# Copyright (c) 2025 WeNet Community
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Streaming keyword spotting with a WeNet ASR model.

This script reuses the CTC prefix-beam keyword logic from
``wekws/bin/stream_kws_ctc.py``, but replaces the wekws backbone with a
standard WeNet ASR encoder + CTC head.  Put this file under your WeNet repo as
``wenet/bin/stream_kws.py`` if you prefer to keep it on the WeNet side only.

Example::

    python wekws/bin/stream_kws_wenet.py \\
        --config /path/to/train.yaml \\
        --checkpoint /path/to/step247499.pt \\
        --dict /path/to/dict \\
        --keywords "你好小问,嗨小问" \\
        --wav_path test.wav
"""

from __future__ import print_function

import argparse
import logging
import os
import struct
from typing import Dict, Optional, Tuple

import librosa
import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi
import yaml

from wekws.utils.ctc_kws_decoder import CtcKeywordDecoder
from wenet.text.char_tokenizer import CharTokenizer
from wenet.utils.init_model import init_model


def get_args():
    parser = argparse.ArgumentParser(
        description='Streaming keyword spotting with WeNet ASR + CTC decode.')
    parser.add_argument('--config', required=True,
                        help='WeNet train.yaml used for ASR training')
    parser.add_argument('--checkpoint', required=True,
                        help='WeNet checkpoint, e.g. step247499.pt')
    parser.add_argument('--dict', required=True,
                        help='CharTokenizer dict dir or dict.txt path')
    parser.add_argument('--wav_path', default=None, help='16k mono wav file')
    parser.add_argument('--wav_scp', default=None, help='Kaldi-style wav.scp')
    parser.add_argument('--result_file', default=None, help='Output result')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU id, -1 for CPU')
    parser.add_argument('--keywords', required=True,
                        help='Keywords split by comma')
    parser.add_argument('--score_beam_size', type=int, default=3)
    parser.add_argument('--path_beam_size', type=int, default=20)
    parser.add_argument('--threshold', type=float, default=0.0)
    parser.add_argument('--min_frames', type=int, default=5)
    parser.add_argument('--max_frames', type=int, default=250)
    parser.add_argument('--interval_frames', type=int, default=50)
    parser.add_argument('--chunk_seconds', type=float, default=0.3,
                        help='Simulated streaming chunk size in seconds')
    parser.add_argument('--decoding_chunk_size', type=int, default=16,
                        help='WeNet encoder chunk size passed to encoder')
    parser.add_argument('--num_decoding_left_chunks', type=int, default=-1)
    return parser.parse_args()


def _resolve_dict_paths(dict_arg: str) -> Tuple[str, Optional[str]]:
    if os.path.isdir(dict_arg):
        dict_txt = os.path.join(dict_arg, 'dict.txt')
        words_txt = os.path.join(dict_arg, 'words.txt')
        if not os.path.isfile(words_txt):
            words_txt = None
        return dict_txt, words_txt
    return dict_arg, None


def _load_dataset_conf(configs: dict) -> dict:
    dataset_conf = configs['dataset_conf']
    if 'fbank_conf' in dataset_conf:
        return dataset_conf
    if 'feature_extraction_conf' in dataset_conf:
        dataset_conf = dict(dataset_conf)
        dataset_conf['fbank_conf'] = dataset_conf['feature_extraction_conf']
    return dataset_conf


def _prepare_configs(config_path: str, configs: dict) -> dict:
    config_dir = os.path.dirname(os.path.abspath(config_path))
    if configs.get('cmvn', None) == 'global_cmvn':
        cmvn_file = configs['cmvn_conf']['cmvn_file']
        if not os.path.isabs(cmvn_file):
            configs['cmvn_conf']['cmvn_file'] = os.path.join(
                config_dir, cmvn_file)
    tokenizer_conf = configs.get('tokenizer_conf', {})
    for key, value in list(tokenizer_conf.items()):
        if isinstance(value, str) and not os.path.isabs(value):
            candidate = os.path.join(config_dir, os.path.basename(value))
            if os.path.exists(candidate):
                tokenizer_conf[key] = candidate
    return configs


class WeNetKeywordSpotter:
    """WeNet encoder/CTC frontend + wekws-style CTC keyword decoder."""

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        dict_path: str,
        keywords: str,
        threshold: float,
        min_frames: int,
        max_frames: int,
        interval_frames: int,
        score_beam: int,
        path_beam: int,
        gpu: int,
        decoding_chunk_size: int = 16,
        num_decoding_left_chunks: int = -1,
    ):
        if gpu >= 0 and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{gpu}')
        else:
            self.device = torch.device('cpu')

        with open(config_path, 'r', encoding='utf-8') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
        configs = _prepare_configs(config_path, configs)
        self.configs = configs
        self.dataset_conf = _load_dataset_conf(configs)

        fbank_conf = self.dataset_conf['fbank_conf']
        self.sample_rate = self.dataset_conf.get('resample_conf',
                                                 {}).get('resample_rate', 16000)
        self.num_mel_bins = fbank_conf['num_mel_bins']
        self.frame_length = fbank_conf['frame_length']
        self.frame_shift = fbank_conf['frame_shift']
        self.subsampling = 1
        self.frame_resolution = self.frame_shift / 1000.0

        args = argparse.Namespace(checkpoint=checkpoint_path)
        model, _ = init_model(args, configs)
        if not hasattr(model, 'ctc_logprobs'):
            raise RuntimeError(
                'Only WeNet ASRModel with CTC head is supported. '
                f'Got model type: {configs.get("model", "unknown")}')
        self.model = model.to(self.device)
        self.model.eval()

        dict_txt, _ = _resolve_dict_paths(dict_path)
        non_lang_syms = configs.get('tokenizer_conf', {}).get(
            'non_lang_syms', None)
        self.tokenizer = CharTokenizer(dict_txt,
                                       non_lang_syms,
                                       split_with_space=True)

        if hasattr(self.model, 'subsampling_rate'):
            self.subsampling = int(self.model.subsampling_rate())
            self.frame_resolution = self.frame_shift / 1000.0 * self.subsampling

        blank_id = configs.get('ctc_conf', {}).get('ctc_blank_id', 0)

        def tokenize_keyword(keyword: str):
            tokens = self.tokenizer.text2tokens(keyword)
            ids = self.tokenizer.tokens2ids(tokens)
            if len(ids) == 0:
                raise ValueError(
                    f'Keyword "{keyword}" cannot be tokenized with {dict_txt}')
            return ids

        keywords_token, keywords_idxset = CtcKeywordDecoder.build_keywords(
            keywords, tokenize_keyword)
        self.decoder = CtcKeywordDecoder(
            keywords_token=keywords_token,
            keywords_idxset=keywords_idxset,
            threshold=threshold,
            min_frames=min_frames,
            max_frames=max_frames,
            interval_frames=interval_frames,
            score_beam=score_beam,
            path_beam=path_beam,
            frame_resolution=self.frame_resolution,
            blank_id=blank_id,
        )

        self.decoding_chunk_size = decoding_chunk_size
        self.num_decoding_left_chunks = num_decoding_left_chunks
        self.blank_id = blank_id

        self.wave_remained = np.array([], dtype=np.float32)
        self.all_feats: Optional[torch.Tensor] = None
        self.processed_encoder_frames = 0

        logging.info('Loaded WeNet checkpoint %s', checkpoint_path)

    def accept_wave(self, wave: bytes) -> Optional[torch.Tensor]:
        data = []
        for i in range(0, len(wave), 2):
            value = struct.unpack('<h', wave[i:i + 2])[0]
            data.append(value)

        wave_arr = np.array(data, dtype=np.float32)
        wave_arr = np.append(self.wave_remained, wave_arr)
        min_samples = int(self.frame_length * self.sample_rate / 1000)
        if wave_arr.size < min_samples:
            self.wave_remained = wave_arr
            return None

        wave_tensor = torch.from_numpy(wave_arr).float()
        feats = kaldi.fbank(
            wave_tensor.unsqueeze(0),
            num_mel_bins=self.num_mel_bins,
            frame_length=self.frame_length,
            frame_shift=self.frame_shift,
            dither=0.0,
            energy_floor=0.0,
            sample_frequency=self.sample_rate,
        )

        frame_shift_samples = int(self.frame_shift / 1000 * self.sample_rate)
        self.wave_remained = wave_arr[feats.size(0) * frame_shift_samples:]
        return feats

    def _run_ctc_probs(self, feats: torch.Tensor) -> torch.Tensor:
        speech = feats.unsqueeze(0).to(self.device)
        speech_lengths = torch.tensor([speech.size(1)],
                                        dtype=torch.int32,
                                        device=self.device)
        with torch.no_grad():
            encoder_out, encoder_mask = self.model.encoder(
                speech,
                speech_lengths,
                decoding_chunk_size=self.decoding_chunk_size,
                num_decoding_left_chunks=self.num_decoding_left_chunks,
            )
            ctc_log_probs = self.model.ctc_logprobs(
                encoder_out, blank_id=self.blank_id)
            valid_len = int(encoder_mask.squeeze(1).sum().item())
            return ctc_log_probs[0, :valid_len]

    def forward(self, wave_chunk: bytes) -> Dict:
        feats = self.accept_wave(wave_chunk)
        if feats is None or feats.size(0) < 1:
            return {}

        if self.all_feats is None:
            self.all_feats = feats
        else:
            self.all_feats = torch.cat([self.all_feats, feats], dim=0)

        ctc_log_probs = self._run_ctc_probs(self.all_feats)
        if ctc_log_probs.size(0) <= self.processed_encoder_frames:
            return self.decoder.result

        probs = ctc_log_probs[self.processed_encoder_frames:].exp().cpu()
        activated = False
        for local_t, prob in enumerate(probs):
            enc_t = self.processed_encoder_frames + local_t
            mel_frame = enc_t * self.subsampling
            self.decoder.decode_frame(mel_frame, prob)
            if self.decoder.activated:
                activated = True
                self.decoder.reset()
                break

        new_frames = probs.size(0)
        self.decoder.finish_chunk(new_frames * self.subsampling)
        self.processed_encoder_frames += new_frames

        return self.decoder.result

    def reset_all(self):
        self.decoder.reset_all()
        self.wave_remained = np.array([], dtype=np.float32)
        self.all_feats = None
        self.processed_encoder_frames = 0


def _wav_to_pcm_bytes(wav_path: str) -> bytes:
    y, _ = librosa.load(wav_path, sr=16000, mono=True)
    return (y * (1 << 15)).astype('int16').tobytes()


def _run_on_pcm(kws: WeNetKeywordSpotter, wav: bytes, chunk_seconds: float):
    interval = int(chunk_seconds * 16000) * 2
    for i in range(0, len(wav), interval):
        chunk = wav[i:min(i + interval, len(wav))]
        result = kws.forward(chunk)
        if result:
            print(result)


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    kws = WeNetKeywordSpotter(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        dict_path=args.dict,
        keywords=args.keywords,
        threshold=args.threshold,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        interval_frames=args.interval_frames,
        score_beam=args.score_beam_size,
        path_beam=args.path_beam_size,
        gpu=args.gpu,
        decoding_chunk_size=args.decoding_chunk_size,
        num_decoding_left_chunks=args.num_decoding_left_chunks,
    )

    fout = open(args.result_file, 'w', encoding='utf-8') if args.result_file \
        else None

    if args.wav_path:
        _run_on_pcm(kws, _wav_to_pcm_bytes(args.wav_path), args.chunk_seconds)

    if args.wav_scp:
        with open(args.wav_scp, 'r', encoding='utf-8') as fscp:
            for line in fscp:
                utt, wav_path = line.strip().split(maxsplit=1)
                kws.reset_all()
                wav = _wav_to_pcm_bytes(wav_path)
                activated = False
                interval = int(args.chunk_seconds * 16000) * 2
                for i in range(0, len(wav), interval):
                    chunk = wav[i:min(i + interval, len(wav))]
                    result = kws.forward(chunk)
                    if result.get('state') == 1:
                        activated = True
                        if fout:
                            fout.write('{} detected {} {:.3f}\n'.format(
                                utt, result['keyword'], result['score']))
                if not activated and fout:
                    fout.write('{} rejected\n'.format(utt))

    if fout:
        fout.close()


if __name__ == '__main__':
    main()
