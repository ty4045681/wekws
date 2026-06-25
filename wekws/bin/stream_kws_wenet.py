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
        --symbol_table /path/to/new_10364.txt \\
        --bpe_model /path/to/unigram5000.model \\
        --cmvn /path/to/global_cmvn \\
        --keywords "hey eva" \\
        --wav_path test.wav
"""

from __future__ import print_function

import argparse
import logging
import os
import struct
from typing import Dict, Optional

import librosa
import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi
import yaml

from wekws.utils.ctc_kws_decoder import CtcKeywordDecoder
from wenet.utils.init_model import init_model
from wenet.utils.init_tokenizer import init_tokenizer


def get_args():
    parser = argparse.ArgumentParser(
        description='Streaming keyword spotting with WeNet ASR + CTC decode.')
    parser.add_argument('--config', required=True,
                        help='WeNet train.yaml used for ASR training')
    parser.add_argument('--checkpoint', required=True,
                        help='WeNet checkpoint, e.g. step247499.pt')
    parser.add_argument('--symbol_table', required=True,
                        help='Tokenizer symbol table (dict.txt / new_10364.txt)')
    parser.add_argument('--bpe_model', default=None,
                        help='BPE model path; required when train.yaml '
                             'tokenizer is bpe')
    parser.add_argument('--cmvn', default=None,
                        help='Global CMVN file; required when train.yaml '
                             'cmvn is global_cmvn')
    parser.add_argument('--non_lang_syms', default=None,
                        help='Optional non-linguistic symbols file')
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
    parser.add_argument('--debug', action='store_true',
                        help='Print per-frame CTC decode debug info')
    parser.add_argument('--debug_topk', type=int, default=5,
                        help='Vocab top-k tokens to print per frame with --debug')
    return parser.parse_args()


def _load_dataset_conf(configs: dict) -> dict:
    dataset_conf = configs['dataset_conf']
    if 'fbank_conf' in dataset_conf:
        return dataset_conf
    if 'feature_extraction_conf' in dataset_conf:
        dataset_conf = dict(dataset_conf)
        dataset_conf['fbank_conf'] = dataset_conf['feature_extraction_conf']
    return dataset_conf


def _apply_asset_paths(
    configs: dict,
    symbol_table: str,
    cmvn: Optional[str] = None,
    bpe_model: Optional[str] = None,
    non_lang_syms: Optional[str] = None,
) -> dict:
    """Override train.yaml asset paths with explicit CLI arguments."""
    if configs.get('cmvn') == 'global_cmvn':
        if cmvn is None:
            raise ValueError(
                '--cmvn is required when train.yaml uses global_cmvn')
        configs['cmvn_conf']['cmvn_file'] = cmvn

    tokenizer_conf = configs.setdefault('tokenizer_conf', {})
    tokenizer_conf['symbol_table_path'] = symbol_table

    tokenizer_type = configs.get('tokenizer', 'char')
    if tokenizer_type == 'bpe':
        if bpe_model is None:
            raise ValueError(
                '--bpe_model is required when train.yaml tokenizer is bpe')
        tokenizer_conf['bpe_path'] = bpe_model
    elif bpe_model is not None:
        logging.warning('--bpe_model is ignored: tokenizer is %s',
                        tokenizer_type)

    if non_lang_syms is not None:
        tokenizer_conf['non_lang_syms_path'] = non_lang_syms

    return configs


def _check_files_exist(*paths: Optional[str]) -> None:
    for path in paths:
        if path is not None and not os.path.isfile(path):
            raise FileNotFoundError(f'File not found: {path}')


class WeNetKeywordSpotter:
    """WeNet encoder/CTC frontend + wekws-style CTC keyword decoder."""

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        symbol_table: str,
        keywords: str,
        threshold: float,
        min_frames: int,
        max_frames: int,
        interval_frames: int,
        score_beam: int,
        path_beam: int,
        gpu: int,
        cmvn: Optional[str] = None,
        bpe_model: Optional[str] = None,
        non_lang_syms: Optional[str] = None,
        decoding_chunk_size: int = 16,
        num_decoding_left_chunks: int = -1,
        debug: bool = False,
        debug_topk: int = 5,
    ):
        if gpu >= 0 and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{gpu}')
        else:
            self.device = torch.device('cpu')

        with open(config_path, 'r', encoding='utf-8') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
        configs = _apply_asset_paths(configs, symbol_table, cmvn, bpe_model,
                                     non_lang_syms)
        _check_files_exist(
            checkpoint_path,
            symbol_table,
            cmvn,
            bpe_model,
            non_lang_syms,
        )
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

        self.tokenizer = init_tokenizer(configs)

        if hasattr(self.model, 'subsampling_rate'):
            self.subsampling = int(self.model.subsampling_rate())

        blank_id = configs.get('ctc_conf', {}).get('ctc_blank_id', 0)
        self.blank_id = blank_id

        def tokenize_keyword(keyword: str):
            tokens = self.tokenizer.text2tokens(keyword)
            ids = self.tokenizer.tokens2ids(tokens)
            if len(ids) == 0:
                raise ValueError(
                    f'Keyword "{keyword}" cannot be tokenized: {tokens}')
            if any(token_id == blank_id for token_id in ids):
                raise ValueError(
                    f'Keyword "{keyword}" contains blank id {blank_id}: '
                    f'{tokens} -> {ids}')
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
            debug=debug,
            debug_topk=debug_topk,
            token_repr=self._token_repr,
        )

        self.debug = debug

        self.decoding_chunk_size = decoding_chunk_size
        self.num_decoding_left_chunks = num_decoding_left_chunks

        self.wave_remained = np.array([], dtype=np.float32)
        self.all_feats: Optional[torch.Tensor] = None
        self.processed_encoder_frames = 0

        logging.info('Loaded WeNet checkpoint %s', checkpoint_path)

    def _token_repr(self, token_id: int) -> str:
        if token_id == self.blank_id:
            return '<blank>'
        if hasattr(self.tokenizer, 'ids2tokens'):
            tokens = self.tokenizer.ids2tokens([token_id])
            if tokens:
                return tokens[0]
        if hasattr(self.tokenizer, 'detokenize'):
            return self.tokenizer.detokenize([token_id])
        return str(token_id)

    def _print_chunk_top1(self, probs: torch.Tensor, start_enc: int):
        parts = []
        for local_t, prob in enumerate(probs):
            enc_t = start_enc + local_t
            top_prob, top_id = prob.max(dim=0)
            token_id = int(top_id.item())
            parts.append(
                f'enc{enc_t}:{token_id}({self._token_repr(token_id)})'
                f'={top_prob.item():.4f}')
        print(f'[KWS-DEBUG] chunk top1: {" | ".join(parts)}', flush=True)

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
        if self.debug and probs.size(0) > 0:
            print(
                f'[KWS-DEBUG] chunk: +{probs.size(0)} encoder frames '
                f'(enc {self.processed_encoder_frames}..'
                f'{self.processed_encoder_frames + probs.size(0) - 1})',
                flush=True,
            )
            self._print_chunk_top1(probs, self.processed_encoder_frames)
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


def _run_on_pcm(kws: WeNetKeywordSpotter, wav: bytes, chunk_seconds: float,
                debug: bool = False):
    interval = int(chunk_seconds * 16000) * 2
    for i in range(0, len(wav), interval):
        chunk = wav[i:min(i + interval, len(wav))]
        result = kws.forward(chunk)
        if not debug and result.get('state') == 1:
            print(result)


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    kws = WeNetKeywordSpotter(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        symbol_table=args.symbol_table,
        cmvn=args.cmvn,
        bpe_model=args.bpe_model,
        non_lang_syms=args.non_lang_syms,
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
        debug=args.debug,
        debug_topk=args.debug_topk,
    )

    fout = open(args.result_file, 'w', encoding='utf-8') if args.result_file \
        else None

    if args.wav_path:
        _run_on_pcm(kws, _wav_to_pcm_bytes(args.wav_path), args.chunk_seconds,
                    debug=args.debug)

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
