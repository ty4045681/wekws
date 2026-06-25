# Copyright (c) 2023 Jing Du(thuduj12@163.com)
#               2025 WeNet Community
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
"""Streaming CTC keyword decoder shared by wekws and WeNet ASR frontends."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import torch


def is_sublist(main_list: Sequence[int], check_list: Sequence[int]) -> int:
    if len(main_list) < len(check_list):
        return -1

    if len(main_list) == len(check_list):
        return 0 if list(main_list) == list(check_list) else -1

    for i in range(len(main_list) - len(check_list)):
        if main_list[i] == check_list[0]:
            for j in range(len(check_list)):
                if main_list[i + j] != check_list[j]:
                    break
            else:
                return i
    return -1


def ctc_prefix_beam_search(
    t: int,
    probs: torch.Tensor,
    cur_hyps: List[Tuple[tuple, tuple]],
    keywords_idxset: Optional[Set[int]],
    score_beam_size: int,
    blank_id: int = 0,
) -> List[Tuple[tuple, tuple]]:
    next_hyps = defaultdict(lambda: (0.0, 0.0, []))

    top_k_probs, top_k_index = probs.topk(score_beam_size)

    filter_index = []
    for prob, idx in zip(top_k_probs.tolist(), top_k_index.tolist()):
        if keywords_idxset is not None:
            if prob > 0.05 and idx in keywords_idxset:
                filter_index.append(idx)
        elif prob > 0.05:
            filter_index.append(idx)

    if len(filter_index) == 0:
        return cur_hyps

    for s in filter_index:
        ps = probs[s].item()

        for prefix, (pb, pnb, cur_nodes) in cur_hyps:
            last = prefix[-1] if len(prefix) > 0 else None
            if s == blank_id:
                n_pb, n_pnb, nodes = next_hyps[prefix]
                n_pb = n_pb + pb * ps + pnb * ps
                nodes = cur_nodes.copy()
                next_hyps[prefix] = (n_pb, n_pnb, nodes)
            elif s == last:
                if not math.isclose(pnb, 0.0, abs_tol=0.000001):
                    n_pb, n_pnb, nodes = next_hyps[prefix]
                    n_pnb = n_pnb + pnb * ps
                    nodes = cur_nodes.copy()
                    if ps > nodes[-1]['prob']:
                        nodes[-1]['prob'] = ps
                        nodes[-1]['frame'] = t
                    next_hyps[prefix] = (n_pb, n_pnb, nodes)

                if not math.isclose(pb, 0.0, abs_tol=0.000001):
                    n_prefix = prefix + (s, )
                    n_pb, n_pnb, nodes = next_hyps[n_prefix]
                    n_pnb = n_pnb + pb * ps
                    nodes = cur_nodes.copy()
                    nodes.append(dict(token=s, frame=t, prob=ps))
                    next_hyps[n_prefix] = (n_pb, n_pnb, nodes)
            else:
                n_prefix = prefix + (s, )
                n_pb, n_pnb, nodes = next_hyps[n_prefix]
                if nodes:
                    if ps > nodes[-1]['prob']:
                        nodes.pop()
                        nodes.append(dict(token=s, frame=t, prob=ps))
                else:
                    nodes = cur_nodes.copy()
                    nodes.append(dict(token=s, frame=t, prob=ps))
                n_pnb = n_pnb + pb * ps + pnb * ps
                next_hyps[n_prefix] = (n_pb, n_pnb, nodes)

    next_hyps = sorted(next_hyps.items(),
                       key=lambda x: (x[1][0] + x[1][1]),
                       reverse=True)
    return next_hyps


class CtcKeywordDecoder:
    """Frame-synchronous CTC keyword detector (logic from stream_kws_ctc.py)."""

    def __init__(
        self,
        keywords_token: Dict[str, Dict[str, Union[Tuple[int, ...], str]]],
        keywords_idxset: Set[int],
        threshold: float = 0.0,
        min_frames: int = 5,
        max_frames: int = 250,
        interval_frames: int = 50,
        score_beam: int = 3,
        path_beam: int = 20,
        frame_resolution: float = 0.01,
        blank_id: int = 0,
    ):
        self.keywords_token = keywords_token
        self.keywords_idxset = keywords_idxset
        self.threshold = threshold
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.interval_frames = interval_frames
        self.score_beam = score_beam
        self.path_beam = path_beam
        self.frame_resolution = frame_resolution
        self.blank_id = blank_id

        self.cur_hyps = [(tuple(), (1.0, 0.0, []))]
        self.hit_score = 1.0
        self.activated = False
        self.last_active_pos = -1
        self.total_frames = 0
        self.result: Dict = {}

    @staticmethod
    def build_keywords(
        keywords: str,
        tokenize_fn,
    ) -> Tuple[Dict[str, Dict[str, Union[Tuple[int, ...], str]]], Set[int]]:
        keywords_list = [kw.strip() for kw in keywords.split(',') if kw.strip()]
        keywords_token = {}
        keywords_idxset = {0}
        for keyword in keywords_list:
            token_ids = tuple(tokenize_fn(keyword))
            keywords_token[keyword] = {
                'token_id': token_ids,
                'token_str': ''.join('%s ' % i for i in token_ids),
            }
            keywords_idxset.update(token_ids)

        token_print = ' '.join(
            f'{k}:{v["token_id"]}' for k, v in keywords_token.items())
        logging.info('Keyword token ids: %s', token_print)
        return keywords_token, keywords_idxset

    def decode_frame(self, frame_idx: int, prob: torch.Tensor):
        next_hyps = ctc_prefix_beam_search(frame_idx, prob, self.cur_hyps,
                                           self.keywords_idxset,
                                           self.score_beam, self.blank_id)
        self.cur_hyps = next_hyps[:self.path_beam]
        self._execute_detection(frame_idx)

        if len(self.cur_hyps) > 0 and len(self.cur_hyps[0][0]) > 0:
            keyword_may_start = int(self.cur_hyps[0][1][2][0]['frame'])
            if (self.total_frames - keyword_may_start) > self.max_frames:
                self.reset()

    def finish_chunk(self, num_frames: int):
        self.total_frames += num_frames

    def _execute_detection(self, absolute_time: int):
        hit_keyword = None
        start = 0
        end = 0
        self.hit_score = 1.0

        hyps = [(y[0], y[1][0] + y[1][1], y[1][2]) for y in self.cur_hyps]
        for one_hyp in hyps:
            prefix_ids = one_hyp[0]
            prefix_nodes = one_hyp[2]
            if len(prefix_ids) != len(prefix_nodes):
                continue
            for word, meta in self.keywords_token.items():
                lab = meta['token_id']
                offset = is_sublist(prefix_ids, lab)
                if offset != -1:
                    hit_keyword = word
                    start = prefix_nodes[offset]['frame']
                    end = prefix_nodes[offset + len(lab) - 1]['frame']
                    for idx in range(offset, offset + len(lab)):
                        self.hit_score *= prefix_nodes[idx]['prob']
                    break
            if hit_keyword is not None:
                self.hit_score = math.sqrt(self.hit_score)
                break

        duration = end - start
        self.activated = False
        if hit_keyword is not None:
            if (self.hit_score >= self.threshold and
                    self.min_frames <= duration <= self.max_frames and
                (self.last_active_pos == -1 or
                 end - self.last_active_pos >= self.interval_frames)):
                self.activated = True
                self.last_active_pos = end
                logging.info(
                    'Frame %d detect %s from %d to %d frame, '
                    'duration %d, score %.4f, activated.',
                    absolute_time, hit_keyword, start, end, duration,
                    self.hit_score)
            else:
                logging.debug(
                    'Frame %d detect %s but rejected (score=%.4f, '
                    'duration=%d).', absolute_time, hit_keyword,
                    self.hit_score, duration)

        self.result = {
            'state': 1 if self.activated else 0,
            'keyword': hit_keyword if self.activated else None,
            'start': start * self.frame_resolution if self.activated else None,
            'end': end * self.frame_resolution if self.activated else None,
            'score': self.hit_score if self.activated else None,
        }

    def reset(self):
        self.cur_hyps = [(tuple(), (1.0, 0.0, []))]
        self.activated = False
        self.hit_score = 1.0

    def reset_all(self):
        self.reset()
        self.last_active_pos = -1
        self.total_frames = 0
        self.result = {}
