# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import List, Optional

import torch
from torch.distributions.utils import lazy_property

from supar.structs.dist import StructuredDistribution
from supar.structs.semiring import LogSemiring, Semiring


class Levenshtein(StructuredDistribution):

    def __init__(
        self,
        scores: torch.Tensor,
        lens: Optional[torch.LongTensor] = None
    ) -> Levenshtein:
        super().__init__(scores)

        batch_size, _, seq_len, src_len = scores.shape[:4]
        if lens is not None:
            self.lens = lens
        else:
            self.lens = (scores.new_zeros(batch_size, 2) + scores.new_tensor(src_len, seq_len)).long()
        self.src_lens, self.tgt_lens = lens.unbind(-1)
        self.src_mask = self.src_lens.unsqueeze(-1).gt(self.lens.new_tensor(range(src_len)))
        self.tgt_mask = self.tgt_lens.unsqueeze(-1).gt(self.lens.new_tensor(range(seq_len)))

    def __add__(self, other):
        return Levenshtein(torch.stack((self.scores, other.scores)), self.lens)

    @lazy_property
    def argmax(self):
        margs = self.backward(self.max.sum())
        margs, edits = margs.argmax(1).transpose(1, 2), [torch.where(i) for i in margs.sum(1).transpose(1, 2).unbind()]
        return [torch.stack((e[0], e[1], m[e])).t().tolist() for e, m in zip(edits, margs)]

    def score(self, value: List) -> torch.Tensor:
        lens = self.lens.new_tensor([len(i) for i in value])
        edit_mask = lens.unsqueeze(-1).gt(lens.new_tensor(range(max(lens))))
        edits = list(self.lens.new_tensor([(i,) + span for i, spans in enumerate(value) for span in spans]).unbind(-1))
        s_edit = self.scores[edits[0], edits[3], edits[2], edits[1]]
        s = s_edit.new_full(edit_mask.shape, LogSemiring.one).masked_scatter_(edit_mask, s_edit)
        return LogSemiring.prod(s)

    def forward(self, semiring: Semiring) -> torch.Tensor:
        # [4, seq_len, src_len, batch_size, ...]
        s_edit = semiring.convert(self.scores.movedim(0, 3))

        _, seq_len, src_len, batch_size = s_edit.shape[:4]
        tgt_lens, src_lens, src_mask = self.tgt_lens, self.src_lens, self.src_mask.t()
        # [seq_len, src_len, batch_size]
        alpha = semiring.zeros_like(s_edit[0])
        trans = semiring.cumprod(torch.cat((semiring.ones_like(s_edit[0, :, :1]), s_edit[0, :, 1:]), 1), 1)
        # [seq_len, src_len, src_len, batch_size]
        trans = trans.unsqueeze(2) - trans.unsqueeze(1)
        trans_mask = src_mask.unsqueeze(0) & torch.ones_like(src_mask).unsqueeze(1)
        # [src_len, src_len, batch_size]
        trans_mask = trans_mask & src_mask.new_ones(src_len, src_len).tril(-1).unsqueeze(-1)

        for t in range(seq_len):
            s_a = alpha[t - 1] if t > 0 else semiring.ones_like(trans[0, 0])
            # INSERT
            s_i = semiring.mul(s_a, s_edit[1, t])
            # KEEP
            s_k = torch.cat((semiring.zeros_like(s_a[:1]), semiring.mul(s_a[:-1], s_edit[2, t, 1:])), 0)
            # REPLACE
            s_r = torch.cat((semiring.zeros_like(s_a[:1]), semiring.mul(s_a[:-1], s_edit[3, t, 1:])), 0)
            # SWAP
            s_s = torch.cat((semiring.zeros_like(s_a[:1]), semiring.mul(s_a[:-1], s_edit[4, t, 1:])), 0)
            # [src_len, batch_size]
            s_a = semiring.sum(torch.stack((s_i, s_k, s_r, s_s)), 0)
            # DELETE
            s_d = semiring.sum(semiring.zero_mask_(semiring.mul(trans[t], s_a.unsqueeze(0)), ~trans_mask), 1)
            # [src_len, batch_size]
            alpha[t] = semiring.add(s_d, s_a)
        # the full input is consumed when the final output symbol is generated
        return semiring.unconvert(alpha[tgt_lens - 1, src_lens - 1, range(batch_size)])
