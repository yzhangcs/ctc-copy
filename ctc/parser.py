# -*- coding: utf-8 -*-

import os
import tempfile

import errant
import torch
import torch.distributed as dist
from torch.optim import AdamW, Optimizer

from supar.config import Config
from supar.parser import Parser
from supar.utils import Dataset
from supar.utils.field import Field
from supar.utils.logging import get_logger
from supar.utils.parallel import gather, is_dist, is_master
from supar.utils.tokenizer import TransformerTokenizer
from supar.utils.transform import Batch

from .metric import PerplexityMetric
from .model import CTCModel
from .transform import Text

logger = get_logger(__name__)


class CTCParser(Parser):

    NAME = 'ctc'
    MODEL = CTCModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.SRC = self.transform.SRC
        self.TGT = self.transform.TGT
        self.annotator = errant.load("en")

    def init_optimizer(self) -> Optimizer:
        return AdamW(params=[{'params': p, 'lr': self.args.lr * (1 if n.startswith('encoder') else self.args.lr_rate)}
                             for n, p in self.model.named_parameters()],
                     lr=self.args.lr,
                     betas=(self.args.get('mu', 0.9), self.args.get('nu', 0.999)),
                     eps=self.args.get('eps', 1e-8),
                     weight_decay=self.args.get('weight_decay', 0))

    def train_step(self, batch: Batch) -> torch.Tensor:
        src, tgt = batch
        src_mask, tgt_mask = batch.mask, tgt.ne(self.args.pad_index)
        mask = tgt_mask.sum(-1).lt(src_mask.sum(-1) * self.args.upsampling)
        src, tgt, src_mask, tgt_mask = src[mask], tgt[mask], src_mask[mask], tgt_mask[mask]
        x = self.model(src)
        loss = self.model.loss(x, src, tgt, src_mask, tgt_mask, self.args.glat)
        return loss

    @torch.no_grad()
    def eval_step(self, batch: Batch) -> PerplexityMetric:
        src, tgt = batch
        src_mask, tgt_mask = batch.mask, tgt.ne(self.args.pad_index)
        mask = tgt_mask.sum(-1).lt(src_mask.sum(-1) * self.args.upsampling)
        src, tgt, src_mask, tgt_mask = src[mask], tgt[mask], src_mask[mask], tgt_mask[mask]
        x = self.model(src)
        loss = self.model.loss(x, src, tgt, src_mask, tgt_mask)
        preds = golds = None
        if self.args.eval_tgt:
            golds = [(s.values[0], s.values[1], s.fields['src'].tolist(), t.tolist())
                     for s, t in zip(batch.sentences, tgt[tgt_mask].split(tgt_mask.sum(-1).tolist()))]
            preds = self.model.decode(x, src, batch.mask)[:, 0]
            pred_mask = preds.ne(self.args.pad_index)
            preds = [i.tolist() for i in preds[pred_mask].split(pred_mask.sum(-1).tolist())]
            preds = [(s.values[0], self.TGT.tokenize.decode(i), s.fields['src'].tolist(), i)
                     for s, i in zip(batch.sentences, preds)]
        return PerplexityMetric(loss,
                                preds,
                                golds,
                                tgt_mask,
                                (None if self.args.lev else self.annotator),
                                not self.args.eval_tgt)

    @torch.no_grad()
    def pred_step(self, batch: Batch) -> Batch:
        src, = batch
        mask = batch.mask
        for _ in range(self.args.iteration):
            x = self.model(src)
            tgt = self.model.decode(x, src, mask)
            src = tgt[:, 0]
            mask = src.ne(self.args.pad_index)
        batch.tgt = [[self.TGT.tokenize.decode(cand).strip() for cand in i] for i in tgt.tolist()]
        return batch

    @classmethod
    def build(cls, path, min_freq=2, fix_len=20, **kwargs):
        r"""
        Build a brand-new Parser, including initialization of all data fields and model parameters.

        Args:
            path (str):
                The path of the model to be saved.
            min_freq (str):
                The minimum frequency needed to include a token in the vocabulary. Default: 2.
            fix_len (int):
                The max length of all subword pieces. The excess part of each piece will be truncated.
                Required if using CharLSTM/BERT.
                Default: 20.
            kwargs (dict):
                A dict holding the unconsumed arguments.
        """

        args = Config(**locals())
        os.makedirs(os.path.dirname(path) or './', exist_ok=True)
        if os.path.exists(path) and not args.build:
            return cls.load(**args)

        logger.info("Building the fields")
        t = TransformerTokenizer(args.bert)
        SRC = Field('src', pad=t.pad, unk=t.unk, bos=t.bos, eos=t.eos, tokenize=t)
        TGT = Field('tgt', pad=t.pad, unk=t.unk, bos=t.bos, eos=t.eos, tokenize=t)
        transform = Text(SRC=SRC, TGT=TGT)
        if args.vocab:
            if is_master():
                t.extend(Dataset(transform, args.train, **args).src)
            if is_dist():
                with tempfile.TemporaryDirectory(dir='.') as td:
                    td = gather(td)[0]
                    if is_master():
                        torch.save(t, f'{td}/t')
                    dist.barrier()
                    t = torch.load(f'{td}/t')
        SRC.vocab = TGT.vocab = t.vocab

        args.update({'n_words': len(SRC.vocab) + 2,
                     'pad_index': SRC.pad_index,
                     'unk_index': SRC.unk_index,
                     'bos_index': SRC.bos_index,
                     'eos_index': SRC.eos_index,
                     'mask_index': t.mask_token_id,
                     'keep_index': len(SRC.vocab),
                     'nul_index': len(SRC.vocab) + 1})
        logger.info(f"{transform}")
        logger.info("Building the model")
        model = cls.MODEL(**args)
        logger.info(f"{model}\n")

        parser = cls(args, model, transform)
        parser.model.to(parser.device)
        return parser
