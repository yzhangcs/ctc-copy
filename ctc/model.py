# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from supar.model import Model
from supar.modules import TokenDropout
from supar.modules.transformer import (TransformerDecoder,
                                       TransformerDecoderLayer)
from supar.config import Config
from supar.utils.common import INF, MIN
from supar.utils.fn import pad


class CTCModel(Model):
    r"""
    The implementation of CTC Parser.

    Args:
        n_words (int):
            The size of the word vocabulary.
        n_tags (int):
            The number of POS tags, required if POS tag embeddings are used. Default: ``None``.
        n_chars (int):
            The number of characters, required if character-level representations are used. Default: ``None``.
        n_lemmas (int):
            The number of lemmas, required if lemma embeddings are used. Default: ``None``.
        encoder (str):
            Encoder to use.
            ``'lstm'``: BiLSTM encoder.
            ``'bert'``: BERT-like pretrained language model (for finetuning), e.g., ``'bert-base-cased'``.
            Default: ``'lstm'``.
        n_embed (int):
            The size of word embeddings. Default: 100.
        n_pretrained (int):
            The size of pretrained word embeddings. Default: 125.
        n_feat_embed (int):
            The size of feature representations. Default: 100.
        n_char_embed (int):
            The size of character embeddings serving as inputs of CharLSTM, required if using CharLSTM. Default: 50.
        n_char_hidden (int):
            The size of y states of CharLSTM, required if using CharLSTM. Default: 100.
        char_pad_index (int):
            The index of the padding token in the character vocabulary, required if using CharLSTM. Default: 0.
        elmo (str):
            Name of the pretrained ELMo registered in `ELMoEmbedding.OPTION`. Default: ``'original_5b'``.
        elmo_bos_eos (tuple[bool]):
            A tuple of two boolean values indicating whether to keep start/end boundaries of elmo outputs.
            Default: ``(True, False)``.
        bert (str):
            Specifies which kind of language model to use, e.g., ``'bert-base-cased'``.
            This is required if ``encoder='bert'`` or using BERT features. The full list can be found in `transformers`_.
            Default: ``None``.
        n_bert_layers (int):
            Specifies how many last layers to use, required if ``encoder='bert'`` or using BERT features.
            The final outputs would be weighted sum of the y states of these layers.
            Default: 4.
        mix_dropout (float):
            The dropout ratio of BERT layers, required if ``encoder='bert'`` or using BERT features. Default: .0.
        bert_pooling (str):
            Pooling way to get token embeddings.
            ``first``: take the first subtoken. ``last``: take the last subtoken. ``mean``: take a mean over all.
            Default: ``mean``.
        bert_pad_index (int):
            The index of the padding token in BERT vocabulary, required if ``encoder='bert'`` or using BERT features.
            Default: 0.
        freeze (bool):
            If ``True``, freezes BERT parameters, required if using BERT features. Default: ``True``.
        embed_dropout (float):
            The dropout ratio of input embeddings. Default: .2.
        n_encoder_hidden (int):
            The size of LSTM y states. Default: 600.
        n_encoder_layers (int):
            The number of LSTM layers. Default: 3.
        encoder_dropout (float):
            The dropout ratio of encoder layer. Default: .33.
        mlp_dropout (float):
            The dropout ratio of unary edge factor MLP layers. Default: .33.
        pad_index (int):
            The index of the padding token in the word vocabulary. Default: 0.
        unk_index (int):
            The index of the unknown token in the word vocabulary. Default: 1.

    .. _transformers:
        https://github.com/huggingface/transformers
    """

    def __init__(self,
                 n_words,
                 n_tags=None,
                 n_chars=None,
                 n_lemmas=None,
                 encoder='lstm',
                 n_embed=100,
                 n_pretrained=100,
                 n_feat_embed=100,
                 n_char_embed=50,
                 n_char_hidden=100,
                 char_pad_index=0,
                 char_dropout=0,
                 elmo='original_5b',
                 elmo_bos_eos=(True, False),
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pooling='mean',
                 bert_pad_index=0,
                 freeze=True,
                 embed_dropout=.33,
                 n_encoder_hidden=512,
                 n_encoder_layers=3,
                 encoder_dropout=.1,
                 dropout=.1,
                 pad_index=0,
                 unk_index=1,
                 **kwargs):
        super().__init__(**Config().update(locals()))

        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(self.args.bert,
                                                 add_pooling_layer=False,
                                                 attention_probs_dropout_prob=self.args.dropout,
                                                 hidden_dropout_prob=self.args.dropout)
        if self.args.vocab:
            self.encoder.resize_token_embeddings(self.args.n_words)
        self.token_dropout = TokenDropout(self.args.get('token_dropout', 0))
        self.proj = nn.Linear(self.args.n_encoder_hidden, self.args.upsampling * self.args.n_encoder_hidden)
        self.decoder = TransformerDecoder(layer=TransformerDecoderLayer(n_model=self.args.n_encoder_hidden,
                                                                        dropout=self.args.dropout),
                                          n_layers=self.args.n_decoder_layers)
        self.classifier = nn.Linear(self.args.n_encoder_hidden, self.args.n_words)

    def forward(self, words):
        r"""
        Args:
            words (~torch.LongTensor): ``[batch_size, seq_len]``.
                Word indices.

        Returns:
            ~torch.Tensor:
                Representations for the src sentences of the shape ``[batch_size, seq_len, n_model]``.
        """
        x = self.encoder(inputs_embeds=self.token_dropout(self.encoder.embeddings.word_embeddings(words)),
                         attention_mask=words.ne(self.args.pad_index))[0]
        return self.encoder_dropout(x)

    def resize(self, x):
        batch_size, seq_len, *_, upsampling = x.shape
        resized = x.new_zeros(batch_size, seq_len * upsampling, *_)
        for i, j in enumerate(x.unbind(-1)):
            resized[:, i::upsampling] = j
        return resized

    def loss(self, x, src, tgt, src_mask, tgt_mask, ratio=0):
        x_tgt, glat_mask = self.resize(self.proj(x).view(*x.shape, self.args.upsampling)), None
        if ratio > 0:
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                mask = self.resize(src_mask.unsqueeze(-1).repeat(1, 1, self.args.upsampling))
                preds, s_x = self.decode(x, src, src_mask, True)
                align = self.align(s_x.log_softmax(2).transpose(0, 1), src, tgt, src_mask, tgt_mask)
                probs = ((align.ne(preds) & mask).sum(-1) / mask.sum(-1) * ratio).clamp_(0, 1)
                glat_mask = (src.new_zeros(mask.shape) + probs.unsqueeze(-1)).bernoulli().bool()
            e_tgt = self.encoder.embeddings(torch.where(align.ge(self.args.n_words-2), self.args.mask_index, align))
            x_tgt = torch.where(glat_mask.unsqueeze(-1), e_tgt, x_tgt)
        x = self.decoder(x_tgt=x_tgt,
                         x_src=x,
                         tgt_mask=self.resize(src_mask.unsqueeze(-1).repeat(1, 1, self.args.upsampling)),
                         src_mask=src_mask)
        # [tgt_len, batch_size, n_words]
        s_x = self.classifier(x).log_softmax(2).transpose(0, 1)
        return self.ctc(s_x, src, tgt, src_mask, tgt_mask)

    def ctc(self, s_x, src, tgt, src_mask, tgt_mask, glat_mask=None):
        src = self.resize(src.unsqueeze(-1).repeat(1, 1, self.args.upsampling))
        # [tgt_len, batch_size]
        s_k, s_b = s_x[..., self.args.keep_index], s_x[..., self.args.nul_index]
        # [tgt_len, seq_len, batch_size]
        s_x = s_x.gather(-1, tgt.repeat(s_b.shape[0], 1, 1)).transpose(1, 2)
        s_x = torch.where(src.unsqueeze(-1).eq(tgt.unsqueeze(1)).movedim(0, -1), s_k.unsqueeze(1), s_x)
        if glat_mask is not None:
            glat_mask = glat_mask.t()
            s_b = s_b.masked_fill(glat_mask, 0)
            s_x = s_x.masked_fill(glat_mask.unsqueeze(1), 0)
        src_lens, tgt_lens = src_mask.sum(-1) * self.args.upsampling, tgt_mask.sum(-1)
        tgt_len, seq_len, batch_size = s_x.shape
        # [tgt_len, 2, seq_len + 1, batch_size]
        s = s_x.new_full((tgt_len, 2, seq_len + 1, batch_size), MIN)
        s[0, 0, 0], s[0, 1, 0] = s_b[0], s_x[0, 0]
        for t in range(1, tgt_len):
            s0 = torch.cat((torch.full_like(s[0, 0, :1], MIN), s[t-1, 1, :-1]))
            s1 = s[t-1, 0]
            s2 = s[t-1, 1]
            s[t, 0] = torch.stack((s0, s1)).logsumexp(0) + s_b[t]
            s[t, 1, :-1] = torch.stack((s0[:-1], s1[:-1], s2[:-1])).logsumexp(0) + s_x[t]
        s = s[src_lens - 1, 0, tgt_lens, range(batch_size)].logaddexp(s[src_lens - 1, 1, tgt_lens - 1, range(batch_size)])
        return -s.sum() / tgt_lens.sum()

    def decode(self, x, src, src_mask, score=False):
        batch_size, *_ = x.shape
        beam_size, n_words = self.args.beam_size, self.args.n_words
        keep_index, nul_index, pad_index = self.args.keep_index, self.args.nul_index, self.args.pad_index
        indices = src.new_tensor(range(batch_size)).unsqueeze(1).repeat(1, beam_size).view(-1)
        x = self.decoder(x_tgt=self.resize(self.proj(x).view(*x.shape, self.args.upsampling)),
                         x_src=x,
                         tgt_mask=self.resize(src_mask.unsqueeze(-1).repeat(1, 1, self.args.upsampling)),
                         src_mask=src_mask)
        src = self.resize(src.unsqueeze(-1).repeat(1, 1, self.args.upsampling))
        src_mask = self.resize(src_mask.unsqueeze(-1).repeat(1, 1, self.args.upsampling))

        if not self.args.prefix:
            s_x = self.classifier(x)
            # [batch_size, tgt_len, topk]
            tgt = s_x.topk(self.args.topk, -1)[1]
            tgt = torch.where(tgt.eq(keep_index), src.unsqueeze(-1), tgt)
            # [batch_size, topk, tgt_len]
            tgt = tgt.masked_fill_(~src_mask.unsqueeze(2), self.args.pad_index).transpose(1, 2)
            if score:
                return tgt[:, 0], s_x
            # [batch_size, topk, tgt_len]
            tgt = [[j.unique_consecutive() for j in i.unbind(0)] for i in tgt.unbind(0)]
            tgt = pad([pad([j[j.ne(nul_index)] for j in i], pad_index) for i in tgt], pad_index)
            return tgt

        # [batch_size * beam_size, tgt_len, ...]
        x, src, src_mask = x[indices], src[indices], src_mask[indices]
        # [batch_size * beam_size, max_len]
        tgt = x.new_full((batch_size * beam_size, x.shape[1]), nul_index, dtype=torch.long)
        lens = tgt.new_full((tgt.shape[0],), 0)
        # [batch_size]
        batches = tgt.new_tensor(range(batch_size)) * beam_size
        # accumulated scores
        # [2, batch_size * beam_size]
        s = torch.stack((x.new_full((batch_size, beam_size), -INF).index_fill_(-1, tgt.new_tensor(0), 0).view(-1),
                         x.new_full((batch_size * beam_size,), -INF)))

        def merge(s_b, s_n, tgt, lens, ends):
            # merge the prefixes that have grown in the new step
            s_n = s_n.view(batch_size, beam_size, -1)
            tgt, lens, ends = tgt.view(batch_size, beam_size, -1), lens.view(batch_size, -1), ends.view(batch_size, -1)
            # [batch_size, beam_size, beam_size]
            mask = tgt.scatter(-1, (lens.clamp(1) - 1).unsqueeze(-1), nul_index).unsqueeze(2).eq(tgt.unsqueeze(1)).all(-1)
            mask = mask & lens.gt(0).unsqueeze(2)
            s_g = s_n.gather(-1, ends.unsqueeze(2))
            s_n[..., nul_index] = s_n[..., nul_index].logaddexp(s_g.transpose(1, 2).masked_fill(~mask, -INF).logsumexp(2))
            s_n.scatter_(-1, ends.unsqueeze(2), torch.where(mask.any(2, True), -INF, s_g))
            s_n = s_n.view(batch_size * beam_size, -1)
            return s_b, s_n

        for t in range(x.shape[1]):
            # [batch_size * beam_size]
            mask = src_mask[:, t]
            # the past prefixes
            ends = tgt[range(tgt.shape[0]), lens - 1]
            # [batch_size * beam_size, n_words]
            s_t = self.classifier(x[:, t]).log_softmax(1)
            s_k = s_t.gather(-1, src[:, t].unsqueeze(-1)).logaddexp(s_t[:, keep_index].unsqueeze(-1))
            s_t = s_t.scatter_(-1, src[:, t].unsqueeze(-1), s_k)
            s_t[:, keep_index] = -INF
            s_e = s_t.gather(1, ends.unsqueeze(1))
            s_p = s.logsumexp(0).unsqueeze(-1)
            # [batch_size * beam_size]
            # the position for blanks are used for storing prefixes kept unchanged
            #  *a - -> *a
            s_b = s_p + s_t.masked_fill(tgt.new_tensor(range(n_words)).ne(nul_index).unsqueeze(0), -INF)
            #  *a b -> *ab
            s_n = s_p + s_t
            # *a- a -> *aa
            s_n = s_n.scatter_(1, ends.unsqueeze(1), s[0].unsqueeze(1) + s_e)
            #  *a a -> *a
            s_n[:, nul_index] = s[1] + s_e.squeeze(1)
            # [2, batch_size * beam_size, n_words]
            s = torch.stack((merge(s_b, s_n, tgt, lens, ends)))
            # [batch_size, beam_size]
            cands = s.logsumexp(0).view(batch_size, -1).topk(beam_size, -1)[1]
            # [2, batch_size * beam_size]
            s = s.view(2, batch_size, -1).gather(-1, cands.repeat(2, 1, 1)).view(2, -1)
            # beams, tokens = cands // n_words, cands % n_words
            beams, tokens = cands.div(n_words, rounding_mode='floor'), (cands % n_words).view(-1, 1)
            indices = (batches.unsqueeze(-1) + beams).view(-1)
            lens[mask] = lens[indices[mask]]
            # [batch_size * beam_size, max_len]
            tgt[mask] = tgt[indices[mask]].scatter_(1, lens[mask].unsqueeze(1), tokens[mask])
            lens += tokens.ne(nul_index).squeeze(1) & mask
        cands = s.logsumexp(0).view(batch_size, -1).topk(self.args.topk, -1)[1]
        tgt = tgt[(batches.unsqueeze(-1) + cands).view(-1)].view(batch_size, self.args.topk, -1)
        tgt = pad([pad([j[j.ne(nul_index)] for j in i], pad_index) for i in tgt], pad_index)
        return tgt

    def align(self, s_x, src, tgt, src_mask, tgt_mask):
        src = self.resize(src.unsqueeze(-1).repeat(1, 1, self.args.upsampling))
        # [tgt_len, batch_size]
        s_k, s_b = s_x[..., self.args.keep_index], s_x[..., self.args.nul_index]
        # [tgt_len, seq_len, batch_size]
        s_x = s_x.gather(-1, tgt.repeat(s_b.shape[0], 1, 1)).transpose(1, 2)
        s_x = torch.where(src.unsqueeze(-1).eq(tgt.unsqueeze(1)).movedim(0, -1), s_k.unsqueeze(1), s_x)
        src_lens, tgt_lens = src_mask.sum(-1) * self.args.upsampling, tgt_mask.sum(-1)
        tgt_len, seq_len, batch_size = s_x.shape
        # [tgt_len, 2, seq_len + 1, batch_size]
        s = s_x.new_full((tgt_len, 2, seq_len + 1, batch_size), -INF)
        p = tgt.new_full((tgt_len, 2, seq_len + 1, batch_size), -1)
        s[0, 0, 0], s[0, 1, 0] = s_b[0], s_x[0, 0]

        for t in range(1, tgt_len):
            s0 = torch.cat((torch.full_like(s[0, 0, :1], -INF), s[t-1, 1, :-1]))
            s1 = s[t-1, 0]
            s2 = s[t-1, 1]
            s_t, p[t, 0] = torch.stack((s0, s1)).max(0)
            s[t, 0] = s_t + s_b[t]
            s_t, p[t, 1, :-1] = torch.stack((s0[:-1], s1[:-1], s2[:-1])).max(0)
            s[t, 1, :-1] = s_t + s_x[t]
        _, p_t = torch.stack((s[src_lens - 1, 0, tgt_lens, range(batch_size)],
                              s[src_lens - 1, 1, tgt_lens - 1, range(batch_size)])).max(0)

        def backtrack(p, tgt, notnul):
            j, pred = [len(p[0][0])-1, len(p[0][0])-2], []
            for i in reversed(range(len(p))):
                prev = p[i][notnul][j[notnul]]
                pred.append(tgt[j[notnul]] if bool(notnul) else self.args.nul_index)
                if notnul == 0:
                    if prev == 0:
                        notnul = 1
                        j[notnul] = j[1-notnul] - 1
                elif notnul == 1:
                    if prev == 0:
                        j[notnul] -= 1
                    if prev == 1:
                        notnul = 0
                        j[notnul] = j[1-notnul]
            return tuple(reversed(pred))
        p_t, tgt, preds = p_t.tolist(), tgt.tolist(), torch.full_like(src, self.args.pad_index)
        for i, (src_len, tgt_len) in enumerate(zip(src_lens.tolist(), tgt_lens.tolist())):
            preds[i, :src_len] = src.new_tensor(backtrack(p[:src_len, :, :tgt_len+1, i].tolist(), tgt[i][:tgt_len], p_t[i]))
        return preds
