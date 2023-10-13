# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from io import StringIO
from typing import Iterable, List, Optional, Union

import pathos.multiprocessing as mp
import spacy
import spacy.parts_of_speech as POS
import torch.distributed as dist
from rapidfuzz.distance import Indel
from spacy.tokens import Doc

from supar.utils import Field
from supar.utils.fn import binarize, debinarize
from supar.utils.logging import progress_bar
from supar.utils.parallel import gather, is_dist, is_master
from supar.utils.tokenizer import Tokenizer
from supar.utils.transform import Sentence, Transform


class Alignment:
    # Protected class resource
    _open_pos = {POS.ADJ, POS.ADV, POS.NOUN, POS.VERB}

    # Input 1: An original text string parsed by spacy
    # Input 2: A corrected text string parsed by spacy
    # Input 3: A flag for standard Levenshtein alignment
    def __init__(self, orig, cor, lev=False, nlp=None):
        # Set orig and cor
        self.nlp = nlp
        self.orig_toks, self.cor_toks = orig, cor
        self.orig = self.parse(orig)
        self.cor = self.parse(cor)
        # Align orig and cor and get the cost and op matrices
        self.cost_matrix, self.op_matrix = self.align(lev)
        # Get the cheapest align sequence from the op matrix
        self.align_seq = self.get_cheapest_align_seq()

    # Input: A flag for standard Levenshtein alignment
    # Output: The cost matrix and the operation matrix of the alignment
    def align(self, lev):
        # Sentence lengths
        o_len = len(self.orig)
        c_len = len(self.cor)
        # Lower case token IDs (for transpositions)
        # Create the cost_matrix and the op_matrix
        cost_matrix = [[0.0 for j in range(c_len+1)] for i in range(o_len+1)]
        op_matrix = [["O" for j in range(c_len+1)] for i in range(o_len+1)]
        # Fill in the edges
        for i in range(1, o_len+1):
            cost_matrix[i][0] = cost_matrix[i-1][0] + 1
            op_matrix[i][0] = "D"
        for j in range(1, c_len+1):
            cost_matrix[0][j] = cost_matrix[0][j-1] + 1
            op_matrix[0][j] = "I"

        # Loop through the cost_matrix
        for i in range(o_len):
            for j in range(c_len):
                # Matches
                if self.orig[i].orth == self.cor[j].orth and self.orig_toks[i] == self.cor_toks[j]:
                    cost_matrix[i+1][j+1] = cost_matrix[i][j]
                    op_matrix[i+1][j+1] = "M"
                # Non-matches
                else:
                    del_cost = cost_matrix[i][j+1] + 1
                    ins_cost = cost_matrix[i+1][j] + 1
                    trans_cost = float("inf")  # currently ignore swap/transpose
                    k = 0
                    # Standard Levenshtein (S = 1)
                    if lev:
                        sub_cost = cost_matrix[i][j] + 1
                    # Linguistic Damerau-Levenshtein
                    else:
                        # Custom substitution
                        sub_cost = cost_matrix[i][j] + self.get_sub_cost(self.orig[i], self.cor[j])
                    # Costs
                    costs = [trans_cost, sub_cost, ins_cost, del_cost]
                    # Get the index of the cheapest (first cheapest if tied)
                    l = costs.index(min(costs))
                    # Save the cost and the op in the matrices
                    cost_matrix[i+1][j+1] = costs[l]
                    if l == 0:
                        op_matrix[i+1][j+1] = "T"+str(k+1)
                    elif l == 1:
                        op_matrix[i+1][j+1] = "S"
                    elif l == 2:
                        op_matrix[i+1][j+1] = "I"
                    else:
                        op_matrix[i+1][j+1] = "D"
        # Return the matrices
        return cost_matrix, op_matrix

    # Input 1: A spacy orig Token
    # Input 2: A spacy cor Token
    # Output: A linguistic cost between 0 < x < 2
    def get_sub_cost(self, o, c):
        # Short circuit if the only difference is case
        if o.lower == c.lower:
            return 0
        # Lemma cost
        if o.lemma == c.lemma:
            lemma_cost = 0
        else:
            lemma_cost = 0.499
        # POS cost
        if o.pos == c.pos:
            pos_cost = 0
        elif o.pos in self._open_pos and c.pos in self._open_pos:
            pos_cost = 0.25
        else:
            pos_cost = 0.5
        # Char cost
        char_cost = Indel.normalized_distance(o.text, c.text)
        # Combine the costs
        return lemma_cost + pos_cost + char_cost

    # Get the cheapest alignment sequence and indices from the op matrix
    def get_cheapest_align_seq(self):
        i = len(self.op_matrix)-1
        j = len(self.op_matrix[0])-1
        op_set = {'D': 0, 'I': 1, 'M': 2, 'S': 3}
        align_seq = [(i, j, op_set['M'])]
        # Work backwards from bottom right until we hit top left
        while i + j != 0:
            # Get the edit operation in the current cell
            op = self.op_matrix[i][j]
            # Matches and substitutions
            if op in {"M", "S"}:
                i -= 1
                j -= 1
            # Deletions
            elif op == "D":
                i -= 1
            # Insertions
            elif op == "I":
                j -= 1
            align_seq.append((i, j, op_set[op]))
        # Reverse the list to go from left to right and return
        align_seq.reverse()
        return align_seq

    # Alignment object string representation
    def __str__(self):
        orig = " ".join(["Orig:"]+[tok.text for tok in self.orig])
        cor = " ".join(["Cor:"]+[tok.text for tok in self.cor])
        cost_matrix = "\n".join(["Cost Matrix:"]+[str(row) for row in self.cost_matrix])
        op_matrix = "\n".join(["Operation Matrix:"]+[str(row) for row in self.op_matrix])
        seq = "Best alignment: "+str(self.align_seq)
        return "\n".join([orig, cor, cost_matrix, op_matrix, seq])

    def parse(self, text):
        if isinstance(text, str):
            new_text = []
            for tok in text.split():  # remove bpe delimeter
                new_text.append(tok if tok[-4:] != "</w>" else tok[:-4])
            text = Doc(self.nlp.vocab, new_text)
        else:
            new_text = []
            for tok in text:
                new_text.append(tok if tok[-4:] != "</w>" else tok[:-4])
            text = Doc(self.nlp.vocab, new_text)
        self.nlp.tagger(text)
        self.nlp.parser(text)
        return text


class Text(Transform):

    fields = ['SRC', 'TGT']

    def __init__(
        self,
        SRC: Optional[Union[Field, Iterable[Field]]] = None,
        TGT: Optional[Union[Field, Iterable[Field]]] = None,
    ) -> Text:
        super().__init__()

        self.SRC = SRC
        self.TGT = TGT

    @property
    def src(self):
        return self.SRC,

    @property
    def tgt(self):
        return self.TGT,

    def load(
        self,
        data: Union[str, Iterable],
        lang: Optional[str] = None,
        **kwargs
    ) -> Iterable[TextSentence]:
        r"""
        Loads the data in Text-X format.
        Also supports for loading data from Text-U file with comments and non-integer IDs.

        Args:
            data (str or Iterable):
                A filename or a list of instances.
            lang (str):
                Language code (e.g., ``en``) or language name (e.g., ``English``) for the text to tokenize.
                ``None`` if tokenization is not required.
                Default: ``None``.

        Returns:
            A list of :class:`TextSentence` instances.
        """

        if lang is not None:
            tokenizer = Tokenizer(lang)
        if isinstance(data, str) and os.path.exists(data):
            f = open(data)
            if data.endswith('.txt'):
                lines = (i
                         for s in f
                         if len(s) > 1
                         for i in StringIO((s.split() if lang is None else tokenizer(s)) + '\n'))
            else:
                lines = f
        else:
            if lang is not None:
                data = [tokenizer(s) for s in ([data] if isinstance(data, str) else data)]
            else:
                data = [data] if isinstance(data[0], str) else data
            lines = (i for s in data for i in StringIO(s + '\n'))

        index, sentence, nlp = 0, [], spacy.load("en", disable=["ner"])
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                yield TextSentence(self, sentence, index, nlp)
                index += 1
                sentence = []
            else:
                sentence.append(line)


class TextSentence(Sentence):

    def __init__(self, transform: Text, lines: List[str], index: Optional[int] = None, nlp=None) -> TextSentence:
        super().__init__(transform, index)

        self.cands = [(line+'\t').split('\t')[1] for line in lines[1:]]
        src, tgt = lines[0].split('\t')[1], self.cands[0]
        self.values = [src, tgt]

    def __repr__(self):
        self.cands = self.values[1] if isinstance(self.values[1], list) else [self.values[1]]
        lines = ['S\t' + self.values[0]]
        lines.extend(['T\t' + i for i in self.cands])
        return '\n'.join(lines) + '\n'

    @classmethod
    def align(cls, src, tgt, nlp):
        return Alignment(src, tgt, nlp=nlp).align_seq
