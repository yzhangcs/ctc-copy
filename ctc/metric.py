# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import os
import tempfile
from collections import Counter
from typing import Any, List, Optional, Set, Tuple

import torch
from errant import Annotator

from supar.structs.fn import levenshtein
from supar.utils.metric import Metric


class PerplexityMetric(Metric):

    def __init__(
        self,
        loss: Optional[float] = None,
        preds: Optional[Tuple[torch.Tensor, List, List]] = None,
        golds: Optional[Tuple[torch.Tensor, List, List]] = None,
        mask: Optional[torch.BoolTensor] = None,
        annotator: Annotator = None,
        reverse: bool = False,
        eps: float = 1e-12
    ) -> PerplexityMetric:
        super().__init__(reverse=reverse, eps=eps)

        self.n_tokens = 0.

        self.tp = 0.0
        self.pred = 0.0
        self.gold = 0.0
        self.total_loss = 0.

        if loss is not None:
            self(loss, preds, golds, mask, annotator)

    def __repr__(self):
        s = f"loss: {self.loss:.4f} PPL: {self.ppl:.4f}"
        if self.tp > 0:
            s += f" - TGT: P: {self.p:6.2%} R: {self.r:6.2%} F0.5: {self.f:6.2%}"
        return s

    def __call__(
        self,
        loss: float,
        preds: Tuple[torch.Tensor, List, List],
        golds: Tuple[torch.Tensor, List, List],
        mask: torch.BoolTensor,
        annotator: Any
    ) -> PerplexityMetric:
        n_tokens = mask.sum().item()
        self.n += len(mask)
        self.count += 1
        self.n_tokens += n_tokens
        self.total_loss += float(loss) * n_tokens

        if preds is not None:
            if annotator is not None:
                with tempfile.TemporaryDirectory() as t:
                    fsrc, fpred, fgold = os.path.join(t, 'src'), os.path.join(t, 'pred'), os.path.join(t, 'gold')
                    pred_m2, gold_m2 = os.path.join(t, 'pred.m2'), os.path.join(t, 'gold.m2')
                    with open(fsrc, 'w') as fs, open(fpred, 'w') as f:
                        for s, i, *_ in preds:
                            fs.write(s + '\n')
                            f.write(i + '\n')
                    with open(fgold, 'w') as f:
                        for _, i, *_ in golds:
                            f.write(i + '\n')
                    self.errant_parallel(fsrc, fpred, pred_m2, annotator)
                    self.errant_parallel(fsrc, fgold, gold_m2, annotator)
                    out = self.errant_compare(pred_m2, gold_m2)
                    tp, fp, fn = out['tp'], out['fp'], out['fn']
                    self.tp += tp
                    self.pred += tp + fp
                    self.gold += tp + fn
            else:
                for p, g in zip(preds, golds):
                    e_p = self.compare(p[2], p[3])
                    e_g = self.compare(g[2], g[3])
                    self.tp += len(e_p & e_g)
                    self.pred += len(e_p)
                    self.gold += len(e_g)
        return self

    def __add__(self, other: PerplexityMetric) -> PerplexityMetric:
        metric = PerplexityMetric(eps=self.eps)
        metric.n = self.n + other.n
        metric.count = self.count + other.count
        metric.n_tokens = self.n_tokens + other.n_tokens
        metric.total_loss = self.total_loss + other.total_loss

        metric.tp = self.tp + other.tp
        metric.pred = self.pred + other.pred
        metric.gold = self.gold + other.gold
        metric.reverse = self.reverse or other.reverse
        return metric

    @property
    def score(self):
        return self.f

    @property
    def loss(self):
        return self.total_loss / self.n_tokens

    @property
    def ppl(self):
        return math.pow(2, (self.loss / math.log(2)))

    @property
    def p(self):
        return self.tp / (self.pred + self.eps)

    @property
    def r(self):
        return self.tp / (self.gold + self.eps)

    @property
    def f(self):
        return (1 + 0.5**2) * self.p * self.r / (0.5**2 * self.p + self.r + self.eps)

    @property
    def values(self):
        return {'P': self.p,
                'R': self.r,
                'F0.5': self.f}

    def compare(self, s, t) -> Set:
        return {(i, edit) for i, _, edit in levenshtein(s, t, align=True)[1] if edit != 0}

    def errant_parallel(self, forig: str, fcor: str, fout: str, annotator: Any) -> None:
        from contextlib import ExitStack

        def noop_edit(id=0):
            return "A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||"+str(id)
        with ExitStack() as stack, open(fout, "w") as out_m2:
            in_files = [stack.enter_context(open(i)) for i in [forig]+[fcor]]
            # Process each line of all input files
            for line in zip(*in_files):
                # Get the original and all the corrected texts
                orig = line[0].strip()
                cors = line[1:]
                # Skip the line if orig is empty
                if not orig:
                    continue
                # Parse orig with spacy
                orig = annotator.parse(orig)
                # Write orig to the output m2 file
                out_m2.write(" ".join(["S"]+[token.text for token in orig])+"\n")
                # Loop through the corrected texts
                for cor_id, cor in enumerate(cors):
                    cor = cor.strip()
                    # If the texts are the same, write a noop edit
                    if orig.text.strip() == cor:
                        out_m2.write(noop_edit(cor_id)+"\n")
                    # Otherwise, do extra processing
                    else:
                        # Parse cor with spacy
                        cor = annotator.parse(cor)
                        # Align the texts and extract and classify the edits
                        edits = annotator.annotate(orig, cor)
                        # Loop through the edits
                        for edit in edits:
                            # Write the edit to the output m2 file
                            out_m2.write(edit.to_m2(cor_id)+"\n")
                # Write a newline when we have processed all corrections for each line
                out_m2.write("\n")

    def errant_compare(self, fhyp: str, fref: str):
        from argparse import Namespace

        # Input: An m2 format sentence with edits.
        # Output: A list of lists. Each edit: [start, end, cat, cor, coder]

        def simplify_edits(sent):
            out_edits = []
            # Get the edit lines from an m2 block.
            edits = sent.split("\n")[1:]
            # Loop through the edits
            for edit in edits:
                # Preprocessing
                edit = edit[2:].split("|||")  # Ignore "A " then split.
                span = edit[0].split()
                start = int(span[0])
                end = int(span[1])
                cat = edit[1]
                cor = edit[2]
                coder = int(edit[-1])
                out_edit = [start, end, cat, cor, coder]
                out_edits.append(out_edit)
            return out_edits

        # Input 1: A list of edits. Each edit: [start, end, cat, cor, coder]
        # Output: A dict; key is coder, value is edit dict.
        def process_edits(edits, args):
            coder_dict = {}
            # Add an explicit noop edit if there are no edits.
            if not edits:
                edits = [[-1, -1, "noop", "-NONE-", 0]]
            # Loop through the edits
            for edit in edits:
                # Name the edit elements for clarity
                start = edit[0]
                end = edit[1]
                cat = edit[2]
                cor = edit[3]
                coder = edit[4]
                # Add the coder to the coder_dict if necessary
                if coder not in coder_dict:
                    coder_dict[coder] = {}

                # Optionally apply filters based on args
                # 1. UNK type edits are only useful for detection, not correction.
                if not args.dt and not args.ds and cat == "UNK":
                    continue
                # 2. Only evaluate single token edits; i.e. 0:1, 1:0 or 1:1
                if args.single and (end-start >= 2 or len(cor.split()) >= 2):
                    continue
                # 3. Only evaluate multi token edits; i.e. 2+:n or n:2+
                if args.multi and end-start < 2 and len(cor.split()) < 2:
                    continue
                # 4. If there is a filter, ignore the specified error types
                if args.filt and cat in args.filt:
                    continue

                # Token Based Detection
                if args.dt:
                    # Preserve noop edits.
                    if start == -1:
                        if (start, start) in coder_dict[coder].keys():
                            coder_dict[coder][(start, start)].append(cat)
                        else:
                            coder_dict[coder][(start, start)] = [cat]
                    # Insertions defined as affecting the token on the right
                    elif start == end and start >= 0:
                        if (start, start+1) in coder_dict[coder].keys():
                            coder_dict[coder][(start, start+1)].append(cat)
                        else:
                            coder_dict[coder][(start, start+1)] = [cat]
                    # Edit spans are split for each token in the range.
                    else:
                        for tok_id in range(start, end):
                            if (tok_id, tok_id+1) in coder_dict[coder].keys():
                                coder_dict[coder][(tok_id, tok_id+1)].append(cat)
                            else:
                                coder_dict[coder][(tok_id, tok_id+1)] = [cat]

                # Span Based Detection
                elif args.ds:
                    if (start, end) in coder_dict[coder].keys():
                        coder_dict[coder][(start, end)].append(cat)
                    else:
                        coder_dict[coder][(start, end)] = [cat]

                # Span Based Correction
                else:
                    # With error type classification
                    if args.cse:
                        if (start, end, cat, cor) in coder_dict[coder].keys():
                            coder_dict[coder][(start, end, cat, cor)].append(cat)
                        else:
                            coder_dict[coder][(start, end, cat, cor)] = [cat]
                    # Without error type classification
                    else:
                        if (start, end, cor) in coder_dict[coder].keys():
                            coder_dict[coder][(start, end, cor)].append(cat)
                        else:
                            coder_dict[coder][(start, end, cor)] = [cat]
            return coder_dict

        # Input 1-3: True positives, false positives, false negatives
        # Input 4: Value of beta in F-score.
        # Output 1-3: Precision, Recall and F-score rounded to 4dp.

        def computeFScore(tp, fp, fn, beta):
            p = float(tp)/(tp+fp) if fp else 1.0
            r = float(tp)/(tp+fn) if fn else 1.0
            f = float((1+(beta**2))*p*r)/(((beta**2)*p)+r) if p+r else 0.0
            return round(p, 4), round(r, 4), round(f, 4)
        # Input 1: A hyp dict; key is coder_id, value is dict of processed hyp edits.
        # Input 2: A ref dict; key is coder_id, value is dict of processed ref edits.
        # Input 3: A dictionary of the best corpus level TP, FP and FN counts so far.
        # Input 4: Sentence ID (for verbose output only)
        # Input 5: Command line args
        # Output 1: A dict of the best corpus level TP, FP and FN for the input sentence.
        # Output 2: The corresponding error type dict for the above dict.

        # Input 1: A dictionary of hypothesis edits for a single system.
        # Input 2: A dictionary of reference edits for a single annotator.
        # Output 1-3: The TP, FP and FN for the hyp vs the given ref annotator.
        # Output 4: A dictionary of the error type counts.
        def compareEdits(hyp_edits, ref_edits):
            tp = 0    # True Positives
            fp = 0    # False Positives
            fn = 0    # False Negatives
            cat_dict = {}  # {cat: [tp, fp, fn], ...}

            for h_edit, h_cats in hyp_edits.items():
                # noop hyp edits cannot be TP or FP
                if h_cats[0] == "noop":
                    continue
                # TRUE POSITIVES
                if h_edit in ref_edits.keys():
                    # On occasion, multiple tokens at same span.
                    for h_cat in ref_edits[h_edit]:  # Use ref dict for TP
                        tp += 1
                        # Each dict value [TP, FP, FN]
                        if h_cat in cat_dict.keys():
                            cat_dict[h_cat][0] += 1
                        else:
                            cat_dict[h_cat] = [1, 0, 0]
                # FALSE POSITIVES
                else:
                    # On occasion, multiple tokens at same span.
                    for h_cat in h_cats:
                        fp += 1
                        # Each dict value [TP, FP, FN]
                        if h_cat in cat_dict.keys():
                            cat_dict[h_cat][1] += 1
                        else:
                            cat_dict[h_cat] = [0, 1, 0]
            for r_edit, r_cats in ref_edits.items():
                # noop ref edits cannot be FN
                if r_cats[0] == "noop":
                    continue
                # FALSE NEGATIVES
                if r_edit not in hyp_edits.keys():
                    # On occasion, multiple tokens at same span.
                    for r_cat in r_cats:
                        fn += 1
                        # Each dict value [TP, FP, FN]
                        if r_cat in cat_dict.keys():
                            cat_dict[r_cat][2] += 1
                        else:
                            cat_dict[r_cat] = [0, 0, 1]
            return tp, fp, fn, cat_dict

        def evaluate_edits(hyp_dict, ref_dict, best, sent_id, original_sentence, args):
            # Store the best sentence level scores and hyp+ref combination IDs
            # best_f is initialised as -1 cause 0 is a valid result.
            best_tp, best_fp, best_fn, best_f, _, _ = 0, 0, 0, -1, 0, 0
            best_cat = {}
            # Compare each hyp and ref combination
            for hyp_id in hyp_dict.keys():
                for ref_id in ref_dict.keys():
                    # Get the local counts for the current combination.
                    tp, fp, fn, cat_dict = compareEdits(hyp_dict[hyp_id], ref_dict[ref_id])
                    # Compute the local sentence scores (for verbose output only)
                    loc_p, loc_r, loc_f = computeFScore(tp, fp, fn, args.beta)
                    # Compute the global sentence scores
                    p, r, f = computeFScore(
                        tp+best["tp"], fp+best["fp"], fn+best["fn"], args.beta)
                    # Save the scores if they are better in terms of:
                    # 1. Higher F-score
                    # 2. Same F-score, higher TP
                    # 3. Same F-score and TP, lower FP
                    # 4. Same F-score, TP and FP, lower FN
                    if (f > best_f) or \
                        (f == best_f and tp > best_tp) or \
                        (f == best_f and tp == best_tp and fp < best_fp) or \
                            (f == best_f and tp == best_tp and fp == best_fp and fn < best_fn):
                        best_tp, best_fp, best_fn = tp, fp, fn
                        best_f, _, _ = f, hyp_id, ref_id
                        best_cat = cat_dict
            # Save the best TP, FP and FNs as a dict, and return this and the best_cat dict
            best_dict = {"tp": best_tp, "fp": best_fp, "fn": best_fn}
            return best_dict, best_cat

        def merge_dict(dict1, dict2):
            for cat, stats in dict2.items():
                if cat in dict1.keys():
                    dict1[cat] = [x+y for x, y in zip(dict1[cat], stats)]
                else:
                    dict1[cat] = stats
            return dict1
        args = Namespace(beta=0.5,
                         dt=False,
                         ds=False,
                         cs=False,
                         cse=False,
                         single=False,
                         multi=False,
                         filt=[],
                         cat=1)
        # Open hypothesis and reference m2 files and split into chunks
        with open(fhyp) as fhyp, open(fref) as fref:
            hyp_m2 = fhyp.read().strip().split("\n\n")
            ref_m2 = fref.read().strip().split("\n\n")
        # Make sure they have the same number of sentences
        assert len(hyp_m2) == len(ref_m2)

        # Store global corpus level best counts here
        best_dict = Counter({"tp": 0, "fp": 0, "fn": 0})
        best_cats = {}
        # Process each sentence
        sents = zip(hyp_m2, ref_m2)
        for sent_id, sent in enumerate(sents):
            # Simplify the edits into lists of lists
            hyp_edits = simplify_edits(sent[0])
            ref_edits = simplify_edits(sent[1])
            # Process the edits for detection/correction based on args
            hyp_dict = process_edits(hyp_edits, args)
            ref_dict = process_edits(ref_edits, args)
            # original sentence for logging
            original_sentence = sent[0][2:].split("\nA")[0]
            # Evaluate edits and get best TP, FP, FN hyp+ref combo.
            count_dict, cat_dict = evaluate_edits(
                hyp_dict, ref_dict, best_dict, sent_id, original_sentence, args)
            # Merge these dicts with best_dict and best_cats
            best_dict += Counter(count_dict)
            best_cats = merge_dict(best_cats, cat_dict)
        return best_dict


class ExactMatchMetric(Metric):

    def __init__(
        self,
        loss: Optional[float] = None,
        preds: Optional[Tuple[torch.Tensor, List, List]] = None,
        golds: Optional[Tuple[torch.Tensor, List, List]] = None,
        mask: Optional[torch.BoolTensor] = None,
        reverse: bool = True,
        eps: float = 1e-12
    ) -> ExactMatchMetric:
        super().__init__(reverse=reverse, eps=eps)

        self.n_tokens = 0.

        self.tp = 0.0
        self.total = 0.0
        self.total_loss = 0.

        if loss is not None:
            self(loss, preds, golds, mask)

    def __repr__(self):
        return f"loss: {self.loss:.4f} EM: {self.em:6.2%}"

    def __call__(
        self,
        loss: float,
        preds: Tuple[torch.Tensor, List, List],
        golds: Tuple[torch.Tensor, List, List],
        mask: torch.BoolTensor
    ) -> ExactMatchMetric:
        n_tokens = mask.sum().item()
        self.n += len(mask)
        self.count += 1
        self.n_tokens += n_tokens
        self.total_loss += float(loss) * n_tokens

        if preds is not None:
            self.tp += sum([p[3].equal(g[3]) for p, g in zip(preds, golds)])
            self.total += len(preds)
        return self

    def __add__(self, other: ExactMatchMetric) -> ExactMatchMetric:
        metric = ExactMatchMetric(eps=self.eps)
        metric.n = self.n + other.n
        metric.count = self.count + other.count
        metric.n_tokens = self.n_tokens + other.n_tokens
        metric.total_loss = self.total_loss + other.total_loss

        metric.tp = self.tp + other.tp
        metric.total = self.total + other.total
        metric.reverse = self.reverse or other.reverse
        return metric

    @property
    def score(self):
        return self.em

    @property
    def loss(self):
        return self.total_loss / self.n_tokens

    @property
    def em(self):
        return self.tp / (self.total + self.eps)

    @property
    def values(self):
        return {'EM': self.em}
