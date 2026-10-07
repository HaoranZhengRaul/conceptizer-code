"""The concept vocabulary and each tokenizer's concept memberships (paper Section 3, Equation 2).

A concept is a numbered list of member words (`data/concept_vocabulary.json`; numbers 0-9060).
A token belongs to every concept that lists it as a word or as the first token of a longer word
(a first-token membership), once per concept. A token in no concept belongs to the special
no-concept outcome, written here as the extra index ``C`` (the paper's bottom symbol).

The membership weights split a token's probability evenly over its concepts,

    omega[v, c] = 1[c in H(v)] / |H(v)|,

and summing per concept turns a next-token distribution into the model concept distribution
(Equation 4). Nothing here has trainable parameters.
"""
from __future__ import annotations

import collections
import json
import math
import re
import unicodedata
from pathlib import Path

import torch


def load_concepts(path):
    """Return {concept number: [member words]} from the released concept vocabulary file."""
    obj = json.loads(Path(path).read_text())
    if isinstance(obj, list):  # the released file is a one-element list holding the mapping
        assert len(obj) == 1 and isinstance(obj[0], dict), "unexpected concept file layout"
        obj = obj[0]
    return {int(c): list(words) for c, words in obj.items()}


def load_wordmap(path):
    """Return {word: {"first_id": int, "n_sub": int}} for one tokenizer.

    ``first_id`` is the first token of the word written after a space (as it appears after
    another word); ``n_sub`` is the number of tokens it takes. ``n_sub == 1`` means the word is a
    single whole-word token for this tokenizer.
    """
    return json.loads(Path(path).read_text())


# ------------------------------------------------------------------ word normalization
# Member strings are compared with judged words after the same normalization the data pipeline
# applied: NFKC, invisible characters removed, dash variants folded to "-", CJK strings dropped,
# and case duplicates inside one concept removed (the first spelling is kept).
_INVISIBLE = re.compile("[\u200b-\u200f\u00ad\u2011\u00a0\ufeff\u202a-\u202e]")
_CJK = re.compile("[\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff"
                  "\uf900-\ufaff\uff00-\uffef\uac00-\ud7af]")
_DASHES = re.compile("[\u2010-\u2015\u2212]")


def normalize_members(concepts):
    out = {}
    for c, words in concepts.items():
        seen, keep = set(), []
        for w in words:
            if _CJK.search(w):
                continue
            w2 = _DASHES.sub("-", _INVISIBLE.sub("", unicodedata.normalize("NFKC", w)))
            if not w2.strip() or w2.lower() in seen:
                continue
            seen.add(w2.lower())
            keep.append(w2)
        out[int(c)] = keep
    return out


def home_index(concepts):
    """word (lowercased) -> sorted concept numbers that list it (used to choose the closest concept)."""
    idx = collections.defaultdict(set)
    for c, words in normalize_members(concepts).items():
        for w in words:
            idx[w.lower()].add(int(c))
    return {w: sorted(cs) for w, cs in idx.items()}


def member_sets(concepts):
    """concept number -> set of its lowercased (normalized) member words."""
    return {c: {w.lower() for w in words} for c, words in normalize_members(concepts).items()}


class Memberships:
    """Concept memberships H(v) of one tokenizer, built from the concept vocabulary and its word map.

    Concepts are indexed by "lanes" 0..C-1 in increasing concept number (for the released
    vocabulary and the three tokenizers every concept has a member token, so lane == concept
    number); lane C is the no-concept outcome. ``output_size`` is the model's logit width, which can
    exceed the tokenizer's size (Qwen2.5): those extra rows are in no concept.
    """

    def __init__(self, concepts, wordmap, output_size):
        tokens_of = {}
        for c, words in concepts.items():
            toks = sorted({int(wordmap[w]["first_id"]) for w in words if wordmap.get(w)})
            if toks:
                tokens_of[int(c)] = toks
        self.concept_numbers = sorted(tokens_of)
        self.lane_of = {c: i for i, c in enumerate(self.concept_numbers)}
        self.C = len(self.concept_numbers)
        self.V = int(output_size)
        self.tokens_of = {self.lane_of[c]: t for c, t in tokens_of.items()}
        homes = collections.defaultdict(list)
        for lane in range(self.C):
            for t in self.tokens_of[lane]:
                assert 0 <= t < self.V, (t, self.V, "member token outside the output vocabulary")
                homes[t].append(lane)
        self.homes = dict(homes)
        self.uncovered = sorted(set(range(self.V)) - set(self.homes))
        # The (token, concept) membership pairs, tokens ascending and each token's concepts
        # ascending, with log omega = -log|H(v)| computed in float64.
        pairs = [(t, lane) for t in sorted(self.homes) for lane in self.homes[t]]
        self._pair_tok = torch.tensor([t for t, _ in pairs], dtype=torch.long)
        self._pair_lane = torch.tensor([lane for _, lane in pairs], dtype=torch.long)
        self._pair_n = torch.tensor([float(len(self.homes[t])) for t, _ in pairs], dtype=torch.float64)
        self._pair_log_omega = torch.tensor([-math.log(len(self.homes[t])) for t, _ in pairs],
                                            dtype=torch.float64)
        self._uncovered = torch.tensor(self.uncovered, dtype=torch.long)
        self._tables_on = {}
        self.n_pairs = len(pairs)

    # ---- H(v) and |H(v)|
    def concepts_of(self, token):
        """H(v) as lanes; the no-concept outcome [C] for a token in no concept."""
        return self.homes.get(int(token), [self.C])

    # ---- Equation 4
    def _tables(self, device, dtype):
        """The pair tables on one device, with omega = exp(log omega - logsumexp of the token's log
        omegas): 1/|H(v)| up to rounding in ``dtype`` (float32 in training), the arithmetic of the
        paper's runs, so that training repeats them exactly."""
        key = (str(device), dtype)
        if key not in self._tables_on:
            log_w = self._pair_log_omega.to(dtype).to(device)
            n = self._pair_n.to(dtype).to(device)
            omega = torch.exp(log_w - (log_w + torch.log(n)))
            self._tables_on[key] = (self._pair_tok.to(device), self._pair_lane.to(device), omega,
                                    self._uncovered.to(device))
        return self._tables_on[key]

    def concept_distribution(self, p_tokens):
        """[N, V] next-token probabilities -> [N, C+1] model concept distribution (rows sum to 1).

        p_c(c) = sum over the member tokens v of c of omega[v, c] p_w(v); the last column is the
        no-concept outcome, the total probability of the tokens in no concept. Differentiable in
        ``p_tokens``; computed in its dtype (float32 in training) over the membership pairs.
        """
        tok, lane, omega, uncovered = self._tables(p_tokens.device, p_tokens.dtype)
        n = p_tokens.shape[0]
        contrib = p_tokens.index_select(1, tok) * omega
        covered = torch.zeros((n, self.C), dtype=p_tokens.dtype, device=p_tokens.device).scatter_add(
            1, lane.unsqueeze(0).expand(n, -1), contrib)
        none = p_tokens.index_select(1, uncovered).sum(-1)
        return torch.cat([covered, none.unsqueeze(1)], dim=1)

    # ---- the one-concept "owner" of each token (used only as the closest-concept fallback)
    def owners(self):
        """Each member token's owner: normally its lowest-numbered concept; a concept that would own
        no token takes one from a concept that owns at least 2 (deterministic, in concept order)."""
        eff = {t: min(lanes) for t, lanes in self.homes.items()}
        owned = collections.defaultdict(set)
        for t, lane in eff.items():
            owned[lane].add(t)
        for lane in range(self.C):
            if self.tokens_of[lane] and not owned[lane]:
                for t in self.tokens_of[lane]:
                    donor = eff[t]
                    if donor != lane and len(owned[donor]) >= 2:
                        owned[donor].discard(t)
                        owned[lane].add(t)
                        eff[t] = lane
                        break
        return {t: self.concept_numbers[lane] for t, lane in eff.items()}
