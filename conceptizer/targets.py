"""Judge weights, contextual synonyms, the closest concept and the concept target (paper Section 3).

For one collected occurrence with observed word w_0:

* the LLM judge's kept words (its synonym pool) lose every word the tokenizer splits into
  several tokens; the rest are ranked by how many of the judge's 3 answers kept them, cut to at
  most 10 and de-duplicated by token: these are the contextual synonyms w_1..w_m;
* each word has a judge weight r_j = (k_j + 1) / (n_j + 2) (Equation 1), counted over all collected
  occurrences of w_0's lemma and part of speech in which w_j was proposed (n_j) and kept (k_j);
  w_0 has r_0 = 1;
* the closest concept c* is, among the concepts listing w_0, the one listing the most of
  w_1..w_m (ties to the lowest number; if none lists any, the concept owning w_0's token);
* the concept target is Equation 3:  q = sum_j rho_j [ sigma_j e_{c*} + (1 - sigma_j) omega_{w_j} ],
  with rho_j = r_j / sum_l r_l and sigma_j = 1 for w_0, 1/2 for a synonym whose token is a member
  of c*, 0 otherwise.

The released training targets store, per occurrence, the token ids of w_0..w_m (``set_ids``), their
judge weights (``set_kr``), the observed token (``gold_id``) and c* (``home_lane``);
:func:`concept_target` turns those four fields into q, the target the paper's models trained on.
"""
from __future__ import annotations

import collections
import math
import struct

GOLD_WEIGHT = 1.0      # r_0: the observed word
RESIDUAL = 0.5         # a synonym with no recorded counts (never reached with the released table)
MAX_SYNONYMS = 10      # the cap of Zhang et al.
SIGMA_SYNONYM_IN_CLOSEST = 0.5


# ------------------------------------------------------------------ Equation 1
def judge_weight(kept, proposed):
    """The judge weight (k + 1) / (n + 2) (Laplace smoothing of the share kept), stored to 4 decimals."""
    return round((kept + 1.0) / (proposed + 2.0), 4)


def judge_weight_table(judge_rows):
    """{ "lemma|POS": {word: r} } from the judge's rows (one row per collected occurrence).

    A word counts as proposed in an occurrence when a source offered it or the judge added it,
    and as kept when any of the judge's 3 answers kept it.
    """
    seen = collections.defaultdict(collections.Counter)
    kept = collections.defaultdict(collections.Counter)
    for row in judge_rows:
        t = row.get("target") or {}
        key = f"{(t.get('lemma') or t.get('word') or '').lower()}|{t.get('pos')}"
        k = {w.lower() for w in ((row.get("lm_disc") or {}).get("kept") or {})}
        pool = {c.lower() for c in (row.get("pool") or [])}
        for w in pool | k:
            seen[key][w] += 1
        for w in k:
            kept[key][w] += 1
    return {key: {w: judge_weight(kept[key][w], n) for w, n in counts.items()}
            for key, counts in seen.items()}


def judge_weights_for(lemma_key, synonyms, table):
    """[r_1..r_m] for the given synonym words (lowercase lookup)."""
    row = table.get(lemma_key) or {}
    return [float(row.get(s.lower(), RESIDUAL)) for s in synonyms]


# ------------------------------------------------------------------ w_1..w_m
def word_pieces(tokenizer):
    """The token ids of a word written after a space, as it appears after another word."""
    def pieces(word):
        return tokenizer(" " + word, add_special_tokens=False)["input_ids"]
    return pieces


def contextual_synonyms(kept, gold_word, gold_id, pieces):
    """The occurrence's contextual synonyms, in their stored order (token ids ascending).

    kept      the judge's kept words with their vote fractions, e.g. {"protection": 1.0, "care": 0.333}
    gold_id   the observed token as tokenized in the document
    pieces    word -> token ids with a leading space (:func:`word_pieces`)

    Returns (set_ids, words): token ids of w_0..w_m sorted ascending and the aligned words.
    """
    single = {w: s for w, s in kept.items() if len(pieces(w)) == 1}   # multi-token synonyms dropped
    ranked = sorted(single.items(), key=lambda kv: -kv[1])          # stable: ties keep judge order
    chosen = [(w, s) for w, s in ranked if s > 0][:MAX_SYNONYMS]
    ids, words = [int(gold_id)], [gold_word]
    for w, _ in chosen:
        fid = int(pieces(w)[0])
        if fid not in ids:                                           # one entry per token
            ids.append(fid)
            words.append(w)
    order = sorted(range(len(ids)), key=lambda i: ids[i])
    return [ids[i] for i in order], [words[i] for i in order]


# ------------------------------------------------------------------ c*
def closest_concept(gold_word, words, homes, members, fallback):
    """c*: among the concepts listing ``gold_word``, the one listing the most of the other ``words``.

    homes     word (lowercased) -> concepts listing it (vocabulary.home_index)
    members   concept -> its lowercased member words (vocabulary.member_sets)
    fallback  the concept that owns the observed word's token (Memberships.owners())
    Ties go to the lowest concept number. If no concept of the word lists a synonym, the fallback
    is used when it lists the word, otherwise the lowest-numbered concept that lists it.
    """
    candidates = homes.get(gold_word.lower())
    if not candidates:
        return fallback
    if len(candidates) == 1:
        return candidates[0]
    others = {w.lower() for w in words} - {gold_word.lower()}
    best_n, best_c = max((len(members[c] & others), -c) for c in candidates)
    if best_n == 0:
        return fallback if fallback in candidates else candidates[0]
    return -best_c


# ------------------------------------------------------------------ Equation 3
def concept_target(set_ids, set_kr, gold_id, home_lane, memberships,
                   sigma=SIGMA_SYNONYM_IN_CLOSEST):
    """The concept target q as {lane: probability} (lane ``memberships.C`` is the no-concept outcome).

    set_ids/set_kr  token ids of w_0..w_m and their judge weights r_j (r_0 = 1 on the observed token)
    home_lane       the closest concept c* (a concept number)
    """
    assert len(set_ids) == len(set_kr) and int(gold_id) in set_ids
    tot = math.fsum(float(r) for r in set_kr)
    c_star = memberships.lane_of[int(home_lane)]
    assert c_star in memberships.concepts_of(gold_id), \
        "the closest concept must contain the observed token (eligibility rule 5)"
    q = collections.defaultdict(float)
    for t, r in zip(set_ids, set_kr):
        share = _float32(float(r) / tot)              # rho_j, kept in float32 as in training
        if int(t) == int(gold_id):                    # sigma_0 = 1
            q[c_star] += share
            continue
        lanes = memberships.homes.get(int(t))
        if not lanes:                                 # a synonym in no concept
            q[memberships.C] += share
            continue
        if c_star in lanes:                           # sigma_j = 1/2
            q[c_star] += share * sigma
            share *= 1.0 - sigma
        for lane in lanes:                            # the rest evenly over all of H(w_j)
            q[lane] += share / len(lanes)
    return dict(sorted(q.items()))


def _float32(x):
    """x rounded to the nearest float32 value."""
    return struct.unpack("f", struct.pack("f", x))[0]


def target_from_entry(entry, memberships):
    """Concept target of one stored training entry (fields set_ids, set_kr, gold_id, home_lane)."""
    return concept_target(entry["set_ids"], entry["set_kr"], entry["gold_id"],
                          entry["home_lane"], memberships)
