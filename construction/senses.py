"""Build the concept vocabulary from judged synonym pools (paper Section 3, Appendix C).

Input: the concept-construction sample, one record per collected occurrence:
    {"key": "defence|NOUN", "gold": "defence",       # the word's lemma|POS and the word as written
     "kept": {"protection": 1.0, "shield": 0.333},   # the judge's kept words (fraction of answers)
     "induction": true}                               # in the 60% of documents that build concepts

1. Senses. Within each word, occurrences are compared by the Dice similarity 2|A&B| / (|A| + |B|)
   of their kept words after folding inflections (WordNet morphy under the word's POS, then any
   other POS, then suffix rules), and grouped by average linkage at Dice >= 0.15. Words with fewer
   than 5 induction occurrences build no sense. A sense's final pool is the union of its occurrences'
   kept words (surface forms) plus the word as written in each occurrence.
2. Concepts. Senses of different words form a candidate pair when their final pools (lowercased)
   share at least 2 words and reach Jaccard |A&B| / |A|B| >= 1/3; a word in more than 200 final
   pools is not used to look up pairs. Pairs are taken in descending similarity and complete linkage
   merges 2 groups only when every pair between them qualifies. Two senses of one word never merge.
3. Every occurrence takes the label of its sense's concept (non-induction occurrences are routed to
   the sense whose words they share most); concepts are numbered by first appearance in the sample,
   and a concept's member words are the union of its induction occurrences' kept words and words as
   written.

WordNet folding depends on the NLTK version: use nltk==3.9.2.
"""
from __future__ import annotations

import collections
import itertools

import numpy as np

DICE_THRESHOLD = 0.15
JACCARD_THRESHOLD = 1 / 3
MIN_SHARED_WORDS = 2
MAX_POOLS_PER_LOOKUP_WORD = 200
MIN_INDUCTION_OCCURRENCES = 5
MORPHY_POS = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}
_KEEP_S = ("ss", "us", "is", "ous", "ness")       # a final s that is part of the word
_FOLD = {}


def _wordnet():
    from nltk.corpus import wordnet
    wordnet.ensure_loaded()
    return wordnet


def crude_stem(w):
    """Suffix rules for words WordNet does not know."""
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    for suf in ("ing", "ed"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[:-len(suf)]
    if w.endswith("es") and len(w) > 4 and w[:-2].endswith(("s", "x", "z", "ch", "sh")):
        return w[:-2]
    if w.endswith("s") and len(w) > 3 and not w.endswith(_KEEP_S):
        return w[:-1]
    return w


def fold_word(w, pos):
    """The form used only to compare kept sets (members keep their surface forms)."""
    k = (w, pos)
    if k not in _FOLD:
        wn, lw = _wordnet(), w.lower()
        tag = MORPHY_POS.get(pos, "n")
        m = wn.morphy(lw, tag)
        if m:
            _FOLD[k] = m
        else:
            alt = next((p for p in "nvar" if p != tag and wn.morphy(lw, p)), None)
            _FOLD[k] = wn.morphy(lw, alt) if alt else crude_stem(lw)
    return _FOLD[k]


def dice(a, b):
    inter = len(a & b)
    return 2 * inter / (len(a) + len(b)) if inter else 0.0


def cluster_occurrences(sets, thr=DICE_THRESHOLD):
    """Average-linkage groups of one word's occurrences at Dice >= thr (labels 1..k)."""
    n = len(sets)
    if n <= 1:
        return np.zeros(n, dtype=np.int32)
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = 1.0 - dice(sets[i], sets[j])
    Z = linkage(squareform(D, checks=False), method="average")
    return fcluster(Z, t=1.0 - thr, criterion="distance").astype(np.int32)


def route_by_kept(kept_row, members_of_word):
    """The word's sense sharing the most words with this occurrence (ties: lowest), -1 if none."""
    best, best_ov = -1, 0
    for cid in sorted(members_of_word):
        ov = sum(1 for w in kept_row if w in members_of_word[cid])
        if ov > best_ov:
            best, best_ov = cid, ov
    return best if best_ov > 0 else -1


def build_senses(records):
    """Step 1: per-word senses. Returns (sense label per record, sense -> final pool, sense -> key)."""
    induct_by_key, all_by_key = collections.defaultdict(list), collections.defaultdict(list)
    for i, r in enumerate(records):
        all_by_key[r["key"]].append(i)
        if r["induction"]:
            induct_by_key[r["key"]].append(i)
    labels = np.full(len(records), -1, dtype=np.int32)
    members, key_of, base = {}, {}, 0
    for key in sorted(induct_by_key):
        idx = sorted(induct_by_key[key])
        if len(idx) < MIN_INDUCTION_OCCURRENCES:
            continue
        pos = key.rsplit("|", 1)[-1]
        sets = [{fold_word(w, pos) for w in records[i]["kept"]} for i in idx]
        groups = collections.defaultdict(list)
        for j, lab in enumerate(cluster_occurrences(sets)):
            groups[int(lab)].append(j)
        for k, (_, js) in enumerate(sorted(groups.items())):
            sid = base + k
            pool = set()
            for j in js:
                i = idx[j]
                labels[i] = sid
                pool |= set(records[i]["kept"])
                pool.add(records[i]["gold"])
            members[sid], key_of[sid] = sorted(pool), key
        base += len(groups)
    by_key = collections.defaultdict(dict)
    for sid, key in key_of.items():
        by_key[key][sid] = set(members[sid])
    for key, idx in all_by_key.items():                 # route the remaining occurrences
        for i in idx:
            if labels[i] < 0 and by_key.get(key):
                row = dict(records[i]["kept"]); row[records[i]["gold"]] = 1.0
                labels[i] = route_by_kept(row, by_key[key])
    return labels, members, key_of


def merge_senses(members, key_of, jaccard=JACCARD_THRESHOLD):
    """Step 2: complete-linkage merge of senses of different words. Returns groups of sense ids."""
    sets = {c: {w.lower() for w in ws} for c, ws in members.items()}
    inv = collections.defaultdict(list)
    for c, s in sets.items():
        for w in s:
            inv[w].append(c)
    edges, seen = [], set()
    for w in sorted(inv):
        cs = inv[w]
        if len(cs) > MAX_POOLS_PER_LOOKUP_WORD:
            continue
        for a, b in itertools.combinations(sorted(cs), 2):
            if (a, b) in seen:
                continue
            seen.add((a, b))
            if key_of.get(a) == key_of.get(b):          # two senses of one word never pair
                continue
            inter = len(sets[a] & sets[b])
            if inter >= MIN_SHARED_WORDS:
                j = inter / (len(sets[a]) + len(sets[b]) - inter)
                if j >= jaccard:
                    edges.append((round(j, 6), a, b))
    edges.sort(key=lambda e: (-e[0], e[1], e[2]))
    ok = {(a, b) for _, a, b in edges} | {(b, a) for _, a, b in edges}
    group_of, grp = {c: c for c in sets}, {c: [c] for c in sets}
    for _, a, b in edges:
        ga, gb = group_of[a], group_of[b]
        if ga == gb or not all((x, y) in ok for x in grp[ga] for y in grp[gb]):
            continue
        keep, drop = (ga, gb) if ga < gb else (gb, ga)
        for c in grp[drop]:
            group_of[c] = keep
        grp[keep].extend(grp.pop(drop))
    groups = sorted((sorted(v) for v in grp.values()), key=lambda g: g[0])
    out = []                                            # detach any second sense of one word
    for g in groups:
        by_word = collections.defaultdict(list)
        for c in g:
            by_word[key_of.get(c)].append(c)
        dup = [c for k, cs in by_word.items() if k is not None and len(cs) >= 2 for c in cs]
        out.append(sorted(set(g) - set(dup)))
        out.extend([[c] for c in dup])
    return sorted((g for g in out if g), key=lambda g: g[0]), len(edges)


def build_concept_vocabulary(records):
    """Steps 1-3. Returns {concept number: sorted member words} (numbers 0..K-1)."""
    labels, members, key_of = build_senses(records)
    groups, _ = merge_senses(members, key_of)
    to_group = {c: gi for gi, g in enumerate(groups) for c in g}
    merged = np.array([to_group.get(int(l), -1) if l >= 0 else -1 for l in labels], dtype=np.int64)
    number, out = {}, collections.defaultdict(set)
    for i, lab in enumerate(merged):                    # number by first appearance
        if lab >= 0 and lab not in number:
            number[lab] = len(number)
    for i, lab in enumerate(merged):
        if lab >= 0 and records[i]["induction"]:
            out[number[lab]].update(records[i]["kept"])
            out[number[lab]].add(records[i]["gold"])
    return {c: sorted(ws) for c, ws in sorted(out.items())}


def read_sample(path):
    """The concept-construction sample's records, in their order (the order numbers the concepts)."""
    import gzip
    import json
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main(argv=None):
    import argparse
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", default=str(root / "data/construction/construction_sample.jsonl.gz"))
    ap.add_argument("--out", required=True, help="where to write the rebuilt vocabulary (JSON)")
    a = ap.parse_args(argv)
    vocab = build_concept_vocabulary(read_sample(a.sample))
    Path(a.out).write_text(json.dumps([{str(c): ws for c, ws in vocab.items()}]))
    print(f"{len(vocab)} concepts")


if __name__ == "__main__":
    main()
