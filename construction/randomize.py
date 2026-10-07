"""The Randomized-synonyms control (paper Section 5): the same training file with scrambled targets.

Every concept-target entry keeps its observed word (token, position, judge weight 1) and receives
the contextual synonyms w_1..w_m, with their judge weights, of another entry: entries are grouped by
(file role, number of words), and each group is shuffled by a seeded derangement (seed 0) under two
hard constraints, no entry keeps its own synonyms and none receives a block containing its own
observed token, and a soft one, the donor's observed word is a different word. The closest concept
c* is then redrawn uniformly from the concepts of the observed token (its concept memberships), with
one random generator per entry address. Windows without a concept target are copied unchanged.

    python -m construction.randomize --family llama3 --source data/llama3 --out data/llama3_randomized
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import unicodedata
from pathlib import Path

from conceptizer import vocabulary as vb

ROOT = Path(__file__).resolve().parents[1]
ROLES = (("train", "train_windows.jsonl"), ("development", "development_windows.jsonl"),
         ("final", "final_windows.jsonl"))
SEED_NAME = {"train": "train", "development": "gov", "final": "adj"}   # the names the seeds were drawn with
ATTEMPTS = 64


def row_rng(seed, role, wi, si):
    h = hashlib.sha256(f"{int(seed)}|{role}|{int(wi)}|{int(si)}".encode()).hexdigest()[:16]
    return random.Random(int(h, 16))


def surface(word):
    return unicodedata.normalize("NFKC", str(word)).strip().casefold()


def _matching(n, allowed):
    """A perfect matching of entries to blocks over the allowed pairs (Kuhn's algorithm), or None."""
    owner = [-1] * n

    def augment(i, seen):
        for j in range(n):
            if not seen[j] and allowed(i, j):
                seen[j] = True
                if owner[j] == -1 or augment(owner[j], seen):
                    owner[j] = i
                    return True
        return False
    for i in range(n):
        if not augment(i, [False] * n):
            return None
    perm = [0] * n
    for j, i in enumerate(owner):
        perm[i] = j
    return perm


def class_permutation(seed_text, gold_ids, gold_surfaces, blocks):
    n = len(gold_ids)
    if n <= 1:
        return [0] * n
    rng = random.Random(int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16))

    def ok(i, j):
        return j != i and gold_ids[i] not in blocks[j]

    def ok_soft(i, j):
        return ok(i, j) and gold_surfaces[i] != gold_surfaces[j]
    perm = list(range(n))
    for _ in range(ATTEMPTS):                         # 1. rejection sampling for all 3 constraints
        rng.shuffle(perm)
        if all(ok_soft(i, perm[i]) for i in range(n)):
            return perm
    perm = list(range(n))                             # 2. Sattolo derangement + swap repair
    for i in range(n - 1, 0, -1):
        j = rng.randrange(i)
        perm[i], perm[j] = perm[j], perm[i]

    def swap_until(i, good):
        for _ in range(ATTEMPTS):
            j = rng.randrange(n)
            if j != i and good(i, perm[j]) and good(j, perm[i]):
                perm[i], perm[j] = perm[j], perm[i]
                return
        for j in range(n):
            if j != i and good(i, perm[j]) and good(j, perm[i]):
                perm[i], perm[j] = perm[j], perm[i]
                return
    for i in range(n):
        if not ok(i, perm[i]):
            swap_until(i, ok)
    if any(not ok(i, perm[i]) for i in range(n)):     # 3. exhaustive matching
        perm = _matching(n, ok)
        assert perm is not None, "no valid assignment exists for this group"
    for i in range(n):                                # 4. soft constraint where possible
        if not ok_soft(i, perm[i]):
            swap_until(i, ok_soft)
    return perm


def scramble(files, memberships, seed=0):
    """files: {role: [window, ...]} -> the same windows with scrambled concept targets."""
    token_homes = {t: sorted(memberships.concept_numbers[l] for l in lanes)
                   for t, lanes in memberships.homes.items()}
    rows = []
    for role, _ in ROLES:
        for wi, w in enumerate(files[role]):
            for si, e in enumerate(w.get("sup") or []):
                if "set_ids" in e:
                    gi = [int(x) for x in e["set_ids"]].index(int(e["gold_id"]))
                    rows.append((role, wi, si, e, gi, (SEED_NAME[role], len(e["set_ids"]), bool(e.get("set_kr")))))
    classes = collections.defaultdict(list)
    for k, r in enumerate(rows):
        classes[r[5]].append(k)
    donor = [None] * len(rows)
    for key in sorted(classes):
        mem = classes[key]
        golds = [int(rows[k][3]["gold_id"]) for k in mem]
        surfs = [surface(rows[k][3]["words"][rows[k][4]]) for k in mem]
        blocks = [frozenset(int(x) for i, x in enumerate(rows[k][3]["set_ids"]) if i != rows[k][4]) for k in mem]
        perm = class_permutation(f"{int(seed)}|{key[0]}|{int(key[1])}|kr{int(key[2])}", golds, surfs, blocks)
        for p, k in enumerate(mem):
            donor[k] = mem[perm[p]]
    out = {role: [dict(w, sup=list(w.get("sup") or [])) for w in files[role]] for role, _ in ROLES}
    for k, (role, wi, si, e, gi, _) in enumerate(rows):
        d = rows[donor[k]]
        de, dgi = d[3], d[4]
        n = len(de["set_ids"])
        kr_d, sw_d = de.get("set_kr") or [None] * n, de.get("set_w")
        sw_d = [None] * n if sw_d is None else sw_d
        block = [(int(de["set_ids"][i]), str(de["words"][i]), None if kr_d[i] is None else float(kr_d[i]),
                  None if sw_d[i] is None else float(sw_d[i])) for i in range(n) if i != dgi]
        o = dict(e)
        ids, words = list(e["set_ids"]), list(e["words"])
        kr = list(e["set_kr"]) if e.get("set_kr") else None
        sw = list(e["set_w"]) if e.get("set_w") is not None else None
        j = 0
        for i in range(len(ids)):
            if i == gi:
                continue                                 # the observed word's slot is untouched
            ids[i], words[i] = block[j][0], block[j][1]
            if kr is not None:
                kr[i] = block[j][2]
            if sw is not None:
                sw[i] = block[j][3]                     # the judge's vote fraction moves with its word
            j += 1
        o["set_ids"], o["words"] = ids, words
        if sw is not None:
            o["set_w"] = sw
        if kr is not None:
            o["set_kr"] = kr
        cand = token_homes[int(e["gold_id"])]
        o["home_lane"] = int(cand[row_rng(seed, SEED_NAME[role], wi, si).randrange(len(cand))])
        out[role][wi]["sup"][si] = o
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True, choices=["llama3", "qwen2.5", "falcon3"])
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--output-size", type=int, required=True, help="the model's logit width")
    a = ap.parse_args(argv)
    mem = vb.Memberships(vb.load_concepts(ROOT / "data/concept_vocabulary.json"),
                         vb.load_wordmap(ROOT / f"data/wordmaps/{a.family}.json"), a.output_size)
    files = {role: [json.loads(l) for l in open(Path(a.source) / name)] for role, name in ROLES}
    out = scramble(files, mem)
    Path(a.out).mkdir(parents=True, exist_ok=False)
    for role, name in ROLES:
        with open(Path(a.out) / name, "w") as f:
            for w in out[role]:
                f.write(json.dumps(w) + "\n")


if __name__ == "__main__":
    main()
