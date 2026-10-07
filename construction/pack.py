"""Training and held-out files for one tokenizer: windows and concept targets (paper Section 3, Appendix C).

Documents and roles. The documents are those holding at least one judged occurrence. A salted hash
of each FineWeb-Edu document id holds out 10% (half development, half final evaluation); the
concept-construction documents never enter a held-out role. Each role's documents, in first-seen
order of the judged file, are shuffled once (seed 0); training reads them in that order.

Windows. Each document is tokenized whole (no special tokens) and cut into 256-token windows; a
final tail shorter than 32 tokens is dropped, and so is a window holding no judged occurrence.
A window's first token is never predicted, so an occurrence that opens a window has no entry.
Each line of a file is {"input_ids", "sup"} (conceptizer/data.py); lines.json gives every line's
document and window number, the index that the released labels carry (construction/rebuild.py).

Concept targets. An occurrence (its observed token = the token holding the word's first
character) gets a concept target only if it passes every check below, in this order (the counts of
Table 4 follow it):
  G1  the observed word is a single whole-word token        G2  fewer than half of the judge's
  G3  some source proposed a candidate                           answers mark it a name or code
  G4  the judge kept a synonym                               G5  a kept synonym is a single token
  G6  the observed word's token belongs to a concept         G7  that concept has < 200 words
  G9  dropping a negating prefix cannot invert the word      G10 a single-token synonym is in a concept
  -- multi-token synonyms dropped; ranked, cut to 10, de-duplicated by token; judge weights attached;
     the closest concept c* chosen --
  G13 c* has a member token                                  G14 c* contains the exact observed token
  G15 lexical support (rule 6 of Appendix C)                 G1w the word starts and ends at word
                                                                 boundaries in the document
In a training file a rejected occurrence disappears (its window stays, for NTP); in a held-out file
it stays as a content-word entry {pos, gold_id, gate}.

    python -m construction.pack --tokenizer meta-llama/Llama-3.2-1B-Instruct --family llama3 \
        --judged judged_final.jsonl --documents fineweb_docs.jsonl --judge-weights judge_weights.json \
        --construction-documents data/construction/construction_documents.json --out data/llama3
"""
from __future__ import annotations

import argparse
import bisect
import collections
import hashlib
import json
import random
import re
from pathlib import Path

from conceptizer import targets as tg
from conceptizer import vocabulary as vb

ROOT = Path(__file__).resolve().parents[1]
WINDOW, TAIL_MIN = 256, 32
SALT = "s25_gated_2026_08_09"
NAME_OR_CODE_MAJORITY = 0.5
MAX_CONCEPT_WORDS = 200
NEGATING = ("non", "anti", "un", "in", "im", "il", "ir", "dis", "mis", "ill", "de", "counter")


# ------------------------------------------------------------------ roles and locations
def role_of(doc_id, salt=SALT):
    d = int(hashlib.md5(f"{salt}|{doc_id}".encode()).hexdigest(), 16)
    if d % 10 != 0:
        return "train"
    return "development" if (d // 10) % 10 <= 4 else "final"


def relocate(text, start, end, word):
    """The occurrence's span, moved to the nearest exact match of the word if the offsets drifted."""
    if text[start:end] == word:
        return start, end
    best, i = None, text.find(word)
    while i != -1:
        if best is None or abs(i - start) < abs(best - start):
            best = i
        i = text.find(word, i + 1)
    return (best, best + len(word)) if best is not None else (start, end)


def locate_sentences(doc_text, sents):
    """Find each judged sentence in its document (monotonic cursor; one retry from the start)."""
    cur, out = 0, []
    for s in sents:
        i = doc_text.find(s["text"], cur)
        if i == -1:
            i = doc_text.find(s["text"])
            if i == -1:
                continue
        out.append((s, i))
        cur = i + len(s["text"])
    return out


# ------------------------------------------------------------------ G9
_DASH_COMPOUND, _DASH_CLAUSE = "\u2010\u2011-", "\u2012\u2013\u2014"
_NEG_RE = re.compile("([A-Za-z]+)(?:[" + re.escape(_DASH_COMPOUND) + r"][ \t\r\n\f\v]*|["
                     + re.escape(_DASH_CLAUSE) + r"])\Z")


def negating_prefix(text, start, end, word):
    st, en = relocate(text, start, end, word)
    before, after = text[max(0, st - 1):st], text[en:en + 1]
    if before == "-":
        m = re.search(r"([A-Za-z]+)-$", text[max(0, st - 24):st])
        pre = m.group(1).lower() if m else None
        if pre in NEGATING:
            return pre
    elif after == "-":
        pass
    seg = max(0, st - 40)
    m = _NEG_RE.search(text[seg:st])
    if not m or m.group(1).lower() not in NEGATING or (m.start(1) == 0 and seg > 0):
        return None
    p, dash = m.group(1).lower(), m.group(0)[len(m.group(1))]
    if dash in _DASH_CLAUSE and p in ("in", "ill", "de", "counter"):
        return None
    return p


# ------------------------------------------------------------------ G1w
LEFT_GLUE_TAIL = "'\u2019\u2010\u2011-"


def boundary_fns(tok):
    raw, dec = {}, {}

    def decoded(i):
        if i not in dec:
            dec[i] = tok.decode([i])
        return dec[i]

    def opens(i):
        if i not in raw:
            raw[i] = tok.convert_ids_to_tokens(i)
        return bool(raw[i]) and raw[i][0] in "\u0120 "

    def continues(i):
        s = decoded(i)
        return not opens(i) and bool(s) and (s[0].isalnum() or s[0] in "'\u2019")

    def left_glue(i):
        s = decoded(i)
        return bool(s) and (s[-1].isalnum() or s[-1] in LEFT_GLUE_TAIL)
    return opens, continues, left_glue


def whole_word_in_document(tids, j, fns, left_in_token="", right_in_token=""):
    opens, continues, left_glue = fns
    if left_in_token:
        if left_in_token[-1].isalnum() or left_in_token[-1] in LEFT_GLUE_TAIL:
            return False
    elif not opens(tids[j]) and j > 0 and left_glue(tids[j - 1]):
        return False
    if right_in_token:
        return not (right_in_token[0].isalnum() or right_in_token[0] in LEFT_GLUE_TAIL)
    return j + 1 < len(tids) and not continues(tids[j + 1])


# ------------------------------------------------------------------ G15 (WordNet 3.0, one suffix step)
class Morphology:
    RULES = {"n": (("s", ""), ("ses", "s"), ("ves", "f"), ("xes", "x"), ("zes", "z"), ("ches", "ch"),
                   ("shes", "sh"), ("men", "man"), ("ies", "y")),
             "v": (("s", ""), ("ies", "y"), ("es", "e"), ("es", ""), ("ed", "e"), ("ed", ""),
                   ("ing", "e"), ("ing", "")),
             "a": (("er", ""), ("est", ""), ("er", "e"), ("est", "e")), "r": ()}
    FILES = {"n": "noun", "v": "verb", "a": "adj", "r": "adv"}
    ALPHA = re.compile(r"[a-z]+\Z")

    def __init__(self, wordnet_dir):
        d = Path(wordnet_dir)

        def read(name):
            return (d / name).read_text(encoding="utf-8")
        self.lexicon, self.exceptions = {}, {}
        for pos, suf in self.FILES.items():
            self.lexicon[pos] = frozenset(
                line.split()[0] for line in read(f"index.{suf}").splitlines()
                if line and not line[0].isspace())
            exc = {}
            for line in read(f"{suf}.exc").splitlines():
                f = line.split()
                exc.setdefault(f[0], set()).update(f[1:])
            self.exceptions[pos] = exc
        self._cache = {}

    def analyses(self, word):
        word = word.lower()
        if word not in self._cache:
            res = set()
            if self.ALPHA.fullmatch(word):
                for pos in self.FILES:
                    cands = [word] + sorted(self.exceptions[pos].get(word, ()))
                    cands += [word[:-len(o)] + n for o, n in self.RULES[pos] if word.endswith(o)]
                    res |= {(pos, b) for b in cands if self.ALPHA.fullmatch(b) and b in self.lexicon[pos]}
            self._cache[word] = frozenset(res)
        return self._cache[word]

    def unique(self, word):
        return len({lemma for _, lemma in self.analyses(word)}) == 1

    def lexically_supported(self, gold, words, members):
        """True when some emitted synonym with a unique WordNet lemma shares a (POS, lemma) with a
        member word of unique lemma and shares no lemma with the observed word."""
        gold = gold.lower()
        gold_lemmas = {lemma for _, lemma in self.analyses(gold)}
        for w in sorted({x.lower() for x in words} - {gold}):
            pairs = self.analyses(w)
            if {lemma for _, lemma in pairs} & gold_lemmas:
                continue
            for m in sorted({x.lower() for x in members}):
                if pairs & self.analyses(m) and self.unique(w) and self.unique(m):
                    return True
        return False


# ------------------------------------------------------------------ the builder
class Builder:
    def __init__(self, tok, concepts, wordmap, judge_weights, wordnet_dir):
        self.tok = tok
        self.mem = vb.Memberships(concepts, wordmap, output_size=len(tok))
        self.owner = self.mem.owners()                               # token -> concept number
        self.size = {c: len(ws) for c, ws in vb.normalize_members(concepts).items()}
        self.homes, self.members = vb.home_index(concepts), vb.member_sets(concepts)
        self.token_homes = {t: {self.mem.concept_numbers[l] for l in lanes} for t, lanes in self.mem.homes.items()}
        self.table = judge_weights
        self.pieces = tg.word_pieces(tok)
        self.boundary = boundary_fns(tok)
        self.morph = Morphology(wordnet_dir)

    def concept_of_word(self, word):
        p = self.pieces(word)
        return self.owner.get(p[0]) if p else None

    def gate(self, row, text):
        """G1-G10 on one judged row; returns the first failed check or None."""
        ld = row.get("lm_disc") or {}
        kept, pool, word = ld.get("kept") or {}, row.get("pool") or [], row["target"]["word"]
        if len(self.pieces(word)) != 1:
            return "G1_whole_word_gold"
        if float(ld.get("drop", 0) or 0) >= NAME_OR_CODE_MAJORITY:
            return "G2_no_majority_drop"
        if not pool:
            return "G3_candidates_offered"
        if not kept:
            return "G4_synonym_survived"
        if not any(len(self.pieces(s)) == 1 for s in kept):
            return "G5_whole_word_synonym"
        c = self.concept_of_word(word)
        if c is None:
            return "G6_gold_has_a_drawer"
        if self.size.get(c, 0) >= MAX_CONCEPT_WORDS:
            return "G7_drawer_not_giant"
        t = row["target"]
        if negating_prefix(text, t["start"], t["end"], word) is not None:
            return "G9_negating_prefix"
        if not any(self.concept_of_word(s) is not None for s in kept if len(self.pieces(s)) == 1):
            return "G10_set_has_no_lane"
        return None

    def entry(self, row, text, tids, j, pos, boundary):
        """The training entry of one located occurrence, or (None, failed check)."""
        failed = self.gate(row, text)
        if failed:
            return None, failed
        word, gid = row["target"]["word"], int(tids[j])
        kept = {w: s for w, s in row["lm_disc"]["kept"].items() if len(self.pieces(w)) == 1}
        set_ids, words = tg.contextual_synonyms(kept, word, gid, self.pieces)
        key = f"{(row['target'].get('lemma') or word).lower()}|{row['target'].get('pos')}"
        set_kr = [tg.GOLD_WEIGHT if t == gid else tg.judge_weights_for(key, [w], self.table)[0]
                  for t, w in zip(set_ids, words)]
        c_star = tg.closest_concept(word, words, self.homes, self.members, self.concept_of_word(word))
        eligible = whole_word_in_document(tids, j, self.boundary, **boundary)
        if eligible and c_star not in self.mem.lane_of:
            return None, "G13_home_uninstantiable"
        if eligible and c_star not in self.token_homes.get(gid, ()):
            return None, "G14_home_cannot_emit_gold"
        if eligible and not self.homes.get(word.lower()) \
                and not ({w.lower() for w in words} - {word.lower()}) & self.members[c_star] \
                and not self.morph.lexically_supported(word, words, self.members[c_star]):
            return None, "G15_native_word_support"
        if not eligible:
            return None, "G1w_window_whole_word"
        set_w = [float(kept.get(word, 0.0)) if t == gid else float(kept[w]) for t, w in zip(set_ids, words)]
        return {"pos": pos, "set_ids": set_ids, "set_w": set_w, "gold_id": gid, "set_kr": set_kr,
                "words": words, "home_lane": int(c_star)}, None

    def document(self, doc_id, text, sents, held_out):
        """[(window number, {"input_ids", "sup"})] of one document, in the files' line format."""
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        tids, offs = enc["input_ids"], enc["offset_mapping"]
        if not tids:
            return []
        starts = [o[0] for o in offs]
        by_window = collections.defaultdict(list)
        for s, off in locate_sentences(text, sents):
            for row in s["rows"]:
                t = row["target"]
                s0, e0 = relocate(s["text"], t["start"], t["end"], t["word"])
                if s["text"][s0:e0] != t["word"]:
                    continue
                a = off + s0
                j = bisect.bisect_right(starts, a) - 1
                if j < 0 or not (offs[j][0] <= a < offs[j][1]):
                    continue
                w, p = divmod(j, WINDOW)
                p -= 1
                if p < 0:
                    continue
                boundary = {"left_in_token": text[offs[j][0]:a],
                            "right_in_token": text[off + e0:offs[j][1]] if off + e0 <= offs[j][1] else ""}
                e, failed = self.entry(row, s["text"], tids, j, p, boundary)
                if e is None:
                    e = {"pos": p, "gold_id": int(tids[j]), "gate": failed}
                payload = [doc_id, a, off + e0, t["word"]]
                e["source_occurrence"] = {
                    "schema": "conceptizer_source_occurrence_v1",
                    "id": hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
                    "doc_id": doc_id, "char_start": a, "char_end": off + e0, "surface": t["word"]}
                by_window[w].append(e)
        out = []
        n_full, tail = divmod(len(tids), WINDOW)
        for w in range(n_full + (1 if tail else 0)):
            if w == n_full and tail < TAIL_MIN:
                break
            sup = by_window.get(w, [])
            if not sup:
                continue
            if not held_out:
                sup = [e for e in sup if "gate" not in e]
            out.append((w, {"input_ids": tids[w * WINDOW:(w + 1) * WINDOW], "sup": sup}))
        return out


def judged_by_document(path):
    """{doc_id: [{"text", "rows"}...]} in sentence order, plus the documents' first-seen order."""
    docs, order = collections.OrderedDict(), []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            d = r["doc_id"]
            if d not in docs:
                docs[d] = {}
                order.append(d)
            s = docs[d].setdefault(r["sent_id"], {"text": r["text"], "rows": []})
            s["rows"].append(r)
    return {d: [s for _, s in sorted(v.items())] for d, v in docs.items()}, order


def main(argv=None):
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--family", required=True, choices=["llama3", "qwen2.5", "falcon3"])
    ap.add_argument("--judged", required=True, help="judged occurrences (construction.label output)")
    ap.add_argument("--documents", required=True, help="JSONL {doc_id, text} of the FineWeb-Edu documents")
    ap.add_argument("--judge-weights", required=True)
    ap.add_argument("--construction-documents", required=True)
    ap.add_argument("--wordnet-dir", required=True, help="the WordNet 3.0 dict directory (index.noun, noun.exc, ...)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from conceptizer import models
    tok = AutoTokenizer.from_pretrained(a.tokenizer, **models.hub_kwargs(a.tokenizer))
    b = Builder(tok, vb.load_concepts(ROOT / "data/concept_vocabulary.json"),
                vb.load_wordmap(ROOT / f"data/wordmaps/{a.family}.json"), json.load(open(a.judge_weights)),
                a.wordnet_dir)
    judged, order = judged_by_document(a.judged)
    construction = set(json.load(open(a.construction_documents)))
    texts = {}
    with open(a.documents) as f:
        for line in f:
            r = json.loads(line)
            d = r.get("doc_id", r.get("id"))
            if d in judged:
                texts[d] = r["text"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    lines = {}                                   # the document id and window number of every line
    for role, name in (("train", "train_windows"), ("development", "development_windows"), ("final", "final_windows")):
        ids = [d for d in order if role_of(d) == role and (role == "train" or d not in construction)]
        random.Random(0).shuffle(ids)
        lines[role] = []
        with open(out / f"{name}.jsonl", "w") as f:
            for d in ids:
                if d in texts:
                    for w, window in b.document(d, texts[d], judged[d], held_out=(role != "train")):
                        f.write(json.dumps(window) + "\n")
                        lines[role].append([d, w])
    (out / "lines.json").write_text(json.dumps(lines))


if __name__ == "__main__":
    main()
