"""Synonym candidates and the LLM judge (paper Section 3, Appendices C and H), with vLLM.

Stages (run in order; each reads the previous stage's file in --workdir):
  prep     occurrences.jsonl -> occurrences of words (lemma, POS) with at least 3 occurrences
  wordnet  single-word WordNet lemma names of each word under its POS (NLTK)
  gen      Llama-3.3-70B-Instruct reads each sentence (first occurrence of the word marked) and
           proposes replacements; 5 samples at temperature 1.0, seed 0, 128 tokens
  freegen  Qwen2.5-72B-Instruct proposes synonyms of each word across all its senses, once per word;
           5 samples at temperature 1.0, seed 0, 192 tokens
  pool     the lowercased union of the 3 sources per occurrence, with each candidate's sources
  judge    gpt-oss-120b keeps the candidates that fit the sentence (it may add up to 2); 3 answers at
           temperature 0.7, seed 0, high reasoning effort, 8,192 tokens. A candidate's value is the
           fraction of readable answers that kept it; "drop" is the fraction that marked the word a
           name or code (at least half removes the occurrence).
  refill   occurrences with fewer than 3 readable answers (answers that ran out of tokens) get only
           their missing answers, with 32,768 tokens: rounds 1-5 at seed 0, rounds 6-10 at seeds 1-5.

    python -m construction.label prep --workdir work --occurrences occurrences.jsonl
    python -m construction.label gen --workdir work            # and so on
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POS_HUMAN = {"NOUN": "noun", "VERB": "verb", "ADJ": "adjective"}
PROMPTS = {k: json.loads((ROOT / f"data/prompts/{k}.json").read_text())
           for k in ("candidates_in_context", "candidates_without_context", "judge")}


def mark_first(text, word):
    """Mark the first (case-insensitive) occurrence of the word's spelling in the sentence."""
    i = text.lower().find(word.lower())
    i = i if i >= 0 else 0
    return text[:i] + "«" + text[i:i + len(word)] + "»" + text[i + len(word):]


def key_of(r):
    t = r["target"]
    return f"{r['doc_id']}|{r['sent_id']}|{t['start']}"


# ------------------------------------------------------------------ messages
def gen_messages(marked, word, pos):
    p = PROMPTS["candidates_in_context"]
    msgs = [{"role": "system", "content": p["system"]}]
    for e in p["examples"]:
        msgs += [{"role": "user", "content": f'Sentence: {e["sentence"]}\nTarget: "{e["target"]}" ({e["pos"]})'},
                 {"role": "assistant", "content": e["answer"]}]
    return msgs + [{"role": "user", "content": f'Sentence: {marked}\nTarget: "{word}" ({pos})'}]


def freegen_messages(word, pos):
    p = PROMPTS["candidates_without_context"]
    msgs = [{"role": "system", "content": p["system"]}]
    for e in p["examples"]:
        msgs += [{"role": "user", "content": f'Word: "{e["word"]}" ({e["pos"]})'},
                 {"role": "assistant", "content": e["answer"]}]
    return msgs + [{"role": "user", "content": f'Word: "{word}" ({pos})'}]


def judge_messages(marked, word, pos, pool):
    p = PROMPTS["judge"]
    msgs = [{"role": "system", "content": p["system"]}]
    for e in p["examples"]:
        msgs += [{"role": "user", "content": f'Sentence: {e["sentence"]}\nTarget: "{e["target"]}" ({e["pos"]})\n'
                                             f'Candidates: {e["candidates"]}'},
                 {"role": "assistant", "content": e["answer"]}]
    return msgs + [{"role": "user", "content": f'Sentence: {marked}\nTarget: "{word}" ({pos})\n'
                                               f'Candidates: [{", ".join(pool)}]'}]


# ------------------------------------------------------------------ parsing
_LIST = re.compile(r"\[.*?\]", re.S)
_OBJ = re.compile(r"\{[^{}]*\}", re.S)


def parse_list(raw):
    m = _LIST.search(raw)
    if not m:
        return None
    try:
        return [w for w in json.loads(m.group(0)) if isinstance(w, str)]
    except Exception:
        return None


def clean(words, target, dedupe=True):
    """Drop multi-word phrases, the target itself and phrases containing it (and duplicates)."""
    tl, out, seen = target.lower().strip(), [], set()
    for w in words or []:
        wl = w.lower().strip()
        if not wl or " " in wl or wl == tl or tl in wl.split() or (dedupe and wl in seen):
            continue
        seen.add(wl)
        out.append(w.strip() if dedupe else w)
    return out


def aggregate(outputs, target):
    """Union of the samples' words, most frequent first, with the fraction of samples proposing each."""
    counts, surface, n = Counter(), {}, 0
    for o in outputs:
        lst = parse_list(o.text)
        if lst is None:
            continue
        n += 1
        for w in clean(lst, target):
            wl = w.lower()
            counts[wl] += 1
            surface.setdefault(wl, w)
    kept = counts.most_common()
    return [surface[w] for w, _ in kept], ({surface[w]: round(c / n, 2) for w, c in kept} if n else {})


def parse_verdict(raw):
    for m in reversed(_OBJ.findall(raw)):
        try:
            o = json.loads(m)
            if isinstance(o, dict) and "synonyms" in o:
                return bool(o.get("drop", False)), [w for w in o.get("synonyms", []) if isinstance(w, str)]
        except Exception:
            pass
    return None, None


def tally(outputs, target):
    """Kept fraction per word and the name-or-code fraction over the readable answers."""
    cnt, drops, n = Counter(), 0, 0
    for o in outputs:
        drop, syns = parse_verdict(o.text)
        if syns is None:
            continue
        n += 1
        drops += int(bool(drop))
        for w in clean(syns, target, dedupe=False):
            cnt[w.lower()] += 1
    return cnt, drops, n


def verdict(cnt, drops, n):
    return {"kept": {w: round(c / n, 3) for w, c in cnt.items()} if n else {},
            "drop": round(drops / n, 3) if n else 0, "n_parsed": n}


# ------------------------------------------------------------------ stages
def _rows(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def stage_prep(work, occurrences):
    cnt = defaultdict(int)
    for r in _rows(occurrences):
        for t in r["targets"]:
            cnt[(t["lemma"], t["pos"])] += 1
    ok = {k for k, v in cnt.items() if v >= 3}
    with open(work / "instances.jsonl", "w") as f:
        for r in _rows(occurrences):
            for t in r["targets"]:
                if (t["lemma"], t["pos"]) in ok:
                    f.write(json.dumps({"doc_id": r["doc_id"], "sent_id": r["sent_id"], "text": r["text"],
                                        "target": t}) + "\n")
    json.dump(sorted(f"{l}|{p}" for l, p in ok), open(work / "words.json", "w"))


def stage_wordnet(work):
    from nltk.corpus import wordnet as wn
    pm = {"NOUN": wn.NOUN, "VERB": wn.VERB, "ADJ": wn.ADJ}
    out = {}
    for lk in json.load(open(work / "words.json")):
        lemma, pos = lk.rsplit("|", 1)
        syns = {l.name().lower() for ss in wn.synsets(lemma, pos=pm.get(pos)) for l in ss.lemmas()
                if "_" not in l.name() and l.name().lower() != lemma.lower()}
        if syns:
            out[lk] = sorted(syns)
    json.dump(out, open(work / "wordnet.json", "w"))


def _llm(model, tp, max_len):
    from vllm import LLM
    return LLM(model=model, tensor_parallel_size=tp, dtype="bfloat16" if "gpt-oss" not in model else "auto",
               gpu_memory_utilization=0.9, max_model_len=max_len, disable_custom_all_reduce=True)


def stage_gen(work, chunk=2000):
    from vllm import SamplingParams
    p = PROMPTS["candidates_in_context"]
    llm = _llm(p["model"], 2, 4096)
    sp = SamplingParams(temperature=1.0, max_tokens=128, n=5, seed=0)
    rows = list(_rows(work / "instances.jsonl"))
    with open(work / "gen.jsonl", "w") as f:
        for i in range(0, len(rows), chunk):
            batch = rows[i:i + chunk]
            msgs = [gen_messages(mark_first(r["text"], r["target"]["word"]), r["target"]["word"],
                                 POS_HUMAN[r["target"]["pos"]]) for r in batch]
            for r, o in zip(batch, llm.chat(msgs, sp)):
                syns, w = aggregate(o.outputs, r["target"]["word"])
                f.write(json.dumps({"_k": key_of(r), **r, "gen": syns, "gen_w": w}) + "\n")


def stage_freegen(work, chunk=4000):
    from vllm import SamplingParams
    p = PROMPTS["candidates_without_context"]
    llm = _llm(p["model"], 2, 4096)
    sp = SamplingParams(temperature=1.0, max_tokens=192, n=5, seed=0)
    words = json.load(open(work / "words.json"))
    with open(work / "freegen.jsonl", "w") as f:
        for i in range(0, len(words), chunk):
            batch = words[i:i + chunk]
            msgs = [freegen_messages(lk.rsplit("|", 1)[0], POS_HUMAN[lk.rsplit("|", 1)[1]]) for lk in batch]
            for lk, o in zip(batch, llm.chat(msgs, sp)):
                syns, w = aggregate(o.outputs, lk.rsplit("|", 1)[0])
                f.write(json.dumps({"_k": lk, "freegen": syns, "freegen_w": w}) + "\n")


def stage_pool(work):
    fg = {r["_k"]: r for r in _rows(work / "freegen.jsonl")}
    wn = json.load(open(work / "wordnet.json"))
    with open(work / "pool.jsonl", "w") as f:
        for r in _rows(work / "gen.jsonl"):
            lk = f"{r['target']['lemma']}|{r['target']['pos']}"
            g = {w.lower() for w in r.get("gen", [])}
            q = {w.lower() for w in fg.get(lk, {}).get("freegen", [])}
            n = {w.lower() for w in wn.get(lk, [])}
            pool = sorted(g | q | n)
            src = {w: ("g" if w in g else "") + ("f" if w in q else "") + ("w" if w in n else "") for w in pool}
            f.write(json.dumps({"_k": r["_k"], "doc_id": r["doc_id"], "sent_id": r["sent_id"], "text": r["text"],
                                "target": r["target"], "pool": pool, "src": src}) + "\n")


def stage_judge(work, chunk=500):
    from vllm import SamplingParams
    llm = _llm(PROMPTS["judge"]["model"], 1, 12288)
    sp = SamplingParams(temperature=0.7, max_tokens=8192, n=3, seed=0)
    rows = list(_rows(work / "pool.jsonl"))
    with open(work / "judged.jsonl", "w") as f:
        for i in range(0, len(rows), chunk):
            batch = rows[i:i + chunk]
            msgs = [judge_messages(mark_first(r["text"], r["target"]["word"]), r["target"]["word"],
                                   POS_HUMAN[r["target"]["pos"]], r["pool"]) for r in batch]
            for r, o in zip(batch, llm.chat(msgs, sp, chat_template_kwargs={"reasoning_effort": "high"})):
                r["lm_disc"] = verdict(*tally(o.outputs, r["target"]["word"]))
                f.write(json.dumps(r) + "\n")


def stage_refill(work, max_rounds=10):
    """Complete occurrences with fewer than 3 readable answers; existing answers are kept."""
    from vllm import SamplingParams
    rows = list(_rows(work / "judged.jsonl"))
    need = {i: 3 - r["lm_disc"]["n_parsed"] for i, r in enumerate(rows) if r["lm_disc"]["n_parsed"] < 3}
    if not need:                                         # nothing to complete: the judged file as is
        with open(work / "judged_final.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return
    llm = _llm(PROMPTS["judge"]["model"], 1, 36864)
    for i in need:                                       # recover integer votes from the fractions
        ld, n = rows[i]["lm_disc"], rows[i]["lm_disc"]["n_parsed"]
        rows[i]["_votes"] = [Counter({w: int(round(v * n)) for w, v in ld["kept"].items()}),
                             int(round(ld["drop"] * n)), n]
    for rnd in range(1, max_rounds + 1):
        todo = [i for i in need if rows[i]["_votes"][2] < 3]
        if not todo:
            break
        for k in (1, 2, 3):
            grp = [i for i in todo if 3 - rows[i]["_votes"][2] == k]
            if not grp:
                continue
            sp = SamplingParams(temperature=0.7, max_tokens=32768, n=k, seed=0 if rnd <= 5 else rnd - 5)
            msgs = [judge_messages(mark_first(rows[i]["text"], rows[i]["target"]["word"]), rows[i]["target"]["word"],
                                   POS_HUMAN[rows[i]["target"]["pos"]], rows[i]["pool"]) for i in grp]
            for i, o in zip(grp, llm.chat(msgs, sp, chat_template_kwargs={"reasoning_effort": "high"})):
                cnt, drops, n = tally(o.outputs, rows[i]["target"]["word"])
                v = rows[i]["_votes"]
                v[0].update(cnt); v[1] += drops; v[2] += n
    with open(work / "judged_final.jsonl", "w") as f:
        for r in rows:
            v = r.pop("_votes", None)
            if v is not None:
                r["lm_disc"] = {"kept": {w: round(c / v[2], 3) for w, c in sorted(v[0].items())} if v[2] else {},
                                "drop": round(v[1] / v[2], 3) if v[2] else 0, "n_parsed": v[2]}
            f.write(json.dumps(r) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["prep", "wordnet", "gen", "freegen", "pool", "judge", "refill"])
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--occurrences")
    a = ap.parse_args(argv)
    work = Path(a.workdir)
    work.mkdir(parents=True, exist_ok=True)
    {"prep": lambda: stage_prep(work, a.occurrences), "wordnet": lambda: stage_wordnet(work),
     "gen": lambda: stage_gen(work), "freegen": lambda: stage_freegen(work), "pool": lambda: stage_pool(work),
     "judge": lambda: stage_judge(work), "refill": lambda: stage_refill(work)}[a.stage]()


if __name__ == "__main__":
    main()
