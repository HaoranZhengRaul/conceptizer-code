"""Rebuild the training and held-out files of one tokenizer from FineWeb-Edu and the released labels.

A window is cut from its document's own tokens, so each file is rebuilt from three inputs: the
FineWeb-Edu documents (two public parquet files), the tokenizer, and data/labels/, which gives
for every line of every file its document, its window number and its entries without the parts
the document determines. The rebuilt files are byte-identical to the files the paper's models were
trained on.

    python -m construction.rebuild --tokenizer meta-llama/Llama-3.2-1B-Instruct --family llama3 \
        --fineweb fineweb/sample/100BT --out data/llama3 --randomized data/llama3_randomized

FineWeb-Edu (ODC-By 1.0), revision FINEWEB_REVISION: sample/100BT/000_00000.parquet and
000_00001.parquet hold every document.

Label files (gzip-compressed, split into parts of at most a few MB, read in part order):
  documents/<role>.partNN.txt.gz   one FineWeb-Edu document id per line; line k is document k
  <family>/<role>.partNN.jsonl.gz  one record per line of the file:
      {"doc": k, "window": w, "sup": [entry, ...]}
  The window is tokens 256w .. 256w+255 of document k, tokenized whole without special tokens.
  An entry is the file's own entry with its source occurrence reduced to "char": [start, end],
  the occurrence's character offsets in the document; the surface word is the document's text
  between them and the occurrence id is a hash of (document, start, end, surface).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WINDOW = 256
ROLES = (("train", "train_windows.jsonl"), ("development", "development_windows.jsonl"),
         ("final", "final_windows.jsonl"))
FINEWEB_REPO = "HuggingFaceFW/fineweb-edu"
FINEWEB_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
FINEWEB_FILES = ("000_00000.parquet", "000_00001.parquet")
OCCURRENCE_SCHEMA = "conceptizer_source_occurrence_v1"


# ------------------------------------------------------------------ label records
def occurrence_id(doc_id, start, end, surface):
    payload = [doc_id, start, end, surface]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def expand(rec, doc_id, text, token_ids):
    """A label record -> the file's line (a dict), given its document's text and tokens."""
    w = int(rec["window"])
    sup = []
    for c in rec["sup"]:
        e = dict(c)
        if "char" in e:
            s, t = e.pop("char")
            surface = text[s:t]
            e["source_occurrence"] = {"schema": OCCURRENCE_SCHEMA, "id": occurrence_id(doc_id, s, t, surface),
                                      "doc_id": doc_id, "char_start": s, "char_end": t, "surface": surface}
        sup.append(e)
    return {"input_ids": list(token_ids[w * WINDOW:(w + 1) * WINDOW]), "sup": sup}


# ------------------------------------------------------------------ part files
def read_parts(prefix, ext):
    """The lines of every part of <prefix>, in part order."""
    prefix = Path(prefix)
    parts = sorted(prefix.parent.glob(f"{prefix.name}.part[0-9][0-9].{ext}.gz"))
    if not parts:
        raise FileNotFoundError(f"no parts {prefix}.partNN.{ext}.gz")
    for p in parts:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                yield line.rstrip("\n")


def read_documents(labels, role):
    return list(read_parts(Path(labels) / "documents" / role, "txt"))


def read_records(labels, family, role):
    return [json.loads(line) for line in read_parts(Path(labels) / family / role, "jsonl")]


# ------------------------------------------------------------------ documents
def fineweb_texts(paths, wanted):
    """{FineWeb-Edu id: text} for the wanted ids, read from the parquet files."""
    import pyarrow.parquet as pq
    texts = {}
    for p in paths:
        pf = pq.ParquetFile(p)
        for g in range(pf.num_row_groups):
            t = pf.read_row_group(g, columns=["id", "text"])
            for d, x in zip(t.column("id").to_pylist(), t.column("text").to_pylist()):
                if d in wanted:
                    texts[d] = x
    return texts


def tokenized(doc_per_line, texts, tok, batch=256):
    """(doc_id, text, token ids) for each line, documents tokenized whole in batches."""
    order, first = [], {}
    for d in doc_per_line:
        if d not in first:
            first[d] = len(order)
            order.append(d)
    ids = {}
    for d in doc_per_line:
        if d not in ids:
            chunk = order[first[d]:first[d] + batch]
            enc = tok([texts[x] for x in chunk], add_special_tokens=False)["input_ids"]
            ids = dict(zip(chunk, enc))
        yield d, texts[d], ids[d]


# ------------------------------------------------------------------ the rebuild
def rebuild(tok, family, labels, texts, out, randomized=None, output_size=None):
    """Writes the 3 files of `family` to `out` (and the Randomized-synonyms files to `randomized`)."""
    from construction import randomize
    from conceptizer import vocabulary as vb
    records = {role: read_records(labels, family, role) for role, _ in ROLES}
    docs = {role: read_documents(labels, role) for role, _ in ROLES}
    scrambled = None
    if randomized is not None:
        mem = vb.Memberships(vb.load_concepts(ROOT / "data/concept_vocabulary.json"),
                             vb.load_wordmap(ROOT / f"data/wordmaps/{family}.json"),
                             output_size or len(tok))
        scrambled = randomize.scramble({role: [{"sup": r["sup"]} for r in records[role]] for role, _ in ROLES}, mem)
    written = []
    for role, name in ROLES:
        per_line = [docs[role][r["doc"]] for r in records[role]]
        targets = [(f"{family}/{name}", Path(out) / name, None)]
        if scrambled is not None:
            targets.append((f"{family}_randomized/{name}", Path(randomized) / name, scrambled[role]))
        handles = [(open(p, "w", encoding="utf-8"), s) for _, p, s in targets]
        for li, (r, (d, text, ids)) in enumerate(zip(records[role], tokenized(per_line, texts, tok))):
            for f, s in handles:
                rec = r if s is None else dict(r, sup=s[li]["sup"])
                f.write(json.dumps(expand(rec, d, text, ids)) + "\n")
        for f, _ in handles:
            f.close()
        written += [(key, p) for key, p, _ in targets]
    return written


def main(argv=None):
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True, help="the base model's name or local directory")
    ap.add_argument("--revision", help="tokenizer revision (default: the paper's, for its 9 models)")
    ap.add_argument("--family", required=True, choices=["llama3", "qwen2.5", "falcon3"])
    ap.add_argument("--fineweb", required=True, help="directory holding the two FineWeb-Edu parquet files")
    ap.add_argument("--labels", default=str(ROOT / "data/labels"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--randomized", help="also write the Randomized-synonyms files here")
    ap.add_argument("--output-size", type=int, help="logit width for the memberships (default: tokenizer size)")
    a = ap.parse_args(argv)
    from conceptizer import models
    tok = AutoTokenizer.from_pretrained(a.tokenizer, **models.hub_kwargs(a.tokenizer, a.revision))
    wanted = set()
    for role, _ in ROLES:
        docs = read_documents(a.labels, role)
        wanted.update(docs[r["doc"]] for r in read_records(a.labels, a.family, role))
    paths = [Path(a.fineweb) / n for n in FINEWEB_FILES]
    for p in paths:
        if not p.exists():
            raise SystemExit(f"missing {p}: download {FINEWEB_REPO} sample/100BT/{p.name} at revision {FINEWEB_REVISION}")
    texts = fineweb_texts(paths, wanted)
    if len(texts) != len(wanted):
        raise SystemExit(f"{len(wanted) - len(texts)} documents not found in {a.fineweb}")
    for d in (a.out, a.randomized):
        if d:
            Path(d).mkdir(parents=True, exist_ok=False)
    for key, path in rebuild(tok, a.family, a.labels, texts, a.out, a.randomized, a.output_size):
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
