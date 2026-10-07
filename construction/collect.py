"""Collect content-word occurrences from FineWeb-Edu (paper Section 3, Appendix C).

Documents are read in a fixed order (sorted shard files, file order within a shard). A document is
used if its language is "en" with language_score >= 0.90, it has at most 1,500,000 characters and
its first 2,000 characters pass the math/code filter. spaCy en_core_web_sm (parser and NER off,
a sentencizer added) splits sentences and tags words; sentences that fail the math/code filter are
skipped. Every alphabetic non-stopword NOUN, VERB or ADJ is an occurrence of its (lowercased lemma,
POS) until that pair has 25 occurrences (first come, first served). Pairs with fewer than 3
occurrences are dropped before labeling (construction/label.py prep).

    python -m construction.collect --shards 'fineweb-edu/*.parquet' --out occurrences.jsonl

Output, one line per sentence with at least one occurrence:
    {"doc_id", "sent_id", "text", "targets": [{"word", "start", "end", "pos", "lemma"}]}
"""
import argparse
import glob
import json
import re
from collections import Counter

CODE_RE = re.compile(r"[{}<>]|;\s|=>|::|</?[a-zA-Z]|\b(def|class|import|return|function|var|const|public|void|SELECT|FROM)\b")
LATEX_RE = re.compile(r"\$\$?|\\begin|\\frac|\\[a-zA-Z]+\s*\{|\^\{?|_\{")
CONTENT_POS = {"NOUN", "VERB", "ADJ"}
PER_WORD_CAP = 25
LANGUAGE_SCORE_MIN = 0.90


def mathy_or_code(text):
    n = len(text)
    if n < 20:
        return True
    if sum(1 for c in text if not (c.isalpha() or c.isspace())) / n > 0.25:
        return True
    if sum(c.isdigit() for c in text) / n > 0.15:
        return True
    return bool(CODE_RE.search(text) or LATEX_RE.search(text))


def documents(shards):
    import pyarrow.parquet as pq
    for path in shards:
        pf = pq.ParquetFile(path)
        cols = [c for c in ("text", "id", "language", "language_score") if c in pf.schema.names]
        for batch in pf.iter_batches(batch_size=512, columns=cols):
            d = batch.to_pydict()
            for text, doc_id, lang, score in zip(d["text"], d["id"], d.get("language", ["en"] * len(d["text"])),
                                                 d.get("language_score", [1.0] * len(d["text"]))):
                if lang == "en" and (score or 0.0) >= LANGUAGE_SCORE_MIN:
                    yield doc_id, text


def main(argv=None):
    import spacy
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", required=True, help="glob of FineWeb-Edu parquet files (read in sorted order)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-process", type=int, default=16)
    a = ap.parse_args(argv)
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    if "sentencizer" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    nlp.max_length = 2_000_000

    def stream():
        for doc_id, text in documents(sorted(glob.glob(a.shards))):
            if text and len(text) <= 1_500_000 and not mathy_or_code(text[:2000]):
                yield text, doc_id

    count = Counter()
    with open(a.out, "w") as out:
        for doc, doc_id in nlp.pipe(stream(), as_tuples=True, n_process=a.n_process, batch_size=64):
            for si, sent in enumerate(doc.sents):
                s = sent.text.strip()
                if mathy_or_code(s):
                    continue
                targets = []
                for tok in sent:
                    if tok.pos_ not in CONTENT_POS or not tok.is_alpha or tok.is_stop:
                        continue
                    key = (tok.lemma_.lower(), tok.pos_)
                    if count[key] >= PER_WORD_CAP:
                        continue
                    count[key] += 1
                    start = tok.idx - sent.start_char
                    targets.append({"word": tok.text, "start": start, "end": start + len(tok.text),
                                    "pos": tok.pos_, "lemma": tok.lemma_.lower()})
                if targets:
                    out.write(json.dumps({"doc_id": doc_id, "sent_id": si, "text": s, "targets": targets}) + "\n")


if __name__ == "__main__":
    main()
