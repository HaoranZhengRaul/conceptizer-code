"""Word map of one tokenizer: word -> first token and number of tokens (paper Section 3).

The universe is every observed word and kept word of the concept-construction sample (57,554
words; the 9,061 concepts use 49,387 of them). Each word is encoded with a leading space, as it
appears after another word, without special tokens. ``data/wordmaps/*.json`` are the maps the paper
used for Llama 3, Qwen2.5 and Falcon3.

    python -m construction.wordmap --tokenizer meta-llama/Llama-3.2-1B-Instruct \
        --sample data/construction/construction_sample.jsonl.gz --out wordmap.json
"""
import argparse
import gzip
import json


def build_wordmap(words, tokenizer):
    wm = {}
    for w in sorted(words):
        ids = tokenizer(" " + w, add_special_tokens=False)["input_ids"]
        if ids:
            wm[w] = {"first_id": int(ids[0]), "n_sub": len(ids)}
    return wm


def main(argv=None):
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    words = set()
    with gzip.open(a.sample, "rt") as f:
        for line in f:
            r = json.loads(line)
            words.add(r["gold"]); words.update(r["kept"])
    from conceptizer import models
    wm = build_wordmap(words, AutoTokenizer.from_pretrained(a.tokenizer, **models.hub_kwargs(a.tokenizer)))
    json.dump(wm, open(a.out, "w"))
    print(f"{len(wm)} words")


if __name__ == "__main__":
    main()
