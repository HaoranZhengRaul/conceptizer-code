# Data

Large files are gzip-compressed and split into parts (`*.partNN.*.gz`, read in part order).

## Corpus

FineWeb-Edu (`HuggingFaceFW/fineweb-edu`, ODC-By 1.0), revision
`87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`: `sample/100BT/000_00000.parquet` and `000_00001.parquet`
hold every document used.

## Concept vocabulary (`data/`)

- `concept_vocabulary.json`: `[{"0": [...], ..., "9060": [...]}]`, concept number to member words.
- `wordmaps/{llama3,qwen2.5,falcon3}.json`: `{word: {"first_id": ..., "n_sub": ...}}`, the first token
  of the word written after a space and its number of tokens.

## Labels (`data/labels/`)

- `documents/{train,development,final}.partNN.txt.gz`: FineWeb-Edu ids, one per line; line k is
  document k.
- `{llama3,qwen2.5,falcon3}/{train,development,final}.partNN.jsonl.gz`: one record per line of the
  file, `{"doc": k, "window": w, "sup": [entry, ...]}`; the line's tokens are tokens 256w to
  256w + 255 of document k, tokenized whole without special tokens. An entry:
  - `pos`: the logit position that predicts the observed token `gold_id`;
  - `words`, `set_ids`: the observed word w_0 and its contextual synonyms w_1..w_m, and their tokens;
  - `set_kr`: their judge weights r_j (1.0 for the observed word);
  - `set_w`: the share of the judge's 3 answers that kept each word in this occurrence;
  - `home_lane`: the closest concept c*;
  - `char`: the occurrence's character offsets in the document.

  The development and final files also keep each content word without a concept target as
  `{"pos", "gold_id", "gate", "char"}`, `gate` naming the first check it failed (Table 4; the codes
  are listed in `construction/pack.py`). The Randomized-synonyms files are derived from these
  records (`construction.rebuild --randomized`).

## Concept-construction sample (`data/construction/`)

- `construction_sample.jsonl.gz`: the concept-construction sample, 149,919 judged occurrences, of which
  88,367 build the concepts, in the order that numbers the concepts:
  `{"key", "gold", "kept", "induction", "fineweb_id", "split"}` (`kept`: the judge's kept words with the
  share of its readable answers, 3 for most occurrences).
- `construction_documents.json`: their documents, which never enter a held-out file.

## Not included

The judged file (every occurrence's candidates and the judge's answers, 3.4 GB), the judge-weight
table (147 MB; the labels carry the weights) and the trained LoRA adapters.
