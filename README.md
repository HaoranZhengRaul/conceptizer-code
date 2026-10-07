# Conceptizer: Next-Token Prediction Is Not All You Need

Code and data for the paper (anonymous submission).

## This repository contains

- **The concept vocabulary: 9,061 concepts** over 49,387 member words, with word maps for the
  Llama 3, Qwen2.5 and Falcon3 tokenizers (`data/concept_vocabulary.json`, `data/wordmaps/`).
- **The concept-labeled training data:** the concept targets of every training and held-out file
  of the paper, which rebuild the files from public FineWeb-Edu documents (`data/labels/`), and the
  concept-construction sample, the 149,919 judged occurrences the concepts are built from
  (`data/construction/`).
- **The construction pipeline:** occurrences, LLM and WordNet candidates, the LLM judge and its
  prompts, senses and concepts, word maps, training files (`construction/`, `data/prompts/`).
- **The training pipeline:** concept memberships, concept targets, the concept loss, and LoRA
  post-training of Conceptizer, NTP post-training, Randomized synonyms and Weighted NTP
  (`conceptizer/`).

## Install

```bash
pip install -r requirements.txt
python -m nltk.downloader wordnet
```

Python 3.11. Training needs a GPU (the paper used one H100 per run); rebuilding the files and the
concept vocabulary runs on CPU. The Llama models require accepting Meta's license on
Hugging Face. Every model and tokenizer loads at the revision the paper used (`conceptizer/models.py`).

## 1. Get the training and held-out files

Download the two FineWeb-Edu files that hold every document the paper uses (4.3 GB):

```bash
python - <<'EOF'
from huggingface_hub import hf_hub_download
for name in ("000_00000.parquet", "000_00001.parquet"):
    hf_hub_download("HuggingFaceFW/fineweb-edu", f"sample/100BT/{name}", repo_type="dataset",
                    revision="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9", local_dir="fineweb")
EOF
```

Rebuild the files of each tokenizer family (CPU, a few minutes and about 3 GB each):

```bash
python -m construction.rebuild --tokenizer meta-llama/Llama-3.2-1B-Instruct --family llama3 \
    --fineweb fineweb/sample/100BT --out data/llama3 --randomized data/llama3_randomized
python -m construction.rebuild --tokenizer Qwen/Qwen2.5-1.5B-Instruct --family qwen2.5 \
    --fineweb fineweb/sample/100BT --out data/qwen2.5 --randomized data/qwen2.5_randomized
python -m construction.rebuild --tokenizer tiiuae/Falcon3-1B-Instruct --family falcon3 \
    --fineweb fineweb/sample/100BT --out data/falcon3 --randomized data/falcon3_randomized
```

Each writes `train_windows.jsonl`, `development_windows.jsonl` and `final_windows.jsonl`, with the
real concept targets in `data/<family>/` and the Randomized-synonyms targets in
`data/<family>_randomized/`. A line is a window of up to 256 tokens of one document with its
concept-target entries (`docs/DATA.md`). Every model of a family trains on that family's files.

## 2. Train

```bash
python -m conceptizer.train --model meta-llama/Llama-3.2-1B-Instruct --family llama3 \
    --train data/llama3/train_windows.jsonl --method conceptizer --seed 0 --out runs/llama1b/conceptizer_s0
```

- `--method`: `conceptizer` (Conceptizer: NTP + concept loss + retention), `ntp` (NTP post-training:
  NTP + retention), `weighted_ntp` (Weighted NTP: extra weight on the observed token at the
  concept-target positions instead of the concept loss), or `randomized` (Randomized synonyms:
  Conceptizer with `--train data/llama3_randomized/train_windows.jsonl`).
- `--model` and `--family`: the paper's 9 models are Llama-3.2-1B/3B-Instruct and
  Llama-3.1-8B-Instruct (`llama3`), Qwen2.5-1.5B/3B/7B-Instruct (`qwen2.5`) and
  Falcon3-1B/3B/7B-Instruct (`falcon3`), all under their Hugging Face names.
- `--seed`: the paper used 0, 37 and 42.

One recipe for every model and method (`RECIPE` in `conceptizer/train.py`): LoRA rank 16, alpha 32,
dropout 0.05 on every linear attention and feed-forward projection (the output head excluded); AdamW
at 7e-5, 100 warmup steps, cosine decay to zero over 50,000 steps of 16 windows; concept-loss weight
0.6, retention weight 0.5; bfloat16. The run writes the LoRA adapter to `<out>/adapter` and the step
losses to `<out>/training.json`. In the paper, a 50,000-step run took about 7 to 9 hours of job time
on one H100 for the smallest models and about 20 hours for Llama 8B.

## 3. Rebuild the concept vocabulary

```bash
python -m construction.senses --out concepts.json
python -m construction.wordmap --tokenizer meta-llama/Llama-3.2-1B-Instruct \
    --sample data/construction/construction_sample.jsonl.gz --out llama3.json
```

The first builds the 9,061 concepts from the 149,919 judged occurrences in `data/construction/`
(with NLTK 3.9.2: WordNet folding depends on its version); the second builds a tokenizer's word map.

## 4. Label new data from scratch

The labeling runs three LLMs with vLLM: Llama-3.3-70B-Instruct and Qwen2.5-72B-Instruct on 2 GPUs
each, gpt-oss-120b on 1. They sample, so new labels differ from the released ones.

```bash
python -m construction.collect --shards 'fineweb/sample/100BT/000_0000[01].parquet' --out work/occurrences.jsonl
python -m construction.label prep --workdir work --occurrences work/occurrences.jsonl
python -m construction.label wordnet --workdir work      # WordNet candidates
python -m construction.label gen --workdir work          # Llama-3.3-70B-Instruct, with the sentence
python -m construction.label freegen --workdir work      # Qwen2.5-72B-Instruct, without it
python -m construction.label pool --workdir work         # the candidates of each occurrence
python -m construction.label judge --workdir work        # gpt-oss-120b, 3 answers per occurrence
python -m construction.label refill --workdir work       # -> work/judged_final.jsonl
```

Then the judge weights and the documents, and the training and held-out files of a tokenizer:

```bash
python - <<'EOF'
import json, pyarrow.parquet as pq
from conceptizer import targets
rows = (json.loads(line) for line in open("work/judged_final.jsonl"))
json.dump(targets.judge_weight_table(rows), open("work/judge_weights.json", "w"))
with open("work/documents.jsonl", "w") as f:
    for name in ("000_00000.parquet", "000_00001.parquet"):
        t = pq.read_table(f"fineweb/sample/100BT/{name}", columns=["id", "text"])
        for d, x in zip(t["id"].to_pylist(), t["text"].to_pylist()):
            f.write(json.dumps({"doc_id": d, "text": x}) + "\n")
EOF
python -m construction.pack --tokenizer meta-llama/Llama-3.2-1B-Instruct --family llama3 \
    --judged work/judged_final.jsonl --documents work/documents.jsonl \
    --judge-weights work/judge_weights.json --construction-documents data/construction/construction_documents.json \
    --wordnet-dir ~/nltk_data/corpora/wordnet --out data/llama3_new
python -m construction.randomize --family llama3 --source data/llama3_new \
    --out data/llama3_new_randomized --output-size 128256
```

`--output-size` is the model's logit width (128256 for Llama 3, 151936 for Qwen2.5, 131072 for
Falcon3); `--wordnet-dir` is the unzipped WordNet folder of `nltk.downloader`. The concepts
themselves are built from the released concept-construction sample (step 3).

## Licenses

Code: MIT (`LICENSE`). Data: derived from FineWeb-Edu (ODC-By 1.0), outputs of
Llama-3.3-70B-Instruct (Llama 3.3 Community License; built with Llama), Qwen2.5-72B-Instruct (Qwen
license) and gpt-oss-120b (Apache 2.0), and WordNet 3.0; use it under those terms.
