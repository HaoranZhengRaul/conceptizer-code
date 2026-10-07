"""Training and held-out files: windows of token ids with their concept-target entries.

One JSON line per window (the running example, Llama-3 training file)::

    {"input_ids": [...],                                  # 256 tokens of one document, no BOS token
     "sup": [{"pos": 160,                                 # the logit at pos predicts input_ids[pos + 1]
              "set_ids": [2512, 7296, 9313, 27344],       # tokens of w_0 and its contextual synonyms w_1..w_m
              "set_w": [0.333, 0.0, 1.0, 0.667],          # share of the judge's 3 answers keeping each word
              "gold_id": 7296,                            # the observed token, input_ids[pos + 1]
              "set_kr": [0.1111, 1.0, 0.963, 0.625],      # judge weights r_j (1.0 for the observed word)
              "words": ["care", "safety", "protection", "prevention"],
              "home_lane": 1450,                          # the closest concept c* (its concept number)
              "source_occurrence": {"doc_id": "<urn:uuid:...>", "char_start": ..., "char_end": ...,
                                    "surface": "safety", ...}}]}

A document's last window may be shorter (at least 32 tokens). In a training file every entry
carries a concept target; a held-out file also keeps each rejected content word as
{"pos", "gold_id", "gate", "source_occurrence"}, gate naming the first check it failed. Training reads
the windows once, 16 consecutive windows per optimizer step, in the file's order (the documents
were shuffled once when the file was built; every method and seed reads the same order).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

WINDOW_TOKENS = 256
WINDOWS_PER_STEP = 16


def read_windows(path):
    with Path(path).open() as f:
        for line in f:
            yield json.loads(line)


def steps(path, windows_per_step=WINDOWS_PER_STEP, max_steps=None):
    """Consecutive groups of windows, one group per optimizer step, in file order."""
    batch, n = [], 0
    for w in read_windows(path):
        batch.append(w)
        if len(batch) == windows_per_step:
            yield batch
            batch, n = [], n + 1
            if max_steps is not None and n >= max_steps:
                return
    if batch and (max_steps is None or n < max_steps):
        yield batch


def collate(batch, pad_id, device):
    """Right-padded input ids and attention mask for a list of windows."""
    length = max(len(w["input_ids"]) for w in batch)
    ids = torch.full((len(batch), length), int(pad_id), dtype=torch.long)
    att = torch.zeros((len(batch), length), dtype=torch.long)
    for r, w in enumerate(batch):
        ids[r, :len(w["input_ids"])] = torch.tensor(w["input_ids"], dtype=torch.long)
        att[r, :len(w["input_ids"])] = 1
    return ids.to(device), att.to(device)


def concept_target_positions(batch):
    """The step's concept-target positions: (row, pos, entry) for every scorable entry."""
    out = []
    for r, w in enumerate(batch):
        n = len(w["input_ids"])
        for e in w.get("sup", []):
            p = int(e["pos"])
            if p >= n - 1 or not e.get("set_ids"):
                continue
            assert int(w["input_ids"][p + 1]) == int(e["gold_id"]), "entry is not aligned with its token"
            assert int(e["gold_id"]) in e["set_ids"]
            out.append((r, p, e))
    return out
