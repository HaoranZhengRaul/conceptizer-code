"""LoRA post-training with NTP, the concept loss and the retention loss (paper Section 4, Appendix D).

    python -m conceptizer.train --model meta-llama/Llama-3.2-1B-Instruct --family llama3 \
        --train data/llama3/train_windows.jsonl --method conceptizer --seed 0 --out runs/llama1b_conceptizer_s0

Methods: ``ntp`` (NTP post-training: NTP + retention), ``conceptizer`` (adds the concept loss),
``randomized`` (Conceptizer on the Randomized-synonyms training file), ``weighted_ntp`` (the concept
loss replaced by lambda times the mean cross-entropy of the observed token at the same positions).
Every method uses the same file order, recipe and retention positions. With strict deterministic
kernels a run repeats exactly on the same GPU type and software.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch

from . import data, losses, models, targets, vocabulary

RECIPE = {
    "lora_rank": 16, "lora_alpha": 32, "lora_dropout": 0.05,
    "learning_rate": 7e-5, "weight_decay": 0.01, "betas": (0.9, 0.999), "adam_eps": 1e-8,
    "warmup_steps": 100, "steps": 50000, "windows_per_step": 16, "clip_norm": 1.0,
    "concept_weight": losses.CONCEPT_WEIGHT, "retention_weight": losses.RETENTION_WEIGHT,
    "sigma": targets.SIGMA_SYNONYM_IN_CLOSEST, "dtype": "bfloat16",
}
ROOT = Path(__file__).resolve().parents[1]


def lr_multiplier(step, warmup, horizon):
    """Linear warmup over ``warmup`` steps, then cosine decay to zero at ``horizon``."""
    if step < warmup:
        return (step + 1) / warmup
    p = min(1.0, max(0.0, (step - warmup) / max(horizon - warmup, 1)))
    return 0.5 * (1.0 + math.cos(math.pi * p))


def deterministic_runtime():
    """The paper's numerical policy: strict deterministic GPU kernels, so a run repeats exactly on
    the same GPU type and software (cuBLAS reads its workspace setting before its first use)."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)


def build_model(name, seed, device, revision=None):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    deterministic_runtime()
    torch.manual_seed(seed)      # before loading, as in the paper's runs: the seed sets the LoRA
    #                              initialization (CPU generator) and the dropout masks (GPU generator)
    hub = models.hub_kwargs(name, revision)
    tok = AutoTokenizer.from_pretrained(name, **hub)
    on_gpu = str(device).startswith("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16, **({"device_map": {"": torch.device(device).index or 0}} if on_gpu else {}), **hub)
    if not on_gpu:
        model = model.to(device)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=RECIPE["lora_rank"], lora_alpha=RECIPE["lora_alpha"], lora_dropout=RECIPE["lora_dropout"],
        bias="none", task_type="CAUSAL_LM", target_modules="all-linear"))   # output head excluded
    return tok, model


def step_loss(model, batch, method, memberships, pad_id, device):
    """The loss of one step, in a fixed order: the frozen base model's logits first, then NTP, the
    concept term and retention, all read off one float32 copy of the logits (the order fixes the
    floating-point rounding, so that runs repeat step for step)."""
    ids, att = data.collate(batch, pad_id, device)
    valid = att[:, 1:].bool()
    sel = data.concept_target_positions(batch)
    rows = torch.tensor([r for r, _, _ in sel], dtype=torch.long, device=device)
    cols = torch.tensor([p for _, p, _ in sel], dtype=torch.long, device=device)
    keep = valid.clone()                          # retention: every valid position that is not a
    if sel:                                       # concept-target position (all methods)
        keep[rows, cols] = False
    base = None
    if bool(keep.any()):                          # the frozen base model; draws no random numbers
        with torch.no_grad(), model.disable_adapter():
            base = model(input_ids=ids, attention_mask=att).logits[:, :-1][keep].float()
    z = losses.shifted_logits(model(input_ids=ids, attention_mask=att).logits)
    ce, _ = losses.token_cross_entropy(z, ids, att)
    l_ntp = losses.ntp_loss(ce, valid)
    l_aux = l_ret = None
    if sel and method == "weighted_ntp":
        l_aux = losses.weighted_ntp_term(ce, rows, cols)
    elif sel and method in ("conceptizer", "randomized"):
        q = losses.dense_targets([targets.target_from_entry(e, memberships) for _, _, e in sel],
                                 memberships.C + 1, device)
        l_aux = losses.concept_loss(z, rows, cols, q, memberships)
    if base is not None:
        l_ret = losses.retention_loss(z, base, keep)
    total = losses.objective(l_ntp, l_aux, l_ret)
    return total, {"ntp": float(l_ntp), "aux": None if l_aux is None else float(l_aux),
                   "retention": None if l_ret is None else float(l_ret),
                   "n_targets": len(sel), "n_tokens": int(valid.sum())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", help="model revision (default: the paper's, for its 9 models)")
    ap.add_argument("--family", required=True, choices=["llama3", "qwen2.5", "falcon3"])
    ap.add_argument("--train", required=True, help="training windows (JSONL)")
    ap.add_argument("--method", required=True, choices=["ntp", "conceptizer", "randomized", "weighted_ntp"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=RECIPE["steps"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    tok, model = build_model(args.model, args.seed, args.device, args.revision)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    memberships = vocabulary.Memberships(
        vocabulary.load_concepts(ROOT / "data/concept_vocabulary.json"),
        vocabulary.load_wordmap(ROOT / f"data/wordmaps/{args.family}.json"),
        output_size=model.get_output_embeddings().weight.shape[0])
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=RECIPE["learning_rate"], betas=RECIPE["betas"],
                            eps=RECIPE["adam_eps"], weight_decay=RECIPE["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: lr_multiplier(k, RECIPE["warmup_steps"], args.steps))
    model.train()
    log, t0 = [], time.time()
    for step, batch in enumerate(data.steps(args.train, RECIPE["windows_per_step"], args.steps)):
        total, rec = step_loss(model, batch, args.method, memberships, pad_id, args.device)
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, RECIPE["clip_norm"])
        opt.step()
        sched.step()
        if step % args.log_every == 0:
            rec.update(step=step, loss=float(total), seconds=round(time.time() - t0, 1))
            log.append(rec)
            print(json.dumps(rec), flush=True)
    model.save_pretrained(out / "adapter")
    (out / "training.json").write_text(json.dumps({"args": vars(args), "recipe": RECIPE, "log": log}, indent=1))


if __name__ == "__main__":
    main()
