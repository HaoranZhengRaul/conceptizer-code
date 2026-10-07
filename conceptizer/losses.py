"""The training losses and objective (paper Section 4, Equations 4-8).

In one optimizer step, with I the positions that predict a next token, S the concept-target
positions and U = I \\ S the others:

    L_NTP       = -1/|I| sum_{i in I} log p_w,i(g_i)                               (5)
    L_concept   = -1/|S| sum_{i in S} sum_c q_i(c) log(p_c,i(c) + eps)               (6)
    L_retention =  1/|U| sum_{i in U} KL(p_base_w,i || p_w,i)                        (7)
    L = L_NTP + lambda L_concept + beta L_retention,  lambda = 0.6, beta = 0.5      (8)

A loss with no positions in a step is zero. p_c,i = omega^T p_w,i is the model concept
distribution (Equation 4, :meth:`Memberships.concept_distribution`). Weighted NTP replaces the
concept loss with lambda times the mean cross-entropy of the observed token over the same S.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-12
CONCEPT_WEIGHT = 0.6     # lambda
RETENTION_WEIGHT = 0.5   # beta


def shifted_logits(logits):
    """The one float32 copy of the logits that every term reads: column j predicts input_ids[:, j + 1].

    NTP, the concept loss and retention are all computed from this single tensor, so their
    gradients are summed in float32 before they reach the bfloat16 model."""
    return logits[:, :-1].float()


def token_cross_entropy(z, input_ids, attention_mask):
    """Per-position -log p_w(g_i) on the shifted grid (z from :func:`shifted_logits`).

    Returns (ce [B, T-1], valid [B, T-1] bool)."""
    tgt = input_ids[:, 1:]
    valid = attention_mask[:, 1:].bool()
    ce = F.cross_entropy(z.reshape(-1, z.shape[-1]), tgt.reshape(-1),
                         reduction="none").view(tgt.shape)
    return ce, valid


def ntp_loss(ce, valid):
    """Equation 5: the mean over every valid position of the step."""
    return (ce * valid).sum() / valid.sum().clamp(min=1)


def dense_targets(targets, n_lanes, device):
    """[{lane: prob}, ...] -> [n, C+1] dense concept targets."""
    q = torch.zeros((len(targets), n_lanes), dtype=torch.float32, device=device)
    for r, t in enumerate(targets):
        for lane, v in t.items():
            q[r, int(lane)] += float(v)
    return q


def concept_loss(z, rows, cols, q, memberships):
    """Equation 6 at the step's concept-target positions (z from :func:`shifted_logits`).

    rows, cols   batch and shifted-grid indices of the concept-target positions (one per target)
    q            [n, C+1] dense concept targets aligned with rows/cols
    """
    p_w = torch.softmax(z[rows, cols].float(), dim=-1)               # next-token distributions
    p_c = memberships.concept_distribution(p_w)                      # Equation 4
    per_target = -(q * torch.log(p_c + EPS)).sum(-1)                 # l_i (Appendix B)
    return per_target.mean()


def weighted_ntp_term(ce, rows, cols):
    """Weighted NTP's extra term: the mean -log p_w(g_i) over the same concept-target positions."""
    return ce[rows, cols].mean()


def retention_loss(z, base_masked, mask):
    """Equation 7: mean KL(p_base || p_model) over the masked positions (valid, no concept target).

    z is :func:`shifted_logits` of the model; base_masked are the frozen base model's float32
    logits at the masked positions (the LoRA adapter disabled), in the same order as z[mask].
    """
    lp_model = torch.log_softmax(z[mask].float(), dim=-1)
    lp_base = torch.log_softmax(base_masked.float(), dim=-1)
    return (lp_base.exp() * (lp_base - lp_model)).sum(-1).mean()


def objective(l_ntp, l_concept=None, l_retention=None, lam=CONCEPT_WEIGHT, beta=RETENTION_WEIGHT):
    """Equation 8, added in this order; a term with no positions in the step is absent (zero)."""
    total = l_ntp
    if l_concept is not None:
        total = total + lam * l_concept
    if l_retention is not None:
        total = total + beta * l_retention
    return total
