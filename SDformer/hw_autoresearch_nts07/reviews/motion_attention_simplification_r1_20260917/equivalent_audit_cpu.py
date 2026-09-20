"""CPU-only equivalence + STE-gradient audit for the h60 simplification candidates.

Read-only: imports the overlay module via importlib; no repo edits, no GPU, no training.

Checks
  C1  A3 canonical popcount identity (s, o, q_act, k_act) == legacy 4-mask TX  (forward exact)
  C2  A2 shared ternarization: motion XOR on an already-materialized k_event == legacy path
  C3  STE gradient scaling audit: legacy vs simplified score wrt q_orig/k_orig
      (STE gradient of sign() is 1 everywhere, so each removed duplicate pass removes one
       unit of incoming gradient; the ratio tells the exact compensation factor)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

torch.set_num_threads(1)

BSA = Path(
    "/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/"
    "H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/bsa_attention.py"
)
spec = importlib.util.spec_from_file_location("bsa_overlay_readonly3", BSA)
bsa = importlib.util.module_from_spec(spec)
sys.modules["bsa_overlay_readonly3"] = bsa
spec.loader.exec_module(bsa)

KNOBS = dict(
    enabled=True, mode="h60", center_scores=True, preserve_mean=True, alpha0=0.02,
    mismatch_penalty=0.0, score_scale=1.0, consensus_bias=0.02, consensus_score_norm="head_dim",
    single_active_penalty=0.0, bipolar_mu=0.0, k_magnitude_alpha=0.0,
    hardware_quant_enabled=False, binary_motion_xor_alpha=0.125,
)
cfg = bsa.config_from_dict(KNOBS)

T, W, Hd, N, D = 2, 6, 12, 225, 32
torch.manual_seed(0)
raw_q = torch.randn(T, W, Hd, N, D)
raw_k = torch.randn(W, Hd, T * N, D)
# spiking-like ternary-ish magnitudes (values below v_th=0.1 are silent after the neuron)
q_orig = torch.where(raw_q.abs() < 0.15, torch.zeros_like(raw_q), raw_q)
k_orig = torch.where(raw_k.abs() < 0.15, torch.zeros_like(raw_k), raw_k)


def legacy_tx(q, k):
    return bsa._ternary_alpha_xnor_token_scores(q, k, cfg)


def canon_tx(q, k):
    qe = bsa._ternary_sign_ste(bsa._qkformer_token_q(q))
    ke = bsa._ternary_sign_ste(k)
    s = (qe * ke).sum(dim=-1, keepdim=True)                 # +1 / -1 evidence
    both = ((qe != 0) & (ke != 0)).to(qe.dtype).sum(dim=-1, keepdim=True)   # o
    qa = qe.abs().sum(dim=-1, keepdim=True)
    ka = ke.abs().sum(dim=-1, keepdim=True)
    same_nz = (both + s) / 2.0
    opposite = (both - s) / 2.0
    same_zero = float(D) - qa - ka + both
    single = qa + ka - 2.0 * both
    score = (
        same_nz
        + float(cfg.alpha0) * same_zero
        - float(cfg.mismatch_penalty) * opposite
        - float(cfg.single_active_penalty) * single
    )
    score = score + float(cfg.binary_motion_xor_alpha) * bsa._binary_temporal_k_xor_popcount(q, k)
    return bsa._normalize_consensus_score(score, D, cfg)


def canon_tx_shared(q, k):
    """Same identity, but k is ternarized once and reused for motion XOR."""
    qe = bsa._ternary_sign_ste(bsa._qkformer_token_q(q))
    ke = bsa._ternary_sign_ste(k)
    s = (qe * ke).sum(dim=-1, keepdim=True)
    both = ((qe != 0) & (ke != 0)).to(qe.dtype).sum(dim=-1, keepdim=True)
    qa = qe.abs().sum(dim=-1, keepdim=True)
    ka = ke.abs().sum(dim=-1, keepdim=True)
    same_nz = (both + s) / 2.0
    same_zero = float(D) - qa - ka + both
    motion = (
        ke.reshape(k.shape[0], k.shape[1], T, N, D)
        .sub(ke.reshape(k.shape[0], k.shape[1], T, N, D).flip(dims=(2,)))
        .abs()
        .sum(dim=-1, keepdim=True)
        .reshape(k.shape[0], k.shape[1], T * N, 1)
    )
    score = same_nz + float(cfg.alpha0) * same_zero + float(cfg.binary_motion_xor_alpha) * motion
    return bsa._normalize_consensus_score(score, D, cfg)


tx_legacy = legacy_tx(q_orig, k_orig)
tx_canon = canon_tx(q_orig, k_orig)
tx_shared = canon_tx_shared(q_orig, k_orig)
print("C1 canonical popcount identity: max|diff| = %.3e  (bit-exact: %s)"
      % ((tx_legacy - tx_canon).abs().max(), bool(torch.equal(tx_legacy, tx_canon))))
print("C2 shared-ternarization variant: max|diff| = %.3e  (bit-exact: %s)"
      % ((tx_legacy - tx_shared).abs().max(), bool(torch.equal(tx_legacy, tx_shared))))

# ---- full current-config score (mu=0 => SC dropped) vs legacy fusion ----
def legacy_fused(q, k):
    tx, sc = bsa._tx_sc_fusion_score_pair(q, k, cfg)
    mu = bsa._apply_hardware_mu_quant(bsa._scheduled_bipolar_mu(_S(), cfg), cfg)
    return tx + mu * sc


class _S:
    training = False
    _h9_global_step = 0


fused_legacy = legacy_fused(q_orig, k_orig)
fused_min = canon_tx_shared(q_orig, k_orig)
print("C1b live-config score (mu=0, drop SC): max|diff| = %.3e  (bit-exact: %s)"
      % ((fused_legacy - fused_min).abs().max(), bool(torch.equal(fused_legacy, fused_min))))

# ---- C3 gradient audit ----
def grad_norm(fn, q, k):
    qg = q.clone().requires_grad_(True)
    kg = k.clone().requires_grad_(True)
    out = fn(qg, kg)
    out.sum().backward()
    return qg.grad.norm().item(), kg.grad.norm().item(), qg.grad.abs().max().item(), kg.grad.abs().max().item()


q1, k1, qm1, km1 = grad_norm(legacy_fused, q_orig, k_orig)
q2, k2, qm2, km2 = grad_norm(canon_tx_shared, q_orig, k_orig)
print("C3 STE grad legacy  : |dq|=%.4f max=%.3f |dk|=%.4f max=%.3f" % (q1, qm1, k1, km1))
print("C3 STE grad cand    : |dq|=%.4f max=%.3f |dk|=%.4f max=%.3f" % (q2, qm2, k2, km2))
print("C3 ratio |dq| legacy/cand = %.4f ; |dk| legacy/cand = %.4f" % (q1 / q2, k1 / k2))

# ---- C4 per-segment gradient audit (who actually carries gradient in the live config?) ----
print("\nC4 per-segment gradient audit (max|grad| per input, live config mu=0):")
for name, fn in (
    ("TX (args: q, k)", lambda q, k: legacy_tx(q, k)),
    ("SC (args: q, k)", lambda q, k: bsa._signed_consensus_token_scores(q, k, cfg)),
    ("motion-XOR only (args: q, k)", lambda q, k: bsa._binary_temporal_k_xor_popcount(q, k) * 1.0),
    ("live fused mu=0 (args: q, k)", lambda q, k: legacy_fused(q, k)),
    ("candidate canon+shared", lambda q, k: canon_tx_shared(q, k)),
    ("gate (shiftmax of live fused)", lambda q, k: bsa.shiftmax(
        legacy_fused(q, k) - legacy_fused(q, k).mean(dim=2, keepdim=True), dim=2, eps=1e-6)),
):
    qg = q_orig.clone().requires_grad_(True)
    kg = k_orig.clone().requires_grad_(True)
    out = fn(qg, kg)
    out.sum().backward()
    gq = "None(no grad path)" if qg.grad is None else "max=%.6f nonzero=%d/%d" % (
        float(qg.grad.abs().max()), int((qg.grad.abs() > 0).sum()), qg.grad.numel())
    gk = "None(no grad path)" if kg.grad is None else "max=%.6f nonzero=%d/%d" % (
        float(kg.grad.abs().max()), int((kg.grad.abs() > 0).sum()), kg.grad.numel())
    print("  %-32s dq %-28s dk %s" % (name, gq, gk))
