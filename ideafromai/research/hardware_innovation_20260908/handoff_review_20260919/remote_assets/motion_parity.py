"""Parity check: h60 live path (H67 ep35 cfg) vs simplified popcount datapaths.

Path A: exact replication of bsa_attention.py h60 branch (lines 6092-6141)
        using the repo's own helper functions.
Path B: S1 strip-dead-compute + popcount identity
        64*s = 65*o - (q+k) + 16*m  (exact integer identity of A's pre-quant score)
Path C: S2 shift-L1 gate: nonnegative integer score normalized by next pow2
        row sum, no amax / 2^x LUT (deviation measured, not exact).
"""
import sys, math
import torch

sys.path.insert(0, "/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN")
import bsa_attention as B

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"

# H67 ep35 config (dsec_fullres_w15_H67_ep35_M29_rank3_factor_atlif_ft5_20260822.yml)
cfg = B.config_from_dict({
    "enabled": True, "mode": "h60",
    "center_scores": True, "preserve_mean": True,
    "alpha0": 0.015625, "mismatch_penalty": 0.0,
    "consensus_bias": 0.02, "consensus_score_norm": "head_dim",
    "single_active_penalty": 0.0, "bipolar_mu": 0.0,
    "k_magnitude_alpha": 0.0, "hardware_quant_enabled": True,
    "hardware_score_step": 0.0078125, "hardware_score_min": -2.0, "hardware_score_max": 2.0,
    "hardware_gate_step": 0.0078125, "hardware_gate_min": 0.0, "hardware_gate_max": 2.0,
    "binary_motion_xor_alpha": 0.25,
})

T, Bc, H, N, D = 2, 2, 3, 225, 32
M = T * N
# realistic ternary events: ~25% active, symmetric sign
def rand_event(shape):
    x = torch.randn(shape, device=dev)
    e = torch.zeros(shape, device=dev)
    act = torch.rand(shape, device=dev) < 0.25
    e[act] = torch.sign(x[act])
    return e

q_orig = rand_event((T, Bc, H, N, D))
k_orig = rand_event((Bc, H, M, D))

n_tokens = k_orig.shape[2]

# ---------- Path A: exact h60 code path ----------
def path_a(q_orig, k_orig):
    tx_scores, sc_scores = B._tx_sc_fusion_score_pair(q_orig, k_orig, cfg)
    scores = tx_scores + 0.0 * sc_scores          # bipolar_mu = 0
    if cfg.center_scores:
        scores = scores - scores.mean(dim=2, keepdim=True)
    scores = B._apply_hardware_score_quant(scores, cfg)
    gate = B.shiftmax(scores, dim=2, eps=cfg.eps)
    if cfg.preserve_mean:
        gate = gate * float(n_tokens)
    gate = B._apply_hardware_gate_quant(gate, cfg)
    attn = k_orig.mul(gate)
    return attn, gate, scores

# ---------- Path B: popcount identity (S1) ----------
def path_b(q_orig, k_orig):
    q_e = B._ternary_sign_ste(B._qkformer_token_q(q_orig))       # [B,H,M,D]
    k_e = B._ternary_sign_ste(k_orig)
    kp = k_e.reshape(Bc, H, T, N, D).flip(dims=(2,)).reshape(Bc, H, M, D)
    qp = (q_e > 0); qn = (q_e < 0)
    kp1 = (k_e > 0); kn = (k_e < 0); ka = (k_e != 0)
    o = ((qp & kp1) | (qn & kn)).sum(-1, keepdim=True).to(torch.int64)   # same_nonzero
    opp = ((qp & kn) | (qn & kp1)).sum(-1, keepdim=True).to(torch.int64)  # opposite polarity
    q_cnt = (qp | qn).sum(-1, keepdim=True).to(torch.int64)
    k_cnt = ka.sum(-1, keepdim=True).to(torch.int64)
    m = (k_e - kp).abs().sum(-1, keepdim=True).to(torch.int64)   # |ΔK| popcount (opposite=2)
    # exact integer identity: s = o + alpha0*(D - q - k + o + opp) + alpha_m*m
    # alpha0 = 1/64, alpha_m = 1/4  =>  64*s*D = 65*o - q - k + opp + 16*m + D
    s64 = 65 * o - q_cnt - k_cnt + opp + 16 * m + D
    scores = s64.to(torch.float32) / (64.0 * D)
    if cfg.center_scores:
        scores = scores - scores.mean(dim=2, keepdim=True)
    scores = B._apply_hardware_score_quant(scores, cfg)
    gate = B.shiftmax(scores, dim=2, eps=cfg.eps)
    if cfg.preserve_mean:
        gate = gate * float(n_tokens)
    gate = B._apply_hardware_gate_quant(gate, cfg)
    attn = k_orig.mul(gate)
    return attn, gate, scores

# ---------- Path C: shift-L1 gate, integer score, no 2^x LUT (S2) ----------
def path_c(q_orig, k_orig):
    q_e = B._ternary_sign_ste(B._qkformer_token_q(q_orig))
    k_e = B._ternary_sign_ste(k_orig)
    kp = k_e.reshape(Bc, H, T, N, D).flip(dims=(2,)).reshape(Bc, H, M, D)
    qp = (q_e > 0); qn = (q_e < 0)
    kp1 = (k_e > 0); kn = (k_e < 0)
    o = ((qp & kp1) | (qn & kn)).sum(-1, keepdim=True).to(torch.int64)
    q_cnt = (qp | qn).sum(-1, keepdim=True).to(torch.int64)
    k_cnt = (k_e != 0).sum(-1, keepdim=True).to(torch.int64)
    m = (k_e - kp).abs().sum(-1, keepdim=True).to(torch.int64)   # |ΔK| popcount (opposite=2)
    s = (65 * o - q_cnt - k_cnt + 16 * m).clamp_min(0)      # nonnegative integer weight
    row = s.sum(dim=2, keepdim=True).clamp_min(1)
    denom_pow = torch.ceil(torch.log2(row.to(torch.float32)))
    gate = s.to(torch.float32) / torch.pow(2.0, denom_pow)
    if cfg.preserve_mean:
        gate = gate * float(n_tokens)
    gate = B._apply_hardware_gate_quant(gate, cfg)
    attn = k_orig.mul(gate)
    return attn, gate, s.to(torch.float32)

with torch.no_grad():
    attn_a, gate_a, sc_a = path_a(q_orig, k_orig)
    attn_b, gate_b, sc_b = path_b(q_orig, k_orig)
    attn_c, gate_c, s_c = path_c(q_orig, k_orig)

print("shapes:", tuple(attn_a.shape))
print("A vs B score  max|Δ|: %.3e" % (sc_a - sc_b).abs().max().item())
print("A vs B gate   max|Δ|: %.3e" % (gate_a - gate_b).abs().max().item())
print("A vs B attn   max|Δ|: %.3e  bitexact=%s" % ((attn_a - attn_b).abs().max().item(),
      torch.equal(attn_a, attn_b)))
print("A vs C gate   max|Δ|: %.3e  mean|Δ|: %.3e" % ((gate_a - gate_c).abs().max().item(),
      (gate_a - gate_c).abs().mean().item()))
print("A vs C attn   rel err: %.3e" % ((attn_a - attn_c).abs().max().item() / attn_a.abs().max().item()))
