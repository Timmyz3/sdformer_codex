"""CPU-only micro-profile of the deployed h60 attention score path (read-only overlay).

Purpose: measure the *relative* cost split of the live h60 segments vs the dead /
duplicated segments. No GPU, no training process, no repo modification: the overlay
module is imported read-only via importlib and only tensor ops are executed on CPU.

Usage:
  /opt/conda/envs/sdformerflow/bin/python prof_h60_attention_cpu.py [--json out.json]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import torch

BSA = Path(
    "/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/"
    "H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/bsa_attention.py"
)

T49_KNOBS = dict(
    enabled=True,
    mode="h60",
    center_scores=True,
    preserve_mean=True,
    alpha0=0.02,
    mismatch_penalty=0.0,
    score_scale=1.0,
    consensus_bias=0.02,
    consensus_score_norm="head_dim",
    eps=1.0e-06,
    single_active_penalty=0.0,
    bipolar_mu=0.0,
    bipolar_gate_min=-1.0,
    bipolar_gate_max=1.8,
    k_magnitude_alpha=0.0,
    sc_mu_schedule_enabled=False,
    hardware_quant_enabled=False,
    binary_motion_xor_alpha=0.125,
    castling_matrix_aux_weight=0.0,
)

DEPLOY_KNOBS = dict(
    T49_KNOBS,
    alpha0=0.015625,
    binary_motion_xor_alpha=0.25,
    hardware_quant_enabled=True,
    hardware_mu_pow2_shift=0,
    hardware_score_step=1.0 / 128.0,
    hardware_score_min=-2.0,
    hardware_score_max=2.0,
    hardware_gate_step=1.0 / 128.0,
    hardware_gate_min=0.0,
    hardware_gate_max=2.0,
)

# representative mid-stage window block: heads=12 (stage2), 2 x 15 x 15 tokens
WINDOWS = 32          # batch-window rows processed in one attention call
HEADS = 12
N_SPATIAL = 225
HEAD_DIM = 32
T_STEPS = 2


def load_bsa():
    import sys

    spec = importlib.util.spec_from_file_location("bsa_overlay_readonly", BSA)
    module = importlib.util.module_from_spec(spec)
    sys.modules["bsa_overlay_readonly"] = module
    spec.loader.exec_module(module)
    return module


def timed(fn, repeats=10, warmup=2):
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=None)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--windows", type=int, default=WINDOWS)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    bsa = load_bsa()

    wins = args.windows
    q_orig = torch.randn(T_STEPS, wins, HEADS, N_SPATIAL, HEAD_DIM) * 0.5
    k_orig = torch.randn(wins, HEADS, T_STEPS * N_SPATIAL, HEAD_DIM) * 0.5
    # spiking-like sparsity (events 0/+-1 mostly) for a second pass
    q_sp = torch.sign(q_orig) * (torch.rand_like(q_orig) < 0.1)
    k_sp = torch.sign(k_orig) * (torch.rand_like(k_orig) < 0.1)

    cfg_train = bsa.config_from_dict(T49_KNOBS)
    cfg_deploy = bsa.config_from_dict(DEPLOY_KNOBS)

    class _Stub:
        training = False
        _h9_global_step = 0

    rows = []

    def bench(name, fn, per="call"):
        secs = timed(fn)
        rows.append({"segment": name, "seconds_per_call": secs, "unit": per,
                     "ms": secs * 1e3, "ms_per_window": secs * 1e3 / wins})

    # ---- individual segments (deployed knobs = deploy config) ----
    bench("q_token_permute_reshape", lambda: bsa._qkformer_token_q(q_orig))
    bench("ternarize_q_token_STE", lambda: bsa._ternary_sign_ste(bsa._qkformer_token_q(q_orig)))
    bench("ternarize_k_STE", lambda: bsa._ternary_sign_ste(k_orig))
    bench("TX_full(alpha0,motion=1/8)", lambda: bsa._ternary_alpha_xnor_token_scores(q_orig, k_orig, cfg_deploy))
    cfg_nomotion = bsa.config_from_dict(dict(DEPLOY_KNOBS, binary_motion_xor_alpha=0.0))
    bench("TX_nomotion", lambda: bsa._ternary_alpha_xnor_token_scores(q_orig, k_orig, cfg_nomotion))
    bench("motion_xor_popcount_only", lambda: bsa._binary_temporal_k_xor_popcount(q_orig, k_orig))
    bench("SC_signed_consensus", lambda: bsa._signed_consensus_token_scores(q_orig, k_orig, cfg_deploy))
    bench("tx_sc_fusion_pair(dead SC path)", lambda: bsa._tx_sc_fusion_score_pair(q_orig, k_orig, cfg_deploy))
    scores = bsa._ternary_alpha_xnor_token_scores(q_orig, k_orig, cfg_deploy)
    bench("center_scores(mean dim=2)", lambda: scores - scores.mean(dim=2, keepdim=True))
    centered = scores - scores.mean(dim=2, keepdim=True)
    bench("shiftmax", lambda: bsa.shiftmax(centered, dim=2, eps=1e-6))
    gate0 = bsa.shiftmax(centered, dim=2, eps=1e-6)
    bench("preserve_mean_mul", lambda: gate0 * float(T_STEPS * N_SPATIAL))
    gate = gate0 * float(T_STEPS * N_SPATIAL)
    bench("attn=k_orig.mul(gate)", lambda: k_orig.mul(gate))
    bench("hw_score_quant(Q7 clamp+round STE)",
          lambda: bsa._apply_hardware_score_quant(centered, cfg_deploy))
    bench("hw_gate_quant(Q7)", lambda: bsa._apply_hardware_gate_quant(gate, cfg_deploy))
    bench("temperature_off(call+check)", lambda: bsa._event_selective_temperature(centered, q_orig, k_orig, cfg_deploy))
    bench("castling_weight_off", lambda: bsa._castling_aux_weight(_Stub(), cfg_deploy))
    bench("context_broadcast_off", lambda: bsa._window_context_broadcast(gate, cfg_deploy))
    bench("mu_quant_call", lambda: bsa._apply_hardware_mu_quant(0.0, cfg_deploy))
    bench("profile_emit_disabled(getattr early-return)",
          lambda: bsa._maybe_emit_h60_profile(None, q_orig=q_orig, k_orig=k_orig,
                                              tx_scores=scores, sc_scores=scores,
                                              fused_scores=scores, pre_quant_scores=scores,
                                              gate=gate, cfg=cfg_deploy))

    def tail_diag():
        rs = gate.sum(dim=2)
        float(rs.detach().mean().cpu())
        float(rs.detach().min().cpu())
        float(rs.detach().max().cpu())
        float(gate.detach().mean().cpu())
        float(scores.detach().mean().cpu())

    bench("tail_diag_5_host_syncs", tail_diag)

    # ---- end-to-end h60 score block replica (deploy knobs, mu=0) ----
    def h60_block(cfg):
        tx, sc = bsa._tx_sc_fusion_score_pair(q_orig, k_orig, cfg)
        mu = bsa._apply_hardware_mu_quant(bsa._scheduled_bipolar_mu(_Stub(), cfg), cfg)
        fused = tx + mu * sc
        fused = fused - fused.mean(dim=2, keepdim=True)
        fused = bsa._event_selective_temperature(fused, q_orig, k_orig, cfg)
        fused = bsa._apply_hardware_score_quant(fused, cfg)
        g = bsa.shiftmax(fused, dim=2, eps=cfg.eps)
        if cfg.preserve_mean:
            g = g * float(T_STEPS * N_SPATIAL)
        g = bsa._apply_hardware_gate_quant(g, cfg)
        return k_orig.mul(g)

    bench("h60_full_score_block_replica", lambda: h60_block(cfg_deploy))
    bench("h60_full_score_block_replica_train_knobs", lambda: h60_block(cfg_train))

    # ---- sparsity-typical inputs (spiking-like) ----
    bench("TX_full_sparse_events", lambda: bsa._ternary_alpha_xnor_token_scores(q_sp, k_sp, cfg_deploy))
    bench("motion_xor_sparse_events", lambda: bsa._binary_temporal_k_xor_popcount(q_sp, k_sp))
    bench("SC_sparse_events", lambda: bsa._signed_consensus_token_scores(q_sp, k_sp, cfg_deploy))

    out = {
        "note": "CPU-only micro-profile, torch=%s threads=%d windows=%d heads=%d tokens=%d head_dim=%d"
                % (torch.__version__, args.threads, wins, HEADS, T_STEPS * N_SPATIAL, HEAD_DIM),
        "rows": rows,
    }
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
    for row in rows:
        print("%-42s %9.3f ms    %8.4f ms/window" % (row["segment"], row["ms"], row["ms_per_window"]))


if __name__ == "__main__":
    main()
