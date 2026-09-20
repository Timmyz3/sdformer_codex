"""Focused A/B re-check of the h60 TX motion-XOR delta (CPU only, read-only overlay)."""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(1)

BSA = Path(
    "/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/"
    "H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/bsa_attention.py"
)
spec = importlib.util.spec_from_file_location("bsa_overlay_readonly2", BSA)
bsa = importlib.util.module_from_spec(spec)
sys.modules["bsa_overlay_readonly2"] = bsa
spec.loader.exec_module(bsa)

KNOBS = dict(
    enabled=True, mode="h60", center_scores=True, preserve_mean=True, alpha0=0.02,
    mismatch_penalty=0.0, score_scale=1.0, consensus_bias=0.02, consensus_score_norm="head_dim",
    single_active_penalty=0.0, bipolar_mu=0.0, k_magnitude_alpha=0.0,
    hardware_quant_enabled=False, binary_motion_xor_alpha=0.125,
)

WINS, HEADS, NSP, D, T = 32, 12, 225, 32, 2
torch.manual_seed(0)
q_orig = torch.randn(T, WINS, HEADS, NSP, D) * 0.5
k_orig = torch.randn(WINS, HEADS, T * NSP, D) * 0.5

cfg_m = bsa.config_from_dict(KNOBS)
cfg_0 = bsa.config_from_dict(dict(KNOBS, binary_motion_xor_alpha=0.0))

REPS = 10


def best(fn):
    fn()
    times = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times), sum(times) / len(times)


def report(name, fn):
    lo, avg = best(fn)
    print("%-34s min %8.2f ms   avg %8.2f ms" % (name, lo * 1e3, avg * 1e3))
    return lo


tx_m = report("TX motion=1/8", lambda: bsa._ternary_alpha_xnor_token_scores(q_orig, k_orig, cfg_m))
tx_0 = report("TX motion=0", lambda: bsa._ternary_alpha_xnor_token_scores(q_orig, k_orig, cfg_0))
mot = report("motion-popcount isolated", lambda: bsa._binary_temporal_k_xor_popcount(q_orig, k_orig))
sc = report("SC isolated", lambda: bsa._signed_consensus_token_scores(q_orig, k_orig, cfg_m))
terq = report("ternarize q_token STE", lambda: bsa._ternary_sign_ste(bsa._qkformer_token_q(q_orig)))
terk = report("ternarize k STE", lambda: bsa._ternary_sign_ste(k_orig))
print("delta TX(motion) - TX(no motion) = %.2f ms ; isolated motion = %.2f ms ; SC = %.2f ms"
      % ((tx_m - tx_0) * 1e3, mot * 1e3, sc * 1e3))

# greedy: what fraction of TX is the two ternarizations?
print("share: tern(q)+tern(k) = %.1f%% of TX(no motion); isolated motion/SC ratio = %.2f"
      % (100.0 * (terq + terk) / tx_0, mot / sc))
