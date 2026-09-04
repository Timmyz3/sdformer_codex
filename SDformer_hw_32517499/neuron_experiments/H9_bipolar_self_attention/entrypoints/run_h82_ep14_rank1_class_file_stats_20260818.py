#!/usr/bin/env python3
"""H82 ep14 rank-1 class-file statistics over valid825 (frozen operator).

Collects the data the hardware side gates on:
- C distribution per (window, head): p50/p95/max (gate: p95 C <= 192)
- temporal equal-pair rate, overall and active-pair only (both tokens nonzero score)
- descriptor ratio (2*PAIRS - eq_pairs) / 450 (gate: <= 0.60)
- within-window adjacent-column member Jaccard (H86 member-delta axis),
  class-level survive/insert/delete counts per adjacent column pair
- adjacent-window member Jaccard in scan order (row-major; pairs that wrap
  a row boundary are not spatially adjacent - caveat recorded in report)

Reuses the standard eval pipeline (eval_DSEC_flow_SNN.valid_test) and
monkeypatches _class_major_shiftmax_gate to capture per-window stats.
Does not modify the overlay file on disk.

Usage:
  python run_h82_ep14_rank1_class_file_stats_20260818.py [--max-samples N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
CFG = EXP / "configs/generated/dsec_fullres_w15_H82_class_major_ttx_ft15.yml"
CKPT = (
    EXP
    / "results/dsec_fullres_w15_H82_class_major_ttx_ft15_20260817/checkpoint_epoch14.pth"
)
OUT = (
    EXP
    / "results/dsec_fullres_w15_H82_class_major_ttx_ft15_20260817/rank1_class_file_stats_20260818"
)
FROZEN_SHA = "807a50e0c63f4800fbda778adf9e47b7a4dd2610138aec75ca105ca7e3ba2250"

os.environ.setdefault("SDFORMER_USE_MLFLOW", "0")
os.environ.setdefault("SDFORMER_MLFLOW_MODEL_LOGGING", "0")
os.environ.setdefault("SDFORMER_SNN_BACKEND", "cupy")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(REPO / "third_party/SDformerFlow"))

parser = argparse.ArgumentParser()
parser.add_argument("--max-samples", type=int, default=0)
cli, _ = parser.parse_known_args()
MAX_SAMPLES = cli.max_samples or 0

# eval_DSEC_flow_SNN auto-installs the H9 overlay from sys.argv at import time.
sys.argv = [
    sys.argv[0],
    "--config", str(CFG),
    "--checkpoint", str(CKPT),
    "--path_results", str(OUT),
    "--mode", "valid",
]
if MAX_SAMPLES:
    sys.argv += ["--max-samples", str(MAX_SAMPLES)]

import torch  # noqa: E402
import eval_DSEC_flow_SNN as E  # noqa: E402

# After overlay install, import the h82 operator module and patch the gate.
import models.STSwinNet_SNN.bsa_attention as bsa  # noqa: E402

_COLLECT = {
    "C": [],            # per (window, head): occupied classes
    "eq": [],           # per (window, head): temporal equal-pair rate
    "eq_active": [],    # per (window, head): equal rate over both-nonzero pairs
    "eq_active_frac": [],  # per (window, head): fraction of pairs both-nonzero
    "jaccard_col": [],  # within-window adjacent-column member Jaccard (surviving)
    "col_pair_survive": [],  # class-level survive/insert/delete per col pair (mean per window)
    "col_pair_insert": [],
    "col_pair_delete": [],
    "jaccard_win": [],  # adjacent-window (scan order) member Jaccard (surviving)
    "win_pair_survive": [],
    "win_pair_insert": [],
    "win_pair_delete": [],
    "n_windows": 0,
    "n_calls": 0,
}

_orig_gate = bsa._class_major_shiftmax_gate


def _wrap_class_major_shiftmax_gate(scores, cfg):
    gate, stats = _orig_gate(scores, cfg)
    try:
        _collect(stats, scores)
    except Exception as exc:  # never break the eval on collector bugs
        print(f"[rank1-stats] collector error: {exc}", flush=True)
    return gate, stats


def _collect(stats: dict, scores: torch.Tensor) -> None:
    with torch.no_grad():
        codes = stats["codes"].detach()            # [B,H,N] long
        mult = stats["multiplicity"].detach()      # [B,H,513]
        b_dim, heads, tokens = codes.shape
        if tokens != 450:
            raise ValueError(f"unexpected window tokens={tokens}, expected 450")

        c_occ = (mult > 0).sum(-1).to(torch.int32).cpu().numpy()      # [B,H]
        _COLLECT["C"].append(c_occ)

        codes2 = codes.reshape(b_dim, heads, 2, tokens // 2)          # [B,H,2,225]
        s2 = scores.detach().squeeze(-1).reshape(b_dim, heads, 2, tokens // 2)
        eq_pair = codes2[:, :, 0].eq(codes2[:, :, 1])                 # [B,H,225]
        both_active = (s2[:, :, 0] != 0) & (s2[:, :, 1] != 0)
        _COLLECT["eq"].append(eq_pair.float().mean(-1).cpu().numpy())
        denom = both_active.sum(-1).clamp_min(1)
        _COLLECT["eq_active"].append(
            (eq_pair & both_active).float().sum(-1).div(denom).cpu().numpy()
        )
        _COLLECT["eq_active_frac"].append(
            both_active.float().mean(-1).cpu().numpy()
        )

        # Within-window adjacent-column member Jaccard (H86 axis).
        # one-hot class sets per (b,h,t,row,col) over the 513-bin grid.
        grid = codes.reshape(b_dim, heads, 2, 15, 15)                 # [B,H,t,row,col]
        col_sets = torch.zeros(
            b_dim, heads, 2, 15, 15, 513, dtype=torch.bool, device=codes.device
        )
        col_sets.scatter_(5, grid.unsqueeze(-1).long(), True)
        prev = col_sets[:, :, :, :, :-1]                              # [B,H,t,row,14,513]
        curr = col_sets[:, :, :, :, 1:]
        inter = (prev & curr).sum(-1)
        union = (prev | curr).sum(-1)
        both_live = (prev.any(-1)) & (curr.any(-1))                   # class sets both nonempty
        j_col = inter.float() / union.float().clamp_min(1.0)
        surv_mask = both_live
        # class-level insert/delete per adjacent column pair
        n_insert = ((curr & ~prev).sum(-1)).float()
        n_delete = ((prev & ~curr).sum(-1)).float()
        n_survive = inter.float()
        _COLLECT["jaccard_col"].append(j_col[surv_mask].cpu().numpy())
        _COLLECT["col_pair_survive"].append(n_survive.cpu().numpy())
        _COLLECT["col_pair_insert"].append(n_insert.cpu().numpy())
        _COLLECT["col_pair_delete"].append(n_delete.cpu().numpy())

        # Adjacent-window (scan order) member Jaccard, all consecutive pairs.
        win_sets = col_sets.any(2).any(2).any(2)                      # [B,H,513] per window
        if b_dim > 1:
            w_prev, w_curr = win_sets[:-1], win_sets[1:]
            w_inter = (w_prev & w_curr).sum(-1)
            w_union = (w_prev | w_curr).sum(-1)
            w_live = (w_prev.any(-1)) & (w_curr.any(-1))
            j_win = w_inter.float() / w_union.float().clamp_min(1.0)
            _COLLECT["jaccard_win"].append(j_win[w_live].cpu().numpy())
            _COLLECT["win_pair_survive"].append(w_inter.float().cpu().numpy())
            _COLLECT["win_pair_insert"].append(
                ((w_curr & ~w_prev).sum(-1)).float().cpu().numpy()
            )
            _COLLECT["win_pair_delete"].append(
                ((w_prev & ~w_curr).sum(-1)).float().cpu().numpy()
            )
        _COLLECT["n_windows"] += int(b_dim * heads)
        _COLLECT["n_calls"] += 1


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    bsa._class_major_shiftmax_gate = _wrap_class_major_shiftmax_gate

    args = argparse.Namespace(
        runid="local", config=str(CFG), checkpoint=str(CKPT),
        path_results=str(OUT), mode="valid", bn_policy=None,
        max_samples=MAX_SAMPLES,
    )
    try:
        metrics = E.valid_test(args, E.YAMLParser(args.config))
    finally:
        bsa._class_major_shiftmax_gate = _orig_gate

    import numpy as np

    def flat(key):
        return np.concatenate([a.flatten() for a in _COLLECT[key]]) if _COLLECT[key] else np.zeros(0)

    C = flat("C")
    eq = flat("eq")
    eq_a = flat("eq_active")
    eq_af = flat("eq_active_frac")
    j_col = flat("jaccard_col")
    j_win = flat("jaccard_win")
    c_surv = flat("col_pair_survive")
    c_ins = flat("col_pair_insert")
    c_del = flat("col_pair_delete")
    w_surv = flat("win_pair_survive")
    w_ins = flat("win_pair_insert")
    w_del = flat("win_pair_delete")

    def q(arr, p):
        return float(np.quantile(arr, p / 100.0)) if arr.size else float("nan")

    PAIRS = 225
    desc_ratio = (2 * PAIRS - eq * PAIRS) / 450.0  # (2*225 - eq_pairs)/450 per window

    report = {
        "schema": "h82_rank1_class_file_stats_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(CKPT),
        "config": str(CFG),
        "operator_frozen_sha": FROZEN_SHA,
        "population": "local_DSEC_valid_file_list 825 frames"
        if not MAX_SAMPLES
        else f"max-samples smoke subset ({MAX_SAMPLES})",
        "n_windows": _COLLECT["n_windows"],
        "n_attention_calls": _COLLECT["n_calls"],
        "occupied_classes_C": {
            "p50": q(C, 50), "p95": q(C, 95), "p99": q(C, 99), "max": float(C.max()),
            "mean": float(C.mean()),
        },
        "equal_pair_rate": {
            "overall_mean": float(eq.mean()),
            "active_only_mean": float(eq_a.mean()),
            "active_pair_fraction_mean": float(eq_af.mean()),
        },
        "descriptor_ratio": {
            "mean": float(desc_ratio.mean()),
            "p95": q(desc_ratio, 95),
            "gate_lte_0.60": bool(np.nanmean(desc_ratio <= 0.60)),
        },
        "member_jaccard": {
            "within_window_adjacent_columns": {
                "mean": float(j_col.mean()),
                "p10": q(j_col, 10), "p50": q(j_col, 50), "p90": q(j_col, 90),
                "n_pairs": int(j_col.size),
            },
            "adjacent_windows_scan_order": {
                "mean": float(j_win.mean()),
                "p10": q(j_win, 10), "p50": q(j_win, 50), "p90": q(j_win, 90),
                "n_pairs": int(j_win.size),
                "caveat": "row-major scan order; pairs crossing a row boundary are not spatially adjacent",
            },
        },
        "col_pair_class_delta": {
            "survive_mean": float(c_surv.mean()),
            "insert_mean": float(c_ins.mean()),
            "delete_mean": float(c_del.mean()),
            "edits_over_450_mean": float((c_ins.mean() + c_del.mean()) / 450.0),
        },
        "win_pair_class_delta": {
            "survive_mean": float(w_surv.mean()),
            "insert_mean": float(w_ins.mean()),
            "delete_mean": float(w_del.mean()),
            "edits_over_450_mean": float((w_ins.mean() + w_del.mean()) / 450.0),
        },
        "eval_metrics": metrics,
    }
    (OUT / "rank1_stats.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
