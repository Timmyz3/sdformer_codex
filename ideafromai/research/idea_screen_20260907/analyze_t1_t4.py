#!/usr/bin/env python3.12
"""T1–T4 grain analysis on existing packed Q/K (ep35-path). Research only."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

QK_DIR = Path(
    "/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/"
    "system_handoff/received/h67_ep35_system_trace_handoff_20260821/"
    "h67_ep35_system_trace_handoff_20260821/trace_qk_100sample_12block"
)
OUT = Path("/home/zhumd/work/sdformer_codex/ideafromai/research/idea_screen_20260907")


def unpack(p: Path):
    z = np.load(p)
    shp = tuple(int(x) for x in z["q_shape"])
    q = np.unpackbits(z["q_bits_packed"]).reshape(shp)
    k = np.unpackbits(z["k_bits_packed"]).reshape(shp)
    return q.astype(np.uint8), k.astype(np.uint8)


def run_lengths(mask: np.ndarray) -> list[int]:
    """Contiguous True runs along last axis. mask shape (..., S)."""
    out = []
    flat = mask.reshape(-1, mask.shape[-1])
    for row in flat:
        n = 0
        for v in row.tolist():
            if v:
                n += 1
            elif n:
                out.append(n)
                n = 0
        if n:
            out.append(n)
    return out


def main():
    files = sorted(QK_DIR.glob("sample*_S*_B*_attn.npz"))
    tok = 0
    token_dirty = 0
    k_zero_both = 0
    leaf_needed = 0
    q_only_dirty = 0
    k_only_dirty = 0
    both_dirty = 0
    # grains
    head_win = 0
    head_win_dirty = 0
    pack450 = 0
    pack450_dirty = 0
    runs = []
    joint = defaultdict(int)  # (kzero, dirty)
    per_stage = defaultdict(lambda: dict(tok=0, dirty=0, kzero=0, leaf=0, win=0, win_dirty=0))
    n_files = 0
    for p in files:
        q, k = unpack(p)
        # (T, X, H, S, C) = (2,1,3,225,32) typically
        assert q.shape[0] == 2
        t, x, h, s, c = q.shape
        q0, q1 = q[0], q[1]
        k0, k1 = k[0], k[1]
        q_d = (q0 != q1).any(-1)  # (X,H,S)
        k_d = (k0 != k1).any(-1)
        dirty = q_d | k_d
        kz = (k0.sum(-1) == 0) & (k1.sum(-1) == 0)
        leaf = dirty & (~kz)
        n = int(dirty.size)
        tok += n
        token_dirty += int(dirty.sum())
        k_zero_both += int(kz.sum())
        leaf_needed += int(leaf.sum())
        q_only_dirty += int((q_d & ~k_d).sum())
        k_only_dirty += int((k_d & ~q_d).sum())
        both_dirty += int((q_d & k_d).sum())
        for a, b in zip(kz.reshape(-1), dirty.reshape(-1)):
            joint[(int(a), int(b))] += 1
        # head window: OR over S
        win = dirty.reshape(x, h, s).any(-1)
        head_win += int(win.size)
        head_win_dirty += int(win.sum())
        # 450-token Shiftmax row: concat T as token, OR dirty over S and T for each head
        # dirty already ORs t0-t1; window OR over S is the 450-key row if T is stacked in the window
        pack450 += int(win.size)
        pack450_dirty += int(win.sum())
        runs.extend(run_lengths(dirty.reshape(-1, s)))
        st = p.stem.split("_")[1]
        per_stage[st]["tok"] += n
        per_stage[st]["dirty"] += int(dirty.sum())
        per_stage[st]["kzero"] += int(kz.sum())
        per_stage[st]["leaf"] += int(leaf.sum())
        per_stage[st]["win"] += int(win.size)
        per_stage[st]["win_dirty"] += int(win.sum())
        n_files += 1

    hist = defaultdict(int)
    for r in runs:
        if r <= 8:
            hist[r] += 1
        elif r <= 16:
            hist["9-16"] += 1
        else:
            hist["17+"] += 1

    report = {
        "identity": "ep35_path_100sample_12block packed QK, NOT ep34",
        "n_files": n_files,
        "tokens": tok,
        "T1_token": {
            "dirty_frac": token_dirty / tok,
            "q_only_dirty_frac": q_only_dirty / tok,
            "k_only_dirty_frac": k_only_dirty / tok,
            "both_qk_dirty_frac": both_dirty / tok,
            "k_zero_both_t_frac": k_zero_both / tok,
            "leaf_needed_frac": leaf_needed / tok,
            "ideal_token_skip_frac": 1.0 - leaf_needed / tok,
        },
        "T1_head_window_OR_over_15x15": {
            "n": head_win,
            "dirty_frac": head_win_dirty / head_win,
            "fully_clean_frac": 1.0 - head_win_dirty / head_win,
            "fail_closed_note": "If Shiftmax denom is over the 15x15 window (or 450=Tw*15*15), skip is only legal when this is 0. Fully-clean windows are the only lossless score-memo rows.",
        },
        "T3_joint_kzero_dirty": {f"kzero{a}_dirty{b}": joint[(a, b)] / tok for a in (0, 1) for b in (0, 1)},
        "T4_dirty_spatial_run_lengths": {
            "n_runs": len(runs),
            "mean": float(np.mean(runs)) if runs else None,
            "median": float(np.median(runs)) if runs else None,
            "histogram": {str(k): hist[k] for k in list(range(1, 9)) + ["9-16", "17+"]},
        },
        "per_stage": {
            st: {
                "token_dirty": v["dirty"] / v["tok"],
                "k_zero": v["kzero"] / v["tok"],
                "leaf_needed": v["leaf"] / v["tok"],
                "head_window_dirty": v["win_dirty"] / v["win"],
                "head_window_clean": 1.0 - v["win_dirty"] / v["win"],
            }
            for st, v in sorted(per_stage.items())
        },
        "verdict": {
            "token_skip_looks_real": (1.0 - leaf_needed / tok) >= 0.3,
            "window_row_skip_looks_weak": (1.0 - head_win_dirty / head_win) < 0.2,
            "do_not_sell_token_skip_as_Shiftmax_skip": True,
        },
    }
    (OUT / "T1_T4_ep35.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2)[:4000])


if __name__ == "__main__":
    main()
