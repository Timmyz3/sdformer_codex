#!/usr/bin/env python3
"""Qualitative flow/error maps for frozen density-selected DSEC frames."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(
    "/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/dsec_density_table_g_four_line_20260817"
)
OUT = Path(
    "/root/private_data/work/sdformer_codex/SDformer/neuron_autoresearch/figures/date_four_line_20260817"
)
METHODS = [("nb0", "NB0"), ("h81", "H81"), ("h67", "H67"), ("local5", "Local5")]


def flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    u = flow[0]
    v = flow[1]
    mag = np.sqrt(u * u + v * v)
    ang = np.arctan2(v, u)
    hsv = np.zeros(u.shape + (3,), dtype=np.float32)
    hsv[..., 0] = (ang + np.pi) / (2.0 * np.pi)
    hsv[..., 1] = 1.0
    scale = float(np.percentile(mag, 99)) + 1e-6
    hsv[..., 2] = np.clip(mag / scale, 0.0, 1.0)
    return matplotlib.colors.hsv_to_rgb(hsv)


def load_npz(method: str, stem: str) -> dict | None:
    path = ROOT / method / "selected_frames" / f"{stem}.npz"
    if not path.is_file():
        return None
    return dict(np.load(path, allow_pickle=True))


def main() -> int:
    frame_list = ROOT / "selected_frames.txt"
    if not frame_list.is_file():
        print("no selected frame list")
        return 1
    files = [line.strip() for line in frame_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    available = []
    for name in files:
        packs = {mid: load_npz(mid, Path(name).stem) for mid, _ in METHODS}
        if all(packs.values()):
            available.append((name, packs))
    if not available:
        print("selected-frame dumps not ready")
        return 2
    n_row = len(available)
    n_col = 1 + len(METHODS)
    fig, axes = plt.subplots(n_row, n_col, figsize=(2.15 * n_col, 1.85 * n_row))
    if n_row == 1:
        axes = np.expand_dims(axes, 0)
    for r, (name, packs) in enumerate(available):
        gt = packs["nb0"]["gt"]
        mask = packs["nb0"]["mask"]
        vis = np.squeeze(mask) > 0
        axes[r, 0].imshow(flow_to_rgb(gt))
        axes[r, 0].set_ylabel(Path(name).stem.replace("_", "\n"), fontsize=6)
        if r == 0:
            axes[r, 0].set_title("GT", fontsize=8)
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
        for c, (mid, label) in enumerate(METHODS, start=1):
            aee = packs[mid]["aee"]
            show = np.where(vis, aee, np.nan)
            im = axes[r, c].imshow(show, cmap="magma", vmin=0.0, vmax=4.0)
            mean_aee = float(np.nanmean(show))
            if r == 0:
                axes[r, c].set_title(label, fontsize=8)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            axes[r, c].text(
                0.03,
                0.97,
                f"{mean_aee:.2f}",
                transform=axes[r, c].transAxes,
                va="top",
                ha="left",
                fontsize=6,
                color="white",
            )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.04, wspace=0.04, hspace=0.08)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_dsec_qualitative_density.png", dpi=300)
    fig.savefig(OUT / "fig_dsec_qualitative_density.pdf")
    plt.close(fig)
    print(f"wrote {OUT / 'fig_dsec_qualitative_density.png'} frames={len(available)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
