#!/usr/bin/env python3
"""Table G quartile bars and a cleaner qualitative error grid."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TABLE = Path(
    "/root/private_data/work/sdformer_codex/SDformer/neuron_autoresearch/DSEC_DENSITY_QUARTILE_TABLE_G_20260817.json"
)
ROOT = Path(
    "/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/dsec_density_table_g_four_line_20260817"
)
OUT = Path(
    "/root/private_data/work/sdformer_codex/SDformer/neuron_autoresearch/figures/date_four_line_20260817"
)
COLORS = {"NB0": "#607D8B", "H81": "#0072B2", "H67": "#D55E00", "Local5": "#009E73"}
LABELS = {"NB0": "NB0", "H81": "H81 TTX", "H67": "H67 Motion", "Local5": "Local5"}
FRAME_LABELS = {
    "zurich_city_05_a_0191.npy": "Q1 lowest density",
    "zurich_city_11_c_0041.npy": "Q1 median",
    "zurich_city_02_e_0301.npy": "Q4 median",
    "zurich_city_06_a_0361.npy": "Q4 highest density",
}
METHODS = [("nb0", "NB0"), ("h81", "H81"), ("h67", "H67"), ("local5", "Local5")]


def flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    u, v = flow[0], flow[1]
    mag = np.sqrt(u * u + v * v)
    ang = np.arctan2(v, u)
    hsv = np.zeros(u.shape + (3,), dtype=np.float32)
    hsv[..., 0] = (ang + np.pi) / (2.0 * np.pi)
    hsv[..., 1] = 1.0
    hsv[..., 2] = np.clip(mag / (float(np.percentile(mag, 99)) + 1e-6), 0.0, 1.0)
    return matplotlib.colors.hsv_to_rgb(hsv)


def plot_table_g(table: dict) -> None:
    quartiles = ["Q1", "Q2", "Q3", "Q4"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5))
    x = np.arange(len(quartiles))
    width = 0.18
    for ax, metric, ylabel, ylim in (
        (axes[0], "AEE", "Frame-equal AEE", (0.0, 1.85)),
        (axes[1], "DSEC_Fl", "Frame-equal Fl (%)", (0.0, 13.5)),
    ):
        for i, name in enumerate(("NB0", "H81", "H67", "Local5")):
            block = next(item for item in table["lines"] if item["id"] == name)
            vals = [block["quartiles"][q][metric] for q in quartiles]
            ax.bar(
                x + (i - 1.5) * width,
                vals,
                width,
                color=COLORS[name],
                label=LABELS[name],
                edgecolor="white",
                linewidth=0.4,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(["Q1 low", "Q2", "Q3", "Q4 high"])
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.28)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_dsec_density_table_g.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig_dsec_density_table_g.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_qual() -> None:
    files = [line.strip() for line in (ROOT / "selected_frames.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    n_row, n_col = len(files), 1 + len(METHODS)
    fig, axes = plt.subplots(n_row, n_col, figsize=(2.3 * n_col, 1.95 * n_row))
    last_im = None
    for r, name in enumerate(files):
        packs = {}
        for mid, _ in METHODS:
            packs[mid] = dict(np.load(ROOT / mid / "selected_frames" / f"{Path(name).stem}.npz", allow_pickle=True))
        gt = packs["nb0"]["gt"]
        mask = np.squeeze(packs["nb0"]["mask"]) > 0
        axes[r, 0].imshow(flow_to_rgb(gt))
        axes[r, 0].set_ylabel(FRAME_LABELS.get(name, Path(name).stem), fontsize=7)
        if r == 0:
            axes[r, 0].set_title("GT flow", fontsize=8)
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
        for c, (mid, label) in enumerate(METHODS, start=1):
            aee = packs[mid]["aee"]
            show = np.full_like(aee, np.nan, dtype=np.float32)
            show[mask] = aee[mask]
            axes[r, c].set_facecolor("#111111")
            last_im = axes[r, c].imshow(show, cmap="inferno", vmin=0.0, vmax=3.0)
            mean_aee = float(np.nanmean(show))
            if r == 0:
                axes[r, c].set_title(label, fontsize=8)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            axes[r, c].text(
                0.04,
                0.96,
                f"{mean_aee:.2f}",
                transform=axes[r, c].transAxes,
                va="top",
                ha="left",
                fontsize=7,
                color="white",
                bbox={"facecolor": "black", "alpha": 0.55, "pad": 1.5, "edgecolor": "none"},
            )
    fig.subplots_adjust(left=0.10, right=0.90, top=0.93, bottom=0.08, wspace=0.04, hspace=0.08)
    cax = fig.add_axes([0.92, 0.12, 0.015, 0.72])
    cb = fig.colorbar(last_im, cax=cax)
    cb.set_label("endpoint error (px)", fontsize=8)
    fig.savefig(OUT / "fig_dsec_qualitative_density.png", dpi=300)
    fig.savefig(OUT / "fig_dsec_qualitative_density.pdf")
    plt.close(fig)


def main() -> int:
    table = json.loads(TABLE.read_text(encoding="utf-8"))
    plot_table_g(table)
    plot_qual()
    print(f"wrote Table G and qualitative figures under {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
