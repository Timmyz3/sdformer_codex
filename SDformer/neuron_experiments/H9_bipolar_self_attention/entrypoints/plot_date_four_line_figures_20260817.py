#!/usr/bin/env python3
"""Publication figures for the four DATE algorithm lines."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LEDGER = Path(
    "/root/private_data/work/sdformer_codex/SDformer/neuron_autoresearch/DATE_FOUR_LINE_LEDGER_20260817.json"
)
OUT = Path(
    "/root/private_data/work/sdformer_codex/SDformer/neuron_autoresearch/figures/date_four_line_20260817"
)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

COLORS = {
    "NB0": "#607D8B",
    "H81": "#0072B2",
    "H67": "#D55E00",
    "Local5": "#009E73",
    "Local5_FT": "#CC79A7",
}
LABELS = {
    "NB0": "NB0",
    "H81": "H81 TTX",
    "H67": "H67 Motion-TTX",
    "Local5": "Local5 TTX",
    "Local5_FT": "Local5 DSEC-FT",
}
MARKERS = {"NB0": "s", "H81": "o", "H67": "D", "Local5": "^"}
SEQS = ["outdoor_day1", "indoor_flying1", "indoor_flying2", "indoor_flying3"]
SEQ_LABS = ["OD1", "IF1", "IF2", "IF3"]


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


def plot_pareto(ledger: dict) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for name in ("NB0", "H81", "H67", "Local5"):
        row = ledger["dsec"][name]["rank1"]
        ax.scatter(
            row["total_spikes_g"],
            row["AEE"],
            s=70,
            color=COLORS[name],
            marker=MARKERS[name],
            zorder=3,
            label=LABELS[name],
        )
        offsets = {
            "NB0": (8, 8),
            "H81": (8, 10),
            "H67": (10, -18),
            "Local5": (8, -18),
        }
        ax.annotate(
            f"{row['AEE']:.3f} / {row['total_spikes_g']:.1f}G",
            (row["total_spikes_g"], row["AEE"]),
            textcoords="offset points",
            xytext=offsets[name],
            fontsize=7,
            color=COLORS[name],
        )
    ax.set_xlabel("DSEC valid825 spikes (G)")
    ax.set_ylabel("DSEC valid825 AEE")
    ax.set_xlim(72, 138)
    ax.set_ylim(1.24, 1.50)
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig_dsec_aee_spikes_pareto")


def plot_budget(ledger: dict) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for name in ("NB0", "H81", "H67", "Local5"):
        pts = ledger["dsec"][name]["budget"]
        xs = [p["epoch"] for p in pts]
        ys = [p["AEE"] for p in pts]
        ax.plot(
            xs,
            ys,
            color=COLORS[name],
            marker=MARKERS[name],
            linewidth=1.6,
            markersize=5.5,
            label=LABELS[name],
        )
        rank_ep = ledger["dsec"][name]["rank1_epoch"]
        rank_aee = ledger["dsec"][name]["rank1"]["AEE"]
        ax.scatter([rank_ep], [rank_aee], s=80, facecolors="none", edgecolors=COLORS[name], linewidths=1.4, zorder=4)
    ax.axvline(40, color="#888888", linestyle="--", linewidth=0.8)
    ax.text(36.6, 1.463, "budget 40", fontsize=7, color="#555555")
    ax.set_xlabel("Checkpoint epoch")
    ax.set_ylabel("DSEC valid825 AEE")
    ax.set_xticks([29, 30, 34, 35, 39, 40, 44, 49])
    ax.legend(frameon=False, loc="center right")
    save(fig, "fig_dsec_budget_curve")


def plot_mvsec(ledger: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.7), gridspec_kw={"width_ratios": [3.1, 1.15]})
    ax = axes[0]
    methods = ["NB0", "H81", "H67", "Local5"]
    x = np.arange(len(SEQS))
    width = 0.18
    for i, name in enumerate(methods):
        vals = [ledger["mvsec"][name]["full"]["AEE"][seq] for seq in SEQS]
        bars = ax.bar(
            x + (i - 1.5) * width,
            vals,
            width,
            color=COLORS[name],
            label=LABELS[name],
            edgecolor="white",
            linewidth=0.4,
        )
        # mark IF1 failures
        if name in ("H81", "Local5") and vals[1] > ledger["mvsec"]["NB0"]["full"]["AEE"]["indoor_flying1"]:
            ax.scatter(
                [x[1] + (i - 1.5) * width],
                [vals[1] + 0.06],
                marker="x",
                s=28,
                color="#111111",
                zorder=5,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(SEQ_LABS)
    ax.set_ylabel("MVSEC full-sequence AEE")
    ax.set_ylim(0, 3.1)
    ax.legend(frameon=False, ncol=2, loc="upper left")

    ax2 = axes[1]
    names = ["NB0", "H81", "H67", "Local5", "Local5_FT"]
    spikes = [ledger["mvsec"][name]["full"]["spikes_g"] for name in names]
    colors = [COLORS[name] for name in names]
    bars = ax2.bar(
        range(len(names)),
        spikes,
        color=colors,
        edgecolor="white",
        linewidth=0.4,
    )
    bars[-1].set_hatch("///")
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(["NB0", "H81", "H67", "L5", "L5-FT"], rotation=0)
    ax2.set_ylabel("MVSEC full spikes (G)")
    ax2.set_ylim(0, 280)
    for bar, val in zip(bars, spikes):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 4, f"{val:.0f}", ha="center", va="bottom", fontsize=7)
    save(fig, "fig_mvsec_four_sequence")


def plot_scorecard(ledger: dict) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    keys = list(ledger["paper_fit_weights"])
    methods = ["H67", "Local5", "H81", "NB0"]
    y = np.arange(len(methods))
    left = np.zeros(len(methods))
    pal = ["#D55E00", "#009E73", "#0072B2", "#E69F00", "#56B4E9", "#999999"]
    lookup = {row["id"]: row["scores"] for row in ledger["paper_fit_scorecard"]}
    for i, key in enumerate(keys):
        vals = [lookup[m][key] for m in methods]
        ax.barh(y, vals, left=left, color=pal[i], label=key.replace("_", " "), height=0.62)
        left = left + np.array(vals)
    totals = {row["id"]: row["total"] for row in ledger["paper_fit_scorecard"]}
    for i, name in enumerate(methods):
        ax.text(totals[name] + 1.2, y[i], f"{totals[name]:.1f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[m] for m in methods])
    ax.set_xlabel("DATE-weighted paper-fit score")
    ax.set_xlim(0, 105)
    ax.legend(
        frameon=False,
        fontsize=7,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.45, 1.18),
    )
    save(fig, "fig_paper_fit_scorecard")


def main() -> int:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    plot_pareto(ledger)
    plot_budget(ledger)
    plot_mvsec(ledger)
    plot_scorecard(ledger)
    print(f"wrote figures under {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
