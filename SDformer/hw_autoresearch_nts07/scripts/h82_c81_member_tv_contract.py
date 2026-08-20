#!/usr/bin/env python3
"""C8.1-v2 contract: survive-class member TV, not intra-window score TV.

The live H82 regularizer penalizes spatial score TV inside one T450 window.
A CPU sliding-window proxy shows that this collapses occupied-class count and
only lifts east-neighbor member Jaccard from ~0.001 to ~0.021. The frozen T450
pack needs member Jaccard >> 0.30 before a delta directory can touch the 41%
active store. This file is the next-FT operator add-on. It is not wired into
the running trainer and must not be imported by bsa_attention.py while H82
ft15 is live.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.h82_class_file_reference import (  # noqa: E402
    TOKENS,
    member_jaccard_surviving,
    q7_codes,
)


def member_tv_loss(codes_a: np.ndarray, codes_b: np.ndarray) -> float:
    """Mean Hamming of membership indicators over surviving classes.

    codes_* are integer class_id rows of length 450. Empty surviving set → 0.
    """

    if codes_a.shape != codes_b.shape or codes_a.size != TOKENS:
        raise ValueError("member TV expects two length-450 class_id rows")
    set_a = {int(code) for code in np.unique(codes_a)}
    set_b = {int(code) for code in np.unique(codes_b)}
    surviving = set_a & set_b
    if not surviving:
        return 0.0
    losses = []
    for class_id in surviving:
        memb_a = codes_a == class_id
        memb_b = codes_b == class_id
        losses.append(
            float(np.mean(np.abs(memb_a.astype(np.float64) - memb_b.astype(np.float64))))
        )
    return float(np.mean(losses))


def adjacent_row_pairs(codes_rows: np.ndarray) -> list[tuple[int, int]]:
    """Hardware-scan adjacency: consecutive T450 rows in the same head."""

    n_rows = int(codes_rows.shape[0])
    return [(index, index + 1) for index in range(n_rows - 1)]


def evaluate_member_tv(codes_rows: np.ndarray) -> dict[str, float]:
    pairs = adjacent_row_pairs(codes_rows)
    tvs = []
    jaccards = []
    for left, right in pairs:
        tvs.append(member_tv_loss(codes_rows[left], codes_rows[right]))
        jaccards.append(
            member_jaccard_surviving(codes_rows[left], codes_rows[right])[
                "member_jaccard_surviving"
            ]
        )
    return {
        "n_row_pairs": float(len(pairs)),
        "mean_member_tv": float(np.mean(tvs)) if tvs else 0.0,
        "mean_member_jaccard": float(np.mean(jaccards)) if jaccards else 1.0,
    }


def demo_loss_tracks_jaccard(seed: int = 821) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    stable = np.tile(rng.integers(0, 12, size=(1, TOKENS)), (8, 1))
    # flip 5% of assignments on later rows
    churned = stable.copy()
    for row in range(1, 8):
        flip = rng.random(TOKENS) < 0.35
        churned[row, flip] = rng.integers(0, 12, size=int(flip.sum()))
    stable_m = evaluate_member_tv(stable)
    churn_m = evaluate_member_tv(churned)
    return {
        "stable_tv": stable_m["mean_member_tv"],
        "stable_jaccard": stable_m["mean_member_jaccard"],
        "churn_tv": churn_m["mean_member_tv"],
        "churn_jaccard": churn_m["mean_member_jaccard"],
    }


def contract() -> dict:
    return {
        "schema": "h82_c81_member_tv_v2",
        "status": "NEXT_FT_ONLY_NOT_WIRED",
        "replaces": "intra-window spatial score TV weight 0.01",
        "loss": "mean Hamming of surviving-class membership across adjacent T450 rows",
        "software_hook": (
            "Dump class_id[row, 450] in hardware scan order. Do not change the "
            "running H82 ft15 graph."
        ),
        "rtl_gate": "enable ST_PATCH only if rank-1 surviving member Jaccard >= 0.60",
        "patch_threshold_rationale": (
            "Frozen pack is 0.30. 0.60 means most of a surviving class roster "
            "is a delta, so the 41% store can be a patch RAM plus a small rebuild."
        ),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "results" / "h82_class_file_isa_20260817"
    out.mkdir(parents=True, exist_ok=True)
    demo = demo_loss_tracks_jaccard()
    payload = {"contract": contract(), "demo": demo}
    (out / "c81_member_tv_v2.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
