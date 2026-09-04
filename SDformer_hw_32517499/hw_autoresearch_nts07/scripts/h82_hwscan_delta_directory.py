#!/usr/bin/env python3
"""Delta directory across *hardware-scan* T450 rows, not intra-window rows.

H85 deltas adjacent spatial rows inside one 15x15 window. The 41% PTPX store
is the active list that is rebuilt every hardware T450 row (138-row scan of
windows). This TLM treats consecutive sliding windows as consecutive scan
rows and encodes:

    shared / insert / delete  of class names
    member insert / delete    at the same relative pair index

Class-set reuse without member reuse cannot cut that store.
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
    build_class_file,
    jaccard,
    member_sets,
    q7_codes,
    window_codes,
    SCORE_LO,
    SCORE_HI,
)


def field_windows(field: np.ndarray) -> list[np.ndarray]:
    """Row-major 15x15 windows. Adjacent list items are adjacent scan rows."""

    height, width, _ = field.shape
    rows = []
    for row in range(height - 15 + 1):
        for col in range(width - 15 + 1):
            rows.append(window_codes(field, row, col))
    return rows


def pair_members(codes: np.ndarray) -> dict[int, frozenset[int]]:
    """Membership by token index 0..449 (T0 then T1). This is the 41% roster."""

    grouped: dict[int, set[int]] = {}
    for index, class_id in enumerate(np.asarray(codes).tolist()):
        grouped.setdefault(int(class_id), set()).add(int(index))
    return {class_id: frozenset(tokens) for class_id, tokens in grouped.items()}


def scan_delta(prev: np.ndarray, curr: np.ndarray) -> dict:
    set_a = {int(x) for x in np.unique(prev)}
    set_b = {int(x) for x in np.unique(curr)}
    shared = set_a & set_b
    members_a = pair_members(prev)
    members_b = pair_members(curr)
    member_insert = 0
    member_delete = 0
    jaccards = []
    for class_id in shared:
        was = members_a.get(class_id, frozenset())
        now = members_b.get(class_id, frozenset())
        jaccards.append(jaccard(was, now))
        member_insert += len(now - was)
        member_delete += len(was - now)
    return {
        "class_insert": len(set_b - set_a),
        "class_delete": len(set_a - set_b),
        "class_shared": len(shared),
        "class_jaccard": jaccard(set_a, set_b),
        "member_jaccard": float(np.mean(jaccards)) if jaccards else 1.0,
        "member_insert": member_insert,
        "member_delete": member_delete,
        "member_edits": member_insert + member_delete,
        "full_rebuild_tokens": 450,
    }


def summarize(field: np.ndarray) -> dict:
    rows = field_windows(field)
    deltas = [scan_delta(rows[i], rows[i + 1]) for i in range(len(rows) - 1)]
    occupied = [int(np.unique(row).size) for row in rows]
    return {
        "n_scan_rows": len(rows),
        "n_seams": len(deltas),
        "mean_occupied": float(np.mean(occupied)),
        "mean_class_jaccard": float(np.mean([d["class_jaccard"] for d in deltas])),
        "mean_member_jaccard": float(np.mean([d["member_jaccard"] for d in deltas])),
        "mean_member_edits": float(np.mean([d["member_edits"] for d in deltas])),
        "mean_class_edits": float(
            np.mean([d["class_insert"] + d["class_delete"] for d in deltas])
        ),
        "edit_vs_rebuild": float(np.mean([d["member_edits"] / 450.0 for d in deltas])),
        "delta_beats_class_set": (
            "class-set-only delta is cheap and does not replace the 450-token roster"
        ),
        "delta_beats_members": (
            "member-delta beats full rebuild only if mean_member_edits << 450"
        ),
    }


def demo(seed: int = 138) -> dict:
    rng = np.random.default_rng(seed)
    height, width = 18, 18
    raw = rng.normal(0.0, 0.4, size=(height, width, 2)).clip(SCORE_LO, SCORE_HI)
    smooth = raw.copy()
    for _ in range(8):
        pad = np.pad(smooth, ((1, 1), (1, 1), (0, 0)), mode="edge")
        smooth = 0.65 * smooth + 0.35 * 0.25 * (
            pad[0:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, 0:-2] + pad[1:-1, 2:]
        )
    return {"raw": summarize(raw), "spatial_smooth": summarize(smooth)}


def main() -> None:
    out = _ROOT / "results" / "h82_class_file_isa_20260817"
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "h82_hwscan_delta_directory_v1",
        "status": "TLM_ONLY_NOT_RTL",
        "demo": demo(),
        "note": (
            "This is the 41% object. H85 intra-window row delta is a different "
            "axis and must not be sold as this directory."
        ),
    }
    (out / "hwscan_delta.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
