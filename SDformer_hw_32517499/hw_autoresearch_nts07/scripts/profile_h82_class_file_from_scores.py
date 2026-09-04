#!/usr/bin/env python3
"""Attach Class File stats to dumped Q7 score rows. CPU only. Refuses GPU.

Expected dump: npz with scores[row, 450] float32 already Q7-quantized or raw.
Does not load a checkpoint and does not touch the live H82 trainer.
"""

from __future__ import annotations

import argparse
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
    member_jaccard_surviving,
    q7_codes,
)


def analyze_rows(scores: np.ndarray) -> dict:
    if scores.ndim != 2 or scores.shape[1] != TOKENS:
        raise ValueError(f"expected [R, {TOKENS}] scores, got {scores.shape}")
    occupied = []
    pair_equal = []
    class_j = []
    member_j = []
    files = []
    prev_codes = None
    for row in scores:
        class_file = build_class_file(row, preserve_mean=True)
        codes = np.asarray(class_file.codes, dtype=np.int64)
        occupied.append(class_file.n_occupied)
        left, right = codes[:225], codes[225:]
        pair_equal.append(float(np.mean(left == right)))
        if prev_codes is not None:
            stats = member_jaccard_surviving(prev_codes, codes)
            class_j.append(stats["class_set_jaccard"])
            member_j.append(stats["member_jaccard_surviving"])
        prev_codes = codes
        files.append(class_file.n_occupied)
    return {
        "n_rows": int(scores.shape[0]),
        "mean_occupied": float(np.mean(occupied)),
        "min_occupied": int(np.min(occupied)),
        "max_occupied": int(np.max(occupied)),
        "mean_pair_class_equal": float(np.mean(pair_equal)),
        "mean_class_set_jaccard": float(np.mean(class_j)) if class_j else None,
        "mean_member_jaccard_surviving": float(np.mean(member_j)) if member_j else None,
        "unique_q7_codes_used": int(len({int(code) for row in scores for code in q7_codes(row)})),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz")
    parser.add_argument("--scores-key", default="scores")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = np.load(args.npz)
    report = analyze_rows(np.asarray(payload[args.scores_key], dtype=np.float64))
    report["source"] = str(args.npz)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
