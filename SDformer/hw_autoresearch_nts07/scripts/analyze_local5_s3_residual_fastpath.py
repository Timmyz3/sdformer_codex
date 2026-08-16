#!/usr/bin/env python3
"""Decide the next Local5 exact residual fast-path from a real window."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def read_memh(path: Path) -> list[int]:
    return [int(line.strip(), 16) for line in path.read_text().splitlines() if line.strip()]


def unpack(value: int, width: int, count: int) -> list[int]:
    mask = (1 << width) - 1
    return [(value >> (index * width)) & mask for index in range(count)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.vector_dir / "manifest.json").read_text())
    q_words = read_memh(args.vector_dir / "input_q.memh")
    k_words = read_memh(args.vector_dir / "input_candidate_k.memh")
    valid_words = read_memh(args.vector_dir / "input_valid.memh")
    score_words = read_memh(args.vector_dir / "expected_scores.memh")
    n = len(q_words)
    if not (n == len(k_words) == len(valid_words) == len(score_words)):
        raise ValueError("vector length mismatch")

    q0 = 0
    qnz = 0
    all_k_equal = 0
    all_score_equal = 0
    pairwise_k = 0
    self_k_unique_among_valid = 0
    dest_repeat_k = 0
    qnz_valid_hist = Counter()
    prev_k = None
    for q_value, k_packed, valid, score_packed in zip(
        q_words, k_words, valid_words, score_words
    ):
        ks = unpack(k_packed, 32, 5)
        scores = unpack(score_packed, 16, 5)
        signed_scores = [s - 65536 if s >= 32768 else s for s in scores]
        present = [ks[i] for i in range(5) if (valid >> i) & 1]
        present_scores = [signed_scores[i] for i in range(5) if (valid >> i) & 1]
        if q_value == 0:
            q0 += 1
            prev_k = tuple(ks)
            continue
        qnz += 1
        qnz_valid_hist[len(present)] += 1
        if present and len(set(present)) == 1:
            all_k_equal += 1
        if present_scores and len(set(present_scores)) == 1:
            all_score_equal += 1
        if len(present) >= 2:
            pairs = 0
            same = 0
            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    pairs += 1
                    if present[i] == present[j]:
                        same += 1
            if pairs and same == pairs:
                pairwise_k += 1
        if present and ks[0] in present and present.count(ks[0]) == 1:
            self_k_unique_among_valid += 1
        cur = tuple(ks)
        if prev_k is not None and cur == prev_k:
            dest_repeat_k += 1
        prev_k = cur

    qnz_rate = qnz / n if n else 0.0
    all_k_frac = all_k_equal / qnz if qnz else 0.0
    all_score_frac = all_score_equal / qnz if qnz else 0.0
    # Structural win: Q!=0 and all valid K identical so one AXNOR can broadcast.
    # Need enough of the residual (not just of the whole window).
    promote = all_k_frac >= 0.30 and qnz_rate >= 0.10
    decision = "PROMOTE_IDENTICAL_K_BROADCAST" if promote else "REJECT_WRITTEN"
    reason = (
        "Q!=0 rows have >=30% identical-K among valid candidates; "
        "a one-AXNOR broadcast remains bit-exact."
        if promote
        else (
            "Q!=0 identical-K / equal-score rates are too low or unstructured "
            "to pay a residual sidecar. Query-Silent already covers Q==0."
        )
    )
    report = {
        "schema": "local5_s3_residual_fastpath_decision_v1",
        "vector_dir": str(args.vector_dir.resolve()),
        "manifest_evidence": manifest.get("evidence"),
        "dest_rows": n,
        "q_zero": q0,
        "q_nonzero": qnz,
        "q_zero_rate": q0 / n if n else 0.0,
        "q_nonzero_rate": qnz_rate,
        "qnz_all_valid_k_equal": all_k_equal,
        "qnz_all_valid_k_equal_frac": all_k_frac,
        "qnz_all_valid_score_equal": all_score_equal,
        "qnz_all_valid_score_equal_frac": all_score_frac,
        "qnz_all_pairs_k_equal": pairwise_k,
        "qnz_self_k_unique": self_k_unique_among_valid,
        "consecutive_dest_identical_k": dest_repeat_k,
        "qnz_valid_count_hist": {str(k): v for k, v in sorted(qnz_valid_hist.items())},
        "decision": decision,
        "reason": reason,
        "claim_boundary": [
            "This is a window-local statistic, not a 21600-group claim.",
            "Reject does not weaken Query-Silent; it refuses a second slogan.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    md = [
        "# Local5 S3 residual exact-path decision",
        "",
        f"- dest rows: {n}",
        f"- Q==0: {q0} ({100 * report['q_zero_rate']:.2f}%)",
        f"- Q!=0: {qnz} ({100 * qnz_rate:.2f}%)",
        f"- Q!=0 all-valid-K identical: {all_k_equal} ({100 * all_k_frac:.2f}%)",
        f"- Q!=0 all-valid-score identical: {all_score_equal} ({100 * all_score_frac:.2f}%)",
        f"- consecutive dest identical-K: {dest_repeat_k}",
        "",
        f"**Decision: {decision}**",
        "",
        reason,
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"PASS residual decision {decision} q0={q0}/{n} identK={all_k_frac:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
