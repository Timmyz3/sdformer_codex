#!/usr/bin/env python3
"""Decide whether leftover Q!=0 / non-identical-K rows deserve another exact path."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read_memh(path: Path) -> list[int]:
    return [int(line.strip(), 16) for line in path.read_text().splitlines() if line.strip()]


def unpack(value: int, width: int, count: int) -> list[int]:
    mask = (1 << width) - 1
    return [(value >> (index * width)) & mask for index in range(count)]


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def analyze(vector_dir: Path) -> dict[str, object]:
    q_words = read_memh(vector_dir / "input_q.memh")
    k_words = read_memh(vector_dir / "input_candidate_k.memh")
    valid_words = read_memh(vector_dir / "input_valid.memh")
    score_words = read_memh(vector_dir / "expected_scores.memh")
    lengths = {
        "input_q": len(q_words),
        "input_candidate_k": len(k_words),
        "input_valid": len(valid_words),
        "expected_scores": len(score_words),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"memh length mismatch in {vector_dir}: {lengths}")
    if not q_words:
        raise ValueError(f"empty vectors in {vector_dir}")
    leftover = 0
    almost4 = 0
    one_bit = 0
    score_tie = 0
    max_delta = Counter()
    for q_value, k_packed, valid, score_packed in zip(
        q_words, k_words, valid_words, score_words
    ):
        if q_value == 0:
            continue
        ks = unpack(k_packed, 32, 5)
        scores = unpack(score_packed, 16, 5)
        signed = [s - 65536 if s >= 32768 else s for s in scores]
        present = [ks[i] for i in range(5) if (valid >> i) & 1]
        present_scores = [signed[i] for i in range(5) if (valid >> i) & 1]
        if not present:
            continue
        if len(set(present)) == 1:
            continue
        leftover += 1
        counts = Counter(present)
        top = counts.most_common(1)[0][1]
        if top >= 4:
            almost4 += 1
        diffs = []
        for left in range(len(present)):
            for right in range(left + 1, len(present)):
                diffs.append(hamming(present[left], present[right]))
        md = min(diffs) if diffs else 32
        max_delta[md] += 1
        if md == 1:
            one_bit += 1
        if present_scores and len(set(present_scores)) == 1:
            score_tie += 1
    n = leftover or 1
    promote = leftover > 0 and (almost4 / n >= 0.40) and (almost4 >= 200)
    return {
        "vector_dir": str(vector_dir.resolve()),
        "leftover_qnz_not_identk": leftover,
        "almost_4of5_k_equal": almost4,
        "almost_4of5_frac": almost4 / n if leftover else 0.0,
        "min_hamming_1": one_bit,
        "min_hamming_1_frac": one_bit / n if leftover else 0.0,
        "equal_score_despite_diff_k": score_tie,
        "min_hamming_hist": {str(k): v for k, v in sorted(max_delta.items())},
        "decision": "PROMOTE_NEAR_IDENT_K" if promote else "REJECT_WRITTEN",
        "reason": (
            "Leftover rows are dominated by 4/5 identical K; a near-ident sidecar "
            "would still be exact."
            if promote
            else (
                "Leftover Q!=0 rows are the residual XOR walk the sealed leaf "
                "already handles (small Hamming or unstructured). A third exact "
                "path would duplicate the leaf without a new contract."
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    stages = []
    for path in args.vector_dirs:
        if not (path / "input_q.memh").is_file():
            raise ValueError(f"missing input_q.memh in requested vector dir {path}")
        stages.append(analyze(path))
    if not stages:
        raise ValueError("no vector dirs with input_q.memh")
    leftover = sum(int(item["leftover_qnz_not_identk"]) for item in stages)
    almost = sum(int(item["almost_4of5_k_equal"]) for item in stages)
    decision = (
        "PROMOTE_NEAR_IDENT_K"
        if leftover and almost / leftover >= 0.40 and almost >= 200
        else "REJECT_WRITTEN"
    )
    report = {
        "schema": "local5_residual_leftover_decision_v1",
        "stages": stages,
        "leftover_total": leftover,
        "almost_4of5_total": almost,
        "decision": decision,
        "claim_boundary": [
            "Window-local statistic. Not a 21600-group claim.",
            "Reject keeps Query-Silent + identical-K as one cascade.",
        ],
    }
    if decision == "REJECT_WRITTEN":
        report["reason"] = (
            "Across complete windows the leftover residual is already the "
            "sealed leaf's XOR walk. No third exact sidecar."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    md = [
        "# Local5 leftover residual decision",
        "",
        f"- leftover Q!=0 non-ident-K dest: {leftover}",
        f"- 4/5 K equal: {almost}",
        f"**Decision: {decision}**",
        "",
        report.get("reason", stages[0]["reason"]),
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"PASS leftover residual {decision} leftover={leftover} almost4={almost}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
