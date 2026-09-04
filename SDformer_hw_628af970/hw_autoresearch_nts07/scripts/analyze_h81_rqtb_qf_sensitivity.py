#!/usr/bin/env python3
"""Recompute H81 no-motion score equality for QF5-QF8 from ordered traces."""

from __future__ import annotations

import argparse
import base64
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import zlib

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_trace(encoded: dict[str, object]) -> np.ndarray:
    if encoded.get("codec") != "zlib_base64" or encoded.get("dtype") != "int16_le":
        raise ValueError("unsupported ordered-trace encoding")
    raw = zlib.decompress(base64.b64decode(str(encoded["data"])))
    array = np.frombuffer(raw, dtype="<i2")
    return array.reshape(tuple(int(value) for value in encoded["shape"]))


def rne_div_pow2(values: np.ndarray, denominator: int) -> np.ndarray:
    if denominator <= 0 or denominator & (denominator - 1):
        raise ValueError("denominator must be a positive power of two")
    quotient = values // denominator
    remainder = values % denominator
    half = denominator // 2
    increment = (remainder > half) | ((remainder == half) & ((quotient & 1) != 0))
    return quotient + increment.astype(quotient.dtype)


def summarize(counts: dict[str, int]) -> dict[str, object]:
    pairs = counts["pairs"]
    empty = counts["empty"]
    equal = counts["equal"]
    nonempty = pairs - empty
    nonempty_equal = equal - empty
    if not 0 <= empty <= equal <= pairs:
        raise ValueError("all-four-vector-empty is not a subset of score equal")
    return {
        "pairs": pairs,
        "empty_pairs": empty,
        "equal_pairs": equal,
        "equal_ratio": equal / pairs if pairs else 0.0,
        "nonempty_pairs": nonempty,
        "nonempty_equal_pairs": nonempty_equal,
        "nonempty_equal_ratio": nonempty_equal / nonempty if nonempty else 0.0,
        "modeled_slot_reduction_all_pairs": 0.5 * equal / pairs if pairs else 0.0,
        "modeled_slot_reduction_nonempty_pairs": (
            0.5 * nonempty_equal / nonempty if nonempty else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path(
            "results/h81_rqtb_g0_20260816/profile10/"
            "nts11_hardware_p0_profile.json"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/h81_rqtb_qf_sensitivity_20260816"),
    )
    args = parser.parse_args()

    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    records = profile["summary"]["h60_records"]
    if profile.get("ordered_trace") is not True or len(records) != 120:
        raise ValueError("expected sealed H81 ordered 10-sample / 120-block profile")

    totals = {bits: defaultdict(int) for bits in (5, 6, 7, 8)}
    stages = {
        stage: {bits: defaultdict(int) for bits in (5, 6, 7, 8)}
        for stage in range(4)
    }
    qf7_record_mismatches = 0
    for record in records:
        q_count = decode_trace(record["pair_q_count_ordered_trace"]).astype(np.int32)
        k_count = decode_trace(record["pair_k_count_ordered_trace"]).astype(np.int32)
        overlap = decode_trace(record["pair_overlap_ordered_trace"]).astype(np.int32)
        union = decode_trace(record["pair_four_vector_union_ordered_trace"])
        if q_count.shape != k_count.shape or q_count.shape != overlap.shape:
            raise ValueError("Q/K/overlap trace shapes differ")
        if q_count.shape[0] != 2:
            raise ValueError("trace does not contain a temporal pair")
        same_zero = 32 - q_count - k_count + overlap
        numerator = 64 * overlap + same_zero
        empty_mask = union == 0
        if empty_mask.shape != q_count.shape[1:]:
            raise ValueError("empty-mask shape differs from temporal-pair shape")
        stage = int(record["stage"])
        pairs = int(empty_mask.size)
        empty = int(np.count_nonzero(empty_mask))
        for bits in (5, 6, 7, 8):
            score = rne_div_pow2(numerator, 1 << (11 - bits))
            equal = int(np.count_nonzero(score[0] == score[1]))
            for bucket in (totals[bits], stages[stage][bits]):
                bucket["pairs"] += pairs
                bucket["empty"] += empty
                bucket["equal"] += equal
            if bits == 7 and equal != int(record["pair_score_equal_ttx"]):
                qf7_record_mismatches += 1

    if qf7_record_mismatches:
        raise ValueError(f"QF7 record miter mismatches={qf7_record_mismatches}")

    report = {
        "schema": "h81_rqtb_qf_sensitivity_v1",
        "status": "PASS_READ_ONLY_QF_SENSITIVITY",
        "evidence": "[prof] ordered no-motion H81 score-count replay",
        "claim_boundary": (
            "QF5-QF8 score-equality and modeled slot statistics only. No model "
            "accuracy, RTL, cycles, energy, selector, MVSEC, encoder, or PPA claim."
        ),
        "profile": str(args.profile.resolve()),
        "profile_sha256": sha256(args.profile),
        "records": len(records),
        "checks": {
            "ordered_trace": True,
            "qf7_record_mismatches": qf7_record_mismatches,
            "all_four_vector_empty_subset": True,
        },
        "overall": {str(bits): summarize(totals[bits]) for bits in (5, 6, 7, 8)},
        "per_stage": {
            str(stage): {
                str(bits): summarize(stages[stage][bits]) for bits in (5, 6, 7, 8)
            }
            for stage in range(4)
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "report.json"
    md_path = args.out_dir / "report.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    rows = []
    for bits in (5, 6, 7, 8):
        row = report["overall"][str(bits)]
        rows.append(
            f"| QF{bits} | {row['equal_ratio']:.6%} | "
            f"{row['nonempty_equal_ratio']:.6%} | "
            f"{row['modeled_slot_reduction_all_pairs']:.6%} |"
        )
    md_path.write_text(
        "# H81 RQTB QF5-QF8 read-only sensitivity\n\n"
        "Status: `PASS_READ_ONLY_QF_SENSITIVITY`.\n\n"
        "| Score precision | all-pair equal | nonempty equal | modeled slot reduction |\n"
        "|---|---:|---:|---:|\n"
        + "\n".join(rows)
        + "\n\n"
        + report["claim_boundary"]
        + "\n",
        encoding="utf-8",
    )
    print(md_path.resolve())


if __name__ == "__main__":
    main()
