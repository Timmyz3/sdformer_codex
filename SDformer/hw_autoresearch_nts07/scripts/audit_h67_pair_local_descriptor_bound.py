#!/usr/bin/env python3
"""Audit the pair-local RQTB descriptor lower bound on ordered H67 traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from scripts.profile_h67_zkqi_multisample_ordered import (
        BLOCK_RE,
        decode_trace,
        h67_score_from_counts,
    )
except ModuleNotFoundError:
    from profile_h67_zkqi_multisample_ordered import (
        BLOCK_RE,
        decode_trace,
        h67_score_from_counts,
    )


PAIRS_PER_ROW = 225


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values: list[int]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.int64)
    if data.size == 0:
        raise ValueError("empty distribution")
    return {
        "min": int(data.min()),
        "mean": float(data.mean()),
        "p50": float(np.quantile(data, 0.50)),
        "p95": float(np.quantile(data, 0.95)),
        "p99": float(np.quantile(data, 0.99)),
        "max": int(data.max()),
    }


def audit_profile(path: Path) -> dict:
    profile = json.loads(path.read_text())
    records = profile.get("summary", {}).get("h60_records", [])
    if len(records) != 1200:
        raise ValueError(f"expected 1200 H60 records in {path}, got {len(records)}")

    totals = defaultdict(int)
    by_stage = defaultdict(lambda: defaultdict(int))
    by_sample = defaultdict(lambda: defaultdict(int))
    row_descriptor_counts: list[int] = []
    row_equal_counts: list[int] = []
    record_receipts = []

    for record_index, record in enumerate(records):
        name = str(record.get("name", ""))
        match = BLOCK_RE.fullmatch(name)
        if not match:
            raise ValueError(f"bad block name at record {record_index}: {name}")
        stage = int(match.group("stage"))
        sample_id = int(record["sample_id"])
        q = decode_trace(record["pair_q_count_ordered_trace"])
        k = decode_trace(record["pair_k_count_ordered_trace"])
        overlap = decode_trace(record["pair_overlap_ordered_trace"])
        motion = decode_trace(record["pair_motion_ordered_trace"])
        score = h67_score_from_counts(q, k, overlap, motion)
        if score.ndim != 4 or score.shape[0] != 2 or score.shape[-1] != PAIRS_PER_ROW:
            raise ValueError(f"score shape mismatch at {name}: {score.shape}")

        equal = score[0] == score[1]
        rows = int(equal.shape[0] * equal.shape[1])
        pairs = rows * PAIRS_PER_ROW
        equal_per_row = np.count_nonzero(equal, axis=-1).reshape(-1)
        # One descriptor is necessary for an equal pair and two for an unequal
        # pair under the pair-local, one-Q7-class-per-descriptor contract.
        lower_bound = 2 * PAIRS_PER_ROW - equal_per_row
        actual = np.where(equal, 1, 2).sum(axis=-1).reshape(-1)
        membership = np.where(equal, 2, 2).sum(axis=-1).reshape(-1)
        if not np.array_equal(actual, lower_bound):
            raise ValueError(f"descriptor lower bound not attained at {name}")
        if not np.all(membership == 2 * PAIRS_PER_ROW):
            raise ValueError(f"temporal membership not conserved at {name}")

        record_equal = int(equal_per_row.sum())
        record_actual = int(actual.sum())
        if int(record.get("row_total", -1)) != rows:
            raise ValueError(f"row_total mismatch at {name}")
        if int(record.get("pair_total", -1)) != pairs:
            raise ValueError(f"pair_total mismatch at {name}")
        if int(record.get("pair_score_equal_h67", -1)) != record_equal:
            raise ValueError(f"stored equal count mismatch at {name}")

        row_descriptor_counts.extend(int(value) for value in actual)
        row_equal_counts.extend(int(value) for value in equal_per_row)
        for bucket in (totals, by_stage[stage], by_sample[sample_id]):
            bucket["records"] += 1
            bucket["rows"] += rows
            bucket["pairs"] += pairs
            bucket["equal_pairs"] += record_equal
            bucket["fixed_descriptors"] += 2 * pairs
            bucket["rqtb_descriptors"] += record_actual
            bucket["membership_tokens"] += 2 * pairs
            bucket["attained_rows"] += rows
        record_receipts.append(
            {
                "sample_id": sample_id,
                "name": name,
                "rows": rows,
                "pairs": pairs,
                "equal_pairs": record_equal,
                "rqtb_descriptors": record_actual,
            }
        )

    expected_samples = list(range(100))
    if sorted(by_sample) != expected_samples:
        raise ValueError("sample coverage is not exactly 0..99")
    if totals["rows"] != 672_000 or totals["pairs"] != 151_200_000:
        raise ValueError(f"unexpected all-window coverage: {dict(totals)}")
    if totals["rqtb_descriptors"] + totals["equal_pairs"] != 2 * totals["pairs"]:
        raise ValueError("global descriptor identity failed")

    def summarize(bucket: dict[str, int]) -> dict:
        result = dict(bucket)
        result["slot_reduction"] = (
            1.0 - bucket["rqtb_descriptors"] / bucket["fixed_descriptors"]
        )
        result["equal_pair_ratio"] = bucket["equal_pairs"] / bucket["pairs"]
        return result

    sample_reductions = [summarize(by_sample[s])["slot_reduction"] for s in expected_samples]
    receipt_payload = json.dumps(
        record_receipts, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    artifact = profile.get("artifact_identity", {})
    return {
        "profile": str(path),
        "profile_sha256": sha256(path),
        "checkpoint_path": artifact.get("checkpoint_path"),
        "checkpoint_sha256": artifact.get("checkpoint_sha256"),
        "totals": summarize(totals),
        "by_stage": {str(stage): summarize(by_stage[stage]) for stage in sorted(by_stage)},
        "sample_slot_reduction": {
            "min": min(sample_reductions),
            "mean": float(np.mean(sample_reductions)),
            "p50": float(np.quantile(sample_reductions, 0.50)),
            "p95": float(np.quantile(sample_reductions, 0.95)),
            "max": max(sample_reductions),
        },
        "row_rqtb_descriptor_distribution": distribution(row_descriptor_counts),
        "row_equal_pair_distribution": distribution(row_equal_counts),
        "record_receipts": {
            "count": len(record_receipts),
            "sha256": hashlib.sha256(receipt_payload).hexdigest(),
            "first": record_receipts[0],
            "last": record_receipts[-1],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="append", type=Path, required=True)
    parser.add_argument("--frozen-doc", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    audits = [audit_profile(path) for path in args.profile]
    report = {
        "schema": "h67_pair_local_descriptor_bound_v1",
        "status": "PASS_PAIR_LOCAL_BOUND_ATTAINED_PROFILE_SEMANTIC",
        "evidence": "[prof] ordered Q/K/count trace; not multisample RTL",
        "contract": {
            "scope": "pair-local descriptors",
            "descriptor": "one Q7 class plus temporal membership in one spatial pair",
            "lower_bound": "D_min = E + 2(P-E) = 2P-E",
            "attainment": "RQTB emits one descriptor for equal pairs and two for unequal pairs",
            "excluded": [
                "global cross-pair class bitmap descriptors",
                "accuracy or full-encoder performance",
                "independent RTL descriptor observation beyond the frozen 138-row anchor",
            ],
        },
        "audits": audits,
        "admission": {
            "all_profiles_attained": all(
                audit["totals"]["attained_rows"] == audit["totals"]["rows"]
                for audit in audits
            ),
            "date_main_table": False,
            "innovation_score_ceiling": 3.2,
            "frozen_doc_sha256": sha256(args.frozen_doc),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
