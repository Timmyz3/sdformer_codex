#!/usr/bin/env python3
"""Independent receipt-blind audit of the frozen M501 result.

This checker intentionally does not import the production M501 analyzer.  It
reaggregates the saved detailed rows and, with --raw-replay, decodes every
frozen payload to recompute only the predeclared horizontal-G2 point.
"""

import argparse
import hashlib
import json
import math
import re
import statistics
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "results/m501_h67_exact_adjacent_overlap_fastkill_r1_20260827"
RESULT_PATH = RESULT_DIR / "m501_h67_exact_adjacent_overlap_fastkill_result_r1.json"
M40_PATH = ROOT / "results/m40_h67_ep35_bottleneck_packed_sources_s10_r6_20260822/m40_bottleneck_packed_source_manifest.json"
M73_PATH = ROOT / "system_handoff/incoming/m73_h67_ep35_train_calibration_sources_s32_r1_20260823/m73_train_calibration_source_manifest.json"
DOCS359_PATH = ROOT / "docs/359_DATE终局冻结_20260813.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def sequence_key(sample_key: str) -> str:
    match = re.match(r"^(.*)_([0-9]+)$", Path(sample_key).stem)
    assert match is not None, sample_key
    return match.group(1)


def verify_seals() -> dict:
    verified = {}
    for raw_line in (RESULT_DIR / "SHA256SUMS").read_text().splitlines():
        expected, relative = raw_line.split(None, 1)
        relative = relative.strip()
        actual = sha256_file(ROOT / relative)
        assert actual == expected, (relative, expected, actual)
        verified[relative] = actual
    seal_line = (RESULT_DIR / "SHA256SUMS.seal.sha256").read_text().strip()
    expected_seal = seal_line.split()[0]
    actual_seal = sha256_file(RESULT_DIR / "SHA256SUMS")
    assert actual_seal == expected_seal
    assert sha256_file(DOCS359_PATH) == "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
    return {
        "manifest_entries": len(verified),
        "manifest_sha256": actual_seal,
        "docs359_sha256": sha256_file(DOCS359_PATH),
    }


def aggregate(rows: list) -> dict:
    baseline = sum(int(row["baseline_events"]) for row in rows)
    overlap = sum(int(row["exact_overlap_events"]) for row in rows)
    redundant = sum(int(row["redundant_events"]) for row in rows)
    candidate = sum(int(row["candidate_events"]) for row in rows)
    ratio = baseline / candidate
    fraction = redundant / baseline
    return {
        "records": len(rows),
        "baseline_events": baseline,
        "exact_overlap_events": overlap,
        "redundant_events": redundant,
        "candidate_events": candidate,
        "event_reduction_ratio": ratio,
        "redundant_fraction": fraction,
    }


def verify_saved_rows(result: dict) -> dict:
    selected = None
    train_ratios = []
    row_count = 0
    for cohort in result["cohorts"]:
        detailed = cohort["detailed"]
        row_count += len(detailed)
        for row in detailed:
            assert int(row["candidate_events"]) + int(row["redundant_events"]) == int(row["baseline_events"])
            assert int(row["redundant_events"]) == (int(row["group_size"]) - 1) * int(row["exact_overlap_events"])
            assert close(row["event_reduction_ratio"], int(row["baseline_events"]) / int(row["candidate_events"]))
            assert close(row["redundant_fraction"], int(row["redundant_events"]) / int(row["baseline_events"]))
            assert int(row["overlap_scratch_bits"]) == 768 * 3 * 3 * 19
            assert int(row["overlap_scratch_bytes"]) == 16416

        for fields, aggregate_name in (
            (("axis", "group_size"), "overall"),
            (("operator", "axis", "group_size"), "per_operator"),
            (("sequence", "axis", "group_size"), "per_sequence"),
        ):
            buckets = defaultdict(list)
            for row in detailed:
                buckets[tuple(row[field] for field in fields)].append(row)
            saved_rows = {
                tuple(row[field] for field in fields): row
                for row in cohort["aggregate"][aggregate_name]
            }
            assert set(saved_rows) == set(buckets)
            for key, bucket in buckets.items():
                expected = aggregate(bucket)
                saved = saved_rows[key]
                for name in ("records", "baseline_events", "exact_overlap_events", "redundant_events", "candidate_events"):
                    assert int(saved[name]) == int(expected[name]), (cohort["cohort"], aggregate_name, key, name)
                for name in ("event_reduction_ratio", "redundant_fraction"):
                    assert close(saved[name], expected[name]), (cohort["cohort"], aggregate_name, key, name)

        hg2 = [
            row for row in cohort["aggregate"]["overall"]
            if row["axis"] == "horizontal" and int(row["group_size"]) == 2
        ]
        assert len(hg2) == 1
        if cohort["cohort"] == "validation_s10":
            selected = hg2[0]
        elif cohort["cohort"] == "train_calibration_s32":
            train_ratios = sorted(
                float(row["event_reduction_ratio"])
                for row in cohort["aggregate"]["per_sequence"]
                if row["axis"] == "horizontal" and int(row["group_size"]) == 2
            )

    assert selected is not None
    assert len(train_ratios) == 18
    event_ratio = int(selected["baseline_events"]) / int(selected["candidate_events"])
    conv_share = 79630957 / 620302905
    ideal = 1 / (1 - conv_share + conv_share / event_ratio)
    decision = result["decision"]
    assert close(decision["event_reduction_ratio"], event_ratio)
    assert close(decision["four_bottleneck_conv_share"], conv_share)
    assert close(decision["ideal_envelope_sensitivity"], ideal)
    saved_distribution = decision["train_calibration_horizontal_g2_sequence_distribution"]
    assert int(saved_distribution["sequences"]) == 18
    assert close(saved_distribution["minimum"], train_ratios[0])
    assert close(saved_distribution["median"], statistics.median(train_ratios))
    assert close(saved_distribution["maximum"], train_ratios[-1])
    assert saved_distribution["heldout"] is False

    return {
        "detailed_rows": row_count,
        "validation_horizontal_g2": {
            "baseline_events": int(selected["baseline_events"]),
            "overlap_events": int(selected["exact_overlap_events"]),
            "candidate_events": int(selected["candidate_events"]),
            "event_reduction_ratio": event_ratio,
        },
        "four_conv_share": conv_share,
        "ideal_envelope_sensitivity": ideal,
        "train_horizontal_g2_sequence_distribution": {
            "sequences": 18,
            "minimum": train_ratios[0],
            "median": statistics.median(train_ratios),
            "maximum": train_ratios[-1],
        },
    }


def replay_manifest(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    totals = defaultdict(lambda: [0, 0, 0])
    operator_codewords = defaultdict(set)
    records = manifest["records"]
    for record in records:
        payload_path = manifest_path.parent / record["value_payload_file"]
        assert sha256_file(payload_path) == record["value_payload_sha256"]
        compressed = payload_path.read_bytes()
        assert len(compressed) == int(record["value_payload_compressed_bytes"])
        raw = zlib.decompress(compressed)
        assert len(raw) == int(record["input_content_bytes"])
        assert hashlib.sha256(raw).hexdigest() == record["input_content_sha256"]
        shape = tuple(int(value) for value in record["shape"])
        assert shape == (10, 1, 768, 15, 20)
        values = np.frombuffer(raw, dtype="<f4").reshape(shape)
        bits = values.view("<u4")
        left = values[..., 0::2]
        right = values[..., 1::2]
        overlap = int(np.count_nonzero((left != 0.0) & (right != 0.0) & (left.view("<u4") == right.view("<u4"))))
        baseline = int(np.count_nonzero(values))
        assert baseline == int(record["nonzero_count"])
        candidate = baseline - overlap
        sequence = sequence_key(record["sample_key"])
        for key in ("__overall__", sequence):
            totals[key][0] += baseline
            totals[key][1] += overlap
            totals[key][2] += candidate

        codebook = record["value_bit_pattern_population"]
        assert int(codebook["unique_float32_bit_patterns"]) == 2
        assert codebook["codebook"][0]["float32_bits_hex"] == "00000000"
        nonzero_bits = codebook["codebook"][1]["float32_bits_hex"]
        assert int(codebook["codebook"][1]["count"]) == baseline
        assert int(record["negative_count"]) == 0
        operator_codewords[record["operator"]].add(nonzero_bits)

    assert all(len(codewords) == 1 for codewords in operator_codewords.values())
    ratios = {
        key: values[0] / values[2]
        for key, values in totals.items()
        if key != "__overall__"
    }
    overall = totals["__overall__"]
    return {
        "records": len(records),
        "sequences": len(ratios),
        "baseline_events": overall[0],
        "overlap_events": overall[1],
        "candidate_events": overall[2],
        "event_reduction_ratio": overall[0] / overall[2],
        "operator_codewords": {
            operator: next(iter(codewords))
            for operator, codewords in sorted(operator_codewords.items())
        },
        "sequence_ratios": ratios,
    }


def verify_raw(result: dict) -> dict:
    validation = replay_manifest(M40_PATH)
    train = replay_manifest(M73_PATH)
    assert validation["records"] == 40 and validation["sequences"] == 1
    assert train["records"] == 128 and train["sequences"] == 18
    assert validation["operator_codewords"] == train["operator_codewords"]
    assert close(validation["event_reduction_ratio"], result["decision"]["event_reduction_ratio"])
    saved_train = result["decision"]["train_calibration_horizontal_g2_sequence_distribution"]
    train_ratios = sorted(train["sequence_ratios"].values())
    assert close(train_ratios[0], saved_train["minimum"])
    assert close(statistics.median(train_ratios), saved_train["median"])
    assert close(train_ratios[-1], saved_train["maximum"])
    return {
        "validation": validation,
        "train": {
            **{key: value for key, value in train.items() if key != "sequence_ratios"},
            "sequence_distribution": {
                "minimum": train_ratios[0],
                "median": statistics.median(train_ratios),
                "maximum": train_ratios[-1],
            },
        },
        "all_168_records_positive_two_codeword": True,
        "exact_overlap_equals_support_intersection_on_frozen_trace": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-replay", action="store_true")
    args = parser.parse_args()
    result = json.loads(RESULT_PATH.read_text())
    assert result["schema"] == "m501_h67_exact_adjacent_overlap_fastkill_result_v1"
    assert result["status"] == "PASS_EXACT_OPPORTUNITY_AUDIT_NO_RTL_ADMISSION"
    assert result["decision"]["opportunity_gate_pass"] is True
    assert result["decision"]["next_action"] == "ALLOW_SAME_RESOURCE_CYCLE_FASTKILL_ONLY"
    assert result["decision"]["new_rtl_admitted"] is False
    for forbidden in ("same_resource_cycles", "rtl", "synopsys", "energy", "ppa", "full_network", "system_speedup", "date_headline"):
        assert result["claim_boundary"][forbidden] is False
    scratch = result["decision"]["selected_overlap_scratch"]
    assert int(scratch["bits"]) == 768 * 3 * 3 * 19 == 131328
    assert int(scratch["bytes"]) == 16416
    assert close(scratch["kibibytes"], 16.03125)
    assert scratch["costs_unpriced"] is True

    report = {
        "schema": "m501_result_independent_audit_v1",
        "seals": verify_seals(),
        "saved_row_reaggregation": verify_saved_rows(result),
        "raw_horizontal_g2_replay": verify_raw(result) if args.raw_replay else "NOT_RUN",
        "verdict": "PASS",
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
