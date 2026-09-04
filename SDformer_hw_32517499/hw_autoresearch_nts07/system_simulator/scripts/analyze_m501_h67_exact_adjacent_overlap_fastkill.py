#!/usr/bin/env python3
"""M501: exact adjacent-position overlap opportunity on frozen H67 Conv traces.

This is a fail-closed opportunity audit inspired by ExSpike APEC.  It is not a
cycle model and cannot admit RTL, PPA, energy, or system-speedup claims.
"""

import argparse
import hashlib
import json
import math
import re
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_key(sample_key: str) -> str:
    stem = Path(sample_key).stem
    match = re.match(r"^(.*)_([0-9]+)$", stem)
    if match is None:
        raise RuntimeError(f"sample key has no numeric suffix: {sample_key}")
    return match.group(1)


def safe_ratio(numerator, denominator):
    if denominator <= 0:
        return 1.0 if numerator == 0 else math.inf
    return float(numerator) / float(denominator)


def load_record(manifest_dir: Path, record: Dict[str, Any]) -> np.ndarray:
    payload_path = manifest_dir / record["value_payload_file"]
    if sha256_file(payload_path) != record["value_payload_sha256"]:
        raise RuntimeError(f"compressed payload SHA mismatch: {payload_path}")
    if payload_path.stat().st_size != record["value_payload_compressed_bytes"]:
        raise RuntimeError(f"compressed payload byte mismatch: {payload_path}")
    raw = zlib.decompress(payload_path.read_bytes())
    if len(raw) != record["input_content_bytes"]:
        raise RuntimeError(f"decoded byte mismatch: {payload_path}")
    if hashlib.sha256(raw).hexdigest() != record["input_content_sha256"]:
        raise RuntimeError(f"decoded payload SHA mismatch: {payload_path}")
    if record["input_dtype"] != "float32":
        raise RuntimeError(f"unsupported dtype: {record['input_dtype']}")
    shape = tuple(int(value) for value in record["shape"])
    if len(shape) != 5:
        raise RuntimeError(f"expected T,B,C,H,W shape, got {shape}")
    array = np.frombuffer(raw, dtype="<f4")
    if array.size != math.prod(shape):
        raise RuntimeError(f"decoded element mismatch: {payload_path}")
    if int(record["elements"]) != math.prod(shape):
        raise RuntimeError(f"manifest element mismatch: {payload_path}")
    if int(record["input_content_bytes"]) != int(record["elements"]) * 4:
        raise RuntimeError(f"manifest float32 byte mismatch: {payload_path}")
    array = array.reshape(shape)
    if not bool(np.isfinite(array).all()):
        raise RuntimeError(f"NaN/Inf in decoded payload: {payload_path}")
    if int(np.count_nonzero(array)) != int(record["nonzero_count"]):
        raise RuntimeError(f"nonzero-count mismatch: {payload_path}")
    if int(np.count_nonzero(array < np.float32(0.0))) != int(record["negative_count"]):
        raise RuntimeError(f"negative-count mismatch: {payload_path}")
    return array


def count_exact_overlap(
    values: np.ndarray,
    *,
    axis: str,
    group_size: int,
) -> Dict[str, Any]:
    if group_size not in (2, 4, 8):
        raise RuntimeError(f"unsupported group size: {group_size}")
    if axis not in ("horizontal", "vertical"):
        raise RuntimeError(f"unsupported axis: {axis}")

    baseline_events = int(np.count_nonzero(values))
    bit_values = values.view("<u4")
    active = values != np.float32(0.0)
    if axis == "horizontal":
        spatial_extent = values.shape[4]
        full_extent = (spatial_extent // group_size) * group_size
        grouped_bits = bit_values[..., :full_extent].reshape(
            *values.shape[:4], full_extent // group_size, group_size
        )
        grouped_active = active[..., :full_extent].reshape(
            *values.shape[:4], full_extent // group_size, group_size
        )
    else:
        spatial_extent = values.shape[3]
        full_extent = (spatial_extent // group_size) * group_size
        moved_bits = np.moveaxis(bit_values[:, :, :, :full_extent, :], 3, -1)
        moved_active = np.moveaxis(active[:, :, :, :full_extent, :], 3, -1)
        grouped_bits = moved_bits.reshape(
            *moved_bits.shape[:-1], full_extent // group_size, group_size
        )
        grouped_active = moved_active.reshape(
            *moved_active.shape[:-1], full_extent // group_size, group_size
        )

    all_active = np.all(grouped_active, axis=-1)
    exact_equal = np.all(grouped_bits == grouped_bits[..., :1], axis=-1)
    exact_overlap_events = int(np.count_nonzero(all_active & exact_equal))
    redundant_events = (group_size - 1) * exact_overlap_events
    candidate_events = baseline_events - redundant_events
    if candidate_events < 0:
        raise RuntimeError("negative candidate event count")
    if candidate_events + redundant_events != baseline_events:
        raise RuntimeError("event conservation failed")
    if redundant_events != (group_size - 1) * exact_overlap_events:
        raise RuntimeError("overlap redundancy identity failed")
    grouped_baseline = int(np.count_nonzero(grouped_active))
    if exact_overlap_events > grouped_baseline // group_size:
        raise RuntimeError("overlap exceeds grouped-event upper bound")
    return {
        "baseline_events": baseline_events,
        "exact_overlap_events": exact_overlap_events,
        "redundant_events": redundant_events,
        "candidate_events": candidate_events,
        "event_reduction_ratio": safe_ratio(baseline_events, candidate_events),
        "redundant_fraction": safe_ratio(redundant_events, baseline_events),
        "full_grouped_spatial_extent": full_extent,
        "ungrouped_spatial_extent": spatial_extent - full_extent,
    }


def aggregate_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    baseline = sum(int(row["baseline_events"]) for row in rows)
    overlap = sum(int(row["exact_overlap_events"]) for row in rows)
    redundant = sum(int(row["redundant_events"]) for row in rows)
    candidate = sum(int(row["candidate_events"]) for row in rows)
    return {
        "records": len(rows),
        "baseline_events": baseline,
        "exact_overlap_events": overlap,
        "redundant_events": redundant,
        "candidate_events": candidate,
        "event_reduction_ratio": safe_ratio(baseline, candidate),
        "redundant_fraction": safe_ratio(redundant, baseline),
    }


def analyze_manifest(
    *,
    manifest_path: Path,
    cohort: str,
    group_sizes: List[int],
    axes: List[str],
    accumulator_bits: int,
    expectation: Dict[str, Any],
) -> Dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != expectation["schema"]:
        raise RuntimeError(f"manifest schema mismatch: {manifest_path}")
    if manifest.get("status") != expectation["status"]:
        raise RuntimeError(f"manifest status mismatch: {manifest_path}")
    records = manifest.get("records", [])
    if len(records) != int(expectation["records"]):
        raise RuntimeError(f"manifest record-count mismatch: {manifest_path}")
    sample_ids = {int(record["sample_id"]) for record in records}
    if sample_ids != set(range(int(expectation["samples"]))):
        raise RuntimeError(f"manifest sample IDs mismatch: {manifest_path}")
    sequences = {sequence_key(record["sample_key"]) for record in records}
    if len(sequences) != int(expectation["sequences"]):
        raise RuntimeError(f"manifest sequence-count mismatch: {manifest_path}")
    expected_operators = set(expectation["operators"])
    observed_operators = {record["operator"] for record in records}
    if observed_operators != expected_operators:
        raise RuntimeError(f"manifest operator set mismatch: {manifest_path}")
    sample_operator = [
        (int(record["sample_id"]), record["operator"]) for record in records
    ]
    expected_pairs = {
        (sample_id, operator)
        for sample_id in sample_ids
        for operator in expected_operators
    }
    if len(sample_operator) != len(set(sample_operator)) or set(sample_operator) != expected_pairs:
        raise RuntimeError(f"manifest sample/operator Cartesian product mismatch: {manifest_path}")
    manifest_dir = manifest_path.parent
    detailed = []  # type: List[Dict[str, Any]]
    positive_two_codeword_records = 0
    for record in records:
        if record.get("value_payload_codec") != "ZLIB_LEVEL9_RAW_C_ORDER_FLOAT32_NATIVE_LE":
            raise RuntimeError(
                f"record lacks reconstructable exact float payload: {record.get('sample_key')}"
            )
        values = load_record(manifest_dir, record)
        if list(record["shape"]) != list(expectation["shape"]):
            raise RuntimeError(f"input shape mismatch: {record['sample_key']}")
        if list(record["output_shape"]) != list(expectation["shape"]):
            raise RuntimeError(f"output shape mismatch: {record['sample_key']}")
        output_channels = int(record["module_geometry"]["out_channels"])
        kernel = tuple(int(value) for value in record["module_geometry"]["kernel_size"])
        geometry = record["module_geometry"]
        for field in ("in_channels", "out_channels", "groups"):
            if int(geometry[field]) != int(expectation["geometry"][field]):
                raise RuntimeError(f"geometry {field} mismatch: {record['sample_key']}")
        for field in ("kernel_size", "stride", "padding", "dilation"):
            if list(geometry[field]) != list(expectation["geometry"][field]):
                raise RuntimeError(f"geometry {field} mismatch: {record['sample_key']}")
        if kernel != (3, 3) or output_channels != 768:
            raise RuntimeError(f"M501 only admits frozen 768-channel 3x3 Conv")
        codebook = record.get("value_bit_pattern_population", {})
        if int(codebook.get("unique_float32_bit_patterns", -1)) != 2:
            raise RuntimeError(f"expected exact two-codeword trace: {record['sample_key']}")
        if not bool(codebook.get("full_codebook_in_manifest", False)):
            raise RuntimeError(f"incomplete codebook: {record['sample_key']}")
        entries = codebook.get("codebook", [])
        if len(entries) != 2 or entries[0].get("float32_bits_hex") != "00000000":
            raise RuntimeError(f"unexpected two-codeword layout: {record['sample_key']}")
        if int(entries[0]["count"]) + int(entries[1]["count"]) != int(record["elements"]):
            raise RuntimeError(f"codebook count mismatch: {record['sample_key']}")
        nonzero_bits = int(entries[1]["float32_bits_hex"], 16)
        nonzero_value = np.array([nonzero_bits], dtype="<u4").view("<f4")[0]
        if not bool(np.isfinite(nonzero_value)) or not float(nonzero_value) > 0.0:
            raise RuntimeError(f"nonzero codeword is not finite-positive: {record['sample_key']}")
        if int(record["negative_count"]) != 0:
            raise RuntimeError(f"M501 frozen cohort is not positive-only: {record['sample_key']}")
        if int(np.count_nonzero(values.view("<u4") == nonzero_bits)) != int(record["nonzero_count"]):
            raise RuntimeError(f"decoded nonzero codeword mismatch: {record['sample_key']}")
        positive_two_codeword_records += 1
        scratch_bits = output_channels * math.prod(kernel) * accumulator_bits
        for axis in axes:
            for group_size in group_sizes:
                metrics = count_exact_overlap(
                    values, axis=axis, group_size=group_size
                )
                detailed.append(
                    {
                        "cohort": cohort,
                        "sample_id": int(record["sample_id"]),
                        "sample_key": record["sample_key"],
                        "sequence": sequence_key(record["sample_key"]),
                        "operator": record["operator"],
                        "axis": axis,
                        "group_size": group_size,
                        "output_channels": output_channels,
                        "kernel": list(kernel),
                        "overlap_scratch_bits": scratch_bits,
                        "overlap_scratch_bytes": (scratch_bits + 7) // 8,
                        **metrics,
                    }
                )

    aggregate = {}  # type: Dict[str, Any]
    for fields, name in (
        (("axis", "group_size"), "overall"),
        (("operator", "axis", "group_size"), "per_operator"),
        (("sequence", "axis", "group_size"), "per_sequence"),
    ):
        buckets = defaultdict(list)  # type: Dict[Any, List[Dict[str, Any]]]
        for row in detailed:
            buckets[tuple(row[field] for field in fields)].append(row)
        aggregate[name] = [
            {
                **{field: key[index] for index, field in enumerate(fields)},
                **aggregate_rows(bucket_rows),
            }
            for key, bucket_rows in sorted(buckets.items())
        ]

    return {
        "cohort": cohort,
        "manifest": {
            "path": str(manifest_path.relative_to(ROOT)),
            "sha256": sha256_file(manifest_path),
            "schema": manifest.get("schema"),
            "status": manifest.get("status"),
        },
        "samples": len({row["sample_key"] for row in detailed}),
        "sequences": len({row["sequence"] for row in detailed}),
        "operators": len({row["operator"] for row in detailed}),
        "records": len(records),
        "trace_codebook_facts": {
            "positive_two_codeword_records": positive_two_codeword_records,
            "all_records_zero_plus_one_operator_constant_positive_amplitude": (
                positive_two_codeword_records == len(records)
            ),
            "all_records_negative_count_zero": True,
            "exact_value_overlap_equals_support_intersection_on_this_trace": True,
            "general_signed_analog_novelty_activated": False,
        },
        "detailed": detailed,
        "aggregate": aggregate,
    }


def select_point(cohort: Dict[str, Any], *, axis: str, group_size: int) -> Dict[str, Any]:
    matches = [
        row
        for row in cohort["aggregate"]["overall"]
        if row["axis"] == axis and int(row["group_size"]) == group_size
    ]
    if len(matches) != 1:
        raise RuntimeError(f"missing unique point {axis}/G{group_size}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    contract_path = args.contract.resolve()
    contract = json.loads(contract_path.read_text())
    if contract.get("schema") != "m501_h67_exact_adjacent_overlap_fastkill_contract_v1":
        raise RuntimeError("wrong M501 contract schema")
    docs359 = ROOT / contract["frozen_inputs"]["docs359"]["path"]
    if sha256_file(docs359) != contract["frozen_inputs"]["docs359"]["sha256"]:
        raise RuntimeError("docs/359 SHA mismatch")

    cohorts = []
    for cohort, key in (("validation_s10", "m40_r6"), ("train_calibration_s32", "m73")):
        item = contract["frozen_inputs"][key]
        path = ROOT / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"{key} manifest SHA mismatch")
        cohorts.append(
            analyze_manifest(
                manifest_path=path,
                cohort=cohort,
                group_sizes=[int(value) for value in contract["dse"]["group_sizes"]],
                axes=list(contract["dse"]["axes"]),
                accumulator_bits=int(contract["cost_model"]["overlap_accumulator_bits"]),
                expectation=contract["cohort_expectations"][key],
            )
        )

    validation = next(row for row in cohorts if row["cohort"] == "validation_s10")
    validation_g2 = select_point(validation, axis="horizontal", group_size=2)
    envelope = contract["frozen_envelope"]
    conv_share = float(envelope["four_bottleneck_conv_cycles"]) / float(
        envelope["total_cycles"]
    )
    opportunity_speedup = float(validation_g2["event_reduction_ratio"])
    ideal_envelope_sensitivity = 1.0 / (
        1.0 - conv_share + conv_share / opportunity_speedup
    )
    event_gate = opportunity_speedup >= float(
        contract["decision_gates"]["minimum_validation_event_reduction_ratio"]
    )
    sensitivity_gate = ideal_envelope_sensitivity >= float(
        contract["decision_gates"]["minimum_ideal_envelope_sensitivity"]
    )
    opportunity_gate = event_gate and sensitivity_gate
    selected_scratch_bits = 768 * 3 * 3 * int(
        contract["cost_model"]["overlap_accumulator_bits"]
    )
    train_calibration = next(
        row for row in cohorts if row["cohort"] == "train_calibration_s32"
    )
    train_g2_sequence_rows = [
        row
        for row in train_calibration["aggregate"]["per_sequence"]
        if row["axis"] == "horizontal" and int(row["group_size"]) == 2
    ]
    train_g2_ratios = sorted(
        float(row["event_reduction_ratio"]) for row in train_g2_sequence_rows
    )
    result = {
        "schema": "m501_h67_exact_adjacent_overlap_fastkill_result_v1",
        "status": "PASS_EXACT_OPPORTUNITY_AUDIT_NO_RTL_ADMISSION",
        "contract": {
            "path": str(contract_path.relative_to(ROOT)),
            "sha256": sha256_file(contract_path),
        },
        "cohorts": cohorts,
        "decision": {
            "selected_point": "validation_s10 horizontal G2",
            "event_reduction_ratio": opportunity_speedup,
            "four_bottleneck_conv_share": conv_share,
            "ideal_envelope_sensitivity": ideal_envelope_sensitivity,
            "event_gate_pass": event_gate,
            "sensitivity_gate_pass": sensitivity_gate,
            "opportunity_gate_pass": opportunity_gate,
            "next_action": (
                "ALLOW_SAME_RESOURCE_CYCLE_FASTKILL_ONLY"
                if opportunity_gate
                else "KILL_ADJACENT_OVERLAP_LINE"
            ),
            "selected_overlap_scratch": {
                "accumulator_width_bits": int(
                    contract["cost_model"]["overlap_accumulator_bits"]
                ),
                "width_source": contract["cost_model"]["selected_width_source"],
                "bits": selected_scratch_bits,
                "bytes": (selected_scratch_bits + 7) // 8,
                "kibibytes": ((selected_scratch_bits + 7) // 8) / 1024.0,
                "costs_unpriced": True,
            },
            "train_calibration_horizontal_g2_sequence_distribution": {
                "sequences": len(train_g2_ratios),
                "minimum": train_g2_ratios[0],
                "median": float(np.median(np.asarray(train_g2_ratios))),
                "maximum": train_g2_ratios[-1],
                "heldout": False,
            },
            "new_rtl_admitted": False,
            "reason": (
                "Opportunity-only exact overlap counts do not price grouping, overlap-psum "
                "storage/ports, weight latency, buffer cycles, or shared destination commit. "
                "ExSpike APEC is direct prior art; a new RTL remains closed until an H67-native "
                "same-resource cycle model and novelty delta both pass independent review."
            ),
        },
        "claim_boundary": contract["claim_boundary"],
        "external_reference": contract["external_reference"],
    }

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    result_path = output_dir / "m501_h67_exact_adjacent_overlap_fastkill_result_r1.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    lines = [
        "# M501 H67 exact adjacent-overlap fast-kill",
        "",
        f"Status: `{result['status']}`",
        "",
        "| Cohort | Axis | Group | Event reduction | Redundant fraction |",
        "|---|---|---:|---:|---:|",
    ]
    for cohort in cohorts:
        for row in cohort["aggregate"]["overall"]:
            lines.append(
                f"| {cohort['cohort']} | {row['axis']} | {row['group_size']} | "
                f"{row['event_reduction_ratio']:.6f}x | {row['redundant_fraction']:.6%} |"
            )
    lines += [
        "",
        f"Selected validation horizontal G2 event reduction: `{opportunity_speedup:.6f}x`.",
        f"Ideal four-Conv envelope sensitivity only: `{ideal_envelope_sensitivity:.6f}x`.",
        f"Opportunity gate: `{opportunity_gate}`; next action: `{result['decision']['next_action']}`.",
        f"Selected overlap scratch proxy: `{selected_scratch_bits} bit` "
        f"(`{result['decision']['selected_overlap_scratch']['kibibytes']:.5f} KiB`, "
        f"{contract['cost_model']['selected_width_source']}).",
        "All frozen records contain only zero plus one operator-constant positive amplitude;",
        "therefore exact-value overlap equals support intersection here and does not activate",
        "a general signed-analog novelty delta.",
        "",
        "`new_rtl_admitted=false`: this is exact event-work opportunity, not a same-resource cycle,",
        "energy, PPA, full-network, or system-speedup result. ExSpike APEC is direct prior art.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")
    (output_dir / "RUN_COMPLETE.txt").write_text(result["status"] + "\n")


if __name__ == "__main__":
    main()
