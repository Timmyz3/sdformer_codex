#!/usr/bin/env python3
"""Audit and summarize cross-simulator H67 multisample RQTB RTL logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


EXPECTED_BLOCKS = {0: 2, 1: 2, 2: 6, 3: 2}
EXPECTED_HEADS = {0: 3, 1: 6, 2: 12, 3: 24}
ROOT = Path(__file__).resolve().parents[1]
ROWS_PER_SAMPLE = sum(
    EXPECTED_BLOCKS[stage] * heads for stage, heads in EXPECTED_HEADS.items()
)
ROW_RE = re.compile(r"^RQTB_ROW (?P<body>.+)$")
PASS_RE = re.compile(r"^PASS H67 RQTB 2S physical flow (?P<body>.+)$")
PAIR_RE = re.compile(r"(?P<key>[A-Za-z0-9_]+)=(?P<value>-?[0-9]+)")
ROW_KEYS = {
    "row", "stage", "block", "head", "active", "equal",
    "fixed_cycles", "rqtb_cycles", "fixed_slots", "rqtb_slots",
    "fixed_desc", "rqtb_desc", "fixed_exp", "rqtb_exp",
    "fixed_pair_stall", "rqtb_pair_stall",
    "fixed_desc_stall", "rqtb_desc_stall",
    "fixed_out_stall", "rqtb_out_stall",
    "fixed_fifo_max", "rqtb_fifo_max",
}
PASS_KEYS = {
    "rows", "checked", "fixed_cycles", "rqtb_cycles", "fixed_slots",
    "rqtb_slots", "fixed_exp", "rqtb_exp", "acc32_mismatch",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_pairs(body: str, expected_keys: set[str]) -> dict[str, int]:
    matches = list(PAIR_RE.finditer(body))
    if not matches:
        raise ValueError("key=value payload is empty")
    values = {match.group("key"): int(match.group("value")) for match in matches}
    if len(values) != len(matches) or set(values) != expected_keys:
        raise ValueError(f"key=value schema mismatch: {set(values)}")
    residue = PAIR_RE.sub("", body).strip()
    if residue:
        raise ValueError(f"unparsed key=value residue: {residue}")
    return values


def parse_log(path: Path) -> tuple[list[dict[str, int]], dict[str, int]]:
    rows: list[dict[str, int]] = []
    passes: list[dict[str, int]] = []
    for line in path.read_text(encoding="utf-8", errors="strict").splitlines():
        if match := ROW_RE.fullmatch(line):
            rows.append(parse_pairs(match.group("body"), ROW_KEYS))
        elif match := PASS_RE.fullmatch(line):
            passes.append(parse_pairs(match.group("body"), PASS_KEYS))
    if len(passes) != 1:
        raise ValueError(f"log must contain exactly one final PASS: {path}")
    if len(rows) < 2 * ROWS_PER_SAMPLE:
        raise ValueError(f"log has fewer than {2 * ROWS_PER_SAMPLE} rows: {path}")
    final = passes[0]
    if final["acc32_mismatch"] != 0 or final["rows"] != len(rows):
        raise ValueError(f"log final receipt mismatch: {path}")
    return rows, final


def expected_row_sequence(sample_count: int) -> list[tuple[int, int, int, int]]:
    return [
        (sample_id, stage, block, head)
        for sample_id in range(sample_count)
        for stage, depth in EXPECTED_BLOCKS.items()
        for block in range(depth)
        for head in range(EXPECTED_HEADS[stage])
    ]


def load_row_index(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="ascii").splitlines()
        if line.strip()
    ]
    if len(rows) < 2 * ROWS_PER_SAMPLE or len(rows) % ROWS_PER_SAMPLE:
        raise ValueError("row index must contain complete all12 data for at least two samples")
    sample_count = len(rows) // ROWS_PER_SAMPLE
    expected = expected_row_sequence(sample_count)
    sample_keys: dict[int, str] = {}
    seen_keys: set[str] = set()
    required = {
        "row_tag", "sample_id", "sample_key", "stage", "block", "head",
        "record_order", "expected_outputs", "expected_folded",
    }
    for row_tag, (row, position) in enumerate(zip(rows, expected)):
        if set(row) != required:
            raise ValueError(f"row index schema mismatch at row {row_tag}")
        if row["row_tag"] != row_tag:
            raise ValueError(f"row index is not globally ordered at row {row_tag}")
        actual = (row["sample_id"], row["stage"], row["block"], row["head"])
        if actual != position:
            raise ValueError(f"all12/head order mismatch at row {row_tag}: {actual}")
        expected_record_order = sum(
            EXPECTED_BLOCKS[stage] for stage in range(row["stage"])
        ) + row["block"]
        if row["record_order"] != expected_record_order:
            raise ValueError(f"record order mismatch at row {row_tag}")
        sample_id = row["sample_id"]
        key = row["sample_key"]
        if not isinstance(key, str) or not key:
            raise ValueError(f"empty sample_key at row {row_tag}")
        if sample_id not in sample_keys:
            if key in seen_keys:
                raise ValueError(f"sample_key is reused by sample {sample_id}")
            sample_keys[sample_id] = key
            seen_keys.add(key)
        elif sample_keys[sample_id] != key:
            raise ValueError(f"sample_key changed within sample {sample_id}")
    return rows


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot summarize an empty distribution")
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def validate_vector_manifest(
    manifest_path: Path, row_index_path: Path, expected_rows: int
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    if (manifest.get("schema") != "h67_multisample_checkpoint_t450_vectors_v1"
            or manifest.get("status") != "PASS"
            or manifest.get("row_count") != expected_rows
            or manifest.get("sample_count") != expected_rows // ROWS_PER_SAMPLE
            or manifest.get("rows_per_sample") != ROWS_PER_SAMPLE
            or manifest.get("tokens_per_row") != 450):
        raise ValueError("vector manifest coverage contract mismatch")
    artifacts = manifest.get("artifacts", {})
    vector_path = Path(artifacts.get("vector_file", ""))
    bound_index = Path(artifacts.get("row_index", ""))
    if bound_index.resolve() != row_index_path.resolve():
        raise ValueError("row index path differs from vector manifest binding")
    if (not vector_path.is_file()
            or file_sha256(vector_path) != artifacts.get("vector_sha256")
            or file_sha256(row_index_path) != artifacts.get("row_index_sha256")):
        raise ValueError("vector or row-index SHA256 mismatch")
    with vector_path.open("r", encoding="ascii") as handle:
        header = handle.readline().strip()
    if header != f"{expected_rows} 450":
        raise ValueError("vector header differs from manifest")
    source_manifest = Path(manifest.get("source_manifest", ""))
    if (not source_manifest.is_file()
            or file_sha256(source_manifest) != manifest.get("source_manifest_sha256")):
        raise ValueError("source trace manifest SHA256 mismatch")
    generator = Path(artifacts.get("generator", ""))
    legacy_generator = Path(artifacts.get("legacy_semantic_generator", ""))
    if (not generator.is_file()
            or file_sha256(generator) != artifacts.get("generator_sha256")
            or not legacy_generator.is_file()
            or file_sha256(legacy_generator)
               != artifacts.get("legacy_semantic_generator_sha256")):
        raise ValueError("vector generator source SHA256 mismatch")
    records = manifest.get("records", [])
    if len(records) != 12 * manifest["sample_count"]:
        raise ValueError("vector manifest record coverage mismatch")
    for record in records:
        source = Path(record.get("source", ""))
        if (not source.is_file()
                or file_sha256(source) != record.get("source_sha256")):
            raise ValueError(f"source NPZ SHA256 mismatch: {source}")
    return manifest


def implementation_source_hashes(rtl_source_dir: Path | None = None) -> dict[str, str]:
    sources = [
        "rtl_ttx/ttx_ceil_log2_u32.sv",
        "rtl_ttx/ttx_exp2_lut_q8.sv",
        "rtl_ttx/ttx_gate_quant_q17.sv",
        "rtl_h67/h67_motionxor_score_q7.sv",
        "rtl_h67/h67_temporal_slot_encoder.sv",
        "rtl_h67/h67_sync_dual_bank_k_store.sv",
        "rtl_h67/h67_temporal_slot_fifo_2s.sv",
        "rtl_h67/h67_temporal_weighted_scs_directory_2s.sv",
        "rtl_h67/h67_temporal_slot_shiftmax_sync_k_2s_top.sv",
        "verif_h67/h67_temporal_slot_flow_2s_assertions.sv",
        "tb_h67/tb_h67_temporal_slot_flow_real_trace_2s.sv",
        "sim_h67/run_h67_rqtb_multisample_real_rtl.sh",
        "scripts/summarize_h67_rqtb_multisample_real_rtl.py",
    ]
    result = {}
    for relative in sources:
        if (rtl_source_dir is not None
                and (relative.startswith("rtl_") or relative.startswith("tb_h67/"))):
            path = rtl_source_dir / Path(relative).name
        else:
            path = ROOT / relative
        if not path.is_file():
            raise ValueError(f"implementation source is missing: {path}")
        result[relative] = file_sha256(path)
    return result


def summarize(
    icarus_log: Path,
    verilator_log: Path,
    row_index_path: Path,
    vector_manifest_path: Path,
    rtl_source_dir: Path | None = None,
) -> dict[str, Any]:
    icarus_rows, icarus_final = parse_log(icarus_log)
    verilator_rows, verilator_final = parse_log(verilator_log)
    if icarus_rows != verilator_rows or icarus_final != verilator_final:
        raise ValueError("Icarus and Verilator row/final receipts differ")
    index_rows = load_row_index(row_index_path)
    if len(index_rows) != len(icarus_rows):
        raise ValueError("RTL row count differs from row index")
    vector_manifest = validate_vector_manifest(
        vector_manifest_path, row_index_path, len(index_rows)
    )

    totals = defaultdict(int)
    sample_totals: dict[int, dict[str, Any]] = {}
    stage_totals: dict[int, dict[str, int]] = {}
    for receipt, index in zip(icarus_rows, index_rows):
        row_tag = index["row_tag"]
        if (receipt["row"] != row_tag
                or receipt["stage"] != index["stage"]
                or receipt["block"] != index["block"]
                or receipt["head"] != index["head"]
                or receipt["active"] != index["expected_outputs"]
                or receipt["fixed_slots"] != 450
                or receipt["rqtb_slots"] != 450 - receipt["equal"]):
            raise ValueError(f"RTL/index workload mismatch at row {row_tag}")
        for key in ("fixed_cycles", "rqtb_cycles", "fixed_slots", "rqtb_slots",
                    "fixed_exp", "rqtb_exp", "active", "equal"):
            totals[key] += receipt[key]
        stage = receipt["stage"]
        stage_entry = stage_totals.setdefault(
            stage,
            {
                "stage": stage,
                "rows": 0,
                "fixed_cycles": 0,
                "rqtb_cycles": 0,
                "fixed_slots": 0,
                "rqtb_slots": 0,
                "fixed_exp": 0,
                "rqtb_exp": 0,
                "equal_pairs": 0,
                "gated_k_outputs_checked": 0,
            },
        )
        stage_entry["rows"] += 1
        stage_entry["equal_pairs"] += receipt["equal"]
        stage_entry["gated_k_outputs_checked"] += receipt["active"]
        for key in ("fixed_cycles", "rqtb_cycles", "fixed_slots", "rqtb_slots",
                    "fixed_exp", "rqtb_exp"):
            stage_entry[key] += receipt[key]
        sample_id = index["sample_id"]
        entry = sample_totals.setdefault(
            sample_id,
            {
                "sample_id": sample_id,
                "sample_key": index["sample_key"],
                "rows": 0,
                "fixed_cycles": 0,
                "rqtb_cycles": 0,
                "fixed_slots": 0,
                "rqtb_slots": 0,
                "equal_pairs": 0,
            },
        )
        entry["rows"] += 1
        entry["equal_pairs"] += receipt["equal"]
        for key in ("fixed_cycles", "rqtb_cycles", "fixed_slots", "rqtb_slots"):
            entry[key] += receipt[key]

    final_map = {
        "fixed_cycles": icarus_final["fixed_cycles"],
        "rqtb_cycles": icarus_final["rqtb_cycles"],
        "fixed_slots": icarus_final["fixed_slots"],
        "rqtb_slots": icarus_final["rqtb_slots"],
        "fixed_exp": icarus_final["fixed_exp"],
        "rqtb_exp": icarus_final["rqtb_exp"],
    }
    if any(totals[key] != value for key, value in final_map.items()):
        raise ValueError("row totals differ from final PASS receipt")
    if totals["active"] != icarus_final["checked"]:
        raise ValueError("active-output total differs from final checked count")
    if totals["fixed_slots"] - totals["rqtb_slots"] != totals["equal"]:
        raise ValueError("pair-local slot/equality conservation mismatch")

    samples = [sample_totals[index] for index in range(len(sample_totals))]
    for sample in samples:
        if sample["rows"] != ROWS_PER_SAMPLE or sample["rqtb_cycles"] <= 0:
            raise ValueError(f"sample-level coverage is incomplete: {sample['sample_id']}")
        sample["speedup"] = sample["fixed_cycles"] / sample["rqtb_cycles"]
        sample["cycle_reduction_ratio"] = (
            1.0 - sample["rqtb_cycles"] / sample["fixed_cycles"]
        )
        sample["slot_reduction_ratio"] = (
            1.0 - sample["rqtb_slots"] / sample["fixed_slots"]
        )
        if sample["fixed_slots"] - sample["rqtb_slots"] != sample["equal_pairs"]:
            raise ValueError(
                f"sample slot/equality conservation mismatch: {sample['sample_id']}"
            )

    stages = [stage_totals[index] for index in range(len(EXPECTED_BLOCKS))]
    for stage in stages:
        expected_rows = (
            vector_manifest["sample_count"]
            * EXPECTED_BLOCKS[stage["stage"]]
            * EXPECTED_HEADS[stage["stage"]]
        )
        if stage["rows"] != expected_rows or stage["rqtb_cycles"] <= 0:
            raise ValueError(f"stage-level coverage is incomplete: {stage['stage']}")
        if stage["fixed_slots"] - stage["rqtb_slots"] != stage["equal_pairs"]:
            raise ValueError(
                f"stage slot/equality conservation mismatch: {stage['stage']}"
            )
        stage["speedup"] = stage["fixed_cycles"] / stage["rqtb_cycles"]
        stage["cycle_reduction_ratio"] = (
            1.0 - stage["rqtb_cycles"] / stage["fixed_cycles"]
        )
        stage["slot_reduction_ratio"] = (
            1.0 - stage["rqtb_slots"] / stage["fixed_slots"]
        )

    row_fixed = [row["fixed_cycles"] for row in icarus_rows]
    row_rqtb = [row["rqtb_cycles"] for row in icarus_rows]
    row_speedup = [fixed / rqtb for fixed, rqtb in zip(row_fixed, row_rqtb)]
    global_speedup = totals["fixed_cycles"] / totals["rqtb_cycles"]
    result = {
        "schema": "h67_rqtb_multisample_real_rtl_v1",
        "status": "PASS",
        "evidence_level": "[rtl]",
        "scope": (
            "real checkpoint Q/K/gate traces, one selected window per all12 "
            "attention block per sample, Fixed2S versus RQTB2S"
        ),
        "coverage": {
            "samples": len(samples),
            "rows": len(icarus_rows),
            "rows_per_sample": ROWS_PER_SAMPLE,
            "tokens_per_row": 450,
            "gated_k_outputs_checked": icarus_final["checked"],
            "acc32_mismatch": 0,
            "cross_simulator_exact": True,
        },
        "cycles": {
            "fixed_total": totals["fixed_cycles"],
            "rqtb_total": totals["rqtb_cycles"],
            "global_speedup": global_speedup,
            "global_cycle_reduction_ratio": 1.0 - 1.0 / global_speedup,
            "row_distribution": {
                "fixed_cycles": distribution(row_fixed),
                "rqtb_cycles": distribution(row_rqtb),
                "speedup": distribution(row_speedup),
            },
            "sample_distribution": {
                "fixed_cycles": distribution([s["fixed_cycles"] for s in samples]),
                "rqtb_cycles": distribution([s["rqtb_cycles"] for s in samples]),
                "speedup": distribution([s["speedup"] for s in samples]),
            },
        },
        "work": {
            "fixed_slots": totals["fixed_slots"],
            "rqtb_slots": totals["rqtb_slots"],
            "slot_reduction_ratio": 1.0 - totals["rqtb_slots"] / totals["fixed_slots"],
            "fixed_exp": totals["fixed_exp"],
            "rqtb_exp": totals["rqtb_exp"],
            "exp_reduction_ratio": 1.0 - totals["rqtb_exp"] / totals["fixed_exp"],
        },
        "samples": samples,
        "stages": stages,
        "synthetic_acc32_boundary": {
            "status": "PASS_SYNTHETIC_ONLY",
            "contract": "TB lane_weight(lane)=(lane%17)-8",
            "meaning": "integer checksum boundary, not real projection weights",
            "forbidden_claims": [
                "real-weight projection equivalence",
                "full-encoder equivalence",
                "deployment accuracy",
            ],
        },
        "provenance": {
            "icarus_log": str(icarus_log.resolve()),
            "icarus_log_sha256": file_sha256(icarus_log),
            "verilator_sva_log": str(verilator_log.resolve()),
            "verilator_sva_log_sha256": file_sha256(verilator_log),
            "row_index": str(row_index_path.resolve()),
            "row_index_sha256": file_sha256(row_index_path),
            "vector_manifest": str(vector_manifest_path.resolve()),
            "vector_manifest_sha256": file_sha256(vector_manifest_path),
            "source_trace_manifest_sha256": vector_manifest["source_manifest_sha256"],
            "implementation_source_mode": (
                "frozen_flat_directory" if rtl_source_dir is not None else "live_tree"
            ),
            "implementation_source_root": str(
                (rtl_source_dir if rtl_source_dir is not None else ROOT).resolve()
            ),
            "implementation_source_sha256": implementation_source_hashes(
                rtl_source_dir
            ),
        },
        "claim_boundary": [
            "Results cover one preregistered window per block, not all spatial windows.",
            "Cycle counts are RTL simulation cycles under the TB fixed-seed backpressure.",
            "No ASIC PPA, FPS, full encoder, or real-weight Acc32 claim is made.",
        ],
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--icarus-log", type=Path, required=True)
    parser.add_argument("--verilator-log", type=Path, required=True)
    parser.add_argument("--row-index", type=Path, required=True)
    parser.add_argument("--vector-manifest", type=Path, required=True)
    parser.add_argument("--rtl-source-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.icarus_log,
        args.verilator_log,
        args.row_index,
        args.vector_manifest,
        args.rtl_source_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(
        f"PASS H67 multisample real RTL samples={result['coverage']['samples']} "
        f"rows={result['coverage']['rows']} "
        f"speedup={result['cycles']['global_speedup']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
