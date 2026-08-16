#!/usr/bin/env python3
"""Fail-closed G2 preflight and exact EDP gate calculator for Local5 EREP."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts/local5_erep_g2_preimplementation_contract_v1_20260810.json"
CANDIDATES = tuple(f"C{index}" for index in range(6))
BOUNDARY = "local5_erep_relation_to_acc32_v1"
RECEIPT_TOP_FIELDS = frozenset({"schema", "status", "shared", "artifacts", "candidates"})
CANDIDATE_FIELDS = frozenset(
    {
        "id",
        "boundary_id",
        "clock_period_ns",
        "pvt_corner",
        "sdc_sha256",
        "memory_macro_policy_sha256",
        "common_activity_stimulus_sha256",
        "idle_clock_gating_policy_sha256",
        "rtl_sha256",
        "filelist_sha256",
        "parameter_sha256",
    }
)
RESULT_TOP_FIELDS = frozenset({"schema", "g2b_receipt_sha256", "candidates"})
RESULT_CANDIDATE_FIELDS = frozenset(
    {
        "id",
        "timing_pass",
        "weighted_energy_joule",
        "weighted_latency_second",
        "activity_annotation_coverage_percent",
        "unknown_toggle_count",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise ValueError(f"required artifact is absent: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error


def strict_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def positive_decimal(value: Any, name: str) -> Decimal:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an exact decimal string")
    try:
        result = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"{name} must be an exact decimal string") from error
    if not result.is_finite() or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _shared_receipt_fields(contract: dict[str, Any]) -> set[str]:
    return set(contract["g2b_target_asic_receipt"]["required_fields"])


def validate_g2b_preflight(receipt_path: Path) -> dict[str, Any]:
    contract = load_json(CONTRACT)
    receipt = load_json(receipt_path)
    if not isinstance(receipt, dict) or set(receipt) != RECEIPT_TOP_FIELDS:
        raise ValueError("G2b receipt has a non-frozen top-level field set")
    if receipt["schema"] != "local5_erep_g2b_run_receipt_v1" or receipt["status"] != "READY":
        raise ValueError("G2b receipt schema/status must be READY")
    shared = receipt["shared"]
    expected_shared = _shared_receipt_fields(contract)
    if not isinstance(shared, dict) or set(shared) != expected_shared:
        raise ValueError("G2b shared receipt has a non-frozen field set")
    for name, value in shared.items():
        if value is None or value == "":
            raise ValueError(f"G2b required field is empty: {name}")
        if name.endswith("_sha256"):
            strict_sha(value, name)
    for name in (
        "receipt_id",
        "receipt_created_utc",
        "target_process_node",
        "target_library_name",
        "pvt_corner",
        "operating_condition_name",
        "dc_version",
        "sta_tool_version",
        "saif_tool_version",
        "ptpx_version",
        "sdc_path",
    ):
        nonempty_string(shared[name], name)
    if type(shared["supply_voltage_v"]) not in (int, float) or shared["supply_voltage_v"] <= 0:
        raise ValueError("supply_voltage_v must be a positive number")
    if type(shared["junction_temperature_c"]) not in (int, float):
        raise ValueError("junction_temperature_c must be a number")

    artifacts = receipt["artifacts"]
    sha_fields = {name for name in expected_shared if name.endswith("_sha256")}
    if not isinstance(artifacts, dict) or set(artifacts) != sha_fields:
        raise ValueError("G2b artifacts must bind every shared SHA field exactly once")
    for field, relative in artifacts.items():
        path = Path(nonempty_string(relative, f"artifacts[{field}]")).expanduser()
        if not path.is_absolute():
            path = (receipt_path.parent / path).resolve()
        if not path.is_file() or sha256_file(path) != shared[field]:
            raise ValueError(f"G2b artifact hash mismatch: {field}")
    sdc_path = Path(shared["sdc_path"]).expanduser()
    if not sdc_path.is_absolute():
        sdc_path = (receipt_path.parent / sdc_path).resolve()
    if not sdc_path.is_file() or sha256_file(sdc_path) != shared["sdc_sha256"]:
        raise ValueError("G2b SDC path/SHA binding failed")

    candidates = receipt["candidates"]
    if not isinstance(candidates, list) or len(candidates) != 6:
        raise ValueError("G2b receipt must bind exactly C0-C5")
    if [row.get("id") for row in candidates if isinstance(row, dict)] != list(CANDIDATES):
        raise ValueError("G2b candidate order must be exactly C0-C5")
    common = {
        "boundary_id": BOUNDARY,
        "clock_period_ns": 5.0,
        "pvt_corner": shared["pvt_corner"],
        "sdc_sha256": shared["sdc_sha256"],
        "memory_macro_policy_sha256": shared["sram_macro_port_latency_contract_sha256"],
        "common_activity_stimulus_sha256": shared["common_activity_stimulus_sha256"],
        "idle_clock_gating_policy_sha256": shared["idle_clock_gating_policy_sha256"],
    }
    for index, row in enumerate(candidates):
        if not isinstance(row, dict) or set(row) != CANDIDATE_FIELDS:
            raise ValueError(f"candidate C{index} has a non-frozen field set")
        for field, expected in common.items():
            if row[field] != expected:
                raise ValueError(f"candidate C{index} violates common field {field}")
        for field in ("rtl_sha256", "filelist_sha256", "parameter_sha256"):
            strict_sha(row[field], f"candidate C{index} {field}")
    return {
        "schema": "local5_erep_g2b_preflight_result_v1",
        "status": "PASS",
        "receipt_sha256": sha256_file(receipt_path),
        "candidate_ids": list(CANDIDATES),
        "boundary_id": BOUNDARY,
        "clock_period_ns": 5.0,
        "pvt_corner": shared["pvt_corner"],
    }


def evaluate_g2_results(results_path: Path, receipt_path: Path) -> dict[str, Any]:
    preflight = validate_g2b_preflight(receipt_path)
    results = load_json(results_path)
    if not isinstance(results, dict) or set(results) != RESULT_TOP_FIELDS:
        raise ValueError("G2 result bundle has a non-frozen field set")
    if (
        results["schema"] != "local5_erep_g2_result_bundle_v1"
        or results["g2b_receipt_sha256"] != preflight["receipt_sha256"]
    ):
        raise ValueError("G2 result bundle is not bound to the preflight receipt")
    rows = results["candidates"]
    if not isinstance(rows, list) or len(rows) != 6 or [row.get("id") for row in rows] != list(CANDIDATES):
        raise ValueError("G2 result bundle must contain C0-C5 in order")
    edp: dict[str, Decimal] = {}
    normalized = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != RESULT_CANDIDATE_FIELDS:
            raise ValueError(f"G2 result C{index} has a non-frozen field set")
        if row["timing_pass"] is not True:
            raise ValueError(f"candidate C{index} failed the common timing target")
        if type(row["unknown_toggle_count"]) is not int or row["unknown_toggle_count"] != 0:
            raise ValueError(f"candidate C{index} has unknown activity toggles")
        coverage = positive_decimal(row["activity_annotation_coverage_percent"], f"C{index} coverage")
        if coverage < Decimal("95") or coverage > Decimal("100"):
            raise ValueError(f"candidate C{index} annotation coverage is outside [95,100]")
        energy = positive_decimal(row["weighted_energy_joule"], f"C{index} energy")
        latency = positive_decimal(row["weighted_latency_second"], f"C{index} latency")
        with localcontext() as context:
            context.prec = 80
            edp[row["id"]] = energy * latency
        normalized.append(
            {
                "id": row["id"],
                "weighted_energy_joule": str(energy),
                "weighted_latency_second": str(latency),
                "edp_joule_second": str(edp[row["id"]]),
            }
        )
    with localcontext() as context:
        context.prec = 80
        primary_pass = edp["C0"] * Decimal(4) >= edp["C3"] * Decimal(5)
        c4_pass = edp["C4"] * Decimal(19) >= edp["C3"] * Decimal(20)
    gates = [
        {
            "name": "c3_edp_reduction_vs_c0_at_least_20_percent",
            "exact_test": "4*EDP_C0 >= 5*EDP_C3",
            "ratio_threshold": "5/4",
            "passed": primary_pass,
        },
        {
            "name": "c3_edp_reduction_vs_c4_at_least_5_percent",
            "exact_test": "19*EDP_C4 >= 20*EDP_C3",
            "ratio_threshold": "20/19",
            "passed": c4_pass,
        },
    ]
    return {
        "schema": "local5_erep_g2_gate_result_v1",
        "receipt_sha256": preflight["receipt_sha256"],
        "candidates": normalized,
        "gates": gates,
        "g2_passed": all(gate["passed"] for gate in gates),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("receipt", type=Path)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("receipt", type=Path)
    evaluate.add_argument("results", type=Path)
    args = parser.parse_args()
    if args.command == "preflight":
        report = validate_g2b_preflight(args.receipt)
    else:
        report = evaluate_g2_results(args.results, args.receipt)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
