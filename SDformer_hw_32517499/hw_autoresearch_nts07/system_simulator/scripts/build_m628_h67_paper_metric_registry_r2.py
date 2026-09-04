#!/usr/bin/env python3
"""Validate and render the M628 Table A/B/C registry without running EDA/GPU."""

import argparse
import hashlib
import json
import math
import os
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any, Dict, List, Set


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "hw_autoresearch_nts07/system_simulator/config/m628_h67_paper_metric_registry_r2_20260828.json"
TABLE_A_KEYS = {
    "row_id", "role", "fidelity", "cycles", "energy_mj", "area_mm2", "accuracy",
    "source_id", "measurement_class", "population_id", "workload_id",
    "resource_manifest_sha256", "completion_receipt_sha256", "decoder_complete",
    "memory_timing_included", "full_network_completion", "logic_sram_dram_energy_closed",
    "logic_macro_area_closed", "sta_closed", "independent_hammer_pass", "blockers",
}


class RegistryError(ValueError):
    pass


def _no_duplicates(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise RegistryError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def load_json(path):
    try:
        obj = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_no_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(RegistryError(f"non-finite JSON number: {token}")),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryError(f"cannot load {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise RegistryError("registry root must be an object")
    return obj


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def secure_repo_file(relative):
    if not relative or os.path.isabs(relative):
        raise RegistryError(f"source path must be non-empty and repo-relative: {relative!r}")
    candidate = REPO_ROOT / relative
    current = REPO_ROOT
    for part in Path(relative).parts:
        if part in ("", ".", ".."):
            raise RegistryError(f"unsafe source path component: {relative}")
        current = current / part
        if current.is_symlink():
            raise RegistryError(f"symlink source component refused: {relative}")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(REPO_ROOT.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise RegistryError(f"source escapes repo or is missing: {relative}") from exc
    if not resolved.is_file():
        raise RegistryError(f"source is not a regular file: {relative}")
    return resolved


def validate_sources(config):
    sources = config.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise RegistryError("sources must be a non-empty object")
    validated = {}
    for source_id, source in sources.items():
        if not isinstance(source, dict) or set(source) != {"path", "sha256"}:
            raise RegistryError(f"source {source_id} must contain exactly path and sha256")
        expected = source["sha256"]
        if not isinstance(expected, str) or len(expected) != 64:
            raise RegistryError(f"source {source_id} has invalid sha256")
        actual = sha256_file(secure_repo_file(source["path"]))
        if actual != expected:
            raise RegistryError(f"source SHA mismatch for {source_id}: expected {expected}, got {actual}")
        validated[source_id] = actual
    return validated


def _require_exact_fields(row, expected, label):
    actual = set(row)
    if actual != expected:
        raise RegistryError(f"{label} fields differ: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")


def _validate_source_refs(rows, source_ids, table):
    for row in rows:
        refs = row.get("source_ids")
        if not isinstance(refs, list) or not refs or any(ref not in source_ids for ref in refs):
            raise RegistryError(f"{table} row {row.get('metric_id')} has invalid source_ids")


def validate_tables(config, source_ids):
    table_a = config.get("table_a_schema")
    if not isinstance(table_a, dict):
        raise RegistryError("table_a_schema missing")
    if set(table_a.get("required_fields", [])) != TABLE_A_KEYS:
        raise RegistryError("Table A required_fields do not match the executable schema")
    rows = table_a.get("rows")
    if not isinstance(rows, list):
        raise RegistryError("Table A rows must be a list")
    for row in rows:
        if not isinstance(row, dict):
            raise RegistryError("Table A row must be an object")
        _require_exact_fields(row, TABLE_A_KEYS, f"Table A row {row.get('row_id')}")
        if not isinstance(row["blockers"], list):
            raise RegistryError(f"Table A row {row['row_id']} blockers must be a list")
    row_ids = [row["row_id"] for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise RegistryError("duplicate Table A row_id")
    required = table_a.get("required_row_ids")
    optional = table_a.get("optional_row_ids")
    if not isinstance(required, list) or not set(required).issubset(row_ids):
        raise RegistryError("Table A required row is missing")
    if set(row_ids) != set(required) | set(optional or []):
        raise RegistryError("Table A contains an unregistered row_id")

    for table_name, expected in (
        ("table_b_schema", {"metric_id","source_ids","value","unit","scope","measurement_class","headline_eligible","limitation"}),
        ("table_c_schema", {"metric_id","source_ids","value","unit","provenance","ours","headline_eligible","limitation"}),
    ):
        table = config.get(table_name)
        if not isinstance(table, dict) or set(table.get("required_fields", [])) != expected:
            raise RegistryError(f"{table_name} schema mismatch")
        metric_rows = table.get("rows")
        if not isinstance(metric_rows, list):
            raise RegistryError(f"{table_name} rows must be a list")
        for row in metric_rows:
            _require_exact_fields(row, expected, f"{table_name} row {row.get('metric_id')}")
            if row["headline_eligible"] is not False:
                raise RegistryError(f"{table_name} row {row['metric_id']} cannot be headline eligible")
            if table_name == "table_c_schema" and row["ours"] is not False:
                raise RegistryError(f"Table C row {row['metric_id']} cannot be labelled ours")
        _validate_source_refs(metric_rows, source_ids, table_name)


def recompute_analytical(config):
    item = config.get("analytical_diagnostic")
    if not isinstance(item, dict) or item.get("admitted") is not False:
        raise RegistryError("analytical diagnostic must be explicitly non-admitted")
    getcontext().prec = 50
    numerator = Decimal(item["fixed_numerator_cycles"])
    candidate_low = Decimal(item["candidate_cycles_low"])
    candidate_high = Decimal(item["candidate_cycles_high"])
    if not (numerator > 0 and candidate_low > 0 and candidate_high >= candidate_low):
        raise RegistryError("invalid analytical cycle range")
    speedup_low = numerator / candidate_high
    speedup_high = numerator / candidate_low
    expected_low = Decimal(item["expected_speedup_low"])
    expected_high = Decimal(item["expected_speedup_high"])
    tolerance = Decimal("1e-27")
    if abs(speedup_low - expected_low) > tolerance or abs(speedup_high - expected_high) > tolerance:
        raise RegistryError("analytical speedup anchors do not independently recompute")
    reasons = item.get("rejection_reasons")
    if not isinstance(reasons, list) or len(reasons) < 6:
        raise RegistryError("analytical rejection reasons are incomplete")
    return {
        "fixed_numerator_cycles": int(numerator),
        "candidate_cycles_low": int(candidate_low),
        "candidate_cycles_high": int(candidate_high),
        "speedup_low": format(speedup_low, ".28f"),
        "speedup_high": format(speedup_high, ".28f"),
        "admitted": False,
        "rejection_reasons": reasons,
    }


def evaluate_headline_gate(config):
    policy = config.get("headline_policy")
    if not isinstance(policy, dict) or policy.get("table_a_only") is not True:
        raise RegistryError("headline policy must be Table-A-only")
    if policy.get("prohibit_ratio_multiplication") is not True or policy.get("prohibit_table_b_or_c_promotion") is not True:
        raise RegistryError("headline policy promotion safeguards must be enabled")
    rows = config["table_a_schema"]["rows"]
    eligible = []
    failures = {}
    required_true = [
        "decoder_complete", "memory_timing_included", "full_network_completion",
        "logic_sram_dram_energy_closed", "logic_macro_area_closed", "sta_closed",
        "independent_hammer_pass",
    ]
    required_values = [
        "cycles", "energy_mj", "area_mm2", "accuracy", "source_id", "population_id",
        "workload_id", "resource_manifest_sha256", "completion_receipt_sha256",
    ]
    for row in rows:
        reasons = list(row["blockers"])
        if row["measurement_class"] != policy["allowed_measurement_class"]:
            reasons.append("measurement_class_not_direct_unified")
        for field in required_values:
            if row[field] is None:
                reasons.append(f"missing_{field}")
        for field in required_true:
            if row[field] is not True:
                reasons.append(f"{field}_false")
        if not reasons:
            if row["source_id"] not in config["sources"]:
                reasons.append("unknown_source_id")
            elif not all(isinstance(row[field], (int, float)) and math.isfinite(row[field]) for field in ("cycles","energy_mj","area_mm2","accuracy")):
                reasons.append("nonfinite_or_nonnumeric_measurement")
        if reasons:
            failures[row["row_id"]] = sorted(set(reasons))
        else:
            eligible.append(row["row_id"])
    required_rows = set(config["table_a_schema"]["required_row_ids"])
    all_required_eligible = required_rows.issubset(eligible)
    manifest = config["table_a_schema"].get("run_manifest")
    manifest_keys = {
        "population_id", "workload_id", "resource_manifest_sha256", "sequence_ids",
        "density_strata", "aggregates_present", "views_present",
    }
    if not isinstance(manifest, dict) or set(manifest) != manifest_keys:
        raise RegistryError("Table A run_manifest schema mismatch")
    global_failures = []
    required_objects = [row for row in rows if row["row_id"] in required_rows]
    for identity in ("population_id", "workload_id", "resource_manifest_sha256"):
        if manifest[identity] is None:
            global_failures.append("missing_manifest_" + identity)
        elif any(row[identity] != manifest[identity] for row in required_objects):
            global_failures.append("row_manifest_" + identity + "_mismatch")
    sequence_ids = manifest["sequence_ids"]
    density_strata = manifest["density_strata"]
    coverage_ok = (
        isinstance(sequence_ids, list) and len(set(sequence_ids)) >= policy["coverage_rule"]["minimum_sequences"]
    ) or (
        isinstance(density_strata, list)
        and set(policy["coverage_rule"]["alternative_preregistered_density_strata"]).issubset(density_strata)
    )
    if not coverage_ok:
        global_failures.append("population_coverage_not_met")
    if not isinstance(manifest["aggregates_present"], list) or not set(policy["required_aggregates"]).issubset(manifest["aggregates_present"]):
        global_failures.append("mandatory_aggregates_missing")
    if not isinstance(manifest["views_present"], list) or not set(policy["required_views"]).issubset(manifest["views_present"]):
        global_failures.append("mandatory_views_missing")
    by_id = {row["row_id"]: row for row in rows}
    numerator = by_id[policy["fixed_numerator_row_id"]]["cycles"]
    candidate = by_id[policy["candidate_row_id"]]["cycles"]
    direct_speedup = None
    if isinstance(numerator, (int, float)) and isinstance(candidate, (int, float)) and candidate > 0:
        direct_speedup = numerator / candidate
        if direct_speedup < policy["minimum_direct_speedup_for_accept"]:
            global_failures.append("direct_speedup_below_accept_floor")
    else:
        global_failures.append("direct_speedup_unavailable")
    admitted = all_required_eligible and not global_failures
    if config["claim_boundary"].get("table_a_admitted_rows") != len(eligible):
        raise RegistryError("claim_boundary table_a_admitted_rows disagrees with executable gate")
    if config["claim_boundary"].get("paper_headline_admitted") is not admitted:
        raise RegistryError("claim_boundary paper_headline_admitted disagrees with executable gate")
    return {
        "admitted": admitted,
        "eligible_row_ids": eligible,
        "eligible_row_count": len(eligible),
        "required_row_count": len(required_rows),
        "row_failures": failures,
        "global_failures": sorted(set(global_failures)),
        "direct_speedup": direct_speedup,
        "default_aggregate": policy["default_headline_aggregate"],
        "fixed_numerator_row_id": policy["fixed_numerator_row_id"],
    }


def build(config_path=DEFAULT_CONFIG):
    config = load_json(config_path)
    if config.get("schema") != "m628.h67.paper_metric_registry.r2":
        raise RegistryError("unexpected registry schema")
    sources = validate_sources(config)
    protected = config.get("protected_file")
    if not isinstance(protected, dict) or set(protected) != {"path", "sha256"}:
        raise RegistryError("protected_file binding missing")
    if sha256_file(secure_repo_file(protected["path"])) != protected["sha256"]:
        raise RegistryError("protected docs359 SHA mismatch")
    validate_tables(config, set(sources))
    analytical = recompute_analytical(config)
    gate = evaluate_headline_gate(config)
    if config.get("strong_accept_minimum_evidence_set", {}).get("fourth_matcher_required") is not False:
        raise RegistryError("M628 must not propose a fourth matcher")
    if config.get("claim_boundary", {}).get("eda_or_gpu_run") is not False:
        raise RegistryError("M628 is a no-EDA/no-GPU package")
    return {
        "schema": "m628.h67.paper_metric_registry.r2.preview",
        "status": config["status"],
        "source_hashes_validated": sources,
        "table_a": config["table_a_schema"]["rows"],
        "table_b": config["table_b_schema"]["rows"],
        "table_c": config["table_c_schema"]["rows"],
        "analytical_diagnostic": analytical,
        "headline_gate": gate,
        "strong_accept_minimum_evidence_set": config["strong_accept_minimum_evidence_set"],
        "no_eda_work_queue": config["no_eda_work_queue"],
        "claim_boundary": config["claim_boundary"],
        "protected_file_validated": protected,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--emit-json", action="store_true")
    args = parser.parse_args()
    try:
        result = build(args.config)
    except RegistryError as exc:
        print(f"M628_REGISTRY_FAIL: {exc}")
        return 2
    if args.emit_json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(
            "M628_REGISTRY_PASS "
            f"sources={len(result['source_hashes_validated'])} "
            f"table_a_eligible={result['headline_gate']['eligible_row_count']} "
            f"headline_admitted={str(result['headline_gate']['admitted']).lower()} "
            f"analytical_admitted={str(result['analytical_diagnostic']['admitted']).lower()}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
