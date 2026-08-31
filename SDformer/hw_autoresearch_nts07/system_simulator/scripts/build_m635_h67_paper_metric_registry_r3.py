#!/usr/bin/env python3
"""Build the M635 H67 paper registry r3 without running GPU or EDA.

This supersedes M628's registry methodology.  It deliberately does not alter
or reinterpret any sealed M628 artifact.  In particular, a Table-A row is not
admitted from values or booleans written in this registry.  It must be an
exact projection of a dedicated, repo-local direct-unified evidence bundle.
"""

import argparse
import hashlib
import json
import math
import os
from decimal import Decimal, getcontext
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "hw_autoresearch_nts07/system_simulator/config/m635_h67_paper_metric_registry_r3_20260828.json"

MANDATORY_ROW_SPECS = (
    ("dense96_fixed_t10", "numerator", "exact"),
    ("ptb_like_structured", "baseline", "exact"),
    ("exact_bit_k1", "baseline", "exact"),
    ("exact_bit_k1x8", "strongest_same_page_baseline", "exact"),
    ("exact_typed_k8", "baseline", "exact"),
    ("ours_exact", "candidate", "exact"),
)
OPTIONAL_ROW_SPECS = (("ours_lossy", "candidate_secondary", "lossy"),)
FIXED_NUMERATOR_ROW_ID = "dense96_fixed_t10"
STRONGEST_BASELINE_ROW_ID = "exact_bit_k1x8"
CANDIDATE_ROW_ID = "ours_exact"
ALLOWED_MEASUREMENT_CLASS = "DIRECT_UNIFIED_CYCLE_SIM"

TABLE_A_KEYS = {
    "row_id", "role", "fidelity", "cycles", "energy_mj", "area_mm2", "accuracy",
    "source_id", "measurement_class", "population_id", "workload_id",
    "resource_manifest_sha256", "completion_receipt_sha256", "decoder_complete",
    "memory_timing_included", "full_network_completion", "logic_sram_dram_energy_closed",
    "logic_macro_area_closed", "sta_closed", "independent_hammer_pass", "blockers",
}
TABLE_B_KEYS = {
    "metric_id", "source_ids", "value", "unit", "scope", "measurement_class",
    "headline_eligible", "limitation",
}
TABLE_C_KEYS = {
    "metric_id", "source_ids", "value", "unit", "provenance", "ours",
    "headline_eligible", "limitation",
}
SOURCE_KEYS = {"path", "sha256", "media_type"}
BUNDLE_ARTIFACTS = {
    "direct_result": "hw_autoresearch_nts07/results/",
    "completion_receipt": "hw_autoresearch_nts07/results/",
    "resource_manifest": "hw_autoresearch_nts07/system_simulator/",
    "coverage_receipt": "hw_autoresearch_nts07/results/",
    "independent_hammer_receipt": "hw_autoresearch_nts07/reviews/",
}
REQUIRED_CLOSURES = (
    "decoder_complete", "memory_timing_included", "full_network_completion",
    "logic_sram_dram_energy_closed", "logic_macro_area_closed", "sta_closed",
)
AGGREGATE_NAMES = ("arithmetic_mean", "geometric_mean", "ratio_of_sums", "minimum", "maximum")
VIEW_NAMES = ("iso_lane", "iso_service")


class RegistryError(ValueError):
    pass


def _no_duplicates(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise RegistryError("duplicate JSON key: %s" % key)
        out[key] = value
    return out


def _reject_nonfinite(value, label="JSON"):
    if isinstance(value, float) and not math.isfinite(value):
        raise RegistryError("non-finite JSON number in %s" % label)
    if isinstance(value, dict):
        for item in value.values():
            _reject_nonfinite(item, label)
    elif isinstance(value, list):
        for item in value:
            _reject_nonfinite(item, label)


def load_json(path, label="JSON evidence"):
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_no_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                RegistryError("non-finite JSON number: %s" % token)
            ),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryError("cannot load %s: %s" % (path, exc))
    _reject_nonfinite(value, label)
    if not isinstance(value, dict):
        raise RegistryError("%s root must be an object" % label)
    return value


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def secure_repo_file(relative):
    if not isinstance(relative, str) or not relative or os.path.isabs(relative):
        raise RegistryError("path must be non-empty and repo-relative: %r" % relative)
    current = REPO_ROOT
    for part in Path(relative).parts:
        if part in ("", ".", ".."):
            raise RegistryError("unsafe path component: %s" % relative)
        current = current / part
        if current.is_symlink():
            raise RegistryError("symlink path component refused: %s" % relative)
    try:
        resolved = (REPO_ROOT / relative).resolve(strict=True)
        resolved.relative_to(REPO_ROOT.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise RegistryError("path escapes repo or is missing: %s" % relative)
    if not resolved.is_file():
        raise RegistryError("path is not a regular file: %s" % relative)
    return resolved


def _validate_file_spec(spec, label, required_prefix=None):
    if not isinstance(spec, dict) or set(spec) != SOURCE_KEYS:
        raise RegistryError("%s must contain exactly path, sha256 and media_type" % label)
    if spec["media_type"] != "application/json":
        raise RegistryError("%s requires application/json" % label)
    if required_prefix is not None and not spec["path"].startswith(required_prefix):
        raise RegistryError("%s path is outside its dedicated namespace" % label)
    expected = spec["sha256"]
    if not isinstance(expected, str) or len(expected) != 64:
        raise RegistryError("%s has invalid sha256" % label)
    path = secure_repo_file(spec["path"])
    actual = sha256_file(path)
    if actual != expected:
        raise RegistryError("%s SHA mismatch: expected %s, got %s" % (label, expected, actual))
    # Every registered evidence source is parsed strictly, not merely hashed.
    parsed = load_json(path, label)
    return path, parsed, actual


def validate_sources(config):
    sources = config.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise RegistryError("sources must be a non-empty object")
    validated = {}
    documents = {}
    for source_id, spec in sources.items():
        path, parsed, actual = _validate_file_spec(spec, "source %s" % source_id)
        validated[source_id] = actual
        documents[source_id] = parsed
    return validated, documents


def _require_exact_fields(value, expected, label):
    if not isinstance(value, dict):
        raise RegistryError("%s must be an object" % label)
    actual = set(value)
    if actual != expected:
        raise RegistryError(
            "%s fields differ: missing=%s, extra=%s"
            % (label, sorted(expected - actual), sorted(actual - expected))
        )


def _positive_finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise RegistryError("%s must be a positive finite number" % label)


def _sha(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise RegistryError("%s must be a lowercase sha256" % label)


def _mandatory_specs_dict():
    return {row_id: (role, fidelity) for row_id, role, fidelity in MANDATORY_ROW_SPECS}


def validate_ladder_and_tables(config, source_ids):
    table_a = config.get("table_a_schema")
    if not isinstance(table_a, dict):
        raise RegistryError("table_a_schema missing")
    mandatory_ids = [row[0] for row in MANDATORY_ROW_SPECS]
    optional_ids = [row[0] for row in OPTIONAL_ROW_SPECS]
    if table_a.get("required_row_ids") != mandatory_ids:
        raise RegistryError("Table A mandatory row IDs/order differ from code-level ladder")
    if table_a.get("optional_row_ids") != optional_ids:
        raise RegistryError("Table A optional row IDs/order differ from code-level ladder")
    if set(table_a.get("required_fields", [])) != TABLE_A_KEYS:
        raise RegistryError("Table A required_fields differ from executable schema")
    rows = table_a.get("rows")
    if not isinstance(rows, list):
        raise RegistryError("Table A rows must be a list")
    expected_all = mandatory_ids + optional_ids
    if [row.get("row_id") if isinstance(row, dict) else None for row in rows] != expected_all:
        raise RegistryError("Table A rows/order differ from code-level mandatory+optional ladder")
    specs = dict(_mandatory_specs_dict())
    specs.update({row_id: (role, fidelity) for row_id, role, fidelity in OPTIONAL_ROW_SPECS})
    for row in rows:
        _require_exact_fields(row, TABLE_A_KEYS, "Table A row %s" % row.get("row_id"))
        if (row["role"], row["fidelity"]) != specs[row["row_id"]]:
            raise RegistryError("Table A role/fidelity mutation for %s" % row["row_id"])
        if not isinstance(row["blockers"], list):
            raise RegistryError("Table A blockers must be a list")

    for table_name, keys in (("table_b_schema", TABLE_B_KEYS), ("table_c_schema", TABLE_C_KEYS)):
        table = config.get(table_name)
        if not isinstance(table, dict) or set(table.get("required_fields", [])) != keys:
            raise RegistryError("%s schema mismatch" % table_name)
        for row in table.get("rows", []):
            _require_exact_fields(row, keys, "%s row" % table_name)
            refs = row.get("source_ids")
            if not isinstance(refs, list) or not refs or any(ref not in source_ids for ref in refs):
                raise RegistryError("%s row has invalid source_ids" % table_name)
            if row["headline_eligible"] is not False:
                raise RegistryError("%s row cannot be headline eligible" % table_name)
            if table_name == "table_c_schema" and row["ours"] is not False:
                raise RegistryError("Table C row cannot be labelled ours")
    return rows


def validate_policy(config):
    policy = config.get("headline_policy")
    if not isinstance(policy, dict):
        raise RegistryError("headline_policy missing")
    anchors = (
        policy.get("fixed_numerator_row_id"),
        policy.get("strongest_same_page_baseline_row_id"),
        policy.get("candidate_row_id"),
    )
    if anchors != (FIXED_NUMERATOR_ROW_ID, STRONGEST_BASELINE_ROW_ID, CANDIDATE_ROW_ID):
        raise RegistryError("headline anchors differ from code-level anchors")
    if policy.get("allowed_measurement_class") != ALLOWED_MEASUREMENT_CLASS:
        raise RegistryError("allowed Table A measurement class mutated")
    if policy.get("required_aggregates") != list(AGGREGATE_NAMES):
        raise RegistryError("required aggregate list/order mutated")
    if policy.get("required_views") != list(VIEW_NAMES):
        raise RegistryError("required view list/order mutated")
    if policy.get("table_a_only") is not True:
        raise RegistryError("headline must be Table-A-only")
    if policy.get("prohibit_ratio_multiplication") is not True or policy.get("prohibit_table_b_or_c_promotion") is not True:
        raise RegistryError("headline promotion safeguards must remain enabled")
    return policy


def validate_m518_binding(config, source_docs):
    expected_sha = "513c5d916859b0f48b9ffeced6853ad89a8ace5ea6a9b264baf05d1ed1966665"
    if config["sources"].get("m518", {}).get("sha256") != expected_sha:
        raise RegistryError("M518 must bind the r11 post-run receipt hammer SHA")
    verdict = source_docs["m518"]
    if verdict.get("schema") != "m518_r11_matched_fixed_t10_atlif_vcs_receipt_blind_hammer_verdict_v1":
        raise RegistryError("M518 source is not the r11 post-run receipt hammer")
    severity = verdict.get("severity_counts", {})
    authorization = verdict.get("authorization", {})
    if severity.get("p0") != 0 or severity.get("p1") != 0:
        raise RegistryError("M518 receipt hammer has blocking findings")
    if authorization.get("rtl_cycle_anchors_admitted") is not True:
        raise RegistryError("M518 receipt hammer does not admit RTL cycle anchors")
    expected_cycle = verdict.get("cycle_anchors", {}).get("issue_cycles_per_tile")
    rows = config["table_b_schema"]["rows"]
    selected = [row for row in rows if row["metric_id"] == "m518_fixed_t10_directed_issue_cycle_anchor"]
    if len(selected) != 1 or selected[0]["source_ids"] != ["m518"] or selected[0]["value"] != expected_cycle:
        raise RegistryError("M518 Table B value/source does not match the receipt hammer")


def _validate_direct_result(doc, label):
    keys = {
        "schema", "status", "measurement_class", "population_id", "workload_id",
        "resource_manifest_sha256", "rows", "samples", "aggregates", "views",
    }
    _require_exact_fields(doc, keys, label)
    if doc["schema"] != "m635.h67.direct_unified.result.r1" or doc["status"] != "PASS_COMPLETE":
        raise RegistryError("%s schema/status mismatch" % label)
    if doc["measurement_class"] != ALLOWED_MEASUREMENT_CLASS:
        raise RegistryError("%s is not DIRECT_UNIFIED" % label)
    _sha(doc["resource_manifest_sha256"], "%s resource manifest" % label)
    for identity in ("population_id", "workload_id"):
        if not isinstance(doc[identity], str) or not doc[identity]:
            raise RegistryError("%s missing %s" % (label, identity))
    row_keys = {"row_id", "role", "fidelity", "cycles", "energy_mj", "area_mm2", "accuracy"}
    rows = doc["rows"]
    if not isinstance(rows, list) or [row.get("row_id") for row in rows] != [item[0] for item in MANDATORY_ROW_SPECS]:
        raise RegistryError("%s rows do not match mandatory ladder" % label)
    specs = _mandatory_specs_dict()
    for row in rows:
        _require_exact_fields(row, row_keys, "%s result row" % label)
        if (row["role"], row["fidelity"]) != specs[row["row_id"]]:
            raise RegistryError("%s result role/fidelity mismatch" % label)
        for field in ("cycles", "energy_mj", "area_mm2", "accuracy"):
            _positive_finite(row[field], "%s %s %s" % (label, row["row_id"], field))
    if not isinstance(doc["samples"], list) or not doc["samples"]:
        raise RegistryError("%s requires raw sample measurements" % label)
    mandatory_ids = [item[0] for item in MANDATORY_ROW_SPECS]
    sample_keys = {"sample_id", "sequence_id", "density_stratum", "view", "row_cycles"}
    sample_ids = []
    samples_by_view = {view: [] for view in VIEW_NAMES}
    for sample in doc["samples"]:
        _require_exact_fields(sample, sample_keys, "%s sample" % label)
        for field in ("sample_id", "sequence_id"):
            if not isinstance(sample[field], str) or not sample[field]:
                raise RegistryError("%s sample missing %s" % (label, field))
        if sample["density_stratum"] not in ("low", "mid", "high"):
            raise RegistryError("%s sample density stratum invalid" % label)
        if sample["view"] not in VIEW_NAMES:
            raise RegistryError("%s sample view invalid" % label)
        if not isinstance(sample["row_cycles"], dict) or list(sample["row_cycles"]) != mandatory_ids:
            raise RegistryError("%s sample row_cycles do not match ordered ladder" % label)
        for row_id, cycles in sample["row_cycles"].items():
            _positive_finite(cycles, "%s sample %s cycles" % (label, row_id))
        sample_ids.append(sample["sample_id"])
        samples_by_view[sample["view"]].append(sample)
    if len(sample_ids) != len(set(sample_ids)):
        raise RegistryError("%s duplicate sample_id" % label)
    if any(not samples_by_view[view] for view in VIEW_NAMES):
        raise RegistryError("%s must contain raw samples for both required views" % label)

    def recompute_view(view_samples):
        ratios = [
            float(sample["row_cycles"][FIXED_NUMERATOR_ROW_ID])
            / float(sample["row_cycles"][CANDIDATE_ROW_ID])
            for sample in view_samples
        ]
        return {
            "arithmetic_mean": math.fsum(ratios) / len(ratios),
            "geometric_mean": math.exp(math.fsum(math.log(value) for value in ratios) / len(ratios)),
            "ratio_of_sums": math.fsum(float(sample["row_cycles"][FIXED_NUMERATOR_ROW_ID]) for sample in view_samples)
            / math.fsum(float(sample["row_cycles"][CANDIDATE_ROW_ID]) for sample in view_samples),
            "minimum": min(ratios),
            "maximum": max(ratios),
        }

    def validate_aggregate_map(actual, expected, aggregate_label):
        if not isinstance(actual, dict) or list(actual) != list(AGGREGATE_NAMES):
            raise RegistryError("%s aggregate values/order missing" % aggregate_label)
        for name in AGGREGATE_NAMES:
            _positive_finite(actual[name], "%s %s" % (aggregate_label, name))
            if not math.isclose(float(actual[name]), expected[name], rel_tol=1e-12, abs_tol=1e-12):
                raise RegistryError("%s %s does not recompute from raw samples" % (aggregate_label, name))

    if not isinstance(doc["views"], dict) or list(doc["views"]) != list(VIEW_NAMES):
        raise RegistryError("%s required views/order missing" % label)
    recomputed_views = {}
    for view in VIEW_NAMES:
        recomputed_views[view] = recompute_view(samples_by_view[view])
        validate_aggregate_map(doc["views"][view], recomputed_views[view], "%s view %s" % (label, view))
    validate_aggregate_map(doc["aggregates"], recomputed_views["iso_service"], "%s default aggregates" % label)
    if doc["aggregates"] != doc["views"]["iso_service"]:
        raise RegistryError("%s default aggregates must equal iso_service values" % label)

    # Table-A cycles are the direct ratio-of-sums operands for the default
    # iso-service view, not products of local speedup ratios.
    summary = {row["row_id"]: row for row in rows}
    for row_id in mandatory_ids:
        expected_cycles = math.fsum(float(sample["row_cycles"][row_id]) for sample in samples_by_view["iso_service"])
        if not math.isclose(float(summary[row_id]["cycles"]), expected_cycles, rel_tol=0.0, abs_tol=1e-9):
            raise RegistryError("%s summary cycles for %s do not sum raw iso_service samples" % (label, row_id))
    return summary, sample_ids, sorted(set(sample["sequence_id"] for sample in doc["samples"])), sorted(
        set(sample["density_stratum"] for sample in doc["samples"])
    )


def _validate_evidence_bundle(bundle_id, spec):
    if not isinstance(bundle_id, str) or not bundle_id.startswith("direct_unified_"):
        raise RegistryError("Table A evidence bundle IDs require direct_unified_ prefix")
    if not isinstance(spec, dict) or set(spec) != set(BUNDLE_ARTIFACTS):
        raise RegistryError("bundle %s must bind exactly five dedicated artifacts" % bundle_id)
    docs = {}
    hashes = {}
    paths = set()
    for name, prefix in BUNDLE_ARTIFACTS.items():
        path, doc, actual = _validate_file_spec(spec[name], "bundle %s %s" % (bundle_id, name), prefix)
        if path in paths:
            raise RegistryError("bundle %s reuses one file for multiple authorities" % bundle_id)
        paths.add(path)
        docs[name] = doc
        hashes[name] = actual

    result = docs["direct_result"]
    result_rows, result_sample_ids, result_sequence_ids, result_density_strata = _validate_direct_result(
        result, "bundle %s direct result" % bundle_id
    )
    if result["resource_manifest_sha256"] != hashes["resource_manifest"]:
        raise RegistryError("bundle %s direct result resource SHA mismatch" % bundle_id)

    resource = docs["resource_manifest"]
    _require_exact_fields(resource, {"schema", "status", "population_id", "workload_id", "resource_tuple"}, "resource manifest")
    if resource["schema"] != "m635.h67.common_resource_manifest.r1" or resource["status"] != "FROZEN":
        raise RegistryError("bundle %s resource manifest schema/status mismatch" % bundle_id)
    if resource["population_id"] != result["population_id"] or resource["workload_id"] != result["workload_id"]:
        raise RegistryError("bundle %s resource identities mismatch" % bundle_id)
    if not isinstance(resource["resource_tuple"], dict) or not resource["resource_tuple"]:
        raise RegistryError("bundle %s empty resource tuple" % bundle_id)

    completion = docs["completion_receipt"]
    completion_keys = {
        "schema", "status", "direct_result_sha256", "resource_manifest_sha256",
        "population_id", "workload_id", "completed_row_ids", "closures",
    }
    _require_exact_fields(completion, completion_keys, "completion receipt")
    if completion["schema"] != "m635.h67.direct_unified.completion_receipt.r1" or completion["status"] != "PASS_COMPLETE":
        raise RegistryError("bundle %s completion receipt schema/status mismatch" % bundle_id)
    if completion["direct_result_sha256"] != hashes["direct_result"] or completion["resource_manifest_sha256"] != hashes["resource_manifest"]:
        raise RegistryError("bundle %s completion receipt hash binding mismatch" % bundle_id)
    if completion["population_id"] != result["population_id"] or completion["workload_id"] != result["workload_id"]:
        raise RegistryError("bundle %s completion identity mismatch" % bundle_id)
    if completion["completed_row_ids"] != [row[0] for row in MANDATORY_ROW_SPECS]:
        raise RegistryError("bundle %s completion does not cover mandatory ladder" % bundle_id)
    closures = completion["closures"]
    if not isinstance(closures, dict) or set(closures) != set(REQUIRED_CLOSURES) or not all(closures.values()):
        raise RegistryError("bundle %s completion closures are not all proven" % bundle_id)

    coverage = docs["coverage_receipt"]
    coverage_keys = {
        "schema", "status", "direct_result_sha256", "population_id", "workload_id",
        "sample_ids", "sequence_receipts", "density_preregistration_receipt", "aggregates", "views",
    }
    _require_exact_fields(coverage, coverage_keys, "coverage receipt")
    if coverage["schema"] != "m635.h67.coverage_receipt.r1" or coverage["status"] != "PASS":
        raise RegistryError("bundle %s coverage receipt schema/status mismatch" % bundle_id)
    if coverage["direct_result_sha256"] != hashes["direct_result"]:
        raise RegistryError("bundle %s coverage result binding mismatch" % bundle_id)
    if coverage["population_id"] != result["population_id"] or coverage["workload_id"] != result["workload_id"]:
        raise RegistryError("bundle %s coverage identity mismatch" % bundle_id)
    if coverage["aggregates"] != result["aggregates"] or coverage["views"] != result["views"]:
        raise RegistryError("bundle %s aggregate/view values are not evidence-bound" % bundle_id)
    if coverage["sample_ids"] != result_sample_ids:
        raise RegistryError("bundle %s coverage sample IDs differ from direct result" % bundle_id)
    seq_specs = coverage["sequence_receipts"]
    sequence_ids = []
    if not isinstance(seq_specs, list):
        raise RegistryError("bundle %s sequence receipts must be a list" % bundle_id)
    for index, seq_spec in enumerate(seq_specs):
        _, seq_doc, _ = _validate_file_spec(seq_spec, "bundle %s sequence receipt %d" % (bundle_id, index), "hw_autoresearch_nts07/results/")
        _require_exact_fields(seq_doc, {"schema", "status", "sequence_id", "population_id", "workload_id", "direct_result_sha256"}, "sequence receipt")
        if seq_doc["schema"] != "m635.h67.sequence_completion_receipt.r1" or seq_doc["status"] != "PASS":
            raise RegistryError("bundle %s invalid sequence receipt" % bundle_id)
        if seq_doc["population_id"] != result["population_id"] or seq_doc["workload_id"] != result["workload_id"] or seq_doc["direct_result_sha256"] != hashes["direct_result"]:
            raise RegistryError("bundle %s sequence receipt binding mismatch" % bundle_id)
        sequence_ids.append(seq_doc["sequence_id"])
    density_spec = coverage["density_preregistration_receipt"]
    density_strata = []
    if density_spec is not None:
        _, density_doc, _ = _validate_file_spec(density_spec, "bundle %s density preregistration" % bundle_id, "hw_autoresearch_nts07/contracts/")
        _require_exact_fields(density_doc, {"schema", "status", "population_id", "workload_id", "strata", "frozen_before_results"}, "density preregistration")
        if density_doc["schema"] != "m635.h67.density_preregistration.r1" or density_doc["status"] != "FROZEN" or density_doc["frozen_before_results"] is not True:
            raise RegistryError("bundle %s density preregistration invalid" % bundle_id)
        if density_doc["population_id"] != result["population_id"] or density_doc["workload_id"] != result["workload_id"]:
            raise RegistryError("bundle %s density identity mismatch" % bundle_id)
        density_strata = density_doc["strata"]
    if sorted(set(sequence_ids)) != result_sequence_ids:
        raise RegistryError("bundle %s sequence receipts do not cover result sequences exactly" % bundle_id)
    if density_strata and sorted(set(density_strata)) != result_density_strata:
        raise RegistryError("bundle %s density preregistration does not match result strata" % bundle_id)
    if len(set(sequence_ids)) < 3 and not {"low", "mid", "high"}.issubset(set(density_strata)):
        raise RegistryError("bundle %s lacks bound population coverage" % bundle_id)

    hammer = docs["independent_hammer_receipt"]
    hammer_keys = {
        "schema", "status", "direct_result_sha256", "completion_receipt_sha256",
        "coverage_receipt_sha256", "resource_manifest_sha256", "severity_counts",
        "independence", "recomputed_rows", "recomputed_aggregates", "recomputed_views", "authorization",
    }
    _require_exact_fields(hammer, hammer_keys, "independent hammer receipt")
    if hammer["schema"] != "m635.h67.direct_unified.independent_hammer.r1" or hammer["status"] != "PASS":
        raise RegistryError("bundle %s independent hammer schema/status mismatch" % bundle_id)
    for field, expected in (
        ("direct_result_sha256", hashes["direct_result"]),
        ("completion_receipt_sha256", hashes["completion_receipt"]),
        ("coverage_receipt_sha256", hashes["coverage_receipt"]),
        ("resource_manifest_sha256", hashes["resource_manifest"]),
    ):
        if hammer[field] != expected:
            raise RegistryError("bundle %s independent hammer %s mismatch" % (bundle_id, field))
    if hammer["severity_counts"] != {"P0": 0, "P1": 0}:
        raise RegistryError("bundle %s hammer has blocking findings" % bundle_id)
    if hammer["independence"] != {"author_receipt_used_as_authority": False, "raw_evidence_recomputed": True, "result_modified": False}:
        raise RegistryError("bundle %s hammer independence contract failed" % bundle_id)
    if hammer["authorization"] != {"table_a_methodology_admitted": True, "direct_unified_measurement_admitted": True}:
        raise RegistryError("bundle %s hammer authorization missing" % bundle_id)
    if (
        hammer["recomputed_rows"] != result["rows"]
        or hammer["recomputed_aggregates"] != result["aggregates"]
        or hammer["recomputed_views"] != result["views"]
    ):
        raise RegistryError("bundle %s hammer recomputation differs from direct result" % bundle_id)

    return {
        "hashes": hashes,
        "result": result,
        "rows": result_rows,
        "closures": closures,
        "sequence_ids": sequence_ids,
        "density_strata": density_strata,
    }


def validate_bundles(config):
    bundles = config.get("table_a_evidence_bundles")
    if not isinstance(bundles, dict):
        raise RegistryError("table_a_evidence_bundles must be an object")
    return {bundle_id: _validate_evidence_bundle(bundle_id, spec) for bundle_id, spec in bundles.items()}


def evaluate_headline_gate(config, rows, source_ids, bundles, policy):
    eligible = []
    failures = {}
    for row in rows:
        reasons = list(row["blockers"])
        bundle_id = row["source_id"]
        if bundle_id is None:
            reasons.append("missing_direct_unified_evidence_bundle")
        elif bundle_id in source_ids:
            reasons.append("table_b_or_c_source_structurally_forbidden")
        elif bundle_id not in bundles:
            reasons.append("unknown_direct_unified_evidence_bundle")
        else:
            bundle = bundles[bundle_id]
            if row["row_id"] not in bundle["rows"]:
                reasons.append("row_absent_from_bound_direct_result")
            else:
                evidence = bundle["rows"][row["row_id"]]
                exact_projection = {
                    "cycles": evidence["cycles"], "energy_mj": evidence["energy_mj"],
                    "area_mm2": evidence["area_mm2"], "accuracy": evidence["accuracy"],
                    "measurement_class": ALLOWED_MEASUREMENT_CLASS,
                    "population_id": bundle["result"]["population_id"],
                    "workload_id": bundle["result"]["workload_id"],
                    "resource_manifest_sha256": bundle["hashes"]["resource_manifest"],
                    "completion_receipt_sha256": bundle["hashes"]["completion_receipt"],
                    "decoder_complete": bundle["closures"]["decoder_complete"],
                    "memory_timing_included": bundle["closures"]["memory_timing_included"],
                    "full_network_completion": bundle["closures"]["full_network_completion"],
                    "logic_sram_dram_energy_closed": bundle["closures"]["logic_sram_dram_energy_closed"],
                    "logic_macro_area_closed": bundle["closures"]["logic_macro_area_closed"],
                    "sta_closed": bundle["closures"]["sta_closed"],
                    "independent_hammer_pass": True,
                }
                for field, expected in exact_projection.items():
                    if row[field] != expected:
                        reasons.append("%s_not_exact_evidence_projection" % field)
                if row["blockers"]:
                    reasons.append("blockers_nonempty_despite_bundle")
        if reasons:
            failures[row["row_id"]] = sorted(set(reasons))
        else:
            eligible.append(row["row_id"])

    mandatory_ids = [row[0] for row in MANDATORY_ROW_SPECS]
    mandatory_rows = {row["row_id"]: row for row in rows if row["row_id"] in mandatory_ids}
    bundle_ids = {row["source_id"] for row in mandatory_rows.values()}
    global_failures = []
    if len(bundle_ids) != 1 or None in bundle_ids:
        global_failures.append("mandatory_rows_not_bound_to_one_common_run_bundle")
    all_required_eligible = set(mandatory_ids).issubset(set(eligible))
    numerator = mandatory_rows[FIXED_NUMERATOR_ROW_ID]["cycles"]
    candidate = mandatory_rows[CANDIDATE_ROW_ID]["cycles"]
    direct_speedup = None
    if isinstance(numerator, (int, float)) and isinstance(candidate, (int, float)) and not isinstance(numerator, bool) and not isinstance(candidate, bool) and candidate > 0:
        direct_speedup = numerator / candidate
        if direct_speedup < policy["minimum_direct_speedup_for_accept"]:
            global_failures.append("direct_speedup_below_accept_floor")
    else:
        global_failures.append("direct_speedup_unavailable")
    admitted = all_required_eligible and not global_failures
    claim = config.get("claim_boundary", {})
    if claim.get("table_a_admitted_rows") != len(eligible):
        raise RegistryError("claim_boundary table_a_admitted_rows disagrees with evidence gate")
    if claim.get("paper_headline_admitted") is not admitted:
        raise RegistryError("claim_boundary paper_headline_admitted disagrees with evidence gate")
    return {
        "admitted": admitted,
        "eligible_row_ids": eligible,
        "eligible_row_count": len(eligible),
        "required_row_count": len(mandatory_ids),
        "row_failures": failures,
        "global_failures": sorted(set(global_failures)),
        "direct_speedup": direct_speedup,
        "fixed_numerator_row_id": FIXED_NUMERATOR_ROW_ID,
        "strongest_same_page_baseline_row_id": STRONGEST_BASELINE_ROW_ID,
        "candidate_row_id": CANDIDATE_ROW_ID,
    }


def recompute_analytical(config):
    item = config.get("analytical_diagnostic")
    if not isinstance(item, dict) or item.get("admitted") is not False:
        raise RegistryError("analytical diagnostic must remain non-admitted")
    getcontext().prec = 50
    numerator = Decimal(str(item["fixed_numerator_cycles"]))
    low_cycles = Decimal(str(item["candidate_cycles_low"]))
    high_cycles = Decimal(str(item["candidate_cycles_high"]))
    low = numerator / high_cycles
    high = numerator / low_cycles
    if abs(low - Decimal(item["expected_speedup_low"])) > Decimal("1e-27") or abs(high - Decimal(item["expected_speedup_high"])) > Decimal("1e-27"):
        raise RegistryError("analytical anchors do not independently recompute")
    return {"speedup_low": format(low, ".28f"), "speedup_high": format(high, ".28f"), "admitted": False}


def build(config_path=DEFAULT_CONFIG):
    config = load_json(config_path, "registry config")
    if config.get("schema") != "m635.h67.paper_metric_registry.r3":
        raise RegistryError("unexpected registry schema")
    sources, source_docs = validate_sources(config)
    rows = validate_ladder_and_tables(config, set(sources))
    policy = validate_policy(config)
    validate_m518_binding(config, source_docs)
    bundles = validate_bundles(config)
    analytical = recompute_analytical(config)
    gate = evaluate_headline_gate(config, rows, set(sources), bundles, policy)
    protected = config.get("protected_file")
    if not isinstance(protected, dict) or set(protected) != {"path", "sha256"}:
        raise RegistryError("protected_file binding missing")
    if sha256_file(secure_repo_file(protected["path"])) != protected["sha256"]:
        raise RegistryError("protected docs359 SHA mismatch")
    if config.get("claim_boundary", {}).get("eda_or_gpu_run") is not False:
        raise RegistryError("M635 must remain a no-EDA/no-GPU package")
    return {
        "schema": "m635.h67.paper_metric_registry.r3.preview",
        "status": config["status"],
        "source_hashes_validated": sources,
        "table_a_evidence_bundle_count": len(bundles),
        "table_a": rows,
        "table_b": config["table_b_schema"]["rows"],
        "table_c": config["table_c_schema"]["rows"],
        "analytical_diagnostic": analytical,
        "headline_gate": gate,
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
        print("M635_REGISTRY_FAIL: %s" % exc)
        return 2
    if args.emit_json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(
            "M635_REGISTRY_PASS sources=%d bundles=%d table_a_eligible=%d headline_admitted=%s analytical_admitted=%s"
            % (
                len(result["source_hashes_validated"]),
                result["table_a_evidence_bundle_count"],
                result["headline_gate"]["eligible_row_count"],
                str(result["headline_gate"]["admitted"]).lower(),
                str(result["analytical_diagnostic"]["admitted"]).lower(),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
