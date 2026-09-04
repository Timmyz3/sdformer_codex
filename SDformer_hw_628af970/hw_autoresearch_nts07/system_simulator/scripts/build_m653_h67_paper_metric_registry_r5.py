#!/usr/bin/env python3
"""Build the M653 H67 paper-metric registry r5, fail closed.

This is a non-mutating methodological successor to sealed M645 r4.  It keeps
the canonical authority map empty and adds the four roots required by M651:
the actual frozen M527/checkpoint files, a fixed-throughput-numerator receipt,
strict request/review target projections, and numeric raw-PPA projection.

No GPU, EDA, capture, production simulator, or paper task is launched.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = REPO_ROOT / "hw_autoresearch_nts07"
R4_BUILDER = HW_ROOT / "system_simulator/scripts/build_m645_h67_paper_metric_registry_r4.py"
R4_BUILDER_SHA256 = "3ae5e996902d22033527b867011933eec190f7c98a7c008f9cbb787032d5605a"
DEFAULT_CONFIG = HW_ROOT / "system_simulator/config/m653_h67_paper_metric_registry_r5_20260828.json"
REGISTRY_TESTS = HW_ROOT / "system_simulator/tests/test_m653_h67_paper_metric_registry_r5.py"
REGISTRY_CONTRACT = HW_ROOT / "contracts/m653_h67_paper_metric_registry_r5_contract_r1_20260828.json"
CHECKPOINT = (HW_ROOT / "system_handoff/received/h67_ep35_system_trace_handoff_20260821/"
              "h67_ep35_system_trace_handoff_20260821/checkpoint/checkpoint_epoch35.pth")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_r4():
    if _sha256(R4_BUILDER) != R4_BUILDER_SHA256:
        raise RuntimeError("sealed M645 r4 builder SHA drift")
    spec = importlib.util.spec_from_file_location("m653_sealed_m645_r4", str(R4_BUILDER))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import sealed M645 r4 builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R4 = _load_r4()
M635 = R4.M635
RegistryError = R4.RegistryError
M635_BUILDER = R4.M635_BUILDER
M635_BUILDER_SHA256 = R4.M635_BUILDER_SHA256
M635_CONFIG = R4.M635_CONFIG
M635_CONFIG_SHA256 = R4.M635_CONFIG_SHA256
M527_CONTRACT = R4.M527_CONTRACT
M527_CONTRACT_SHA256 = R4.M527_CONTRACT_SHA256
DOCS359 = R4.DOCS359
DOCS359_SHA256 = R4.DOCS359_SHA256
CHECKPOINT_SHA256 = R4.CHECKPOINT_SHA256
ROW_TO_M527_CONFIGURATION = R4.ROW_TO_M527_CONFIGURATION
MANDATORY_ROW_IDS = R4.MANDATORY_ROW_IDS
VIEW_NAMES = R4.VIEW_NAMES
AGGREGATE_NAMES = R4.AGGREGATE_NAMES
RESOURCE_FIELDS = R4.RESOURCE_FIELDS
CHARGE_FIELDS = R4.CHARGE_FIELDS
FALLBACK_FIELDS = R4.FALLBACK_FIELDS
MECHANISM_FIELDS = R4.MECHANISM_FIELDS
EXPECTED_MECHANISMS = R4.EXPECTED_MECHANISMS
FILE_SPEC_FIELDS = R4.FILE_SPEC_FIELDS

TRUSTED_HAMMER_AUTHORITIES = {}

BUNDLE_FIELDS = set(R4.BUNDLE_FIELDS) | {
    "m527_contract", "checkpoint", "fixed_throughput_numerator_receipt",
}
NUMERATOR_FIELDS = {
    "schema", "status", "m527_contract_sha256", "checkpoint_sha256",
    "measurement_identity_sha256", "complete_trace_manifest_sha256",
    "sequence_population_manifest_sha256", "aggregation_weight_manifest_sha256",
    "population_scalar", "population_unit", "frame_definition", "op_convention",
    "included_operator_scope", "excluded_operator_scope_with_reason",
    "excluded_numerator_ops_cycles_energy_traffic_charged",
    "dense_equivalent_ops_scalar", "dense_equivalent_ops_unit",
    "original_useful_nonzero_ops_scalar", "original_useful_nonzero_ops_unit",
    "configuration_ids",
}
OP_CONVENTION_FIELDS = {
    "multiply_ops", "add_ops", "mac_ops", "comparison_ops",
    "state_update_ops", "normalization_ops", "address_and_control_ops",
}
REQUEST_TARGET_FIELDS = {
    "registry_builder", "registry_config", "registry_tests", "registry_contract",
    "m527_contract", "checkpoint", "direct_result", "fixed_throughput_numerator_receipt",
}


def _exact(value, fields, label):
    return R4._exact(value, fields, label)


def _sha(value, label):
    return R4._sha(value, label)


def _string(value, label):
    return R4._string(value, label)


def _number(value, label, zero_ok=False):
    return R4._number(value, label, zero_ok)


def _integer(value, label, zero_ok=False):
    return R4._integer(value, label, zero_ok)


def _secure_file(relative, prefix=None):
    return R4._secure_file(relative, prefix)


def _file_spec(spec, label, prefix=None, media_types=("application/json",)):
    return R4._file_spec(spec, label, prefix, media_types)


def _runtime_m527_contract():
    if _sha256(M527_CONTRACT) != M527_CONTRACT_SHA256:
        raise RegistryError("actual frozen M527 contract SHA drift")
    doc = M635.load_json(M527_CONTRACT, "actual frozen M527 contract")
    if doc.get("schema") != "m527_h67_headline_baseline_ladder_contract_v3":
        raise RegistryError("actual frozen M527 contract schema mismatch")
    gates = doc.get("headline_policy", {}).get("independent_required_gates")
    required = {
        "measurement_identity.admission_gate.current_value",
        "fixed_throughput_numerators.admission_gate.current_value",
        "configuration_registry.admission_gate.current_value",
    }
    if set(gates or []) != required or doc.get("headline_policy", {}).get("all_independent_required_gates_must_pass") is not True:
        raise RegistryError("actual M527 independent headline-gate contract drift")
    numerator = doc.get("fixed_throughput_numerators", {})
    if (numerator.get("required_before_headline") is not True or
            numerator.get("receipt_schema", {}).get("schema_exact") !=
            "m527_h67_fixed_throughput_numerator_receipt_v1"):
        raise RegistryError("actual M527 fixed-numerator contract drift")
    return doc


def _validate_exact_root_spec(spec, path, expected_sha, label, media_types):
    resolved, _, digest = _file_spec(spec, label, media_types=media_types)
    if resolved != path or digest != expected_sha:
        raise RegistryError(label + " does not bind the exact repo-local frozen file")
    return digest


def _validate_measurement_identity(spec, checkpoint_sha):
    _, doc, digest = _file_spec(spec, "measurement identity", "hw_autoresearch_nts07/system_simulator/")
    fields = {
        "schema", "status", "m527_contract_sha256", "checkpoint_sha256",
        "complete_trace_manifest", "sequence_population_manifest",
        "aggregation_weight_manifest", "frame_definition", "density_metric",
        "density_bin_boundaries", "operator_ids",
    }
    _exact(doc, fields, "measurement identity")
    if doc["schema"] != "m645.h67.measurement_identity.r1" or doc["status"] != "FROZEN_COMPLETE":
        raise RegistryError("measurement identity schema/status mismatch")
    if doc["m527_contract_sha256"] != M527_CONTRACT_SHA256 or doc["checkpoint_sha256"] != checkpoint_sha:
        raise RegistryError("measurement identity root mismatch")
    manifests = {}
    schemas = {
        "complete_trace_manifest": "m645.h67.complete_trace_manifest.r1",
        "sequence_population_manifest": "m645.h67.sequence_population_manifest.r1",
        "aggregation_weight_manifest": "m645.h67.aggregation_weight_manifest.r1",
    }
    for name, schema in schemas.items():
        path, child, child_sha = _file_spec(doc[name], name, "hw_autoresearch_nts07/system_simulator/")
        if child.get("schema") != schema or child.get("status") != "FROZEN_COMPLETE":
            raise RegistryError(name + " schema/status mismatch")
        if child.get("checkpoint_sha256") != checkpoint_sha:
            raise RegistryError(name + " checkpoint mismatch")
        manifests[name] = {"path": path, "doc": child, "sha256": child_sha}
    trace = manifests["complete_trace_manifest"]["doc"]
    _exact(trace, {"schema", "status", "checkpoint_sha256", "decoder_complete",
                   "operator_scope", "record_count", "trace_members"}, "complete trace manifest")
    if trace["decoder_complete"] is not True or "ConvTranspose2d" not in trace["operator_scope"]:
        raise RegistryError("complete trace is not decoder-complete")
    _integer(trace["record_count"], "complete trace record_count")
    if not isinstance(trace["trace_members"], list) or not trace["trace_members"]:
        raise RegistryError("complete trace member list empty")
    for index, member in enumerate(trace["trace_members"]):
        _file_spec(member, "trace member %d" % index, "hw_autoresearch_nts07/")
    population = manifests["sequence_population_manifest"]["doc"]
    _exact(population, {"schema", "status", "checkpoint_sha256", "population_id", "samples"},
           "population manifest")
    if not isinstance(population["samples"], list) or not population["samples"]:
        raise RegistryError("population samples empty")
    population_samples = {}
    for sample in population["samples"]:
        _exact(sample, {"sample_id", "sequence_id", "density_stratum", "frame_count"}, "population sample")
        sample_id = _string(sample["sample_id"], "sample_id")
        _string(sample["sequence_id"], "sequence_id")
        if sample["density_stratum"] not in ("low", "medium", "high"):
            raise RegistryError("population density stratum must use M527 low/medium/high vocabulary")
        _integer(sample["frame_count"], "frame_count")
        if sample_id in population_samples:
            raise RegistryError("duplicate population sample")
        population_samples[sample_id] = sample
    weights = manifests["aggregation_weight_manifest"]["doc"]
    _exact(weights, {"schema", "status", "checkpoint_sha256", "population_id",
                     "selection_frozen_before_results", "samples"}, "aggregation manifest")
    if weights["population_id"] != population["population_id"] or weights["selection_frozen_before_results"] is not True:
        raise RegistryError("aggregation/population identity mismatch or post-selected")
    aggregation = {}
    for sample in weights["samples"]:
        _exact(sample, {"sample_id", "view", "weight"}, "aggregation sample")
        key = (_string(sample["sample_id"], "aggregation sample_id"), sample["view"])
        if key[0] not in population_samples or key[1] not in VIEW_NAMES or key in aggregation:
            raise RegistryError("aggregation sample is invalid/duplicated")
        aggregation[key] = _number(sample["weight"], "aggregation weight", zero_ok=True)
    expected_keys = {(sample_id, view) for sample_id in population_samples for view in VIEW_NAMES}
    if set(aggregation) != expected_keys:
        raise RegistryError("aggregation does not cover every population sample/view")
    for view in VIEW_NAMES:
        if not math.isclose(math.fsum(aggregation[(sample_id, view)] for sample_id in population_samples),
                            1.0, rel_tol=0.0, abs_tol=1e-12):
            raise RegistryError("aggregation weights do not sum to one")
    return {"sha256": digest, "doc": doc, "manifests": manifests,
            "population_id": population["population_id"], "samples": population_samples,
            "weights": aggregation}


def _validate_numerator_receipt(spec, measurement):
    _, receipt, digest = _file_spec(spec, "M527 fixed-throughput numerator receipt",
                                    "hw_autoresearch_nts07/results/")
    _exact(receipt, NUMERATOR_FIELDS, "M527 fixed-throughput numerator receipt")
    if receipt["schema"] != "m527_h67_fixed_throughput_numerator_receipt_v1" or receipt["status"] != "PASS_RECOMPUTED_FIXED_POPULATION":
        raise RegistryError("M527 fixed-numerator receipt schema/status mismatch")
    expected_roots = (
        M527_CONTRACT_SHA256, CHECKPOINT_SHA256, measurement["sha256"],
        measurement["manifests"]["complete_trace_manifest"]["sha256"],
        measurement["manifests"]["sequence_population_manifest"]["sha256"],
        measurement["manifests"]["aggregation_weight_manifest"]["sha256"],
    )
    actual_roots = (
        receipt["m527_contract_sha256"], receipt["checkpoint_sha256"],
        receipt["measurement_identity_sha256"], receipt["complete_trace_manifest_sha256"],
        receipt["sequence_population_manifest_sha256"],
        receipt["aggregation_weight_manifest_sha256"],
    )
    if actual_roots != expected_roots:
        raise RegistryError("M527 fixed-numerator identity roots mismatch")
    population_scalar = sum(item["frame_count"] for item in measurement["samples"].values())
    if (receipt["population_scalar"] != population_scalar or
            receipt["population_unit"] != "frozen_frames_across_frozen_sequence_population" or
            receipt["frame_definition"] != measurement["doc"]["frame_definition"]):
        raise RegistryError("M527 fixed-numerator population/frame projection mismatch")
    _exact(receipt["op_convention"], OP_CONVENTION_FIELDS, "M527 op convention")
    for key in OP_CONVENTION_FIELDS:
        if isinstance(receipt["op_convention"][key], bool) or receipt["op_convention"][key] not in (0, 1, 2):
            raise RegistryError("M527 op convention values must be 0/1/2")
    if receipt["op_convention"]["mac_ops"] != (receipt["op_convention"]["multiply_ops"] +
                                                   receipt["op_convention"]["add_ops"]):
        raise RegistryError("M527 mac_ops must equal multiply_ops plus add_ops")
    included = receipt["included_operator_scope"]
    excluded = receipt["excluded_operator_scope_with_reason"]
    if (not isinstance(included, list) or len(included) != len(set(included)) or
            not isinstance(excluded, dict) or any(not isinstance(reason, str) or not reason.strip()
                                                 for reason in excluded.values()) or
            set(included) & set(excluded) or
            set(included) | set(excluded) != set(measurement["doc"]["operator_ids"])):
        raise RegistryError("M527 numerator operator scope partition mismatch")
    if receipt["excluded_numerator_ops_cycles_energy_traffic_charged"] is not True:
        raise RegistryError("excluded numerator work must still charge cycles/energy/traffic")
    dense = _integer(receipt["dense_equivalent_ops_scalar"], "dense-equivalent ops")
    useful = _integer(receipt["original_useful_nonzero_ops_scalar"], "original useful nonzero ops")
    if useful > dense:
        raise RegistryError("original useful nonzero ops cannot exceed dense-equivalent ops")
    if (receipt["dense_equivalent_ops_unit"] != "ops_per_frozen_population" or
            receipt["original_useful_nonzero_ops_unit"] != "ops_per_frozen_population"):
        raise RegistryError("M527 fixed-numerator unit mismatch")
    if receipt["configuration_ids"] != [ROW_TO_M527_CONFIGURATION[row] for row in MANDATORY_ROW_IDS]:
        raise RegistryError("M527 fixed numerator is not shared by every Table-A configuration")
    return {"sha256": digest, "doc": receipt,
            "dense_equivalent_ops_scalar": dense,
            "original_useful_nonzero_ops_scalar": useful}


_NUMERIC_LINE = re.compile(r"^([a-z][a-z0-9_]*)[ \t]+([-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?)$")


def _parse_projection_report(spec, label, required_keys):
    path, document, digest = _file_spec(spec, label, "hw_autoresearch_nts07/results/",
                                        ("application/json", "text/plain"))
    if document is not None:
        values = document
    else:
        values = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            match = _NUMERIC_LINE.match(line.strip())
            if match is None or match.group(1) in values:
                raise RegistryError(label + " is not a strict machine-readable numeric report")
            values[match.group(1)] = float(match.group(2))
    _exact(values, required_keys, label + " numeric projection")
    projected = {}
    for key in required_keys:
        projected[key] = _number(values[key], label + " " + key, zero_ok=True)
    return projected, digest


def _validate_typed_receipts(bundle_id, specs, raw, configs, summaries, measurement):
    # Reuse M645's exact raw-derived energy and accuracy arithmetic by validating
    # temporary PPA-free copies locally; PPA itself is handled below.
    _, energy, energy_sha = _file_spec(specs["energy_receipt"], "typed energy receipt",
                                       "hw_autoresearch_nts07/results/")
    _exact(energy, {"schema", "status", "raw_run_index_sha256", "rows"}, "typed energy receipt")
    if energy["schema"] != "m645.h67.logic_sram_dram_energy_receipt.r1" or energy["status"] != "PASS_TYPED" or energy["raw_run_index_sha256"] != raw["sha256"]:
        raise RegistryError("typed energy receipt schema/status/root mismatch")
    energy_rows = {}
    for row in energy["rows"]:
        _exact(row, {"row_id", "configuration_manifest_sha256", "logic_energy_mj",
                     "sram_energy_mj", "dram_energy_mj", "total_energy_mj"}, "energy row")
        row_id = row["row_id"]
        if row_id not in MANDATORY_ROW_IDS or row_id in energy_rows or row["configuration_manifest_sha256"] != configs[row_id]["sha256"]:
            raise RegistryError("typed energy row identity mismatch")
        components = []
        for component in ("logic_energy_mj", "sram_energy_mj", "dram_energy_mj"):
            value = _number(row[component], component, zero_ok=True)
            if not math.isclose(value, summaries[row_id][component], rel_tol=1e-12, abs_tol=1e-15):
                raise RegistryError("typed energy component does not recompute from rooted raw logs")
            components.append(value)
        if (not math.isclose(row["total_energy_mj"], math.fsum(components), rel_tol=1e-12, abs_tol=1e-15) or
                not math.isclose(row["total_energy_mj"], summaries[row_id]["energy_mj"], rel_tol=1e-12, abs_tol=1e-15)):
            raise RegistryError("typed energy total does not recompute from rooted raw logs")
        energy_rows[row_id] = row
    if set(energy_rows) != set(MANDATORY_ROW_IDS):
        raise RegistryError("typed energy receipt lacks six rows")

    _, accuracy, accuracy_sha = _file_spec(specs["accuracy_receipt"], "typed accuracy receipt",
                                           "hw_autoresearch_nts07/results/")
    _exact(accuracy, {"schema", "status", "raw_run_index_sha256", "checkpoint_sha256",
                      "population_manifest_sha256", "rows"}, "typed accuracy receipt")
    if (accuracy["schema"] != "m645.h67.accuracy_receipt.r1" or accuracy["status"] != "PASS_TYPED" or
            accuracy["raw_run_index_sha256"] != raw["sha256"] or accuracy["checkpoint_sha256"] != CHECKPOINT_SHA256 or
            accuracy["population_manifest_sha256"] != measurement["manifests"]["sequence_population_manifest"]["sha256"]):
        raise RegistryError("typed accuracy receipt schema/status/root mismatch")
    accuracy_rows = {}
    for row in accuracy["rows"]:
        _exact(row, {"row_id", "configuration_manifest_sha256", "aee", "dsec_fl_percent"}, "accuracy row")
        row_id = row["row_id"]
        if row_id not in MANDATORY_ROW_IDS or row_id in accuracy_rows or row["configuration_manifest_sha256"] != configs[row_id]["sha256"]:
            raise RegistryError("typed accuracy row identity mismatch")
        if (not math.isclose(row["aee"], summaries[row_id]["accuracy"], rel_tol=1e-12, abs_tol=1e-12) or
                not math.isclose(row["dsec_fl_percent"], summaries[row_id]["dsec_fl_percent"], rel_tol=1e-12, abs_tol=1e-12)):
            raise RegistryError("typed accuracy row does not recompute from rooted raw logs")
        accuracy_rows[row_id] = row
    if set(accuracy_rows) != set(MANDATORY_ROW_IDS):
        raise RegistryError("typed accuracy receipt lacks six rows")

    _, ppa, ppa_sha = _file_spec(specs["ppa_receipt"], "typed PPA receipt", "hw_autoresearch_nts07/results/")
    _exact(ppa, {"schema", "status", "technology_nm", "clock_period_ns", "rows"}, "typed PPA receipt")
    if ppa["schema"] != "m653.h67.logic_macro_sta_power_ppa_receipt.r1" or ppa["status"] != "PASS_RAW_PROJECTED" or ppa["technology_nm"] != 28 or ppa["clock_period_ns"] != 3.0:
        raise RegistryError("typed PPA receipt schema/status/process mismatch")
    ppa_rows = {}
    for row in ppa["rows"]:
        fields = {"row_id", "configuration_manifest_sha256", "logic_area_mm2", "logic_power_mw",
                  "sram_macro_area_mm2", "sram_macro_power_mw", "total_area_mm2", "total_power_mw",
                  "setup_wns_ns", "hold_wns_ns", "logic_report", "sram_report", "sta_report"}
        _exact(row, fields, "PPA row")
        row_id = row["row_id"]
        if row_id not in MANDATORY_ROW_IDS or row_id in ppa_rows or row["configuration_manifest_sha256"] != configs[row_id]["sha256"]:
            raise RegistryError("typed PPA row identity mismatch")
        logic_raw, _ = _parse_projection_report(row["logic_report"], "PPA raw logic report",
                                                {"logic_area_mm2", "logic_power_mw"})
        sram_raw, _ = _parse_projection_report(row["sram_report"], "PPA raw SRAM report",
                                               {"sram_macro_area_mm2", "sram_macro_power_mw"})
        sta_raw, _ = _parse_projection_report(row["sta_report"], "PPA raw STA report",
                                              {"setup_wns_ns", "hold_wns_ns"})
        projection = {}
        projection.update(logic_raw)
        projection.update(sram_raw)
        projection.update(sta_raw)
        for field, value in projection.items():
            typed_value = _number(row[field], "typed PPA " + field, zero_ok=True)
            if not math.isclose(typed_value, value, rel_tol=0.0, abs_tol=1e-12):
                raise RegistryError("typed PPA scalar does not project raw report: " + field)
        area = projection["logic_area_mm2"] + projection["sram_macro_area_mm2"]
        power = projection["logic_power_mw"] + projection["sram_macro_power_mw"]
        if (not math.isclose(row["total_area_mm2"], area, rel_tol=0.0, abs_tol=1e-12) or
                not math.isclose(row["total_power_mw"], power, rel_tol=0.0, abs_tol=1e-12)):
            raise RegistryError("typed PPA total does not project raw area/power")
        ppa_rows[row_id] = row
    if set(ppa_rows) != set(MANDATORY_ROW_IDS):
        raise RegistryError("typed PPA receipt lacks six rows")
    return {"energy_sha256": energy_sha, "accuracy_sha256": accuracy_sha,
            "ppa_sha256": ppa_sha, "energy": energy_rows, "accuracy": accuracy_rows,
            "ppa": ppa_rows}


def _validate_result_and_receipts(bundle_id, specs, configs, common, measurement, raw, recomputed, typed):
    summaries, samples, views = recomputed
    _, result, result_sha = _file_spec(specs["direct_result"], "direct unified result",
                                       "hw_autoresearch_nts07/results/")
    fields = {"schema", "status", "bundle_id", "measurement_class", "population_id",
              "workload_id", "common_resource_manifest_sha256", "raw_run_index_sha256",
              "ppa_receipt_sha256", "energy_receipt_sha256", "accuracy_receipt_sha256",
              "rows", "samples", "aggregates", "views"}
    _exact(result, fields, "direct unified result")
    if (result["schema"] != "m645.h67.direct_unified.result.r1" or result["status"] != "PASS_COMPLETE" or
            result["bundle_id"] != bundle_id or result["measurement_class"] != M635.ALLOWED_MEASUREMENT_CLASS or
            result["population_id"] != measurement["population_id"] or
            result["common_resource_manifest_sha256"] != common["sha256"] or
            result["raw_run_index_sha256"] != raw["sha256"] or
            (result["ppa_receipt_sha256"], result["energy_receipt_sha256"], result["accuracy_receipt_sha256"]) !=
            (typed["ppa_sha256"], typed["energy_sha256"], typed["accuracy_sha256"])):
        raise RegistryError("direct unified result root/identity mismatch")
    _string(result["workload_id"], "workload_id")
    result_rows = {}
    specs_by_id = {row[0]: (row[1], row[2]) for row in M635.MANDATORY_ROW_SPECS}
    if not isinstance(result["rows"], list):
        raise RegistryError("direct result rows missing")
    for row in result["rows"]:
        _exact(row, {"row_id", "role", "fidelity", "cycles", "energy_mj", "area_mm2", "accuracy"},
               "direct result row")
        row_id = row["row_id"]
        if row_id not in MANDATORY_ROW_IDS or row_id in result_rows or (row["role"], row["fidelity"]) != specs_by_id[row_id]:
            raise RegistryError("direct result row identity/role/fidelity mismatch")
        expected = summaries[row_id]
        projection = {"cycles": expected["cycles"], "energy_mj": expected["energy_mj"],
                      "area_mm2": typed["ppa"][row_id]["total_area_mm2"],
                      "accuracy": expected["accuracy"]}
        for field, value in projection.items():
            if not math.isclose(float(row[field]), float(value), rel_tol=1e-12, abs_tol=1e-12):
                raise RegistryError("direct result row is not an exact typed/raw projection")
        result_rows[row_id] = row
    if tuple(result_rows) != MANDATORY_ROW_IDS or result["samples"] != samples:
        raise RegistryError("direct result row/sample projection mismatch")
    if not isinstance(result["views"], dict) or tuple(result["views"]) != VIEW_NAMES:
        raise RegistryError("direct result views missing")
    for view in VIEW_NAMES:
        R4._check_aggregate_map(result["views"][view], views[view], "direct result view " + view)
    R4._check_aggregate_map(result["aggregates"], views["iso_service"], "direct result aggregate")
    if result["aggregates"] != result["views"]["iso_service"]:
        raise RegistryError("default aggregate must be iso_service")

    _, completion, completion_sha = _file_spec(specs["completion_receipt"], "completion receipt",
                                               "hw_autoresearch_nts07/results/")
    _exact(completion, {"schema", "status", "bundle_id", "direct_result_sha256",
                        "raw_run_index_sha256", "ppa_receipt_sha256", "energy_receipt_sha256",
                        "accuracy_receipt_sha256", "completed_row_ids"}, "completion receipt")
    if (completion["schema"] != "m645.h67.direct_unified.completion_receipt.r1" or
            completion["status"] != "PASS_DERIVED_CLOSURES" or completion["bundle_id"] != bundle_id or
            completion["direct_result_sha256"] != result_sha or completion["raw_run_index_sha256"] != raw["sha256"] or
            (completion["ppa_receipt_sha256"], completion["energy_receipt_sha256"], completion["accuracy_receipt_sha256"]) !=
            (typed["ppa_sha256"], typed["energy_sha256"], typed["accuracy_sha256"]) or
            tuple(completion["completed_row_ids"]) != MANDATORY_ROW_IDS):
        raise RegistryError("completion receipt is not derived from all typed roots")

    _, coverage, coverage_sha = _file_spec(specs["coverage_receipt"], "coverage receipt",
                                           "hw_autoresearch_nts07/results/")
    _exact(coverage, {"schema", "status", "direct_result_sha256", "raw_run_index_sha256",
                      "population_manifest_sha256", "aggregation_manifest_sha256", "sample_ids",
                      "sequence_ids", "density_strata", "aggregates", "views"}, "coverage receipt")
    expected_samples = list(measurement["samples"])
    expected_sequences = sorted({item["sequence_id"] for item in measurement["samples"].values()})
    expected_strata = sorted({item["density_stratum"] for item in measurement["samples"].values()})
    if (coverage["schema"] != "m645.h67.coverage_receipt.r1" or coverage["status"] != "PASS_RECOMPUTED" or
            coverage["direct_result_sha256"] != result_sha or coverage["raw_run_index_sha256"] != raw["sha256"] or
            coverage["population_manifest_sha256"] != measurement["manifests"]["sequence_population_manifest"]["sha256"] or
            coverage["aggregation_manifest_sha256"] != measurement["manifests"]["aggregation_weight_manifest"]["sha256"] or
            coverage["sample_ids"] != expected_samples or coverage["sequence_ids"] != expected_sequences or
            coverage["density_strata"] != expected_strata or coverage["aggregates"] != result["aggregates"] or
            coverage["views"] != result["views"]):
        raise RegistryError("coverage receipt does not exactly project frozen population/raw result")
    if len(expected_sequences) < 3 or not {"low", "medium", "high"}.issubset(set(expected_strata)):
        raise RegistryError("Table-A population requires >=3 sequences and low/medium/high density coverage")
    return {"result": result, "result_sha256": result_sha, "rows": result_rows,
            "completion_sha256": completion_sha, "coverage_sha256": coverage_sha}


def _parse_sha256sums(path, label):
    return R4._parse_sha256sums(path, label)


def _validate_outer_seal(manifest_spec, outer_spec, label):
    return R4._validate_outer_seal(manifest_spec, outer_spec, label)


def _map_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                 ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _validate_reviewed_targets(targets, bundle_id, evidence, numerator, result_spec):
    _exact(targets, REQUEST_TARGET_FIELDS, "independent review targets")
    fixed = {
        "registry_builder": Path(__file__), "registry_config": DEFAULT_CONFIG,
        "registry_tests": REGISTRY_TESTS, "registry_contract": REGISTRY_CONTRACT,
        "m527_contract": M527_CONTRACT, "checkpoint": CHECKPOINT,
    }
    for name, path in fixed.items():
        media = ("application/octet-stream",) if name == "checkpoint" else (
            ("text/x-python", "text/plain") if name in ("registry_builder", "registry_tests") else
            ("application/json",))
        resolved, _, _ = _file_spec(targets[name], "review target " + name, media_types=media)
        if resolved != path:
            raise RegistryError("independent review target path mismatch: " + name)
    if targets["direct_result"] != result_spec:
        raise RegistryError("independent review does not bind exact direct result spec")
    if targets["fixed_throughput_numerator_receipt"] != _ACTIVE_REVIEW_CONTEXT["numerator_spec"]:
        raise RegistryError("independent review does not bind fixed-numerator receipt")
    _file_spec(targets["direct_result"], "review target direct result", "hw_autoresearch_nts07/results/")
    _file_spec(targets["fixed_throughput_numerator_receipt"], "review target fixed numerator",
               "hw_autoresearch_nts07/results/")
    return {name: targets[name]["sha256"] for name in sorted(targets)}


def _validate_hammer(spec, authority_id, evidence):
    authority = TRUSTED_HAMMER_AUTHORITIES.get(authority_id)
    if authority is None:
        raise RegistryError("independent hammer authority is not code-trusted in this release")
    required_authority = {"request_document", "request_manifest", "request_outer_seal",
                          "review_document", "review_manifest", "review_outer_seal", "receipt"}
    _exact(authority, required_authority, "code-trusted hammer authority")
    request_rows, _, request_outer_sha = _validate_outer_seal(
        authority["request_manifest"], authority["request_outer_seal"], "trusted request")
    _, request, request_sha = _file_spec(authority["request_document"], "trusted request document",
                                         "hw_autoresearch_nts07/reviews/")
    request_path = _secure_file(authority["request_document"]["path"])
    if request_rows != {request_path.name: request_sha}:
        raise RegistryError("trusted request manifest must seal exactly its typed request document")
    request_fields = {"schema", "status", "authority_id", "bundle_id", "reviewed_targets",
                      "bundle_evidence_sha256", "complete_evidence_root_sha256"}
    _exact(request, request_fields, "trusted typed request")
    if request["schema"] != "m653.h67.direct_unified.hammer_request.r1" or request["status"] != "FROZEN_BEFORE_REVIEW" or request["authority_id"] != authority_id:
        raise RegistryError("trusted request schema/status/authority mismatch")
    context = _ACTIVE_REVIEW_CONTEXT
    if context is None or request["bundle_id"] != context["bundle_id"]:
        raise RegistryError("trusted request bundle identity mismatch")
    combined_evidence = dict(evidence)
    combined_evidence.update(context["extra_evidence"])
    expected_evidence = {name: combined_evidence[name] for name in sorted(combined_evidence)}
    if request["bundle_evidence_sha256"] != expected_evidence or request["complete_evidence_root_sha256"] != _map_sha(expected_evidence):
        raise RegistryError("trusted request does not bind complete evidence-root manifest")
    reviewed_targets = _validate_reviewed_targets(request["reviewed_targets"], context["bundle_id"],
                                                  evidence, context["numerator"], context["result_spec"])

    review_rows, _, review_outer_sha = _validate_outer_seal(
        authority["review_manifest"], authority["review_outer_seal"], "trusted review")
    _, receipt, receipt_sha = _file_spec(spec, "independent hammer receipt", "hw_autoresearch_nts07/reviews/")
    _, review, review_sha = _file_spec(authority["review_document"], "trusted review document",
                                       "hw_autoresearch_nts07/reviews/")
    if spec != authority["receipt"]:
        raise RegistryError("bundle hammer receipt differs from code-trusted authority root")
    receipt_path = _secure_file(spec["path"])
    review_path = _secure_file(authority["review_document"]["path"])
    if review_rows != {receipt_path.name: receipt_sha, review_path.name: review_sha}:
        raise RegistryError("trusted review manifest must seal exactly review and receipt")
    review_fields = {"schema", "status", "authority_id", "request_outer_seal_sha256",
                     "reviewed_targets_sha256", "bundle_evidence_sha256",
                     "complete_evidence_root_sha256", "receipt_sha256", "severity_counts", "verdict"}
    _exact(review, review_fields, "trusted typed review")
    if (review["schema"] != "m653.h67.direct_unified.hammer_review.r1" or
            review["status"] != "COMPLETE" or review["authority_id"] != authority_id or
            review["request_outer_seal_sha256"] != request_outer_sha or
            review["reviewed_targets_sha256"] != reviewed_targets or
            review["bundle_evidence_sha256"] != expected_evidence or
            review["complete_evidence_root_sha256"] != _map_sha(expected_evidence) or
            review["receipt_sha256"] != receipt_sha or review["severity_counts"] != {"P0": 0, "P1": 0} or
            review["verdict"] != "GO"):
        raise RegistryError("trusted typed review does not admit the exact target/result/contract graph")
    if review_outer_sha != authority["review_outer_seal"]["sha256"]:
        raise RegistryError("independent review outer-seal identity mismatch")

    fields = {"schema", "status", "authority_id", "request_outer_seal_sha256",
              "reviewed_targets_sha256", "bundle_evidence_sha256", "severity_counts",
              "independence", "recomputed_fixed_throughput_numerators", "recomputed_rows",
              "recomputed_aggregates", "recomputed_views", "authorization"}
    _exact(receipt, fields, "independent hammer receipt")
    if (receipt["schema"] != "m653.h67.direct_unified.independent_hammer.r1" or
            receipt["status"] != "PASS_INDEPENDENT" or receipt["authority_id"] != authority_id or
            receipt["request_outer_seal_sha256"] != request_outer_sha or
            receipt["reviewed_targets_sha256"] != reviewed_targets or
            receipt["bundle_evidence_sha256"] != expected_evidence or
            receipt["severity_counts"] != {"P0": 0, "P1": 0}):
        raise RegistryError("independent hammer does not bind exact reviewed graph or has blocking findings")
    expected_independence = {"author_receipt_used_as_authority": False,
                             "raw_logs_rehashed_and_recomputed": True,
                             "fixed_numerators_rehashed_and_recomputed": True,
                             "typed_receipts_recomputed": True,
                             "raw_ppa_reports_parsed_and_projected": True,
                             "result_modified": False}
    if receipt["independence"] != expected_independence:
        raise RegistryError("independent hammer did not independently recompute every root")
    expected_numerators = {
        "dense_equivalent_ops_scalar": context["numerator"]["dense_equivalent_ops_scalar"],
        "original_useful_nonzero_ops_scalar": context["numerator"]["original_useful_nonzero_ops_scalar"],
    }
    if receipt["recomputed_fixed_throughput_numerators"] != expected_numerators:
        raise RegistryError("independent hammer fixed-numerator recomputation mismatch")
    expected_auth = {"table_a_methodology_admitted": True,
                     "direct_unified_measurement_admitted": True,
                     "paper_headline_admitted": True}
    if receipt["authorization"] != expected_auth:
        raise RegistryError("independent hammer authorization missing")
    return receipt_sha, receipt


_R4_VALIDATE_BUNDLE = R4._validate_bundle
_ACTIVE_REVIEW_CONTEXT = None


def _validate_bundle(bundle_id, bundle):
    global _ACTIVE_REVIEW_CONTEXT
    _exact(bundle, BUNDLE_FIELDS, "Table-A direct bundle " + bundle_id)
    if bundle["schema"] != "m653.h67.rooted_direct_bundle.r2" or bundle["bundle_id"] != bundle_id:
        raise RegistryError("M653 Table-A bundle schema/id mismatch")
    _runtime_m527_contract()
    m527_sha = _validate_exact_root_spec(bundle["m527_contract"], M527_CONTRACT,
                                         M527_CONTRACT_SHA256, "actual M527 contract",
                                         ("application/json",))
    checkpoint_sha = _validate_exact_root_spec(bundle["checkpoint"], CHECKPOINT,
                                               CHECKPOINT_SHA256, "actual H67 checkpoint",
                                               ("application/octet-stream",))
    if bundle["m527_contract_sha256"] != m527_sha or checkpoint_sha != CHECKPOINT_SHA256:
        raise RegistryError("bundle frozen root mismatch")
    # Validate identity first so the fixed numerator can project its population.
    measurement = _validate_measurement_identity(bundle["measurement_identity"], checkpoint_sha)
    numerator = _validate_numerator_receipt(bundle["fixed_throughput_numerator_receipt"], measurement)
    legacy = copy.deepcopy(bundle)
    for field in ("m527_contract", "checkpoint", "fixed_throughput_numerator_receipt"):
        del legacy[field]
    legacy["schema"] = "m645.h67.rooted_direct_bundle.r1"
    extra_evidence = {
        "m527_contract": m527_sha, "checkpoint": checkpoint_sha,
        "fixed_throughput_numerator_receipt": numerator["sha256"],
    }
    previous = _ACTIVE_REVIEW_CONTEXT
    previous_r4_authorities = R4.TRUSTED_HAMMER_AUTHORITIES
    _ACTIVE_REVIEW_CONTEXT = {"bundle_id": bundle_id, "numerator": numerator,
                              "numerator_spec": bundle["fixed_throughput_numerator_receipt"],
                              "result_spec": bundle["direct_result"], "extra_evidence": extra_evidence}
    R4.TRUSTED_HAMMER_AUTHORITIES = TRUSTED_HAMMER_AUTHORITIES
    try:
        result = _R4_VALIDATE_BUNDLE(bundle_id, legacy)
    finally:
        _ACTIVE_REVIEW_CONTEXT = previous
        R4.TRUSTED_HAMMER_AUTHORITIES = previous_r4_authorities
    result["fixed_numerator_sha256"] = numerator["sha256"]
    result["fixed_throughput_numerators"] = {
        "dense_equivalent_ops_scalar": numerator["dense_equivalent_ops_scalar"],
        "original_useful_nonzero_ops_scalar": numerator["original_useful_nonzero_ops_scalar"],
    }
    result["m527_independent_gates"] = {
        "measurement_identity": True,
        "fixed_throughput_numerators": True,
        "configuration_registry": True,
    }
    return result


def _validate_overlay(config):
    fields = {"schema", "date", "status", "purpose", "base_registry",
              "table_a_evidence_bundles", "table_a_rows", "claim_boundary", "protected_file"}
    _exact(config, fields, "M653 registry overlay")
    if config["schema"] != "m653.h67.paper_metric_registry.r5":
        raise RegistryError("unexpected M653 registry schema")
    _, base, base_sha = _file_spec(config["base_registry"], "sealed M635 base registry",
                                   "hw_autoresearch_nts07/system_simulator/config/")
    if base_sha != M635_CONFIG_SHA256 or config["base_registry"]["path"] != M635_CONFIG.relative_to(REPO_ROOT).as_posix():
        raise RegistryError("M653 must inherit exact sealed M635 canonical registry")
    if base.get("schema") != "m635.h67.paper_metric_registry.r3" or base.get("table_a_evidence_bundles") != {}:
        raise RegistryError("sealed M635 base is not canonical zero-bundle registry")
    protected = config["protected_file"]
    _exact(protected, {"path", "sha256"}, "protected file")
    if protected != {"path": DOCS359.relative_to(REPO_ROOT).as_posix(), "sha256": DOCS359_SHA256}:
        raise RegistryError("protected docs359 binding mismatch")
    if _sha256(_secure_file(protected["path"])) != DOCS359_SHA256:
        raise RegistryError("protected docs359 SHA drift")
    return base


def _evaluate(rows, bundles, claim, policy):
    gate = R4._evaluate(rows, bundles, claim, policy)
    mandatory_sources = {row["source_id"] for row in rows
                         if row["row_id"] in MANDATORY_ROW_IDS and row["source_id"] is not None}
    flags = {"measurement_identity": False, "fixed_throughput_numerators": False,
             "configuration_registry": False}
    if len(mandatory_sources) == 1:
        source_id = next(iter(mandatory_sources))
        if source_id in bundles:
            flags = copy.deepcopy(bundles[source_id]["m527_independent_gates"])
    if gate["admitted"] and not all(flags.values()):
        raise RegistryError("headline reached without all executable M527 independent gates")
    gate["m527_independent_gates"] = flags
    gate["all_m527_independent_gates_pass"] = all(flags.values())
    return gate


def build(config_path=DEFAULT_CONFIG):
    _runtime_m527_contract()
    config = M635.load_json(Path(config_path), "M653 registry")
    base = _validate_overlay(config)
    sources, source_docs = M635.validate_sources(base)
    base_rows = M635.validate_ladder_and_tables(base, set(sources))
    policy = M635.validate_policy(base)
    M635.validate_m518_binding(base, source_docs)
    analytical = M635.recompute_analytical(base)
    rows = copy.deepcopy(base_rows if config["table_a_rows"] is None else config["table_a_rows"])
    validation_copy = copy.deepcopy(base)
    validation_copy["table_a_schema"]["rows"] = copy.deepcopy(rows)
    M635.validate_ladder_and_tables(validation_copy, set(sources))
    bundle_specs = config["table_a_evidence_bundles"]
    if not isinstance(bundle_specs, dict):
        raise RegistryError("table_a_evidence_bundles must be an object")
    bundles = {bundle_id: _validate_bundle(bundle_id, bundle)
               for bundle_id, bundle in bundle_specs.items()}
    gate = _evaluate(rows, bundles, config["claim_boundary"], policy)
    return {"schema": "m653.h67.paper_metric_registry.r5.preview",
            "status": config["status"], "source_hashes_validated": sources,
            "trusted_hammer_authority_count": len(TRUSTED_HAMMER_AUTHORITIES),
            "table_a_evidence_bundle_count": len(bundles),
            "table_a": rows, "table_b": base["table_b_schema"]["rows"],
            "table_c": base["table_c_schema"]["rows"],
            "analytical_diagnostic": analytical, "headline_gate": gate,
            "claim_boundary": config["claim_boundary"],
            "protected_file_validated": config["protected_file"]}


# The sealed r4 bundle walk performs global lookups for these inner validators.
R4._validate_measurement_identity = _validate_measurement_identity
R4._validate_typed_receipts = _validate_typed_receipts
R4._validate_result_and_receipts = _validate_result_and_receipts
R4._validate_hammer = _validate_hammer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--emit-json", action="store_true")
    args = parser.parse_args()
    try:
        result = build(args.config)
    except (RegistryError, RuntimeError) as exc:
        print("M653_REGISTRY_FAIL: %s" % exc)
        return 2
    if args.emit_json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    else:
        print("M653_REGISTRY_PASS sources=%d trusted_authorities=%d bundles=%d table_a_eligible=%d headline_admitted=%s analytical_admitted=%s" %
              (len(result["source_hashes_validated"]), result["trusted_hammer_authority_count"],
               result["table_a_evidence_bundle_count"], result["headline_gate"]["eligible_row_count"],
               str(result["headline_gate"]["admitted"]).lower(),
               str(result["analytical_diagnostic"]["admitted"]).lower()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
