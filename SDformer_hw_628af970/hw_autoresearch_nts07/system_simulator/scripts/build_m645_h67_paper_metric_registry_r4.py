#!/usr/bin/env python3
"""Build the M645 H67 paper-metric registry r4, fail closed.

M645 supersedes only the *methodology* of M635.  It imports the sealed M635
Table-B/Table-C catalogue, but a Table-A row can now be projected only from a
rooted direct-run bundle.  In particular, directory placement under
``reviews/`` is not authority: a measurement hammer must be named in the
code-level ``TRUSTED_HAMMER_AUTHORITIES`` map with exact request/review outer
seals.  The map is intentionally empty in this release, so the canonical
registry and every author-created bundle remain headline-ineligible.

No GPU, EDA, simulator or capture is launched by this program.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = REPO_ROOT / "hw_autoresearch_nts07"
M635_BUILDER = HW_ROOT / "system_simulator/scripts/build_m635_h67_paper_metric_registry_r3.py"
M635_BUILDER_SHA256 = "bbd7dacb8c6fcaea07c07c905451647e7874ceddd352c542f44be36a7e89b058"
M635_CONFIG = HW_ROOT / "system_simulator/config/m635_h67_paper_metric_registry_r3_20260828.json"
M635_CONFIG_SHA256 = "9ed999d4cb31021806eff53bcdd90d1178080cced53506b6935363f662e4a4b8"
M527_CONTRACT = HW_ROOT / "contracts/m527_h67_headline_baseline_ladder_contract_r3_20260827.json"
M527_CONTRACT_SHA256 = "83ea25e43b53d12800ac64e971069a682e3077411ff10851a7861636ef77355b"
DOCS359 = HW_ROOT / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
CHECKPOINT_SHA256 = "4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158"
DEFAULT_CONFIG = HW_ROOT / "system_simulator/config/m645_h67_paper_metric_registry_r4_20260828.json"


def _load_m635():
    if _sha256(M635_BUILDER) != M635_BUILDER_SHA256:
        raise RuntimeError("sealed M635 builder SHA drift")
    spec = importlib.util.spec_from_file_location("m645_sealed_m635", str(M635_BUILDER))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import sealed M635 builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


M635 = _load_m635()
RegistryError = M635.RegistryError


# This is the authority boundary.  It may only be populated in a later,
# separately reviewed source release after a real measurement hammer exists.
# Config files cannot add entries.  Each future entry must contain exact
# repo-relative paths and SHA256 values for request/review manifests, their
# outer seals, and the admitted hammer receipt.
TRUSTED_HAMMER_AUTHORITIES = {}

ROW_TO_M527_CONFIGURATION = {
    "dense96_fixed_t10": "b0_dense96_fixed_t10",
    "ptb_like_structured": "b1_project_defined_ptb_like_structured_k1x8",
    "exact_bit_k1": "b2_exact_bit_sparse_k1",
    "exact_bit_k1x8": "b3_exact_bit_sparse_k1x8",
    "exact_typed_k8": "c2_exact_typed_k8",
    "ours_exact": "c123_ours_exact",
}
MANDATORY_ROW_IDS = tuple(item[0] for item in M635.MANDATORY_ROW_SPECS)
VIEW_NAMES = tuple(M635.VIEW_NAMES)
AGGREGATE_NAMES = tuple(M635.AGGREGATE_NAMES)

RESOURCE_FIELDS = (
    "technology_nm", "clock_period_ns", "source_lanes",
    "service_width_sources_per_cycle", "onchip_sram_bytes_total",
    "dram_bandwidth_bytes_per_second_decimal", "dram_bytes_per_cycle",
    "accumulator_bits", "source_queue_depth", "completion_queue_depth",
    "parent_queue_depth", "weight_sram_bank_count", "state_sram_bank_count",
    "parent_scratch_bank_count", "weight_sram_port_mode",
    "state_sram_port_mode", "parent_scratch_port_mode",
    "external_read_port_count", "external_write_port_count",
)
CHARGE_FIELDS = (
    "extra_matcher_area_charged", "extra_scoreboard_area_charged",
    "extra_control_area_charged", "extra_state_bits_charged",
    "extra_sram_bytes_charged", "extra_sram_ports_charged",
    "all_added_logic_dynamic_energy_charged",
    "all_added_memory_dynamic_energy_charged",
)
FALLBACK_FIELDS = (
    "mode", "must_charge_cycles", "must_charge_traffic",
    "must_charge_energy", "must_charge_area", "unsupported_operator_ids",
)
MECHANISM_FIELDS = (
    "dense_fixed_t10", "structured_group_scan", "exact_bit_sparse",
    "parent_product_capture", "typed_k8", "exact_atlif_service",
    "execution_service_limit_sources_per_cycle",
)
EXPECTED_MECHANISMS = {
    "b0_dense96_fixed_t10": (True, False, False, False, False, False, 8),
    "b1_project_defined_ptb_like_structured_k1x8": (False, True, False, False, False, False, 8),
    "b2_exact_bit_sparse_k1": (False, False, True, False, False, False, 1),
    "b3_exact_bit_sparse_k1x8": (False, False, True, False, False, False, 8),
    "c2_exact_typed_k8": (False, False, True, False, True, False, 8),
    "c123_ours_exact": (False, False, True, True, True, True, 8),
}

FILE_SPEC_FIELDS = {"path", "sha256", "media_type"}
BUNDLE_FIELDS = {
    "schema", "bundle_id", "m527_contract_sha256", "common_resource_manifest",
    "configuration_manifests", "producer", "unified_simulator",
    "invocation_contract", "measurement_identity", "raw_run_index",
    "direct_result", "completion_receipt", "coverage_receipt", "ppa_receipt",
    "energy_receipt", "accuracy_receipt", "independent_hammer_authority_id",
    "independent_hammer_receipt",
}


def _exact(value, fields, label):
    if not isinstance(value, dict) or set(value) != set(fields):
        actual = set(value) if isinstance(value, dict) else set()
        raise RegistryError("%s fields differ: missing=%s extra=%s" %
                            (label, sorted(set(fields) - actual), sorted(actual - set(fields))))


def _sha(value, label):
    if (not isinstance(value, str) or len(value) != 64 or
            any(ch not in "0123456789abcdef" for ch in value)):
        raise RegistryError("%s must be a lowercase SHA256" % label)
    return value


def _string(value, label):
    if not isinstance(value, str) or not value.strip():
        raise RegistryError("%s must be a nonempty string" % label)
    return value


def _number(value, label, zero_ok=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or value < (0 if zero_ok else 1e-300)):
        raise RegistryError("%s must be a %s finite number" %
                            (label, "nonnegative" if zero_ok else "positive"))
    return float(value)


def _integer(value, label, zero_ok=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if zero_ok else 1):
        raise RegistryError("%s must be a %s integer" %
                            (label, "nonnegative" if zero_ok else "positive"))
    return value


def _secure_file(relative, prefix=None):
    if not isinstance(relative, str) or not relative or os.path.isabs(relative):
        raise RegistryError("evidence path must be nonempty and repo-relative")
    if prefix is not None and not relative.startswith(prefix):
        raise RegistryError("evidence path outside required namespace: %s" % relative)
    return M635.secure_repo_file(relative)


def _file_spec(spec, label, prefix=None, media_types=("application/json",)):
    _exact(spec, FILE_SPEC_FIELDS, label)
    if spec["media_type"] not in media_types:
        raise RegistryError("%s media_type is not admitted" % label)
    expected = _sha(spec["sha256"], label + " SHA")
    path = _secure_file(spec["path"], prefix)
    actual = _sha256(path)
    if actual != expected:
        raise RegistryError("%s SHA mismatch" % label)
    document = None
    if spec["media_type"] == "application/json":
        document = M635.load_json(path, label)
    return path, document, actual


def _source_ref(path_value, sha_value, label, prefix=None, json_required=False):
    _string(path_value, label + " path")
    expected = _sha(sha_value, label + " SHA")
    path = _secure_file(path_value, prefix)
    if _sha256(path) != expected:
        raise RegistryError("%s source SHA mismatch" % label)
    if json_required:
        M635.load_json(path, label)
    return path


def _validate_resource(resource):
    _exact(resource, RESOURCE_FIELDS, "M527 resource tuple")
    fixed = {
        "technology_nm": 28,
        "clock_period_ns": 3.0,
        "source_lanes": 96,
        "service_width_sources_per_cycle": 8,
        "onchip_sram_bytes_total": 245760,
        "dram_bandwidth_bytes_per_second_decimal": 64000000000,
        "dram_bytes_per_cycle": 192,
        "accumulator_bits": 24,
    }
    for key, expected in fixed.items():
        if resource[key] != expected:
            raise RegistryError("M527 resource %s mismatch" % key)
    for key in RESOURCE_FIELDS[8:14] + RESOURCE_FIELDS[17:19]:
        _integer(resource[key], "resource " + key)
    for key in RESOURCE_FIELDS[14:17]:
        _string(resource[key], "resource " + key)


def _validate_charge(charge):
    _exact(charge, CHARGE_FIELDS, "charge policy")
    if any(charge[key] is not True for key in CHARGE_FIELDS):
        raise RegistryError("all M527 added logic/memory resources must be charged")


def _validate_fallback(fallback, require_partition=False):
    _exact(fallback, FALLBACK_FIELDS, "fallback policy")
    if fallback["mode"] != "EXECUTE_UNSUPPORTED_WORK_IN_THE_SAME_UNIFIED_MODEL":
        raise RegistryError("fallback mode mismatch")
    if any(fallback[key] is not True for key in FALLBACK_FIELDS[1:5]):
        raise RegistryError("fallback must charge cycles/traffic/energy/area")
    unsupported = fallback["unsupported_operator_ids"]
    if not isinstance(unsupported, list) or any(not isinstance(x, str) or not x for x in unsupported):
        raise RegistryError("fallback unsupported_operator_ids must be explicit")
    if len(unsupported) != len(set(unsupported)):
        raise RegistryError("fallback unsupported_operator_ids contain duplicates")
    if require_partition and not unsupported:
        raise RegistryError("fallback partition cannot be silently empty")


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
            raise RegistryError("%s schema/status mismatch" % name)
        if child.get("checkpoint_sha256") != checkpoint_sha:
            raise RegistryError("%s checkpoint mismatch" % name)
        manifests[name] = {"path": path, "doc": child, "sha256": child_sha}
    trace = manifests["complete_trace_manifest"]["doc"]
    _exact(trace, {"schema", "status", "checkpoint_sha256", "decoder_complete",
                   "operator_scope", "record_count", "trace_members"}, "complete trace manifest")
    if trace["decoder_complete"] is not True:
        raise RegistryError("complete trace does not include decoder")
    _integer(trace["record_count"], "complete trace record_count")
    if not isinstance(trace["operator_scope"], list) or "ConvTranspose2d" not in trace["operator_scope"]:
        raise RegistryError("complete trace operator scope is not decoder-complete")
    if not isinstance(trace["trace_members"], list) or not trace["trace_members"]:
        raise RegistryError("complete trace member list empty")
    for index, member in enumerate(trace["trace_members"]):
        _file_spec(member, "trace member %d" % index, "hw_autoresearch_nts07/")
    population = manifests["sequence_population_manifest"]["doc"]
    _exact(population, {"schema", "status", "checkpoint_sha256", "population_id",
                        "samples"}, "population manifest")
    samples = population["samples"]
    if not isinstance(samples, list) or not samples:
        raise RegistryError("population samples empty")
    population_samples = {}
    for sample in samples:
        _exact(sample, {"sample_id", "sequence_id", "density_stratum", "frame_count"}, "population sample")
        sample_id = _string(sample["sample_id"], "sample_id")
        _string(sample["sequence_id"], "sequence_id")
        if sample["density_stratum"] not in ("low", "mid", "high"):
            raise RegistryError("population density stratum invalid")
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
    if not isinstance(weights["samples"], list):
        raise RegistryError("aggregation samples missing")
    for sample in weights["samples"]:
        _exact(sample, {"sample_id", "view", "weight"}, "aggregation sample")
        key = (_string(sample["sample_id"], "aggregation sample_id"), sample["view"])
        if key[0] not in population_samples or key[1] not in VIEW_NAMES:
            raise RegistryError("aggregation sample is outside frozen population/view")
        _number(sample["weight"], "aggregation weight", zero_ok=True)
        if key in aggregation:
            raise RegistryError("duplicate aggregation sample/view")
        aggregation[key] = float(sample["weight"])
    expected_keys = {(sample_id, view) for sample_id in population_samples for view in VIEW_NAMES}
    if set(aggregation) != expected_keys:
        raise RegistryError("aggregation does not cover every population sample/view")
    for view in VIEW_NAMES:
        total = math.fsum(aggregation[(sample_id, view)] for sample_id in population_samples)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise RegistryError("aggregation weights for %s do not sum to one" % view)
    return {
        "sha256": digest, "doc": doc, "manifests": manifests,
        "population_id": population["population_id"], "samples": population_samples,
        "weights": aggregation,
    }


def _validate_common_resource(spec, producer_sha, simulator_sha, measurement):
    _, doc, digest = _file_spec(spec, "M527 common resource manifest", "hw_autoresearch_nts07/system_simulator/")
    fields = {"schema", "status", "m527_contract_sha256", "checkpoint_sha256",
              "producer_source_sha256", "simulator_source_sha256",
              "measurement_identity_sha256", "resource_tuple", "charge_policy",
              "fallback_policy"}
    _exact(doc, fields, "M527 common resource manifest")
    if doc["schema"] != "m527_h67_common_resource_manifest_v1" or doc["status"] != "FROZEN_EXECUTABLE":
        raise RegistryError("M527 common resource schema/status mismatch")
    expected = (M527_CONTRACT_SHA256, CHECKPOINT_SHA256, producer_sha, simulator_sha, measurement["sha256"])
    actual = (doc["m527_contract_sha256"], doc["checkpoint_sha256"], doc["producer_source_sha256"],
              doc["simulator_source_sha256"], doc["measurement_identity_sha256"])
    if actual != expected:
        raise RegistryError("M527 common resource root mismatch")
    _validate_resource(doc["resource_tuple"])
    _validate_charge(doc["charge_policy"])
    _validate_fallback(doc["fallback_policy"])
    return {"sha256": digest, "doc": doc}


def _validate_configurations(specs, common, producer_sha, simulator_sha, invocation_sha, measurement):
    if not isinstance(specs, dict) or tuple(specs) != MANDATORY_ROW_IDS:
        raise RegistryError("configuration manifests must follow the exact six-row Table-A order")
    resolved = {}
    seen_sources = set()
    for row_id in MANDATORY_ROW_IDS:
        _, doc, digest = _file_spec(specs[row_id], "configuration manifest " + row_id,
                                    "hw_autoresearch_nts07/system_simulator/")
        fields = {"schema", "status", "configuration_id", "configuration_source",
                  "producer_source_sha256", "simulator_source_sha256",
                  "invocation_contract_sha256", "checkpoint_sha256",
                  "complete_trace_manifest_sha256", "sequence_population_manifest_sha256",
                  "aggregation_weight_manifest_sha256", "common_resource_manifest_sha256",
                  "mechanism_enable_map", "optimized_operator_ids", "resource_tuple",
                  "charge_policy", "fallback_policy"}
        _exact(doc, fields, "configuration manifest " + row_id)
        config_id = ROW_TO_M527_CONFIGURATION[row_id]
        if (doc["schema"] != "m527_h67_executable_configuration_manifest_v1" or
                doc["status"] != "FROZEN_EXECUTABLE" or doc["configuration_id"] != config_id):
            raise RegistryError("configuration schema/status/id mismatch for " + row_id)
        source_path, source_doc, source_sha = _file_spec(
            doc["configuration_source"], "configuration source " + row_id,
            "hw_autoresearch_nts07/system_simulator/", ("application/json",))
        if source_path in seen_sources:
            raise RegistryError("each Table-A row requires a distinct executable configuration source")
        seen_sources.add(source_path)
        _exact(source_doc, {"schema", "configuration_id", "mechanism_enable_map",
                            "optimized_operator_ids", "unsupported_operator_ids"},
               "configuration source " + row_id)
        if source_doc["schema"] != "m645.h67.executable_configuration_source.r1" or source_doc["configuration_id"] != config_id:
            raise RegistryError("configuration source identity mismatch")
        roots = (
            doc["producer_source_sha256"], doc["simulator_source_sha256"],
            doc["invocation_contract_sha256"], doc["checkpoint_sha256"],
            doc["complete_trace_manifest_sha256"], doc["sequence_population_manifest_sha256"],
            doc["aggregation_weight_manifest_sha256"], doc["common_resource_manifest_sha256"],
        )
        expected = (
            producer_sha, simulator_sha, invocation_sha, CHECKPOINT_SHA256,
            measurement["manifests"]["complete_trace_manifest"]["sha256"],
            measurement["manifests"]["sequence_population_manifest"]["sha256"],
            measurement["manifests"]["aggregation_weight_manifest"]["sha256"], common["sha256"],
        )
        if roots != expected:
            raise RegistryError("configuration measurement/root SHA mismatch for " + row_id)
        _exact(doc["mechanism_enable_map"], MECHANISM_FIELDS, "mechanism map")
        expected_mechanisms = dict(zip(MECHANISM_FIELDS, EXPECTED_MECHANISMS[config_id]))
        if doc["mechanism_enable_map"] != expected_mechanisms or source_doc["mechanism_enable_map"] != expected_mechanisms:
            raise RegistryError("configuration mechanism map mismatch for " + row_id)
        if doc["optimized_operator_ids"] != source_doc["optimized_operator_ids"]:
            raise RegistryError("configuration optimized operator partition drift")
        unsupported = source_doc["unsupported_operator_ids"]
        if doc["fallback_policy"].get("unsupported_operator_ids") != unsupported:
            raise RegistryError("configuration fallback partition drift")
        operator_ids = measurement["doc"]["operator_ids"]
        optimized = doc["optimized_operator_ids"]
        if (not isinstance(optimized, list) or set(optimized) & set(unsupported) or
                set(optimized) | set(unsupported) != set(operator_ids)):
            raise RegistryError("configuration operator partition is incomplete")
        _validate_resource(doc["resource_tuple"])
        _validate_charge(doc["charge_policy"])
        _validate_fallback(doc["fallback_policy"])
        if (doc["resource_tuple"] != common["doc"]["resource_tuple"] or
                doc["charge_policy"] != common["doc"]["charge_policy"] or
                {key: doc["fallback_policy"][key] for key in FALLBACK_FIELDS[:-1]} !=
                {key: common["doc"]["fallback_policy"][key] for key in FALLBACK_FIELDS[:-1]}):
            raise RegistryError("configuration does not share the exact common-resource/charge/fallback policy")
        resolved[row_id] = {"sha256": digest, "source_sha256": source_sha,
                            "configuration_id": config_id, "doc": doc}
    return resolved


def _aggregate(ratios):
    return {
        "arithmetic_mean": math.fsum(ratios) / len(ratios),
        "geometric_mean": math.exp(math.fsum(math.log(x) for x in ratios) / len(ratios)),
        "ratio_of_sums": None,
        "minimum": min(ratios),
        "maximum": max(ratios),
    }


def _check_aggregate_map(actual, expected, label):
    if not isinstance(actual, dict) or tuple(actual) != AGGREGATE_NAMES:
        raise RegistryError(label + " aggregate fields/order mismatch")
    for name in AGGREGATE_NAMES:
        _number(actual[name], label + " " + name)
        if not math.isclose(float(actual[name]), float(expected[name]), rel_tol=1e-12, abs_tol=1e-12):
            raise RegistryError(label + " does not recompute from rooted raw runs")


def _validate_raw_runs(spec, configs, producer_sha, simulator_sha, invocation_sha, measurement):
    _, index, index_sha = _file_spec(spec, "raw-run index", "hw_autoresearch_nts07/results/")
    _exact(index, {"schema", "status", "producer_source_sha256", "simulator_source_sha256",
                   "invocation_contract_sha256", "measurement_identity_sha256", "runs"}, "raw-run index")
    if index["schema"] != "m645.h67.raw_run_index.r1" or index["status"] != "PASS_COMPLETE":
        raise RegistryError("raw-run index schema/status mismatch")
    if (index["producer_source_sha256"], index["simulator_source_sha256"],
            index["invocation_contract_sha256"], index["measurement_identity_sha256"]) != (
            producer_sha, simulator_sha, invocation_sha, measurement["sha256"]):
        raise RegistryError("raw-run index executable roots mismatch")
    runs = index["runs"]
    if not isinstance(runs, list) or not runs:
        raise RegistryError("raw-run index is empty")
    observed = {}
    run_hashes = []
    expected_keys = {(sample_id, view, row_id) for sample_id in measurement["samples"]
                     for view in VIEW_NAMES for row_id in MANDATORY_ROW_IDS}
    for position, item in enumerate(runs):
        _exact(item, {"run_id", "sample_id", "view", "row_id", "log"}, "raw-run index item")
        key = (item["sample_id"], item["view"], item["row_id"])
        if key not in expected_keys or key in observed:
            raise RegistryError("raw-run population is duplicated or outside the frozen Cartesian product")
        _, log, log_sha = _file_spec(item["log"], "raw-run log %d" % position,
                                     "hw_autoresearch_nts07/results/")
        fields = {"schema", "status", "run_id", "sample_id", "sequence_id",
                  "density_stratum", "view", "row_id", "configuration_manifest_sha256",
                  "producer_source_sha256", "simulator_source_sha256",
                  "invocation_contract_sha256", "checkpoint_sha256",
                  "complete_trace_manifest_sha256", "sequence_population_manifest_sha256",
                  "aggregation_weight_manifest_sha256", "direct_cycles",
                  "logic_energy_nj", "sram_energy_nj", "dram_energy_nj", "aee",
                  "dsec_fl_percent"}
        _exact(log, fields, "raw-run log")
        if log["schema"] != "m645.h67.direct_raw_run.r1" or log["status"] != "PASS_DIRECT":
            raise RegistryError("raw-run log schema/status mismatch")
        population = measurement["samples"][key[0]]
        if (log["run_id"], log["sample_id"], log["view"], log["row_id"],
                log["sequence_id"], log["density_stratum"]) != (
                item["run_id"], key[0], key[1], key[2], population["sequence_id"],
                population["density_stratum"]):
            raise RegistryError("raw-run index/population identity mismatch")
        roots = (log["configuration_manifest_sha256"], log["producer_source_sha256"],
                 log["simulator_source_sha256"], log["invocation_contract_sha256"],
                 log["checkpoint_sha256"], log["complete_trace_manifest_sha256"],
                 log["sequence_population_manifest_sha256"], log["aggregation_weight_manifest_sha256"])
        expected = (configs[key[2]]["sha256"], producer_sha, simulator_sha, invocation_sha,
                    CHECKPOINT_SHA256,
                    measurement["manifests"]["complete_trace_manifest"]["sha256"],
                    measurement["manifests"]["sequence_population_manifest"]["sha256"],
                    measurement["manifests"]["aggregation_weight_manifest"]["sha256"])
        if roots != expected:
            raise RegistryError("raw-run log is not rooted in the executable configuration/measurement")
        _integer(log["direct_cycles"], "direct cycles")
        for name in ("logic_energy_nj", "sram_energy_nj", "dram_energy_nj", "aee"):
            _number(log[name], "raw-run " + name, zero_ok=True)
        fl = _number(log["dsec_fl_percent"], "raw-run DSEC-Fl", zero_ok=True)
        if fl > 100:
            raise RegistryError("raw-run DSEC-Fl exceeds 100 percent")
        observed[key] = log
        run_hashes.append(log_sha)
    if set(observed) != expected_keys:
        raise RegistryError("raw-run index does not cover every row/sample/view exactly once")
    return {"sha256": index_sha, "runs": observed, "run_hashes": run_hashes}


def _recompute(runs, measurement):
    summaries = {}
    samples = []
    for view in VIEW_NAMES:
        for sample_id, population in measurement["samples"].items():
            row_cycles = {row_id: runs[(sample_id, view, row_id)]["direct_cycles"]
                          for row_id in MANDATORY_ROW_IDS}
            samples.append({"sample_id": sample_id, "sequence_id": population["sequence_id"],
                            "density_stratum": population["density_stratum"], "view": view,
                            "row_cycles": row_cycles})
    views = {}
    for view in VIEW_NAMES:
        view_samples = [sample for sample in samples if sample["view"] == view]
        ratios = [sample["row_cycles"]["dense96_fixed_t10"] /
                  sample["row_cycles"]["ours_exact"] for sample in view_samples]
        aggregate = _aggregate(ratios)
        aggregate["ratio_of_sums"] = (math.fsum(sample["row_cycles"]["dense96_fixed_t10"] for sample in view_samples) /
                                      math.fsum(sample["row_cycles"]["ours_exact"] for sample in view_samples))
        views[view] = aggregate
    for row_id in MANDATORY_ROW_IDS:
        iso = [(sample_id, measurement["weights"][(sample_id, "iso_service")])
               for sample_id in measurement["samples"]]
        cycles = sum(runs[(sample_id, "iso_service", row_id)]["direct_cycles"]
                     for sample_id in measurement["samples"])
        energy_components_mj = {}
        for component in ("logic", "sram", "dram"):
            energy_components_mj[component + "_energy_mj"] = math.fsum(
                weight * runs[(sample_id, "iso_service", row_id)][component + "_energy_nj"]
                for sample_id, weight in iso) / 1e6
        aee = math.fsum(weight * runs[(sample_id, "iso_service", row_id)]["aee"]
                        for sample_id, weight in iso)
        fl = math.fsum(weight * runs[(sample_id, "iso_service", row_id)]["dsec_fl_percent"]
                       for sample_id, weight in iso)
        summaries[row_id] = {
            "cycles": cycles,
            "energy_mj": math.fsum(energy_components_mj.values()),
            "accuracy": aee,
            "dsec_fl_percent": fl,
        }
        summaries[row_id].update(energy_components_mj)
    return summaries, samples, views


def _validate_typed_receipts(bundle_id, specs, raw, configs, summaries, measurement):
    # Energy is recomputed from every raw log, never accepted from a positive scalar alone.
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
            observed_component = _number(row[component], component, zero_ok=True)
            if not math.isclose(observed_component, summaries[row_id][component], rel_tol=1e-12, abs_tol=1e-15):
                raise RegistryError("typed energy component does not recompute from rooted raw logs")
            components.append(observed_component)
        expected = summaries[row_id]["energy_mj"]
        if (not math.isclose(row["total_energy_mj"], math.fsum(components), rel_tol=1e-12, abs_tol=1e-15) or
                not math.isclose(row["total_energy_mj"], expected, rel_tol=1e-12, abs_tol=1e-15)):
            raise RegistryError("typed energy row does not recompute from rooted raw logs")
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

    _, ppa, ppa_sha = _file_spec(specs["ppa_receipt"], "typed PPA receipt",
                                 "hw_autoresearch_nts07/results/")
    _exact(ppa, {"schema", "status", "technology_nm", "clock_period_ns", "rows"}, "typed PPA receipt")
    if ppa["schema"] != "m645.h67.logic_macro_sta_ppa_receipt.r1" or ppa["status"] != "PASS_TYPED" or ppa["technology_nm"] != 28 or ppa["clock_period_ns"] != 3.0:
        raise RegistryError("typed PPA receipt schema/status/process mismatch")
    ppa_rows = {}
    for row in ppa["rows"]:
        _exact(row, {"row_id", "configuration_manifest_sha256", "logic_area_mm2",
                     "sram_macro_area_mm2", "total_area_mm2", "setup_wns_ns", "hold_wns_ns",
                     "logic_report", "sram_report", "sta_report"}, "PPA row")
        row_id = row["row_id"]
        if row_id not in MANDATORY_ROW_IDS or row_id in ppa_rows or row["configuration_manifest_sha256"] != configs[row_id]["sha256"]:
            raise RegistryError("typed PPA row identity mismatch")
        logic = _number(row["logic_area_mm2"], "logic area", zero_ok=True)
        sram = _number(row["sram_macro_area_mm2"], "SRAM area", zero_ok=True)
        total = _number(row["total_area_mm2"], "total area")
        if not math.isclose(total, logic + sram, rel_tol=1e-12, abs_tol=1e-12):
            raise RegistryError("total PPA area is not logic plus SRAM macro")
        for field in ("setup_wns_ns", "hold_wns_ns"):
            if isinstance(row[field], bool) or not isinstance(row[field], (int, float)) or not math.isfinite(row[field]) or row[field] < 0:
                raise RegistryError("PPA receipt is not setup/hold closed")
        for report_name in ("logic_report", "sram_report", "sta_report"):
            _file_spec(row[report_name], "PPA raw " + report_name,
                       "hw_autoresearch_nts07/results/", ("application/json", "text/plain"))
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
        _exact(row, {"row_id", "role", "fidelity", "cycles", "energy_mj", "area_mm2", "accuracy"}, "direct result row")
        row_id = row["row_id"]
        if row_id not in MANDATORY_ROW_IDS or row_id in result_rows or (row["role"], row["fidelity"]) != specs_by_id[row_id]:
            raise RegistryError("direct result row identity/role/fidelity mismatch")
        expected = summaries[row_id]
        projection = {"cycles": expected["cycles"], "energy_mj": expected["energy_mj"],
                      "area_mm2": typed["ppa"][row_id]["total_area_mm2"], "accuracy": expected["accuracy"]}
        for field, value in projection.items():
            if not math.isclose(float(row[field]), float(value), rel_tol=1e-12, abs_tol=1e-12):
                raise RegistryError("direct result row is not an exact typed/raw projection")
        result_rows[row_id] = row
    if tuple(result_rows) != MANDATORY_ROW_IDS:
        raise RegistryError("direct result row order/content mismatch")
    if result["samples"] != samples:
        raise RegistryError("direct result samples are not recomputed raw cycles")
    if not isinstance(result["views"], dict) or tuple(result["views"]) != VIEW_NAMES:
        raise RegistryError("direct result views missing")
    for view in VIEW_NAMES:
        _check_aggregate_map(result["views"][view], views[view], "direct result view " + view)
    _check_aggregate_map(result["aggregates"], views["iso_service"], "direct result aggregate")
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
                      "population_manifest_sha256", "aggregation_manifest_sha256",
                      "sample_ids", "sequence_ids", "density_strata", "aggregates", "views"}, "coverage receipt")
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
    if len(expected_sequences) < 3 or not {"low", "mid", "high"}.issubset(set(expected_strata)):
        raise RegistryError("Table-A population requires >=3 sequences and low/mid/high density coverage")
    return {"result": result, "result_sha256": result_sha, "rows": result_rows,
            "completion_sha256": completion_sha, "coverage_sha256": coverage_sha}


def _parse_sha256sums(path, label):
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2:
            raise RegistryError(label + " contains a malformed SHA256SUMS line")
        digest, name = parts
        _sha(digest, label + " member SHA")
        if (not name or name.startswith("/") or ".." in Path(name).parts or
                name in rows):
            raise RegistryError(label + " contains an unsafe/duplicate member")
        rows[name] = digest
    if not rows:
        raise RegistryError(label + " is empty")
    return rows


def _validate_outer_seal(manifest_spec, outer_spec, label):
    manifest_path, _, manifest_sha = _file_spec(
        manifest_spec, label + " manifest", "hw_autoresearch_nts07/reviews/", ("text/plain",))
    outer_path, _, _ = _file_spec(
        outer_spec, label + " outer seal", "hw_autoresearch_nts07/reviews/", ("text/plain",))
    if manifest_path.parent != outer_path.parent:
        raise RegistryError(label + " manifest and outer seal are not colocated")
    expected = "%s  %s\n" % (manifest_sha, manifest_path.name)
    if outer_path.read_text(encoding="utf-8") != expected:
        raise RegistryError(label + " outer seal does not exactly seal its manifest")
    rows = _parse_sha256sums(manifest_path, label + " manifest")
    for name, digest in rows.items():
        member = manifest_path.parent / name
        if not member.is_file() or member.is_symlink() or _sha256(member) != digest:
            raise RegistryError(label + " sealed member SHA mismatch: " + name)
    return rows, manifest_sha, _sha256(outer_path)


def _validate_hammer(spec, authority_id, evidence):
    # Do not accept authority metadata from the candidate config or bundle.
    authority = TRUSTED_HAMMER_AUTHORITIES.get(authority_id)
    if authority is None:
        raise RegistryError("independent hammer authority is not code-trusted in this release")
    required_authority = {"request_manifest", "request_outer_seal", "review_manifest",
                          "review_outer_seal", "receipt"}
    _exact(authority, required_authority, "code-trusted hammer authority")
    _, _, request_outer_sha = _validate_outer_seal(
        authority["request_manifest"], authority["request_outer_seal"], "trusted request")
    review_rows, _, review_outer_sha = _validate_outer_seal(
        authority["review_manifest"], authority["review_outer_seal"], "trusted review")
    _, receipt, receipt_sha = _file_spec(spec, "independent hammer receipt",
                                         "hw_autoresearch_nts07/reviews/")
    if spec != authority["receipt"]:
        raise RegistryError("bundle hammer receipt differs from code-trusted authority root")
    receipt_path = _secure_file(spec["path"])
    if (receipt_path.parent != _secure_file(authority["review_manifest"]["path"]).parent or
            review_rows.get(receipt_path.name) != receipt_sha):
        raise RegistryError("trusted review outer seal does not transitively seal the hammer receipt")
    fields = {"schema", "status", "authority_id", "request_outer_seal_sha256",
              "bundle_evidence_sha256", "severity_counts",
              "independence", "recomputed_rows", "recomputed_aggregates",
              "recomputed_views", "authorization"}
    _exact(receipt, fields, "independent hammer receipt")
    if receipt["schema"] != "m645.h67.direct_unified.independent_hammer.r1" or receipt["status"] != "PASS_INDEPENDENT":
        raise RegistryError("independent hammer schema/status mismatch")
    if receipt["authority_id"] != authority_id:
        raise RegistryError("independent hammer authority identity mismatch")
    if receipt["request_outer_seal_sha256"] != request_outer_sha:
        raise RegistryError("independent hammer request outer-seal binding mismatch")
    if review_outer_sha != authority["review_outer_seal"]["sha256"]:
        raise RegistryError("independent hammer review outer-seal identity mismatch")
    expected_hashes = {name: evidence[name] for name in sorted(evidence)}
    if receipt["bundle_evidence_sha256"] != expected_hashes:
        raise RegistryError("independent hammer does not bind the complete raw/typed bundle")
    if receipt["severity_counts"] != {"P0": 0, "P1": 0}:
        raise RegistryError("independent hammer has blocking findings")
    expected_independence = {"author_receipt_used_as_authority": False,
                             "raw_logs_rehashed_and_recomputed": True,
                             "typed_receipts_recomputed": True,
                             "result_modified": False}
    if receipt["independence"] != expected_independence:
        raise RegistryError("independent hammer is self-attested or did not recompute roots")
    expected_auth = {"table_a_methodology_admitted": True,
                     "direct_unified_measurement_admitted": True,
                     "paper_headline_admitted": True}
    if receipt["authorization"] != expected_auth:
        raise RegistryError("independent hammer authorization missing")
    return receipt_sha, receipt


def _validate_bundle(bundle_id, bundle):
    _exact(bundle, BUNDLE_FIELDS, "Table-A direct bundle " + bundle_id)
    if bundle["schema"] != "m645.h67.rooted_direct_bundle.r1" or bundle["bundle_id"] != bundle_id:
        raise RegistryError("Table-A direct bundle schema/id mismatch")
    if not bundle_id.startswith("direct_unified_"):
        raise RegistryError("Table-A bundle ID must use direct_unified_ prefix")
    if bundle["m527_contract_sha256"] != M527_CONTRACT_SHA256:
        raise RegistryError("Table-A bundle does not bind the real frozen M527 r3 contract")
    # A code-level trust root is checked before author-controlled run content.
    # This makes a synchronized fabricated bundle unable to consume resources
    # or reach admission; all detailed validators below define the only future
    # path after an exact real hammer authority is frozen.
    authority_id = _string(bundle["independent_hammer_authority_id"], "hammer authority id")
    if authority_id not in TRUSTED_HAMMER_AUTHORITIES:
        raise RegistryError("independent hammer authority is not code-trusted in this release")

    _, _, producer_sha = _file_spec(bundle["producer"], "unified producer",
                                     "hw_autoresearch_nts07/system_simulator/",
                                     ("text/x-python", "text/plain"))
    _, _, simulator_sha = _file_spec(bundle["unified_simulator"], "unified simulator",
                                      "hw_autoresearch_nts07/system_simulator/",
                                      ("text/x-python", "text/plain"))
    _, invocation, invocation_sha = _file_spec(bundle["invocation_contract"], "invocation contract",
                                                "hw_autoresearch_nts07/contracts/")
    _exact(invocation, {"schema", "status", "producer_source_sha256", "simulator_source_sha256",
                        "checkpoint_sha256", "command_argv", "environment_allowlist",
                        "direct_cycles_required", "fallback_charged"}, "invocation contract")
    if (invocation["schema"] != "m645.h67.direct_invocation_contract.r1" or
            invocation["status"] != "FROZEN_BEFORE_RUN" or
            (invocation["producer_source_sha256"], invocation["simulator_source_sha256"],
             invocation["checkpoint_sha256"]) != (producer_sha, simulator_sha, CHECKPOINT_SHA256) or
            invocation["direct_cycles_required"] is not True or invocation["fallback_charged"] is not True or
            not isinstance(invocation["command_argv"], list) or not invocation["command_argv"]):
        raise RegistryError("invocation contract is not an executable pre-run root")
    measurement = _validate_measurement_identity(bundle["measurement_identity"], CHECKPOINT_SHA256)
    common = _validate_common_resource(bundle["common_resource_manifest"], producer_sha, simulator_sha, measurement)
    configs = _validate_configurations(bundle["configuration_manifests"], common, producer_sha,
                                       simulator_sha, invocation_sha, measurement)
    raw = _validate_raw_runs(bundle["raw_run_index"], configs, producer_sha, simulator_sha,
                             invocation_sha, measurement)
    recomputed = _recompute(raw["runs"], measurement)
    typed = _validate_typed_receipts(bundle_id, bundle, raw, configs, recomputed[0], measurement)
    result = _validate_result_and_receipts(bundle_id, bundle, configs, common, measurement,
                                           raw, recomputed, typed)
    evidence_hashes = {
        "common_resource_manifest": common["sha256"],
        "measurement_identity": measurement["sha256"],
        "raw_run_index": raw["sha256"],
        "direct_result": result["result_sha256"],
        "completion_receipt": result["completion_sha256"],
        "coverage_receipt": result["coverage_sha256"],
        "ppa_receipt": typed["ppa_sha256"],
        "energy_receipt": typed["energy_sha256"],
        "accuracy_receipt": typed["accuracy_sha256"],
    }
    for row_id in MANDATORY_ROW_IDS:
        evidence_hashes["configuration_manifest:" + row_id] = configs[row_id]["sha256"]
    for index, digest in enumerate(raw["run_hashes"]):
        evidence_hashes["raw_log:%06d" % index] = digest
    hammer_sha, hammer = _validate_hammer(bundle["independent_hammer_receipt"], authority_id,
                                          evidence_hashes)
    if (hammer["recomputed_rows"] != result["result"]["rows"] or
            hammer["recomputed_aggregates"] != result["result"]["aggregates"] or
            hammer["recomputed_views"] != result["result"]["views"]):
        raise RegistryError("trusted hammer recomputation differs from direct result")
    return {"rows": result["rows"], "result": result["result"],
            "hashes": evidence_hashes, "hammer_sha256": hammer_sha,
            "completion_sha256": result["completion_sha256"]}


def _validate_overlay(config):
    fields = {"schema", "date", "status", "purpose", "base_registry",
              "table_a_evidence_bundles", "table_a_rows", "claim_boundary", "protected_file"}
    _exact(config, fields, "M645 registry overlay")
    if config["schema"] != "m645.h67.paper_metric_registry.r4":
        raise RegistryError("unexpected M645 registry schema")
    _, base, base_sha = _file_spec(config["base_registry"], "sealed M635 base registry",
                                   "hw_autoresearch_nts07/system_simulator/config/")
    if base_sha != M635_CONFIG_SHA256 or config["base_registry"]["path"] != M635_CONFIG.relative_to(REPO_ROOT).as_posix():
        raise RegistryError("M645 must inherit the exact sealed M635 canonical registry")
    if base.get("schema") != "m635.h67.paper_metric_registry.r3" or base.get("table_a_evidence_bundles") != {}:
        raise RegistryError("sealed M635 base is not the zero-bundle canonical registry")
    protected = config["protected_file"]
    _exact(protected, {"path", "sha256"}, "protected file")
    if protected != {"path": DOCS359.relative_to(REPO_ROOT).as_posix(), "sha256": DOCS359_SHA256}:
        raise RegistryError("protected docs359 binding mismatch")
    if _sha256(_secure_file(protected["path"])) != DOCS359_SHA256:
        raise RegistryError("protected docs359 SHA drift")
    return base


def _evaluate(rows, bundles, claim, policy):
    eligible = []
    failures = {}
    for row in rows:
        reasons = list(row["blockers"])
        bundle_id = row["source_id"]
        if bundle_id is None:
            reasons.append("missing_rooted_direct_bundle")
        elif bundle_id not in bundles:
            reasons.append("unknown_or_untrusted_rooted_direct_bundle")
        elif row["row_id"] not in bundles[bundle_id]["rows"]:
            reasons.append("row_absent_from_rooted_result")
        else:
            bundle = bundles[bundle_id]
            evidence = bundle["rows"][row["row_id"]]
            projection = {
                "cycles": evidence["cycles"], "energy_mj": evidence["energy_mj"],
                "area_mm2": evidence["area_mm2"], "accuracy": evidence["accuracy"],
                "measurement_class": M635.ALLOWED_MEASUREMENT_CLASS,
                "population_id": bundle["result"]["population_id"],
                "workload_id": bundle["result"]["workload_id"],
                "resource_manifest_sha256": bundle["hashes"]["common_resource_manifest"],
                "completion_receipt_sha256": bundle["completion_sha256"],
                "decoder_complete": True, "memory_timing_included": True,
                "full_network_completion": True, "logic_sram_dram_energy_closed": True,
                "logic_macro_area_closed": True, "sta_closed": True,
                "independent_hammer_pass": True,
            }
            for field, expected in projection.items():
                if row[field] != expected:
                    reasons.append(field + "_not_exact_rooted_projection")
        if reasons:
            failures[row["row_id"]] = sorted(set(reasons))
        else:
            eligible.append(row["row_id"])
    mandatory = {row["row_id"]: row for row in rows if row["row_id"] in MANDATORY_ROW_IDS}
    bundle_ids = {mandatory[row_id]["source_id"] for row_id in MANDATORY_ROW_IDS}
    global_failures = []
    if len(bundle_ids) != 1 or None in bundle_ids:
        global_failures.append("mandatory_rows_not_bound_to_one_rooted_common_run")
    direct_speedup = None
    numerator = mandatory["dense96_fixed_t10"]["cycles"]
    candidate = mandatory["ours_exact"]["cycles"]
    if isinstance(numerator, (int, float)) and not isinstance(numerator, bool) and isinstance(candidate, (int, float)) and not isinstance(candidate, bool) and candidate > 0:
        direct_speedup = float(numerator) / float(candidate)
        if direct_speedup < policy["minimum_direct_speedup_for_accept"]:
            global_failures.append("direct_speedup_below_accept_floor")
    else:
        global_failures.append("direct_speedup_unavailable")
    admitted = set(MANDATORY_ROW_IDS).issubset(set(eligible)) and not global_failures
    if claim != {"table_a_admitted_rows": len(eligible), "paper_headline_admitted": admitted,
                 "analytical_range_admitted": False, "methodology_registry_only": True,
                 "paper_body_modified": False, "eda_or_gpu_run": False}:
        raise RegistryError("claim boundary disagrees with executable M645 gate")
    return {"admitted": admitted, "eligible_row_ids": eligible,
            "eligible_row_count": len(eligible), "required_row_count": len(MANDATORY_ROW_IDS),
            "row_failures": failures, "global_failures": sorted(set(global_failures)),
            "direct_speedup": direct_speedup,
            "fixed_numerator_row_id": M635.FIXED_NUMERATOR_ROW_ID,
            "strongest_same_page_baseline_row_id": M635.STRONGEST_BASELINE_ROW_ID,
            "candidate_row_id": M635.CANDIDATE_ROW_ID}


def build(config_path=DEFAULT_CONFIG):
    config = M635.load_json(Path(config_path), "M645 registry")
    base = _validate_overlay(config)
    # Re-run the sealed non-Table-A method checks rather than trusting its text.
    sources, source_docs = M635.validate_sources(base)
    base_rows = M635.validate_ladder_and_tables(base, set(sources))
    policy = M635.validate_policy(base)
    M635.validate_m518_binding(base, source_docs)
    analytical = M635.recompute_analytical(base)
    rows = copy.deepcopy(base_rows if config["table_a_rows"] is None else config["table_a_rows"])
    validation_copy = copy.deepcopy(base)
    validation_copy["table_a_schema"]["rows"] = copy.deepcopy(rows)
    M635.validate_ladder_and_tables(validation_copy, set(sources))
    bundles_spec = config["table_a_evidence_bundles"]
    if not isinstance(bundles_spec, dict):
        raise RegistryError("table_a_evidence_bundles must be an object")
    bundles = {bundle_id: _validate_bundle(bundle_id, bundle)
               for bundle_id, bundle in bundles_spec.items()}
    gate = _evaluate(rows, bundles, config["claim_boundary"], policy)
    return {
        "schema": "m645.h67.paper_metric_registry.r4.preview",
        "status": config["status"],
        "source_hashes_validated": sources,
        "trusted_hammer_authority_count": len(TRUSTED_HAMMER_AUTHORITIES),
        "table_a_evidence_bundle_count": len(bundles),
        "table_a": rows, "table_b": base["table_b_schema"]["rows"],
        "table_c": base["table_c_schema"]["rows"],
        "analytical_diagnostic": analytical, "headline_gate": gate,
        "claim_boundary": config["claim_boundary"],
        "protected_file_validated": config["protected_file"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--emit-json", action="store_true")
    args = parser.parse_args()
    try:
        result = build(args.config)
    except (RegistryError, RuntimeError) as exc:
        print("M645_REGISTRY_FAIL: %s" % exc)
        return 2
    if args.emit_json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    else:
        print("M645_REGISTRY_PASS sources=%d trusted_authorities=%d bundles=%d table_a_eligible=%d headline_admitted=%s analytical_admitted=%s" %
              (len(result["source_hashes_validated"]), result["trusted_hammer_authority_count"],
               result["table_a_evidence_bundle_count"], result["headline_gate"]["eligible_row_count"],
               str(result["headline_gate"]["admitted"]).lower(),
               str(result["analytical_diagnostic"]["admitted"]).lower()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
