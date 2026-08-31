#!/usr/bin/env python3
"""Build or verify the fail-closed M527 five-configuration payload.

This tool closes only the executable-configuration registry *payload* gap.  It
does not change the immutable M527 r3 contract gates and never admits a
headline, a waterfall, or a performance number.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[3]
M527_CONTRACT = ROOT / "hw_autoresearch_nts07/contracts/m527_h67_headline_baseline_ladder_contract_r3_20260827.json"
M527_CONTRACT_SHA256 = "83ea25e43b53d12800ac64e971069a682e3077411ff10851a7861636ef77355b"
DOCS359 = ROOT / "hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

PAYLOAD_SCHEMA = "m634_m527_configuration_payload_registry_v1"
COMMON_SCHEMA = "m527_h67_common_resource_manifest_v1"
CONFIG_MANIFEST_SCHEMA = "m527_h67_executable_configuration_manifest_v1"
MEASUREMENT_SCHEMA = "m634_h67_measurement_identity_binding_v1"
COMMON_SOURCE_SCHEMA = "m634_h67_common_resource_source_v1"
CONFIG_SOURCE_SCHEMA = "m634_h67_configuration_source_v1"
RECEIPT_SCHEMA = "m634_m527_configuration_payload_verification_receipt_v1"

CONFIGURATION_IDS = (
    "b0_dense96_fixed_t10",
    "b1_project_defined_ptb_like_structured_k1x8",
    "b2_exact_bit_sparse_k1",
    "b3_exact_bit_sparse_k1x8",
    "c123_ours_exact",
)

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
COMMON_MANIFEST_FIELDS = (
    "schema", "m527_contract", "docs359", "builder_source",
    "measurement_identity_manifest", "measurement_identity",
    "common_resource_source", "simulator_source", "resource_tuple",
    "charge_policy", "fallback_policy", "physical_service_width_note",
)
CONFIG_MANIFEST_FIELDS = (
    "schema", "configuration_id", "configuration_source_path",
    "configuration_source_sha256", "simulator_source_path",
    "simulator_source_sha256", "complete_trace_manifest_path",
    "complete_trace_manifest_sha256", "measurement_identity_manifest_path",
    "measurement_identity_manifest_sha256", "common_resource_manifest_path",
    "common_resource_manifest_sha256", "mechanism_enable_map",
    "optimized_operator_ids", "resource_tuple", "charge_policy",
    "fallback_policy", "claim_boundary",
)
REGISTRY_FIELDS = (
    "schema", "m527_contract_sha256", "common_resource_manifest",
    "configuration_manifests", "measurement_identity_manifest_sha256",
    "complete_trace_manifest_sha256", "simulator_source_sha256",
    "configuration_payload_ready", "m527_contract_admission_gate_current_value",
    "headline_admitted",
)
RECEIPT_FIELDS = (
    "schema", "status", "validated_configuration_ids",
    "common_resource_tuple_identical_across_five",
    "charge_policy_identical_and_fully_charged",
    "fallback_charges_cycles_traffic_energy_area", "source_hashes_verified",
    "double_seal_created_by_atomic_writer", "claim_boundary",
)

EXPECTED_MECHANISMS = {
    "b0_dense96_fixed_t10": {
        "dense_fixed_t10": True, "structured_group_scan": False,
        "exact_bit_sparse": False, "parent_product_capture": False,
        "typed_k8": False, "exact_atlif_service": False,
        "execution_service_limit_sources_per_cycle": 8,
    },
    "b1_project_defined_ptb_like_structured_k1x8": {
        "dense_fixed_t10": False, "structured_group_scan": True,
        "exact_bit_sparse": False, "parent_product_capture": False,
        "typed_k8": False, "exact_atlif_service": False,
        "execution_service_limit_sources_per_cycle": 8,
    },
    "b2_exact_bit_sparse_k1": {
        "dense_fixed_t10": False, "structured_group_scan": False,
        "exact_bit_sparse": True, "parent_product_capture": False,
        "typed_k8": False, "exact_atlif_service": False,
        "execution_service_limit_sources_per_cycle": 1,
    },
    "b3_exact_bit_sparse_k1x8": {
        "dense_fixed_t10": False, "structured_group_scan": False,
        "exact_bit_sparse": True, "parent_product_capture": False,
        "typed_k8": False, "exact_atlif_service": False,
        "execution_service_limit_sources_per_cycle": 8,
    },
    "c123_ours_exact": {
        "dense_fixed_t10": False, "structured_group_scan": False,
        "exact_bit_sparse": True, "parent_product_capture": True,
        "typed_k8": True, "exact_atlif_service": True,
        "execution_service_limit_sources_per_cycle": 8,
    },
}


class PayloadError(RuntimeError):
    """A fail-closed input, payload, or identity error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PayloadError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_object(items: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in items:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def strict_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                PayloadError("non-finite JSON token: " + token)
            ),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise PayloadError(f"cannot read strict JSON {path}: {exc}") from exc
    require(isinstance(value, dict), "JSON root must be an object: " + str(path))
    return value


def secure_file(path: Path) -> Path:
    path = path if path.is_absolute() else ROOT / path
    absolute = path.absolute()
    try:
        relative = absolute.relative_to(ROOT)
    except ValueError as exc:
        raise PayloadError("source escapes repository: " + str(path)) from exc
    cursor = ROOT
    for part in relative.parts:
        cursor = cursor / part
        require(not cursor.is_symlink(), "symlink source component refused: " + str(cursor))
    require(absolute.is_file(), "source is not a regular file: " + str(absolute))
    return absolute


def repo_path(path: Path) -> str:
    return secure_file(path).relative_to(ROOT).as_posix()


def sha_text(value: Any, label: str) -> str:
    require(isinstance(value, str) and len(value) == 64 and
            all(ch in "0123456789abcdef" for ch in value),
            label + " must be a non-null lowercase SHA256")
    return value


def exact_keys(value: Mapping[str, Any], expected: Sequence[str], label: str) -> None:
    require(set(value) == set(expected),
            f"{label} fields drift: expected {sorted(expected)}, got {sorted(value)}")


def source_entry(path_value: Any, sha_value: Any, label: str) -> Dict[str, str]:
    require(isinstance(path_value, str) and path_value, label + " path is null/empty")
    expected = sha_text(sha_value, label + " sha256")
    path = secure_file(Path(path_value))
    observed = sha256(path)
    require(observed == expected, f"{label} SHA mismatch: expected {expected}, observed {observed}")
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": observed}


def _positive_integer(value: Any, label: str, allow_zero: bool = False) -> int:
    require(isinstance(value, int) and not isinstance(value, bool), label + " must be integer")
    require(value >= (0 if allow_zero else 1), label + " out of range")
    return value


def verify_frozen_contract() -> Dict[str, Any]:
    contract_path = secure_file(M527_CONTRACT)
    docs_path = secure_file(DOCS359)
    require(sha256(contract_path) == M527_CONTRACT_SHA256, "M527 r3 contract SHA drift")
    require(sha256(docs_path) == DOCS359_SHA256, "docs/359 SHA drift")
    contract = strict_json(contract_path)
    require(contract.get("schema") == "m527_h67_headline_baseline_ladder_contract_v3",
            "M527 contract schema drift")
    schema = contract.get("configuration_manifest_schema")
    require(isinstance(schema, dict), "M527 configuration schema missing")
    require(schema.get("schema_exact") == CONFIG_MANIFEST_SCHEMA,
            "M527 configuration manifest schema drift")
    require(tuple(schema.get("resource_tuple_required_fields", [])) == RESOURCE_FIELDS,
            "M527 resource field order/content drift")
    require(tuple(schema.get("charge_policy_required_fields", [])) == CHARGE_FIELDS,
            "M527 charge field order/content drift")
    registry_ids = tuple(entry.get("configuration_id")
                         for entry in contract.get("configuration_registry", {}).get("entries", []))
    for config_id in CONFIGURATION_IDS:
        require(config_id in registry_ids, "M527 registry lacks required configuration: " + config_id)
    return contract


def verify_measurement_binding(path: Path, contract: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    path = secure_file(path)
    value = strict_json(path)
    required = (
        "schema", "m527_contract_sha256", "checkpoint_sha256",
        "complete_trace_manifest", "sequence_population_manifest",
        "aggregation_weight_manifest", "frame_definition", "density_metric",
        "density_bin_boundaries", "operator_ids",
    )
    exact_keys(value, required, "measurement identity")
    require(value["schema"] == MEASUREMENT_SCHEMA, "measurement identity schema drift")
    require(sha_text(value["m527_contract_sha256"], "measurement contract SHA") == M527_CONTRACT_SHA256,
            "measurement identity binds another M527 contract")
    require(sha_text(value["checkpoint_sha256"], "measurement checkpoint SHA") ==
            contract["identity"]["checkpoint_sha256"], "measurement checkpoint mismatch")
    resolved: Dict[str, Any] = {}
    for name in ("complete_trace_manifest", "sequence_population_manifest", "aggregation_weight_manifest"):
        entry = value[name]
        require(isinstance(entry, dict), name + " must be an object")
        exact_keys(entry, ("path", "sha256"), name)
        resolved[name] = source_entry(entry["path"], entry["sha256"], name)
    for name in ("frame_definition", "density_metric"):
        require(isinstance(value[name], str) and value[name].strip(), name + " is null/empty")
    bins = value["density_bin_boundaries"]
    require(isinstance(bins, list) and len(bins) >= 2, "density bins must have at least two values")
    require(all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in bins),
            "density bins contain non-finite/non-numeric value")
    require(all(left < right for left, right in zip(bins, bins[1:])), "density bins are not strictly increasing")
    operator_ids = value["operator_ids"]
    require(isinstance(operator_ids, list) and operator_ids, "operator_ids must be nonempty")
    require(all(isinstance(item, str) and item.strip() for item in operator_ids), "invalid operator id")
    require(len(operator_ids) == len(set(operator_ids)), "duplicate operator id")
    canonical = dict(value)
    canonical.update(resolved)
    return canonical, {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}


def verify_resource_source(path: Path, contract: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    path = secure_file(path)
    value = strict_json(path)
    exact_keys(value, ("schema", "resource_tuple", "charge_policy", "fallback_policy"),
               "common resource source")
    require(value["schema"] == COMMON_SOURCE_SCHEMA, "common resource source schema drift")
    resource = value["resource_tuple"]
    require(isinstance(resource, dict), "resource_tuple must be an object")
    exact_keys(resource, RESOURCE_FIELDS, "resource_tuple")
    axes = contract["common_resource_axes"]
    fixed = {
        "technology_nm": axes["technology_nm"],
        "clock_period_ns": axes["clock_period_ns"],
        "source_lanes": axes["source_lanes"],
        "service_width_sources_per_cycle": 8,
        "onchip_sram_bytes_total": axes["onchip_sram_bytes"],
        "dram_bandwidth_bytes_per_second_decimal": axes["dram_bandwidth_GB_per_s_decimal"] * 1_000_000_000,
        "dram_bytes_per_cycle": axes["dram_bytes_per_3ns_cycle"],
        "accumulator_bits": axes["accumulator_bits"],
    }
    for key, expected in fixed.items():
        require(resource[key] == expected, f"common resource {key} mismatch: {resource[key]} != {expected}")
    for key in RESOURCE_FIELDS[8:14] + RESOURCE_FIELDS[17:19]:
        _positive_integer(resource[key], "resource_tuple." + key)
    for key in RESOURCE_FIELDS[14:17]:
        require(isinstance(resource[key], str) and resource[key].strip(), "resource_tuple." + key + " empty")
    charge = value["charge_policy"]
    require(isinstance(charge, dict), "charge_policy must be an object")
    exact_keys(charge, CHARGE_FIELDS, "charge_policy")
    require(all(charge[key] is True for key in CHARGE_FIELDS), "every added resource/energy charge must be true")
    fallback = value["fallback_policy"]
    require(isinstance(fallback, dict), "fallback_policy must be an object")
    exact_keys(fallback, FALLBACK_FIELDS[:-1], "common fallback policy")
    require(fallback["mode"] == "EXECUTE_UNSUPPORTED_WORK_IN_THE_SAME_UNIFIED_MODEL",
            "fallback mode mismatch")
    require(all(fallback[key] is True for key in FALLBACK_FIELDS[1:5]),
            "fallback must charge cycles/traffic/energy/area")
    return value, {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}


def verify_configuration_source(path: Path, config_id: str,
                                operator_ids: Sequence[str]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    path = secure_file(path)
    value = strict_json(path)
    exact_keys(value, ("schema", "configuration_id", "mechanism_enable_map",
                       "optimized_operator_ids", "unsupported_operator_ids"),
               "configuration source " + config_id)
    require(value["schema"] == CONFIG_SOURCE_SCHEMA, "configuration source schema drift: " + config_id)
    require(value["configuration_id"] == config_id, "configuration source ID mismatch: " + config_id)
    mechanism = value["mechanism_enable_map"]
    require(isinstance(mechanism, dict), "mechanism map must be object: " + config_id)
    exact_keys(mechanism, MECHANISM_FIELDS, "mechanism map " + config_id)
    require(mechanism == EXPECTED_MECHANISMS[config_id], "mechanism map mismatch: " + config_id)
    optimized = value["optimized_operator_ids"]
    unsupported = value["unsupported_operator_ids"]
    for label, items in (("optimized", optimized), ("unsupported", unsupported)):
        require(isinstance(items, list), f"{config_id} {label} ids must be a list")
        require(all(isinstance(item, str) and item.strip() for item in items),
                f"{config_id} has invalid {label} id")
        require(len(items) == len(set(items)), f"{config_id} has duplicate {label} id")
    require(set(optimized).isdisjoint(unsupported), config_id + " optimized/fallback overlap")
    require(set(optimized) | set(unsupported) == set(operator_ids),
            config_id + " operator partition does not equal frozen trace universe")
    return value, {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}


def build_documents(measurement_path: Path, common_source_path: Path,
                    simulator_path: Path, configuration_paths: Mapping[str, Path]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Dict[str, Any]]]:
    contract = verify_frozen_contract()
    require(set(configuration_paths) == set(CONFIGURATION_IDS),
            "exactly five B0/B1/B2/B3/Ours configuration sources are required")
    measurement, measurement_ref = verify_measurement_binding(measurement_path, contract)
    common_source, common_source_ref = verify_resource_source(common_source_path, contract)
    simulator_path = secure_file(simulator_path)
    simulator_ref = {"path": simulator_path.relative_to(ROOT).as_posix(), "sha256": sha256(simulator_path)}
    sha_text(simulator_ref["sha256"], "simulator source SHA")

    sources: Dict[str, Tuple[Dict[str, Any], Dict[str, str]]] = {}
    for config_id in CONFIGURATION_IDS:
        sources[config_id] = verify_configuration_source(
            configuration_paths[config_id], config_id, measurement["operator_ids"])

    common_manifest = {
        "schema": COMMON_SCHEMA,
        "m527_contract": {"path": M527_CONTRACT.relative_to(ROOT).as_posix(), "sha256": M527_CONTRACT_SHA256},
        "docs359": {"path": DOCS359.relative_to(ROOT).as_posix(), "sha256": DOCS359_SHA256},
        "builder_source": {"path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
                           "sha256": sha256(Path(__file__).resolve())},
        "measurement_identity_manifest": measurement_ref,
        "measurement_identity": measurement,
        "common_resource_source": common_source_ref,
        "simulator_source": simulator_ref,
        "resource_tuple": common_source["resource_tuple"],
        "charge_policy": common_source["charge_policy"],
        "fallback_policy": common_source["fallback_policy"],
        "physical_service_width_note": "All five rows charge one physical K1x8 pool; B2 K1 is an execution cap, not deletion of seven services.",
    }
    common_bytes = (json.dumps(common_manifest, indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n").encode("utf-8")
    common_sha = hashlib.sha256(common_bytes).hexdigest()

    manifests: Dict[str, Dict[str, Any]] = {}
    for config_id in CONFIGURATION_IDS:
        source, source_ref = sources[config_id]
        fallback = dict(common_source["fallback_policy"])
        fallback["unsupported_operator_ids"] = source["unsupported_operator_ids"]
        manifests[config_id] = {
            "schema": CONFIG_MANIFEST_SCHEMA,
            "configuration_id": config_id,
            "configuration_source_path": source_ref["path"],
            "configuration_source_sha256": source_ref["sha256"],
            "simulator_source_path": simulator_ref["path"],
            "simulator_source_sha256": simulator_ref["sha256"],
            "complete_trace_manifest_path": measurement["complete_trace_manifest"]["path"],
            "complete_trace_manifest_sha256": measurement["complete_trace_manifest"]["sha256"],
            "measurement_identity_manifest_path": measurement_ref["path"],
            "measurement_identity_manifest_sha256": measurement_ref["sha256"],
            "common_resource_manifest_path": "common_resource_manifest.json",
            "common_resource_manifest_sha256": common_sha,
            "mechanism_enable_map": source["mechanism_enable_map"],
            "optimized_operator_ids": source["optimized_operator_ids"],
            "resource_tuple": common_source["resource_tuple"],
            "charge_policy": common_source["charge_policy"],
            "fallback_policy": fallback,
            "claim_boundary": {
                "configuration_payload_ready": True,
                "m527_contract_registry_gate_changed": False,
                "waterfall_admitted": False,
                "system_speedup": False,
                "paper_headline": False,
            },
        }
    return common_manifest, measurement, manifests


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")


def _seal(directory: Path, names: Sequence[str]) -> None:
    manifest = "".join(f"{sha256(directory / name)}  {name}\n" for name in sorted(names))
    (directory / "SHA256SUMS").write_text(manifest, encoding="utf-8")
    (directory / "SHA256SUMS.seal.sha256").write_text(
        f"{sha256(directory / 'SHA256SUMS')}  SHA256SUMS\n", encoding="utf-8")


def build_payload(measurement_path: Path, common_source_path: Path,
                  simulator_path: Path, configuration_paths: Mapping[str, Path],
                  output_dir: Path) -> Dict[str, Any]:
    common, measurement, configs = build_documents(
        measurement_path, common_source_path, simulator_path, configuration_paths)
    output_dir = output_dir.absolute()
    require(output_dir.parent.is_dir() and not output_dir.exists(),
            "output must be absent and its parent must exist")
    staging = Path(tempfile.mkdtemp(prefix=".m634_m527_payload_", dir=output_dir.parent))
    try:
        _write_json(staging / "common_resource_manifest.json", common)
        config_entries = []
        member_names = ["common_resource_manifest.json"]
        for config_id in CONFIGURATION_IDS:
            name = config_id + ".json"
            _write_json(staging / name, configs[config_id])
            member_names.append(name)
            config_entries.append({"configuration_id": config_id, "path": name,
                                   "sha256": sha256(staging / name)})
        registry = {
            "schema": PAYLOAD_SCHEMA,
            "m527_contract_sha256": M527_CONTRACT_SHA256,
            "common_resource_manifest": {"path": "common_resource_manifest.json",
                                         "sha256": sha256(staging / "common_resource_manifest.json")},
            "configuration_manifests": config_entries,
            "measurement_identity_manifest_sha256": common["measurement_identity_manifest"]["sha256"],
            "complete_trace_manifest_sha256": measurement["complete_trace_manifest"]["sha256"],
            "simulator_source_sha256": common["simulator_source"]["sha256"],
            "configuration_payload_ready": True,
            "m527_contract_admission_gate_current_value": False,
            "headline_admitted": False,
        }
        _write_json(staging / "registry.json", registry)
        member_names.append("registry.json")
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "status": "CONFIGURATION_PAYLOAD_COMPLETE__M527_HEADLINE_STILL_BLOCKED",
            "validated_configuration_ids": list(CONFIGURATION_IDS),
            "common_resource_tuple_identical_across_five": True,
            "charge_policy_identical_and_fully_charged": True,
            "fallback_charges_cycles_traffic_energy_area": True,
            "source_hashes_verified": True,
            "double_seal_created_by_atomic_writer": True,
            "claim_boundary": {
                "configuration_registry_payload": True,
                "fixed_numerator_receipt": False,
                "unified_measurement": False,
                "system_speedup": False,
                "paper_headline": False,
            },
        }
        _write_json(staging / "verification_receipt.json", receipt)
        member_names.append("verification_receipt.json")
        _seal(staging, member_names)
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return verify_payload(output_dir)


def _parse_sha_line(text: str, expected_name: str) -> str:
    parts = text.rstrip("\n").split("  ")
    require(len(parts) == 2 and parts[1] == expected_name, "invalid seal line for " + expected_name)
    return sha_text(parts[0], expected_name + " seal")


def verify_payload(payload_dir: Path) -> Dict[str, Any]:
    verify_frozen_contract()
    payload_dir = payload_dir.absolute()
    require(payload_dir.is_dir() and not payload_dir.is_symlink(), "payload directory missing/symlinked")
    expected_members = {"common_resource_manifest.json", "registry.json", "verification_receipt.json",
                        *(config_id + ".json" for config_id in CONFIGURATION_IDS)}
    actual = {path.name for path in payload_dir.iterdir() if path.name not in
              {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    require(actual == expected_members, "payload member set drift")
    for name in expected_members | {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
        path = payload_dir / name
        require(path.is_file() and not path.is_symlink(), "missing/symlinked payload member: " + name)
    manifest_sha = _parse_sha_line((payload_dir / "SHA256SUMS.seal.sha256").read_text(encoding="utf-8"),
                                  "SHA256SUMS")
    require(manifest_sha == sha256(payload_dir / "SHA256SUMS"), "outer seal mismatch")
    lines = (payload_dir / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    require(len(lines) == len(expected_members), "member manifest line count drift")
    sealed = {}
    for line in lines:
        parts = line.split("  ")
        require(len(parts) == 2 and parts[1] in expected_members and parts[1] not in sealed,
                "invalid/duplicate member manifest entry")
        sealed[parts[1]] = sha_text(parts[0], "member SHA")
    require(set(sealed) == expected_members, "member manifest set drift")
    for name, expected in sealed.items():
        require(sha256(payload_dir / name) == expected, "payload member SHA mismatch: " + name)

    registry = strict_json(payload_dir / "registry.json")
    exact_keys(registry, REGISTRY_FIELDS, "payload registry")
    require(registry.get("schema") == PAYLOAD_SCHEMA and registry.get("headline_admitted") is False,
            "registry schema/claim boundary drift")
    common = strict_json(payload_dir / "common_resource_manifest.json")
    exact_keys(common, COMMON_MANIFEST_FIELDS, "common resource manifest")
    require(common.get("schema") == COMMON_SCHEMA, "common manifest schema drift")
    require(registry["common_resource_manifest"]["sha256"] == sha256(payload_dir / "common_resource_manifest.json"),
            "registry/common manifest SHA mismatch")
    builder = source_entry(common["builder_source"]["path"], common["builder_source"]["sha256"],
                           "live M634 builder source")
    require(builder["path"] == Path(__file__).resolve().relative_to(ROOT).as_posix(),
            "payload binds another builder path")
    require(source_entry(common["measurement_identity_manifest"]["path"],
                         common["measurement_identity_manifest"]["sha256"],
                         "live measurement identity") is not None, "unreachable")
    measurement, _ = verify_measurement_binding(
        Path(common["measurement_identity_manifest"]["path"]), verify_frozen_contract())
    resource_source, _ = verify_resource_source(
        Path(common["common_resource_source"]["path"]), verify_frozen_contract())
    simulator = source_entry(common["simulator_source"]["path"], common["simulator_source"]["sha256"],
                             "live simulator source")
    require(common["resource_tuple"] == resource_source["resource_tuple"], "common resource source drift")
    require(common["charge_policy"] == resource_source["charge_policy"], "common charge source drift")
    require(common["fallback_policy"] == resource_source["fallback_policy"], "common fallback source drift")

    entries = registry.get("configuration_manifests")
    require(isinstance(entries, list) and len(entries) == 5, "registry configuration count drift")
    require(tuple(entry.get("configuration_id") for entry in entries) == CONFIGURATION_IDS,
            "registry configuration order/ID drift")
    common_sha = sha256(payload_dir / "common_resource_manifest.json")
    for entry in entries:
        config_id = entry["configuration_id"]
        name = config_id + ".json"
        require(entry == {"configuration_id": config_id, "path": name,
                          "sha256": sha256(payload_dir / name)},
                "registry configuration entry drift: " + config_id)
        manifest = strict_json(payload_dir / name)
        exact_keys(manifest, CONFIG_MANIFEST_FIELDS, "configuration manifest " + config_id)
        require(manifest.get("schema") == CONFIG_MANIFEST_SCHEMA and
                manifest.get("configuration_id") == config_id,
                "configuration manifest schema/ID drift: " + config_id)
        source, _ = verify_configuration_source(Path(manifest["configuration_source_path"]),
                                                config_id, measurement["operator_ids"])
        require(manifest["configuration_source_sha256"] ==
                sha256(secure_file(Path(manifest["configuration_source_path"]))),
                "configuration source SHA drift: " + config_id)
        require(manifest["simulator_source_sha256"] == simulator["sha256"],
                "simulator SHA differs across configurations")
        require(manifest["complete_trace_manifest_sha256"] ==
                measurement["complete_trace_manifest"]["sha256"],
                "trace SHA differs across configurations")
        require(manifest["common_resource_manifest_sha256"] == common_sha,
                "common resource SHA differs across configurations")
        require(manifest["resource_tuple"] == common["resource_tuple"],
                "resource tuple differs across configurations")
        require(manifest["charge_policy"] == common["charge_policy"],
                "charge policy differs across configurations")
        expected_fallback = dict(common["fallback_policy"])
        expected_fallback["unsupported_operator_ids"] = source["unsupported_operator_ids"]
        require(manifest["fallback_policy"] == expected_fallback,
                "fallback policy drift: " + config_id)
        require(manifest["mechanism_enable_map"] == EXPECTED_MECHANISMS[config_id],
                "mechanism manifest drift: " + config_id)
        boundary = manifest.get("claim_boundary", {})
        require(boundary.get("system_speedup") is False and boundary.get("paper_headline") is False,
                "configuration overclaims admission: " + config_id)
    receipt = strict_json(payload_dir / "verification_receipt.json")
    exact_keys(receipt, RECEIPT_FIELDS, "verification receipt")
    require(receipt.get("schema") == RECEIPT_SCHEMA and
            receipt.get("claim_boundary", {}).get("paper_headline") is False,
            "receipt schema/claim boundary drift")
    return {
        "status": "PASS_M634_CONFIGURATION_PAYLOAD__M527_HEADLINE_BLOCKED",
        "payload_dir": str(payload_dir),
        "configuration_count": 5,
        "member_manifest_sha256": sha256(payload_dir / "SHA256SUMS"),
        "outer_seal_file_sha256": sha256(payload_dir / "SHA256SUMS.seal.sha256"),
        "system_speedup": False,
        "paper_headline": False,
    }


def parse_configuration_sources(values: Sequence[str]) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for value in values:
        require("=" in value, "configuration source must be ID=PATH")
        config_id, raw_path = value.split("=", 1)
        require(config_id in CONFIGURATION_IDS and config_id not in result and raw_path,
                "invalid/duplicate configuration source: " + config_id)
        result[config_id] = Path(raw_path)
    require(set(result) == set(CONFIGURATION_IDS), "all five configuration sources are required")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    build = subparsers.add_parser("build")
    build.add_argument("--measurement-identity", type=Path, required=True)
    build.add_argument("--common-resource-source", type=Path, required=True)
    build.add_argument("--simulator-source", type=Path, required=True)
    build.add_argument("--configuration-source", action="append", default=[], metavar="ID=PATH")
    build.add_argument("--output-dir", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--payload-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command is None:
        parser.error("one of {build,validate} is required")
    try:
        if args.command == "build":
            result = build_payload(args.measurement_identity, args.common_resource_source,
                                   args.simulator_source,
                                   parse_configuration_sources(args.configuration_source),
                                   args.output_dir)
        else:
            result = verify_payload(args.payload_dir)
    except PayloadError as exc:
        print("FAIL_CLOSED_M634: " + str(exc))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
