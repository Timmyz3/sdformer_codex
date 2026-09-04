#!/usr/bin/env python3
"""Build/verify the M644 fail-closed M527 configuration-identity payload.

This supersedes the M634 r1 builder without modifying its sealed artifact.  A
PASS proves only that five executable configuration identities, their common
resource tuple, and an independently verified decoder-complete measurement
identity are mutually SHA-bound.  It never opens an M527 admission gate and
never admits a waterfall, system speedup, effective GOP/s, or paper headline.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[3]
BASE_BUILDER = ROOT / "hw_autoresearch_nts07/system_simulator/scripts/build_verify_m634_m527_configuration_payload.py"
BASE_BUILDER_SHA256 = "b53429d9444e44f33cb9a240f696a3d847323da1af7929ed43e473e87fa564fa"
_SPEC = importlib.util.spec_from_file_location("m634_frozen_base", str(BASE_BUILDER))
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("cannot load frozen M634 base")
BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(BASE)

PayloadError = BASE.PayloadError
M527_CONTRACT = BASE.M527_CONTRACT
M527_CONTRACT_SHA256 = BASE.M527_CONTRACT_SHA256
DOCS359 = BASE.DOCS359
DOCS359_SHA256 = BASE.DOCS359_SHA256
CONFIGURATION_IDS = BASE.CONFIGURATION_IDS
RESOURCE_FIELDS = BASE.RESOURCE_FIELDS
CHARGE_FIELDS = BASE.CHARGE_FIELDS
FALLBACK_FIELDS = BASE.FALLBACK_FIELDS
EXPECTED_MECHANISMS = BASE.EXPECTED_MECHANISMS
COMMON_SOURCE_SCHEMA = BASE.COMMON_SOURCE_SCHEMA
CONFIG_SOURCE_SCHEMA = BASE.CONFIG_SOURCE_SCHEMA
CONFIG_MANIFEST_SCHEMA = BASE.CONFIG_MANIFEST_SCHEMA

PAYLOAD_SCHEMA = "m644_m527_configuration_identity_payload_registry_v2"
COMMON_SCHEMA = "m527_h67_common_resource_manifest_v1"
MEASUREMENT_SCHEMA = "m644_h67_measurement_identity_binding_v2"
UPSTREAM_SCHEMA = "m644_h67_decoder_complete_semantic_verification_receipt_v1"
UPSTREAM_STATUS = "PASS_DECODER_COMPLETE_TRACE_SEMANTICS__NO_PERFORMANCE_ADMISSION"
RECEIPT_SCHEMA = "m644_m527_configuration_identity_payload_verification_receipt_v2"
RECEIPT_STATUS = "CONFIGURATION_IDENTITY_PAYLOAD_COMPLETE__M527_ADMISSION_GATES_STILL_FALSE"

MEASUREMENT_FIELDS = (
    "schema", "m527_contract_sha256", "checkpoint_sha256",
    "complete_trace_manifest", "sequence_population_manifest",
    "aggregation_weight_manifest", "upstream_semantic_verification_receipt",
    "frame_definition", "density_metric", "density_bin_boundaries",
    "population_scalar", "population_unit", "operator_ids",
)
UPSTREAM_FIELDS = (
    "schema", "status", "checkpoint_sha256",
    "complete_trace_manifest_sha256", "sequence_population_manifest_sha256",
    "aggregation_weight_manifest_sha256", "population_scalar",
    "population_unit", "frame_definition", "density_metric",
    "density_bin_boundaries", "operator_ids", "verification", "claim_boundary",
)
UPSTREAM_VERIFICATION = {
    "complete_trace_schema_verified": True,
    "decoder_trace_population_complete": True,
    "sequence_population_verified": True,
    "aggregation_weights_verified": True,
    "operator_universe_verified": True,
    "checkpoint_identity_verified": True,
}
UPSTREAM_CLAIM_BOUNDARY = {
    "semantic_identity_ready": True,
    "configuration_registry_ready": False,
    "fixed_numerator_ready": False,
    "unified_cycles_ready": False,
    "system_speedup": False,
    "paper_headline": False,
}
COMMON_MANIFEST_FIELDS = (
    "schema", "m527_contract", "docs359", "builder_source",
    "frozen_base_builder_source", "measurement_identity_manifest",
    "measurement_identity", "upstream_semantic_verification_receipt",
    "common_resource_source", "simulator_source", "resource_tuple",
    "charge_policy", "fallback_policy", "physical_service_width_note",
)
CONFIG_MANIFEST_FIELDS = BASE.CONFIG_MANIFEST_FIELDS
REGISTRY_FIELDS = (
    "schema", "m527_contract_sha256", "common_resource_manifest",
    "configuration_manifests", "measurement_identity_manifest_sha256",
    "upstream_semantic_verification_receipt_sha256",
    "complete_trace_manifest_sha256", "simulator_source_sha256",
    "configuration_identity_payload_ready",
    "m527_contract_admission_gate_current_value", "waterfall_admitted",
    "system_speedup", "headline_admitted",
)
RECEIPT_FIELDS = (
    "schema", "status", "validated_configuration_ids",
    "common_resource_tuple_identical_across_five",
    "charge_policy_identical_and_fully_charged",
    "fallback_charges_cycles_traffic_energy_area",
    "all_live_source_paths_and_hashes_verified",
    "upstream_decoder_complete_semantics_verified",
    "staging_verified_before_atomic_publish", "double_seal_verified",
    "claim_boundary",
)
CONFIG_CLAIM_BOUNDARY = {
    "configuration_identity_payload_ready": True,
    "m527_contract_registry_gate_changed": False,
    "waterfall_admitted": False,
    "system_speedup": False,
    "effective_gops_admitted": False,
    "paper_headline": False,
}
RECEIPT_CLAIM_BOUNDARY = {
    "configuration_identity_payload_ready": True,
    "m527_configuration_registry_ready": False,
    "fixed_numerator_receipt": False,
    "unified_measurement": False,
    "waterfall_admitted": False,
    "system_speedup": False,
    "effective_gops_admitted": False,
    "paper_headline": False,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PayloadError(message)


def sha256(path: Path) -> str:
    return BASE.sha256(path)


def strict_json(path: Path) -> Dict[str, Any]:
    return BASE.strict_json(path)


def exact_keys(value: Mapping[str, Any], expected: Sequence[str], label: str) -> None:
    BASE.exact_keys(value, expected, label)


def secure_file(path: Path) -> Path:
    return BASE.secure_file(path)


def source_entry(path_value: Any, sha_value: Any, label: str) -> Dict[str, str]:
    return BASE.source_entry(path_value, sha_value, label)


def verify_frozen_contract() -> Dict[str, Any]:
    require(sha256(secure_file(BASE_BUILDER)) == BASE_BUILDER_SHA256,
            "frozen M634 base builder SHA drift")
    return BASE.verify_frozen_contract()


def _positive_integer(value: Any, label: str) -> int:
    require(isinstance(value, int) and not isinstance(value, bool) and value > 0,
            label + " must be a positive integer")
    return value


def _ref(path: Path) -> Dict[str, str]:
    path = secure_file(path)
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _json_sha(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _repo_confined_no_symlink_path(path: Path, require_directory: bool) -> Path:
    absolute = (path if path.is_absolute() else ROOT / path).absolute()
    try:
        relative = absolute.relative_to(ROOT)
    except ValueError as exc:
        raise PayloadError("path escapes repository: " + str(path)) from exc
    cursor = ROOT
    require(not cursor.is_symlink(), "repository root is symlinked")
    parts = relative.parts if require_directory else relative.parts[:-1]
    for part in parts:
        cursor = cursor / part
        require(not cursor.is_symlink(), "symlink path component refused: " + str(cursor))
    if require_directory:
        require(absolute.is_dir() and not absolute.is_symlink(),
                "payload directory missing/symlinked")
    else:
        require(not absolute.is_symlink(), "output leaf symlink refused: " + str(absolute))
        require(absolute.parent.is_dir() and not absolute.parent.is_symlink(),
                "output parent must be an existing real directory")
        require(not absolute.exists(), "output must be absent")
    return absolute


def verify_upstream_receipt(path: Path, measurement: Mapping[str, Any],
                            contract: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    path = secure_file(path)
    value = strict_json(path)
    exact_keys(value, UPSTREAM_FIELDS, "upstream semantic-verification receipt")
    require(value["schema"] == UPSTREAM_SCHEMA, "upstream receipt schema drift")
    require(value["status"] == UPSTREAM_STATUS, "upstream receipt status is not PASS")
    require(value["verification"] == UPSTREAM_VERIFICATION,
            "upstream semantic verification proof drift")
    require(value["claim_boundary"] == UPSTREAM_CLAIM_BOUNDARY,
            "upstream receipt overclaims admission")
    expected = {
        "checkpoint_sha256": contract["identity"]["checkpoint_sha256"],
        "complete_trace_manifest_sha256": measurement["complete_trace_manifest"]["sha256"],
        "sequence_population_manifest_sha256": measurement["sequence_population_manifest"]["sha256"],
        "aggregation_weight_manifest_sha256": measurement["aggregation_weight_manifest"]["sha256"],
        "population_scalar": measurement["population_scalar"],
        "population_unit": measurement["population_unit"],
        "frame_definition": measurement["frame_definition"],
        "density_metric": measurement["density_metric"],
        "density_bin_boundaries": measurement["density_bin_boundaries"],
        "operator_ids": measurement["operator_ids"],
    }
    for key, expected_value in expected.items():
        require(value[key] == expected_value, "upstream receipt identity drift: " + key)
    return value, _ref(path)


def verify_measurement_binding(path: Path, contract: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str], Dict[str, Any]]:
    path = secure_file(path)
    value = strict_json(path)
    exact_keys(value, MEASUREMENT_FIELDS, "measurement identity")
    require(value["schema"] == MEASUREMENT_SCHEMA, "measurement identity schema drift")
    require(value["m527_contract_sha256"] == M527_CONTRACT_SHA256,
            "measurement identity binds another M527 contract")
    require(value["checkpoint_sha256"] == contract["identity"]["checkpoint_sha256"],
            "measurement checkpoint mismatch")
    canonical = dict(value)
    for name in ("complete_trace_manifest", "sequence_population_manifest",
                 "aggregation_weight_manifest", "upstream_semantic_verification_receipt"):
        entry = value[name]
        require(isinstance(entry, dict), name + " must be an object")
        exact_keys(entry, ("path", "sha256"), name)
        canonical[name] = source_entry(entry["path"], entry["sha256"], name)
    for name in ("frame_definition", "density_metric", "population_unit"):
        require(isinstance(value[name], str) and value[name].strip(), name + " is null/empty")
    _positive_integer(value["population_scalar"], "population_scalar")
    bins = value["density_bin_boundaries"]
    require(isinstance(bins, list) and len(bins) >= 2, "density bins incomplete")
    require(all(isinstance(item, (int, float)) and not isinstance(item, bool) and
                math.isfinite(item) for item in bins),
            "density bins must be finite numeric values")
    require(all(float(left) < float(right) for left, right in zip(bins, bins[1:])),
            "density bins are not strictly increasing")
    operator_ids = value["operator_ids"]
    require(isinstance(operator_ids, list) and operator_ids and
            all(isinstance(item, str) and item.strip() for item in operator_ids) and
            len(operator_ids) == len(set(operator_ids)), "operator universe invalid")
    upstream, upstream_ref = verify_upstream_receipt(
        Path(canonical["upstream_semantic_verification_receipt"]["path"]), canonical, contract)
    require(upstream_ref == canonical["upstream_semantic_verification_receipt"],
            "upstream receipt path/SHA drift")
    return canonical, _ref(path), upstream


def build_expected_documents(measurement_path: Path, common_source_path: Path,
                             simulator_path: Path,
                             configuration_paths: Mapping[str, Path]) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    contract = verify_frozen_contract()
    require(set(configuration_paths) == set(CONFIGURATION_IDS),
            "exactly five B0/B1/B2/B3/Ours configuration sources are required")
    measurement, measurement_ref, upstream = verify_measurement_binding(measurement_path, contract)
    common_source, common_source_ref = BASE.verify_resource_source(common_source_path, contract)
    simulator_ref = _ref(simulator_path)
    sources = {}
    for config_id in CONFIGURATION_IDS:
        sources[config_id] = BASE.verify_configuration_source(
            configuration_paths[config_id], config_id, measurement["operator_ids"])

    common = {
        "schema": COMMON_SCHEMA,
        "m527_contract": _ref(M527_CONTRACT),
        "docs359": _ref(DOCS359),
        "builder_source": _ref(Path(__file__).resolve()),
        "frozen_base_builder_source": _ref(BASE_BUILDER),
        "measurement_identity_manifest": measurement_ref,
        "measurement_identity": measurement,
        "upstream_semantic_verification_receipt": measurement["upstream_semantic_verification_receipt"],
        "common_resource_source": common_source_ref,
        "simulator_source": simulator_ref,
        "resource_tuple": common_source["resource_tuple"],
        "charge_policy": common_source["charge_policy"],
        "fallback_policy": common_source["fallback_policy"],
        "physical_service_width_note": "All five rows charge one physical K1x8 pool; B2 K1 is an execution cap, not deletion of seven services.",
    }
    common_sha = _json_sha(common)
    configs = {}
    for config_id in CONFIGURATION_IDS:
        source, source_ref = sources[config_id]
        fallback = dict(common_source["fallback_policy"])
        fallback["unsupported_operator_ids"] = source["unsupported_operator_ids"]
        configs[config_id] = {
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
            "claim_boundary": CONFIG_CLAIM_BOUNDARY,
        }

    config_entries = []
    for config_id in CONFIGURATION_IDS:
        name = config_id + ".json"
        config_entries.append({
            "configuration_id": config_id,
            "path": name,
            "sha256": _json_sha(configs[config_id]),
        })
    registry = {
        "schema": PAYLOAD_SCHEMA,
        "m527_contract_sha256": M527_CONTRACT_SHA256,
        "common_resource_manifest": {"path": "common_resource_manifest.json", "sha256": common_sha},
        "configuration_manifests": config_entries,
        "measurement_identity_manifest_sha256": measurement_ref["sha256"],
        "upstream_semantic_verification_receipt_sha256": measurement["upstream_semantic_verification_receipt"]["sha256"],
        "complete_trace_manifest_sha256": measurement["complete_trace_manifest"]["sha256"],
        "simulator_source_sha256": simulator_ref["sha256"],
        "configuration_identity_payload_ready": True,
        "m527_contract_admission_gate_current_value": False,
        "waterfall_admitted": False,
        "system_speedup": False,
        "headline_admitted": False,
    }
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": RECEIPT_STATUS,
        "validated_configuration_ids": list(CONFIGURATION_IDS),
        "common_resource_tuple_identical_across_five": True,
        "charge_policy_identical_and_fully_charged": True,
        "fallback_charges_cycles_traffic_energy_area": True,
        "all_live_source_paths_and_hashes_verified": True,
        "upstream_decoder_complete_semantics_verified": True,
        "staging_verified_before_atomic_publish": True,
        "double_seal_verified": True,
        "claim_boundary": RECEIPT_CLAIM_BOUNDARY,
    }
    # Keep the upstream object live and validated even though only its ref is embedded.
    require(upstream["status"] == UPSTREAM_STATUS, "unreachable upstream status drift")
    return common, configs, registry, receipt


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(_json_bytes(value))


def _seal(directory: Path, names: Sequence[str]) -> None:
    manifest = "".join("{}  {}\n".format(sha256(directory / name), name)
                       for name in sorted(names))
    (directory / "SHA256SUMS").write_text(manifest, encoding="utf-8")
    (directory / "SHA256SUMS.seal.sha256").write_text(
        "{}  SHA256SUMS\n".format(sha256(directory / "SHA256SUMS")), encoding="utf-8")


def _parse_sha_line(text: str, expected_name: str) -> str:
    parts = text.rstrip("\n").split("  ")
    require(len(parts) == 2 and parts[1] == expected_name,
            "invalid seal line for " + expected_name)
    return BASE.sha_text(parts[0], expected_name + " seal")


def _verify_seals(payload_dir: Path) -> None:
    expected_members = {"common_resource_manifest.json", "registry.json",
                        "verification_receipt.json"}
    expected_members.update(config_id + ".json" for config_id in CONFIGURATION_IDS)
    actual = {path.name for path in payload_dir.iterdir()
              if path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    require(actual == expected_members, "payload member set drift")
    for name in expected_members | {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
        path = payload_dir / name
        require(path.is_file() and not path.is_symlink(),
                "missing/symlinked payload member: " + name)
    manifest_sha = _parse_sha_line(
        (payload_dir / "SHA256SUMS.seal.sha256").read_text(encoding="utf-8"), "SHA256SUMS")
    require(manifest_sha == sha256(payload_dir / "SHA256SUMS"), "outer seal mismatch")
    lines = (payload_dir / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    require(len(lines) == len(expected_members), "member manifest line count drift")
    sealed = {}
    for line in lines:
        parts = line.split("  ")
        require(len(parts) == 2 and parts[1] in expected_members and parts[1] not in sealed,
                "invalid/duplicate member manifest entry")
        sealed[parts[1]] = BASE.sha_text(parts[0], "member SHA")
    require(set(sealed) == expected_members, "member manifest set drift")
    for name, expected in sealed.items():
        require(sha256(payload_dir / name) == expected, "payload member SHA mismatch: " + name)


def verify_payload(payload_dir: Path, measurement_path: Path,
                   common_source_path: Path, simulator_path: Path,
                   configuration_paths: Mapping[str, Path]) -> Dict[str, Any]:
    payload_dir = _repo_confined_no_symlink_path(payload_dir, True)
    _verify_seals(payload_dir)
    common, configs, registry, receipt = build_expected_documents(
        measurement_path, common_source_path, simulator_path, configuration_paths)
    actual_common = strict_json(payload_dir / "common_resource_manifest.json")
    require(actual_common == common,
            "common resource manifest differs from live-source reconstruction")
    for config_id in CONFIGURATION_IDS:
        actual = strict_json(payload_dir / (config_id + ".json"))
        require(actual == configs[config_id],
                "configuration manifest differs from live-source reconstruction: " + config_id)
    require(strict_json(payload_dir / "registry.json") == registry,
            "registry differs from live-source reconstruction")
    require(strict_json(payload_dir / "verification_receipt.json") == receipt,
            "verification receipt differs from exact non-admission receipt")
    return {
        "status": "PASS_M644_CONFIGURATION_IDENTITY_PAYLOAD__ALL_M527_ADMISSION_GATES_BLOCKED",
        "payload_dir": str(payload_dir),
        "configuration_count": 5,
        "member_manifest_sha256": sha256(payload_dir / "SHA256SUMS"),
        "outer_seal_file_sha256": sha256(payload_dir / "SHA256SUMS.seal.sha256"),
        "configuration_identity_payload_ready": True,
        "m527_configuration_registry_ready": False,
        "waterfall_admitted": False,
        "system_speedup": False,
        "paper_headline": False,
    }


def _post_publish_verify(payload_dir: Path, measurement_path: Path,
                         common_source_path: Path, simulator_path: Path,
                         configuration_paths: Mapping[str, Path]) -> Dict[str, Any]:
    return verify_payload(payload_dir, measurement_path, common_source_path,
                          simulator_path, configuration_paths)


def _quarantine_failed_publish(output_dir: Path, reason: Exception) -> Path:
    token = hashlib.sha256(os.urandom(32)).hexdigest()[:16]
    quarantine = output_dir.parent / (output_dir.name + ".m644_quarantine_" + token)
    try:
        os.replace(str(output_dir), str(quarantine))
        failure = {
            "schema": "m644_post_publish_failure_receipt_v1",
            "status": "QUARANTINED_POST_PUBLISH_VERIFICATION_FAILURE",
            "canonical_output_removed": True,
            "error": str(reason),
            "claim_boundary": {"payload_ready": False, "system_speedup": False,
                               "paper_headline": False},
        }
        _write_json(quarantine / "POST_PUBLISH_FAILURE.json", failure)
    except Exception:
        if output_dir.exists() and not output_dir.is_symlink():
            shutil.rmtree(str(output_dir), ignore_errors=True)
        raise
    return quarantine


def build_payload(measurement_path: Path, common_source_path: Path,
                  simulator_path: Path, configuration_paths: Mapping[str, Path],
                  output_dir: Path) -> Dict[str, Any]:
    output_dir = _repo_confined_no_symlink_path(output_dir, False)
    common, configs, registry, receipt = build_expected_documents(
        measurement_path, common_source_path, simulator_path, configuration_paths)
    staging = Path(tempfile.mkdtemp(prefix=".m644_m527_payload_", dir=str(output_dir.parent)))
    published = False
    try:
        _write_json(staging / "common_resource_manifest.json", common)
        names = ["common_resource_manifest.json"]
        for config_id in CONFIGURATION_IDS:
            name = config_id + ".json"
            _write_json(staging / name, configs[config_id])
            names.append(name)
        _write_json(staging / "registry.json", registry)
        _write_json(staging / "verification_receipt.json", receipt)
        names.extend(["registry.json", "verification_receipt.json"])
        _seal(staging, names)
        # The full live-source verifier runs before the canonical name exists.
        verify_payload(staging, measurement_path, common_source_path,
                       simulator_path, configuration_paths)
        os.replace(str(staging), str(output_dir))
        published = True
        try:
            return _post_publish_verify(output_dir, measurement_path, common_source_path,
                                        simulator_path, configuration_paths)
        except Exception as exc:
            quarantine = _quarantine_failed_publish(output_dir, exc)
            raise PayloadError("post-publish verification failed; quarantined at " +
                               str(quarantine)) from exc
    except Exception:
        if not published:
            shutil.rmtree(str(staging), ignore_errors=True)
        raise


def parse_configuration_sources(values: Sequence[str]) -> Dict[str, Path]:
    return BASE.parse_configuration_sources(values)


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--measurement-identity", type=Path, required=True)
    parser.add_argument("--common-resource-source", type=Path, required=True)
    parser.add_argument("--simulator-source", type=Path, required=True)
    parser.add_argument("--configuration-source", action="append", default=[], metavar="ID=PATH")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    build = subparsers.add_parser("build")
    _add_source_arguments(build)
    build.add_argument("--output-dir", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    _add_source_arguments(validate)
    validate.add_argument("--payload-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command is None:
        parser.error("one of {build,validate} is required")
    try:
        configs = parse_configuration_sources(args.configuration_source)
        if args.command == "build":
            result = build_payload(args.measurement_identity, args.common_resource_source,
                                   args.simulator_source, configs, args.output_dir)
        else:
            result = verify_payload(args.payload_dir, args.measurement_identity,
                                    args.common_resource_source, args.simulator_source, configs)
    except PayloadError as exc:
        print("FAIL_CLOSED_M644: " + str(exc))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
