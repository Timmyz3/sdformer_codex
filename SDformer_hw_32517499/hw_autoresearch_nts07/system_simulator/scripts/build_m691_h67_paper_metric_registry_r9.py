#!/usr/bin/env python3
"""Build the M691 Table-A registry r9 canonical zero overlay."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


class RegistryError(ValueError):
    pass


REPO_ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = REPO_ROOT / "hw_autoresearch_nts07"
R8_BUILDER = HW_ROOT / "system_simulator/scripts/build_m671_h67_paper_metric_registry_r8.py"
R8_BUILDER_SHA256 = "67b88cfbed3f3250798b35ef763e476569a72c667ed775c95791f34674c3c91c"
R8_CONFIG = HW_ROOT / "system_simulator/config/m671_h67_paper_metric_registry_r8_20260828.json"
R8_CONFIG_SHA256 = "d924b7ed3c91288a60eb4b895af5eaac099da5ebc188b43b6d580a5b826a57c7"
EXTRACTOR = HW_ROOT / "system_simulator/scripts/extract_m691_native_synopsys_run_provenance.py"
EXTRACTOR_SHA256 = "3d8ee74b58df9ecdeb1ed8fb87c7feb3cbf3a6ba81ec49b6b972557d16fec420"
DEFAULT_CONFIG = HW_ROOT / "system_simulator/config/m691_h67_paper_metric_registry_r9_20260828.json"
DEFAULT_CONFIG_SHA256 = "ffa3e72fa3252a0726372b856ec1a20edd1ac28b553e6f3115657e17dfbcda88"
M687_REVIEW = HW_ROOT / "reviews/m687_m684_table_a_registry_r8_fresh_hammer_r1_20260828/review.json"
M687_REVIEW_SHA256 = "84d926e8279ee4379f803f1fafa9c1a0ae5f3b351423ffbf8a4ff388b16bc868"
M687_MANIFEST = M687_REVIEW.parent / "SHA256SUMS"
M687_MANIFEST_SHA256 = "0b517fdd6567a5e0a72857fdc878387c9d9901104520e10e9605c02025cce205"
M687_OUTER = M687_REVIEW.parent / "SHA256SUMS.seal.sha256"
M687_OUTER_SHA256 = "a386634c36d4d782a542a203fcb280cab94a833c1bafcd1f4b56a37f9f5770a6"
DOCS359 = HW_ROOT / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(name, path, expected_sha):
    if _sha256(path) != expected_sha:
        raise RegistryError("sealed dependency SHA drift: " + str(path))
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RegistryError("cannot import sealed dependency")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R8 = _load("m691_sealed_r8_builder", R8_BUILDER, R8_BUILDER_SHA256)
EX = _load("m691_r9_extractor", EXTRACTOR, EXTRACTOR_SHA256)
ROW_TO_CONFIGURATION = dict(R8.ROW_TO_M527_CONFIGURATION)


def _exact(value, fields, label):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise RegistryError(label + " fields differ")


def _validate_m687_root():
    if (_sha256(M687_REVIEW) != M687_REVIEW_SHA256 or
            _sha256(M687_MANIFEST) != M687_MANIFEST_SHA256 or
            _sha256(M687_OUTER) != M687_OUTER_SHA256):
        raise RegistryError("sealed M687 root drift")
    review = json.loads(M687_REVIEW.read_text(encoding="utf-8"))
    if (review.get("verdict") !=
            "PASS_CANONICAL_ZERO__NO_GO_PRODUCTION_ADMISSION__R9_REPAIR_REQUIRED" or
            review.get("severity_counts") != {"P0": 0, "P1": 4, "P2": 1}):
        raise RegistryError("sealed M687 semantic root mismatch")
    if M687_OUTER.read_text(encoding="utf-8") != M687_MANIFEST_SHA256 + "  SHA256SUMS\n":
        raise RegistryError("sealed M687 outer seal mismatch")


def _validate_run_manifest(spec, expected_row=None):
    path = EX._file_spec(spec, "r9 production run manifest", ("application/json",),
                         "hw_autoresearch_nts07/results/")
    manifest = EX._load_json(path, "r9 production run manifest")
    row_id = manifest.get("row_id")
    if expected_row is not None and row_id != expected_row:
        raise RegistryError("r9 production row mismatch")
    if row_id not in ROW_TO_CONFIGURATION:
        raise RegistryError("r9 production row is not mandatory")
    if manifest.get("m527_configuration_id") != ROW_TO_CONFIGURATION[row_id]:
        raise RegistryError("r9 production configuration identity mismatch")
    expected_scope = R8._map_sha(R8._required_operator_scope())
    if (manifest.get("operator_scope_sha256") != expected_scope or
            manifest.get("design_name") != "h67_table_a_" + ROW_TO_CONFIGURATION[row_id]):
        raise RegistryError("r9 full-scope/design identity mismatch")
    try:
        extracted = EX.extract_from_manifest(spec["path"])
    except (OSError, EX.ExtractionError) as exc:
        raise RegistryError("r9 production extraction failed: %s" % exc)
    if (extracted["run_identity"]["operator_scope_sha256"] != expected_scope or
            extracted["run_identity"]["design_name"] != manifest["design_name"] or
            extracted.get("area_mode") != "DC_TOTAL_INCLUDES_MACROS"):
        raise RegistryError("r9 production derived identity mismatch")
    if (extracted["memory_inventory"]["macro_rounded_total_bytes"] != 245760 or
            extracted["memory_inventory"]["macros"][2]["port_type"] != "1R1W"):
        raise RegistryError("r9 production memory closure mismatch")
    total_area = (extracted["values"]["logic_area_mm2"] +
                  extracted["values"]["sram_macro_area_mm2"])
    if total_area <= 0.0 or extracted["values"]["total_power_mw"] <= 0.0:
        raise RegistryError("r9 production physical values are not positive")
    return {"row_id": row_id, "run_manifest_sha256": _sha256(path),
            "production_proof_sha256": extracted["production_proof_sha256"],
            "total_area_mm2": total_area,
            "total_power_mw": extracted["values"]["total_power_mw"]}


def build(config_path=DEFAULT_CONFIG):
    _validate_m687_root()
    if (_sha256(R8_CONFIG) != R8_CONFIG_SHA256 or
            _sha256(EXTRACTOR) != EXTRACTOR_SHA256 or
            _sha256(DOCS359) != DOCS359_SHA256):
        raise RegistryError("r9 frozen source/config/protected identity drift")
    path = Path(config_path)
    if path == DEFAULT_CONFIG and _sha256(path) != DEFAULT_CONFIG_SHA256:
        raise RegistryError("canonical r9 config SHA drift")
    config = json.loads(path.read_text(encoding="utf-8"))
    _exact(config, {"schema", "date", "status", "purpose", "sealed_r8_registry",
                    "production_run_manifests", "claim_boundary", "protected_file"},
           "r9 registry config")
    if (config["schema"] != "m691.h67.paper_metric_registry.r9" or
            config["sealed_r8_registry"] != {
                "path": R8_CONFIG.relative_to(REPO_ROOT).as_posix(),
                "sha256": R8_CONFIG_SHA256, "media_type": "application/json"} or
            config["protected_file"] != {
                "path": DOCS359.relative_to(REPO_ROOT).as_posix(),
                "sha256": DOCS359_SHA256}):
        raise RegistryError("r9 overlay identity mismatch")
    expected_boundary = {
        "trusted_hammer_authorities": 0, "table_a_evidence_bundles": 0,
        "table_a_admitted_rows": 0, "paper_headline_admitted": False,
        "analytical_range_admitted": False, "production_measurement_admitted": False,
        "methodology_registry_only": True, "eda_or_gpu_run": False,
    }
    if config["claim_boundary"] != expected_boundary:
        raise RegistryError("r9 canonical claim boundary mismatch")
    runs = config["production_run_manifests"]
    if not isinstance(runs, dict):
        raise RegistryError("production run manifest map must be an object")
    validated = {}
    for row_id, spec in runs.items():
        if row_id in validated:
            raise RegistryError("duplicate production row")
        validated[row_id] = _validate_run_manifest(spec, row_id)
    predecessor = R8.build(R8_CONFIG)
    if (predecessor["trusted_hammer_authority_count"] != 0 or
            predecessor["table_a_evidence_bundle_count"] != 0 or
            predecessor["headline_gate"]["admitted"] or
            predecessor["analytical_diagnostic"]["admitted"]):
        raise RegistryError("sealed r8 predecessor is not canonical zero")
    return {
        "schema": "m691.h67.paper_metric_registry.r9.preview",
        "status": config["status"],
        "validated_production_run_count": len(validated),
        "validated_production_runs": validated,
        "trusted_hammer_authority_count": 0,
        "table_a_evidence_bundle_count": 0,
        "table_a": predecessor["table_a"],
        "table_b": predecessor["table_b"],
        "table_c": predecessor["table_c"],
        "headline_gate": predecessor["headline_gate"],
        "analytical_diagnostic": predecessor["analytical_diagnostic"],
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
    except (OSError, ValueError, RuntimeError) as exc:
        print("M691_REGISTRY_FAIL: %s" % exc)
        return 2
    if args.emit_json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2,
                         allow_nan=False))
    else:
        print("M691_REGISTRY_PASS production_runs=%d authority=0 bundles=0 eligible=%d headline=false analytical=false" %
              (result["validated_production_run_count"],
               result["headline_gate"]["eligible_row_count"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
