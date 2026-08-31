#!/usr/bin/env python3
"""Build the M698 Table-A registry r10 canonical-zero overlay.

The r10 parser can validate a complete candidate evidence grammar, but this
registry intentionally has no trusted production authority pinned yet.  Thus a
self-authored or synthetic bundle cannot enter validated production runs.  A
future additive revision may pin one exact fresh-review SHA after a real native
run; changing this file in place is forbidden by the contract.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


class RegistryError(ValueError):
    pass


REPO_ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = REPO_ROOT / "hw_autoresearch_nts07"
R9_BUILDER = HW_ROOT / "system_simulator/scripts/build_m691_h67_paper_metric_registry_r9.py"
R9_BUILDER_SHA256 = "1db8bf8ccc96e8c4631116d3cb351e8d44e54b503c0e0e7e35cd86cf95cef08c"
R9_CONFIG = HW_ROOT / "system_simulator/config/m691_h67_paper_metric_registry_r9_20260828.json"
R9_CONFIG_SHA256 = "ffa3e72fa3252a0726372b856ec1a20edd1ac28b553e6f3115657e17dfbcda88"
EXTRACTOR = HW_ROOT / "system_simulator/scripts/extract_m698_native_synopsys_run_provenance_r10.py"
# Filled from the authored extractor before release; exact-SHA self-check is
# applied by _load below.
EXTRACTOR_SHA256 = "66b5b988e51f1d1e0a6be4234c08c04d041517b65f29f23531ae955f5585f85d"
DEFAULT_CONFIG = HW_ROOT / "system_simulator/config/m698_h67_paper_metric_registry_r10_20260828.json"
DEFAULT_CONFIG_SHA256 = "6d9dedb378acfc43330a09315274e4cbe372c2abf1b9749916b606261ab2e5a3"
M695_ROOT = HW_ROOT / "reviews/m695_m691_table_a_registry_r9_fresh_hammer_r1_20260828"
M695_REVIEW = M695_ROOT / "review.json"
M695_REVIEW_SHA256 = "cedadf56a5b0966aba392d1e409a357d58141098b32c48ca173cf076552c0a7a"
M695_MANIFEST = M695_ROOT / "SHA256SUMS"
M695_MANIFEST_SHA256 = "5a25b1c214f0892c5bda996c93a5cf591eec47e318995475b5e24a4dcf3ff2b9"
M695_OUTER = M695_ROOT / "SHA256SUMS.seal.sha256"
M695_OUTER_SHA256 = "b09058d71683180f0c8c07d5c7715e7501c58ed2b2bc04c1b76e748cc7dde473"
DOCS359 = HW_ROOT / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

# Fail closed by construction.  A future additive builder must bind exact
# fresh-review SHA(s); r10 itself can never trust a review merely because a
# user-authored JSON claims P0=P1=0.
PINNED_PRODUCTION_AUTHORITIES = {}


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


R9 = _load("m698_sealed_m691_builder", R9_BUILDER, R9_BUILDER_SHA256)
EX = _load("m698_r10_extractor", EXTRACTOR, EXTRACTOR_SHA256)
ROW_TO_CONFIGURATION = dict(R9.ROW_TO_CONFIGURATION)


def _exact(value, fields, label):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise RegistryError(label + " fields differ")


def _validate_m695_root():
    if (_sha256(M695_REVIEW) != M695_REVIEW_SHA256 or
            _sha256(M695_MANIFEST) != M695_MANIFEST_SHA256 or
            _sha256(M695_OUTER) != M695_OUTER_SHA256):
        raise RegistryError("sealed M695 root drift")
    review = json.loads(M695_REVIEW.read_text(encoding="utf-8"))
    if (review.get("verdict") !=
            "PASS_CANONICAL_ZERO__NO_GO_PRODUCTION_ADMISSION__R10_REPAIR_REQUIRED" or
            review.get("severity_counts") != {"P0": 0, "P1": 5, "P2": 1}):
        raise RegistryError("sealed M695 semantic root mismatch")
    if M695_OUTER.read_text(encoding="utf-8") != M695_MANIFEST_SHA256 + "  SHA256SUMS\n":
        raise RegistryError("sealed M695 outer seal mismatch")


def _spec_path(spec, label, media, prefix):
    try:
        return EX._file_spec(spec, label, (media,)) if prefix.startswith(
            "hw_autoresearch_nts07/results/") else R9.EX._file_spec(
                spec, label, (media,), prefix)
    except (EX.ExtractionError, R9.EX.ExtractionError) as exc:
        raise RegistryError(str(exc))


def _validate_candidate_structure(spec, expected_row, allow_synthetic=False):
    _exact(spec, {"run_manifest", "trust_extension", "authority"},
           "r10 production candidate")
    run_path = _spec_path(spec["run_manifest"], "r10 run manifest", "application/json",
                          "hw_autoresearch_nts07/results/")
    extension_path = _spec_path(spec["trust_extension"], "r10 trust extension",
                                "application/json", "hw_autoresearch_nts07/results/")
    try:
        extracted = EX.extract_from_bundle(
            run_path.relative_to(REPO_ROOT).as_posix(),
            extension_path.relative_to(REPO_ROOT).as_posix(),
            allow_synthetic_grammar=allow_synthetic)
    except (OSError, EX.ExtractionError) as exc:
        raise RegistryError("r10 candidate structural extraction failed: %s" % exc)
    manifest = json.loads(run_path.read_text(encoding="utf-8"))
    row_id = manifest.get("row_id")
    if (row_id != expected_row or row_id not in ROW_TO_CONFIGURATION or
            manifest.get("m527_configuration_id") != ROW_TO_CONFIGURATION[row_id] or
            manifest.get("design_name") != "h67_table_a_" + ROW_TO_CONFIGURATION[row_id]):
        raise RegistryError("r10 mandatory-row/configuration identity mismatch")
    return {
        "row_id": row_id,
        "run_manifest_sha256": _sha256(run_path),
        "trust_extension_sha256": _sha256(extension_path),
        "evidence_class": extracted["r10_evidence_class"],
        "structural_evidence_pass": True,
        "total_area_mm2": (extracted["values"]["logic_area_mm2"] +
                           extracted["values"]["sram_macro_area_mm2"]),
        "total_power_mw": extracted["values"]["total_power_mw"],
    }


def _validate_pinned_authority(spec, candidate):
    path = _spec_path(spec, "r10 fresh native-run authority", "application/json",
                      "hw_autoresearch_nts07/reviews/")
    digest = _sha256(path)
    expected = PINNED_PRODUCTION_AUTHORITIES.get(candidate["row_id"])
    if expected is None or digest != expected:
        raise RegistryError("r10 production authority is not code-pinned")
    authority = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema", "verdict", "severity_counts", "execution_authorized",
                "row_id", "run_manifest_sha256", "trust_extension_sha256"}
    _exact(authority, required, "r10 production authority")
    if (authority["schema"] != "m698.h67.native_run_fresh_authority.r1" or
            authority["verdict"] != "GO_TABLE_A_PRODUCTION_INGESTION" or
            authority["severity_counts"].get("P0") != 0 or
            authority["severity_counts"].get("P1") != 0 or
            authority["execution_authorized"] is not True or
            authority["row_id"] != candidate["row_id"] or
            authority["run_manifest_sha256"] != candidate["run_manifest_sha256"] or
            authority["trust_extension_sha256"] != candidate["trust_extension_sha256"]):
        raise RegistryError("r10 pinned authority semantic mismatch")
    return digest


def build(config_path=DEFAULT_CONFIG):
    _validate_m695_root()
    if (_sha256(R9_CONFIG) != R9_CONFIG_SHA256 or
            _sha256(EXTRACTOR) != EXTRACTOR_SHA256 or
            _sha256(DOCS359) != DOCS359_SHA256):
        raise RegistryError("r10 frozen source/config/protected identity drift")
    path = Path(config_path)
    if path == DEFAULT_CONFIG and _sha256(path) != DEFAULT_CONFIG_SHA256:
        raise RegistryError("canonical r10 config SHA drift")
    config = json.loads(path.read_text(encoding="utf-8"))
    _exact(config, {"schema", "date", "status", "purpose", "sealed_r9_registry",
                    "production_run_bundles", "claim_boundary", "protected_file"},
           "r10 registry config")
    if (config["schema"] != "m698.h67.paper_metric_registry.r10" or
            config["sealed_r9_registry"] != {
                "path": R9_CONFIG.relative_to(REPO_ROOT).as_posix(),
                "sha256": R9_CONFIG_SHA256, "media_type": "application/json"} or
            config["protected_file"] != {
                "path": DOCS359.relative_to(REPO_ROOT).as_posix(),
                "sha256": DOCS359_SHA256}):
        raise RegistryError("r10 overlay identity mismatch")
    expected_boundary = {
        "trusted_hammer_authorities": 0, "table_a_evidence_bundles": 0,
        "table_a_admitted_rows": 0, "paper_headline_admitted": False,
        "analytical_range_admitted": False, "production_measurement_admitted": False,
        "methodology_registry_only": True, "eda_or_gpu_run": False,
        "code_pinned_production_authority_count": 0,
    }
    if config["claim_boundary"] != expected_boundary:
        raise RegistryError("r10 canonical claim boundary mismatch")
    bundles = config["production_run_bundles"]
    if not isinstance(bundles, dict):
        raise RegistryError("production run bundle map must be an object")
    validated = {}
    for row_id, spec in bundles.items():
        if row_id in validated:
            raise RegistryError("duplicate r10 production row")
        _exact(spec, {"run_manifest", "trust_extension", "authority"},
               "r10 production candidate")
        expected_authority = PINNED_PRODUCTION_AUTHORITIES.get(row_id)
        if expected_authority is None:
            raise RegistryError("r10 production authority is not code-pinned")
        authority_path = _spec_path(
            spec["authority"], "r10 fresh native-run authority", "application/json",
            "hw_autoresearch_nts07/reviews/")
        if _sha256(authority_path) != expected_authority:
            raise RegistryError("r10 production authority SHA is not code-pinned")
        # A bundle cannot even be considered native without a code-pinned
        # independent authority.  Structural grammar may be tested only via
        # _validate_candidate_structure(..., allow_synthetic=True), never here.
        candidate = _validate_candidate_structure(spec, row_id, False)
        if candidate["evidence_class"] != "NATIVE_SYNOPSYS_EXECUTION":
            raise RegistryError("r10 production bundle is not native evidence")
        candidate["authority_sha256"] = _validate_pinned_authority(
            spec["authority"], candidate)
        validated[row_id] = candidate
    predecessor = R9.build(R9_CONFIG)
    if (predecessor["validated_production_run_count"] != 0 or
            predecessor["trusted_hammer_authority_count"] != 0 or
            predecessor["table_a_evidence_bundle_count"] != 0 or
            predecessor["headline_gate"]["admitted"] or
            predecessor["analytical_diagnostic"]["admitted"]):
        raise RegistryError("sealed r9 predecessor is not canonical zero")
    return {
        "schema": "m698.h67.paper_metric_registry.r10.preview",
        "status": config["status"],
        "validated_production_run_count": len(validated),
        "validated_production_runs": validated,
        "trusted_hammer_authority_count": len(validated),
        "table_a_evidence_bundle_count": 0,
        "table_a": predecessor["table_a"], "table_b": predecessor["table_b"],
        "table_c": predecessor["table_c"], "headline_gate": predecessor["headline_gate"],
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
        print("M698_REGISTRY_FAIL: %s" % exc)
        return 2
    if args.emit_json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2,
                         allow_nan=False))
    else:
        print("M698_REGISTRY_PASS production_runs=%d authority=%d bundles=0 eligible=%d headline=false analytical=false" %
              (result["validated_production_run_count"],
               result["trusted_hammer_authority_count"],
               result["headline_gate"]["eligible_row_count"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
