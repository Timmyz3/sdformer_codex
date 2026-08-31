#!/usr/bin/env python3
"""Build the M658 H67 paper-metric registry r6, fail closed.

M658 is a methodology-only successor to M653 r5.  It executes the frozen
M527 ten-class operator scope and requires PPA extraction receipts rooted in
DC, PTPX, PrimeTime and SRAM-compiler report text plus exact extractor/tool/
command/library/corner/unit identities.  Canonical authority remains empty.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = REPO_ROOT / "hw_autoresearch_nts07"
R5_BUILDER = HW_ROOT / "system_simulator/scripts/build_m653_h67_paper_metric_registry_r5.py"
R5_BUILDER_SHA256 = "97ce23afec30f91acfc612c06d4d5344680922842a04fd0c747675899156b9fd"
EXTRACTOR = HW_ROOT / "system_simulator/scripts/extract_m658_synopsys_ppa_reports.py"
EXTRACTOR_SHA256 = "a29431d7d2478a7089961069663ac6ecfdb7d99db8701152f16cd0e7511a297d"
DEFAULT_CONFIG = HW_ROOT / "system_simulator/config/m658_h67_paper_metric_registry_r6_20260828.json"
REGISTRY_TESTS = HW_ROOT / "system_simulator/tests/test_m658_h67_paper_metric_registry_r6.py"
REGISTRY_CONTRACT = HW_ROOT / "contracts/m658_h67_paper_metric_registry_r6_contract_r1_20260828.json"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(name, path, expected_sha):
    if _sha256(path) != expected_sha:
        raise RuntimeError("sealed dependency SHA drift: " + str(path))
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import sealed dependency: " + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R5 = _load("m658_sealed_m653_r5", R5_BUILDER, R5_BUILDER_SHA256)
EX = _load("m658_exact_ppa_extractor", EXTRACTOR, EXTRACTOR_SHA256)
R4 = R5.R4
M635 = R5.M635
RegistryError = R5.RegistryError
M635_CONFIG = R5.M635_CONFIG
M635_CONFIG_SHA256 = R5.M635_CONFIG_SHA256
M527_CONTRACT = R5.M527_CONTRACT
M527_CONTRACT_SHA256 = R5.M527_CONTRACT_SHA256
CHECKPOINT = R5.CHECKPOINT
CHECKPOINT_SHA256 = R5.CHECKPOINT_SHA256
DOCS359 = R5.DOCS359
DOCS359_SHA256 = R5.DOCS359_SHA256
MANDATORY_ROW_IDS = R5.MANDATORY_ROW_IDS
ROW_TO_M527_CONFIGURATION = R5.ROW_TO_M527_CONFIGURATION
VIEW_NAMES = R5.VIEW_NAMES
TRUSTED_HAMMER_AUTHORITIES = {}
BUNDLE_FIELDS = R5.BUNDLE_FIELDS
REQUEST_TARGET_FIELDS = R5.REQUEST_TARGET_FIELDS

RAW_REPORT_FIELDS = {"dc_area", "ptpx_power", "pt_sta", "sram_macro"}
EXTRACTION_RECEIPT_FIELDS = {
    "schema", "status", "row_id", "configuration_manifest_sha256",
    "extractor_source", "extraction_argv", "raw_reports", "synopsys_tools",
    "libraries", "corners", "library_identity_sha256", "units", "extracted_values",
}
EXTRACTED_FIELDS = {
    "logic_area_mm2", "logic_power_mw", "sram_macro_area_mm2",
    "sram_macro_power_mw", "setup_wns_ns", "hold_wns_ns",
}


def _exact(value, fields, label):
    return R5._exact(value, fields, label)


def _number(value, label, zero_ok=False):
    return R5._number(value, label, zero_ok)


def _file_spec(spec, label, prefix=None, media_types=("application/json",)):
    return R5._file_spec(spec, label, prefix, media_types)


def _required_operator_scope():
    contract = R5._runtime_m527_contract()
    scope = contract.get("identity", {}).get("required_operator_scope")
    if (not isinstance(scope, list) or len(scope) != 10 or len(scope) != len(set(scope)) or
            any(not isinstance(item, str) or not item for item in scope)):
        raise RegistryError("frozen M527 required_operator_scope is not the exact ten-class population")
    return list(scope)


_R5_VALIDATE_MEASUREMENT = R5._validate_measurement_identity
_R5_VALIDATE_NUMERATOR = R5._validate_numerator_receipt


def _validate_measurement_identity(spec, checkpoint_sha):
    measurement = _R5_VALIDATE_MEASUREMENT(spec, checkpoint_sha)
    required = _required_operator_scope()
    if measurement["doc"]["operator_ids"] != required:
        raise RegistryError("measurement operator_ids do not exactly project frozen M527 required_operator_scope")
    trace_scope = measurement["manifests"]["complete_trace_manifest"]["doc"]["operator_scope"]
    if trace_scope != required:
        raise RegistryError("complete trace operator_scope does not exactly project frozen M527 required_operator_scope")
    measurement["required_operator_scope"] = required
    return measurement


def _validate_numerator_receipt(spec, measurement):
    receipt = _R5_VALIDATE_NUMERATOR(spec, measurement)
    required = _required_operator_scope()
    doc = receipt["doc"]
    partition = list(doc["included_operator_scope"]) + list(doc["excluded_operator_scope_with_reason"])
    if (set(partition) != set(required) or len(partition) != len(required) or
            measurement["doc"]["operator_ids"] != required):
        raise RegistryError("fixed numerator scope does not exactly partition frozen M527 required_operator_scope")
    receipt["required_operator_scope"] = required
    return receipt


def _expected_argv(raw_reports):
    return [
        "python3", EXTRACTOR.relative_to(REPO_ROOT).as_posix(),
        "--dc-area-report", raw_reports["dc_area"]["path"],
        "--ptpx-power-report", raw_reports["ptpx_power"]["path"],
        "--pt-sta-report", raw_reports["pt_sta"]["path"],
        "--sram-macro-report", raw_reports["sram_macro"]["path"],
        "--emit-json",
    ]


def _identity_projection(extracted):
    identities = extracted["identities"]
    return (
        {key: {"tool": identities[key]["tool"], "version": identities[key]["version"]}
         for key in sorted(identities)},
        {key: identities[key]["library"] for key in sorted(identities)},
        {key: identities[key]["corner"] for key in sorted(identities)},
    )


def _validate_extraction_receipt(spec, row_id, configuration_sha):
    _, receipt, digest = _file_spec(spec, "PPA extraction receipt " + row_id,
                                    "hw_autoresearch_nts07/results/")
    _exact(receipt, EXTRACTION_RECEIPT_FIELDS, "PPA extraction receipt")
    if (receipt["schema"] != "m658.h67.synopsys_ppa_extraction_receipt.r1" or
            receipt["status"] != "PASS_EXTRACTED_FROM_BOUND_REPORTS" or
            receipt["row_id"] != row_id or
            receipt["configuration_manifest_sha256"] != configuration_sha):
        raise RegistryError("PPA extraction receipt schema/status/row root mismatch")
    extractor_path, _, extractor_sha = _file_spec(
        receipt["extractor_source"], "PPA extractor source", media_types=("text/x-python", "text/plain"))
    if extractor_path != EXTRACTOR or extractor_sha != EXTRACTOR_SHA256:
        raise RegistryError("PPA receipt does not bind the exact reviewed extractor source")
    if not isinstance(receipt["raw_reports"], dict) or set(receipt["raw_reports"]) != RAW_REPORT_FIELDS:
        raise RegistryError("PPA extraction receipt raw report set mismatch")
    report_paths = {}
    report_hashes = {}
    for name in sorted(RAW_REPORT_FIELDS):
        path, _, report_sha = _file_spec(receipt["raw_reports"][name], "bound raw report " + name,
                                         "hw_autoresearch_nts07/results/", ("text/plain",))
        report_paths[name] = path
        report_hashes[name] = report_sha
    if receipt["extraction_argv"] != _expected_argv(receipt["raw_reports"]):
        raise RegistryError("PPA extraction argv is not exact")
    try:
        extracted = EX.extract(report_paths["dc_area"], report_paths["ptpx_power"],
                               report_paths["pt_sta"], report_paths["sram_macro"])
    except (OSError, EX.ExtractionError) as exc:
        raise RegistryError("bound PPA report is not an accepted Synopsys report: %s" % exc)
    tools, libraries, corners = _identity_projection(extracted)
    if receipt["synopsys_tools"] != tools or receipt["libraries"] != libraries or receipt["corners"] != corners:
        raise RegistryError("PPA tool/version/library/corner identity does not project bound reports")
    if receipt["library_identity_sha256"] != R5._map_sha(libraries):
        raise RegistryError("PPA library identity digest mismatch")
    expected_units = {"logic_area_mm2": "mm2", "logic_power_mw": "mW",
                      "sram_macro_area_mm2": "mm2", "sram_macro_power_mw": "mW",
                      "setup_wns_ns": "ns", "hold_wns_ns": "ns"}
    if receipt["units"] != expected_units:
        raise RegistryError("PPA extraction units mismatch")
    _exact(receipt["extracted_values"], EXTRACTED_FIELDS, "PPA extracted values")
    for field, expected in extracted["values"].items():
        actual = _number(receipt["extracted_values"][field], "extracted " + field, zero_ok=True)
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise RegistryError("PPA extraction receipt value does not project bound raw report: " + field)
    return {"sha256": digest, "values": extracted["values"],
            "report_hashes": report_hashes, "extractor_sha256": extractor_sha}


def _validate_typed_receipts(bundle_id, specs, raw, configs, summaries, measurement):
    # Energy and accuracy retain M653's raw-run recomputation.
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
            raise RegistryError("typed energy total mismatch")
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
            raise RegistryError("typed accuracy row mismatch")
        accuracy_rows[row_id] = row
    if set(accuracy_rows) != set(MANDATORY_ROW_IDS):
        raise RegistryError("typed accuracy receipt lacks six rows")

    _, ppa, ppa_sha = _file_spec(specs["ppa_receipt"], "typed PPA receipt",
                                 "hw_autoresearch_nts07/results/")
    _exact(ppa, {"schema", "status", "technology_nm", "clock_period_ns", "rows"}, "typed PPA receipt")
    if ppa["schema"] != "m658.h67.synopsys_rooted_ppa_receipt.r1" or ppa["status"] != "PASS_RAW_REPORT_EXTRACTED" or ppa["technology_nm"] != 28 or ppa["clock_period_ns"] != 3.0:
        raise RegistryError("typed PPA receipt schema/status/process mismatch")
    ppa_rows = {}
    for row in ppa["rows"]:
        fields = {"row_id", "configuration_manifest_sha256", "logic_area_mm2", "logic_power_mw",
                  "sram_macro_area_mm2", "sram_macro_power_mw", "total_area_mm2", "total_power_mw",
                  "setup_wns_ns", "hold_wns_ns", "extraction_receipt"}
        _exact(row, fields, "PPA row")
        row_id = row["row_id"]
        if row_id not in MANDATORY_ROW_IDS or row_id in ppa_rows or row["configuration_manifest_sha256"] != configs[row_id]["sha256"]:
            raise RegistryError("typed PPA row identity mismatch")
        extracted = _validate_extraction_receipt(row["extraction_receipt"], row_id, configs[row_id]["sha256"])
        for field, expected in extracted["values"].items():
            actual = _number(row[field], "typed PPA " + field, zero_ok=True)
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
                raise RegistryError("typed PPA scalar does not project rooted Synopsys extraction: " + field)
        area = row["logic_area_mm2"] + row["sram_macro_area_mm2"]
        power = row["logic_power_mw"] + row["sram_macro_power_mw"]
        if (not math.isclose(row["total_area_mm2"], area, rel_tol=0.0, abs_tol=1e-12) or
                not math.isclose(row["total_power_mw"], power, rel_tol=0.0, abs_tol=1e-12)):
            raise RegistryError("typed PPA total does not project rooted area/power")
        ppa_rows[row_id] = row
    if set(ppa_rows) != set(MANDATORY_ROW_IDS):
        raise RegistryError("typed PPA receipt lacks six rows")
    return {"energy_sha256": energy_sha, "accuracy_sha256": accuracy_sha,
            "ppa_sha256": ppa_sha, "energy": energy_rows, "accuracy": accuracy_rows,
            "ppa": ppa_rows}


def _collect_ppa_evidence(ppa_spec):
    _, ppa, _ = _file_spec(ppa_spec, "PPA evidence collection", "hw_autoresearch_nts07/results/")
    evidence = {"ppa_extractor_source": EXTRACTOR_SHA256}
    if not isinstance(ppa.get("rows"), list):
        raise RegistryError("PPA evidence rows missing")
    for row in ppa["rows"]:
        row_id = row.get("row_id")
        extraction_spec = row.get("extraction_receipt")
        if row_id not in MANDATORY_ROW_IDS or not isinstance(extraction_spec, dict):
            raise RegistryError("PPA evidence row/extraction receipt missing")
        _, receipt, receipt_sha = _file_spec(extraction_spec, "PPA evidence extraction receipt",
                                             "hw_autoresearch_nts07/results/")
        evidence["ppa_extraction_receipt:" + row_id] = receipt_sha
        reports = receipt.get("raw_reports")
        if not isinstance(reports, dict) or set(reports) != RAW_REPORT_FIELDS:
            raise RegistryError("PPA evidence raw report set missing")
        for name in sorted(reports):
            _, _, report_sha = _file_spec(reports[name], "PPA evidence report",
                                           "hw_autoresearch_nts07/results/", ("text/plain",))
            evidence["ppa_raw_report:%s:%s" % (row_id, name)] = report_sha
    return evidence


def _validate_reviewed_targets(targets, bundle_id, evidence, numerator, result_spec):
    _exact(targets, REQUEST_TARGET_FIELDS, "independent review targets")
    fixed = {"registry_builder": Path(__file__), "registry_config": DEFAULT_CONFIG,
             "registry_tests": REGISTRY_TESTS, "registry_contract": REGISTRY_CONTRACT,
             "m527_contract": M527_CONTRACT, "checkpoint": CHECKPOINT}
    for name, path in fixed.items():
        media = (("application/octet-stream",) if name == "checkpoint" else
                 (("text/x-python", "text/plain") if name in ("registry_builder", "registry_tests")
                  else ("application/json",)))
        resolved, _, _ = _file_spec(targets[name], "review target " + name, media_types=media)
        if resolved != path:
            raise RegistryError("independent review target path mismatch: " + name)
    if targets["direct_result"] != result_spec:
        raise RegistryError("independent review does not bind exact direct result spec")
    context = R5._ACTIVE_REVIEW_CONTEXT
    if targets["fixed_throughput_numerator_receipt"] != context["numerator_spec"]:
        raise RegistryError("independent review does not bind exact fixed-numerator receipt")
    _file_spec(targets["direct_result"], "review target direct result", "hw_autoresearch_nts07/results/")
    _file_spec(targets["fixed_throughput_numerator_receipt"], "review target fixed numerator",
               "hw_autoresearch_nts07/results/")
    return {name: targets[name]["sha256"] for name in sorted(targets)}


_R5_VALIDATE_HAMMER = R5._validate_hammer
_ACTIVE_PPA_EVIDENCE = None


def _validate_hammer(spec, authority_id, evidence):
    if _ACTIVE_PPA_EVIDENCE is None:
        raise RegistryError("PPA provenance evidence context missing")
    extended = dict(evidence)
    extended.update(_ACTIVE_PPA_EVIDENCE)
    return _R5_VALIDATE_HAMMER(spec, authority_id, extended)


_R5_VALIDATE_BUNDLE = R5._validate_bundle


def _validate_bundle(bundle_id, bundle):
    global _ACTIVE_PPA_EVIDENCE
    _exact(bundle, BUNDLE_FIELDS, "M658 Table-A direct bundle")
    if bundle["schema"] != "m658.h67.rooted_direct_bundle.r3" or bundle["bundle_id"] != bundle_id:
        raise RegistryError("M658 Table-A bundle schema/id mismatch")
    ppa_evidence = _collect_ppa_evidence(bundle["ppa_receipt"])
    _, measurement_doc, _ = _file_spec(bundle["measurement_identity"], "M527 scope evidence identity",
                                        "hw_autoresearch_nts07/system_simulator/")
    ppa_evidence["m527_required_operator_scope"] = R5._map_sha(_required_operator_scope())
    ppa_evidence["m527_complete_trace_manifest"] = measurement_doc["complete_trace_manifest"]["sha256"]
    legacy = copy.deepcopy(bundle)
    legacy["schema"] = "m653.h67.rooted_direct_bundle.r2"
    previous = _ACTIVE_PPA_EVIDENCE
    previous_authorities = R5.TRUSTED_HAMMER_AUTHORITIES
    _ACTIVE_PPA_EVIDENCE = ppa_evidence
    R5.TRUSTED_HAMMER_AUTHORITIES = TRUSTED_HAMMER_AUTHORITIES
    try:
        result = _R5_VALIDATE_BUNDLE(bundle_id, legacy)
    finally:
        _ACTIVE_PPA_EVIDENCE = previous
        R5.TRUSTED_HAMMER_AUTHORITIES = previous_authorities
    result["m527_required_operator_scope_gate"] = True
    result["synopsys_ppa_provenance_gate"] = True
    result["ppa_provenance_evidence"] = ppa_evidence
    return result


def _validate_overlay(config):
    fields = {"schema", "date", "status", "purpose", "base_registry",
              "table_a_evidence_bundles", "table_a_rows", "claim_boundary", "protected_file"}
    _exact(config, fields, "M658 registry overlay")
    if config["schema"] != "m658.h67.paper_metric_registry.r6":
        raise RegistryError("unexpected M658 registry schema")
    _, base, base_sha = _file_spec(config["base_registry"], "sealed M635 base registry",
                                   "hw_autoresearch_nts07/system_simulator/config/")
    if base_sha != M635_CONFIG_SHA256 or config["base_registry"]["path"] != M635_CONFIG.relative_to(REPO_ROOT).as_posix():
        raise RegistryError("M658 must inherit exact sealed M635 canonical registry")
    if base.get("schema") != "m635.h67.paper_metric_registry.r3" or base.get("table_a_evidence_bundles") != {}:
        raise RegistryError("sealed M635 base is not canonical zero-bundle registry")
    protected = config["protected_file"]
    _exact(protected, {"path", "sha256"}, "protected file")
    if protected != {"path": DOCS359.relative_to(REPO_ROOT).as_posix(), "sha256": DOCS359_SHA256}:
        raise RegistryError("protected docs359 binding mismatch")
    if _sha256(R5._secure_file(protected["path"])) != DOCS359_SHA256:
        raise RegistryError("protected docs359 SHA drift")
    return base


def _evaluate(rows, bundles, claim, policy):
    gate = R5._evaluate(rows, bundles, claim, policy)
    source_ids = {row["source_id"] for row in rows
                  if row["row_id"] in MANDATORY_ROW_IDS and row["source_id"] is not None}
    scope_gate = False
    ppa_gate = False
    if len(source_ids) == 1 and next(iter(source_ids)) in bundles:
        bundle = bundles[next(iter(source_ids))]
        scope_gate = bundle["m527_required_operator_scope_gate"]
        ppa_gate = bundle["synopsys_ppa_provenance_gate"]
    if gate["admitted"] and not (scope_gate and ppa_gate):
        raise RegistryError("headline reached without M527 scope and Synopsys PPA provenance gates")
    gate["m527_required_operator_scope_gate"] = scope_gate
    gate["synopsys_ppa_provenance_gate"] = ppa_gate
    return gate


def build(config_path=DEFAULT_CONFIG):
    _required_operator_scope()
    if _sha256(EXTRACTOR) != EXTRACTOR_SHA256:
        raise RegistryError("exact PPA extractor SHA drift")
    config = M635.load_json(Path(config_path), "M658 registry")
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
    return {"schema": "m658.h67.paper_metric_registry.r6.preview", "status": config["status"],
            "source_hashes_validated": sources,
            "trusted_hammer_authority_count": len(TRUSTED_HAMMER_AUTHORITIES),
            "table_a_evidence_bundle_count": len(bundles), "table_a": rows,
            "table_b": base["table_b_schema"]["rows"], "table_c": base["table_c_schema"]["rows"],
            "analytical_diagnostic": analytical, "headline_gate": gate,
            "claim_boundary": config["claim_boundary"], "protected_file_validated": config["protected_file"]}


# Patch only the imported successor's inner future-bundle walk.
R5._validate_measurement_identity = _validate_measurement_identity
R5._validate_numerator_receipt = _validate_numerator_receipt
R5._validate_reviewed_targets = _validate_reviewed_targets
R5.R4._validate_measurement_identity = _validate_measurement_identity
R5.R4._validate_typed_receipts = _validate_typed_receipts
R5.R4._validate_hammer = _validate_hammer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--emit-json", action="store_true")
    args = parser.parse_args()
    try:
        result = build(args.config)
    except (RegistryError, RuntimeError) as exc:
        print("M658_REGISTRY_FAIL: %s" % exc)
        return 2
    if args.emit_json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    else:
        print("M658_REGISTRY_PASS sources=%d trusted_authorities=%d bundles=%d table_a_eligible=%d headline_admitted=%s analytical_admitted=%s" %
              (len(result["source_hashes_validated"]), result["trusted_hammer_authority_count"],
               result["table_a_evidence_bundle_count"], result["headline_gate"]["eligible_row_count"],
               str(result["headline_gate"]["admitted"]).lower(),
               str(result["analytical_diagnostic"]["admitted"]).lower()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
