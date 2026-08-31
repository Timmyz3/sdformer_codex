#!/usr/bin/env python3
"""Build the M663 H67 paper-metric registry r7, fail closed.

M663 is a methodology-only successor to M658.  It retains the exact frozen
ten-operator population and replaces author-shaped PPA text with direct native
DC/PT/PTPX/memory-compiler parsing plus typed row/config/run identity.
Canonical authority remains empty.
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
R6_BUILDER = HW_ROOT / "system_simulator/scripts/build_m658_h67_paper_metric_registry_r6.py"
R6_BUILDER_SHA256 = "2880e1356c71b25cee344eb6ffb389a93cbe8cff3b12d012073ee6a7255bea5d"
EXTRACTOR = HW_ROOT / "system_simulator/scripts/extract_m663_native_synopsys_ppa_reports.py"
EXTRACTOR_SHA256 = "2a7456d8fe0c6336f094c857cb37c9d54a48425f77f5c0fd914c34436f0733a4"
DEFAULT_CONFIG = HW_ROOT / "system_simulator/config/m663_h67_paper_metric_registry_r7_20260828.json"
REGISTRY_TESTS = HW_ROOT / "system_simulator/tests/test_m663_h67_paper_metric_registry_r7.py"
REGISTRY_CONTRACT = HW_ROOT / "contracts/m663_h67_paper_metric_registry_r7_contract_r1_20260828.json"


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


R6 = _load("m663_sealed_m658_r6", R6_BUILDER, R6_BUILDER_SHA256)
EX = _load("m663_native_ppa_extractor", EXTRACTOR, EXTRACTOR_SHA256)
R5 = R6.R5
M635 = R6.M635
RegistryError = R6.RegistryError
M635_CONFIG = R6.M635_CONFIG
M635_CONFIG_SHA256 = R6.M635_CONFIG_SHA256
M527_CONTRACT = R6.M527_CONTRACT
M527_CONTRACT_SHA256 = R6.M527_CONTRACT_SHA256
CHECKPOINT = R6.CHECKPOINT
CHECKPOINT_SHA256 = R6.CHECKPOINT_SHA256
DOCS359 = R6.DOCS359
DOCS359_SHA256 = R6.DOCS359_SHA256
MANDATORY_ROW_IDS = R6.MANDATORY_ROW_IDS
ROW_TO_M527_CONFIGURATION = R6.ROW_TO_M527_CONFIGURATION
VIEW_NAMES = R6.VIEW_NAMES
BUNDLE_FIELDS = R6.BUNDLE_FIELDS
REQUEST_TARGET_FIELDS = R6.REQUEST_TARGET_FIELDS
TRUSTED_HAMMER_AUTHORITIES = {}

RAW_REPORT_FIELDS = set(EX.REPORT_FIELDS)
RUN_MANIFEST_FIELDS = set(EX.MANIFEST_FIELDS)
EXTRACTION_RECEIPT_FIELDS = {
    "schema", "status", "row_id", "configuration_manifest_sha256",
    "run_manifest", "extractor_source", "extraction_argv", "raw_reports",
    "native_identities", "tools", "libraries", "corners", "units",
    "extracted_values",
}
EXTRACTED_FIELDS = {
    "logic_area_mm2", "sram_macro_area_mm2",
    "logic_internal_power_mw", "logic_switching_power_mw",
    "logic_dynamic_power_mw", "logic_leakage_power_mw", "logic_total_power_mw",
    "sram_internal_power_mw", "sram_switching_power_mw",
    "sram_dynamic_power_mw", "sram_leakage_power_mw", "sram_total_power_mw",
    "total_internal_power_mw", "total_switching_power_mw",
    "total_dynamic_power_mw", "total_leakage_power_mw", "total_power_mw",
    "setup_wns_ns", "hold_wns_ns",
}
PPA_ROW_FIELDS = set(EXTRACTED_FIELDS) | {
    "row_id", "configuration_manifest_sha256", "total_area_mm2",
    "extraction_receipt",
}


def _exact(value, fields, label):
    return R6._exact(value, fields, label)


def _number(value, label, zero_ok=False):
    return R6._number(value, label, zero_ok)


def _file_spec(spec, label, prefix=None, media_types=("application/json",)):
    return R6._file_spec(spec, label, prefix, media_types)


def _required_operator_scope():
    return R6._required_operator_scope()


def _map_sha(value):
    return R5._map_sha(value)


def _expected_design(row_id):
    return "h67_table_a_" + ROW_TO_M527_CONFIGURATION[row_id]


def _expected_run_id(row_id, configuration_sha, report_hashes):
    clean = re.sub(r"[^a-zA-Z0-9_]+", "_", row_id)
    return "m663_%s_%s_%s" % (clean, configuration_sha[:12], _map_sha(report_hashes)[:12])


def _expected_argv(run_manifest_spec):
    return ["python3", EXTRACTOR.relative_to(REPO_ROOT).as_posix(),
            "--run-manifest", run_manifest_spec["path"], "--emit-json"]


def _validate_run_manifest(spec, row_id, configuration_sha):
    path, doc, digest = _file_spec(spec, "native Synopsys run manifest " + row_id,
                                   "hw_autoresearch_nts07/results/")
    _exact(doc, RUN_MANIFEST_FIELDS, "native Synopsys run manifest")
    if (doc["schema"] != "m663.h67.native_synopsys_run_manifest.r1" or
            doc["status"] != "FROZEN_NATIVE_REPORTS" or doc["row_id"] != row_id or
            doc["configuration_manifest_sha256"] != configuration_sha or
            doc["m527_configuration_id"] != ROW_TO_M527_CONFIGURATION[row_id] or
            doc["operator_scope_sha256"] != _map_sha(_required_operator_scope()) or
            doc["design_name"] != _expected_design(row_id)):
        raise RegistryError("native run manifest row/config/operator/design mismatch")
    if not isinstance(doc["macro_name"], str) or not doc["macro_name"]:
        raise RegistryError("native run manifest macro identity missing")
    if not isinstance(doc["raw_reports"], dict) or set(doc["raw_reports"]) != RAW_REPORT_FIELDS:
        raise RegistryError("native run manifest raw report set mismatch")
    report_hashes = {}
    expected_parent = path.parent / "reports"
    for name in sorted(RAW_REPORT_FIELDS):
        report_path, _, report_sha = _file_spec(doc["raw_reports"][name],
                                                "native report " + name,
                                                "hw_autoresearch_nts07/results/",
                                                ("text/plain",))
        if report_path.parent != expected_parent:
            raise RegistryError("native reports must be colocated in the typed run directory")
        report_hashes[name] = report_sha
    expected_run = _expected_run_id(row_id, configuration_sha, report_hashes)
    if doc["run_id"] != expected_run or path.parent.name != expected_run:
        raise RegistryError("native run identity/path mismatch")
    expected_tools = {"dc_area": "dc_shell", "ptpx_power": "pt_shell",
                      "pt_setup": "pt_shell", "pt_hold": "pt_shell",
                      "sram_macro": "memory_compiler"}
    for field in ("tools", "libraries", "corners"):
        if not isinstance(doc[field], dict) or set(doc[field]) != RAW_REPORT_FIELDS:
            raise RegistryError("native run %s map mismatch" % field)
    for name, tool in expected_tools.items():
        entry = doc["tools"][name]
        if (not isinstance(entry, dict) or set(entry) != {"tool", "version"} or
                entry["tool"] != tool or not isinstance(entry["version"], str) or not entry["version"]):
            raise RegistryError("native run tool/version identity mismatch")
        if not isinstance(doc["libraries"][name], str) or not doc["libraries"][name]:
            raise RegistryError("native run library identity missing")
        if not isinstance(doc["corners"][name], str) or not doc["corners"][name]:
            raise RegistryError("native run corner identity missing")
    return {"path": path, "doc": doc, "sha256": digest,
            "report_hashes": report_hashes}


def _validate_extraction_receipt(spec, row_id, configuration_sha):
    _, receipt, digest = _file_spec(spec, "native PPA extraction receipt " + row_id,
                                    "hw_autoresearch_nts07/results/")
    _exact(receipt, EXTRACTION_RECEIPT_FIELDS, "native PPA extraction receipt")
    if (receipt["schema"] != "m663.h67.native_synopsys_ppa_extraction_receipt.r1" or
            receipt["status"] != "PASS_DIRECT_NATIVE_REPORT_PARSE" or
            receipt["row_id"] != row_id or
            receipt["configuration_manifest_sha256"] != configuration_sha):
        raise RegistryError("native PPA receipt schema/status/row root mismatch")
    run = _validate_run_manifest(receipt["run_manifest"], row_id, configuration_sha)
    extractor_path, _, extractor_sha = _file_spec(
        receipt["extractor_source"], "native PPA extractor source",
        media_types=("text/x-python", "text/plain"))
    if extractor_path != EXTRACTOR or extractor_sha != EXTRACTOR_SHA256:
        raise RegistryError("native PPA receipt does not bind the reviewed extractor")
    if receipt["raw_reports"] != run["doc"]["raw_reports"]:
        raise RegistryError("native PPA receipt raw reports differ from run manifest")
    if receipt["extraction_argv"] != _expected_argv(receipt["run_manifest"]):
        raise RegistryError("native PPA extraction argv is not exact")
    try:
        extracted = EX.extract_from_manifest(run["path"])
    except (OSError, EX.ExtractionError) as exc:
        raise RegistryError("bound native PPA report set is invalid: %s" % exc)
    expected_run_identity = {
        "row_id": row_id, "configuration_manifest_sha256": configuration_sha,
        "m527_configuration_id": ROW_TO_M527_CONFIGURATION[row_id],
        "operator_scope_sha256": _map_sha(_required_operator_scope()),
        "design_name": _expected_design(row_id), "macro_name": run["doc"]["macro_name"],
        "run_id": run["doc"]["run_id"],
    }
    if extracted["run_identity"] != expected_run_identity:
        raise RegistryError("native extraction run identity mismatch")
    if (receipt["native_identities"] != extracted["identities"] or
            receipt["tools"] != run["doc"]["tools"] or
            receipt["libraries"] != run["doc"]["libraries"] or
            receipt["corners"] != run["doc"]["corners"] or
            extracted["libraries"] != run["doc"]["libraries"] or
            extracted["corners"] != run["doc"]["corners"]):
        raise RegistryError("native tool/report/library/corner identity mismatch")
    expected_units = {field: ("mm2" if field.endswith("area_mm2") else
                              "ns" if field.endswith("wns_ns") else "mW")
                      for field in EXTRACTED_FIELDS}
    if receipt["units"] != expected_units:
        raise RegistryError("native PPA extraction units mismatch")
    _exact(receipt["extracted_values"], EXTRACTED_FIELDS, "native PPA extracted values")
    for field, expected in extracted["values"].items():
        actual = _number(receipt["extracted_values"][field], "native extracted " + field,
                         zero_ok=True)
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise RegistryError("native PPA receipt value drift: " + field)
    values = extracted["values"]
    for prefix in ("logic", "sram", "total"):
        dynamic = values[prefix + "_dynamic_power_mw"]
        if not math.isclose(dynamic,
                            values[prefix + "_internal_power_mw"] +
                            values[prefix + "_switching_power_mw"],
                            rel_tol=0.0, abs_tol=1e-12):
            raise RegistryError(prefix + " dynamic power identity mismatch")
        total_field = (prefix + "_total_power_mw" if prefix != "total" else
                       "total_power_mw")
        if not math.isclose(values[total_field],
                            dynamic + values[prefix + "_leakage_power_mw"],
                            rel_tol=2e-6, abs_tol=1e-9):
            raise RegistryError(prefix + " total power lacks dynamic or leakage")
    return {"sha256": digest, "values": values, "run_manifest_sha256": run["sha256"],
            "report_hashes": run["report_hashes"], "extractor_sha256": extractor_sha}


def _validate_typed_receipts(bundle_id, specs, raw, configs, summaries, measurement):
    _, energy, energy_sha = _file_spec(specs["energy_receipt"], "typed energy receipt",
                                       "hw_autoresearch_nts07/results/")
    _exact(energy, {"schema", "status", "raw_run_index_sha256", "rows"},
           "typed energy receipt")
    if (energy["schema"] != "m645.h67.logic_sram_dram_energy_receipt.r1" or
            energy["status"] != "PASS_TYPED" or energy["raw_run_index_sha256"] != raw["sha256"]):
        raise RegistryError("typed energy receipt schema/status/root mismatch")
    energy_rows = {}
    for row in energy["rows"]:
        _exact(row, {"row_id", "configuration_manifest_sha256", "logic_energy_mj",
                     "sram_energy_mj", "dram_energy_mj", "total_energy_mj"}, "energy row")
        row_id = row["row_id"]
        if (row_id not in MANDATORY_ROW_IDS or row_id in energy_rows or
                row["configuration_manifest_sha256"] != configs[row_id]["sha256"]):
            raise RegistryError("typed energy row identity mismatch")
        components = []
        for component in ("logic_energy_mj", "sram_energy_mj", "dram_energy_mj"):
            value = _number(row[component], component, zero_ok=True)
            if not math.isclose(value, summaries[row_id][component], rel_tol=1e-12,
                                abs_tol=1e-15):
                raise RegistryError("typed energy component does not project rooted logs")
            components.append(value)
        if (not math.isclose(row["total_energy_mj"], math.fsum(components), rel_tol=1e-12,
                             abs_tol=1e-15) or
                not math.isclose(row["total_energy_mj"], summaries[row_id]["energy_mj"],
                                 rel_tol=1e-12, abs_tol=1e-15)):
            raise RegistryError("typed energy total mismatch")
        energy_rows[row_id] = row
    if set(energy_rows) != set(MANDATORY_ROW_IDS):
        raise RegistryError("typed energy receipt lacks six rows")

    _, accuracy, accuracy_sha = _file_spec(specs["accuracy_receipt"], "typed accuracy receipt",
                                           "hw_autoresearch_nts07/results/")
    _exact(accuracy, {"schema", "status", "raw_run_index_sha256", "checkpoint_sha256",
                      "population_manifest_sha256", "rows"}, "typed accuracy receipt")
    if (accuracy["schema"] != "m645.h67.accuracy_receipt.r1" or
            accuracy["status"] != "PASS_TYPED" or
            accuracy["raw_run_index_sha256"] != raw["sha256"] or
            accuracy["checkpoint_sha256"] != CHECKPOINT_SHA256 or
            accuracy["population_manifest_sha256"] !=
            measurement["manifests"]["sequence_population_manifest"]["sha256"]):
        raise RegistryError("typed accuracy receipt schema/status/root mismatch")
    accuracy_rows = {}
    for row in accuracy["rows"]:
        _exact(row, {"row_id", "configuration_manifest_sha256", "aee", "dsec_fl_percent"},
               "accuracy row")
        row_id = row["row_id"]
        if (row_id not in MANDATORY_ROW_IDS or row_id in accuracy_rows or
                row["configuration_manifest_sha256"] != configs[row_id]["sha256"] or
                not math.isclose(row["aee"], summaries[row_id]["accuracy"], rel_tol=1e-12,
                                 abs_tol=1e-12) or
                not math.isclose(row["dsec_fl_percent"], summaries[row_id]["dsec_fl_percent"],
                                 rel_tol=1e-12, abs_tol=1e-12)):
            raise RegistryError("typed accuracy row mismatch")
        accuracy_rows[row_id] = row
    if set(accuracy_rows) != set(MANDATORY_ROW_IDS):
        raise RegistryError("typed accuracy receipt lacks six rows")

    _, ppa, ppa_sha = _file_spec(specs["ppa_receipt"], "typed native PPA receipt",
                                 "hw_autoresearch_nts07/results/")
    _exact(ppa, {"schema", "status", "technology_nm", "clock_period_ns", "rows"},
           "typed native PPA receipt")
    if (ppa["schema"] != "m663.h67.native_synopsys_rooted_ppa_receipt.r1" or
            ppa["status"] != "PASS_DIRECT_NATIVE_REPORT_PARSE" or
            ppa["technology_nm"] != 28 or ppa["clock_period_ns"] != 3.0):
        raise RegistryError("typed native PPA receipt schema/status/process mismatch")
    ppa_rows = {}
    for row in ppa["rows"]:
        _exact(row, PPA_ROW_FIELDS, "typed native PPA row")
        row_id = row["row_id"]
        if (row_id not in MANDATORY_ROW_IDS or row_id in ppa_rows or
                row["configuration_manifest_sha256"] != configs[row_id]["sha256"]):
            raise RegistryError("typed native PPA row identity mismatch")
        extracted = _validate_extraction_receipt(row["extraction_receipt"], row_id,
                                                  configs[row_id]["sha256"])
        for field, expected in extracted["values"].items():
            actual = _number(row[field], "typed native PPA " + field, zero_ok=True)
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
                raise RegistryError("typed native PPA scalar drift: " + field)
        if not math.isclose(row["total_area_mm2"],
                            row["logic_area_mm2"] + row["sram_macro_area_mm2"],
                            rel_tol=0.0, abs_tol=1e-12):
            raise RegistryError("typed native PPA total area mismatch")
        ppa_rows[row_id] = row
    if set(ppa_rows) != set(MANDATORY_ROW_IDS):
        raise RegistryError("typed native PPA receipt lacks six rows")
    return {"energy_sha256": energy_sha, "accuracy_sha256": accuracy_sha,
            "ppa_sha256": ppa_sha, "energy": energy_rows, "accuracy": accuracy_rows,
            "ppa": ppa_rows}


def _collect_ppa_evidence(ppa_spec):
    _, ppa, _ = _file_spec(ppa_spec, "native PPA evidence collection",
                           "hw_autoresearch_nts07/results/")
    evidence = {"ppa_native_extractor_source": EXTRACTOR_SHA256}
    if not isinstance(ppa.get("rows"), list) or len(ppa["rows"]) != 6:
        raise RegistryError("native PPA evidence must contain six rows")
    seen = set()
    for row in ppa["rows"]:
        row_id = row.get("row_id")
        if row_id not in MANDATORY_ROW_IDS or row_id in seen:
            raise RegistryError("native PPA evidence row identity mismatch")
        seen.add(row_id)
        _, receipt, receipt_sha = _file_spec(row.get("extraction_receipt"),
                                             "native PPA evidence extraction receipt",
                                             "hw_autoresearch_nts07/results/")
        evidence["ppa_extraction_receipt:" + row_id] = receipt_sha
        _, manifest, manifest_sha = _file_spec(receipt.get("run_manifest"),
                                               "native PPA evidence run manifest",
                                               "hw_autoresearch_nts07/results/")
        evidence["ppa_run_manifest:" + row_id] = manifest_sha
        reports = manifest.get("raw_reports")
        if not isinstance(reports, dict) or set(reports) != RAW_REPORT_FIELDS:
            raise RegistryError("native PPA evidence report set mismatch")
        for name in sorted(reports):
            _, _, report_sha = _file_spec(reports[name], "native PPA evidence report",
                                           "hw_autoresearch_nts07/results/", ("text/plain",))
            evidence["ppa_native_report:%s:%s" % (row_id, name)] = report_sha
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
        raise RegistryError("independent review does not bind exact fixed numerator")
    _file_spec(targets["direct_result"], "review direct result",
               "hw_autoresearch_nts07/results/")
    _file_spec(targets["fixed_throughput_numerator_receipt"], "review fixed numerator",
               "hw_autoresearch_nts07/results/")
    return {name: targets[name]["sha256"] for name in sorted(targets)}


def _validate_bundle(bundle_id, bundle):
    _exact(bundle, BUNDLE_FIELDS, "M663 Table-A direct bundle")
    if bundle["schema"] != "m663.h67.rooted_direct_bundle.r4" or bundle["bundle_id"] != bundle_id:
        raise RegistryError("M663 Table-A bundle schema/id mismatch")
    legacy = copy.deepcopy(bundle)
    legacy["schema"] = "m658.h67.rooted_direct_bundle.r3"
    previous = R6.TRUSTED_HAMMER_AUTHORITIES
    R6.TRUSTED_HAMMER_AUTHORITIES = TRUSTED_HAMMER_AUTHORITIES
    try:
        return R6._validate_bundle(bundle_id, legacy)
    finally:
        R6.TRUSTED_HAMMER_AUTHORITIES = previous


def _validate_overlay(config):
    fields = {"schema", "date", "status", "purpose", "base_registry",
              "table_a_evidence_bundles", "table_a_rows", "claim_boundary", "protected_file"}
    _exact(config, fields, "M663 registry overlay")
    if config["schema"] != "m663.h67.paper_metric_registry.r7":
        raise RegistryError("unexpected M663 registry schema")
    _, base, base_sha = _file_spec(config["base_registry"], "sealed M635 base registry",
                                   "hw_autoresearch_nts07/system_simulator/config/")
    if (base_sha != M635_CONFIG_SHA256 or
            config["base_registry"]["path"] != M635_CONFIG.relative_to(REPO_ROOT).as_posix() or
            base.get("schema") != "m635.h67.paper_metric_registry.r3" or
            base.get("table_a_evidence_bundles") != {}):
        raise RegistryError("M663 must inherit exact canonical M635 zero-bundle registry")
    protected = config["protected_file"]
    _exact(protected, {"path", "sha256"}, "protected file")
    if protected != {"path": DOCS359.relative_to(REPO_ROOT).as_posix(),
                      "sha256": DOCS359_SHA256}:
        raise RegistryError("protected docs359 binding mismatch")
    if _sha256(R5._secure_file(protected["path"])) != DOCS359_SHA256:
        raise RegistryError("protected docs359 SHA drift")
    return base


def build(config_path=DEFAULT_CONFIG):
    _required_operator_scope()
    if _sha256(EXTRACTOR) != EXTRACTOR_SHA256:
        raise RegistryError("native PPA extractor SHA drift")
    config = M635.load_json(Path(config_path), "M663 registry")
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
    gate = R6._evaluate(rows, bundles, config["claim_boundary"], policy)
    return {"schema": "m663.h67.paper_metric_registry.r7.preview", "status": config["status"],
            "source_hashes_validated": sources,
            "trusted_hammer_authority_count": len(TRUSTED_HAMMER_AUTHORITIES),
            "table_a_evidence_bundle_count": len(bundles), "table_a": rows,
            "table_b": base["table_b_schema"]["rows"], "table_c": base["table_c_schema"]["rows"],
            "analytical_diagnostic": analytical, "headline_gate": gate,
            "claim_boundary": config["claim_boundary"],
            "protected_file_validated": config["protected_file"]}


# Patch only the sealed predecessor's future-bundle call graph.
R6._collect_ppa_evidence = _collect_ppa_evidence
R6.R5._validate_reviewed_targets = _validate_reviewed_targets
R6.R5.R4._validate_typed_receipts = _validate_typed_receipts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--emit-json", action="store_true")
    args = parser.parse_args()
    try:
        result = build(args.config)
    except (RegistryError, RuntimeError) as exc:
        print("M663_REGISTRY_FAIL: %s" % exc)
        return 2
    if args.emit_json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    else:
        print("M663_REGISTRY_PASS sources=%d trusted_authorities=%d bundles=%d table_a_eligible=%d headline_admitted=%s analytical_admitted=%s" %
              (len(result["source_hashes_validated"]), result["trusted_hammer_authority_count"],
               result["table_a_evidence_bundle_count"], result["headline_gate"]["eligible_row_count"],
               str(result["headline_gate"]["admitted"]).lower(),
               str(result["analytical_diagnostic"]["admitted"]).lower()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
