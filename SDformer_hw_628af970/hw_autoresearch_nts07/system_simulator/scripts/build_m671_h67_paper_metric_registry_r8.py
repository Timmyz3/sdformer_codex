#!/usr/bin/env python3
"""Build the M671 H67 paper-metric registry r8, fail closed.

M671 is a methodology-only repair of the three M668 P1 findings and two P2
path findings.  Canonical production authority remains empty.
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
R7_BUILDER = HW_ROOT / "system_simulator/scripts/build_m663_h67_paper_metric_registry_r7.py"
R7_BUILDER_SHA256 = "19f436f05937845805ddd08ce4989e33cef7f59b7be772a7214a9f4b9b357279"
EXTRACTOR = HW_ROOT / "system_simulator/scripts/extract_m671_native_synopsys_run_provenance.py"
EXTRACTOR_SHA256 = "d635f5b5a63fece0c7a46fdf2f462ee566f1b54a4f0f9acf5e3623ecf063c4fb"
M668_REVIEW = HW_ROOT / "reviews/m668_m663_registry_r7_fresh_independent_hammer_r1_20260828/review.json"
M668_REVIEW_SHA256 = "89e5144c81178e24712efaa86dde105e6bc0d45444905948be73ec0266c38562"
M668_MANIFEST = M668_REVIEW.parent / "SHA256SUMS"
M668_MANIFEST_SHA256 = "697bc1b8cd664bd34a6f1a0758b76cd3e841cb757b25f54a3e313aad307441de"
M668_OUTER = M668_REVIEW.parent / "SHA256SUMS.seal.sha256"
M668_OUTER_SHA256 = "3e4d653e2ea792700b324f57a175d57c623c4eefd33b169057c17ca3f3f648db"
DEFAULT_CONFIG = HW_ROOT / "system_simulator/config/m671_h67_paper_metric_registry_r8_20260828.json"
REGISTRY_TESTS = HW_ROOT / "system_simulator/tests/test_m671_h67_paper_metric_registry_r8.py"
REGISTRY_CONTRACT = HW_ROOT / "contracts/m671_h67_paper_metric_registry_r8_contract_r1_20260828.json"


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


R7 = _load("m671_sealed_m663_r7", R7_BUILDER, R7_BUILDER_SHA256)
EX = _load("m671_native_run_extractor", EXTRACTOR, EXTRACTOR_SHA256)
R6 = R7.R6
R5 = R7.R5
M635 = R7.M635
RegistryError = R7.RegistryError
M635_CONFIG = R7.M635_CONFIG
M635_CONFIG_SHA256 = R7.M635_CONFIG_SHA256
M527_CONTRACT = R7.M527_CONTRACT
M527_CONTRACT_SHA256 = R7.M527_CONTRACT_SHA256
CHECKPOINT = R7.CHECKPOINT
CHECKPOINT_SHA256 = R7.CHECKPOINT_SHA256
DOCS359 = R7.DOCS359
DOCS359_SHA256 = R7.DOCS359_SHA256
MANDATORY_ROW_IDS = R7.MANDATORY_ROW_IDS
ROW_TO_M527_CONFIGURATION = R7.ROW_TO_M527_CONFIGURATION
VIEW_NAMES = R7.VIEW_NAMES
BUNDLE_FIELDS = R7.BUNDLE_FIELDS
REQUEST_TARGET_FIELDS = R7.REQUEST_TARGET_FIELDS
TRUSTED_HAMMER_AUTHORITIES = {}

RAW_REPORT_FIELDS = set(EX.REPORT_FIELDS)
RUN_MANIFEST_FIELDS = set(EX.MANIFEST_FIELDS)
EXTRACTION_RECEIPT_FIELDS = {
    "schema", "status", "row_id", "configuration_manifest_sha256",
    "run_manifest", "tool_run_receipt", "extractor_source", "extraction_argv",
    "raw_reports", "native_identities", "target_corners", "library_dbs",
    "memory_inventory", "provenance_component_root_sha256", "units",
    "extracted_values",
}
EXTRACTED_FIELDS = set(R7.EXTRACTED_FIELDS)
PPA_ROW_FIELDS = set(R7.PPA_ROW_FIELDS)


def _exact(value, fields, label):
    return R7._exact(value, fields, label)


def _number(value, label, zero_ok=False):
    return R7._number(value, label, zero_ok)


def _file_spec(spec, label, prefix=None, media_types=("application/json",)):
    return R7._file_spec(spec, label, prefix, media_types)


def _required_operator_scope():
    return R7._required_operator_scope()


def _map_sha(value):
    return R7._map_sha(value)


def _expected_design(row_id):
    return "h67_table_a_" + ROW_TO_M527_CONFIGURATION[row_id]


def _expected_run_id(row_id, configuration_sha, report_hashes):
    clean = re.sub(r"[^a-zA-Z0-9_]+", "_", row_id)
    return "m671_%s_%s_%s" % (clean, configuration_sha[:12],
                               _map_sha(report_hashes)[:12])


def _expected_argv(run_manifest_spec):
    return ["python3", EXTRACTOR.relative_to(REPO_ROOT).as_posix(),
            "--run-manifest", run_manifest_spec["path"], "--emit-json"]


def _validate_m668_root():
    if (_sha256(M668_REVIEW) != M668_REVIEW_SHA256 or
            _sha256(M668_MANIFEST) != M668_MANIFEST_SHA256 or
            _sha256(M668_OUTER) != M668_OUTER_SHA256):
        raise RegistryError("sealed M668 review root drift")
    review = M635.load_json(M668_REVIEW, "sealed M668 review")
    if (review.get("status") != "COMPLETE_NO_GO_P1_OPEN" or
            review.get("severity_counts") != {"P0": 0, "P1": 3, "P2": 2} or
            review.get("verdict") !=
            "NO_GO_METHODOLOGY__P1_REPAIR_AND_FRESH_HAMMER_REQUIRED"):
        raise RegistryError("sealed M668 review semantic root mismatch")
    manifest_text = M668_MANIFEST.read_text(encoding="utf-8")
    outer_text = M668_OUTER.read_text(encoding="utf-8")
    if (M668_REVIEW_SHA256 + "  review.json" not in manifest_text or
            outer_text != M668_MANIFEST_SHA256 + "  SHA256SUMS\n"):
        raise RegistryError("sealed M668 double seal mismatch")


def _validate_run_manifest(spec, row_id, configuration_sha):
    path, doc, digest = _file_spec(spec, "native rooted run manifest " + row_id,
                                   "hw_autoresearch_nts07/results/")
    _exact(doc, RUN_MANIFEST_FIELDS, "native rooted run manifest")
    if (doc["schema"] != "m671.h67.native_synopsys_run_manifest.r2" or
            doc["status"] != "FROZEN_ROOTED_NATIVE_TOOL_RUN" or
            doc["row_id"] != row_id or
            doc["configuration_manifest_sha256"] != configuration_sha or
            doc["configuration_manifest"].get("sha256") != configuration_sha or
            doc["m527_configuration_id"] != ROW_TO_M527_CONFIGURATION[row_id] or
            doc["operator_scope_sha256"] != _map_sha(_required_operator_scope()) or
            doc["design_name"] != _expected_design(row_id)):
        raise RegistryError("native run row/config/operator/design mismatch")
    configuration_path, _, config_digest = _file_spec(
        doc["configuration_manifest"], "native run configuration " + row_id,
        "hw_autoresearch_nts07/system_simulator/")
    if config_digest != configuration_sha:
        raise RegistryError("native run configuration SHA mismatch")
    try:
        _, parsed, _, _, tool_run, component_hashes = EX._load_manifest(spec["path"])
    except (OSError, EX.ExtractionError) as exc:
        raise RegistryError("native run provenance is invalid: %s" % exc)
    if parsed != doc:
        raise RegistryError("native run parser/document mismatch")
    expected_reports = path.parent / "reports"
    report_hashes = {}
    for name in RAW_REPORT_FIELDS:
        report_path, _, report_sha = _file_spec(
            doc["raw_reports"][name], "native output report " + name,
            "hw_autoresearch_nts07/results/", ("text/plain",))
        if report_path.parent != expected_reports:
            raise RegistryError("native output reports are not colocated in run reports/")
        report_hashes[name] = report_sha
    expected_run = _expected_run_id(row_id, configuration_sha, report_hashes)
    if doc["run_id"] != expected_run or path.parent.name != expected_run:
        raise RegistryError("native run ID/path mismatch")
    tool_path, _, tool_sha = _file_spec(doc["tool_run_receipt"], "tool run receipt " + row_id,
                                        "hw_autoresearch_nts07/results/")
    if tool_path.parent != path.parent:
        raise RegistryError("tool run receipt is not colocated with run manifest")
    clean_row = re.sub(r"[^a-zA-Z0-9_]+", "_", row_id)
    provenance_root = path.parent.parent / ("m671_provenance_%s_%s" %
                                             (clean_row, configuration_sha[:12]))
    rooted_groups = (
        ("command_scripts", "scripts"), ("tool_logs", "logs"),
    )
    for group, directory in rooted_groups:
        for name, item in tool_run[group].items():
            item_path, _, _ = _file_spec(item, group + " " + name,
                                          "hw_autoresearch_nts07/results/",
                                          ("text/plain",))
            if item_path.parent != provenance_root / directory:
                raise RegistryError(group + " is outside typed run directory")
    for name in ("netlist", "sdc", "activity"):
        item_path, _, _ = _file_spec(tool_run[name], "run input " + name,
                                      "hw_autoresearch_nts07/results/", ("text/plain",))
        if item_path.parent != provenance_root / "inputs":
            raise RegistryError("run input is outside typed run directory")
    if (not isinstance(tool_run["rtl_sources"], dict) or
            set(tool_run["rtl_sources"]) != set(EX.RTL_SOURCE_ROLES)):
        raise RegistryError("typed RTL/testbench/assertion source set mismatch")
    for role, item in tool_run["rtl_sources"].items():
        item_path, _, _ = _file_spec(
            item, "run RTL source " + role, "hw_autoresearch_nts07/results/",
            ("text/plain", "text/x-systemverilog"))
        if item_path.parent != provenance_root / "inputs" / "rtl":
            raise RegistryError("run RTL source is outside typed RTL directory")
    for name, entry in tool_run["tool_executables"].items():
        item_path, _, _ = _file_spec(entry["file"], "tool executable " + name,
                                      "hw_autoresearch_nts07/results/",
                                      ("application/octet-stream",))
        if item_path.parent != provenance_root / "tools":
            raise RegistryError("tool executable is outside typed run directory")
    for role, entry in tool_run["library_dbs"].items():
        item_path, _, _ = _file_spec(entry["file"], "library DB " + role,
                                      "hw_autoresearch_nts07/results/",
                                      ("application/octet-stream",))
        if item_path.parent != provenance_root / "libraries":
            raise RegistryError("library DB is outside typed run directory")
    return {"path": path, "doc": doc, "sha256": digest,
            "configuration_path": configuration_path, "report_hashes": report_hashes,
            "tool_run_sha256": tool_sha, "tool_run": tool_run,
            "component_hashes": component_hashes}


def _validate_extraction_receipt(spec, row_id, configuration_sha):
    _, receipt, digest = _file_spec(spec, "native PPA extraction receipt " + row_id,
                                    "hw_autoresearch_nts07/results/")
    _exact(receipt, EXTRACTION_RECEIPT_FIELDS, "native PPA extraction receipt")
    if (receipt["schema"] != "m671.h67.native_synopsys_ppa_extraction_receipt.r2" or
            receipt["status"] != "PASS_ROOTED_NATIVE_TOOL_RUN_PARSE" or
            receipt["row_id"] != row_id or
            receipt["configuration_manifest_sha256"] != configuration_sha):
        raise RegistryError("native PPA receipt schema/status/row mismatch")
    run = _validate_run_manifest(receipt["run_manifest"], row_id, configuration_sha)
    extractor_path, _, extractor_sha = _file_spec(
        receipt["extractor_source"], "native PPA extractor source",
        media_types=("text/x-python", "text/plain"))
    if extractor_path != EXTRACTOR or extractor_sha != EXTRACTOR_SHA256:
        raise RegistryError("native PPA receipt does not bind reviewed extractor")
    if (receipt["tool_run_receipt"] != run["doc"]["tool_run_receipt"] or
            receipt["raw_reports"] != run["doc"]["raw_reports"] or
            receipt["extraction_argv"] != _expected_argv(receipt["run_manifest"])):
        raise RegistryError("native PPA receipt run/report/argv root mismatch")
    try:
        extracted = EX.extract_from_manifest(receipt["run_manifest"]["path"])
    except (OSError, EX.ExtractionError) as exc:
        raise RegistryError("bound native PPA run is invalid: %s" % exc)
    expected_identity = {
        "row_id": row_id, "configuration_manifest_sha256": configuration_sha,
        "m527_configuration_id": ROW_TO_M527_CONFIGURATION[row_id],
        "operator_scope_sha256": _map_sha(_required_operator_scope()),
        "design_name": _expected_design(row_id), "run_id": run["doc"]["run_id"],
    }
    if extracted["run_identity"] != expected_identity:
        raise RegistryError("native extraction run identity mismatch")
    if (receipt["native_identities"] != extracted["identities"] or
            receipt["target_corners"] != extracted["corners"] or
            receipt["library_dbs"] != extracted["library_dbs"] or
            receipt["memory_inventory"] != extracted["memory_inventory"] or
            receipt["provenance_component_root_sha256"] !=
            extracted["provenance_component_root_sha256"] or
            receipt["provenance_component_root_sha256"] !=
            run["tool_run"]["component_root_sha256"]):
        raise RegistryError("native identity/corner/library/memory/provenance mismatch")
    expected_units = {field: ("mm2" if field.endswith("area_mm2") else
                              "ns" if field.endswith("wns_ns") else "mW")
                      for field in EXTRACTED_FIELDS}
    if receipt["units"] != expected_units:
        raise RegistryError("native PPA extraction units mismatch")
    _exact(receipt["extracted_values"], EXTRACTED_FIELDS, "native PPA values")
    for field, expected in extracted["values"].items():
        actual = _number(receipt["extracted_values"][field], "native PPA " + field,
                         zero_ok=True)
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise RegistryError("native PPA receipt value drift: " + field)
    return {"sha256": digest, "values": extracted["values"],
            "run_manifest_sha256": run["sha256"],
            "tool_run_receipt_sha256": run["tool_run_sha256"],
            "report_hashes": run["report_hashes"],
            "component_hashes": run["component_hashes"],
            "extractor_sha256": extractor_sha}


def _validate_typed_receipts(bundle_id, specs, raw, configs, summaries, measurement):
    _, energy, energy_sha = _file_spec(specs["energy_receipt"], "typed energy receipt",
                                       "hw_autoresearch_nts07/results/")
    _exact(energy, {"schema", "status", "raw_run_index_sha256", "rows"},
           "typed energy receipt")
    if (energy["schema"] != "m645.h67.logic_sram_dram_energy_receipt.r1" or
            energy["status"] != "PASS_TYPED" or
            energy["raw_run_index_sha256"] != raw["sha256"]):
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
    _, accuracy, accuracy_sha = _file_spec(specs["accuracy_receipt"],
                                           "typed accuracy receipt",
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
        _exact(row, {"row_id", "configuration_manifest_sha256", "aee",
                     "dsec_fl_percent"}, "accuracy row")
        row_id = row["row_id"]
        if (row_id not in MANDATORY_ROW_IDS or row_id in accuracy_rows or
                row["configuration_manifest_sha256"] != configs[row_id]["sha256"] or
                not math.isclose(row["aee"], summaries[row_id]["accuracy"], rel_tol=1e-12,
                                 abs_tol=1e-12) or
                not math.isclose(row["dsec_fl_percent"],
                                 summaries[row_id]["dsec_fl_percent"], rel_tol=1e-12,
                                 abs_tol=1e-12)):
            raise RegistryError("typed accuracy row mismatch")
        accuracy_rows[row_id] = row
    if set(accuracy_rows) != set(MANDATORY_ROW_IDS):
        raise RegistryError("typed accuracy receipt lacks six rows")
    _, ppa, ppa_sha = _file_spec(specs["ppa_receipt"], "typed native PPA receipt",
                                 "hw_autoresearch_nts07/results/")
    _exact(ppa, {"schema", "status", "technology_nm", "clock_period_ns", "rows"},
           "typed native PPA receipt")
    if (ppa["schema"] != "m671.h67.native_synopsys_rooted_ppa_receipt.r2" or
            ppa["status"] != "PASS_ROOTED_NATIVE_TOOL_RUN_PARSE" or
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
                                             "native extraction receipt",
                                             "hw_autoresearch_nts07/results/")
        evidence["ppa_extraction_receipt:" + row_id] = receipt_sha
        _, manifest, manifest_sha = _file_spec(receipt.get("run_manifest"),
                                               "native run manifest",
                                               "hw_autoresearch_nts07/results/")
        evidence["ppa_run_manifest:" + row_id] = manifest_sha
        _, tool_run, tool_sha = _file_spec(manifest.get("tool_run_receipt"),
                                           "native tool run receipt",
                                           "hw_autoresearch_nts07/results/")
        evidence["ppa_tool_run_receipt:" + row_id] = tool_sha
        for name, item in manifest["raw_reports"].items():
            _, _, digest = _file_spec(item, "native report", "hw_autoresearch_nts07/results/",
                                       ("text/plain",))
            evidence["ppa_native_report:%s:%s" % (row_id, name)] = digest
        for name in ("netlist", "sdc", "activity"):
            _, _, digest = _file_spec(tool_run[name], "native input",
                                       "hw_autoresearch_nts07/results/", ("text/plain",))
            evidence["ppa_run_input:%s:%s" % (row_id, name)] = digest
        for role, item in tool_run["rtl_sources"].items():
            _, _, digest = _file_spec(
                item, "native RTL source", "hw_autoresearch_nts07/results/",
                ("text/plain", "text/x-systemverilog"))
            evidence["ppa_rtl_source:%s:%s" % (row_id, role)] = digest
        for name, item in tool_run["command_scripts"].items():
            _, _, digest = _file_spec(item, "native command script",
                                       "hw_autoresearch_nts07/results/", ("text/plain",))
            evidence["ppa_command_script:%s:%s" % (row_id, name)] = digest
        for name, item in tool_run["tool_logs"].items():
            _, _, digest = _file_spec(item, "native tool log",
                                       "hw_autoresearch_nts07/results/", ("text/plain",))
            evidence["ppa_tool_log:%s:%s" % (row_id, name)] = digest
        for name, entry in tool_run["tool_executables"].items():
            _, _, digest = _file_spec(entry["file"], "native tool executable",
                                       "hw_autoresearch_nts07/results/",
                                       ("application/octet-stream",))
            evidence["ppa_tool_executable:%s:%s" % (row_id, name)] = digest
        for role, entry in tool_run["library_dbs"].items():
            _, _, digest = _file_spec(entry["file"], "native library DB",
                                       "hw_autoresearch_nts07/results/",
                                       ("application/octet-stream",))
            evidence["ppa_library_db:%s:%s" % (row_id, role)] = digest
    return evidence


def _validate_reviewed_targets(targets, bundle_id, evidence, numerator, result_spec):
    _exact(targets, REQUEST_TARGET_FIELDS, "independent review targets")
    fixed = {"registry_builder": Path(__file__), "registry_config": DEFAULT_CONFIG,
             "registry_tests": REGISTRY_TESTS, "registry_contract": REGISTRY_CONTRACT,
             "m527_contract": M527_CONTRACT, "checkpoint": CHECKPOINT}
    for name, path in fixed.items():
        media = (("application/octet-stream",) if name == "checkpoint" else
                 (("text/x-python", "text/plain") if name in
                  ("registry_builder", "registry_tests") else ("application/json",)))
        resolved, _, _ = _file_spec(targets[name], "review target " + name,
                                     media_types=media)
        if resolved != path:
            raise RegistryError("independent review target path mismatch: " + name)
    if targets["direct_result"] != result_spec:
        raise RegistryError("independent review does not bind exact direct result")
    context = R5._ACTIVE_REVIEW_CONTEXT
    if targets["fixed_throughput_numerator_receipt"] != context["numerator_spec"]:
        raise RegistryError("independent review does not bind exact fixed numerator")
    _file_spec(targets["direct_result"], "review direct result",
               "hw_autoresearch_nts07/results/")
    _file_spec(targets["fixed_throughput_numerator_receipt"], "review fixed numerator",
               "hw_autoresearch_nts07/results/")
    return {name: targets[name]["sha256"] for name in sorted(targets)}


def _validate_bundle(bundle_id, bundle):
    _exact(bundle, BUNDLE_FIELDS, "M671 Table-A direct bundle")
    if bundle["schema"] != "m671.h67.rooted_direct_bundle.r5" or bundle["bundle_id"] != bundle_id:
        raise RegistryError("M671 Table-A bundle schema/id mismatch")
    legacy = copy.deepcopy(bundle)
    legacy["schema"] = "m663.h67.rooted_direct_bundle.r4"
    previous = R7.TRUSTED_HAMMER_AUTHORITIES
    R7.TRUSTED_HAMMER_AUTHORITIES = TRUSTED_HAMMER_AUTHORITIES
    try:
        return R7._validate_bundle(bundle_id, legacy)
    finally:
        R7.TRUSTED_HAMMER_AUTHORITIES = previous


def _validate_overlay(config):
    fields = {"schema", "date", "status", "purpose", "base_registry",
              "table_a_evidence_bundles", "table_a_rows", "claim_boundary",
              "protected_file"}
    _exact(config, fields, "M671 registry overlay")
    if config["schema"] != "m671.h67.paper_metric_registry.r8":
        raise RegistryError("unexpected M671 registry schema")
    _, base, base_sha = _file_spec(config["base_registry"], "sealed M635 base registry",
                                   "hw_autoresearch_nts07/system_simulator/config/")
    if (base_sha != M635_CONFIG_SHA256 or
            config["base_registry"]["path"] != M635_CONFIG.relative_to(REPO_ROOT).as_posix() or
            base.get("schema") != "m635.h67.paper_metric_registry.r3" or
            base.get("table_a_evidence_bundles") != {}):
        raise RegistryError("M671 must inherit canonical M635 zero-bundle registry")
    protected = config["protected_file"]
    _exact(protected, {"path", "sha256"}, "protected file")
    if protected != {"path": DOCS359.relative_to(REPO_ROOT).as_posix(),
                      "sha256": DOCS359_SHA256}:
        raise RegistryError("protected docs359 binding mismatch")
    if _sha256(R5._secure_file(protected["path"])) != DOCS359_SHA256:
        raise RegistryError("protected docs359 SHA drift")
    return base


def build(config_path=DEFAULT_CONFIG):
    _validate_m668_root()
    _required_operator_scope()
    if _sha256(EXTRACTOR) != EXTRACTOR_SHA256:
        raise RegistryError("native run extractor SHA drift")
    config = M635.load_json(Path(config_path), "M671 registry")
    base = _validate_overlay(config)
    sources, source_docs = M635.validate_sources(base)
    base_rows = M635.validate_ladder_and_tables(base, set(sources))
    policy = M635.validate_policy(base)
    M635.validate_m518_binding(base, source_docs)
    analytical = M635.recompute_analytical(base)
    rows = copy.deepcopy(base_rows if config["table_a_rows"] is None else
                         config["table_a_rows"])
    validation_copy = copy.deepcopy(base)
    validation_copy["table_a_schema"]["rows"] = copy.deepcopy(rows)
    M635.validate_ladder_and_tables(validation_copy, set(sources))
    bundle_specs = config["table_a_evidence_bundles"]
    if not isinstance(bundle_specs, dict):
        raise RegistryError("table_a_evidence_bundles must be an object")
    bundles = {bundle_id: _validate_bundle(bundle_id, bundle)
               for bundle_id, bundle in bundle_specs.items()}
    gate = R6._evaluate(rows, bundles, config["claim_boundary"], policy)
    return {"schema": "m671.h67.paper_metric_registry.r8.preview",
            "status": config["status"], "source_hashes_validated": sources,
            "trusted_hammer_authority_count": len(TRUSTED_HAMMER_AUTHORITIES),
            "table_a_evidence_bundle_count": len(bundles), "table_a": rows,
            "table_b": base["table_b_schema"]["rows"],
            "table_c": base["table_c_schema"]["rows"],
            "analytical_diagnostic": analytical, "headline_gate": gate,
            "claim_boundary": config["claim_boundary"],
            "protected_file_validated": config["protected_file"]}


# Patch only the sealed predecessor's future-bundle call graph.
R7._collect_ppa_evidence = _collect_ppa_evidence
R7.R5._validate_reviewed_targets = _validate_reviewed_targets
R7.R5.R4._validate_typed_receipts = _validate_typed_receipts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--emit-json", action="store_true")
    args = parser.parse_args()
    try:
        result = build(args.config)
    except (RegistryError, RuntimeError) as exc:
        print("M671_REGISTRY_FAIL: %s" % exc)
        return 2
    if args.emit_json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True,
                         allow_nan=False))
    else:
        print("M671_REGISTRY_PASS sources=%d trusted_authorities=%d bundles=%d table_a_eligible=%d headline_admitted=%s analytical_admitted=%s" %
              (len(result["source_hashes_validated"]),
               result["trusted_hammer_authority_count"],
               result["table_a_evidence_bundle_count"],
               result["headline_gate"]["eligible_row_count"],
               str(result["headline_gate"]["admitted"]).lower(),
               str(result["analytical_diagnostic"]["admitted"]).lower()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
