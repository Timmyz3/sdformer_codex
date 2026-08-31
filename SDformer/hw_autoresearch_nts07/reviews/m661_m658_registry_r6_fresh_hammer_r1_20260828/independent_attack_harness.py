#!/usr/bin/env python3
"""Fresh CPU/static hammer for the sealed M658 registry-r6 methodology.

The reviewed author test module is used only as a disposable graph factory.
All decisions, attacks and assertions below are independent.  No production
authority or Table-A row is created by this harness.
"""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = REPO_ROOT / "hw_autoresearch_nts07"
BUILDER = HW_ROOT / "system_simulator/scripts/build_m658_h67_paper_metric_registry_r6.py"
EXTRACTOR = HW_ROOT / "system_simulator/scripts/extract_m658_synopsys_ppa_reports.py"
TESTS = HW_ROOT / "system_simulator/tests/test_m658_h67_paper_metric_registry_r6.py"
CONFIG = HW_ROOT / "system_simulator/config/m658_h67_paper_metric_registry_r6_20260828.json"
CONTRACT = HW_ROOT / "contracts/m658_h67_paper_metric_registry_r6_contract_r1_20260828.json"
REQUEST_DIR = HW_ROOT / "reviews/m659_m658_registry_r6_fresh_hammer_r1_REQUEST_20260828"
DOCS359 = HW_ROOT / "docs/359_DATE终局冻结_20260813.md"
NATIVE_DC_AREA = (HW_ROOT / "dc_handoff/runs/m62_p48_dc_3p000ns_r1b_20260823/"
                  "reports/area.rpt")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import " + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


T = load_module("m661_m658_author_fixture_factory", TESTS)
T.M658RegistryTests.setUpClass()
M = T.M


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def helper():
    item = T.M658RegistryTests(methodName="test_01_canonical_zero_authority_bundle_row_and_headline")
    item.base = T.M658RegistryTests.base
    return item


def rejected(label, fn):
    try:
        fn()
    except (M.RegistryError, M.EX.ExtractionError) as exc:
        return {"label": label, "rejected": True, "error": str(exc)}
    return {"label": label, "rejected": False, "error": None}


def accepted(label, fn):
    try:
        value = fn()
    except (M.RegistryError, M.EX.ExtractionError) as exc:
        return {"label": label, "accepted": False, "error": str(exc)}
    return {"label": label, "accepted": True, "result": value}


def canonical_check():
    value = M.build(CONFIG)
    return {
        "sources": len(value["source_hashes_validated"]),
        "trusted_authorities": value["trusted_hammer_authority_count"],
        "bundles": value["table_a_evidence_bundle_count"],
        "eligible_rows": value["headline_gate"]["eligible_row_count"],
        "headline": value["headline_gate"]["admitted"],
        "analytical": value["analytical_diagnostic"]["admitted"],
    }


def positive_probe():
    h = helper()
    try:
        with h._rooted_positive_fixture() as (config_path, bundle, _, _):
            value = M.build(config_path)
            measurement = h._load_spec(bundle["measurement_identity"])
            numerator = h._load_spec(bundle["fixed_throughput_numerator_receipt"])
            ppa = h._load_spec(bundle["ppa_receipt"])
            evidence = M._collect_ppa_evidence(bundle["ppa_receipt"])
            reports = []
            for row in ppa["rows"]:
                receipt = h._load_spec(row["extraction_receipt"])
                for report_class, spec in sorted(receipt["raw_reports"].items()):
                    path = REPO_ROOT / spec["path"]
                    reports.append({
                        "row_id": row["row_id"],
                        "class": report_class,
                        "sha256": spec["sha256"],
                        "bytes": path.stat().st_size,
                        "line_count": len(path.read_text(encoding="utf-8").splitlines()),
                    })
            return {
                "accepted": value["headline_gate"]["admitted"],
                "eligible_rows": value["headline_gate"]["eligible_row_count"],
                "required_scope": M._required_operator_scope(),
                "measurement_scope": measurement["operator_ids"],
                "numerator_included": numerator["included_operator_scope"],
                "numerator_excluded": numerator["excluded_operator_scope_with_reason"],
                "ppa_rows": len(ppa["rows"]),
                "raw_reports": len(reports),
                "ppa_provenance_evidence_roots": len(evidence),
                "report_line_count_range": [min(item["line_count"] for item in reports),
                                            max(item["line_count"] for item in reports)],
            }
    finally:
        h.doCleanups()


def scope_attack(location):
    h = helper()
    try:
        with h._rooted_positive_fixture() as (_, bundle, _, _):
            if location == "trace":
                measurement = h._load_spec(bundle["measurement_identity"])
                trace_spec = measurement["complete_trace_manifest"]
                trace = h._load_spec(trace_spec)
                trace["operator_scope"].remove("fc1")
                h._rewrite_json(trace_spec, trace)
                measurement["complete_trace_manifest"] = trace_spec
                h._rewrite_json(bundle["measurement_identity"], measurement)
                return rejected("trace scope missing fc1", lambda: M._validate_measurement_identity(
                    bundle["measurement_identity"], M.CHECKPOINT_SHA256))
            if location == "measurement":
                measurement = h._load_spec(bundle["measurement_identity"])
                measurement["operator_ids"].remove("fc1")
                h._rewrite_json(bundle["measurement_identity"], measurement)
                return rejected("measurement scope missing fc1", lambda: M._validate_measurement_identity(
                    bundle["measurement_identity"], M.CHECKPOINT_SHA256))
            measurement = M._validate_measurement_identity(bundle["measurement_identity"],
                                                             M.CHECKPOINT_SHA256)
            numerator = h._load_spec(bundle["fixed_throughput_numerator_receipt"])
            numerator["included_operator_scope"].remove("fc1")
            h._rewrite_json(bundle["fixed_throughput_numerator_receipt"], numerator)
            return rejected("numerator scope missing fc1", lambda: M._validate_numerator_receipt(
                bundle["fixed_throughput_numerator_receipt"], measurement))
    finally:
        h.doCleanups()


def receipt_attack(kind):
    h = helper()
    try:
        with h._rooted_positive_fixture() as (_, bundle, _, evidence):
            row_id = M.MANDATORY_ROW_IDS[0]
            spec = evidence[row_id]["extraction"]
            receipt = h._load_spec(spec)
            if kind == "three_line":
                report = receipt["raw_reports"]["dc_area"]
                h._rewrite_text(report, "logic_area_mm2 0.6\nlogic_power_mw 0.2\nsetup_wns_ns 0.0\n")
                receipt["raw_reports"]["dc_area"] = report
            elif kind == "tool_version":
                receipt["synopsys_tools"]["dc_area"]["version"] = "UNBOUND"
            elif kind == "extractor_source":
                receipt["extractor_source"] = copy.deepcopy(receipt["raw_reports"]["dc_area"])
                receipt["extractor_source"]["media_type"] = "text/plain"
            elif kind == "argv":
                receipt["extraction_argv"] = ["python3", "unreviewed.py"]
            elif kind == "library":
                receipt["libraries"]["dc_area"] = "unbound.db"
                receipt["library_identity_sha256"] = M.R5._map_sha(receipt["libraries"])
            elif kind == "corner":
                receipt["corners"]["dc_area"] = "unbound_corner"
            elif kind == "units":
                receipt["units"]["logic_area_mm2"] = "um2"
            elif kind == "extracted_value":
                receipt["extracted_values"]["logic_area_mm2"] = 9999.0
            elif kind == "raw_value":
                report = receipt["raw_reports"]["dc_area"]
                path = REPO_ROOT / report["path"]
                text = path.read_text(encoding="utf-8").replace(
                    "Total cell area (um2): 600000.000000",
                    "Total cell area (um2): 9999000000.000000")
                h._rewrite_text(report, text)
                receipt["raw_reports"]["dc_area"] = report
            else:
                raise AssertionError(kind)
            receipt["extraction_argv"] = (M._expected_argv(receipt["raw_reports"])
                                           if kind not in ("argv",) else receipt["extraction_argv"])
            h._rewrite_json(spec, receipt)
            return rejected(kind, lambda: M._validate_extraction_receipt(
                spec, row_id, bundle["configuration_manifests"][row_id]["sha256"]))
    finally:
        h.doCleanups()


def request_attack(kind):
    h = helper()
    try:
        with h._rooted_positive_fixture() as (config_path, _, authority, _):
            path = REPO_ROOT / authority["request_document"]["path"]
            request = json.loads(path.read_text(encoding="utf-8"))
            if kind == "wrong_target":
                request["reviewed_targets"]["registry_contract"] = h._spec(CONFIG, "application/json")
            elif kind == "omit_raw_report":
                key = next(key for key in request["bundle_evidence_sha256"]
                           if key.startswith("ppa_raw_report:"))
                del request["bundle_evidence_sha256"][key]
                request["complete_evidence_root_sha256"] = M.R5._map_sha(
                    request["bundle_evidence_sha256"])
            elif kind == "omit_extraction_receipt":
                key = next(key for key in request["bundle_evidence_sha256"]
                           if key.startswith("ppa_extraction_receipt:"))
                del request["bundle_evidence_sha256"][key]
                request["complete_evidence_root_sha256"] = M.R5._map_sha(
                    request["bundle_evidence_sha256"])
            else:
                raise AssertionError(kind)
            path.write_text(json.dumps(request, separators=(",", ":")), encoding="utf-8")
            authority["request_document"] = h._spec(path, "application/json")
            manifest, outer = h._seal_one(path.parent, authority["request_document"])
            authority["request_manifest"] = manifest
            authority["request_outer_seal"] = outer
            return rejected(kind, lambda: M.build(config_path))
    finally:
        h.doCleanups()


def native_report_probe():
    text = NATIVE_DC_AREA.read_text(encoding="utf-8")
    try:
        value = M.EX._header(text, "area", "dc_shell")
    except M.EX.ExtractionError as exc:
        return {"accepted": False, "error": str(exc), "path": str(NATIVE_DC_AREA.relative_to(REPO_ROOT)),
                "sha256": sha256(NATIVE_DC_AREA), "bytes": NATIVE_DC_AREA.stat().st_size}
    return {"accepted": True, "identity": value, "path": str(NATIVE_DC_AREA.relative_to(REPO_ROOT)),
            "sha256": sha256(NATIVE_DC_AREA), "bytes": NATIVE_DC_AREA.stat().st_size}


def wrong_design_probe():
    h = helper()
    try:
        with h._rooted_positive_fixture() as (_, bundle, _, evidence):
            row_id = M.MANDATORY_ROW_IDS[0]
            spec = evidence[row_id]["extraction"]
            receipt = h._load_spec(spec)
            report = receipt["raw_reports"]["dc_area"]
            path = REPO_ROOT / report["path"]
            text = path.read_text(encoding="utf-8").replace("Design : " + row_id,
                                                            "Design : wrong_configuration")
            h._rewrite_text(report, text)
            receipt["raw_reports"]["dc_area"] = report
            receipt["extraction_argv"] = M._expected_argv(receipt["raw_reports"])
            h._rewrite_json(spec, receipt)
            return accepted("wrong Design identity accepted by row provenance gate", lambda:
                            M._validate_extraction_receipt(
                                spec, row_id,
                                bundle["configuration_manifests"][row_id]["sha256"])["values"])
    finally:
        h.doCleanups()


def cross_row_report_probe():
    h = helper()
    try:
        with h._rooted_positive_fixture() as (_, bundle, _, evidence):
            source_row = M.MANDATORY_ROW_IDS[0]
            target_row = M.MANDATORY_ROW_IDS[1]
            source_receipt = h._load_spec(evidence[source_row]["extraction"])
            target_path = REPO_ROOT / evidence[target_row]["extraction"]["path"]
            spec = h._write_extraction(
                target_path.parent, target_row,
                bundle["configuration_manifests"][target_row]["sha256"],
                source_receipt["raw_reports"])
            return accepted("row-0 reports accepted as row-1 PPA", lambda:
                            M._validate_extraction_receipt(
                                spec, target_row,
                                bundle["configuration_manifests"][target_row]["sha256"])["values"])
    finally:
        h.doCleanups()


def main():
    attacks = [
        scope_attack("trace"), scope_attack("measurement"), scope_attack("numerator"),
        receipt_attack("three_line"), receipt_attack("tool_version"),
        receipt_attack("extractor_source"), receipt_attack("argv"),
        receipt_attack("library"), receipt_attack("corner"), receipt_attack("units"),
        receipt_attack("extracted_value"), receipt_attack("raw_value"),
        request_attack("wrong_target"), request_attack("omit_raw_report"),
        request_attack("omit_extraction_receipt"),
    ]
    result = {
        "schema": "m661.m658.registry_r6.independent_attack_summary.r1",
        "canonical": canonical_check(),
        "positive_methodology_probe": positive_probe(),
        "required_negative_attacks": attacks,
        "native_synopsys_report_compatibility_probe": native_report_probe(),
        "accepted_semantic_holes": [wrong_design_probe(), cross_row_report_probe()],
        "extractor_fields": sorted(M.EXTRACTED_FIELDS),
        "leakage_power_extracted": any("leakage" in key for key in M.EXTRACTED_FIELDS),
        "frozen_roots": {
            "builder": sha256(BUILDER), "extractor": sha256(EXTRACTOR),
            "tests": sha256(TESTS), "config": sha256(CONFIG), "contract": sha256(CONTRACT),
            "r5_builder": sha256(M.R5_BUILDER), "m527": sha256(M.M527_CONTRACT),
            "checkpoint": sha256(M.CHECKPOINT), "docs359": sha256(DOCS359),
            "request_json": sha256(REQUEST_DIR / "request.json"),
        },
        "temporary_fixture_persisted": False,
    }
    if not all(item["rejected"] for item in attacks):
        raise RuntimeError("one or more required negative attacks were admitted")
    if not all(item["accepted"] for item in result["accepted_semantic_holes"]):
        raise RuntimeError("expected semantic-hole probe did not reach the row gate")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
