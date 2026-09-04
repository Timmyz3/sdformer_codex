#!/usr/bin/env python3
"""Fresh independent M668 attacks for the M663 r7 methodology.

This harness writes only an ephemeral directory below results/ and removes it
in a finally block.  It never invokes Synopsys, GPU code, or production runs.
"""

from __future__ import print_function

import copy
import hashlib
import importlib.util
import json
import shutil
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
BUILDER = HW / "system_simulator/scripts/build_m663_h67_paper_metric_registry_r7.py"
RESULTS = HW / "results"
REAL_DC = HW / "dc_handoff/runs/m62_p48_dc_3p000ns_r1b_20260823/reports/area.rpt"
REAL_SETUP = HW / "dc_handoff/runs/m441_m433_to_m439_formality_ptsta_r1d_20260826/ptsta/reports/timing_setup_slow.rpt"
REAL_HOLD = HW / "dc_handoff/runs/m441_m433_to_m439_formality_ptsta_r1d_20260826/ptsta/reports/timing_hold_fast.rpt"
REAL_PTPX = HW / "dc_handoff/runs/m448r4_m431_m438_prelayout_stdcell_ptpx_tt0p9v25c_r4_20260826/reports/ptpx_power_primary_100ps.rpt"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import " + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = load("m668_fresh_m663_target", BUILDER)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spec(path, media_type):
    return {"path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256(path), "media_type": media_type}


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":"),
                               allow_nan=False), encoding="utf-8")


def native_texts(design, macro, dc_library):
    dc = """****************************************
Report : area
Design : %s
Version: V-2023.12-SP3
Date   : Fri Aug 28 12:00:00 2026
****************************************

Library(s) Used:

    %s (File: /fabricated/path/%s.db)

Number of cells: 7
Total cell area: 123456.000000
Total area: undefined
""" % (design, dc_library, dc_library)
    power = """****************************************
Report : Averaged Power
        -significant_digits
        -nosplit
        -unit mW
Design : %s
Version: W-2024.09-SP3
Date   : Fri Aug 28 12:00:00 2026
****************************************

                        Internal Switching Leakage Total (%%)
memory                  0.040000 0.010000 0.010000 0.060000 (20.0%%)

  Net Switching Power  = 0.060000 (20.0%%)
  Cell Internal Power  = 0.220000 (73.3%%)
  Cell Leakage Power   = 0.020000 (6.7%%)
Total Power            = 0.300000 (100.0%%)
""" % design
    setup = """****************************************
Report : timing
        -delay_type max
Design : %s
Version: W-2024.09-SP3
Date   : Fri Aug 28 12:00:00 2026
****************************************

  Path Type: max
  slack (MET) 0.125000
""" % design
    hold = """****************************************
Report : timing
        -delay_type min
Design : %s
Version: W-2024.09-SP3
Date   : Fri Aug 28 12:00:00 2026
****************************************

  Path Type: min
  slack (MET) 0.015000
""" % design
    macro_library = macro.lower()
    ds = """####*********************************************************************************************************************/
#### Software       : TSMC MEMORY COMPILER tsn28hpcpd127spsram_2012.02.00.d.180a */
#### Library Name   : %s (user specify : %s) */
#### Generated Time : 2026/08/28, 12:00:00 */
####*********************************************************************************************************************/

1. Area
  | Width(um) | Height(um) | Area (um^2) |
  | 500.0000 | 400.0000 | 200000.0000 |

2. Timing Specification
   2.2 SRAM timing:(Slow, 0.9000, 125.0000 deg.)

4.1 Static Power
    Leakage Current 10.0000 (uA) diagnostic only
4.2 Dynamic Power - Average
    Read 11.0000 (uA/MHz)
    Write 12.0000 (uA/MHz)
""" % (macro_library, macro)
    return {"dc_area": dc, "ptpx_power": power, "pt_setup": setup,
            "pt_hold": hold, "sram_macro": ds}


def make_run(base, row_id, configuration_sha, design=None, macro=None,
             corners=None, libraries=None):
    base.mkdir(parents=True, exist_ok=False)
    expected_design = M._expected_design(row_id)
    design = expected_design if design is None else design
    macro = ("M668_%s_SRAM" % row_id.upper()) if macro is None else macro
    dc_library = "tcbn28hpcplusbwp35p140ssg0p9v125c"
    texts = native_texts(design, macro, dc_library)
    report_hashes = {name: hashlib.sha256(text.encode("utf-8")).hexdigest()
                     for name, text in texts.items()}
    run_id = M._expected_run_id(row_id, configuration_sha, report_hashes)
    run_dir = base / run_id
    reports = run_dir / "reports"
    reports.mkdir(parents=True)
    names = {"dc_area": "area.rpt", "ptpx_power": "power.rpt",
             "pt_setup": "setup.rpt", "pt_hold": "hold.rpt",
             "sram_macro": "macro.ds"}
    report_specs = {}
    for name, text in texts.items():
        path = reports / names[name]
        path.write_text(text, encoding="utf-8")
        report_specs[name] = spec(path, "text/plain")
    tools = {
        "dc_area": {"tool": "dc_shell", "version": "V-2023.12-SP3"},
        "ptpx_power": {"tool": "pt_shell", "version": "W-2024.09-SP3"},
        "pt_setup": {"tool": "pt_shell", "version": "W-2024.09-SP3"},
        "pt_hold": {"tool": "pt_shell", "version": "W-2024.09-SP3"},
        "sram_macro": {"tool": "memory_compiler",
                       "version": "tsn28hpcpd127spsram_2012.02.00.d.180a"},
    }
    default_libraries = {
        "dc_area": dc_library,
        "ptpx_power": "UNPARSED_PTPX_LIBRARY",
        "pt_setup": "UNPARSED_SETUP_LIBRARY",
        "pt_hold": "UNPARSED_HOLD_LIBRARY",
        "sram_macro": macro.lower(),
    }
    if libraries is not None:
        default_libraries.update(libraries)
    default_corners = {name: "UNPARSED_FAKE_CORNER_" + name
                       for name in M.RAW_REPORT_FIELDS}
    if corners is not None:
        default_corners.update(corners)
    manifest = {
        "schema": "m663.h67.native_synopsys_run_manifest.r1",
        "status": "FROZEN_NATIVE_REPORTS", "row_id": row_id,
        "configuration_manifest_sha256": configuration_sha,
        "m527_configuration_id": M.ROW_TO_M527_CONFIGURATION[row_id],
        "operator_scope_sha256": M._map_sha(M._required_operator_scope()),
        "design_name": design, "macro_name": macro, "run_id": run_id,
        "raw_reports": report_specs, "tools": tools,
        "libraries": default_libraries, "corners": default_corners,
    }
    manifest_path = run_dir / "native_run_manifest.json"
    write_json(manifest_path, manifest)
    manifest_spec = spec(manifest_path, "application/json")
    extracted = M.EX.extract_from_manifest(manifest_path)
    units = {field: ("mm2" if field.endswith("area_mm2") else
                     "ns" if field.endswith("wns_ns") else "mW")
             for field in M.EXTRACTED_FIELDS}
    receipt = {
        "schema": "m663.h67.native_synopsys_ppa_extraction_receipt.r1",
        "status": "PASS_DIRECT_NATIVE_REPORT_PARSE", "row_id": row_id,
        "configuration_manifest_sha256": configuration_sha,
        "run_manifest": manifest_spec,
        "extractor_source": {
            "path": M.EXTRACTOR.relative_to(ROOT).as_posix(),
            "sha256": M.EXTRACTOR_SHA256, "media_type": "text/x-python"},
        "extraction_argv": M._expected_argv(manifest_spec),
        "raw_reports": report_specs,
        "native_identities": extracted["identities"], "tools": tools,
        "libraries": default_libraries, "corners": default_corners,
        "units": units, "extracted_values": extracted["values"],
    }
    receipt_path = run_dir / "native_extraction_receipt.json"
    write_json(receipt_path, receipt)
    return {
        "run_dir": run_dir, "reports_dir": reports,
        "manifest": manifest, "manifest_path": manifest_path,
        "manifest_spec": manifest_spec, "receipt": receipt,
        "receipt_path": receipt_path,
        "receipt_spec": spec(receipt_path, "application/json"),
        "report_specs": report_specs, "extracted": extracted,
    }


def expect_reject(label, function, contains=None):
    try:
        function()
    except Exception as exc:  # independent adversarial harness
        if contains is not None and contains not in str(exc):
            raise RuntimeError(label + " wrong rejection: " + str(exc))
        return {"status": "REJECTED", "exception": type(exc).__name__,
                "message": str(exc)}
    raise RuntimeError(label + " unexpectedly accepted")


def main():
    temp = Path(tempfile.mkdtemp(prefix=".m668_m663_attack_", dir=str(RESULTS)))
    checks = {}
    issues = {}
    try:
        dc_identity, dc_area = M.EX.parse_dc_area(REAL_DC)
        checks["real_repo_dc_area"] = {
            "design": dc_identity["design"], "logic_area_mm2": dc_area,
            "source_sha256": sha256(REAL_DC)}
        setup = M.EX.parse_pt_timing(REAL_SETUP, "max")
        hold = M.EX.parse_pt_timing(REAL_HOLD, "min")
        checks["real_repo_pt_setup_hold"] = {
            "setup_design": setup[0]["design"], "setup_wns_ns": setup[1],
            "hold_design": hold[0]["design"], "hold_wns_ns": hold[1]}
        ptpx = M.EX.parse_ptpx_power(REAL_PTPX)
        checks["real_repo_ptpx"] = {
            "design": ptpx[0]["design"],
            "total_leakage_power_mw": ptpx[1]["total_leakage_power_mw"],
            "total_power_mw": ptpx[1]["total_power_mw"]}

        row = M.MANDATORY_ROW_IDS[0]
        config_sha = hashlib.sha256(b"m668-real-config-placeholder").hexdigest()
        positive = make_run(temp / "positive", row, config_sha)
        run_validation = M._validate_run_manifest(
            positive["manifest_spec"], row, config_sha)
        receipt_validation = M._validate_extraction_receipt(
            positive["receipt_spec"], row, config_sha)
        checks["constructed_complete_native_grammar"] = {
            "status": "PASS", "run_id": run_validation["doc"]["run_id"],
            "report_count": len(run_validation["report_hashes"]),
            "total_power_mw": receipt_validation["values"]["total_power_mw"]}

        wrapper = temp / "numeric_wrapper.rpt"
        wrapper.write_text("logic_area_mm2 0.1\nlogic_power_mw 0.2\nsetup_wns_ns 0.0\n",
                           encoding="utf-8")
        checks["three_line_wrapper"] = expect_reject(
            "three-line wrapper", lambda: M.EX.parse_dc_area(wrapper))

        wrong_design = make_run(temp / "wrong_design", row, config_sha,
                                design="M668_WRONG_DESIGN")
        checks["consistent_wrong_design"] = expect_reject(
            "consistent wrong design",
            lambda: M._validate_run_manifest(
                wrong_design["manifest_spec"], row, config_sha),
            "row/config/operator/design")

        wrong_config = make_run(temp / "wrong_config", row, config_sha)
        manifest = wrong_config["manifest"]
        manifest["m527_configuration_id"] = "wrong_configuration"
        write_json(wrong_config["manifest_path"], manifest)
        wrong_config_spec = spec(wrong_config["manifest_path"], "application/json")
        checks["wrong_m527_configuration"] = expect_reject(
            "wrong M527 configuration",
            lambda: M._validate_run_manifest(wrong_config_spec, row, config_sha),
            "row/config/operator/design")

        wrong_scope = make_run(temp / "wrong_scope", row, config_sha)
        manifest = wrong_scope["manifest"]
        manifest["operator_scope_sha256"] = "0" * 64
        write_json(wrong_scope["manifest_path"], manifest)
        wrong_scope_spec = spec(wrong_scope["manifest_path"], "application/json")
        checks["wrong_operator_scope"] = expect_reject(
            "wrong operator scope",
            lambda: M._validate_run_manifest(wrong_scope_spec, row, config_sha),
            "row/config/operator/design")

        wrong_run = make_run(temp / "wrong_run", row, config_sha)
        manifest = wrong_run["manifest"]
        manifest["run_id"] = "wrong_run_id"
        write_json(wrong_run["manifest_path"], manifest)
        wrong_run_spec = spec(wrong_run["manifest_path"], "application/json")
        checks["wrong_run_id"] = expect_reject(
            "wrong run id",
            lambda: M._validate_run_manifest(wrong_run_spec, row, config_sha),
            "run identity/path")

        wrong_argv = make_run(temp / "wrong_argv", row, config_sha)
        receipt = wrong_argv["receipt"]
        receipt["extraction_argv"] = ["python3", "author_wrapper.py"]
        write_json(wrong_argv["receipt_path"], receipt)
        wrong_argv_spec = spec(wrong_argv["receipt_path"], "application/json")
        checks["wrong_extractor_argv"] = expect_reject(
            "wrong extraction argv",
            lambda: M._validate_extraction_receipt(
                wrong_argv_spec, row, config_sha), "argv")

        wrong_units = make_run(temp / "wrong_units", row, config_sha)
        receipt = wrong_units["receipt"]
        receipt["units"]["total_power_mw"] = "W"
        write_json(wrong_units["receipt_path"], receipt)
        wrong_units_spec = spec(wrong_units["receipt_path"], "application/json")
        checks["wrong_units"] = expect_reject(
            "wrong units",
            lambda: M._validate_extraction_receipt(
                wrong_units_spec, row, config_sha), "units")

        missing_leakage = temp / "missing_leakage.rpt"
        missing_leakage.write_text("\n".join(
            line for line in native_texts("x", "X", "L")["ptpx_power"].splitlines()
            if "Cell Leakage Power" not in line) + "\n", encoding="utf-8")
        checks["missing_leakage"] = expect_reject(
            "missing leakage", lambda: M.EX.parse_ptpx_power(missing_leakage),
            "leakage total")

        bad_total = temp / "bad_total.rpt"
        bad_total.write_text(native_texts("x", "X", "L")["ptpx_power"].replace(
            "Total Power            = 0.300000",
            "Total Power            = 9.300000"), encoding="utf-8")
        checks["total_arithmetic"] = expect_reject(
            "total arithmetic", lambda: M.EX.parse_ptpx_power(bad_total),
            "internal+switching+leakage")

        second_row = M.MANDATORY_ROW_IDS[1]
        second = make_run(temp / "cross_target", second_row, config_sha)
        source = make_run(temp / "cross_source", row, config_sha)
        target_manifest = second["manifest"]
        target_manifest["raw_reports"] = copy.deepcopy(source["report_specs"])
        write_json(second["manifest_path"], target_manifest)
        target_spec = spec(second["manifest_path"], "application/json")
        checks["cross_row_report_reuse"] = expect_reject(
            "cross-row reuse",
            lambda: M._validate_run_manifest(target_spec, second_row, config_sha),
            "colocated")

        wrong_macro = make_run(
            temp / "wrong_macro_consistent", row, config_sha,
            macro="TOTALLY_WRONG_UNBOUND_MACRO")
        macro_accepted = M._validate_extraction_receipt(
            wrong_macro["receipt_spec"], row, config_sha)
        issues["P1_macro_not_bound_to_configuration_or_resource"] = {
            "unexpected_accept": True,
            "macro_name": wrong_macro["manifest"]["macro_name"],
            "sram_library": wrong_macro["manifest"]["libraries"]["sram_macro"],
            "extracted_area_mm2": macro_accepted["values"]["sram_macro_area_mm2"],
            "reason": "manifest and .ds agree, but no expected macro is derived from the row/config/resource manifest",
        }

        metadata = make_run(temp / "unbound_metadata", row, config_sha)
        metadata_accepted = M._validate_extraction_receipt(
            metadata["receipt_spec"], row, config_sha)
        issues["P1_native_library_corner_maps_are_not_report_derived"] = {
            "unexpected_accept": True,
            "accepted_libraries": metadata["manifest"]["libraries"],
            "accepted_corners": metadata["manifest"]["corners"],
            "value_count": len(metadata_accepted["values"]),
            "reason": "PT libraries and every corner are only nonempty manifest strings; DC/SRAM libraries alone are parsed",
        }

        forged = make_run(temp / "handwritten_native_lookalike", row, config_sha)
        forged_accepted = M._validate_extraction_receipt(
            forged["receipt_spec"], row, config_sha)
        issues["P1_native_generation_provenance_is_absent"] = {
            "unexpected_accept": True,
            "accepted_total_power_mw": forged_accepted["values"]["total_power_mw"],
            "missing_run_roots": ["generation_argv", "netlist_sha256", "sdc_sha256",
                                  "library_db_sha256", "saif_sha256", "tool_log_sha256"],
            "reason": "a fully hand-written native-looking report set is indistinguishable from a tool-generated run",
        }

        alias = positive["reports_dir"] / "area_alias.rpt"
        alias.symlink_to(positive["reports_dir"] / "area.rpt")
        alias_manifest = copy.deepcopy(positive["manifest"])
        alias_manifest["raw_reports"]["dc_area"] = spec(alias, "text/plain")
        alias_path = positive["run_dir"] / "alias_manifest.json"
        write_json(alias_path, alias_manifest)
        alias_result = M.EX.extract_from_manifest(alias_path)
        builder_alias_spec = spec(alias_path, "application/json")
        builder_alias = expect_reject(
            "builder symlink", lambda: M._validate_run_manifest(
                builder_alias_spec, row, config_sha), "symlink")
        issues["P2_standalone_extractor_resolves_symlink_before_check"] = {
            "direct_extractor_unexpected_accept": True,
            "direct_design": alias_result["run_identity"]["design_name"],
            "builder_wrapper": builder_alias,
        }

        traversal_manifest = copy.deepcopy(positive["manifest"])
        original = traversal_manifest["raw_reports"]["dc_area"]["path"]
        traversal_manifest["raw_reports"]["dc_area"]["path"] = original.replace(
            "/reports/area.rpt", "/reports/../reports/area.rpt")
        traversal_path = positive["run_dir"] / "traversal_manifest.json"
        write_json(traversal_path, traversal_manifest)
        traversal_result = M.EX.extract_from_manifest(traversal_path)
        traversal_spec = spec(traversal_path, "application/json")
        builder_traversal = expect_reject(
            "builder traversal", lambda: M._validate_run_manifest(
                traversal_spec, row, config_sha), "unsafe path")
        issues["P2_standalone_extractor_accepts_parent_traversal"] = {
            "direct_extractor_unexpected_accept": True,
            "direct_design": traversal_result["run_identity"]["design_name"],
            "builder_wrapper": builder_traversal,
        }

        ppa_rows = []
        for index, row_id in enumerate(M.MANDATORY_ROW_IDS):
            run = make_run(temp / ("evidence_%d" % index), row_id, config_sha)
            values = dict(run["extracted"]["values"])
            values.update({
                "row_id": row_id,
                "configuration_manifest_sha256": config_sha,
                "total_area_mm2": (values["logic_area_mm2"] +
                                   values["sram_macro_area_mm2"]),
                "extraction_receipt": run["receipt_spec"]})
            ppa_rows.append(values)
        ppa_path = temp / "six_row_ppa.json"
        write_json(ppa_path, {
            "schema": "m663.h67.native_synopsys_rooted_ppa_receipt.r1",
            "status": "PASS_DIRECT_NATIVE_REPORT_PARSE",
            "technology_nm": 28, "clock_period_ns": 3.0, "rows": ppa_rows})
        evidence = M._collect_ppa_evidence(spec(ppa_path, "application/json"))
        checks["evidence_roots"] = {
            "total": len(evidence),
            "extractors": len([key for key in evidence
                               if key == "ppa_native_extractor_source"]),
            "receipts": len([key for key in evidence
                             if key.startswith("ppa_extraction_receipt:")]),
            "manifests": len([key for key in evidence
                              if key.startswith("ppa_run_manifest:")]),
            "native_reports": len([key for key in evidence
                                   if key.startswith("ppa_native_report:")]),
        }

        canonical = M.build(M.DEFAULT_CONFIG)
        checks["exact_scope_and_canonical"] = {
            "operator_scope": M._required_operator_scope(),
            "operator_scope_count": len(M._required_operator_scope()),
            "trusted_authorities": canonical["trusted_hammer_authority_count"],
            "bundles": canonical["table_a_evidence_bundle_count"],
            "eligible_rows": canonical["headline_gate"]["eligible_row_count"],
            "headline_admitted": canonical["headline_gate"]["admitted"],
            "claim_boundary": canonical["claim_boundary"],
        }

        output = {
            "schema": "m668_m663_registry_r7_fresh_attack_result_v1",
            "status": "NO_GO_METHODOLOGY_P1_OPEN",
            "checks": checks, "issues": issues,
            "severity_counts": {"P0": 0, "P1": 3, "P2": 2},
        }
        print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
        return 1
    finally:
        shutil.rmtree(str(temp))


if __name__ == "__main__":
    raise SystemExit(main())
