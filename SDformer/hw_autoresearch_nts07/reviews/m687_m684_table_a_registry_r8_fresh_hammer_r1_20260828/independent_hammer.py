#!/usr/bin/env python3
"""Independent CPU-only hammer for the M684/M671 r8 author handoff.

This test intentionally imports the author's fixture only to construct bytes
that the reviewed extractor itself accepts.  It launches no EDA/GPU process
and never writes outside TemporaryDirectory instances owned by that fixture.
"""

import importlib.util
import json
import math
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = REPO_ROOT / "hw_autoresearch_nts07"
EXTRACTOR = HW_ROOT / "system_simulator/scripts/extract_m671_native_synopsys_run_provenance.py"
BUILDER = HW_ROOT / "system_simulator/scripts/build_m671_h67_paper_metric_registry_r8.py"
AUTHOR_TESTS = HW_ROOT / "system_simulator/tests/test_m671_h67_paper_metric_registry_r8.py"
CONFIG = HW_ROOT / "system_simulator/config/m671_h67_paper_metric_registry_r8_20260828.json"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import " + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EX = load("m687_extractor", EXTRACTOR)
M = load("m687_builder", BUILDER)
T = load("m687_author_tests", AUTHOR_TESTS)


def expect_reject(label, function, contains):
    try:
        function()
    except EX.ExtractionError as exc:
        if contains not in str(exc):
            raise AssertionError("%s rejected for wrong reason: %s" % (label, exc))
        return str(exc)
    raise AssertionError(label + " did not reject")


def main():
    checks = {}
    fixture = T.NativeFixture()
    try:
        manifest_relative = fixture.manifest_path.relative_to(REPO_ROOT).as_posix()

        # The reviewed extractor accepts the author's wholly synthetic chain.
        extracted = EX.extract_from_manifest(manifest_relative)
        checks["synthetic_chain_extracts"] = True
        checks["synthetic_vcs_status"] = extracted["identities"]["vcs_simulation"]["status"]
        checks["synthetic_formality_status"] = extracted["identities"]["formality_verification"]["status"]
        checks["synthetic_memory_bytes"] = extracted["memory_inventory"]["macro_rounded_total_bytes"]
        checks["synthetic_parent_port"] = extracted["memory_inventory"]["macros"][2]["port_type"]

        design_text = (REPO_ROOT / fixture.rtl_sources["design_rtl"]["path"]).read_text()
        netlist_text = (REPO_ROOT / fixture.netlist["path"]).read_text()
        sdc_text = (REPO_ROOT / fixture.sdc["path"]).read_text()
        saif_text = (REPO_ROOT / fixture.activity["path"]).read_text()
        checks["design_is_empty_module"] = design_text.strip() == (
            "module %s; endmodule" % fixture.design)
        checks["netlist_is_empty_module"] = netlist_text.strip() == (
            "module %s; endmodule" % fixture.design)
        checks["sdc_is_single_clock_line"] = len(sdc_text.strip().splitlines()) == 1
        checks["saif_is_header_only"] = saif_text.strip() == '(SAIFILE (SAIFVERSION "2.0"))'
        checks["all_tool_snapshots_are_plaintext_fixtures"] = all(
            (REPO_ROOT / entry["file"]["path"]).read_bytes().startswith(
                b"M671_BINARY_TOOL_FIXTURE_")
            for entry in fixture.executables.values())
        checks["all_db_snapshots_are_plaintext_fixtures"] = all(
            (REPO_ROOT / entry["file"]["path"]).read_bytes().startswith(
                b"M671_BINARY_DB_FIXTURE_")
            for entry in fixture.libraries.values())

        receipt = fixture.load_tool_receipt()
        vcs_argv = receipt["generation_argv"]["vcs"]
        checks["vcs_argv_shape"] = ["<rooted-vcs>"] + vcs_argv[1:-1] + ["<rooted-script>"]
        checks["vcs_argv_has_no_run_phase"] = (
            "-R" not in vcs_argv and not any("simv" in item for item in vcs_argv))
        checks["vcs_argv_has_no_source_path"] = all(
            item["path"] not in vcs_argv for item in fixture.rtl_sources.values())
        command_text = {
            step: (REPO_ROOT / spec["path"]).read_text()
            for step, spec in receipt["command_scripts"].items()
        }
        checks["command_scripts_are_metadata_only"] = all(
            text.startswith("M671_COMMAND_ROOT_BEGIN\n") and
            not any(token in text for token in (
                "read_verilog", "analyze -format", "elaborate ", "compile_ultra",
                "read_netlist", "read_sdc", "read_saif", "report_power",
                "set_top", "run 0", "-R"))
            for text in command_text.values())

        # Independent parser fixture: these bytes are authored here, not by the
        # reviewed test suite, and nevertheless satisfy the two "native" proof
        # parsers using only version/result marker text.
        with tempfile.TemporaryDirectory(dir=str(HW_ROOT / "results"),
                                         prefix="m687_independent_parser_fixture_") as parser_tmp:
            parser_tmp = Path(parser_tmp)
            forged_vcs = parser_tmp / "forged_vcs.log"
            forged_vcs.write_text(
                "Compiler version V-2023.12-SP1_Full64; Runtime version "
                "V-2023.12-SP1_Full64; independently forged\n"
                "M671_TABLE_A_VCS_PASS\n", encoding="utf-8")
            forged_fm = parser_tmp / "forged_formality.log"
            forged_fm.write_text(
                " Version V-2023.12-SP3 for linux64 - independently forged\n"
                "Verification SUCCEEDED\n", encoding="utf-8")
            checks["independent_forged_vcs_parser_status"] = (
                EX.parse_vcs_simulation(forged_vcs)["status"])
            checks["independent_forged_formality_parser_status"] = (
                EX.parse_formality_verification(forged_fm)["status"])

        # Canonical state remains safely empty despite the permissive extractor.
        canonical = M.build(CONFIG)
        checks["canonical_trusted_authorities"] = canonical["trusted_hammer_authority_count"]
        checks["canonical_bundles"] = canonical["table_a_evidence_bundle_count"]
        checks["canonical_eligible_rows"] = canonical["headline_gate"]["eligible_row_count"]
        checks["canonical_headline"] = canonical["headline_gate"]["admitted"]
        checks["canonical_analytical"] = canonical["analytical_diagnostic"]["admitted"]

        # Standalone path attacks, including an intermediate symlink.
        path_errors = {}
        attacks = {
            "absolute": "/tmp/m687-nope",
            "dot": "hw_autoresearch_nts07/results/./m687-nope",
            "dotdot": "hw_autoresearch_nts07/results/../results/m687-nope",
            "double_separator": "hw_autoresearch_nts07/results//m687-nope",
            "backslash": "hw_autoresearch_nts07/results\\m687-nope",
        }
        for label, value in attacks.items():
            path_errors[label] = expect_reject(
                label, lambda value=value: EX._secure_repo_file(value, label), "path")
        target_dir = fixture.result_root / "real_dir"
        target_dir.mkdir()
        target_file = target_dir / "payload.txt"
        target_file.write_text("payload\n", encoding="utf-8")
        intermediate = fixture.result_root / "linked_dir"
        intermediate.symlink_to(target_dir.name, target_is_directory=True)
        path_errors["intermediate_symlink"] = expect_reject(
            "intermediate_symlink",
            lambda: EX._secure_repo_file(
                (intermediate / "payload.txt").relative_to(REPO_ROOT).as_posix(),
                "intermediate_symlink"),
            "symlink")
        checks["path_rejections"] = path_errors

        # Obvious negative and zero physical values are rejected.
        with tempfile.TemporaryDirectory(dir=str(fixture.run_dir)) as tmp:
            tmp = Path(tmp)
            zero_area = tmp / "zero_area.rpt"
            original_area = (REPO_ROOT / fixture.reports["dc_area"]["path"]).read_text()
            zero_area.write_text(original_area.replace("600000.0", "0.0"), encoding="utf-8")
            checks["zero_area_reject"] = expect_reject(
                "zero_area", lambda: EX.parse_dc_area(zero_area), "not positive")
            negative_power = tmp / "negative_power.rpt"
            original_power = (REPO_ROOT / fixture.reports["ptpx_power"]["path"]).read_text()
            negative_power.write_text(
                original_power.replace("0.010000 0.010000 0.060000",
                                       "0.010000 -0.010000 0.040000"),
                encoding="utf-8")
            checks["negative_power_reject"] = expect_reject(
                "negative_power", lambda: EX.parse_ptpx_power(negative_power), "negative")

            # Direct extractor's 1e-12 residual tolerance permits a tiny negative
            # logic component after SRAM subtraction.  The registry's typed-number
            # gate later rejects it, so this is a direct-extractor P2, not admission.
            tiny = tmp / "tiny_negative_residual.rpt"
            tiny.write_text(
                original_power.replace(
                    "memory 0.040000 0.010000 0.010000 0.060000",
                    "memory 0.2400000000005 0.000000 0.000000 0.2400000000005"),
                encoding="utf-8")
            _, parsed_power = EX.parse_ptpx_power(tiny)
            residual = (parsed_power["total_internal_power_mw"] -
                        parsed_power["sram_internal_power_mw"])
            checks["tiny_negative_logic_residual_mw"] = residual
            checks["tiny_negative_passes_extractor_guard"] = residual < 0.0 and not residual < -1e-12
            try:
                M._number(residual, "tiny negative", zero_ok=True)
            except M.RegistryError:
                checks["tiny_negative_rejected_by_registry_number_gate"] = True
            else:
                checks["tiny_negative_rejected_by_registry_number_gate"] = False

        # Area semantics: DC Total cell area is labelled logic area and macro
        # datasheet area is then separate; no hierarchical macro subtraction is
        # present in the accepted chain.
        checks["accepted_dc_total_cell_area_mm2_as_logic"] = extracted["values"]["logic_area_mm2"]
        checks["accepted_macro_datasheet_area_mm2"] = extracted["values"]["sram_macro_area_mm2"]
        checks["would_report_sum_mm2"] = (
            extracted["values"]["logic_area_mm2"] +
            extracted["values"]["sram_macro_area_mm2"])

        required_true = (
            "synthetic_chain_extracts", "design_is_empty_module",
            "netlist_is_empty_module", "sdc_is_single_clock_line",
            "saif_is_header_only", "all_tool_snapshots_are_plaintext_fixtures",
            "all_db_snapshots_are_plaintext_fixtures", "vcs_argv_has_no_run_phase",
            "vcs_argv_has_no_source_path", "command_scripts_are_metadata_only",
            "independent_forged_vcs_parser_status",
            "independent_forged_formality_parser_status",
            "tiny_negative_passes_extractor_guard",
            "tiny_negative_rejected_by_registry_number_gate",
        )
        assert all(checks[name] for name in required_true)
        assert checks["independent_forged_vcs_parser_status"] == "PASS"
        assert checks["independent_forged_formality_parser_status"] == "SUCCEEDED"
        assert checks["synthetic_memory_bytes"] == 245760
        assert checks["synthetic_parent_port"] == "1R1W"
        assert checks["canonical_trusted_authorities"] == 0
        assert checks["canonical_bundles"] == 0
        assert checks["canonical_eligible_rows"] == 0
        assert checks["canonical_headline"] is False
        assert checks["canonical_analytical"] is False
        assert len(checks["path_rejections"]) == 6
        print(json.dumps({
            "schema": "m687.m684.registry_r8.independent_hammer.r1",
            "status": "PASS_ATTACKS__CANONICAL_ZERO__PRODUCTION_GATE_NO_GO",
            "checks": checks,
        }, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
        return 0
    finally:
        fixture.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
