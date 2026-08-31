#!/usr/bin/env python3
"""M695 receipt-blind adversarial hammer for the M691 r9 ingress gate.

This test never launches EDA or GPU work.  It reuses the author's public
grammar fixture to ask whether forged evidence can cross the exact production
manifest boundary, then tests executable/argv coupling and the independently
derivable scope/activity/macro claims.
"""

import copy
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = REPO_ROOT / "hw_autoresearch_nts07"
AUTHOR_TEST = HW_ROOT / "system_simulator/tests/test_m691_h67_paper_metric_registry_r9.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load " + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    author = load_module("m695_receipt_blind_m691_fixture", AUTHOR_TEST)

    # Instantiate the author's exact r9 upgrade on the mandatory M527 b0
    # identity.  The published author fixture accidentally retained the older
    # pre-M527 configuration_id, so it only exercises the extractor, not the
    # registry ingress.  This subclass changes identities before any artifact
    # is generated; all downstream SHA/run roots are still built by author
    # code.
    class M695MandatoryNativeFixture(author.R8T.NativeFixture):
        def __init__(self):
            self.result_tmp = tempfile.TemporaryDirectory(
                dir=str(HW_ROOT / "results"), prefix="m695_native_fixture_")
            self.config_tmp = tempfile.TemporaryDirectory(
                dir=str(HW_ROOT / "system_simulator/tests"),
                prefix="m695_config_fixture_")
            self.result_root = Path(self.result_tmp.name)
            self.config_root = Path(self.config_tmp.name)
            self.row_id = "dense96_fixed_t10"
            self.design = "h67_table_a_b0_dense96_fixed_t10"
            self.config = self._write_json(
                self.config_root / "configuration.json",
                {
                    "configuration_id": "b0_dense96_fixed_t10",
                    "resource_tuple": {
                        "onchip_sram_bytes_total": 245760,
                        "weight_sram_bank_count": 8,
                        "state_sram_bank_count": 8,
                        "parent_scratch_bank_count": 1,
                        "weight_sram_port_mode": "1R1W",
                        "state_sram_port_mode": "1R1W",
                        "parent_scratch_port_mode": "1R1W",
                    },
                },
            )
            self.provenance_root = self.result_root / (
                "m671_provenance_%s_%s" %
                (self.row_id, self.config["sha256"][:12]))
            for directory in ("inputs", "inputs/rtl", "libraries", "tools",
                              "scripts", "logs"):
                (self.provenance_root / directory).mkdir(
                    parents=True, exist_ok=True)
            self._make_inputs()
            self._make_run()
            self.manifest_doc["m527_configuration_id"] = (
                "b0_dense96_fixed_t10")

    original_native_fixture = author.R8T.NativeFixture
    author.R8T.NativeFixture = M695MandatoryNativeFixture
    try:
        fixture = author.R9Fixture()
    finally:
        author.R8T.NativeFixture = original_native_fixture
    results = {}
    try:
        manifest_rel = fixture.base.manifest_path.relative_to(REPO_ROOT).as_posix()

        # Attack A: the declared positive fixture uses /bin/true for every
        # Synopsys executable and simv, plus an ELF image with appended bytes
        # for every alleged .db.  If this extracts, media shape is being used
        # as tool/database authenticity.
        extracted = author.EX.extract_from_manifest(manifest_rel)
        true_sha = sha256("/bin/true")
        tool_shas = {
            name: item["file"]["sha256"]
            for name, item in fixture.base.executables.items()
        }
        db_paths = [
            REPO_ROOT / item["file"]["path"]
            for item in fixture.base.libraries.values()
        ]
        if set(tool_shas.values()) != {true_sha}:
            raise AssertionError("author fixture no longer uses one /bin/true image")
        if not all(path.read_bytes().startswith(b"\x7fELF") for path in db_paths):
            raise AssertionError("expected forged ELF-plus-bytes DB fixture")
        results["A_fake_toolchain_and_db_extracts"] = {
            "status": "FALSE_POSITIVE_CONFIRMED",
            "tool_sha_count": len(set(tool_shas.values())),
            "all_tool_shas_equal_bin_true": True,
            "all_fake_dbs_start_with_elf_magic": True,
            "extracted_total_area_mm2": (
                extracted["values"]["logic_area_mm2"] +
                extracted["values"]["sram_macro_area_mm2"]),
            "extracted_total_power_mw": extracted["values"]["total_power_mw"],
        }

        # Attack B: put the same forged manifest into the production map.  It
        # must not become a validated production run merely because it matches
        # the grammar.  No authority or headline is expected here; the attack
        # is against the future production ingress predicate itself.
        canonical = json.loads(author.CONFIG.read_text(encoding="utf-8"))
        canonical["production_run_manifests"] = {
            fixture.base.row_id: fixture.base.manifest_spec
        }
        attack_config = fixture.base.config_root / "m695_fake_production_config.json"
        attack_config.write_text(
            json.dumps(canonical, sort_keys=True, separators=(",", ":"),
                       allow_nan=False),
            encoding="utf-8",
        )
        registry = author.M.build(attack_config)
        if registry["validated_production_run_count"] != 1:
            raise AssertionError("forged production fixture did not cross ingress")
        results["B_fake_manifest_enters_validated_production_map"] = {
            "status": "FALSE_POSITIVE_CONFIRMED",
            "validated_production_run_count": 1,
            "authority_count": registry["trusted_hammer_authority_count"],
            "bundle_count": registry["table_a_evidence_bundle_count"],
            "headline_admitted": registry["headline_gate"]["admitted"],
        }

        # Attack C: execute-file identity is not coupled to argv[0].  Replace
        # the DC step executable with a different ELF while keeping argv[0]
        # rooted at the alleged dc_shell.  Re-root the self-authored text log;
        # a real execution predicate must reject this mismatch.
        original = copy.deepcopy(fixture.proof_original)
        proof = copy.deepcopy(original)
        false_path = fixture.base.provenance_root / "tools/m695_unrelated_false"
        false_path.write_bytes(Path("/bin/false").read_bytes())
        false_path.chmod(0o755)
        false_spec = fixture._spec(false_path, "application/octet-stream")
        dc = proof["execution_steps"]["dc"]
        alleged_argv0 = dc["argv"][0]
        dc["executable"] = false_spec
        log_text = (
            REPO_ROOT / dc["log"]["path"]
        ).read_text(encoding="utf-8")
        log_text = log_text.replace(
            "EXECUTABLE_SHA256 " + original["execution_steps"]["dc"]["executable"]["sha256"],
            "EXECUTABLE_SHA256 " + false_spec["sha256"],
        )
        dc["log"] = fixture._write(
            fixture.base.provenance_root / "r9_logs/dc_m695_decoupled.log",
            log_text,
        )
        fixture.rewrite_proof(proof)
        author.EX.extract_from_manifest(manifest_rel)
        results["C_step_executable_is_decoupled_from_argv0"] = {
            "status": "FALSE_POSITIVE_CONFIRMED",
            "entry_executable": false_spec["path"],
            "argv0": alleged_argv0,
            "paths_differ": false_spec["path"] != alleged_argv0,
            "sha_differ": false_spec["sha256"] !=
                          fixture.base.executables["dc_shell"]["file"]["sha256"],
        }
        fixture.rewrite_proof(copy.deepcopy(original))

        # Attack D: the accepted "full scope" is ten behavioral stubs plus a
        # top that instantiates them; the mapped netlist is the same source.
        # Presence regexes do not establish the frozen 10-op implementation.
        rtl_text = (
            REPO_ROOT / fixture.base.rtl_sources["design_rtl"]["path"]
        ).read_text(encoding="utf-8")
        netlist_text = (
            REPO_ROOT / fixture.base.netlist["path"]
        ).read_text(encoding="utf-8")
        stub_leaf_count = sum(
            ("module %s; wire alive; endmodule" % module) in rtl_text
            for _, module, _ in author.EX.SCOPE_ANCHORS
        )
        if stub_leaf_count != 10 or rtl_text != netlist_text:
            raise AssertionError("expected ten-stub scope fixture")
        author.EX.extract_from_manifest(manifest_rel)
        results["D_stub_scope_and_unmapped_netlist_extract"] = {
            "status": "FALSE_POSITIVE_CONFIRMED",
            "behavioral_stub_leaf_modules": stub_leaf_count,
            "rtl_equals_netlist_byte_for_byte": True,
        }

        # Attack E: one toggling SAIF net is accepted alongside self-reported
        # 99/100 and 198/200 coverage.  No annotation report derives either
        # numerator or denominator from SAIF/netlist.
        saif_text = (
            REPO_ROOT / fixture.base.activity["path"]
        ).read_text(encoding="utf-8")
        tc_count = saif_text.count("(TC ")
        author.EX.extract_from_manifest(manifest_rel)
        results["E_saif_coverage_is_self_reported"] = {
            "status": "FALSE_POSITIVE_CONFIRMED",
            "actual_tc_entries_in_saif": tc_count,
            "claimed_annotated_nets": original["saif_annotation"]["annotated_nets"],
            "claimed_total_nets": original["saif_annotation"]["total_nets"],
            "claimed_annotated_pins": original["saif_annotation"]["annotated_pins"],
            "claimed_total_pins": original["saif_annotation"]["total_pins"],
        }

        # Attack F: the PTPX report contains no SRAM instance names, and the
        # DC area report has no macro/logic split.  Both 17-instance power and
        # area inclusion are supplied solely by production_proof JSON.
        ptpx_text = (
            REPO_ROOT / fixture.base.reports["ptpx_power"]["path"]
        ).read_text(encoding="utf-8")
        dc_text = (
            REPO_ROOT / fixture.base.reports["dc_area"]["path"]
        ).read_text(encoding="utf-8")
        instance_names = [
            row["instance"] for row in original["macro_census"]["instances"]
        ]
        if any(name in ptpx_text for name in instance_names):
            raise AssertionError("fixture unexpectedly has PTPX instance census")
        if "logic_cell_area" in dc_text or "macro_cell_area" in dc_text:
            raise AssertionError("fixture unexpectedly has DC area split")
        author.EX.extract_from_manifest(manifest_rel)
        results["F_macro_area_and_ptpx_census_are_self_reported"] = {
            "status": "FALSE_POSITIVE_CONFIRMED",
            "ptpx_report_contains_exact_macro_instances": False,
            "dc_report_contains_logic_macro_split": False,
            "proof_instance_count": len(instance_names),
        }

        # Controls required by M695.
        author.EX._reject_negative_power({
            "logic_internal_power_mw": 0.0,
            "total_power_mw": 1.0,
        })
        try:
            author.EX._reject_negative_power({
                "logic_internal_power_mw": -5e-13,
                "total_power_mw": 1.0,
            })
        except author.EX.ExtractionError:
            strict_negative = True
        else:
            strict_negative = False
        if not strict_negative:
            raise AssertionError("micro-negative power was accepted")
        canonical_result = author.M.build(author.CONFIG)
        results["G_controls"] = {
            "status": "PASS",
            "micro_negative_power_rejected": True,
            "canonical_production_runs": canonical_result["validated_production_run_count"],
            "canonical_authorities": canonical_result["trusted_hammer_authority_count"],
            "canonical_bundles": canonical_result["table_a_evidence_bundle_count"],
            "canonical_eligible": canonical_result["headline_gate"]["eligible_row_count"],
            "canonical_headline": canonical_result["headline_gate"]["admitted"],
            "canonical_analytical": canonical_result["analytical_diagnostic"]["admitted"],
        }
        print(json.dumps(results, sort_keys=True, indent=2, allow_nan=False))
    finally:
        fixture.cleanup()


if __name__ == "__main__":
    main()
