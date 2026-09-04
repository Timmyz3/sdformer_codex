#!/usr/bin/env python3
"""M698 r10 author tests and reproductions of all M695 P1 attacks."""

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = REPO_ROOT / "hw_autoresearch_nts07"
EXTRACTOR = HW_ROOT / "system_simulator/scripts/extract_m698_native_synopsys_run_provenance_r10.py"
BUILDER = HW_ROOT / "system_simulator/scripts/build_m698_h67_paper_metric_registry_r10.py"
R9_TESTS = HW_ROOT / "system_simulator/tests/test_m691_h67_paper_metric_registry_r9.py"
CONFIG = HW_ROOT / "system_simulator/config/m698_h67_paper_metric_registry_r10_20260828.json"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EX = _load("m698_extractor_test", EXTRACTOR)
M = _load("m698_builder_test", BUILDER)
R9T = _load("m698_r9_fixture", R9_TESTS)


class MandatoryB0Fixture(R9T.R8T.NativeFixture):
    """M695's corrected mandatory-row identity, generated before r9 upgrade."""

    def __init__(self):
        self.result_tmp = tempfile.TemporaryDirectory(
            dir=str(HW_ROOT / "results"), prefix="m698_native_fixture_")
        self.config_tmp = tempfile.TemporaryDirectory(
            dir=str(HW_ROOT / "system_simulator/tests"), prefix="m698_config_fixture_")
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
            "m671_provenance_%s_%s" % (self.row_id, self.config["sha256"][:12]))
        for directory in ("inputs", "inputs/rtl", "libraries", "tools",
                          "scripts", "logs"):
            (self.provenance_root / directory).mkdir(parents=True, exist_ok=True)
        self._make_inputs()
        self._make_run()
        self.manifest_doc["m527_configuration_id"] = "b0_dense96_fixed_t10"


class R10Fixture(object):
    def __init__(self):
        original = R9T.R8T.NativeFixture
        R9T.R8T.NativeFixture = MandatoryB0Fixture
        try:
            self.r9 = R9T.R9Fixture()
        finally:
            R9T.R8T.NativeFixture = original
        self.base = self.r9.base
        self.report_dir = self.base.provenance_root / "r10_reports"
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.report_specs = {}
        self._build_reports()
        self._build_extension()

    def cleanup(self):
        self.r9.cleanup()

    @staticmethod
    def _sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @staticmethod
    def _spec(path, media="text/plain"):
        return {"path": Path(path).relative_to(REPO_ROOT).as_posix(),
                "sha256": R10Fixture._sha(path), "media_type": media}

    def _write_report(self, name, lines):
        path = self.report_dir / (name + ".rpt")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.report_specs[name] = self._spec(path)
        return self.report_specs[name]

    def _build_reports(self):
        loaded = EX.R9._load_manifest(
            self.base.manifest_path.relative_to(REPO_ROOT).as_posix())
        tool_run, proof = loaded[4], loaded[7]

        lines = ["M698_TOOL_IDENTITY_V1"]
        for index, tool in enumerate(EX.TOOL_NAMES):
            digest = tool_run["tool_executables"][tool]["file"]["sha256"]
            build_id = hashlib.sha256(("build:" + tool).encode("utf-8")).hexdigest()[:40]
            version_sha = hashlib.sha256(
                EX.R9.EXPECTED_TOOL_VERSIONS[tool].encode("utf-8")).hexdigest()
            path = "/synthetic/synopsys/%s" % tool
            lines.append("|".join([
                "TOOL", tool, tool, path, path, str(index + 1), str(index + 101),
                build_id, digest, version_sha, digest, "SYNTHETIC_GRAMMAR_ONLY",
            ]))
        for step in EX.STRICT_STEPS:
            entry = proof["execution_steps"][step]
            lines.append("|".join([
                "STEP", step, EX.STEP_TO_LOGICAL_TOOL[step],
                entry["executable"]["path"], entry["executable"]["sha256"],
                entry["argv"][0],
            ]))
        lines.append("M698_TOOL_IDENTITY_END")
        self._write_report("tool_identity", lines)

        dc_sha = tool_run["tool_executables"]["dc_shell"]["file"]["sha256"]
        lines = ["M698_DB_NATIVE_READ_V1"]
        for role in EX.R9.LIBRARY_ROLES:
            item = tool_run["library_dbs"][role]
            fingerprint = EX.R9._map_sha({"role": role, "db": item["file"]["sha256"]})
            lines.append("|".join([
                "DB", role, item["library_name"], item["file"]["sha256"], dc_sha,
                "SYNTHETIC_PARSE_OK", "100", "1", "900", "ns", fingerprint,
            ]))
        lines.append("M698_DB_NATIVE_READ_END")
        self._write_report("db_native_read", lines)

        rtl_sha = tool_run["rtl_sources"]["design_rtl"]["sha256"]
        net_sha = tool_run["netlist"]["sha256"]
        roots = [EX.R9._map_sha({"root": name}) for name in
                 ("mapped_refs", "elaboration", "netlist")]
        lines = ["M698_SCOPE_FORMALITY_V1"]
        lines.append("|".join([
            "TOP", proof["design_name"], rtl_sha, net_sha, "PASS", "100", "0",
            "10", "10", "20", "1", "1", "20", roots[0], roots[1], roots[2],
        ]))
        for op, module, instance in EX.SCOPE_ANCHORS:
            lines.append("|".join([
                "OP", op, module, instance, "1", "1", "2",
                EX.R9._map_sha({"operator": op}), "MAPPED_STDCELL_REFERENCES",
            ]))
        lines.append("M698_SCOPE_FORMALITY_END")
        self._write_report("scope_formality", lines)

        lines = ["M698_PT_SAIF_ANNOTATION_V1",
                 "|".join(["SUMMARY", proof["design_name"],
                           tool_run["activity"]["sha256"], net_sha,
                           "1", "1", "1", "1", "PT_REPORT_ACTIVITY_DERIVED"]),
                 "NET|clk|clk", "PIN|clk|u_clk_pin",
                 "M698_PT_SAIF_ANNOTATION_END"]
        self._write_report("saif_annotation", lines)

        expected = EX._expected_macro_rows(loaded)
        lines = ["M698_NETLIST_MACRO_HIERARCHY_V1"]
        for row in expected:
            lines.append("|".join(["MACRO", row["role"], row["instance"],
                                   row["macro_name"], "LINKED"]))
        lines.append("M698_NETLIST_MACRO_HIERARCHY_END")
        self._write_report("macro_hierarchy", lines)

        expected_macro = sum(row["area_mm2"] for row in expected)
        _, total = EX.R9.R8.parse_dc_area(loaded[3]["dc_area"])
        lines = ["M698_DC_AREA_SPLIT_V1",
                 "AREA|%.15g|%.15g|%.15g|DC_REPORT_HIER_DERIVED" %
                 (total, total - expected_macro, expected_macro),
                 "M698_DC_AREA_SPLIT_END"]
        self._write_report("dc_area_split", lines)

        _, power = EX.R9.R8.parse_ptpx_power(loaded[3]["ptpx_power"])
        each = power["sram_total_power_mw"] / len(expected)
        internal, switching = each * 0.3, each * 0.2
        leakage = each - internal - switching
        lines = ["M698_PTPX_MACRO_POWER_V1"]
        for row in expected:
            lines.append("|".join([
                "MEM_POWER", row["role"], row["instance"], row["macro_name"],
                "%.17g" % internal, "%.17g" % switching, "%.17g" % leakage,
                "%.17g" % each, "PTPX_HIER_DERIVED",
            ]))
        lines.append("M698_PTPX_MACRO_POWER_END")
        self._write_report("ptpx_macro_power", lines)

    def _build_extension(self):
        b = self.base
        extension = {
            "schema": "m698.h67.native_synopsys_trust_extension.r1",
            "status": "STRUCTURAL_EVIDENCE_COMPLETE__NOT_AUTHORITY",
            "evidence_class": "SYNTHETIC_GRAMMAR_ONLY",
            "row_id": b.row_id, "design_name": b.design, "run_id": b.run_id,
            "r9_run_manifest_sha256": self._sha(b.manifest_path),
            "tool_identity_report": self.report_specs["tool_identity"],
            "db_native_read_report": self.report_specs["db_native_read"],
            "scope_and_formality_report": self.report_specs["scope_formality"],
            "pt_saif_annotation_report": self.report_specs["saif_annotation"],
            "netlist_macro_hierarchy_report": self.report_specs["macro_hierarchy"],
            "dc_area_split_report": self.report_specs["dc_area_split"],
            "ptpx_macro_power_report": self.report_specs["ptpx_macro_power"],
            "component_root_sha256": "",
        }
        extension["component_root_sha256"] = EX.R9._map_sha(
            {key: value for key, value in extension.items()
             if key != "component_root_sha256"})
        self.extension = extension
        self.extension_path = b.run_dir / "m698_trust_extension.json"
        self.extension_path.write_text(
            json.dumps(extension, sort_keys=True, separators=(",", ":"),
                       allow_nan=False), encoding="utf-8")
        self.extension_spec = self._spec(self.extension_path, "application/json")
        self.authority_path = b.run_dir / "m698_untrusted_authority.json"
        self.authority_path.write_text(json.dumps({"self_authored": True}),
                                       encoding="utf-8")
        self.authority_spec = self._spec(self.authority_path, "application/json")
        self.candidate = {
            "run_manifest": b.manifest_spec,
            "trust_extension": self.extension_spec,
            "authority": self.authority_spec,
        }

    def rewrite_report(self, name, lines):
        self._write_report(name, lines)
        field = {
            "tool_identity": "tool_identity_report",
            "db_native_read": "db_native_read_report",
            "scope_formality": "scope_and_formality_report",
            "saif_annotation": "pt_saif_annotation_report",
            "macro_hierarchy": "netlist_macro_hierarchy_report",
            "dc_area_split": "dc_area_split_report",
            "ptpx_macro_power": "ptpx_macro_power_report",
        }[name]
        self.extension[field] = self.report_specs[name]
        self.rewrite_extension(self.extension)

    def rewrite_extension(self, extension):
        extension = copy.deepcopy(extension)
        extension["component_root_sha256"] = EX.R9._map_sha(
            {key: value for key, value in extension.items()
             if key != "component_root_sha256"})
        self.extension = extension
        self.extension_path.write_text(
            json.dumps(extension, sort_keys=True, separators=(",", ":"),
                       allow_nan=False), encoding="utf-8")
        self.extension_spec = self._spec(self.extension_path, "application/json")
        self.candidate["trust_extension"] = self.extension_spec

    def config_with_candidate(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        config["production_run_bundles"] = {self.base.row_id: self.candidate}
        path = self.base.config_root / "m698_candidate_config.json"
        path.write_text(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                   allow_nan=False), encoding="utf-8")
        return path


class M698Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = R10Fixture()

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def setUp(self):
        # Every mutation test starts from a fresh fixture so a failed parser
        # cannot leave altered SHA roots for the following attack.
        if self._testMethodName not in ("test_01_canonical_remains_zero",
                                        "test_02_mandatory_b0_synthetic_grammar_reaches_structure"):
            self.local = R10Fixture()
        else:
            self.local = None

    def tearDown(self):
        if self.local is not None:
            self.local.cleanup()

    def test_01_canonical_remains_zero(self):
        result = M.build(CONFIG)
        self.assertEqual(result["validated_production_run_count"], 0)
        self.assertEqual(result["trusted_hammer_authority_count"], 0)
        self.assertEqual(result["table_a_evidence_bundle_count"], 0)
        self.assertFalse(result["headline_gate"]["admitted"])
        self.assertFalse(result["analytical_diagnostic"]["admitted"])

    def test_02_mandatory_b0_synthetic_grammar_reaches_structure(self):
        result = M._validate_candidate_structure(
            self.fixture.candidate, "dense96_fixed_t10", True)
        self.assertTrue(result["structural_evidence_pass"])
        self.assertEqual(result["evidence_class"], "SYNTHETIC_GRAMMAR_ONLY")
        self.assertEqual(result["row_id"], "dense96_fixed_t10")

    def test_03_synthetic_candidate_never_enters_production_map(self):
        with self.assertRaisesRegex(M.RegistryError, "not code-pinned"):
            M.build(self.local.config_with_candidate())

    def test_04_bin_true_native_impersonation_rejects(self):
        extension = copy.deepcopy(self.local.extension)
        extension["evidence_class"] = "NATIVE_SYNOPSYS_EXECUTION"
        self.local.rewrite_extension(extension)
        with self.assertRaisesRegex(EX.ExtractionError, "unapproved native|distinct"):
            EX.extract_from_bundle(
                self.local.base.manifest_path.relative_to(REPO_ROOT).as_posix(),
                self.local.extension_path.relative_to(REPO_ROOT).as_posix(), False)

    def test_05_step_executable_argv0_decoupling_rejects(self):
        path = self.local.report_dir / "tool_identity.rpt"
        lines = path.read_text(encoding="utf-8").splitlines()
        lines = [line.replace("|dc|dc_shell|", "|dc|dc_shell|").replace(
            "|dc_shell|hw_autoresearch_nts07/results/", "|dc_shell|different/")
                 if line.startswith("STEP|dc|") else line for line in lines]
        self.local.rewrite_report("tool_identity", lines)
        with self.assertRaisesRegex(EX.ExtractionError, "step executable"):
            EX.extract_from_bundle(
                self.local.base.manifest_path.relative_to(REPO_ROOT).as_posix(),
                self.local.extension_path.relative_to(REPO_ROOT).as_posix(), True)

    def test_06_elf_bytes_do_not_replace_native_db_read(self):
        path = self.local.report_dir / "db_native_read.rpt"
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if line.startswith("DB|"):
                fields = line.split("|")
                fields[5] = "NATIVE_DC_READ_OK"
                fields[6] = "0"
                lines[index] = "|".join(fields)
                break
        self.local.rewrite_report("db_native_read", lines)
        with self.assertRaisesRegex(EX.ExtractionError, "DB native-read"):
            EX.extract_from_bundle(
                self.local.base.manifest_path.relative_to(REPO_ROOT).as_posix(),
                self.local.extension_path.relative_to(REPO_ROOT).as_posix(), True)

    def test_07_wire_stubs_and_rtl_equal_netlist_reject_native(self):
        loaded = EX.R9._load_manifest(
            self.local.base.manifest_path.relative_to(REPO_ROOT).as_posix())
        with self.assertRaisesRegex(EX.ExtractionError,
                                   "RTL equals|behavioral stub|mapped standard-cell"):
            EX._validate_native_scope_text(loaded[4])

    def test_08_operator_cell_census_must_reconcile(self):
        path = self.local.report_dir / "scope_formality.rpt"
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if line.startswith("OP|patch_embed|"):
                fields = line.split("|")
                fields[4:7] = ["0", "0", "0"]
                lines[index] = "|".join(fields)
                break
        self.local.rewrite_report("scope_formality", lines)
        with self.assertRaisesRegex(EX.ExtractionError, "operator semantic"):
            EX.extract_from_bundle(
                self.local.base.manifest_path.relative_to(REPO_ROOT).as_posix(),
                self.local.extension_path.relative_to(REPO_ROOT).as_posix(), True)

    def test_09_one_tc_cannot_claim_99_annotated_nets(self):
        path = self.local.report_dir / "saif_annotation.rpt"
        lines = path.read_text(encoding="utf-8").splitlines()
        fields = lines[1].split("|")
        fields[4:8] = ["99", "100", "198", "200"]
        lines[1] = "|".join(fields)
        self.local.rewrite_report("saif_annotation", lines)
        with self.assertRaisesRegex(EX.ExtractionError, "coverage reconciliation"):
            EX.extract_from_bundle(
                self.local.base.manifest_path.relative_to(REPO_ROOT).as_posix(),
                self.local.extension_path.relative_to(REPO_ROOT).as_posix(), True)

    def test_10_missing_ptpx_macro_instance_rejects(self):
        path = self.local.report_dir / "ptpx_macro_power.rpt"
        lines = path.read_text(encoding="utf-8").splitlines()
        del lines[-2]
        self.local.rewrite_report("ptpx_macro_power", lines)
        with self.assertRaisesRegex(EX.ExtractionError, "row count"):
            EX.extract_from_bundle(
                self.local.base.manifest_path.relative_to(REPO_ROOT).as_posix(),
                self.local.extension_path.relative_to(REPO_ROOT).as_posix(), True)

    def test_11_dc_logic_macro_total_equation_rejects(self):
        path = self.local.report_dir / "dc_area_split.rpt"
        lines = path.read_text(encoding="utf-8").splitlines()
        fields = lines[1].split("|")
        fields[2] = str(float(fields[2]) + 0.01)
        lines[1] = "|".join(fields)
        self.local.rewrite_report("dc_area_split", lines)
        with self.assertRaisesRegex(EX.ExtractionError, "DC.*equation"):
            EX.extract_from_bundle(
                self.local.base.manifest_path.relative_to(REPO_ROOT).as_posix(),
                self.local.extension_path.relative_to(REPO_ROOT).as_posix(), True)

    def test_12_self_authored_authority_is_not_trusted(self):
        candidate = M._validate_candidate_structure(
            self.local.candidate, "dense96_fixed_t10", True)
        with tempfile.TemporaryDirectory(
                dir=str(HW_ROOT / "reviews"), prefix="m698_untrusted_authority_") as tmp:
            path = Path(tmp) / "review.json"
            path.write_text(json.dumps({"self_authored": True}), encoding="utf-8")
            spec = self.local._spec(path, "application/json")
            with self.assertRaisesRegex(M.RegistryError, "not code-pinned"):
                M._validate_pinned_authority(spec, candidate)


if __name__ == "__main__":
    unittest.main()
