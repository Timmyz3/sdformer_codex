#!/usr/bin/env python3

"""Author tests for the M671 production Synopsys provenance gate.

The generated files deliberately live under the repository because the
production extractor rejects paths outside the repository.  They are grammar
fixtures only and are removed after every test class; they can never become a
trusted Table-A authority.
"""

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = REPO_ROOT / "hw_autoresearch_nts07"
EXTRACTOR = HW_ROOT / "system_simulator/scripts/extract_m671_native_synopsys_run_provenance.py"
BUILDER = HW_ROOT / "system_simulator/scripts/build_m671_h67_paper_metric_registry_r8.py"
CONFIG = HW_ROOT / "system_simulator/config/m671_h67_paper_metric_registry_r8_20260828.json"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EX = _load("m671_native_provenance", EXTRACTOR)
M = _load("m671_registry_r8", BUILDER)


class NativeFixture(object):
    def __init__(self):
        self.result_tmp = tempfile.TemporaryDirectory(
            dir=str(HW_ROOT / "results"), prefix="m671_native_fixture_")
        self.config_tmp = tempfile.TemporaryDirectory(
            dir=str(HW_ROOT / "system_simulator/tests"),
            prefix="m671_config_fixture_")
        self.result_root = Path(self.result_tmp.name)
        self.config_root = Path(self.config_tmp.name)
        self.row_id = "dense96_fixed_t10"
        self.design = "h67_table_a_dense96_fixed_t10"
        self.config = self._write_json(
            self.config_root / "configuration.json",
            {
                "configuration_id": "dense96_fixed_t10",
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

    def cleanup(self):
        self.result_tmp.cleanup()
        self.config_tmp.cleanup()

    @staticmethod
    def _sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @staticmethod
    def _spec(path, media="text/plain"):
        return {"path": Path(path).relative_to(REPO_ROOT).as_posix(),
                "sha256": NativeFixture._sha(path), "media_type": media}

    def _write(self, path, text, media="text/plain"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return self._spec(path, media)

    def _write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                   allow_nan=False), encoding="utf-8")
        return self._spec(path, "application/json")

    def _make_inputs(self):
        inputs = self.provenance_root / "inputs"
        self.rtl_sources = {
            "design_rtl": self._write(inputs / "rtl/design.sv",
                                      "module %s; endmodule\n" % self.design,
                                      "text/x-systemverilog"),
            "testbench": self._write(inputs / "rtl/tb.sv",
                                     "module tb; endmodule\n",
                                     "text/x-systemverilog"),
            "assertions": self._write(inputs / "rtl/assertions.sv",
                                      "module assertions; endmodule\n",
                                      "text/x-systemverilog"),
        }
        self.netlist = self._write(inputs / "mapped.v",
                                   "module %s; endmodule\n" % self.design)
        self.sdc = self._write(inputs / "constraints.sdc",
                               "create_clock -period 3.0 [get_ports clk]\n")
        self.activity = self._write(inputs / "activity.saif",
                                    "(SAIFILE (SAIFVERSION \"2.0\"))\n")
        self.libraries = {}
        for role in EX.LIBRARY_ROLES:
            suffix = "setup" if role.endswith("setup") else (
                "hold" if role.endswith("hold") else "power")
            corner = EX.TARGET_CORNERS[
                "pt_setup" if suffix == "setup" else
                "pt_hold" if suffix == "hold" else "ptpx_power"]
            file_spec = self._write(
                self.provenance_root / "libraries" / (role + ".db"),
                "M671_BINARY_DB_FIXTURE_%s\n" % role,
                "application/octet-stream")
            self.libraries[role] = {
                "role": role, "library_name": EX.EXPECTED_LIBRARY_NAMES[role],
                "corner": corner, "file": file_spec,
            }
        self.executables = {}
        for name in EX.TOOL_NAMES:
            file_spec = self._write(
                self.provenance_root / "tools" / name,
                "M671_BINARY_TOOL_FIXTURE_%s\n" % name,
                "application/octet-stream")
            self.executables[name] = {
                "file": file_spec, "version": EX.EXPECTED_TOOL_VERSIONS[name]}

    def _header(self, report, version):
        return ("****************************************\n"
                "Report : %s\nDesign : %s\nVersion: %s\n"
                "Date   : Fri Aug 28 00:00:00 2026\n"
                "****************************************\n\n") % (report, self.design, version)

    def _environment(self, analysis, corner, roles, tool_version):
        text = self._header("environment", tool_version)
        text += ("Analysis View : %s\nOperating Condition : %s\n"
                 "Process : %s\nVoltage : %.6f V\nTemperature : %.6f C\n" %
                 (analysis, corner["operating_condition"], corner["process"],
                  corner["voltage_v"], corner["temperature_c"]))
        for role in roles:
            item = self.libraries[role]
            text += "Library : %s %s (File: %s)\n" % (
                role, item["library_name"], item["file"]["path"])
        return text

    @staticmethod
    def _macro_text(frozen):
        name = frozen["macro_name"]
        library = name.lower()
        memory_type = "Two Port SRAM" if frozen["port_type"] == "1R1W" else "Single Port SRAM"
        addr_bits = frozen["depth_words"].bit_length() - 1
        return """#### Software : TSMC MEMORY COMPILER %s */
#### Technology : TSMC 28nm HPC+ */
#### Memory Type : %s */
#### Library Name : %s (user specify : %s) */
#### Library Version : 1.0 */
#### Generated Time : 2026/08/28, 00:00:00 */
   2.2 SRAM timing:(Slow, 0.9000, 125.0000 deg.)
  A[%d:0] 1.0
  D[%d:0] 1.0
  | 100.0 | 200.0 | 20000.0 |
  Leakage Current 10.0 (uA)
  Read 11.0 (uA/MHz)
  Write 12.0 (uA/MHz)
""" % (EX.EXPECTED_TOOL_VERSIONS["memory_compiler"], memory_type,
       library, name, addr_bits - 1, frozen["width_bits"] - 1)

    def _report_texts(self):
        dc = self._header("area", EX.EXPECTED_TOOL_VERSIONS["dc_shell"])
        dc += "    %s (File: setup.db)\nTotal cell area: 600000.0\n" % (
            EX.EXPECTED_LIBRARY_NAMES["logic_setup"])
        setup = self._header("timing", EX.EXPECTED_TOOL_VERSIONS["pt_shell"])
        setup += "  -delay_type max\n  Path Type: max\n  slack (MET) 0.100000\n"
        hold = self._header("timing", EX.EXPECTED_TOOL_VERSIONS["pt_shell"])
        hold += "  -delay_type min\n  Path Type: min\n  slack (MET) 0.010000\n"
        power = self._header("Averaged Power", EX.EXPECTED_TOOL_VERSIONS["pt_shell"])
        power += """  -unit mW
memory 0.040000 0.010000 0.010000 0.060000 (18.18%%)
  Cell Internal Power = 0.240000 (72.73%%)
  Net Switching Power = 0.060000 (18.18%%)
  Cell Leakage Power = 0.030000 (9.09%%)
Total Power = 0.330000 (100.00%%)
"""
        vcs = ("Chronologic VCS simulator copyright 1991-2023\n"
               "Compiler version V-2023.12-SP1_Full64; Runtime version "
               "V-2023.12-SP1_Full64; Aug 28 00:00 2026\n"
               "M671_TABLE_A_VCS_PASS\n")
        formality = ("Formality (R)\n"
                     " Version V-2023.12-SP3 for linux64 - Apr 13, 2024\n"
                     "Verification SUCCEEDED\nThank you for using Formality (R)!\n")
        reports = {
            "vcs_simulation": vcs,
            "dc_area": dc,
            "dc_environment": self._environment(
                "dc", EX.TARGET_CORNERS["dc_area"],
                ("logic_setup", "sram_setup"), EX.EXPECTED_TOOL_VERSIONS["dc_shell"]),
            "formality_verification": formality,
            "pt_setup": setup,
            "pt_setup_environment": self._environment(
                "setup", EX.TARGET_CORNERS["pt_setup"],
                ("logic_setup", "sram_setup"), EX.EXPECTED_TOOL_VERSIONS["pt_shell"]),
            "pt_hold": hold,
            "pt_hold_environment": self._environment(
                "hold", EX.TARGET_CORNERS["pt_hold"],
                ("logic_hold", "sram_hold"), EX.EXPECTED_TOOL_VERSIONS["pt_shell"]),
            "ptpx_power": power,
            "ptpx_environment": self._environment(
                "power", EX.TARGET_CORNERS["ptpx_power"],
                ("logic_power", "sram_power"), EX.EXPECTED_TOOL_VERSIONS["pt_shell"]),
        }
        for frozen in EX.EXPECTED_MEMORY_MACROS:
            reports[frozen["report_id"]] = self._macro_text(frozen)
        return reports

    def _make_run(self):
        texts = self._report_texts()
        report_hashes = {name: hashlib.sha256(text.encode("utf-8")).hexdigest()
                         for name, text in texts.items()}
        self.run_id = "m671_%s_%s_%s" % (
            self.row_id, self.config["sha256"][:12],
            EX._map_sha({name: report_hashes[name] for name in sorted(report_hashes)})[:12])
        self.run_dir = self.result_root / self.run_id
        reports_dir = self.run_dir / "reports"
        reports_dir.mkdir(parents=True)
        self.reports = {}
        for name, text in texts.items():
            self.reports[name] = self._write(reports_dir / (name + ".rpt"), text)
        macros = []
        for frozen in EX.EXPECTED_MEMORY_MACROS:
            row = dict(frozen)
            row["library_name"] = row["macro_name"].lower()
            row["macro_rounded_bytes_per_instance"] = (
                row["depth_words"] * row["width_bits"] + 7) // 8
            row["macro_rounded_total_bytes"] = (
                row["macro_rounded_bytes_per_instance"] * row["instance_count"])
            macros.append(row)
        self.memory_inventory = {
            "target_onchip_sram_bytes_total": 245760,
            "macro_rounded_total_bytes": sum(row["macro_rounded_total_bytes"] for row in macros),
            "macros": macros,
        }
        receipt = {
            "schema": "m671.h67.native_tool_run_receipt.r1",
            "status": "PASS_EXIT_ZERO_ROOTED", "row_id": self.row_id,
            "configuration_manifest_sha256": self.config["sha256"],
            "design_name": self.design, "run_id": self.run_id,
            "generation_argv": {}, "command_scripts": {},
            "rtl_sources": self.rtl_sources, "netlist": self.netlist,
            "sdc": self.sdc, "library_dbs": self.libraries,
            "activity": self.activity, "tool_logs": {},
            "tool_executables": self.executables,
            "tool_versions": EX.EXPECTED_TOOL_VERSIONS,
            "exit_status": {step: 0 for step in EX.STEPS},
            "output_reports": self.reports,
            "memory_inventory_sha256": EX._map_sha(self.memory_inventory),
            "component_root_sha256": "",
        }
        for step in EX.STEPS:
            path = self.provenance_root / "scripts" / (step + ".tcl")
            receipt["command_scripts"][step] = {
                "path": path.relative_to(REPO_ROOT).as_posix(),
                "sha256": "0" * 64, "media_type": "text/plain"}
        receipt["generation_argv"] = EX._expected_generation_argv(receipt)
        for step in EX.STEPS:
            outputs = EX._step_outputs(step, receipt)
            rows = []
            for name, digest in outputs.items():
                spec = (self.netlist if name == "mapped_netlist" else
                        self.activity if name == "activity" else self.reports[name])
                rows.append("OUTPUT %s %s %s" % (name, spec["path"], digest))
            body = ["M671_COMMAND_ROOT_BEGIN", "STEP " + step,
                    "DESIGN " + self.design,
                    "INPUT_ROOT_SHA256 " + EX._map_sha(EX._step_input_roots(step, receipt))]
            body.extend(rows)
            body.append("M671_COMMAND_ROOT_END")
            receipt["command_scripts"][step] = self._write(
                self.provenance_root / "scripts" / (step + ".tcl"),
                "\n".join(body) + "\n")
        receipt["generation_argv"] = EX._expected_generation_argv(receipt)
        for step in EX.STEPS:
            tool = ("vcs" if step == "vcs" else "dc_shell" if step == "dc" else
                    "fm_shell" if step == "formality" else
                    "memory_compiler" if step == "memory_compiler" else "pt_shell")
            rows = ["M671_PROVENANCE_BEGIN", "STEP " + step, "TOOL " + tool,
                    "TOOL_EXECUTABLE_SHA256 " + self.executables[tool]["file"]["sha256"],
                    "TOOL_VERSION " + EX.EXPECTED_TOOL_VERSIONS[tool],
                    "ARGV_SHA256 " + EX._map_sha(receipt["generation_argv"][step]),
                    "COMMAND_SCRIPT_SHA256 " + receipt["command_scripts"][step]["sha256"],
                    "INPUT_ROOT_SHA256 " + EX._map_sha(EX._step_input_roots(step, receipt)),
                    "EXIT_STATUS 0"]
            rows.extend("OUTPUT %s %s" % item for item in
                        EX._step_outputs(step, receipt).items())
            rows.append("M671_PROVENANCE_END")
            receipt["tool_logs"][step] = self._write(
                self.provenance_root / "logs" / (step + ".log"),
                "\n".join(rows) + "\n")
        components = {"netlist": self.netlist["sha256"], "sdc": self.sdc["sha256"],
                      "activity": self.activity["sha256"],
                      "memory_inventory": EX._map_sha(self.memory_inventory)}
        components.update({"rtl:" + role: self.rtl_sources[role]["sha256"]
                           for role in EX.RTL_SOURCE_ROLES})
        components.update({"report:" + name: self.reports[name]["sha256"]
                           for name in sorted(EX.REPORT_FIELDS)})
        components.update({"argv:" + step: EX._map_sha(receipt["generation_argv"][step])
                           for step in EX.STEPS})
        components.update({"library:" + role: self.libraries[role]["file"]["sha256"]
                           for role in EX.LIBRARY_ROLES})
        components.update({"executable:" + tool: self.executables[tool]["file"]["sha256"]
                           for tool in EX.TOOL_NAMES})
        components.update({"script:" + step: receipt["command_scripts"][step]["sha256"]
                           for step in EX.STEPS})
        components.update({"log:" + step: receipt["tool_logs"][step]["sha256"]
                           for step in EX.STEPS})
        receipt["component_root_sha256"] = EX._map_sha(components)
        self.tool_receipt = self._write_json(self.run_dir / "tool_run_receipt.json", receipt)
        self.manifest_doc = {
            "schema": "m671.h67.native_synopsys_run_manifest.r2",
            "status": "FROZEN_ROOTED_NATIVE_TOOL_RUN", "row_id": self.row_id,
            "configuration_manifest": self.config,
            "configuration_manifest_sha256": self.config["sha256"],
            "m527_configuration_id": "dense96_fixed_t10",
            "operator_scope_sha256": "1" * 64,
            "design_name": self.design, "run_id": self.run_id,
            "raw_reports": self.reports, "target_corners": EX.TARGET_CORNERS,
            "library_dbs": self.libraries, "memory_inventory": self.memory_inventory,
            "tool_run_receipt": self.tool_receipt,
        }
        self.manifest_path = self.run_dir / "native_run_manifest.json"
        self.manifest_spec = self._write_json(self.manifest_path, self.manifest_doc)

    def rewrite_manifest(self):
        self.manifest_spec = self._write_json(self.manifest_path, self.manifest_doc)

    def load_tool_receipt(self):
        return json.loads((REPO_ROOT / self.tool_receipt["path"]).read_text(encoding="utf-8"))

    def rewrite_tool_receipt(self, receipt):
        self.tool_receipt = self._write_json(REPO_ROOT / self.tool_receipt["path"], receipt)
        self.manifest_doc["tool_run_receipt"] = self.tool_receipt
        self.rewrite_manifest()


class M671RegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = NativeFixture()

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def test_01_canonical_registry_remains_zero(self):
        result = M.build(CONFIG)
        self.assertEqual(result["trusted_hammer_authority_count"], 0)
        self.assertEqual(result["table_a_evidence_bundle_count"], 0)
        self.assertFalse(result["headline_gate"]["admitted"])

    def test_02_complete_native_chain_extracts(self):
        result = EX.extract_from_manifest(self.fixture.manifest_path.relative_to(REPO_ROOT).as_posix())
        self.assertEqual(result["identities"]["vcs_simulation"]["status"], "PASS")
        self.assertEqual(result["identities"]["formality_verification"]["status"],
                         "SUCCEEDED")
        self.assertEqual(result["memory_inventory"]["macro_rounded_total_bytes"], 245760)
        self.assertGreater(result["values"]["logic_area_mm2"] +
                           result["values"]["sram_macro_area_mm2"], 0.0)

    def test_03_vcs_and_formality_fail_closed(self):
        with self.assertRaises(EX.ExtractionError):
            EX.parse_vcs_simulation(self.fixture.run_dir / "reports/formality_verification.rpt")
        with self.assertRaises(EX.ExtractionError):
            EX.parse_formality_verification(self.fixture.run_dir / "reports/vcs_simulation.rpt")
        bad = self.fixture.run_dir / "reports/vcs_bad_direct.rpt"
        bad.write_text((REPO_ROOT / self.fixture.reports["vcs_simulation"]["path"]).read_text(
            encoding="utf-8") + "UVM_FATAL\n", encoding="utf-8")
        self.addCleanup(lambda: bad.unlink() if bad.exists() else None)
        with self.assertRaisesRegex(EX.ExtractionError, "failure signature"):
            EX.parse_vcs_simulation(bad)

    def test_04_parent_macro_is_exact_1r1w_and_total_is_exact(self):
        parent = EX.EXPECTED_MEMORY_MACROS[2]
        self.assertEqual((parent["port_type"], parent["port_count"]), ("1R1W", 2))
        config = json.loads((REPO_ROOT / self.fixture.config["path"]).read_text(encoding="utf-8"))
        identities = {}
        for frozen in EX.EXPECTED_MEMORY_MACROS:
            identities[frozen["report_id"]], _ = EX.parse_sram_macro(
                REPO_ROOT / self.fixture.reports[frozen["report_id"]]["path"])
        projected = EX._validate_memory_inventory(self.fixture.manifest_doc, config, identities)
        self.assertEqual(projected["macro_rounded_total_bytes"], 245760)
        wrong = copy.deepcopy(self.fixture.manifest_doc)
        wrong["memory_inventory"]["macros"][2]["port_type"] = "1RW"
        with self.assertRaisesRegex(EX.ExtractionError, "frozen identity"):
            EX._validate_memory_inventory(wrong, config, identities)

    def test_05_all_db_saif_netlist_and_tool_bytes_are_sha_bound(self):
        db_path = REPO_ROOT / self.fixture.libraries["logic_setup"]["file"]["path"]
        original = db_path.read_bytes()
        try:
            db_path.write_bytes(original + b"drift")
            with self.assertRaisesRegex(EX.ExtractionError, "SHA mismatch"):
                EX.extract_from_manifest(self.fixture.manifest_path.relative_to(REPO_ROOT).as_posix())
        finally:
            db_path.write_bytes(original)

    def test_06_generation_argv_and_exit_status_are_exact(self):
        original = self.fixture.load_tool_receipt()
        receipt = copy.deepcopy(original)
        receipt["generation_argv"]["dc"].append("-unrooted")
        self.fixture.rewrite_tool_receipt(receipt)
        try:
            with self.assertRaisesRegex(EX.ExtractionError, "generation argv"):
                EX.extract_from_manifest(self.fixture.manifest_path.relative_to(REPO_ROOT).as_posix())
        finally:
            self.fixture.rewrite_tool_receipt(original)
        receipt = copy.deepcopy(original)
        receipt["exit_status"]["formality"] = 1
        self.fixture.rewrite_tool_receipt(receipt)
        try:
            with self.assertRaisesRegex(EX.ExtractionError, "exit status"):
                EX.extract_from_manifest(self.fixture.manifest_path.relative_to(REPO_ROOT).as_posix())
        finally:
            self.fixture.rewrite_tool_receipt(original)

    def test_07_noncanonical_and_symlink_paths_reject(self):
        with self.assertRaisesRegex(EX.ExtractionError, "forbidden component"):
            EX._secure_repo_file("hw_autoresearch_nts07/results/../results/nope",
                                 "attack")
        target = self.fixture.run_dir / "reports/vcs_simulation.rpt"
        link = self.fixture.run_dir / "reports/vcs_link.rpt"
        link.symlink_to(target.name)
        self.addCleanup(lambda: link.unlink() if link.exists() or link.is_symlink() else None)
        with self.assertRaisesRegex(EX.ExtractionError, "symlink"):
            EX._secure_repo_file(link.relative_to(REPO_ROOT).as_posix(), "attack")

    def test_08_zero_macro_and_negative_physical_values_reject(self):
        zero_area = self.fixture.run_dir / "reports/dc_zero_direct.rpt"
        text = (REPO_ROOT / self.fixture.reports["dc_area"]["path"]).read_text(
            encoding="utf-8").replace("600000.0", "0.0")
        zero_area.write_text(text, encoding="utf-8")
        self.addCleanup(lambda: zero_area.unlink() if zero_area.exists() else None)
        with self.assertRaisesRegex(EX.ExtractionError, "not positive"):
            EX.parse_dc_area(zero_area)
        negative = self.fixture.run_dir / "reports/power_negative_direct.rpt"
        text = (REPO_ROOT / self.fixture.reports["ptpx_power"]["path"]).read_text(
            encoding="utf-8").replace("0.010000 0.010000 0.060000",
                                      "0.010000 -0.010000 0.040000")
        negative.write_text(text, encoding="utf-8")
        self.addCleanup(lambda: negative.unlink() if negative.exists() else None)
        with self.assertRaisesRegex(EX.ExtractionError, "negative"):
            EX.parse_ptpx_power(negative)


if __name__ == "__main__":
    unittest.main()
