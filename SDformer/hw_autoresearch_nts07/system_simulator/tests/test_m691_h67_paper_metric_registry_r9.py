#!/usr/bin/env python3
"""Author tests and adversarial fixtures for M691 registry r9."""

import copy
import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = REPO_ROOT / "hw_autoresearch_nts07"
EXTRACTOR = HW_ROOT / "system_simulator/scripts/extract_m691_native_synopsys_run_provenance.py"
BUILDER = HW_ROOT / "system_simulator/scripts/build_m691_h67_paper_metric_registry_r9.py"
R8_TESTS = HW_ROOT / "system_simulator/tests/test_m671_h67_paper_metric_registry_r8.py"
CONFIG = HW_ROOT / "system_simulator/config/m691_h67_paper_metric_registry_r9_20260828.json"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EX = _load("m691_extractor", EXTRACTOR)
M = _load("m691_builder", BUILDER)
R8T = _load("m691_r8_fixture", R8_TESTS)


def _scope_design(top):
    modules = []
    declared = set()
    for _, module, _ in EX.SCOPE_ANCHORS:
        modules.append("module %s; wire alive; endmodule" % module)
        declared.add(module)
    for frozen in EX.EXPECTED_MEMORY_MACROS:
        if frozen["macro_name"] not in declared:
            modules.append("module %s; wire alive; endmodule" % frozen["macro_name"])
            declared.add(frozen["macro_name"])
    body = ["module %s;" % top, "wire full_scope_alive;"]
    for _, module, instance in EX.SCOPE_ANCHORS:
        body.append("%s %s();" % (module, instance))
    for index in range(8):
        body.append("%s u_weight_sram_%d();" %
                    (EX.EXPECTED_MEMORY_MACROS[0]["macro_name"], index))
    for index in range(8):
        body.append("%s u_state_sram_%d();" %
                    (EX.EXPECTED_MEMORY_MACROS[1]["macro_name"], index))
    body.append("%s u_parent_scratch();" % EX.EXPECTED_MEMORY_MACROS[2]["macro_name"])
    body.append("endmodule")
    return "\n".join(modules + body) + "\n"


def _saif(top):
    return """(SAIFILE
 (SAIFVERSION \"2.0\")
 (TIMESCALE 1 ns)
 (DURATION 1000)
 (INSTANCE %s
  (NET clk (T0 500) (T1 500) (TX 0) (TC 40))))
""" % top


class R9Fixture(object):
    def __init__(self):
        self.base = R8T.NativeFixture()
        self._upgrade()

    def cleanup(self):
        self.base.cleanup()

    @staticmethod
    def _sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @staticmethod
    def _spec(path, media="text/plain"):
        return {"path": Path(path).relative_to(REPO_ROOT).as_posix(),
                "sha256": R9Fixture._sha(path), "media_type": media}

    @staticmethod
    def _write(path, text, media="text/plain"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text, encoding="utf-8")
        return R9Fixture._spec(path, media)

    @staticmethod
    def _write_json(path, value):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                         allow_nan=False), encoding="utf-8")
        return R9Fixture._spec(path, "application/json")

    def _upgrade(self):
        b = self.base
        report_hashes = {name: spec["sha256"] for name, spec in b.reports.items()}
        new_run_id = "m691_%s_%s_%s" % (
            b.row_id, b.config["sha256"][:12],
            EX._map_sha({name: report_hashes[name] for name in sorted(report_hashes)})[:12])
        old_dir = b.run_dir
        new_dir = old_dir.parent / new_run_id
        old_dir.rename(new_dir)
        b.run_dir, b.run_id = new_dir, new_run_id
        b.manifest_path = new_dir / "native_run_manifest.json"
        b.reports = {name: self._spec(new_dir / "reports" / (name + ".rpt"))
                     for name in EX.REPORT_FIELDS}

        receipt = json.loads((new_dir / "tool_run_receipt.json").read_text(encoding="utf-8"))
        design = _scope_design(b.design)
        rtl_dir = b.provenance_root / "inputs/rtl"
        b.rtl_sources = {
            "design_rtl": self._write(rtl_dir / "design.sv", design, "text/x-systemverilog"),
            "testbench": self._write(rtl_dir / "tb.sv",
                                     "module tb; %s dut(); endmodule\n" % b.design,
                                     "text/x-systemverilog"),
            "assertions": self._write(rtl_dir / "assertions.sv",
                                      "module assertions; property p; 1 |-> 1; endproperty endmodule\n",
                                      "text/x-systemverilog"),
        }
        b.netlist = self._write(b.provenance_root / "inputs/mapped.v", design)
        b.sdc = self._write(b.provenance_root / "inputs/constraints.sdc",
                           "create_clock -period 3.0 [get_ports clk]\n")
        b.activity = self._write(b.provenance_root / "inputs/activity.saif", _saif(b.design))

        elf = Path("/bin/true").read_bytes()
        b.executables = {}
        for name in EX.TOOL_NAMES:
            path = b.provenance_root / "tools" / name
            path.write_bytes(elf)
            path.chmod(0o755)
            b.executables[name] = {
                "file": self._spec(path, "application/octet-stream"),
                "version": EX.EXPECTED_TOOL_VERSIONS[name],
            }
        b.libraries = {}
        for role in EX.LIBRARY_ROLES:
            suffix = "setup" if role.endswith("setup") else (
                "hold" if role.endswith("hold") else "power")
            corner = EX.TARGET_CORNERS[
                "pt_setup" if suffix == "setup" else
                "pt_hold" if suffix == "hold" else "ptpx_power"]
            path = b.provenance_root / "libraries" / (role + ".db")
            path.write_bytes(elf + bytes(bytearray(range(256))) * 20)
            b.libraries[role] = {
                "role": role, "library_name": EX.R8.EXPECTED_LIBRARY_NAMES[role],
                "corner": corner, "file": self._spec(path, "application/octet-stream"),
            }

        receipt.update({
            "run_id": new_run_id, "rtl_sources": b.rtl_sources, "netlist": b.netlist,
            "sdc": b.sdc, "activity": b.activity, "library_dbs": b.libraries,
            "tool_executables": b.executables, "output_reports": b.reports,
            "tool_versions": EX.EXPECTED_TOOL_VERSIONS,
            "exit_status": {step: 0 for step in EX.R8.STEPS},
        })
        self._rebuild_legacy(receipt)
        receipt = self.receipt

        simv_path = b.provenance_root / "tools/simv"
        simv_path.write_bytes(elf)
        simv_path.chmod(0o755)
        simv = self._spec(simv_path, "application/octet-stream")
        proof = {
            "schema": "m691.h67.production_execution_proof.r1",
            "status": "PASS_WRAPPER_ROOTED_NATIVE_EXECUTION", "row_id": b.row_id,
            "configuration_manifest_sha256": b.config["sha256"],
            "design_name": b.design, "run_id": new_run_id,
            "legacy_tool_run_receipt_sha256": b.tool_receipt["sha256"],
            "simv": simv,
            "scope_anchors": [{"operator": op, "rtl_module": module, "instance": instance}
                              for op, module, instance in EX.SCOPE_ANCHORS],
            "execution_steps": {}, "tool_version_runs": {},
            "saif_annotation": {
                "activity_sha256": b.activity["sha256"], "top_instance": b.design,
                "duration": 1000.0, "timescale": "1 ns", "annotated_nets": 99,
                "total_nets": 100, "annotated_pins": 198, "total_pins": 200,
            },
            "macro_census": {}, "ptpx_memory_census": {},
            "component_root_sha256": "",
        }
        macro_diagnostics = {}
        for frozen in EX.EXPECTED_MEMORY_MACROS:
            _, diagnostic = EX.R8.parse_sram_macro(
                REPO_ROOT / b.reports[frozen["report_id"]]["path"])
            macro_diagnostics[frozen["report_id"]] = diagnostic
        instances = EX._expected_macro_instances(macro_diagnostics)
        macro_area = sum(row["area_mm2"] for row in instances)
        _, dc_total = EX.R8.parse_dc_area(REPO_ROOT / b.reports["dc_area"]["path"])
        proof["macro_census"] = {
            "area_mode": "DC_TOTAL_INCLUDES_MACROS", "dc_total_cell_area_mm2": dc_total,
            "logic_cell_area_mm2": dc_total - macro_area,
            "macro_cell_area_mm2": macro_area, "instances": instances,
        }
        proof["ptpx_memory_census"] = {
            "instances": [{"role": row["role"], "instance": row["instance"],
                           "macro_name": row["macro_name"]} for row in instances],
            "sram_total_power_mw": 0.06,
        }
        strict_scripts = b.provenance_root / "r9_scripts"
        strict_logs = b.provenance_root / "r9_logs"
        strict_scripts.mkdir(exist_ok=True)
        strict_logs.mkdir(exist_ok=True)
        scripts = {
            "dc": "read_verilog %s\nread_sdc %s\ncompile_ultra\n" %
                  (b.rtl_sources["design_rtl"]["path"], b.sdc["path"]),
            "formality": "read_verilog %s\nread_verilog %s\nset_top %s\nverify\n" %
                         (b.rtl_sources["design_rtl"]["path"], b.netlist["path"], b.design),
            "pt_setup": "read_verilog %s\nread_sdc %s\nreport_timing -delay_type max\n" %
                        (b.netlist["path"], b.sdc["path"]),
            "pt_hold": "read_verilog %s\nread_sdc %s\nreport_timing -delay_type min\n" %
                       (b.netlist["path"], b.sdc["path"]),
            "ptpx": "read_verilog %s\nread_sdc %s\nread_saif %s\nreport_power\n" %
                    (b.netlist["path"], b.sdc["path"], b.activity["path"]),
            "memory_compiler": "\n".join(
                "compile_memory %s" % row["macro_name"] for row in EX.EXPECTED_MEMORY_MACROS) + "\n",
        }
        script_specs = {step: self._write(strict_scripts / (step + ".tcl"), text)
                        for step, text in scripts.items()}
        for index, step in enumerate(EX.STRICT_STEPS):
            if step == "vcs_compile":
                executable, script = b.executables["vcs"]["file"], None
            elif step == "vcs_run":
                executable, script = simv, None
            else:
                tool = ("dc_shell" if step == "dc" else "fm_shell" if step == "formality" else
                        "memory_compiler" if step == "memory_compiler" else "pt_shell")
                executable, script = b.executables[tool]["file"], script_specs[step]
            proof["execution_steps"][step] = {
                "executable": executable, "argv": [], "script": script, "log": None,
                "exit_status": 0, "start_time_ns": 1000 + 10 * index,
                "end_time_ns": 1001 + 10 * index, "input_sha256": {},
                "output_sha256": {},
            }
        for step in EX.STRICT_STEPS:
            entry = proof["execution_steps"][step]
            entry["argv"] = EX._expected_strict_argv(step, proof, receipt)
            entry["input_sha256"], entry["output_sha256"] = EX._strict_input_output(
                step, proof, receipt)
            rows = ["M691_EXECUTION_BEGIN", "STEP " + step,
                    "EXECUTABLE_SHA256 " + entry["executable"]["sha256"],
                    "ARGV_SHA256 " + EX._map_sha(entry["argv"]),
                    "SCRIPT_SHA256 " + ("NONE" if entry["script"] is None else
                                          entry["script"]["sha256"]),
                    "INPUT_ROOT_SHA256 " + EX._map_sha(entry["input_sha256"]),
                    "START_TIME_NS " + str(entry["start_time_ns"]),
                    "END_TIME_NS " + str(entry["end_time_ns"]), "EXIT_STATUS 0"]
            rows.extend("OUTPUT %s %s" % item for item in entry["output_sha256"].items())
            rows.append("M691_EXECUTION_END")
            entry["log"] = self._write(strict_logs / (step + ".log"), "\n".join(rows) + "\n")
        for tool in EX.TOOL_NAMES:
            executable = b.executables[tool]["file"]
            flag = "-ID" if tool == "vcs" else "-version"
            log = self._write(strict_logs / ("version_" + tool + ".log"),
                              "M691_VERSION %s %s %s\n" %
                              (tool, executable["sha256"], EX.EXPECTED_TOOL_VERSIONS[tool]))
            proof["tool_version_runs"][tool] = {
                "executable_sha256": executable["sha256"],
                "argv": [executable["path"], flag], "log": log, "exit_status": 0,
                "reported_version": EX.EXPECTED_TOOL_VERSIONS[tool],
            }
        proof["component_root_sha256"] = EX._map_sha(
            {key: value for key, value in proof.items() if key != "component_root_sha256"})
        self.proof_path = new_dir / "production_proof.json"
        self.proof_spec = self._write_json(self.proof_path, proof)
        self.proof = proof
        self.proof_original = copy.deepcopy(proof)

        manifest = dict(b.manifest_doc)
        manifest.update({
            "schema": "m691.h67.native_synopsys_run_manifest.r3",
            "status": "FROZEN_ROOTED_NATIVE_TOOL_RUN_R9", "run_id": new_run_id,
            "raw_reports": b.reports, "library_dbs": b.libraries,
            "tool_run_receipt": b.tool_receipt,
            "operator_scope_sha256": EX._map_sha([row[0] for row in EX.SCOPE_ANCHORS]),
            "production_proof": self.proof_spec,
        })
        b.manifest_doc = manifest
        b.manifest_spec = self._write_json(b.manifest_path, manifest)

    def _rebuild_legacy(self, receipt):
        b = self.base
        for step in EX.R8.STEPS:
            path = b.provenance_root / "scripts" / (step + ".tcl")
            receipt["command_scripts"][step] = {
                "path": path.relative_to(REPO_ROOT).as_posix(), "sha256": "0" * 64,
                "media_type": "text/plain"}
        receipt["generation_argv"] = EX.R8._expected_generation_argv(receipt)
        for step in EX.R8.STEPS:
            outputs = EX.R8._step_outputs(step, receipt)
            rows = []
            for name, digest in outputs.items():
                spec = (b.netlist if name == "mapped_netlist" else
                        b.activity if name == "activity" else b.reports[name])
                rows.append("OUTPUT %s %s %s" % (name, spec["path"], digest))
            body = ["M671_COMMAND_ROOT_BEGIN", "STEP " + step, "DESIGN " + b.design,
                    "INPUT_ROOT_SHA256 " + EX._map_sha(EX.R8._step_input_roots(step, receipt))]
            body.extend(rows)
            body.append("M671_COMMAND_ROOT_END")
            receipt["command_scripts"][step] = self._write(
                b.provenance_root / "scripts" / (step + ".tcl"), "\n".join(body) + "\n")
        receipt["generation_argv"] = EX.R8._expected_generation_argv(receipt)
        for step in EX.R8.STEPS:
            tool = ("vcs" if step == "vcs" else "dc_shell" if step == "dc" else
                    "fm_shell" if step == "formality" else
                    "memory_compiler" if step == "memory_compiler" else "pt_shell")
            rows = ["M671_PROVENANCE_BEGIN", "STEP " + step, "TOOL " + tool,
                    "TOOL_EXECUTABLE_SHA256 " + receipt["tool_executables"][tool]["file"]["sha256"],
                    "TOOL_VERSION " + EX.EXPECTED_TOOL_VERSIONS[tool],
                    "ARGV_SHA256 " + EX._map_sha(receipt["generation_argv"][step]),
                    "COMMAND_SCRIPT_SHA256 " + receipt["command_scripts"][step]["sha256"],
                    "INPUT_ROOT_SHA256 " + EX._map_sha(EX.R8._step_input_roots(step, receipt)),
                    "EXIT_STATUS 0"]
            rows.extend("OUTPUT %s %s" % item for item in EX.R8._step_outputs(step, receipt).items())
            rows.append("M671_PROVENANCE_END")
            receipt["tool_logs"][step] = self._write(
                b.provenance_root / "logs" / (step + ".log"), "\n".join(rows) + "\n")
        components = {"netlist": b.netlist["sha256"], "sdc": b.sdc["sha256"],
                      "activity": b.activity["sha256"],
                      "memory_inventory": receipt["memory_inventory_sha256"]}
        components.update({"rtl:" + role: b.rtl_sources[role]["sha256"]
                           for role in EX.RTL_SOURCE_ROLES})
        components.update({"report:" + name: b.reports[name]["sha256"]
                           for name in sorted(EX.REPORT_FIELDS)})
        components.update({"argv:" + step: EX._map_sha(receipt["generation_argv"][step])
                           for step in EX.R8.STEPS})
        components.update({"library:" + role: b.libraries[role]["file"]["sha256"]
                           for role in EX.LIBRARY_ROLES})
        components.update({"executable:" + tool: b.executables[tool]["file"]["sha256"]
                           for tool in EX.TOOL_NAMES})
        components.update({"script:" + step: receipt["command_scripts"][step]["sha256"]
                           for step in EX.R8.STEPS})
        components.update({"log:" + step: receipt["tool_logs"][step]["sha256"]
                           for step in EX.R8.STEPS})
        receipt["component_root_sha256"] = EX._map_sha(components)
        b.tool_receipt = self._write_json(b.run_dir / "tool_run_receipt.json", receipt)
        self.receipt = receipt

    def rewrite_proof(self, proof):
        proof["component_root_sha256"] = EX._map_sha(
            {key: value for key, value in proof.items() if key != "component_root_sha256"})
        self.proof = proof
        self.proof_spec = self._write_json(self.proof_path, proof)
        self.base.manifest_doc["production_proof"] = self.proof_spec
        self.base.manifest_spec = self._write_json(self.base.manifest_path,
                                                   self.base.manifest_doc)


class M691Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = R9Fixture()

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def test_01_canonical_remains_zero(self):
        result = M.build(CONFIG)
        self.assertEqual(result["validated_production_run_count"], 0)
        self.assertEqual(result["trusted_hammer_authority_count"], 0)
        self.assertEqual(result["table_a_evidence_bundle_count"], 0)
        self.assertEqual(result["headline_gate"]["eligible_row_count"], 0)
        self.assertFalse(result["headline_gate"]["admitted"])
        self.assertFalse(result["analytical_diagnostic"]["admitted"])

    def test_02_complete_r9_fixture_extracts(self):
        result = EX.extract_from_manifest(
            self.fixture.base.manifest_path.relative_to(REPO_ROOT).as_posix())
        self.assertEqual(result["area_mode"], "DC_TOTAL_INCLUDES_MACROS")
        self.assertEqual(result["memory_inventory"]["macro_rounded_total_bytes"], 245760)
        self.assertEqual(result["memory_inventory"]["macros"][2]["port_type"], "1R1W")
        self.assertAlmostEqual(result["values"]["logic_area_mm2"] +
                               result["values"]["sram_macro_area_mm2"], 0.6)

    def test_03_plaintext_tool_and_db_reject(self):
        with tempfile.TemporaryDirectory(dir=str(HW_ROOT / "results")) as tmp:
            tmp = Path(tmp)
            tool = self.fixture._write(tmp / "tool", "plain tool\n", "application/octet-stream")
            (tmp / "tool").chmod(0o755)
            with self.assertRaisesRegex(EX.ExtractionError, "ELF"):
                EX._elf_spec(tool, "tool")
            db = self.fixture._write(tmp / "db", "plain db\n" * 1000,
                                     "application/octet-stream")
            with self.assertRaisesRegex(EX.ExtractionError, "binary DB|plaintext DB"):
                EX._binary_db_spec(db, "db")

    def test_04_vcs_compile_and_run_are_exact(self):
        proof = copy.deepcopy(self.fixture.proof)
        proof["execution_steps"]["vcs_compile"]["argv"].remove(
            self.fixture.base.rtl_sources["assertions"]["path"])
        self.fixture.rewrite_proof(proof)
        try:
            with self.assertRaisesRegex(EX.ExtractionError, "strict argv"):
                EX.extract_from_manifest(
                    self.fixture.base.manifest_path.relative_to(REPO_ROOT).as_posix())
        finally:
            self.fixture.rewrite_proof(copy.deepcopy(self.fixture.proof_original))

    def test_05_empty_selected_slice_scope_rejects(self):
        receipt = copy.deepcopy(self.fixture.receipt)
        empty = self.fixture._write(
            self.fixture.base.provenance_root / "inputs/rtl/empty_attack.sv",
            "module %s; endmodule\n" % self.fixture.base.design,
            "text/x-systemverilog")
        receipt["rtl_sources"]["design_rtl"] = empty
        with self.assertRaisesRegex(EX.ExtractionError, "absent or empty"):
            EX._validate_scope(self.fixture.proof, receipt)

    def test_06_saif_duration_top_and_toggle_reject(self):
        with tempfile.TemporaryDirectory(dir=str(HW_ROOT / "results")) as tmp:
            path = Path(tmp) / "bad.saif"
            path.write_text(_saif("wrong_top").replace("(DURATION 1000)",
                                                       "(DURATION 0)"), encoding="utf-8")
            with self.assertRaisesRegex(EX.ExtractionError, "SAIF"):
                EX._parse_saif(path, self.fixture.base.design)

    def test_07_annotation_macro_and_area_attacks_reject(self):
        original = copy.deepcopy(self.fixture.proof)
        proof = copy.deepcopy(original)
        proof["saif_annotation"]["annotated_nets"] = 94
        self.fixture.rewrite_proof(proof)
        try:
            with self.assertRaisesRegex(EX.ExtractionError, "coverage"):
                EX.extract_from_manifest(
                    self.fixture.base.manifest_path.relative_to(REPO_ROOT).as_posix())
        finally:
            self.fixture.rewrite_proof(copy.deepcopy(original))
        proof = copy.deepcopy(original)
        proof["macro_census"]["instances"].pop()
        self.fixture.rewrite_proof(proof)
        try:
            with self.assertRaisesRegex(EX.ExtractionError, "macro instance"):
                EX.extract_from_manifest(
                    self.fixture.base.manifest_path.relative_to(REPO_ROOT).as_posix())
        finally:
            self.fixture.rewrite_proof(copy.deepcopy(original))
        proof = copy.deepcopy(original)
        proof["macro_census"]["logic_cell_area_mm2"] += 0.34
        self.fixture.rewrite_proof(proof)
        try:
            with self.assertRaisesRegex(EX.ExtractionError, "area inclusion"):
                EX.extract_from_manifest(
                    self.fixture.base.manifest_path.relative_to(REPO_ROOT).as_posix())
        finally:
            self.fixture.rewrite_proof(copy.deepcopy(original))

    def test_08_any_negative_derived_power_rejects(self):
        with self.assertRaisesRegex(EX.ExtractionError, "negative derived"):
            EX._reject_negative_power({"logic_internal_power_mw": -5e-13,
                                       "total_power_mw": 1.0})

    def test_09_tool_version_and_vcs_run_receipts_reject_drift(self):
        original = copy.deepcopy(self.fixture.proof)
        proof = copy.deepcopy(original)
        proof["tool_version_runs"]["vcs"]["reported_version"] = "WRONG"
        self.fixture.rewrite_proof(proof)
        try:
            with self.assertRaisesRegex(EX.ExtractionError, "version execution"):
                EX.extract_from_manifest(
                    self.fixture.base.manifest_path.relative_to(REPO_ROOT).as_posix())
        finally:
            self.fixture.rewrite_proof(copy.deepcopy(original))
        proof = copy.deepcopy(original)
        del proof["execution_steps"]["vcs_run"]
        self.fixture.rewrite_proof(proof)
        try:
            with self.assertRaisesRegex(EX.ExtractionError, "step set"):
                EX.extract_from_manifest(
                    self.fixture.base.manifest_path.relative_to(REPO_ROOT).as_posix())
        finally:
            self.fixture.rewrite_proof(copy.deepcopy(original))

    def test_10_ptpx_17_instance_census_rejects_drift(self):
        original = copy.deepcopy(self.fixture.proof)
        proof = copy.deepcopy(original)
        proof["ptpx_memory_census"]["instances"].pop()
        self.fixture.rewrite_proof(proof)
        try:
            with self.assertRaisesRegex(EX.ExtractionError, "PTPX memory instance"):
                EX.extract_from_manifest(
                    self.fixture.base.manifest_path.relative_to(REPO_ROOT).as_posix())
        finally:
            self.fixture.rewrite_proof(copy.deepcopy(original))


if __name__ == "__main__":
    # Frozen restoration root used by mutation tests.
    fixture = None
    unittest.main()
