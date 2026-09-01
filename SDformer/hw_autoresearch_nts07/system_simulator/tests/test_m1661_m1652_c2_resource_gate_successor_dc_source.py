#!/usr/bin/env python3
"""Static author checks for M1661 executable-preflight C2 successor."""
from __future__ import print_function

import hashlib
import json
from pathlib import Path
import re
import stat
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
RUNNER = HW / "dc_handoff/scripts/run_dc_m1661_m1652_c2_resource_gate_successor_exact_sha_r1.sh"
FILELIST = HW / "dc_handoff/filelists/date_m1634_c2_m1609_registered_fault_three_axis_logic_only_dc.f"
CONTRACT = HW / "contracts/m1661_m1652_c2_resource_gate_successor_dc_source_contract_r1_20260901.json"
TCL = HW / "dc_handoff/scripts/run_dc_m519_r8_setup_area_three_axis.tcl"
SDC = HW / "dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc"
M1609 = HW / "rtl_m1609/m1609_m214_fc2_raw4_to_descriptor4_terminal_hint_compactor_registered_fault_successor.sv"
M214 = HW / "rtl_m214/m214_fc2_raw4_to_descriptor4_terminal_hint_compactor.sv"
M216 = HW / "rtl_m216/m216_fc2_raw4_to_source_cap_frontend.sv"
M519 = HW / "rtl_m519/m519_fc2_registered_release_standalone_raw4_acc24.sv"
M803_K8 = HW / "rtl_m803/m803_fc2_k8_channel_split_registered_release_8bank_raw4_acc24.sv"
M1627 = HW / "reviews/m1627_m1613_c2_registered_fault_directed_vcs_result_independent_hammer_r1_20260901"
M903 = HW / "reviews/m903_m872_m803_c2_r16_three_axis_dc_result_hammer_r1_20260829"
M1634_RUNNER = HW / "dc_handoff/scripts/run_dc_m1634_m1609_c2_registered_fault_three_axis_logic_only_exact_sha_r1.sh"
M1634_CONTRACT = HW / "contracts/m1634_m1609_c2_registered_fault_three_axis_logic_only_dc_source_contract_r1_20260901.json"
M1635 = HW / "reviews/m1635_m1634_m1609_c2_three_axis_dc_source_hammer_r1_20260901"
M1636_RELEASE = HW / "contracts/m1636_m1635_m1634_m1609_c2_three_axis_dc_launch_release_r1_20260901.json"
M1641 = HW / "reviews/m1641_m1636_m1634_m1609_c2_three_axis_dc_release_hammer_r1_20260901"
M1652_RUNNER = HW / "dc_handoff/scripts/run_dc_m1652_m1634_c2_resource_gate_successor_exact_sha_r1.sh"
M1652_CONTRACT = HW / "contracts/m1652_m1634_c2_resource_gate_successor_dc_source_contract_r1_20260901.json"
M1653_FAIL = HW / "reviews/m1653_m1652_m1634_c2_resource_gate_successor_dc_source_hammer_r1_20260901"
RESULT = HW / "dc_handoff/runs/m1661_m1652_c2_resource_gate_successor_three_axis_logic_only_dc_3p000ns_r1_20260901"
ATTEMPT = HW / "dc_handoff/runs/.m1661_m1652_c2_resource_gate_successor_three_axis_dc_attempt_consumed"
WORK_GLOB = ".m1661_m1652_c2_resource_gate_successor_three_axis_dc_work.*"
LOCK = HW / "dc_handoff/runs/.m1661_m1652_c2_resource_gate_successor_three_axis_dc_launch_lock"
FUTURE_REVIEW = HW / "reviews/m1662_m1661_m1652_c2_resource_gate_successor_dc_source_hammer_r1_20260901"
FUTURE_RELEASE = HW / "contracts/m1663_m1662_m1661_m1652_c2_resource_gate_successor_dc_launch_release_r1_20260901.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED_ROWS = (
    "rtl_m1609/m1609_m214_fc2_raw4_to_descriptor4_terminal_hint_compactor_registered_fault_successor.sv",
    "rtl_m216/m216_fc2_descriptor4_source_cap_frontend.sv",
    "rtl_m216/m216_fc2_raw4_to_source_cap_frontend.sv",
    "rtl_m218/m218_fc2_tagged_slice_service_island.sv",
    "rtl_m499/m499_fc2_bundle_to_8bank_no_reuse_adapter.sv",
    "rtl_m519/m519_fc2_k1_registered_release_service_island.sv",
    "rtl_m519/m519_fc2_registered_release_standalone_raw4_acc24.sv",
    "rtl_m519/m519_fc2_k1_registered_release_8bank_raw4_acc24.sv",
    "rtl_m519/m519_fc2_k1x8_registered_release_raw4_acc24.sv",
    "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv",
    "rtl_m803/m803_fc2_k8_channel_split_registered_release_8bank_raw4_acc24.sv",
    "rtl_m803/m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24.sv",
)
PREDECESSOR = "rtl_m214/m214_fc2_raw4_to_descriptor4_terminal_hint_compactor.sv"
EXPECTED = {
    RUNNER: "9bf1e220054ff28e3c7bad27b07bc61f50a504625f4b7df0893b0e50162e80e6",
    FILELIST: "03c4dcd546da19d5de231fa80032473e7c365592012661e6ed77019d7bab4f3f",
    CONTRACT: "1e2f04c6c46c69c58659e406b6c5d055f24c91429d6e2dcd9dd7bb1a53df03ed",
    TCL: "c9da61c9a483487b3d1157538481a6c940d7277534e2acef634c4b1a1ff7adbe",
    SDC: "808307c496bd67843907b727acdfe18ea3b48565798f97cb55e689c70c1183f5",
    M1609: "7ee28b3912ae34c99c795a48e80be29df2b59b363e5de2d2b359175ec9dda931",
    M214: "e278da8b0deaa0dda07b0477930453daa40b0331399a3941b743d604d0b102a5",
    M1627 / "review.json": "ab4f2187667301a37fbd5f523687a8971282e642163d42886edcdc138edc43d4",
    M903 / "review.json": "89785b3a06fc5981cb1e652bce18c4ab3853809ccf6dee7d1b96a65bd018b10a",
    M1634_RUNNER: "da9cd0d118021eb85c8b548d93f6779ec6d25b6fec7ca5894bdae988a95840b7",
    M1634_CONTRACT: "9f5e5b1cb40da5cd403270ba48ceac9b5a7d6aecd79b7ad98cf3d644d0f8f030",
    M1652_RUNNER: "57f9b90642641215c801b0f61302636ddecb81e6b37523763f6523f2862dfdb3",
    M1652_CONTRACT: "01ee8cff796705c71a0b3c5875046ca32d08935936026315375da797d02d863c",
    M1653_FAIL / "review.json": "5e3e6c9974e26a28be3e6bae7efc93e661afafaf0ba8b5b9ebf35e5ad0855d6d",
    M1653_FAIL / "SHA256SUMS": "75f5f21569e53c351fa2bd92eef1b5e33075e033baced7783fb9041d5064515d",
    M1653_FAIL / "SHA256SUMS.seal.sha256": "0f29ab34485b854bb492415e51301591798d26c1ed2ab411c0d897d0d70d7113",
    M1635 / "review.json": "215dfaa31a91b372f5318109eb3eac05a7de7a346815916d8296b51e2f0a6620",
    M1636_RELEASE: "0b1945b7060e5b2af9557ceb4b72f5c0a1fb862af48534c3abc59669cbfa5088",
    M1641 / "review.json": "278df1d44232cccabc0c50e45beae9dee60adce834896f1be20f8fc7625bf1e6",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def command_text(text, comment="#"):
    return "\n".join(line.split(comment, 1)[0] for line in text.splitlines())


def module_header(text, module):
    match = re.search(r"\bmodule\s+" + re.escape(module) + r"\b(.*?);", text,
                      re.S)
    if not match:
        raise AssertionError("module header absent " + module)
    return match.group(1)


def declared_ports(header):
    return tuple(re.findall(
        r"\b(?:input|output)\s+logic(?:\s+signed)?(?:\s*\[[^\]]+\])?\s+([A-Za-z_]\w*)",
        header))


def embedded_authorization_preflight(runner):
    snippets = re.findall(r"<<'PY'\n(.*?)\nPY", runner, re.S)
    selected = [text for text in snippets if "contract,runner,m1627,m903" in text]
    if len(selected) != 1:
        raise AssertionError("embedded authorization preflight cardinality drift")
    return selected[0]


def execute_embedded_authorization_preflight(runner, contract_path):
    snippet = embedded_authorization_preflight(runner)
    return subprocess.run(
        [str(Path(__import__("sys").executable)), "-I", "-", str(contract_path),
         str(RUNNER), str(M1627 / "review.json"), str(M903 / "review.json")],
        input=snippet, universal_newlines=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=15, check=False)


def validate_source_texts(runner, rows, contract):
    if tuple(rows) != EXPECTED_ROWS:
        raise AssertionError("filelist topology drift")
    if PREDECESSOR in rows or rows.count(EXPECTED_ROWS[0]) != 1:
        raise AssertionError("compactor selection drift")
    required_runner = (
        "axis_names=(k1 k8 k1x8)", "axis_modes=(0 1 2)",
        "for index in 0 1 2", 'mkdir -- "${ATTEMPT}"',
        '"${DC_SHELL}" -f "${TCL}"', "fresh_all_axes=true",
        "old_netlist_reuse=false", "hold_diagnostic_only=true",
        "M1662_M1661_M1652_C2_RESOURCE_GATE_SOURCE_HAMMER",
        "AUTHORIZE_ONE_M1661_C2_RESOURCE_GATE_SUCCESSOR_DC_ATTEMPT",
        "M1661_EXPECTED_DC_RUNNER_SHA256",
        "M1661_EXPECTED_DC_RELEASE_SHA256",
        "PASS_M1641_M1636_C2_THREE_AXIS_DC_RELEASE_HAMMER__ONE_LAUNCH_ADMITTED",
        "FAIL_M1653_M1652_C2_RESOURCE_GATE_SOURCE_HAMMER__NO_RELEASE",
        "auth=c['authorization']",
        "assert auth['dc_runs_now']==0",
        "assert auth['future_dc_shell_runs_max']==3",
        "assert auth['all_other_eda_runs']==0",
        "assert auth['vcs_runs']==0",
        "assert auth['pt_runs']==0",
        "assert auth['formality_runs']==0",
        "assert auth['ptpx_runs']==0",
        "assert auth['gpu_runs']==0",
        "assert auth['remote_runs']==0",
        "assert auth['attempts_created_now']==0",
        "assert auth['retry'] is False",
        "mapped.v", "mapped.sdc", '.ddc"', '.svf"',
        "timing_setup.rpt", "timing_hold_diagnostic.rpt",
        "area.rpt", "qor.rpt", "automatic_retry':False",
    )
    for token in required_runner:
        if token not in runner:
            raise AssertionError("runner token absent " + token)
    if not (runner.index('verify_dir_seal "${HAMMER_DIR}"') <
            runner.index('mkdir -- "${ATTEMPT}"') <
            runner.index('"${DC_SHELL}" -f "${TCL}"')):
        raise AssertionError("review/attempt/tool order drift")
    if "rm -rf" in runner or "pt_shell" in runner or "fm_shell" in runner:
        raise AssertionError("destructive or unauthorized tool token")
    if contract.get("status") != (
            "SOURCE_ONLY_M1661_M1652_C2_RESOURCE_GATE_SUCCESSOR__NO_EDA_AUTHORIZED"):
        raise AssertionError("contract status drift")
    auth = contract.get("authorization", {})
    if auth.get("dc_runs_now") != 0 or auth.get("future_dc_shell_runs_max") != 3 or \
            auth.get("all_other_eda_runs") != 0 or auth.get("retry") is not False:
        raise AssertionError("authorization drift")
    if runner.count('"${headroom}" -ge 50331648') != 1 or \
            '"${headroom}" -ge 67108864' in runner:
        raise AssertionError("commit-headroom successor drift")
    if runner.count('"${mem_available}" -ge 100663296') != 1 or \
            runner.count('"${swap_free}" -ge 16777216') != 1:
        raise AssertionError("resident-memory or swap gate drift")
    gate = contract.get("resource_gate", {})
    if gate.get("old_commit_headroom_min_kib") != 67108864 or \
            gate.get("commit_headroom_min_kib") != 50331648 or \
            gate.get("mem_available_min_kib") != 100663296 or \
            gate.get("swap_free_min_kib") != 16777216 or \
            gate.get("same_uid_dc_collision_tolerance") != 0 or \
            gate.get("physical_or_result_condition_changed") is not False:
        raise AssertionError("resource-gate contract drift")
    fair = contract.get("fair_three_axis_definition", {})
    if fair.get("axis_order") != ["k1", "k8", "k1x8"] or \
            fair.get("frozen_baseline_netlist_reuse") is not False:
        raise AssertionError("fair-axis contract drift")
    claims = contract.get("claim_boundary", {})
    if claims.get("source_only") is not True or claims.get("dc_authorized") is not False or \
            claims.get("fresh_mapped_k8") is not False or \
            claims.get("hold_closed") is not False or \
            claims.get("power") is not False or claims.get("paper_headline") is not False:
        raise AssertionError("claim boundary drift")
    if contract.get("identity", {}).get("m1627_review_sha256") != EXPECTED[M1627 / "review.json"]:
        raise AssertionError("M1627 binding drift")


class M1661ExecutablePreflightResourceGateSuccessorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = RUNNER.read_text(encoding="utf-8")
        cls.rows = [row for row in FILELIST.read_text(encoding="utf-8").splitlines()
                    if row.strip()]
        cls.contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        cls.tcl = TCL.read_text(encoding="utf-8")
        cls.sdc = SDC.read_text(encoding="utf-8")

    def test_01_exact_identities(self):
        for path, digest in EXPECTED.items():
            self.assertTrue(stat.S_ISREG(path.lstat().st_mode), str(path))
            self.assertFalse(path.is_symlink(), str(path))
            self.assertEqual(sha(path), digest, str(path))
        self.assertEqual(self.contract["identity"]["runner_sha256"], EXPECTED[RUNNER])
        self.assertEqual(self.contract["identity"]["filelist_sha256"], EXPECTED[FILELIST])

    def test_02_unique_m1609_selection(self):
        self.assertEqual(tuple(self.rows), EXPECTED_ROWS)
        self.assertNotIn(PREDECESSOR, self.rows)
        self.assertEqual(self.rows.count(EXPECTED_ROWS[0]), 1)
        for row in self.rows:
            self.assertTrue((HW / row).is_file(), row)
            self.assertFalse((HW / row).is_symlink(), row)

    def test_03_m1609_port_compatible_semantic_delta(self):
        module = "m214_fc2_raw4_to_descriptor4_terminal_hint_compactor"
        new_text = M1609.read_text(encoding="utf-8")
        old_text = M214.read_text(encoding="utf-8")
        self.assertEqual(declared_ports(module_header(new_text, module)),
                         declared_ports(module_header(old_text, module)))
        self.assertIn("assign protocol_error = fault_q;", new_text)
        self.assertIn("assign protocol_error = fault_q || illegal_request;", old_text)
        self.assertEqual(new_text.count("module " + module), 1)

    def test_04_m1609_is_in_all_axis_source_cones(self):
        self.assertIn("m214_fc2_raw4_to_descriptor4_terminal_hint_compactor #(",
                      M216.read_text(encoding="utf-8"))
        self.assertIn("m216_fc2_raw4_to_source_cap_frontend #(",
                      M519.read_text(encoding="utf-8"))
        self.assertIn("m519_fc2_registered_release_standalone_raw4_acc24 #(",
                      M803_K8.read_text(encoding="utf-8"))
        top = (HW / EXPECTED_ROWS[-1]).read_text(encoding="utf-8")
        for branch in ("g_k1", "g_k8", "g_k1x8"):
            self.assertIn(branch, top)
        for mode in ("ARCH_MODE == 0", "ARCH_MODE == 1", "ARCH_MODE == 2"):
            self.assertIn(mode, top)

    def test_05_frozen_setup_area_tcl_and_constraint(self):
        commands = command_text(self.tcl)
        self.assertEqual(len(re.findall(r"(?m)^\s*compile_ultra\s*$", commands)), 1)
        self.assertEqual(len(re.findall(r"(?m)^\s*compile\b", commands)), 0)
        self.assertIn("timing_hold_diagnostic.rpt", self.tcl)
        self.assertIn("report_timing -delay_type min", self.tcl)
        for suffix in ("_mapped.v", "_mapped.sdc", ".ddc", ".svf"):
            self.assertIn(suffix, self.tcl)
        self.assertRegex(self.sdc, r"create_clock[^\n]*-period\s+\$clock_period_ns")
        self.assertIn("set_clock_uncertainty -setup 0.200", self.sdc)
        self.assertIn("set_clock_uncertainty -hold 0.050", self.sdc)
        for token in ("set_false_path", "set_multicycle_path", "set_min_delay",
                      "set_max_delay", "set_disable_timing", "set_case_analysis"):
            self.assertIsNone(re.search(r"(?m)^\s*" + token + r"\b",
                                        commands + "\n" + command_text(self.sdc)))

    def test_06_embedded_authorization_preflight_executes_pass(self):
        completed = execute_embedded_authorization_preflight(self.runner, CONTRACT)
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertNotIn("AssertionError", completed.stdout)
        self.assertNotIn("c['authorization']==", embedded_authorization_preflight(self.runner))

    def test_07_embedded_authorization_preflight_rejects_all_field_mutations(self):
        mutations = (
            ("dc_runs_now", 1),
            ("future_dc_shell_runs_max", 4),
            ("all_other_eda_runs", 1),
            ("vcs_runs", 1),
            ("pt_runs", 1),
            ("formality_runs", 1),
            ("ptpx_runs", 1),
            ("gpu_runs", 1),
            ("remote_runs", 1),
            ("attempts_created_now", 1),
            ("retry", True),
        )
        rejected = []
        with tempfile.TemporaryDirectory(prefix="m1661_inline_preflight_") as directory:
            candidate_path = Path(directory) / "contract.json"
            for key, value in mutations:
                candidate = json.loads(json.dumps(self.contract))
                candidate["authorization"][key] = value
                candidate_path.write_text(
                    json.dumps(candidate, sort_keys=True) + "\n", encoding="utf-8")
                completed = execute_embedded_authorization_preflight(
                    self.runner, candidate_path)
                self.assertNotEqual(completed.returncode, 0, key)
                self.assertIn("AssertionError", completed.stdout, key)
                rejected.append(key)
        self.assertEqual(len(rejected), 11)

    def test_08_runner_shell_and_exact_three_fresh_axes(self):
        completed = subprocess.run(
            ["/usr/bin/bash", "-n", str(RUNNER)], cwd=str(ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, timeout=10, check=False)
        self.assertEqual(completed.returncode, 0, completed.stdout)
        validate_source_texts(self.runner, self.rows, self.contract)
        self.assertEqual(self.runner.count('"${DC_SHELL}" -f "${TCL}"'), 1)
        self.assertIn("for index in 0 1 2", self.runner)
        self.assertNotRegex(self.runner, r"cp[^\n]*(?:M872_RESULT|\.ddc|_mapped\.v)")

    def test_09_fail_closed_result_and_artifact_gate(self):
        for token in (
            "same-UID DC collision", "memory/commit gate not met",
            "M1661_ATTEMPT_CONSUMED", "retry=false", "failed_or_incomplete",
            "TIM-209=0", "OPT-150=0", "slack (MET)",
            "This design has no violated constraints.",
            "PASS_RAW_M1661_M1609_C2_THREE_AXIS_LOGIC_ONLY_DC_PENDING_INDEPENDENT_RESULT_HAMMER",
            "mv -T -n", "independent result hammer required",
        ):
            self.assertIn(token, self.runner)
        self.assertNotIn("rm -rf", self.runner)

    def test_10_claims_and_cycle_provenance_are_closed(self):
        c = self.contract
        self.assertFalse(c["claim_boundary"]["dc_completed"])
        self.assertFalse(c["claim_boundary"]["setup_area"])
        self.assertFalse(c["claim_boundary"]["hold_closed"])
        self.assertFalse(c["claim_boundary"]["power"])
        self.assertFalse(c["claim_boundary"]["system_speedup"])
        old = c["baseline_provenance"]["frozen_directed_cycles_not_refreshed_by_dc"]
        self.assertEqual(old["k8_sum"], 1913)
        self.assertEqual(old["k1x8_sum"], 1945)
        self.assertIn("not a full-network", old["scope"].replace("component workloads", "component workloads; not a full-network"))

    def test_11_resource_gate_only_and_runtime_namespaces_fresh(self):
        self.assertEqual(self.runner.count('"${headroom}" -ge 50331648'), 1)
        self.assertNotIn('"${headroom}" -ge 67108864', self.runner)
        self.assertEqual(self.runner.count('"${mem_available}" -ge 100663296'), 1)
        self.assertEqual(self.runner.count('"${swap_free}" -ge 16777216'), 1)
        for path in (RESULT, ATTEMPT, LOCK):
            self.assertFalse(path.exists(), str(path))
            self.assertFalse(path.is_symlink(), str(path))
        self.assertEqual(list((HW / "dc_handoff/runs").glob(WORK_GLOB)), [])
        self.assertFalse(FUTURE_REVIEW.exists())
        self.assertFalse(FUTURE_RELEASE.exists())

    def test_12_contract_and_authorities_are_double_sealed(self):
        sidecar = Path(str(CONTRACT) + ".sha256")
        outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
        self.assertEqual(sidecar.read_text(encoding="ascii").split(),
                         [sha(CONTRACT), CONTRACT.name])
        self.assertEqual(outer.read_text(encoding="ascii").split(),
                         [sha(sidecar), sidecar.name])
        self.assertEqual(self.contract["functional_authority"]["m1627_score"], 99)
        self.assertEqual(self.contract["baseline_provenance"]["m903_status"],
                         "PASS100_M872_M803_C2_R16_THREE_AXIS_LOGIC_ONLY_DC_RESULT_ADMITTED")

    def test_13_predecessor_failure_and_future_authorities_are_hard_gates(self):
        self.assertIn("m1635_m1634_m1609_c2_three_axis_dc_source_hammer", self.runner)
        self.assertIn("m1636_m1635_m1634_m1609_c2_three_axis_dc_launch_release", self.runner)
        self.assertIn("m1641_m1636_m1634_m1609_c2_three_axis_dc_release_hammer", self.runner)
        self.assertIn("m1653_m1652_m1634_c2_resource_gate_successor_dc_source_hammer", self.runner)
        self.assertIn("m1662_m1661_m1652_c2_resource_gate_successor_dc_source_hammer", self.runner)
        failed = json.loads((M1653_FAIL / "review.json").read_text(encoding="utf-8"))
        self.assertEqual(failed["status"],
                         "FAIL_M1653_M1652_C2_RESOURCE_GATE_SOURCE_HAMMER__NO_RELEASE")
        self.assertEqual(failed["p1_count"], 1)
        self.assertIn("m1663_m1662_m1661_m1652_c2_resource_gate_successor_dc_launch_release", self.runner)
        self.assertFalse(self.contract["future_release_chain"]["present_at_source_authoring"])
        self.assertLess(self.runner.index('verify_dir_seal "${HAMMER_DIR}"'),
                        self.runner.index('mkdir -- "${ATTEMPT}"'))

    def test_14_no_current_eda_or_other_tool_authority(self):
        commands = command_text(self.runner).lower()
        for token in ("/opt/synopsys/vcs", "/opt/synopsys/pt",
                      "/opt/synopsys/fm", "/opt/synopsys/formality"):
            self.assertNotIn(token, commands)
        self.assertEqual(self.contract["authorization"]["dc_runs_now"], 0)
        self.assertEqual(self.contract["authorization"]["all_other_eda_runs"], 0)

    def test_15_mutation_hammer(self):
        attacks = []

        def reject(label, runner=None, rows=None, contract=None):
            try:
                validate_source_texts(
                    self.runner if runner is None else runner,
                    self.rows if rows is None else rows,
                    self.contract if contract is None else contract)
            except AssertionError:
                attacks.append(label)
                return
            self.fail("mutation escaped: " + label)

        reject("predecessor_filelist", rows=[PREDECESSOR] + self.rows[1:])
        reject("duplicate_m1609", rows=[self.rows[0]] + self.rows)
        reject("missing_k1x8", rows=self.rows[:-4] + self.rows[-3:])
        reject("filelist_reorder", rows=[self.rows[1], self.rows[0]] + self.rows[2:])
        reject("axis_name_drop", runner=self.runner.replace(
            "axis_names=(k1 k8 k1x8)", "axis_names=(k1 k8)"))
        reject("axis_mode_alias", runner=self.runner.replace(
            "axis_modes=(0 1 2)", "axis_modes=(0 1 1)"))
        reject("loop_drop", runner=self.runner.replace(
            "for index in 0 1 2", "for index in 0 1"))
        reject("attempt_after_tool", runner=self.runner.replace(
            'mkdir -- "${ATTEMPT}"', 'true', 1))
        reject("fresh_axes_false", runner=self.runner.replace(
            "fresh_all_axes=true", "fresh_all_axes=false"))
        reject("old_netlist_reuse", runner=self.runner.replace(
            "old_netlist_reuse=false", "old_netlist_reuse=true"))
        reject("hold_claim", runner=self.runner.replace(
            "hold_diagnostic_only=true", "hold_diagnostic_only=false"))
        reject("artifact_drop", runner=self.runner.replace(
            "reports/timing_hold_diagnostic.rpt", "reports/removed_hold.rpt"))
        reject("source_hammer_status", runner=self.runner.replace(
            "PASS_M1662_M1661_M1652_C2_RESOURCE_GATE_SOURCE_HAMMER",
            "PASS_FAKE_SOURCE_HAMMER"))
        reject("release_status", runner=self.runner.replace(
            "AUTHORIZE_ONE_M1661_C2_RESOURCE_GATE_SUCCESSOR_DC_ATTEMPT",
            "AUTHORIZE_UNBOUNDED"))
        reject("commit_floor", runner=self.runner.replace(
            '"${headroom}" -ge 50331648', '"${headroom}" -ge 1'))
        reject("mem_available_floor", runner=self.runner.replace(
            '"${mem_available}" -ge 100663296', '"${mem_available}" -ge 1'))
        reject("swap_floor", runner=self.runner.replace(
            '"${swap_free}" -ge 16777216', '"${swap_free}" -ge 1'))
        reject("future_runner_pin", runner=self.runner.replace(
            "M1661_EXPECTED_DC_RUNNER_SHA256", "M1661_UNPINNED_DC_RUNNER_SHA256"))
        reject("future_release_pin", runner=self.runner.replace(
            "M1661_EXPECTED_DC_RELEASE_SHA256", "M1661_UNPINNED_DC_RELEASE_SHA256"))
        reject("predecessor_release_hammer_status", runner=self.runner.replace(
            "PASS_M1641_M1636_C2_THREE_AXIS_DC_RELEASE_HAMMER__ONE_LAUNCH_ADMITTED",
            "PASS_FAKE_RELEASE_HAMMER"))
        reject("m1653_fail_status", runner=self.runner.replace(
            "FAIL_M1653_M1652_C2_RESOURCE_GATE_SOURCE_HAMMER__NO_RELEASE",
            "PASS_FAKE_PREDECESSOR"))
        for key, value in (
                ("dc_runs_now", "1"), ("future_dc_shell_runs_max", "4"),
                ("all_other_eda_runs", "1"), ("vcs_runs", "1"),
                ("pt_runs", "1"), ("formality_runs", "1"),
                ("ptpx_runs", "1"), ("gpu_runs", "1"), ("remote_runs", "1"),
                ("attempts_created_now", "1")):
            old = "assert auth['%s']==%s" % (
                key, "3" if key == "future_dc_shell_runs_max" else "0")
            reject("inline_auth_" + key, runner=self.runner.replace(
                old, "assert auth['%s']==%s" % (key, value)))
        reject("inline_auth_retry", runner=self.runner.replace(
            "assert auth['retry'] is False", "assert auth['retry'] is True"))

        def mutate_contract(path, value):
            candidate=json.loads(json.dumps(self.contract))
            cursor=candidate
            for part in path[:-1]: cursor=cursor[part]
            cursor[path[-1]]=value
            return candidate

        reject("contract_status", contract=mutate_contract(
            ["status"], "AUTHORIZED"))
        reject("dc_now", contract=mutate_contract(
            ["authorization", "dc_runs_now"], 1))
        reject("future_runs", contract=mutate_contract(
            ["authorization", "future_dc_shell_runs_max"], 4))
        reject("other_eda", contract=mutate_contract(
            ["authorization", "all_other_eda_runs"], 1))
        reject("frozen_netlist_reuse", contract=mutate_contract(
            ["fair_three_axis_definition", "frozen_baseline_netlist_reuse"], True))
        reject("claim_dc_authorized", contract=mutate_contract(
            ["claim_boundary", "dc_authorized"], True))
        reject("claim_fresh_k8", contract=mutate_contract(
            ["claim_boundary", "fresh_mapped_k8"], True))
        reject("claim_hold", contract=mutate_contract(
            ["claim_boundary", "hold_closed"], True))
        reject("claim_power", contract=mutate_contract(
            ["claim_boundary", "power"], True))
        reject("m1627_binding", contract=mutate_contract(
            ["identity", "m1627_review_sha256"], "0" * 64))
        reject("contract_commit_floor", contract=mutate_contract(
            ["resource_gate", "commit_headroom_min_kib"], 1))
        reject("contract_old_commit_floor", contract=mutate_contract(
            ["resource_gate", "old_commit_headroom_min_kib"], 1))
        reject("contract_mem_floor", contract=mutate_contract(
            ["resource_gate", "mem_available_min_kib"], 1))
        reject("contract_swap_floor", contract=mutate_contract(
            ["resource_gate", "swap_free_min_kib"], 1))
        reject("contract_physical_changed", contract=mutate_contract(
            ["resource_gate", "physical_or_result_condition_changed"], True))
        self.assertEqual(len(attacks), 47)


if __name__ == "__main__":
    unittest.main(verbosity=2)
