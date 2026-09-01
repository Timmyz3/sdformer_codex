#!/usr/bin/env python3
"""Author tests for source-only M1701 M1695 tool-entity repair."""
from __future__ import print_function

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
TCL = HW / "dc_handoff/scripts/run_dc_m1695_m1665_c1_fastmin_hold_closure_candidate.tcl"
M1695_RUNNER = HW / "dc_handoff/scripts/run_dc_m1695_m1665_c1_fastmin_hold_closure_exact_one_shot.sh"
RUNNER = HW / "dc_handoff/scripts/run_dc_m1701_m1695_c1_tool_entity_repair_exact_one_shot.sh"
TEST = Path(__file__).resolve()
CONTRACT = HW / "contracts/m1701_m1695_c1_tool_entity_repair_source_contract_r1_20260901.json"
WITNESS = HW / "contracts/m1701_m1695_dc_shell_symlink_pre_attempt_failure_witness_r1_20260901.json"
M1697 = HW / "contracts/m1697_m1696_m1695_c1_fastmin_hold_closure_launch_release_r1_20260901.json"
M1702 = HW / "reviews/m1702_m1701_m1695_c1_tool_entity_repair_source_hammer_r1_20260901"
M1703 = HW / "contracts/m1703_m1702_m1701_m1695_c1_tool_entity_repair_launch_release_r1_20260901.json"
RESULT = HW / "dc_handoff/runs/m1701_m1695_c1_tool_entity_repair_dc_r1_20260901"
ATTEMPT = HW / "dc_handoff/runs/.m1701_m1695_c1_tool_entity_repair_dc_attempt_consumed"
M1695_RESULT = HW / "dc_handoff/runs/m1695_m1665_c1_fastmin_hold_closure_dc_r1_20260901"
M1695_ATTEMPT = HW / "dc_handoff/runs/.m1695_m1665_c1_fastmin_hold_closure_dc_attempt_consumed"
DC_ENTRY = Path("/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell")
DC_TARGET = Path("/opt/synopsys/syn/V-2023.12-SP3/bin/snps_shell")

EXPECTED_TCL_SHA = "cb05b053078c7ab9d084cddf5028802aeff52ef1a4aef6d1b026ba6da2f41ad8"
EXPECTED_M1695_RUNNER_SHA = "f470eee1f4f68be76d4d680522efca4157472582e9f442721ef836bd5957ca5d"
EXPECTED_M1697_SHA = "45fe5a6029a182a52a63fab47288eff982ce64a861c113767cdc3db00e3c3fbb"
EXPECTED_TARGET_SHA = "23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def commands(text):
    return "\n".join(row.split("#", 1)[0] for row in text.splitlines())


def verify_file_seal(payload):
    manifest = Path(str(payload) + ".sha256")
    outer = Path(str(payload) + ".sha256.seal.sha256")
    if manifest.read_text(encoding="ascii").split() != [sha256(payload), payload.name]:
        raise AssertionError("payload seal drift " + str(payload))
    if outer.read_text(encoding="ascii").split() != [sha256(manifest), manifest.name]:
        raise AssertionError("outer seal drift " + str(payload))


def validate_entity_shape(raw, direct, resolved, target_is_regular,
                          target_is_symlink, target_sha):
    if raw != "snps_shell":
        raise ValueError("raw link")
    if direct != str(DC_TARGET):
        raise ValueError("direct target")
    if resolved != str(DC_TARGET):
        raise ValueError("resolved target")
    if not target_is_regular or target_is_symlink:
        raise ValueError("target entity")
    if target_sha != EXPECTED_TARGET_SHA:
        raise ValueError("target sha")


def validate_tcl(text):
    cmd = commands(text)
    conditions = [
        len(re.findall(r"(?m)^\s*read_ddc\b", cmd)) == 1,
        len(re.findall(r"(?m)^\s*set_fix_hold\s+\$core_clock\s*$", cmd)) == 1,
        len(re.findall(r"(?m)^\s*compile\s+-incremental_mapping\s+-only_hold_time\s*$", cmd)) == 1,
        len(re.findall(r"(?m)^\s*compile\b", cmd)) == 1,
        text.count("set optimization_hold_uncertainty_ns 0.081") == 1,
        text.count("set reported_hold_uncertainty_ns 0.050") == 1,
        text.count("set_min_library $std_slow_db -min_version $std_fast_db") == 1,
        text.count("set_min_library $macro_slow_db -min_version $macro_fast_db") == 1,
    ]
    if not all(conditions):
        raise ValueError("M1695 hold-only flow drift")


class M1701SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = RUNNER.read_text(encoding="utf-8")
        cls.tcl = TCL.read_text(encoding="utf-8")
        cls.contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        cls.witness = json.loads(WITNESS.read_text(encoding="utf-8"))

    def test_01_predecessor_identities_are_exact(self):
        self.assertEqual(sha256(TCL), EXPECTED_TCL_SHA)
        self.assertEqual(sha256(M1695_RUNNER), EXPECTED_M1695_RUNNER_SHA)
        self.assertEqual(sha256(M1697), EXPECTED_M1697_SHA)

    def test_02_official_entry_is_exact_direct_symlink(self):
        self.assertTrue(stat.S_ISLNK(DC_ENTRY.lstat().st_mode))
        raw = os.readlink(str(DC_ENTRY))
        direct = str((DC_ENTRY.parent / raw).absolute())
        resolved = str(DC_ENTRY.resolve(strict=True))
        target_mode = DC_TARGET.lstat().st_mode
        validate_entity_shape(raw, direct, resolved,
                              stat.S_ISREG(target_mode),
                              stat.S_ISLNK(target_mode), sha256(DC_TARGET))

    def test_03_entity_mutations_are_rejected(self):
        good = ("snps_shell", str(DC_TARGET), str(DC_TARGET), True, False,
                EXPECTED_TARGET_SHA)
        mutations = [
            ("../bin/snps_shell",) + good[1:],
            (good[0], "/opt/synopsys/syn/V-2023.12-SP3/bin/other") + good[2:],
            good[:2] + ("/opt/synopsys/syn/V-2023.12-SP3/bin/other",) + good[3:],
            good[:3] + (False,) + good[4:],
            good[:4] + (True,) + good[5:],
            good[:5] + ("0" * 64,),
        ]
        for row in mutations:
            with self.assertRaises(ValueError):
                validate_entity_shape(*row)

    def test_04_witness_binds_false_negative_and_absence(self):
        w = self.witness
        self.assertEqual(w["status"],
            "PASS_PRE_ATTEMPT_FAILURE_WITNESS__M1695_SHA_EXACT_REJECTED_OFFICIAL_DC_SHELL_SYMLINK__NO_EDA")
        self.assertEqual(w["observed_tool_entity"]["raw_readlink"], "snps_shell")
        self.assertEqual(w["observed_tool_entity"]["resolved_sha256"], EXPECTED_TARGET_SHA)
        self.assertFalse(w["m1695_rejection"]["gate_result"])
        self.assertTrue(w["absence_witness"]["result_absent"])
        self.assertTrue(w["absence_witness"]["attempt_absent"])
        self.assertFalse(M1695_RESULT.exists())
        self.assertFalse(M1695_ATTEMPT.exists())

    def test_05_witness_and_contract_are_double_sealed(self):
        verify_file_seal(WITNESS)
        verify_file_seal(CONTRACT)

    def test_06_shell_syntax_and_only_one_dc_invocation(self):
        run = subprocess.run(["/usr/bin/bash", "-n", str(RUNNER)],
                             cwd=str(ROOT), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, universal_newlines=True,
                             timeout=10, check=False)
        self.assertEqual(run.returncode, 0, run.stdout)
        cmd = commands(self.runner)
        launch = '"${DC_SHELL}" -no_home_init -no_local_init -no_gui -f "${TCL}"'
        self.assertEqual(cmd.count(launch), 1)
        for token in ('"${FM_SHELL}"', '"${PT_SHELL}"', "vcs -full64", "ptpx"):
            self.assertNotIn(token, cmd)

    def test_07_repair_gate_is_narrow_and_rechecks_after_hash(self):
        self.assertEqual(self.runner.count("sha_official_direct_symlink_exact"), 2)
        required = (
            'raw="$(readlink -- "${entry}")"',
            'direct="$(readlink -m -- "$(dirname -- "${entry}")/${raw}")"',
            'resolved="$(readlink -f -- "${entry}")"',
            '[[ -f "${expected_resolved}" && ! -L "${expected_resolved}" ]]',
            "official tool target changed while hashing",
            "official tool entity changed after hashing",
        )
        for token in required:
            self.assertEqual(self.runner.count(token), 1, token)
        self.assertNotIn('sha_exact 23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2 "${DC_SHELL}"', self.runner)

    def test_08_m1695_hold_only_tcl_is_byte_identical_and_preserved(self):
        self.assertEqual(sha256(TCL), EXPECTED_TCL_SHA)
        validate_tcl(self.tcl)
        self.assertNotIn("read_verilog", commands(self.tcl))
        self.assertNotIn("compile_ultra", commands(self.tcl))

    def test_09_constraint_resource_and_result_gates_are_preserved(self):
        for token in (
            '"${headroom}" -ge 25165824',
            'SHARED_QUEUE="/tmp/date_dual_synopsys_same_uid_eda_queue.lock"',
            'fail "same-UID DC collision after shared lock"',
            'fail "same-UID DC collision immediately before launch"',
            "area<=ceiling", "macro_count_post=9", "design_rule_violating_nets",
            "set_fix_hold_count':1", "hold_only_incremental_mapping_count':1",
        ):
            self.assertIn(token, self.runner, token)
        self.assertIn("ancestry=set(); pid=os.getpid()", self.runner)
        self.assertNotIn("flock -u", self.runner)

    def test_10_new_namespaces_and_one_attempt_no_retry(self):
        for token in (
            "m1701_m1695_c1_tool_entity_repair_dc_r1_20260901",
            ".m1701_m1695_c1_tool_entity_repair_dc_attempt_consumed",
            "m1702_m1701_m1695_c1_tool_entity_repair_source_hammer",
            "m1703_m1702_m1701_m1695_c1_tool_entity_repair_launch_release",
            "M1701_EXPECTED_DC_RUNNER_SHA256",
            "M1701_EXPECTED_DC_RELEASE_SHA256",
            "max_dc_runs=1", "retry=false",
        ):
            self.assertIn(token, self.runner, token)
        self.assertFalse(RESULT.exists())
        self.assertFalse(ATTEMPT.exists())
        self.assertFalse(M1702.exists())
        self.assertFalse(M1703.exists())

    def test_11_authority_check_precedes_attempt_and_launch(self):
        authority = self.runner.index('verify_file_seal "${CONTRACT}"')
        attempt = self.runner.index('mkdir -- "${ATTEMPT}"')
        launch = self.runner.index('"${DC_SHELL}" -no_home_init -no_local_init -no_gui -f "${TCL}"')
        self.assertLess(authority, attempt)
        self.assertLess(attempt, launch)
        self.assertEqual(self.runner.count('mkdir -- "${ATTEMPT}"'), 1)

    def test_12_contract_binds_all_repair_sources_and_opens_no_run(self):
        c = self.contract
        self.assertEqual(c["status"],
            "SOURCE_ONLY_M1701_M1695_C1_TOOL_ENTITY_REPAIR__NO_EDA_AUTHORIZED")
        self.assertEqual(c["identity"]["runner_sha256"], sha256(RUNNER))
        self.assertEqual(c["identity"]["tcl_sha256"], EXPECTED_TCL_SHA)
        self.assertEqual(c["identity"]["author_test_sha256"], sha256(TEST))
        self.assertEqual(c["identity"]["failure_witness_sha256"], sha256(WITNESS))
        self.assertEqual(c["identity"]["m1695_runner_sha256"], EXPECTED_M1695_RUNNER_SHA)
        self.assertEqual(c["identity"]["m1697_release_sha256"], EXPECTED_M1697_SHA)
        self.assertEqual(c["authorization"]["dc_runs_now"], 0)
        self.assertEqual(c["authorization"]["future_dc_runs_max"], 1)
        self.assertFalse(c["claim_boundary"]["hold_closed"])
        self.assertFalse(c["claim_boundary"]["paper_citable"])

    def test_13_no_docs359_or_rtl_write_target(self):
        self.assertEqual(self.runner.count('DOC359="${HW_ROOT}/docs/359_'), 1)
        self.assertNotRegex(self.runner, r"(?:>|>>|cp|mv)\s+[^\n]*\$\{DOC359\}")
        self.assertNotIn("rtl_m935", self.tcl)
        self.assertFalse(any("docs/359" in x for x in self.contract["outputs"]))

    def test_14_only_tool_entity_gate_changed_in_scope(self):
        c = self.contract
        self.assertEqual(c["repair_scope"]["changed_item_count"], 1)
        self.assertEqual(c["repair_scope"]["changed_item"], "dc_shell_entry_entity_validation")
        for key, value in c["preserved_m1695_flow"].items():
            if isinstance(value, bool):
                self.assertTrue(value, key)


if __name__ == "__main__":
    unittest.main(verbosity=2)
