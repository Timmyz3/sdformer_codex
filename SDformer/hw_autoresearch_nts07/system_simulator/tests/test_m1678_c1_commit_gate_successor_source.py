#!/usr/bin/env python3
from __future__ import print_function

import hashlib
import json
import unittest
from pathlib import Path


HW = Path(__file__).resolve().parents[2]
OLD = HW / "dc_handoff/scripts/run_m1674_m1665_c1_transitive_formality_ptsta_exact_closed_one_shot.sh"
NEW = HW / "dc_handoff/scripts/run_m1678_m1674_c1_commit_gate_successor_transitive_formality_ptsta_exact_closed_one_shot.sh"
CONTRACT = HW / "contracts/m1678_m1674_c1_commit_gate_successor_transitive_formality_ptsta_source_contract_r1_20260901.json"
M1675 = HW / "reviews/m1675_m1674_m1665_c1_transitive_formality_ptsta_source_hammer_r1_20260901"
M1676 = HW / "contracts/m1676_m1675_m1674_m1665_c1_transitive_formality_ptsta_launch_release_r1_20260901.json"
RESULT = HW / "dc_handoff/runs/m1678_m1674_c1_commit_gate_successor_transitive_formality_ptsta_r1_20260901"
ATTEMPT = HW / "dc_handoff/runs/.m1678_m1674_c1_commit_gate_successor_transitive_formality_ptsta_attempt_consumed"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestM1678CommitGateSuccessor(unittest.TestCase):
    def test_exact_source_identity(self):
        d = json.loads(CONTRACT.read_text())
        self.assertEqual(d["source_files"], {
            "dc_handoff/scripts/run_m1678_m1674_c1_commit_gate_successor_transitive_formality_ptsta_exact_closed_one_shot.sh": sha(NEW),
            "system_simulator/tests/test_m1678_c1_commit_gate_successor_source.py": sha(Path(__file__).resolve()),
        })

    def test_predecessors_are_immutable_and_exact(self):
        d = json.loads(CONTRACT.read_text())
        self.assertEqual(sha(OLD), "55409e053c7392de2e5962d7d8a9430cfc6429483ea3d774cd7ff4906305b944")
        self.assertEqual(sha(M1675 / "review.json"), "644fba82b931b4bcc84287731ce6144a6fae94127fe8b8cf466e2512bf8b88e7")
        self.assertEqual(sha(M1675 / "SHA256SUMS.seal.sha256"), "73a01b08f7f21781512f0f0c2da38189d2a96875568f1848f17bc6a87cd0e07b")
        self.assertEqual(sha(M1676), "121e0843c69dccbb2039d9127e3732754d2d299bf5a818c1c3038b1d940be5a6")
        self.assertEqual(sha(Path(str(M1676) + ".sha256.seal.sha256")), "5cc03cd4c50de76c5c801e59b9f8513115855beffa1b066d3b188bfa68b9be50")
        self.assertEqual(d["immutable_predecessors"]["m1674_runner_sha256"], sha(OLD))
        self.assertEqual(d["immutable_predecessors"]["m1675_review_sha256"], sha(M1675 / "review.json"))
        self.assertEqual(d["immutable_predecessors"]["m1676_release_sha256"], sha(M1676))

    def test_only_live_resource_gate_delta(self):
        old = OLD.read_text()
        new = NEW.read_text()
        self.assertIn('"${commit_headroom}" -ge 50331648', old)
        self.assertIn('"${commit_headroom}" -ge 25165824', new)
        self.assertNotIn('"${commit_headroom}" -ge 50331648', new)
        for token in (
            '"${mem_available}" -ge 16777216',
            '"${disk_available}" -ge 4194304',
            '[[ -z "$(same_uid_eda)" ]] || exit 4',
            'for feature in Formality PrimeTime',
            'Total of\\s+(\\d+)\\s+licenses? issued',
        ):
            self.assertEqual(old.count(token), new.count(token), token)
        self.assertNotIn("SwapFree", old)
        self.assertNotIn("SwapFree", new)

    def test_eda_execution_and_result_gates_are_byte_identical(self):
        old = OLD.read_text()
        new = NEW.read_text()
        start = 'export M1674_SNAPSHOT_ROOT="${HW_ROOT}"'
        end = '/usr/bin/python3 - "${WORK}" <<\'PY\''
        old_slice = old[old.index(start):old.index(end)]
        new_slice = new[new.index(start):new.index(end)]
        self.assertEqual(old_slice, new_slice)
        for token in (
            '"${FM_SHELL}" -f "${RTL_TO_M993_TCL}"',
            '"${FM_SHELL}" -f "${GATE_TO_GATE_TCL}"',
            '"${PT_SHELL}" -f "${PT_TCL}"',
        ):
            self.assertEqual(new.count(token), 1)
        self.assertLess(new.index('"${FM_SHELL}" -f "${RTL_TO_M993_TCL}"'),
                        new.index('"${FM_SHELL}" -f "${GATE_TO_GATE_TCL}"'))
        self.assertLess(new.index('"${FM_SHELL}" -f "${GATE_TO_GATE_TCL}"'),
                        new.index('"${PT_SHELL}" -f "${PT_TCL}"'))

    def test_reuses_exact_m1674_tcls_and_all_physical_gates(self):
        text = NEW.read_text()
        exact = {
            "run_formality_m1674_c1_rtl_to_m993_transitive.tcl": "d3a72876d9b40f73c47834da123388fa40263cf017c61586f2113b352a7bc3de",
            "run_formality_m1674_c1_m993_to_m1665_gate_to_gate.tcl": "6df82c2435ab312263fd133a8e52371ea3de1004bc493d9553879eafaf3d1e12",
            "run_ptsta_m1674_c1_m1665_slowmax_fastmin.tcl": "e289faa0abb9f8e7136305158ef086e20bd7e77d2f960e436f51138a431241a1",
        }
        for name, digest in exact.items():
            self.assertIn(name, text)
            self.assertIn(digest, text)
        for token in (
            "setup<0 or hold<0 or setup_tns!=0 or hold_tns!=0",
            "int(machine['macro_count'])!=9",
            "No unmatched points",
            "No failing compare points",
            "No aborted compare points",
            "No unverified compare points",
            "forbidden timing exception",
        ):
            self.assertIn(token, text)

    def test_fresh_authority_and_namespace(self):
        text = NEW.read_text()
        for token in (
            "M1678_EXPECTED_RUNNER_SHA256",
            "M1678_EXPECTED_RELEASE_SHA256",
            "PASS_M1679_M1678_C1_COMMIT_GATE_SUCCESSOR_SOURCE_HAMMER__AUTHORIZE_ONE_FUTURE_ATTEMPT",
            "AUTHORIZE_ONE_M1678_C1_COMMIT_GATE_SUCCESSOR_FORMALITY_PTSTA_ATTEMPT",
            "m1680_m1679_m1678_m1674_c1_commit_gate_successor_transitive_formality_ptsta_launch_release_r1_v1",
            "future_m1678_attempts",
        ):
            self.assertIn(token, text)
        self.assertFalse(RESULT.exists())
        self.assertFalse(ATTEMPT.exists())

    def test_source_only_contract(self):
        d = json.loads(CONTRACT.read_text())
        self.assertEqual(d["authorization_now"], {
            "formality_runs": 0, "pt_runs": 0, "dc_runs": 0,
            "vcs_runs": 0, "ptpx_runs": 0, "gpu_runs": 0,
            "remote_runs": 0, "attempts_created": 0,
        })
        self.assertEqual(d["resource_gate_delta"], {
            "field": "commit_headroom_min_kib",
            "m1674": 50331648,
            "m1678": 25165824,
            "all_other_runtime_gates_byte_or_predicate_identical": True,
        })
        self.assertEqual(d["future_execution_budget"], {
            "formality_processes_exact": 2,
            "prime_time_processes_exact": 1,
            "all_other_eda_processes": 0,
            "max_attempts": 1,
            "retry": False,
        })


if __name__ == "__main__":
    unittest.main()
