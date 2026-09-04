#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Mutation tests for M1336 release source; never launches VCS or a license query."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


HERE = Path(__file__).resolve().parent
CHECKER = HERE / "static_check_m1336_c2_activity_vcs_release_source.py"
SPEC = importlib.util.spec_from_file_location("m1336_release_checker", CHECKER)
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
assert SPEC.loader is not None
SPEC.loader.exec_module(M)


class Tests(unittest.TestCase):
    def setUp(self):
        self.contract = json.loads(M.CONTRACT.read_text())
        self.runner = M.RUNNER.read_text()
        self.digest = self.contract["identity"]["runner_sha256"]

    def test_01_exact_runner_passes(self):
        self.assertTrue(M.exact_runner_gate(self.runner, self.digest))

    def test_02_any_runner_byte_mutation_rejected(self):
        mutants = (
            self.runner + "\n",
            self.runner.replace("automatic_retry=false", "automatic_retry=true", 1),
            self.runner.replace("for axis in k8 k1x8", "for axis in k8", 1),
            self.runner.replace("for case_id in 0 1 2 3 4", "for case_id in 0", 1),
            self.runner.replace("M1334_SAIF_FILE", "M1334_BAD_SAIF_FILE", 1),
            self.runner.replace("publish_no_replace", "mv", 1),
        )
        for mutant in mutants:
            self.assertNotEqual(mutant, self.runner)
            self.assertFalse(M.exact_runner_gate(mutant, self.digest))

    def test_03_all_external_release_digests_required(self):
        good = {name: "a" * 64 for name in M.ENV_NAMES}
        self.assertTrue(M.env_gate(good))
        for name in M.ENV_NAMES:
            bad = dict(good); bad.pop(name)
            self.assertFalse(M.env_gate(bad))
            bad = dict(good); bad[name] = "A" * 64
            self.assertFalse(M.env_gate(bad))

    def test_04_current_namespaces_are_absent_including_broken_links(self):
        for path in M.namespaces():
            self.assertFalse(os.path.lexists(path), str(path))

    def test_05_two_by_five_workload_is_exact(self):
        self.assertEqual(self.contract["workloads"], {
            "events": [20, 41, 90, 110, 0],
            "k8_cycles": [51, 131, 486, 1231, 14],
            "k1x8_cycles": [53, 133, 499, 1246, 14],
        })
        self.assertEqual(self.contract["future_execution"]["vcs_compiles"], 2)
        self.assertEqual(self.contract["future_execution"]["simv_runs"], 10)

    def test_06_claims_remain_false(self):
        for key in ("functional_vcs_verified", "production_saif", "ptpx",
                    "power", "energy", "performance", "system_speedup",
                    "paper_ppa_ready", "headline"):
            self.assertIs(self.contract["claim_boundary"][key], False)

    def test_07_frozen_netlist_and_protocol_objects_match(self):
        for path, digest in M.EXPECTED.items():
            self.assertEqual(M.sha(path), digest, str(path))

    def test_08_runner_never_consumes_workspace_ucli_state(self):
        self.assertNotIn("ucli.key", self.runner)
        self.assertIn('M1334_SAIF_FILE="${saif}"', self.runner)
        self.assertIn('-ucli -i "${UCLI}"', self.runner)

    def test_09_no_retry_no_replace_and_double_seals_are_active(self):
        self.assertNotIn("automatic_retry=true", self.runner)
        self.assertIn('publish_no_replace "${ATTEMPT_STAGE}" "${ATTEMPT}"', self.runner)
        self.assertIn('publish_no_replace "${RESULT_STAGE}" "${RESULT}"', self.runner)
        self.assertIn('seal_dir "${FAILURE_STAGE}"', self.runner)
        self.assertIn('seal_dir "${RESULT_STAGE}"', self.runner)

    def test_10_runner_is_inert_without_external_release_digests(self):
        before = [os.path.lexists(path) for path in M.namespaces()]
        run = subprocess.run(["/usr/bin/bash", str(M.RUNNER)],
                             env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8",
                                  "LC_ALL": "C.UTF-8"},
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             universal_newlines=True, check=False)
        self.assertEqual(run.returncode, 2)
        self.assertIn("M1336_EXPECTED_RUNNER_SHA256 absent/invalid", run.stderr)
        self.assertEqual(before, [os.path.lexists(path) for path in M.namespaces()])


if __name__ == "__main__":
    unittest.main(verbosity=2)
