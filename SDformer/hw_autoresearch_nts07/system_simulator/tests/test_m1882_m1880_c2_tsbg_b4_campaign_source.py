#!/usr/bin/env python3
"""Dual-interpreter source/mutation tests for inert M1882; never run EDA."""
from __future__ import print_function

import importlib.util
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
CHECKER = HERE.parent / "scripts/check_m1882_m1880_c2_tsbg_b4_campaign_source.py"
SPEC = importlib.util.spec_from_file_location("m1882_checker_tests", str(CHECKER))
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class M1882CampaignSourceTest(unittest.TestCase):
    def test_01_positive_source_and_zero_execution(self):
        value = CHECK.validate_sources()
        self.assertEqual(value["status"],
                         "PASS_M1882_C2_TSBG_B4_CAMPAIGN_SOURCE_STATIC_NO_EDA")
        self.assertEqual(value["author_execution"], {
            "license_queries": 0, "vcs": 0, "simv": 0, "eda": 0,
            "attempts": 0, "results": 0, "releases": 0})
        self.assertTrue(all(flag is False
                            for flag in value["claim_boundary"].values()))

    def test_02_m1880_source_and_filelist_exact_identity(self):
        value = CHECK.validate_sources()
        identity = value["upstream_identity"]
        self.assertEqual(CHECK.sha(CHECK.RTL), identity["m1880_rtl_sha256"])
        self.assertEqual(CHECK.sha(CHECK.SVA), identity["m1880_sva_sha256"])
        self.assertEqual(CHECK.sha(CHECK.TB), identity["m1880_tb_sha256"])
        self.assertEqual(CHECK.sha(CHECK.FILELIST),
                         identity["m1880_filelist_sha256"])

    def test_03_m1881_only_authorizes_campaign_source(self):
        value = CHECK.strict_json(CHECK.M1881 / "review.json")
        self.assertEqual(value["severity_counts"],
                         {"p0": 0, "p1": 0, "p2": 0})
        self.assertTrue(value["authorization"]["author_m1882_campaign_source"])
        self.assertFalse(value["authorization"]["create_naked_release"])
        self.assertFalse(value["authorization"]["run_vcs"])

    def test_04_future_chain_requires_review_release_and_release_audit(self):
        value = CHECK.validate_contract()["future_chain"]
        self.assertEqual(value, {
            "campaign_source_review": "M1884",
            "launch_release": "M1885",
            "launch_release_audit": "M1886",
            "all_three_required_before_attempt": True,
            "one_license_query_one_compile_one_simv": True,
            "result_hammer_required": True,
            "naked_release_forbidden": True})

    def test_05_runner_compile_keeps_sva_enabled(self):
        text = CHECK.RUNNER.read_text(encoding="utf-8")
        self.assertIn('"-assert", "svaext"', text)
        self.assertEqual(text.count('state["license_queries"] += 1'), 1)
        self.assertEqual(text.count('state["vcs_compiles"] += 1'), 1)
        self.assertEqual(text.count('state["simv_runs"] += 1'), 1)

    def test_06_attempt_and_publication_are_fail_closed(self):
        text = CHECK.RUNNER.read_text(encoding="utf-8")
        for token in ("ATTEMPT.mkdir()", "publish_no_replace(STAGE, RESULT)",
                      "publish_no_replace(FAIL_STAGE, FAILURE)",
                      "if state[\"attempt\"] and not state[\"complete\"]",
                      "if success == failure"):
            self.assertIn(token, text)
        for token in ("os.replace(", ".rename(", "shutil.move(", "LOCK_SH"):
            self.assertNotIn(token, text)

    def test_07_lock_path_resource_and_prior_simv_gates_present(self):
        text = CHECK.RUNNER.read_text(encoding="utf-8")
        for token in ("date_dual_synopsys_same_uid_eda_queue.lock",
                      "m1882_m1880_c2_tsbg_b4_directed_vcs.lock",
                      "exact_result_path", "prior private build or simv namespace",
                      "same-UID EDA collision", "MemAvailable below 16 GiB",
                      "commit headroom below 16 GiB"):
            self.assertIn(token, text)

    def test_08_mutation_inventory_covers_governance_categories(self):
        names = [item[0] for item in CHECK.MUTATION_SPECS]
        self.assertGreaterEqual(len(names), 60)
        self.assertEqual(len(names), len(set(names)))
        for prefix in ("call_", "lock_", "path_", "attempt_", "publish_",
                       "future_", "count_"):
            self.assertTrue(any(name.startswith(prefix) for name in names), prefix)


def make_mutation_test(name, old, new):
    def test(self):
        base = CHECK.source_texts()
        runner = base[CHECK.RUNNER]
        self.assertEqual(runner.count(old), 1,
                         "mutation anchor cardinality " + name)
        mutated = dict(base)
        mutated[CHECK.RUNNER] = runner.replace(old, new, 1)
        with self.assertRaises(CHECK.CheckFailure):
            CHECK.validate_semantics(mutated)
    return test


for index, (name, old, new) in enumerate(CHECK.MUTATION_SPECS, 9):
    setattr(M1882CampaignSourceTest,
            "test_{0:03d}_reject_{1}".format(index, name),
            make_mutation_test(name, old, new))


if __name__ == "__main__":
    unittest.main(verbosity=2)
