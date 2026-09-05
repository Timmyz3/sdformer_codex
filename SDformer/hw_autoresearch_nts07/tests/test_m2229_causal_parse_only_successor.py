"""Negative cases use the actual sealed logs, without writes or EDA."""
import importlib.util
import unittest
from pathlib import Path

HW = Path(__file__).resolve().parents[1]
SCRIPT = HW / "system_simulator/scripts/run_m2231_m2229_causal_parse_only_successor.py"
spec = importlib.util.spec_from_file_location("m2229", SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ParseOnlyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.comp = (m.Q / "vcs_compile.log").read_text()
        cls.sim = (m.Q / "simv.log").read_text()
        cls.rc = (m.Q / "simv.rc").read_text()
        cls.tb = (m.REPO / m.TB_REL).read_text()

    def test_real_frozen_log(self):
        actual = m.parse_raw(self.comp, self.sim, self.rc, self.tb)
        self.assertEqual(actual["raw_log_ledger"], m.EXPECTED)

    def test_compile_identity_and_completion_rejected(self):
        for token in ["Chronologic VCS (TM)", "Version V-2023.12-SP1_Full64",
                      "All of 7 modules done", "simv up to date", "to link"]:
            with self.subTest(token=token), self.assertRaises(ValueError):
                m.parse_raw(self.comp.replace(token, "REMOVED"), self.sim, self.rc, self.tb)

    def test_runtime_faults_and_identity_rejected(self):
        changes = [self.sim.replace("Chronologic VCS simulator copyright", "REMOVED"),
            self.sim.replace("Runtime version V-2023.12-SP1_Full64", "REMOVED"),
            self.sim + "\nError: injected\n", self.sim + "\n$fatal injected\n",
            self.sim + "\nassertion failed\n", self.sim + "\nWarning-[NEW] injected\n",
            self.sim.replace("Time: 10330500 ps", "Time: incomplete")]
        for index, sim in enumerate(changes):
            with self.subTest(index=index), self.assertRaises(ValueError):
                m.parse_raw(self.comp, sim, self.rc, self.tb)
        with self.assertRaises(ValueError):
            m.parse_raw(self.comp, self.sim, "1", self.tb)

    def test_warning_allowlist_is_narrow(self):
        changes = [self.comp + "\nWarning-[NEW] new warning\n",
            self.comp + "\nWarning: unreviewed warning form\n",
            self.comp.replace("'context' is a SystemVerilog", "'other' is a SystemVerilog", 1),
            self.comp.replace("Keyword used as identifier\n" + m.TB_REL,
                              "Keyword used as identifier\nhw_autoresearch_nts07/rtl_m2213/changed.sv", 1),
            self.comp.replace("Warning-[KUAI]", "Warning-[UNKNOWN]", 1),
            self.comp.replace("Rocky Linux release 8.10 (Green Obsidian)", "other OS")]
        for index, comp in enumerate(changes):
            with self.subTest(index=index), self.assertRaises(ValueError):
                m.parse_raw(comp, self.sim, self.rc, self.tb)

    def test_count_duplicate_and_cover_rejections(self):
        raw_line = next(line for line in self.sim.splitlines() if line.startswith(m.RAW_PASS))
        changes = [self.sim + "\n" + raw_line + "\n",
            self.sim.replace("preread_reads=576", "preread_reads=577"),
            self.sim.replace("postread_bank_rsp=1728", "postread_bank_rsp=1727"),
            self.sim.replace("golden_mismatches=0", "golden_mismatches=1"),
            self.sim.replace("commits_each=24", "commits_each=23"),
            self.sim.replace("rows=24", "rows=24 rows=24"),
            self.sim.replace("552 match", "0 match"),
            self.sim.replace("cp_postread_commit_terminal", "removed_cover")]
        for index, sim in enumerate(changes):
            with self.subTest(index=index), self.assertRaises(ValueError):
                m.parse_raw(self.comp, sim, self.rc, self.tb)

    def test_source_review_gate(self):
        good = {"status": "PASS_M2230_M2229_PARSE_ONLY_SOURCE__M2231_CPU_PARSE_AUTHORIZED",
            "score_over_100": 98, "severity_counts": {"p0": 0, "p1": 0},
            "identity": {"source_contract_sha256": "contract", "parser_runner_sha256": m.sha(SCRIPT)},
            "authorization": {"cpu_parse_runs": 1, "license_queries": 0,
                "eda_runs": 0, "gpu_runs": 0, "automatic_retry": False}}
        m.authorize(good, "contract")
        import copy
        bad = copy.deepcopy(good)
        bad["authorization"]["eda_runs"] = 1
        with self.assertRaises(ValueError):
            m.authorize(bad, "contract")
        bad = copy.deepcopy(good)
        bad["severity_counts"]["p1"] = 1
        with self.assertRaises(ValueError):
            m.authorize(bad, "contract")
        with self.assertRaises(ValueError):
            m.authorize(good, "wrong contract")


if __name__ == "__main__":
    unittest.main()
