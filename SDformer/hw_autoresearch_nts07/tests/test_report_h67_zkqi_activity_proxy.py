import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class ActivityProxyTest(unittest.TestCase):
    def test_strong_baseline_and_candidate_ledgers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pair = root / "pair.log"
            ttb = root / "ttb.log"
            common = (
                "ROW_RESULT row=0 stage=0 block=0 head=0 bundle_skip={bundle} "
                "active_pairs=2 outputs=3 baseline_preload=225 zkqi_preload=225 "
                "baseline_cycles=10 zkqi_cycles={cycles} baseline_e2e_cycles=235 "
                "zkqi_e2e_cycles={e2e} baseline_slots=225 zkqi_slots=2 seeded=446 "
                "baseline_read_bits=1000 zkqi_read_bits=200 fifo_max=1\n"
            )
            pair.write_text(common.format(bundle=0, cycles=10, e2e=235), encoding="utf-8")
            ttb.write_text(common.format(bundle=1, cycles=5, e2e=230), encoding="utf-8")
            out = root / "out"
            subprocess.run(
                [
                    "python3", "scripts/report_h67_zkqi_activity_proxy.py",
                    "--pair-log", str(pair), "--ttb-log", str(ttb),
                    "--output-dir", str(out),
                ],
                check=True,
            )
            report = json.loads((out / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["event_ledger"]["baseline_rqtb2s"]["score_evaluations"], 225)
            self.assertEqual(report["event_ledger"]["ttb8_zkqi"]["bundle_header_tests"], 29)
            self.assertAlmostEqual(report["reductions_vs_baseline"]["qk_read_bit_reduction"], 0.8)
            self.assertIn("不是门级toggle", (out / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
