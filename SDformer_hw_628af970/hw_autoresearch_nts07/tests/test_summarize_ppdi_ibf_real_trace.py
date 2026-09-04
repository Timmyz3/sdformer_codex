import tempfile
import unittest
from pathlib import Path

from scripts.summarize_ppdi_ibf_real_trace import build_report


class SummarizePpdiIbfRealTraceTest(unittest.TestCase):
    def test_build_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cycles = {
                "scalar_rmw": [100, 200, 300, 400],
                "ppdi_rmw": [90, 190, 280, 350],
                "scalar_ibf": [80, 100, 200, 300],
                "ppdi_ibf": [70, 90, 180, 250],
            }
            for mode, values in cycles.items():
                (root / mode).mkdir()
                for stage, value in enumerate(values):
                    text = (
                        f"PASS DCTF96 REAL TRACE stage=S{stage} heads=3 "
                        f"cycles={value} terms=4 physical_weight_req=12 "
                        f"bias_req={3 if 'ibf' in mode else 486} "
                        "final_checks=15552\n"
                    )
                    (root / mode / f"icarus_s{stage}.log").write_text(
                        text, encoding="utf-8"
                    )
                    if mode == "ppdi_ibf" and stage == 0:
                        (root / mode / "verilator_s0.log").write_text(
                            text, encoding="utf-8"
                        )

            report = build_report(root)

            self.assertEqual(report["total_cycles"]["scalar_rmw"], 1000)
            self.assertAlmostEqual(
                report["speedup_vs_scalar_rmw"]["ppdi_ibf"], 1000 / 590
            )
            self.assertEqual(report["total_bias_requests"]["scalar_ibf"], 12)


if __name__ == "__main__":
    unittest.main()
