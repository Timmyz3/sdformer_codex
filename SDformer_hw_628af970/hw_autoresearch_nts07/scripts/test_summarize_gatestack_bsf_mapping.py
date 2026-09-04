#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

from summarize_gatestack_bsf_mapping import build_report


class BsfMappingSummaryTest(unittest.TestCase):
    @staticmethod
    def _write_result(path: Path, stage: int, cycles: int,
                      requests: int, bsf: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"RESULT stage=S{stage} status=PASS bsf={bsf} "
            f"total_cycles={cycles} bias_req_hs={requests} mismatches=0\n"
        )

    def test_joint_cycle_and_area_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mapping = root / "mapping"
            baseline = root / "baseline"
            bsf = root / "bsf"
            mapping.mkdir()
            for mode, area, cells in (
                ("baseline", 100.0, 1000), ("bsf", 105.0, 1010)
            ):
                (mapping / f"{mode}.log").write_text(
                    "   $mem_v2 3\n"
                    f"   Number of cells: {cells}\n"
                    f"   Chip area for module '\\top': {area:.3f}\n"
                )
            for stage in range(4):
                self._write_result(
                    baseline / f"w96_s{stage}" / "iverilog.log",
                    stage, 100, 162, 0)
                self._write_result(
                    bsf / f"hatf96_s{stage}" / "iverilog.log",
                    stage, 90, 1, 1)

            report = build_report(mapping, baseline, bsf)
            base_row, bsf_row = report["rows"]
            self.assertEqual(base_row["total_cycles"], 400)
            self.assertEqual(bsf_row["total_cycles"], 360)
            self.assertEqual(base_row["bias_requests"], 648)
            self.assertEqual(bsf_row["bias_requests"], 4)
            self.assertEqual(base_row["external_bias_payload_bits"],
                             648 * 96 * 32)
            self.assertEqual(bsf_row["resident_bits_per_supertile"], 96 * 32)
            self.assertAlmostEqual(bsf_row["speedup"], 400 / 360)
            self.assertAlmostEqual(bsf_row["logic_area_delta"], 0.05)


if __name__ == "__main__":
    unittest.main()
