import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_ppdi_ibf_projection import build_report


class SummarizePpdiIbfProjectionTest(unittest.TestCase):
    def test_build_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, cycles in {
                "scalar_rmw": 128,
                "ppdi_rmw": 125,
                "scalar_ibf": 98,
                "ppdi_ibf": 95,
            }.items():
                (root / f"{name}_iverilog.log").write_text(
                    f"PASS DCTF96 BANKLOCAL PROJECTION cycles={cycles}\n",
                    encoding="utf-8",
                )
            for name, cells, area in [
                ("scalar_rmw", 100, 200.0),
                ("ppdi_rmw", 103, 204.0),
                ("scalar_ibf", 104, 207.0),
                ("ppdi_ibf", 105, 210.0),
            ]:
                (root / f"map_{name}.log").write_text(
                    f"Number of cells: {cells}\n"
                    "Chip area for module "
                    f"'\\gatestack_dctf96_banklocal_projection_top': {area}\n",
                    encoding="utf-8",
                )
            profile = root / "profile.json"
            profile.write_text(
                json.dumps(
                    {
                        "sample0_window0": {
                            "scalar_commands": 100,
                            "ppdi_commands": 70,
                            "command_reduction": 0.3,
                        }
                    }
                ),
                encoding="utf-8",
            )

            report = build_report(root, profile)

            self.assertAlmostEqual(report["cycle_speedup"]["combined"], 128 / 95)
            self.assertAlmostEqual(
                report["open_logic_mapping"]["area_ratios"]["ppdi_ibf"], 1.05
            )
            self.assertEqual(
                report["motion_sample0_window0_ppdi"]["ppdi_commands"], 70
            )
            self.assertEqual(
                report["analytical_bias_traffic"]["162"][
                    "scalar_bias_reads_per_three_bank_tile"
                ],
                486,
            )


if __name__ == "__main__":
    unittest.main()
