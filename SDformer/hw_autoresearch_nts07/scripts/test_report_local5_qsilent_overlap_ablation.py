from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import report_local5_qsilent_overlap_ablation as report


def make_log(cycles: int, groups: int = 100, include_overlap: bool = True) -> str:
    rows = []
    base = cycles // groups
    remainder = cycles % groups
    for group in range(groups):
        row_cycles = base + (1 if group < remainder else 0)
        overlap = " overlap=0" if include_overlap else ""
        rows.append(
            "GROUP backend=0 latency=1 group={} cycles={} score_rows=450 "
            "score_service=0 score_direct_rows=0 qsilent_rows=0 identk_rows=0{} "
            "active=0 memory_wait=0 terms=0 updates=0".format(
                group, row_cycles, overlap
            )
        )
    rows.append(
        f"PASS Local5 score-to-projection backend=0 latency=1 "
        f"groups={groups} total_cycles={cycles}"
    )
    return "\n".join(rows) + "\n"


class Local5OverlapReportTest(unittest.TestCase):
    def test_parse_requires_complete_new_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "complete.log"
            path.write_text(make_log(324605), encoding="utf-8")
            parsed = report.parse_log(path)
            self.assertEqual(parsed["total_cycles"], 324605)
            self.assertEqual(parsed["score_rows"], 45000)

    def test_old_log_without_overlap_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.log"
            path.write_text(make_log(324605, include_overlap=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete groups"):
                report.parse_log(path)

    def test_anchor_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vectors = root / "vectors"
            results = root / "results"
            vectors.mkdir()
            results.mkdir()
            (vectors / "manifest.json").write_text(
                json.dumps({"shape": {"out_dim": 2}}), encoding="utf-8"
            )
            for name, cycles in report.SEALED_ANCHORS.items():
                (results / f"{name}.log").write_text(
                    make_log(cycles), encoding="utf-8"
                )
            for name in set(report.CONFIGS) - set(report.SEALED_ANCHORS):
                (results / f"{name}.log").write_text(
                    make_log(200000), encoding="utf-8"
                )
            (results / "q0_serial.log").write_text(
                make_log(191425), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "sealed anchor drift"):
                report.build_report(results, vectors)


if __name__ == "__main__":
    unittest.main()
