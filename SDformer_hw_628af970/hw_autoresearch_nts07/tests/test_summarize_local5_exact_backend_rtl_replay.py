from __future__ import annotations

import sys
import tempfile
import unittest
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import summarize_local5_exact_backend_rtl_replay as summary


class SummarizeLocal5ExactBackendRtlReplayTest(unittest.TestCase):
    def write_manifest(self, root: Path, stages: list[int]) -> Path:
        source_manifest = root / "source_manifest.json"
        source_payload = root / "source_payload.npz"
        source_payload.write_bytes(b"payload")
        source_rows = [
            {
                "tag": index,
                "sample": index % 2,
                "stage": stage,
                "block": 0,
                "window": index,
                "head": 0,
                "flat_group": index,
                "batch_windows": 2,
                "heads": 1,
                "module": f"stage{stage}.block0",
                "selection": "fixture",
            }
            for index, stage in enumerate(stages)
        ]
        source_manifest.write_text(
            json.dumps(
                {
                    "schema": "et3_ordered_term_trace_v2",
                    "qualification": {"qualified": True},
                    "sampling": {
                        "performance_scope": "test fixture",
                        "groups_per_block_sample": 1,
                    },
                    "groups": source_rows,
                }
            ),
            encoding="utf-8",
        )
        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema": "local5_active_projection_postg0_vectors_v1",
                    "source_manifest": str(source_manifest),
                    "source_manifest_sha256": summary.sha256(source_manifest),
                    "source_payload": str(source_payload),
                    "source_payload_sha256": summary.sha256(source_payload),
                    "shape": {"sources": 450, "out_dim": 2},
                    "selection": {
                        "method": "manifest_order_all_groups",
                        "rows": [
                            {
                                **source_rows[index],
                                "input_group_index": index,
                                "vector_group_index": index,
                                "active_sources": 1,
                                "terms": 1,
                                "updates": 1,
                            }
                            for index, stage in enumerate(stages)
                        ],
                    },
                }
            ),
            encoding="utf-8",
        )
        return manifest

    def test_parse_rejects_missing_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.log"
            path.write_text("GROUP group=0 cycles=1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                summary.parse_log(path)

    def test_statistics_names_observed_max(self) -> None:
        row = summary.statistics([1, 2, 3])
        self.assertEqual(row["sample_observed_max"], 3.0)
        self.assertEqual(row["p0"], 1.0)

    def test_weighted_percentile_uses_population_weight(self) -> None:
        self.assertEqual(
            summary.weighted_percentile([1, 10], [99, 1], 95),
            1.0,
        )

    def test_tail_gate_uses_every_stage_p95(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.write_manifest(root, [0, 0, 1, 1])
            direct = root / "direct.log"
            gasr = root / "gasr.log"
            common = (
                " new1rw=1 latency=1 active=1 terms=1 updates=1 term_stall=0 "
                "sram_reads=1 sram_writes=1\n"
            )
            direct.write_text(
                "".join(
                    f"GROUP mode=0 group={index} cycles={cycles}{common}"
                    for index, cycles in enumerate([100, 100, 10, 10])
                )
                + "PASS post-G0 active projection\n",
                encoding="utf-8",
            )
            gasr.write_text(
                "".join(
                    f"GROUP mode=1 group={index} cycles={cycles}{common}"
                    for index, cycles in enumerate([50, 50, 11, 11])
                )
                + "PASS post-G0 active projection\n",
                encoding="utf-8",
            )

            report = summary.build_report(manifest, direct, gasr)

            self.assertTrue(report["local_pre_result_gate"]["overall_p95_non_regression_pass"])
            self.assertFalse(report["local_pre_result_gate"]["each_stage_p95_non_regression_pass"])
            self.assertFalse(report["local_pre_result_gate"]["tail_pass"])
            self.assertAlmostEqual(
                report["aggregate"]["post_hoc_perfect_mode_oracle"][
                    "speedup_over_direct"
                ],
                220 / 120,
            )
            self.assertEqual(report["numeric_miter"]["comparisons_per_mode"], 3600)
            self.assertEqual(
                report["numeric_miter"]["total_comparisons_across_two_modes"], 7200
            )

    def test_rejects_wrong_backend_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.write_manifest(root, [0])
            bad = root / "bad.log"
            bad.write_text(
                "GROUP new1rw=1 mode=0 latency=1 group=0 cycles=10 active=1 "
                "terms=1 updates=1 term_stall=0 sram_reads=1 sram_writes=1\n"
                "PASS post-G0 active projection\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                summary.build_report(manifest, bad, bad)


if __name__ == "__main__":
    unittest.main()
