from __future__ import annotations

import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from scripts.audit_local5_formal_phase_archive_scale import (
    FORMAL_GROUPS,
    FORMAL_PHASES,
    TOKENS,
    V4_EVENT_BYTES,
    build_report,
    estimate_counts,
    load_workload,
    sha256,
    write_markdown,
)


class FormalPhaseArchiveScaleAuditTest(unittest.TestCase):
    def test_tile_expansion_and_template_formulas(self) -> None:
        counts = estimate_counts(
            np.asarray([10, 20]),
            np.asarray([4, 8]),
            np.asarray([6, 12]),
            np.asarray([2, 3]),
            np.asarray([3, 6]),
        )
        self.assertEqual(counts["destination_unique_items"], 30)
        self.assertEqual(counts["source_product_terms"], 12)
        self.assertEqual(counts["destination_deliveries"], 18)
        self.assertEqual(counts["active_records"], 5)
        self.assertEqual(counts["tile_expanded_product_terms"], 60)
        self.assertEqual(counts["tile_expanded_deliveries"], 90)
        self.assertEqual(counts["tile_expanded_records"], 24)
        self.assertEqual(counts["v4_main_expanded_events_excluding_common"], 262)
        self.assertEqual(counts["head_template_events"], 61)

    def test_invalid_shape_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "形状或范围"):
            estimate_counts(
                np.asarray([1, 2]),
                np.asarray([1, 2]),
                np.asarray([1, 2]),
                np.asarray([1]),
                np.asarray([3, 3]),
            )

    def test_patch_capacity_is_explicitly_excluded_from_base_template(self) -> None:
        counts = estimate_counts(
            np.asarray([10, 20]),
            np.asarray([4, 8]),
            np.asarray([6, 12]),
            np.asarray([2, 3]),
            np.asarray([3, 6]),
        )
        report = build_report({"fixture": True}, counts)
        storage = report["storage_model"]
        self.assertIn(
            "base_template_bytes_excluding_tile_patch_by_common_scenario", storage
        )
        self.assertNotIn("parameterized_template_bytes_by_common_scenario", storage)
        patch = storage["tile_patch_capacity_envelope"]
        self.assertEqual(patch["patch_target_events_excluding_common"], 262)
        self.assertEqual(patch["dense_cycle_only_uint32_bytes"], 262 * 4)
        self.assertEqual(patch["dense_cycle_identity_uint32_pair_bytes"], 262 * 8)
        self.assertIsNone(patch["sparse_patch_density"])
        self.assertEqual(report["formal_g0"], "DENY")
        self.assertEqual(len(report["model_source_bindings"]), 6)

    def test_common_scenarios_and_source_bindings_are_reproducible(self) -> None:
        counts = estimate_counts(
            np.asarray([1]),
            np.asarray([1]),
            np.asarray([2]),
            np.asarray([1]),
            np.asarray([3]),
        )
        report = build_report({"fixture": True}, counts)
        storage = report["storage_model"]
        common = storage["common_event_scenarios"]
        self.assertEqual(common["one_event_per_common_phase"], FORMAL_GROUPS * 2)
        self.assertEqual(common["vector_drain_450"], FORMAL_GROUPS * (1 + TOKENS))
        self.assertEqual(
            common["scalar_drain_450x32"], FORMAL_GROUPS * (1 + TOKENS * 32)
        )
        phase_bytes = FORMAL_PHASES * 11 + (FORMAL_PHASES + 1) * 8
        expected = (
            counts["v4_main_expanded_events_excluding_common"]
            + common["one_event_per_common_phase"]
        ) * V4_EVENT_BYTES + phase_bytes
        self.assertEqual(
            storage["v4_uncompressed_bytes_by_common_scenario"]
            ["one_event_per_common_phase"],
            expected,
        )
        for binding in report["model_source_bindings"]:
            path = Path(binding["file"])
            self.assertTrue(path.is_file())
            self.assertEqual(binding["sha256"], sha256(path))

    def test_markdown_forbids_complete_capacity_claim_without_patch(self) -> None:
        counts = estimate_counts(
            np.asarray([1]),
            np.asarray([1]),
            np.asarray([2]),
            np.asarray([1]),
            np.asarray([3]),
        )
        report = build_report({"fixture": True}, counts)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "report.md"
            write_markdown(path, report)
            text = path.read_text(encoding="utf-8")
        self.assertIn("不含 tile patch", text)
        self.assertIn("不是端到端存储缩减", text)
        self.assertNotIn("schema-minimum", text)

    def test_formal_profile_exactly_regenerates_frozen_artifacts(self) -> None:
        profile = Path(
            "results/local5_fullres_bb1e4_joint_heads_profile100_20260809"
        )
        frozen = Path("results/local5_formal_phase_archive_scale_audit_v3_20260811")
        if not profile.is_dir() or not frozen.is_dir():
            self.skipTest("正式 profile 或冻结审计产物不存在")
        bindings, counts = load_workload(profile.resolve())
        report = build_report(bindings, counts)
        expected_json = json.loads(
            (frozen / "phase_archive_scale_audit.json").read_text(encoding="utf-8")
        )
        self.assertEqual(report, expected_json)
        with TemporaryDirectory() as directory:
            regenerated = Path(directory) / "phase_archive_scale_audit.md"
            write_markdown(regenerated, report)
            self.assertEqual(
                regenerated.read_bytes(),
                (frozen / "phase_archive_scale_audit.md").read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
