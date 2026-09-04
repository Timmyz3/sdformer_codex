#!/usr/bin/env python3
"""Fail-closed checks for the M26 factor arithmetic lower-bound evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
GENERATOR = (
    REPO_ROOT
    / "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "audit_m26_atlif_low_rank_hardware.py"
)
CANONICAL = (
    REPO_ROOT
    / "hw_autoresearch_nts07/results/"
    "m26_atlif_factor_arithmetic_lower_bound_r5_receipted_20260822/"
    "m26_atlif_factor_arithmetic_lower_bound.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_generator():
    entrypoint_dir = str(GENERATOR.parent)
    if entrypoint_dir not in sys.path:
        sys.path.insert(0, entrypoint_dir)
    spec = importlib.util.spec_from_file_location("m26_factor_audit", GENERATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load M26 generator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class M26FactorArithmeticLowerBoundTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_generator()
        cls.payload = json.loads(CANONICAL.read_text(encoding="utf-8"))

    def test_discrete_factor_and_fallback_schedule(self) -> None:
        self.assertEqual(
            self.module.factor_service_cycles(16, 10, 3, 96, 16),
            (10, "TWO_STAGE_FACTOR_TILE", 1),
        )
        self.assertEqual(
            self.module.factor_service_cycles(16, 10, 2, 96, 16),
            (8, "TWO_STAGE_FACTOR_TILE", 1),
        )
        self.assertEqual(
            self.module.factor_service_cycles(16, 10, 5, 96, 16),
            (17, "DENSE_FALLBACK", 0),
        )

    def test_tail_tile_is_separately_rounded(self) -> None:
        self.assertEqual(
            self.module.factor_service_cycles(17, 10, 3, 96, 16),
            (12, "TWO_STAGE_FACTOR_TILE", 2),
        )

    def test_generator_and_execution_partition_are_bound(self) -> None:
        self.assertEqual(
            self.payload["identity"]["generator_sha256"], sha256(GENERATOR)
        )
        partition = self.payload["checkpoint_matrix_census"]["execution_partition"]
        self.assertEqual(
            (
                partition["live_modules"],
                partition["deployment_dead_modules"],
                partition["checkpoint_installed_but_uncalled_modules"],
            ),
            (81, 12, 12),
        )
        self.assertEqual(len(partition["uncalled_names"]), 12)
        self.assertEqual(len(partition["deployment_dead_names"]), 12)
        self.assertTrue(
            all(name.endswith(".sn2_q.spiking_neuron") for name in partition["uncalled_names"])
        )
        self.assertTrue(
            all(
                name.endswith(".attn.attn_sn.spiking_neuron")
                for name in partition["deployment_dead_names"]
            )
        )

    def test_all_local_identity_inputs_match_their_hashes(self) -> None:
        identity = self.payload["identity"]
        for field in ("config", "checkpoint", "profile", "m25"):
            path = REPO_ROOT / identity[field]
            self.assertTrue(path.is_file(), str(path))
            self.assertEqual(identity[field + "_sha256"], sha256(path))
        profiler = GENERATOR.parent / "profile_nts11_hardware_p0.py"
        self.assertEqual(identity["profiler_sha256"], sha256(profiler))

    def test_rank2_two_cycle_per_tile_lower_bound_remains_above_two_x(self) -> None:
        rank2 = next(row for row in self.payload["candidates"] if row["rank"] == 2)
        sensitivity = {
            row["overhead_cycles_per_factor_tile"]: row
            for row in rank2["cycle_per_factor_tile_overhead_sensitivity"]
        }
        self.assertTrue(sensitivity[2]["crosses_2x_local"])
        self.assertTrue(sensitivity[2]["crosses_2x_motion"])
        self.assertGreater(sensitivity[2]["local_speedup_vs_fixed"], 2.0)
        self.assertGreater(sensitivity[2]["motion_speedup_vs_fixed"], 2.0)

    def test_rank3_one_cycle_per_tile_revokes_two_x(self) -> None:
        rank3 = next(row for row in self.payload["candidates"] if row["rank"] == 3)
        self.assertEqual(rank3["factor_tiles"], 7_318_350)
        sensitivity = {
            row["overhead_cycles_per_factor_tile"]: row
            for row in rank3["cycle_per_factor_tile_overhead_sensitivity"]
        }
        self.assertTrue(sensitivity[0]["crosses_2x_local"])
        self.assertFalse(sensitivity[1]["crosses_2x_local"])
        self.assertFalse(sensitivity[1]["crosses_2x_motion"])
        self.assertFalse(rank3["precision_contract"]["same_resource_cycle_point_admitted"])
        self.assertFalse(rank3["headline_admitted"])

    def test_rank3_tile_resident_state_and_materialization_penalty(self) -> None:
        rank3 = next(row for row in self.payload["candidates"] if row["rank"] == 3)
        state = rank3["factor_state_contract"]
        self.assertEqual(state["minimum_tile_resident_intermediate_bytes_q24"], 144)
        self.assertEqual(state["minimum_double_buffered_intermediate_bytes_q24"], 288)
        self.assertEqual(
            state["external_write_plus_read_bytes_if_q24_materialized"],
            2_107_684_800,
        )
        self.assertFalse(state["tile_resident_bank_port_rtl_frozen"])


if __name__ == "__main__":
    unittest.main()
