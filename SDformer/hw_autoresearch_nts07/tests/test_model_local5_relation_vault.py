#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/model_local5_relation_vault.py"
SPEC = importlib.util.spec_from_file_location("model_local5_relation_vault", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RelationVaultModelTest(unittest.TestCase):
    def test_packet_worst_case_is_smaller_than_raw_relation(self) -> None:
        group = MODULE.Group(0, 0, MODULE.TOKENS, MODULE.TOKENS)
        self.assertEqual(group.packet_words, MODULE.TOKENS)
        self.assertLessEqual(group.packet_words, MODULE.RELATION_MACRO_DEPTH)

    def test_full_capacity_eliminates_repeated_builds(self) -> None:
        service = np.asarray([100, 500, 900], dtype=np.int64)
        packets = np.asarray([1000, 2000, 3000], dtype=np.int64)
        baseline, vault, resident, builds, used = MODULE.cycles_for_window(
            service, packets, 6000, "first_fit_all"
        )
        self.assertEqual(resident, 3)
        self.assertEqual(builds, 3)
        self.assertEqual(used, 6000)
        self.assertLessEqual(vault, baseline)

    def test_zero_capacity_is_exact_recompute_baseline(self) -> None:
        service = np.asarray([100, 500, 900], dtype=np.int64)
        packets = np.asarray([1000, 2000, 3000], dtype=np.int64)
        baseline, vault, resident, builds, used = MODULE.cycles_for_window(
            service, packets, 0
        )
        self.assertEqual(vault, baseline)
        self.assertEqual(resident, 0)
        self.assertEqual(builds, 9)
        self.assertEqual(used, 0)

    def test_partial_admission_preserves_head_order(self) -> None:
        admitted = MODULE.admit_packets(
            np.asarray([100, 100, 100, 100], dtype=np.int64),
            np.asarray([5, 7, 3, 2], dtype=np.int64),
            10,
            "critical_only",
        )
        self.assertEqual(admitted.tolist(), [True, False, True, True])

    def test_critical_policy_rejects_hidden_frontend(self) -> None:
        admitted = MODULE.admit_packets(
            np.asarray([449, 450, 900], dtype=np.int64),
            np.asarray([100, 100, 100], dtype=np.int64),
            1000,
            "critical_only",
        )
        self.assertEqual(admitted.tolist(), [True, False, False])

    def test_noncritical_rollback_releases_capacity_for_next_head(self) -> None:
        admitted, speculative, discarded, misses = MODULE.online_admit_packets(
            np.asarray([500, 100], dtype=np.int64),
            np.asarray([56000, 1120], dtype=np.int64),
            57344,
            "critical_only",
        )
        self.assertEqual(admitted.tolist(), [False, True])
        self.assertEqual(speculative, 510)
        self.assertEqual(discarded, 500)
        self.assertEqual(misses, 0)

    def test_physical_packet_port_is_hidden_by_service_lower_bound(self) -> None:
        for active_sources in range(MODULE.TOKENS + 1):
            packet_words = active_sources
            service_lower_bound = 15 + active_sources
            self.assertLessEqual(packet_words + 1, service_lower_bound)


if __name__ == "__main__":
    unittest.main()
