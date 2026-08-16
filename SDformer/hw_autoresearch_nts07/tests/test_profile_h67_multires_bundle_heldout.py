import unittest
from unittest.mock import patch

import numpy as np

from scripts.profile_h67_multires_bundle_heldout import (
    EXPECTED_NAMES,
    build_dataset,
    bundle_front_cycles,
    choose_best_size,
    decide_architecture_gate,
    evaluate_strategy,
    freeze_policy,
)


class H67MultiresBundleHeldoutTest(unittest.TestCase):
    def test_all_empty_front_is_group_count(self) -> None:
        active = np.zeros((1, 1, 225), dtype=np.bool_)
        self.assertEqual(int(bundle_front_cycles(active, 4)[0, 0]), 57)
        self.assertEqual(int(bundle_front_cycles(active, 8)[0, 0]), 29)
        self.assertEqual(int(bundle_front_cycles(active, 16)[0, 0]), 15)
        self.assertEqual(int(bundle_front_cycles(active, 32)[0, 0]), 8)

    def test_all_active_has_one_cycle_handoff_cost(self) -> None:
        active = np.ones((1, 1, 225), dtype=np.bool_)
        for size in (4, 8, 16, 32):
            self.assertEqual(int(bundle_front_cycles(active, size)[0, 0]), 226)

    def test_tie_prefers_narrower_selector(self) -> None:
        self.assertEqual(
            choose_best_size({4: 100, 8: 90, 16: 90, 32: 90}), 8
        )

    def test_contract_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "冻结DSE集合"):
            bundle_front_cycles(np.zeros((1, 1, 225), dtype=np.bool_), 64)
        with self.assertRaisesRegex(ValueError, "候选bundle集合"):
            choose_best_size({4: 1, 8: 2})

    def test_random_middle_masks_are_monotonic(self) -> None:
        rng = np.random.default_rng(20260809)
        active = rng.random((7, 11, 225)) < 0.37
        values = {size: bundle_front_cycles(active, size) for size in (4, 8, 16, 32)}
        self.assertTrue(np.all(values[8] <= values[4]))
        self.assertTrue(np.all(values[16] <= values[8]))
        self.assertTrue(np.all(values[32] <= values[16]))

    def test_policy_freeze_does_not_read_heldout_choice(self) -> None:
        dataset = []
        for name in EXPECTED_NAMES:
            stage = int(name[1])
            block = int(name.split(".B", 1)[1].split(".", 1)[0])
            dataset.append(
                {
                    "sample": 0,
                    "stage": stage,
                    "block": block,
                    "name": name,
                    "cycles": {
                        4: np.array([[120]]), 8: np.array([[80]]),
                        16: np.array([[90]]), 32: np.array([[100]]),
                    },
                    "baseline": np.array([[200]]),
                }
            )
            dataset.append(
                {
                    "sample": 50,
                    "stage": stage,
                    "block": block,
                    "name": name,
                    "cycles": {
                        4: np.array([[130]]), 8: np.array([[110]]),
                        16: np.array([[70]]), 32: np.array([[10]]),
                    },
                    "baseline": np.array([[200]]),
                }
            )
        policy = freeze_policy(dataset, {0})
        self.assertEqual(policy["global_size"], 8)
        self.assertEqual(set(policy["stage_sizes"].values()), {8})
        self.assertEqual(set(policy["block_sizes"].values()), {8})
        heldout = evaluate_strategy(dataset, {50}, "global_static", policy)
        self.assertEqual(heldout["cycles"], len(EXPECTED_NAMES) * 110)

    def test_architecture_gate_branches(self) -> None:
        def comparison(gain: float, p95: float = 0.0, p99: float = 0.0):
            return {
                "block_vs_global": {
                    "cycle_reduction": gain,
                    "row_p95_reduction": p95,
                    "row_p99_reduction": p99,
                }
            }

        self.assertEqual(
            decide_architecture_gate(comparison(0.049))["status"],
            "REJECT_AS_PARAMETER_DSE",
        )
        self.assertEqual(
            decide_architecture_gate(comparison(0.051, p99=-0.001))["status"],
            "REJECT_TAIL_REGRESSION",
        )
        self.assertEqual(
            decide_architecture_gate(comparison(0.051))["status"],
            "WAIT_CANONICAL_RTL_AND_PHYSICAL",
        )

    def test_trace_exact_flag_fail_closed(self) -> None:
        metrics = {
            "backend_cycles": np.zeros((1, 1), dtype=np.int64),
            "preload_cycles": np.full((1, 1), 225, dtype=np.int64),
            "ttb_e2e_cycles": np.zeros((1, 1), dtype=np.int64),
            "baseline_e2e_cycles": np.zeros((1, 1), dtype=np.int64),
        }
        checks = {
            "k_count": np.zeros((2, 1, 1, 225), dtype=np.int32),
            "ttb_active_trace_exact": False,
            "ttb_k_trace_exact": True,
            "ttb_motion_trace_exact": True,
        }
        with (
            patch(
                "scripts.profile_h67_multires_bundle_heldout.decode_record",
                return_value=(metrics, checks),
            ),
            patch(
                "scripts.profile_h67_multires_bundle_heldout.block_identity",
                return_value=(0, 0, "S0.B0.attn"),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "ttb_active_trace_exact"):
                build_dataset([{"sample_id": 0, "name": "S0.B0.attn"}])


if __name__ == "__main__":
    unittest.main()
