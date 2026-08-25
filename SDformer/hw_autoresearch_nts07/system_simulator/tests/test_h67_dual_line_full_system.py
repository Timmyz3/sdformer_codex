import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_h67_dual_line_full_system.py"
SPEC = importlib.util.spec_from_file_location("h67_dual_line_full_system", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def operator(**updates):
    row = {
        "name": "op",
        "operator": "Linear",
        "category": "ffn_expand",
        "activity_cycles_at_config_lanes": "40",
        "input_activity": "0.25",
        "input_binary_packed_eligible": "True",
        "replaced_by_attention_rtl_anchor": "False",
    }
    row.update(updates)
    return row


class DualLineFullSystemTest(unittest.TestCase):
    def test_motion_delta_is_bit_exact_for_signed_int8_weights(self):
        weights = [[-3, 2, 7, -1], [5, -8, 1, 4]]
        previous = [0, 1, 1, 0]
        current = [1, 1, 0, 1]
        previous_output = MODULE.dense_binary_accumulate(weights, previous)
        updated = MODULE.motion_delta_accumulate(
            weights, previous, current, previous_output
        )
        self.assertEqual(updated, MODULE.dense_binary_accumulate(weights, current))

    def test_only_binary_nonattention_linear_or_conv_routes_to_both_paths(self):
        self.assertTrue(MODULE.eligible(operator()))
        self.assertTrue(MODULE.eligible(operator(operator="Conv2d")))
        self.assertFalse(MODULE.eligible(operator(operator="Conv3d")))
        self.assertFalse(
            MODULE.eligible(operator(input_binary_packed_eligible="False"))
        )
        self.assertFalse(
            MODULE.eligible(operator(replaced_by_attention_rtl_anchor="True"))
        )

    def test_window_reshaped_attention_qk_is_local_only(self):
        routed = MODULE.route_operator(operator(category="attention_q_projection"))
        self.assertTrue(routed["local_eligible"])
        self.assertFalse(routed["motion_eligible"])
        self.assertEqual(routed["motion_path"], "LOCAL_ONLY_WINDOW_RESHAPED")

    def test_build_is_fail_closed_without_ordered_tile_trace(self):
        result = MODULE.build(
            {
                "status": MODULE.LEDGER_STATUS,
                "attention": {
                    "fixed_cycles_per_frame": 10,
                    "rqtb_cycles_per_frame": 8,
                },
                "cycles_per_frame_model": {"fixed_total": 100},
            },
            [operator(activity_cycles_at_config_lanes="60")],
            {
                "targets": [2.0],
                "local_component_anchor": {"speedup": 2.0},
                "selector": {},
            },
            {
                "full_network_execution_trace": {"available": False},
                "full_network_dual_line_operator_trace": {"available": False},
            },
        )
        self.assertEqual(result["coverage"]["eligible_cycles"], 60)
        self.assertTrue(result["status"].endswith("TRACE_TIMING_BLOCKED"))
        target = result["envelopes"]["targets"][0]
        self.assertAlmostEqual(target["minimum_dual_line_eligible_engine_speedup"], 5.0)


if __name__ == "__main__":
    unittest.main()
