import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_h67_dual_line_operator_trace.py"
SPEC = importlib.util.spec_from_file_location("h67_dual_line_operator_trace_analysis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def row(**updates):
    value = {
        "status": MODULE.PASS,
        "name": "linear",
        "operator": "Linear",
        "scope": "encoder",
        "operator_call_index": "0",
        "temporal_step": "1",
        "state_valid": "True",
        "selector_rows": "2",
        "motion_selected_rows": "1",
        "local_selected_rows": "1",
        "local_work": "30",
        "motion_work": "24",
        "selected_work": "18",
        "selector_saved_work": "12",
        "output_channel_fanout": "6",
        "current_source_count": "5",
        "positive_transition_source_count": "2",
        "negative_transition_source_count": "2",
        "valid_source_work": "60",
    }
    value.update(updates)
    return value


class DualLineOperatorTraceAnalysisTest(unittest.TestCase):
    def test_selector_can_beat_both_aggregate_paths(self):
        result = MODULE.summarize_pass_rows([row()])
        self.assertEqual(result["selector_saved_work"], 12)
        self.assertAlmostEqual(result["local_to_selected_work_ratio"], 30 / 18)
        self.assertEqual(MODULE.audit_rows([row()]), [])

    def test_invalid_state_cannot_select_motion(self):
        errors = MODULE.audit_rows(
            [row(state_valid="False", motion_selected_rows="1")]
        )
        self.assertTrue(any("invalid prior state" in error for error in errors))

    def test_source_fanout_conservation_is_fail_closed(self):
        errors = MODULE.audit_rows([row(current_source_count="4")])
        self.assertTrue(any("Local source/fanout" in error for error in errors))

    def test_selected_work_must_be_fanout_aligned(self):
        errors = MODULE.audit_rows([row(selected_work="17", selector_saved_work="13")])
        self.assertTrue(any("not divisible by fanout" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
