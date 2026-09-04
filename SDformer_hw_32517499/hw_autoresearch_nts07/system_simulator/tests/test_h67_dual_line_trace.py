import importlib.util
from pathlib import Path
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


MODULE = None
if torch is not None:
    SCRIPT = (
        Path(__file__).resolve().parents[3]
        / "neuron_experiments/H9_bipolar_self_attention/entrypoints/h67_dual_line_trace.py"
    )
    SPEC = importlib.util.spec_from_file_location("h67_dual_line_trace", SCRIPT)
    MODULE = importlib.util.module_from_spec(SPEC)
    assert SPEC.loader is not None
    SPEC.loader.exec_module(MODULE)


@unittest.skipUnless(torch is not None, "algorithm-side torch is optional on hardware server")
class DualLineTraceTest(unittest.TestCase):
    def test_linear_uses_motion_only_when_row_transition_is_smaller(self):
        linear = torch.nn.Linear(4, 3, bias=False)
        value = torch.tensor(
            [
                [[1, 1, 1, 0], [0, 0, 0, 0]],
                [[1, 1, 0, 0], [1, 0, 0, 0]],
            ],
            dtype=torch.float32,
        )
        rows = MODULE.profile_operator_temporal_work(
            linear, value, temporal_steps=2
        )
        self.assertEqual(rows[0]["local_work"], 9)
        self.assertEqual(rows[0]["selected_work"], 9)
        self.assertEqual(rows[1]["local_work"], 9)
        self.assertEqual(rows[1]["motion_work"], 6)
        self.assertEqual(rows[1]["selected_work"], 6)
        self.assertEqual(rows[1]["motion_selected_rows"], 1)

    def test_conv2d_counts_padding_geometry_exactly(self):
        conv = torch.nn.Conv2d(1, 2, 3, padding=1, bias=False)
        value = torch.zeros((2, 1, 2, 2), dtype=torch.float32)
        value[0, 0, 0, 0] = 1
        value[1, 0, 0, 0] = 1
        rows = MODULE.profile_operator_temporal_work(conv, value, temporal_steps=2)
        # One corner input contributes to all four 2x2 outputs and two channels.
        self.assertEqual(rows[0]["local_work"], 8)
        self.assertEqual(rows[1]["motion_work"], 0)
        self.assertEqual(rows[1]["selected_work"], 0)

    def test_nonbinary_input_is_explicit_bypass(self):
        linear = torch.nn.Linear(2, 2, bias=False)
        rows = MODULE.profile_operator_temporal_work(
            linear, torch.tensor([[[0.0, -1.0]], [[0.0, 1.0]]]), temporal_steps=2
        )
        self.assertEqual(rows[0]["status"], "NON_BINARY_BYPASS")


if __name__ == "__main__":
    unittest.main()
