import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_h67_system_dse_envelope.py"
SPEC = importlib.util.spec_from_file_location("h67_system_dse_envelope", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SystemDseEnvelopeTest(unittest.TestCase):
    def test_required_scale_hits_target(self):
        non_attention = 600_000_000
        fixed_attention = 4_000_000
        proposed_attention = 3_400_000
        target = 1.05
        scale = MODULE.minimum_non_attention_scale(
            non_attention, fixed_attention, proposed_attention, target
        )
        self.assertIsNotNone(scale)
        scaled = non_attention / scale
        observed = (scaled + fixed_attention) / (scaled + proposed_attention)
        self.assertAlmostEqual(observed, target)

    def test_attention_only_limit_is_unreachable(self):
        scale = MODULE.minimum_non_attention_scale(
            600_000_000, 4_000_000, 3_400_000, 4_000_000 / 3_400_000
        )
        self.assertIsNone(scale)

    def test_object_fit_spills_only_large_objects(self):
        rows = MODULE.object_fit_sweep(
            [("small", 64), ("medium", 128), ("large", 1024)], [128]
        )
        self.assertEqual(rows[0]["fit_objects"], 2)
        self.assertEqual(rows[0]["spill_objects"], 1)
        self.assertEqual(rows[0]["individually_fitting_payload_bytes"], 192)
        self.assertEqual(rows[0]["spill_read_write_bytes"], 2048)


if __name__ == "__main__":
    unittest.main()
