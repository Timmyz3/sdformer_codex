from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from model_vl_gs_ttb_dual_line import motion_eager_header


class VlGsTtbDualLineTest(unittest.TestCase):
    def test_motion_eager_header_exact_fallback(self) -> None:
        value = motion_eager_header([0, 2, 5], [0, 8, 10], 4)
        self.assertEqual(value["active_contexts"], 2)
        self.assertEqual(value["fallback_contexts"], 1)
        self.assertEqual(value["fallback_terms"], 10)
        self.assertEqual(value["baseline_gate_key_bits"], 162)
        # fast: mode1 + count3 + header18 + body16; raw: mode1 + body90
        self.assertEqual(value["vl_gs_ttb_gate_key_bits"], 129)

    def test_rejects_invalid_class_term_relation(self) -> None:
        with self.assertRaisesRegex(ValueError, "不守恒"):
            motion_eager_header([3], [2], 4)


if __name__ == "__main__":
    unittest.main()
