#!/usr/bin/env python3
"""Local5 原位跨头累加 SRAM 组织模型单元测试。"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "model_local5_inplace_acc_sram",
    ROOT / "scripts" / "model_local5_inplace_acc_sram.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class StorageModelTest(unittest.TestCase):
    def test_macro_tiling_and_port_contract(self) -> None:
        model = MODULE.build_model()
        mapping = model["mappings"]
        self.assertEqual(
            mapping["tcfm5_to_target_1r1w"]["macro_count"], 20
        )
        self.assertEqual(
            mapping["scalar_to_local_1rw"]["macro_count"], 57
        )
        self.assertFalse(mapping["tcfm5_to_local_1rw"]["supported"])

    def test_candidate_delta(self) -> None:
        model = MODULE.build_model()
        candidates = model["candidates"]
        self.assertEqual(
            candidates["B0_scalar_recompute"]["logical_acc_bits"], 921600
        )
        self.assertEqual(
            candidates["B2_inplace_recompute"]["logical_acc_bits"], 460800
        )
        self.assertEqual(model["deltas"]["deleted_scalar_transactions"], 259200)


if __name__ == "__main__":
    unittest.main()
