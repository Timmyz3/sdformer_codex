import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "explore_local5_affine_bank_pareto.py"
)
SPEC = importlib.util.spec_from_file_location(
    "explore_local5_affine_bank_pareto",
    MODULE_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Local5AffineBankParetoTest(unittest.TestCase):
    def test_five_bank_mapping_is_zero_replay(self):
        result = MODULE.explore(height=15, width=15, planes=2)
        row = result["best_by_bank_count"]["5"]
        self.assertEqual(row["maximum_replay"], 1)
        self.assertEqual(row["mean_replay"], 1.0)
        self.assertEqual(row["allocated_entries"], 450)

    def test_four_bank_mapping_requires_replay(self):
        result = MODULE.explore(height=15, width=15, planes=2)
        row = result["best_by_bank_count"]["4"]
        self.assertEqual(row["maximum_replay"], 2)
        self.assertGreater(row["mean_replay"], 1.0)

    def test_non_multiple_width_remains_capacity_accounted(self):
        result = MODULE.explore(height=3, width=4, planes=2)
        for row in result["best_by_bank_count"].values():
            self.assertGreaterEqual(row["allocated_entries"], 24)
            self.assertLessEqual(row["storage_utilization"], 1.0)


if __name__ == "__main__":
    unittest.main()
