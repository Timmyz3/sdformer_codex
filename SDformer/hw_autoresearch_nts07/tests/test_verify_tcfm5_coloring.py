import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "verify_tcfm5_coloring.py"
)
SPEC = importlib.util.spec_from_file_location("verify_tcfm5_coloring", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Tcfm5ColoringTest(unittest.TestCase):
    def test_deployment_window_is_conflict_free_and_minimal(self):
        result = MODULE.verify(height=15, width=15, planes=2)
        self.assertEqual(result["tokens"], 450)
        self.assertTrue(result["injective_bank_address"])
        self.assertTrue(result["conflict_free_all_neighborhoods"])
        self.assertEqual(result["minimum_banks_for_one_cycle"], 5)
        self.assertEqual(result["interior_k5_witnesses"], 338)

    def test_non_multiple_of_five_width_remains_injective(self):
        result = MODULE.verify(height=3, width=4, planes=2)
        self.assertEqual(result["tokens"], 24)
        self.assertTrue(result["injective_bank_address"])
        self.assertTrue(result["conflict_free_all_neighborhoods"])


if __name__ == "__main__":
    unittest.main()
