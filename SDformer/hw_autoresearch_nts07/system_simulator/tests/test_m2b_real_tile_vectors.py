import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_m2b_real_tile_vectors.py"
SPEC = importlib.util.spec_from_file_location("m2b_real_tile_vectors", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class M2BRealTileVectorTest(unittest.TestCase):
    def test_uniform_selection_is_deterministic_and_complete(self):
        self.assertEqual(MODULE.select_indices(5, 5), [0, 1, 2, 3, 4])
        self.assertEqual(MODULE.select_indices(10, 4), [0, 3, 6, 9])
        self.assertEqual(MODULE.select_indices(10, 4), MODULE.select_indices(10, 4))

    def test_uniform_selection_rejects_bad_cardinality(self):
        with self.assertRaises(ValueError):
            MODULE.select_indices(0, 1)
        with self.assertRaises(ValueError):
            MODULE.select_indices(1, 0)


if __name__ == "__main__":
    unittest.main()
