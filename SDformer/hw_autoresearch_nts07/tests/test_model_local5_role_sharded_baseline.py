import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "model_local5_role_sharded_baseline.py"
)
SPEC = importlib.util.spec_from_file_location(
    "model_local5_role_sharded_baseline",
    MODULE_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Local5RoleShardedBaselineTest(unittest.TestCase):
    def test_deployment_window_boundary_compression(self):
        result = MODULE.evaluate(
            height=15,
            width=15,
            planes=2,
            out_dim=4,
            acc_width=32,
        )
        role = result["role_sharded_compressed_boundary"]
        self.assertEqual(result["tokens"], 450)
        self.assertEqual(role["acc_entries"], 2130)
        self.assertEqual(role["final_vector_adds"], 1680)
        self.assertAlmostEqual(role["entry_ratio_vs_tcfm5"], 2130 / 450)

    def test_single_token_has_only_self_role(self):
        result = MODULE.evaluate(
            height=1,
            width=1,
            planes=2,
            out_dim=4,
            acc_width=32,
        )
        role = result["role_sharded_compressed_boundary"]
        self.assertEqual(role["acc_entries"], 2)
        self.assertEqual(role["final_vector_adds"], 0)


if __name__ == "__main__":
    unittest.main()
