from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


EXP_ROOT = Path(__file__).resolve().parents[1]


class H2EntrypointPatchTest(unittest.TestCase):
    def test_train_patch_matches_current_baseline(self) -> None:
        module_path = EXP_ROOT / "entrypoints" / "train.py"
        spec = importlib.util.spec_from_file_location("h2_train_entry", module_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        baseline_entry = EXP_ROOT.parents[1] / "third_party" / "SDformerFlow" / "train_flow_parallel_supervised_SNN.py"
        patched = module._patch_source(baseline_entry.read_text(), baseline_entry)

        self.assertIn("install_adaptive_ternary_qk", patched)
        self.assertIn("adaptive_ternary_regularization", patched)
        self.assertIn("[H2] adaptive ternary summary", patched)


if __name__ == "__main__":
    unittest.main()

