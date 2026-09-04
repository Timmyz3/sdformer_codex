from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "entrypoints/train.py"
SPEC = importlib.util.spec_from_file_location("h9_train_entrypoint", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TrainPathArgsTest(unittest.TestCase):
    def test_only_file_arguments_are_absolutized(self) -> None:
        normalized = MODULE._absolutize_path_args(
            [
                "--prev_runid",
                "relative/checkpoint.pth",
                "--save_path=relative/checkpoint_epoch{}.pth",
                "--resume",
                "1",
                "--finetune=1",
            ]
        )
        self.assertTrue(Path(normalized[1]).is_absolute())
        self.assertTrue(Path(normalized[2].split("=", 1)[1]).is_absolute())
        self.assertEqual(normalized[3:5], ["--resume", "1"])
        self.assertEqual(normalized[5], "--finetune=1")


if __name__ == "__main__":
    unittest.main()
