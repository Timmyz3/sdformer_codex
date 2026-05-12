from __future__ import annotations

import unittest
from pathlib import Path


class E5BEntrypointPatchTest(unittest.TestCase):
    def test_train_entrypoint_patches_official_tsn_training_protocol(self) -> None:
        train_py = Path(__file__).resolve().parents[1] / "entrypoints" / "train.py"
        source = train_py.read_text()

        self.assertIn("official_tsn_split_weights", source)
        self.assertIn("torch.optim.SGD", source)
        self.assertIn("CosineAnnealingLR", source)
        self.assertIn("[E5b] official TSN split_weights", source)


if __name__ == "__main__":
    unittest.main()
