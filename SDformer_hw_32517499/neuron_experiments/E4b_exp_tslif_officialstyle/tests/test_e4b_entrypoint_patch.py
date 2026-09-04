from __future__ import annotations

import unittest
from pathlib import Path


class E4BEntrypointPatchTest(unittest.TestCase):
    def test_train_entrypoint_patches_official_style_optimizer(self) -> None:
        train_py = Path(__file__).resolve().parents[1] / "entrypoints" / "train.py"
        source = train_py.read_text()

        self.assertIn("tslif_lr", source)
        self.assertIn("spiking_neuron.core.alpha_s", source)
        self.assertIn("spiking_neuron.core.decay_factor", source)
        self.assertIn("[E4b] official-style optimizer groups", source)


if __name__ == "__main__":
    unittest.main()
