import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from utils import utils


class LocalLoadModelTest(unittest.TestCase):
    def test_v1_remap_applies_interpolated_state_dict(self):
        source = torch.nn.Linear(2, 2)
        target = torch.nn.Linear(2, 2)
        with torch.no_grad():
            source.weight.fill_(3.0)
            source.bias.fill_(-2.0)
            target.weight.zero_()
            target.bias.zero_()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pth"
            torch.save({"model_state_dict": source.state_dict()}, checkpoint)
            with mock.patch.object(utils, "load_pretrained_interpolate") as interpolate:
                loaded = utils.load_model(
                    str(checkpoint),
                    target,
                    torch.device("cpu"),
                    remap="v1",
                )

        interpolate.assert_called_once()
        self.assertTrue(torch.equal(loaded.weight, source.weight))
        self.assertTrue(torch.equal(loaded.bias, source.bias))


if __name__ == "__main__":
    unittest.main()
