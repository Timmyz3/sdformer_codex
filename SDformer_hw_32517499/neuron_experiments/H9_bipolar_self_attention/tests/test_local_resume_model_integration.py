from __future__ import annotations

import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[3]
UPSTREAM = REPO / "third_party/SDformerFlow"
MODULES_BEFORE_IMPORT = set(sys.modules)
sys.path.insert(0, str(UPSTREAM))
try:
    from utils.utils import resume_model  # noqa: E402
finally:
    sys.path.remove(str(UPSTREAM))
    for module_name in set(sys.modules).difference(MODULES_BEFORE_IMPORT):
        if module_name == "models" or module_name.startswith(("models.", "utils")):
            sys.modules.pop(module_name, None)


class DummyScaler:
    def __init__(self) -> None:
        self.scale = 1.0

    def state_dict(self) -> dict[str, float]:
        return {"scale": self.scale}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.scale = float(state["scale"])


class LocalResumeModelIntegrationTest(unittest.TestCase):
    def test_restores_paired_state_and_advances_to_next_epoch(self) -> None:
        expected_lrs = [1e-4, 1e-4, 5e-5, 5e-5, 5e-6]
        parameters = [torch.nn.Parameter(torch.tensor(float(index))) for index in range(5)]
        optimizer = torch.optim.AdamW(
            [{"params": [parameter], "lr": lr} for parameter, lr in zip(parameters, expected_lrs)]
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[13, 20], gamma=0.5
        )
        scheduler.last_epoch = 9
        scheduler._last_lr = list(expected_lrs)
        scaler = DummyScaler()
        scaler.scale = 65536.0

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint_epoch9.pth"
            checkpoint.write_bytes(b"model fixture")
            state_path = Path(temporary) / "checkpoint_epoch9_state_dict.pth"
            torch.save(
                {
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": 9,
                    "scaler": scaler.state_dict(),
                },
                state_path,
            )

            for group in optimizer.param_groups:
                group["lr"] = 1.0
            scheduler.last_epoch = 0
            scheduler.milestones = Counter({1: 1})
            scaler.scale = 1.0

            optimizer, scheduler, scaler, epoch_initial = resume_model(
                str(checkpoint), optimizer, scheduler, scaler, 0, torch.device("cpu")
            )

        self.assertEqual(epoch_initial, 10)
        self.assertEqual([group["lr"] for group in optimizer.param_groups], expected_lrs)
        self.assertEqual(scheduler.last_epoch, 9)
        self.assertEqual(dict(scheduler.milestones), {13: 1, 20: 1})
        self.assertEqual(scheduler._last_lr, expected_lrs)
        self.assertEqual(scaler.scale, 65536.0)


if __name__ == "__main__":
    unittest.main()
