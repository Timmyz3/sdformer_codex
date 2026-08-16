from __future__ import annotations

import importlib.util
import unittest
from collections import Counter
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "entrypoints/audit_local5_ep9_checkpoint_20260805.py"
)
SPEC = importlib.util.spec_from_file_location("local5_ep9_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def state_payload(*, epoch: int = 9, last_epoch: int = 9) -> dict:
    lrs = [1e-4, 1e-4, 5e-5, 5e-5, 5e-6]
    return {
        "epoch": epoch,
        "optimizer": {"param_groups": [{"lr": value} for value in lrs]},
        "scheduler": {
            "last_epoch": last_epoch,
            "milestones": Counter({13: 1, 20: 1}),
            "_last_lr": lrs,
        },
        "scaler": {"scale": 65536.0},
    }


class Local5Ep9CheckpointAuditTest(unittest.TestCase):
    def test_accepts_expected_resume_state(self) -> None:
        facts, checks = MODULE.validate_state(state_payload())
        self.assertEqual(facts["scheduler_milestones"], {13: 1, 20: 1})
        self.assertTrue(all(checks.values()), checks)

    def test_rejects_epoch_drift(self) -> None:
        _, checks = MODULE.validate_state(state_payload(epoch=8, last_epoch=8))
        self.assertFalse(checks["state_epoch9"])
        self.assertFalse(checks["scheduler_epoch9"])

    def test_rejects_lr_drift(self) -> None:
        payload = state_payload()
        payload["optimizer"]["param_groups"][4]["lr"] = 1e-5
        _, checks = MODULE.validate_state(payload)
        self.assertFalse(checks["optimizer_five_group_lrs"])


if __name__ == "__main__":
    unittest.main()
