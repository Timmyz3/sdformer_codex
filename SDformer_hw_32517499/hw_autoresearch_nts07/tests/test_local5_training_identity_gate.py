from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_local5_bb1e4_checkpoint_bound_rtl as rtl  # noqa: E402
import run_local5_bb1e4_postg0_profile as profile  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Local5TrainingIdentityGateTest(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[Path, Path, Path, dict]:
        config = root / "train.yml"
        state = root / "state.pth"
        identity = root / "training_config_identity.json"
        config.write_text("experiment: local5\n", encoding="utf-8")
        state.write_bytes(b"paired-state")
        value = {
            "schema": "local5_training_config_identity_v1",
            "status": "PASS",
            "authority": "ep9_optimizer_scheduler_state",
            "deterministic_regeneration_equal": True,
            "config_path": str(config.resolve()),
            "config_sha256": sha(config),
            "state_path": str(state.resolve()),
            "state_sha256": sha(state),
            "checks": {"scheduler": True, "optimizer": True},
        }
        identity.write_text(json.dumps(value), encoding="utf-8")
        return config, state, identity, value

    def test_profile_gate_accepts_bound_pass_and_rejects_state_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, state, identity, value = self.fixture(Path(temporary))
            with (
                patch.object(profile, "TRAINING_CONFIG", config),
                patch.object(profile, "TRAINING_IDENTITY", identity),
            ):
                self.assertEqual(profile.validate_training_identity(), value)
                state.write_bytes(b"drift")
                with self.assertRaisesRegex(RuntimeError, "state_sha"):
                    profile.validate_training_identity()

    def test_profile_gate_waits_for_pending_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, _, identity, value = self.fixture(Path(temporary))
            value["status"] = "PENDING_EP9_RUNTIME_STATE"
            identity.write_text(json.dumps(value), encoding="utf-8")
            with (
                patch.object(profile, "TRAINING_CONFIG", config),
                patch.object(profile, "TRAINING_IDENTITY", identity),
            ):
                self.assertIsNone(profile.validate_training_identity())

    def test_rtl_gate_requires_postg0_source_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _, identity, value = self.fixture(root)
            run_identity = root / "post_g0_run_identity.json"
            run_identity.write_text(
                json.dumps(
                    {
                        "source_bindings": {
                            "training_config_identity": {
                                "path": str(identity.resolve()),
                                "sha256": sha(identity),
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch.object(rtl, "TRAINING_CONFIG", config),
                patch.object(rtl, "TRAINING_IDENTITY", identity),
                patch.object(rtl, "RUN_IDENTITY", run_identity),
            ):
                self.assertEqual(rtl.validate_training_identity_binding(), value)
                run_identity.write_text('{"source_bindings": {}}', encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "not bound"):
                    rtl.validate_training_identity_binding()


if __name__ == "__main__":
    unittest.main()
