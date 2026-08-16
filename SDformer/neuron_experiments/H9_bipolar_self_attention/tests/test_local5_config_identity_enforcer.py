from pathlib import Path
import hashlib
import json
import sys
import tempfile
from unittest.mock import patch


ENTRYPOINTS = Path(__file__).resolve().parents[1] / "entrypoints"
sys.path.insert(0, str(ENTRYPOINTS))

from enforce_local5_ep9_config_identity_20260805 import (  # noqa: E402
    EXPECTED_MILESTONES,
    repair_scheduler_state,
    repairable_milestone_only,
    state_checks,
    state_facts,
)
import enforce_local5_ep9_config_identity_20260805 as enforcer  # noqa: E402


def payload(milestones: dict[int, int]) -> dict:
    lrs = [1e-4, 1e-4, 5e-5, 5e-5, 5e-6]
    return {
        "epoch": 9,
        "optimizer": {"param_groups": [{"lr": value} for value in lrs]},
        "scheduler": {"last_epoch": 9, "milestones": milestones, "_last_lr": lrs},
        "scaler": {"scale": 65536.0},
    }


def test_accepts_expected_runtime_state() -> None:
    checks = state_checks(state_facts(payload(EXPECTED_MILESTONES)))
    assert all(checks.values())
    assert not repairable_milestone_only(checks)


def test_repairs_only_stale_milestones() -> None:
    state = payload({12: 1, 20: 1})
    checks = state_checks(state_facts(state))
    assert repairable_milestone_only(checks)
    repair_scheduler_state(state)
    assert all(state_checks(state_facts(state)).values())


def test_rejects_non_milestone_drift() -> None:
    state = payload({12: 1, 20: 1})
    state["optimizer"]["param_groups"][0]["lr"] = 2e-4
    assert not repairable_milestone_only(state_checks(state_facts(state)))


def test_active_launch_binding_rejects_stale_source_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        config = root / "config.yml"
        source = (
            root
            / "results/h66d_allbinary_all12_lr_ttx_w720_fastlr_full30_bs8_full30_20260712_setsid/checkpoint_epoch29.pth"
        )
        report = root / "active_launch.json"
        config.write_bytes(b"config")
        source.parent.mkdir(parents=True)
        source.write_bytes(b"source")
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        report.write_text(
            json.dumps(
                {
                    "schema": "local5_active_launch_provenance_v1",
                    "status": "PASS_ACTIVE_CAPTURE",
                    "scope": "launch-only",
                    "active_train": {
                        "pid": 7,
                        "start_utc": "2026-08-05T00:00:00+00:00",
                        "start_epoch": 1.0,
                    },
                    "checks": {"one": True},
                    "artifact_identity": {
                        "config_sha256_at_capture": digest(config),
                        "source_checkpoint_sha256": digest(source),
                    },
                }
            ),
            encoding="utf-8",
        )
        with (
            patch.object(enforcer, "CONFIG", config),
            patch.object(enforcer, "EXP", root),
            patch.object(enforcer, "ACTIVE_LAUNCH_REPORT", report),
        ):
            assert enforcer.active_launch_binding()["train_pid_at_capture"] == 7
            source.write_bytes(b"stale")
            try:
                enforcer.active_launch_binding()
            except RuntimeError as error:
                assert "source_checkpoint_sha" in str(error)
            else:
                raise AssertionError("stale source checkpoint was accepted")
