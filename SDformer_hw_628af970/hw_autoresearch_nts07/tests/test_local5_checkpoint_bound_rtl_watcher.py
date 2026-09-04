from __future__ import annotations

import json
import inspect
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_local5_bb1e4_checkpoint_bound_rtl as watcher


def test_atlif_lock_precedes_score_and_projection_vector_generation() -> None:
    source = inspect.getsource(watcher.main)
    atlif = source.index("atlif_result = run_atlif_checkpoint_replay()")
    score = source.index("score_env = os.environ.copy()")
    projection = source.index("generate_local5_active_projection_postg0_vectors.py")
    assert atlif < score < projection


def test_ordered_source_manifest_binds_checkpoint_and_config() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "ordered.json"
        source.write_text(
            json.dumps(
                {
                    "checkpoint_sha256": "a" * 64,
                    "config_sha256": "b" * 64,
                }
            ),
            encoding="utf-8",
        )
        vector = {
            "source_manifest": str(source),
            "source_manifest_sha256": watcher.file_sha256(source),
        }
        assert watcher.ordered_manifest_identity(vector) == {
            "checkpoint_sha256": "a" * 64,
            "config_sha256": "b" * 64,
        }
        value = json.loads(source.read_text(encoding="utf-8"))
        del value["config_sha256"]
        source.write_text(json.dumps(value), encoding="utf-8")
        vector["source_manifest_sha256"] = watcher.file_sha256(source)
        try:
            watcher.ordered_manifest_identity(vector)
        except RuntimeError as exc:
            assert "checkpoint/config" in str(exc)
        else:
            raise AssertionError("missing config SHA was accepted")


def test_missing_acceptance_runs_supervised_profile_producer() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        acceptance = Path(temporary) / "acceptance.json"

        def produce(*_args, **_kwargs) -> None:
            acceptance.write_text("{}\n", encoding="utf-8")

        with (
            patch.object(watcher, "ACCEPTANCE", acceptance),
            patch.object(
                watcher,
                "validate_profile_acceptance_binding",
                side_effect=[RuntimeError("missing"), ({}, Path(temporary) / "checkpoint.pth")],
            ),
            patch.object(watcher, "run", side_effect=produce) as run,
        ):
            watcher.ensure_profile_acceptance()

        command, label = run.call_args.args[:2]
        assert command == [watcher.PYTHON, "-u", str(watcher.PROFILE_WRAPPER)]
        assert "post-G0" in label


def test_existing_acceptance_does_not_restart_profile_producer() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        acceptance = Path(temporary) / "acceptance.json"
        acceptance.write_text("{}\n", encoding="utf-8")
        with (
            patch.object(watcher, "ACCEPTANCE", acceptance),
            patch.object(
                watcher,
                "validate_profile_acceptance_binding",
                return_value=({}, Path(temporary) / "checkpoint.pth"),
            ),
            patch.object(watcher, "run") as run,
        ):
            watcher.ensure_profile_acceptance()
        run.assert_not_called()


def test_stale_acceptance_restarts_profile_producer() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        acceptance = Path(temporary) / "acceptance.json"
        acceptance.write_text('{"accepted": true}\n', encoding="utf-8")

        with (
            patch.object(watcher, "ACCEPTANCE", acceptance),
            patch.object(
                watcher,
                "validate_profile_acceptance_binding",
                side_effect=[
                    RuntimeError("stale rank-1"),
                    ({"checkpoint_sha256": "new"}, Path(temporary) / "checkpoint.pth"),
                ],
            ),
            patch.object(watcher, "run") as run,
        ):
            watcher.ensure_profile_acceptance()
        run.assert_called_once()


def test_acceptance_binding_tracks_current_rank1_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        run_dir = root / "run"
        run_dir.mkdir()
        checkpoint = run_dir / "checkpoint_epoch19.pth"
        checkpoint.write_bytes(b"checkpoint")
        ranking = run_dir / "profile_ranking_valid825.md"
        ranking.write_text("| rank | epoch |\n|---:|---:|\n| 1 | 19 |\n", encoding="utf-8")
        hardware_config = root / "hardware.yml"
        training_config = root / "training.yml"
        hardware_config.write_bytes(b"hardware")
        training_config.write_bytes(b"training")
        state = run_dir / "checkpoint_epoch9_state_dict.pth"
        state.write_bytes(b"state")
        training_identity = run_dir / "training_config_identity.json"
        training_identity.write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "schema": "local5_training_config_identity_v1",
                    "authority": "ep9_optimizer_scheduler_state",
                    "config_path": str(training_config),
                    "config_sha256": watcher.file_sha256(training_config),
                    "state_path": str(state),
                    "state_sha256": watcher.file_sha256(state),
                    "checks": {"runtime": True},
                }
            ),
            encoding="utf-8",
        )
        run_identity = root / "post_g0_run_identity.json"
        run_identity.write_text(
            json.dumps(
                {
                    "best_epoch": 19,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": watcher.file_sha256(checkpoint),
                    "config": str(hardware_config),
                    "config_sha256": watcher.file_sha256(hardware_config),
                    "source_bindings": {
                        "training_config_identity": {
                            "path": str(training_identity),
                            "sha256": watcher.file_sha256(training_identity),
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        manifest = root / "ordered_term_manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": watcher.file_sha256(checkpoint),
                    "run_identity_file_sha256": watcher.file_sha256(run_identity),
                }
            ),
            encoding="utf-8",
        )
        required = (
            "loader_provenance",
            "formal_qualification",
            "relation_rtl_binding",
            "descriptor_geometry",
            "replay_binding",
            "descriptor_report_binding",
            "reports_recomputed_equal",
            "source_software_binding",
            "release_receipt_binding",
            "checkpoint_projection_weight_binding",
            "checkpoint_projection_payload_recomputed",
            "checkpoint_projection_topology_abi",
            "threshold_training_deployment_semantics",
        )
        acceptance = root / "acceptance.json"
        acceptance.write_text(
            json.dumps(
                {
                    "schema": "local5_post_g0_acceptance_v1",
                    "accepted": True,
                    "samples": 100,
                    "blocks": 12,
                    "manifest": str(manifest),
                    "manifest_sha256": watcher.file_sha256(manifest),
                    "run_identity": str(run_identity),
                    "run_identity_sha256": watcher.file_sha256(run_identity),
                    "checks": {name: True for name in required},
                }
            ),
            encoding="utf-8",
        )
        with (
            patch.object(watcher, "RUN", run_dir),
            patch.object(watcher, "RANKING", ranking),
            patch.object(watcher, "HARDWARE_CONFIG", hardware_config),
            patch.object(watcher, "TRAINING_CONFIG", training_config),
            patch.object(watcher, "TRAINING_IDENTITY", training_identity),
            patch.object(watcher, "RUN_IDENTITY", run_identity),
            patch.object(watcher, "ACCEPTANCE", acceptance),
        ):
            identity, selected = watcher.validate_profile_acceptance_binding()
            assert selected == checkpoint
            assert identity["best_epoch"] == 19
            checkpoint.write_bytes(b"stale replacement")
            try:
                watcher.validate_profile_acceptance_binding()
            except RuntimeError as exc:
                assert "rank-1 binding" in str(exc)
            else:
                raise AssertionError("stale checkpoint was accepted")
