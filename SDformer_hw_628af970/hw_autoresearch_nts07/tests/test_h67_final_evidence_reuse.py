from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_h67_ep30_fullres_t450_profile as ep30
import run_h67_postconvergence_rank1_profile as final_watcher
from evidence_provenance import EXPECTED_ALL12_NAMES


def binding(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": ep30.sha256(path),
        "bytes": path.stat().st_size,
    }


def write_projection_report(
    root: Path, checkpoint: Path, config: Path, trace_manifest: Path
) -> Path:
    assets = root / "projection_assets"
    assets.mkdir(exist_ok=True)
    source_files = {}
    for name in (
        "vector_generator",
        "vector_generator_tests",
        "runner",
        "testbench",
        "sva",
        "bind",
    ):
        path = assets / f"{name}.txt"
        path.write_text(name + "\n", encoding="utf-8")
        source_files[name] = binding(path)
    rtl_bindings = []
    for index in range(7):
        path = assets / f"rtl_{index}.sv"
        path.write_text(f"// rtl {index}\n", encoding="utf-8")
        rtl_bindings.append(binding(path))

    records = []
    payloads = []
    for name in sorted(EXPECTED_ALL12_NAMES):
        vector_dir = assets / name.replace(".", "_")
        vector_dir.mkdir()
        payload = vector_dir / "payload.memh"
        payload.write_text("00\n", encoding="utf-8")
        record_manifest = vector_dir / "manifest.json"
        record_manifest.write_text(json.dumps({"name": name}) + "\n", encoding="utf-8")
        records.append(
            {
                "name": name,
                "vector_dir": str(vector_dir),
                "files": {
                    payload.name: {
                        "sha256": ep30.sha256(payload),
                        "bytes": payload.stat().st_size,
                    }
                },
            }
        )
        payloads.append(
            {
                "name": name,
                "record_manifest": binding(record_manifest),
                "files": [binding(payload)],
            }
        )

    vector_manifest = assets / "vectors_manifest.json"
    vector_manifest.write_text(json.dumps({"records": records}) + "\n", encoding="utf-8")
    report = root / "projection/report.json"
    report.parent.mkdir(exist_ok=True)
    report.write_text(
        json.dumps(
            {
                "schema": "h67_checkpoint_projection_rtl_exact_v2",
                "status": "PASS",
                "scope": (
                    "checkpoint_bound_real_weight_projection_component_"
                    "rtl_exact_not_full_network"
                ),
                "checkpoint_identity": {
                    "checkpoint_sha256": ep30.sha256(checkpoint),
                    "config_sha256": ep30.sha256(config),
                },
                "record_count": 12,
                "required_stage_coverage": [0, 1, 2, 3],
                "temporal_tokens": 450,
                "token_id_width": 9,
                "weight_mode": "checkpoint_dyadic_int8_projection_weight",
                "source_manifest": str(trace_manifest),
                "source_manifest_sha256": ep30.sha256(trace_manifest),
                "vector_manifest": str(vector_manifest),
                "vector_manifest_sha256": ep30.sha256(vector_manifest),
                "records": [{"name": name} for name in sorted(EXPECTED_ALL12_NAMES)],
                "vector_payloads": payloads,
                "source_artifacts": {
                    "source_trace_manifest": binding(trace_manifest),
                    "vector_manifest": binding(vector_manifest),
                    **source_files,
                    "rtl": rtl_bindings,
                },
            }
        ),
        encoding="utf-8",
    )
    return report


def write_reports(root: Path, checkpoint: Path) -> tuple[Path, Path, Path]:
    checkpoint_sha = ep30.sha256(checkpoint)
    config = root / "config.yml"
    if not config.is_file():
        config.write_bytes(b"config")
    score = root / "score/report.json"
    atlif = root / "atlif/report.json"
    projection = root / "projection/report.json"
    score.parent.mkdir(parents=True)
    atlif.parent.mkdir(parents=True)
    projection.parent.mkdir(parents=True)
    score.write_text(
        json.dumps(
            {
                "status": "PASS",
                "scope": "checkpoint_bound_component_rtl_exact_not_full_network",
                "source_trace_manifest": str(root / "trace/manifest.json"),
                "source_trace_manifest_sha256": "deferred",
                "run_context": {
                    "artifact_identity": {
                        "checkpoint_sha256": checkpoint_sha,
                        "config_sha256": ep30.sha256(config),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    atlif.write_text(
        json.dumps(
            {
                "status": "PASS",
                "checkpoint_identity": {
                    "checkpoint_sha256": checkpoint_sha,
                    "config_sha256": ep30.sha256(config),
                },
            }
        ),
        encoding="utf-8",
    )
    return score, atlif, projection


def write_profile_trace_audit(
    root: Path, checkpoint: Path
) -> tuple[Path, Path, Path, Path]:
    config = root / "config.yml"
    config.write_bytes(b"config")
    trace = root / "trace"
    trace.mkdir()
    payload = trace / "record.npz"
    payload.write_bytes(b"trace-payload")
    record = {
        "file": str(payload),
        "sha256": ep30.sha256(payload),
        "temporal_tokens": 450,
    }
    manifest = trace / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "run_context": {
                    "artifact_identity": {
                        "checkpoint_sha256": ep30.sha256(checkpoint),
                        "config_sha256": ep30.sha256(config),
                    }
                },
                "records": [dict(record) for _ in range(12)],
            }
        ),
        encoding="utf-8",
    )
    profile_dir = root / "profile"
    profile_dir.mkdir()
    profile = profile_dir / "nts11_hardware_p0_profile.json"
    profile.write_text(
        json.dumps(
            {
                "samples": 100,
                "bit_trace_records": 12,
                "artifact_identity": {
                    "checkpoint_sha256": ep30.sha256(checkpoint),
                    "config_sha256": ep30.sha256(config),
                },
                "eval_protocol": {
                    "resolution": [480, 640],
                    "crop": None,
                    "window_size": [2, 15, 15],
                    "tokens_per_window": 450,
                },
                "module_counts": {
                    "ATLIFTernaryPSN": 105,
                    "ShiftmaxAttention": 12,
                },
            }
        ),
        encoding="utf-8",
    )
    audit_dir = root / "audit"
    audit_dir.mkdir()
    audit = audit_dir / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "status": "PASS",
                "source_manifest": str(manifest),
                "coverage": {
                    "stages": [0, 1, 2, 3],
                    "four_stage_complete": True,
                },
                "records": [{"sha256_ok": True} for _ in range(12)],
            }
        ),
        encoding="utf-8",
    )
    score_report = root / "score/report.json"
    if score_report.is_file():
        score_value = json.loads(score_report.read_text(encoding="utf-8"))
        score_value["source_trace_manifest_sha256"] = ep30.sha256(manifest)
        score_report.write_text(json.dumps(score_value), encoding="utf-8")
    write_projection_report(root, checkpoint, config, manifest)
    return config, profile, trace, audit


def test_ep30_reuse_requires_both_reports_bound_to_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        checkpoint = root / "checkpoint.pth"
        checkpoint.write_bytes(b"checkpoint")
        score, atlif, projection = write_reports(root, checkpoint)
        config, profile, trace, audit = write_profile_trace_audit(root, checkpoint)
        with (
            patch.object(ep30, "CONFIG", config),
            patch.object(ep30, "PROFILE", profile.parent),
            patch.object(ep30, "TRACE", trace),
            patch.object(ep30, "AUDIT", audit.parent),
            patch.object(ep30, "CHECKPOINT", checkpoint),
            patch.object(ep30, "RTL_RESULT", score.parent),
            patch.object(ep30, "ATLIF_RESULT", atlif.parent),
            patch.object(ep30, "PROJECTION_RESULT", projection.parent),
        ):
            assert ep30.completed_evidence_matches_checkpoint()
            checkpoint.write_bytes(b"changed checkpoint")
            assert not ep30.completed_evidence_matches_checkpoint()


def test_ep30_reuse_emits_the_downstream_complete_marker() -> None:
    messages: list[str] = []
    with tempfile.TemporaryDirectory() as temporary:
        with (
            patch.object(
                ep30, "completed_evidence_matches_checkpoint", return_value=True
            ),
            patch.object(ep30, "record", side_effect=messages.append),
            patch.object(ep30.fcntl, "flock"),
            patch.object(ep30, "LOCK", Path(temporary) / "watcher.lock"),
        ):
            assert ep30.main() == 0
    assert messages == [
        "REUSE completed H67 ep30 T450 profile/trace/RTL evidence",
        ep30.COMPLETE_MARKER,
    ]


def test_ep30_projection_runner_path_is_absolute() -> None:
    assert ep30.PROJECTION_RUNNER.is_absolute()
    assert ep30.PROJECTION_RUNNER.is_file()


def test_final_reuse_requires_current_rank1_and_checkpoint_sha() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        checkpoint = root / "checkpoint_epoch40.pth"
        checkpoint.write_bytes(b"checkpoint")
        score, atlif, projection = write_reports(root, checkpoint)
        config, profile, trace, audit = write_profile_trace_audit(root, checkpoint)
        final = root / "final.json"
        final.write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "rank1_epoch": 40,
                    "checkpoint": str(checkpoint),
                    "hardware_order_config": str(config),
                    "profile": str(profile),
                    "trace_manifest": str(trace / "manifest.json"),
                    "trace_audit": str(audit),
                    "rtl_report": str(score),
                    "atlif_rtl_report": str(atlif),
                    "projection_rtl_report": str(projection),
                    "scope": "checkpoint_bound_component_rtl_exact_not_full_network",
                }
            ),
            encoding="utf-8",
        )
        with patch.object(final_watcher, "FINAL", final):
            assert final_watcher.final_matches_rank1(40, checkpoint)
            assert not final_watcher.final_matches_rank1(35, checkpoint)
            config.write_bytes(b"changed config")
            assert not final_watcher.final_matches_rank1(40, checkpoint)
            config.write_bytes(b"config")
            checkpoint.write_bytes(b"changed checkpoint")
            assert not final_watcher.final_matches_rank1(40, checkpoint)


def test_final_reuse_rejects_stale_trace_audit() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        checkpoint = root / "checkpoint_epoch30.pth"
        checkpoint.write_bytes(b"checkpoint")
        score, atlif, projection = write_reports(root, checkpoint)
        config, profile, trace, audit = write_profile_trace_audit(root, checkpoint)
        final = root / "final.json"
        final.write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "rank1_epoch": 30,
                    "checkpoint": str(checkpoint),
                    "hardware_order_config": str(config),
                    "profile": str(profile),
                    "trace_manifest": str(trace / "manifest.json"),
                    "trace_audit": str(audit),
                    "rtl_report": str(score),
                    "atlif_rtl_report": str(atlif),
                    "projection_rtl_report": str(projection),
                    "scope": "checkpoint_bound_component_rtl_exact_not_full_network",
                }
            ),
            encoding="utf-8",
        )
        with patch.object(final_watcher, "FINAL", final):
            assert final_watcher.final_matches_rank1(30, checkpoint)
            payload = trace / "record.npz"
            payload.write_bytes(b"stale")
            assert not final_watcher.final_matches_rank1(30, checkpoint)


def test_trace_reuse_requires_checkpoint_config_and_npz_sha() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        trace = root / "trace"
        trace.mkdir()
        checkpoint = root / "checkpoint.pth"
        config = root / "config.yml"
        payload = trace / "record.npz"
        checkpoint.write_bytes(b"checkpoint")
        config.write_bytes(b"config")
        payload.write_bytes(b"trace-payload")
        record = {
            "file": str(payload),
            "sha256": ep30.sha256(payload),
            "temporal_tokens": 450,
        }
        (trace / "manifest.json").write_text(
            json.dumps(
                {
                    "run_context": {
                        "artifact_identity": {
                            "checkpoint_sha256": ep30.sha256(checkpoint),
                            "config_sha256": ep30.sha256(config),
                        }
                    },
                    "records": [dict(record) for _ in range(12)],
                }
            ),
            encoding="utf-8",
        )
        with (
            patch.object(ep30, "TRACE", trace),
            patch.object(ep30, "CONFIG", config),
            patch.object(ep30, "CHECKPOINT", checkpoint),
        ):
            assert ep30.trace_matches_checkpoint()
        assert final_watcher.trace_matches_checkpoint(trace, config, checkpoint)
        payload.write_bytes(b"stale")
        assert not final_watcher.trace_matches_checkpoint(trace, config, checkpoint)


def test_profile_reuse_requires_profile_and_trace_pair() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        checkpoint = root / "checkpoint.pth"
        checkpoint.write_bytes(b"checkpoint")
        config, profile, trace, _audit = write_profile_trace_audit(root, checkpoint)
        with (
            patch.object(ep30, "CONFIG", config),
            patch.object(ep30, "CHECKPOINT", checkpoint),
            patch.object(ep30, "PROFILE", profile.parent),
            patch.object(ep30, "TRACE", trace),
        ):
            assert ep30.profile_matches_checkpoint()
            assert ep30.trace_matches_checkpoint()
            profile.unlink()
            assert not ep30.profile_matches_checkpoint()
            assert ep30.trace_matches_checkpoint()
        assert not final_watcher.profile_matches_checkpoint(
            profile, config, checkpoint
        )


def test_h67_release_requires_current_local5_rtl_identity() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        run = root / "run"
        run.mkdir()
        checkpoint = run / "checkpoint_epoch7.pth"
        checkpoint.write_bytes(b"checkpoint")
        ranking = run / "profile_ranking_valid825.md"
        ranking.write_text("| rank | epoch | AEE |\n| 1 | 7 | 1.0 |\n", encoding="utf-8")
        config = root / "hardware.yml"
        config.write_bytes(b"config")
        status = root / "status.log"
        status.write_text(ep30.LOCAL5_RTL_COMPLETE + "\n", encoding="utf-8")
        report = root / "checkpoint_bound_scope.json"
        payload = {
            "status": "PASS",
            "evidence_scope": (
                "checkpoint_bound_component_rtl_exact_not_full_network"
            ),
            "checkpoint_identity": {
                "best_epoch": 7,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": ep30.sha256(checkpoint),
                "config": str(config),
                "config_sha256": ep30.sha256(config),
            },
            "score_shiftmax": {"status": "PASS"},
            "projection": {
                "weight_mode": ep30.LOCAL5_PROJECTION_WEIGHT_MODE,
                "verification": {
                    "checkpoint_weight_binding": "PASS",
                    "random_sva": "PASS",
                    "verilator_lint": "PASS",
                    "yosys_check": "PASS",
                },
            },
            "atlif_temporal_matrix": {"status": "PASS"},
        }
        report.write_text(json.dumps(payload), encoding="utf-8")
        patches = (
            patch.object(ep30, "LOCAL5_RUN", run),
            patch.object(ep30, "LOCAL5_RANKING", ranking),
            patch.object(ep30, "LOCAL5_HARDWARE_CONFIG", config),
            patch.object(ep30, "LOCAL5_RTL", report),
            patch.object(ep30, "LOCAL5_RTL_STATUS", status),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            assert ep30.local5_rtl_evidence_complete()
            payload["projection"]["weight_mode"] = (
                "checkpoint_dyadic_int8_head_slice"
            )
            report.write_text(json.dumps(payload), encoding="utf-8")
            assert not ep30.local5_rtl_evidence_complete()
            payload["projection"]["weight_mode"] = (
                ep30.LOCAL5_PROJECTION_WEIGHT_MODE
            )
            checkpoint.write_bytes(b"stale")
            assert not ep30.local5_rtl_evidence_complete()


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    return unittest.TestSuite(
        unittest.FunctionTestCase(test)
        for test in (
            test_ep30_reuse_requires_both_reports_bound_to_checkpoint,
            test_ep30_reuse_emits_the_downstream_complete_marker,
            test_ep30_projection_runner_path_is_absolute,
            test_final_reuse_requires_current_rank1_and_checkpoint_sha,
            test_final_reuse_rejects_stale_trace_audit,
            test_trace_reuse_requires_checkpoint_config_and_npz_sha,
            test_profile_reuse_requires_profile_and_trace_pair,
            test_h67_release_requires_current_local5_rtl_identity,
        )
    )
