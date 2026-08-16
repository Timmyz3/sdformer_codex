#!/usr/bin/env python3
"""Bind final H67 hardware evidence to the post-convergence rank-1 checkpoint."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from evidence_provenance import validate_projection_provenance


HW_ROOT = Path(__file__).resolve().parents[1]
REPO = HW_ROOT.parent
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
ENTRYPOINTS = EXP / "entrypoints"
sys.path.insert(0, str(ENTRYPOINTS))

from run_dsec_fullres_paper_w15_deploy_followup import make_deploy_configs  # noqa: E402


CONVERGENCE_STATUS = (
    EXP / "results/dsec_fullres_w15_equal_plus10_convergence_20260805.log"
)
# Producer writes Local5/H67/NB0; older waiters only looked for H67/NB0.
CONVERGENCE_DONE_MARKERS = (
    "ALL COMPLETE Local5/H67/NB0 equal +10 convergence audit",
    "ALL COMPLETE H67/NB0 equal +10 convergence audit",
)
CONVERGENCE_SUMMARY = (
    EXP / "results/dsec_fullres_w15_equal_plus10_convergence_summary_20260805.json"
)
RUN = EXP / "results/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_20260805"
RANKING = RUN / "profile_ranking_valid825.md"
SOURCE_CONFIG = EXP / "configs/generated/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40.yml"
EP30_CONFIG = EXP / (
    "configs/generated/dsec_fullres_w15_H67_crop_bb1e4_resume_ep30_"
    "hardware_order_q7q17_deploy.yml"
)
EP30_PROFILE = HW_ROOT / (
    "results/h67_fullres_ep30_t450_profile100_20260805/"
    "nts11_hardware_p0_profile.json"
)
EP30_TRACE = HW_ROOT / "results/h67_fullres_ep30_t450_all12_bit_trace_20260805"
EP30_AUDIT = HW_ROOT / (
    "results/h67_fullres_ep30_t450_all12_bit_trace_audit_20260805/audit.json"
)
EP30_RTL = HW_ROOT / "results/h67_fullres_ep30_t450_score_shiftmax_rtl_20260805/report.json"
EP30_ATLIF_RTL = HW_ROOT / "results/h67_ep30_checkpoint_atlif_dptme_rtl_20260805/report.json"
EP30_PROJECTION_RTL = HW_ROOT / "results/h67_ep30_checkpoint_projection_rtl_20260805/report.json"
FINAL = HW_ROOT / "results/h67_postconvergence_rank1_hardware_evidence_20260805.json"
STATUS = HW_ROOT / "results/h67_postconvergence_rank1_profile_watcher_20260805.log"
LOCK = HW_ROOT / "results/h67_postconvergence_rank1_profile_watcher_20260805.lock"
PYTHON = "/opt/conda/envs/sdformerflow/bin/python"
FINAL_SCOPE = (
    "checkpoint_bound_qk_score_scs_shiftmax_atlif_temporal_matrix_"
    "real_weight_projection_component_rtl_exact_not_full_network"
)


def record(message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with STATUS.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run(
    command: list[str],
    label: str,
    *,
    extra_env: dict[str, str] | None = None,
) -> None:
    record(f"START {label}: {' '.join(command)}")
    env = os.environ.copy()
    env.update(
        {
            "SDFORMER_USE_MLFLOW": "0",
            "SDFORMER_MLFLOW_MODEL_LOGGING": "0",
            "SDFORMER_SNN_BACKEND": "cupy",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(command, cwd=REPO, env=env)
    record(f"END {label}: exit_code={result.returncode}")
    if result.returncode:
        raise RuntimeError(f"{label} failed")


def wait_for_convergence() -> None:
    while True:
        text = (
            CONVERGENCE_STATUS.read_text(encoding="utf-8")
            if CONVERGENCE_STATUS.is_file()
            else ""
        )
        marker_hit = any(marker in text for marker in CONVERGENCE_DONE_MARKERS)
        summary_ok = False
        if CONVERGENCE_SUMMARY.is_file():
            try:
                summary = json.loads(
                    CONVERGENCE_SUMMARY.read_text(encoding="utf-8")
                )
                summary_ok = (
                    summary.get("schema")
                    == "dsec_fullres_equal_plus10_convergence_v1"
                    and isinstance(summary.get("candidates"), dict)
                    and all(
                        name in summary["candidates"]
                        for name in ("Local5", "H67", "NB0")
                    )
                )
            except (json.JSONDecodeError, OSError, TypeError):
                summary_ok = False
        if marker_hit and summary_ok:
            record(
                "RELEASE equal +10 convergence audit "
                f"(marker={'yes' if marker_hit else 'no'}, summary_ok={summary_ok})"
            )
            return
        record(
            "WAIT H67/NB0 equal +10 convergence audit "
            f"(marker={'yes' if marker_hit else 'no'}, summary_ok={summary_ok})"
        )
        time.sleep(300)


def best_epoch() -> int:
    for line in RANKING.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\|\s*1\s*\|\s*(\d+)\s*\|", line)
        if match:
            epoch = int(match.group(1))
            if epoch not in (30, 35, 40):
                raise RuntimeError(f"unexpected H67 rank-1 epoch: {epoch}")
            return epoch
    raise RuntimeError(f"cannot parse rank-1 epoch from {RANKING}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_report(
    report_path: Path, epoch: int, checkpoint: Path, config: Path
) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS":
        raise RuntimeError(f"H67 ep{epoch} RTL report is not PASS")
    scope = str(report.get("scope", ""))
    if "component_rtl_exact" not in scope:
        raise RuntimeError(f"H67 ep{epoch} RTL scope is not component exact: {scope}")
    identity = report.get("run_context", {}).get("artifact_identity", {})
    if identity.get("checkpoint_sha256") != sha256(checkpoint):
        raise RuntimeError(f"H67 ep{epoch} RTL report/checkpoint SHA mismatch")
    if identity.get("config_sha256") != sha256(config):
        raise RuntimeError(f"H67 ep{epoch} RTL report/config SHA mismatch")
    trace_path = Path(str(report.get("source_trace_manifest", "")))
    if (
        not trace_path.is_file()
        or report.get("source_trace_manifest_sha256") != sha256(trace_path)
    ):
        raise RuntimeError(f"H67 ep{epoch} RTL report/trace manifest mismatch")
    return report


def validate_projection_report(
    report_path: Path, epoch: int, checkpoint: Path, config: Path
) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    validate_projection_provenance(report)
    checks = {
        "status": report.get("status") == "PASS",
        "scope": "projection_component_rtl_exact" in str(report.get("scope", "")),
        "checkpoint": report.get("checkpoint_identity", {}).get("checkpoint_sha256")
        == sha256(checkpoint),
        "config": report.get("checkpoint_identity", {}).get("config_sha256")
        == sha256(config),
        "all12": int(report.get("record_count", 0)) == 12,
        "stages": report.get("required_stage_coverage") == [0, 1, 2, 3],
        "tokens": int(report.get("temporal_tokens", 0)) == 450,
        "token_id_width": int(report.get("token_id_width", 0)) == 9,
        "weight_mode": report.get("weight_mode")
        == "checkpoint_dyadic_int8_projection_weight",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"H67 ep{epoch} projection RTL report failed: {failed}")
    return report


def trace_matches_checkpoint(
    trace: Path, config: Path, checkpoint: Path
) -> bool:
    manifest_path = trace / "manifest.json"
    if not manifest_path.is_file() or not config.is_file() or not checkpoint.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = manifest.get("records") or []
        identity = manifest.get("run_context", {}).get("artifact_identity", {})
        return (
            identity.get("checkpoint_sha256") == sha256(checkpoint)
            and identity.get("config_sha256") == sha256(config)
            and len(records) == 12
            and {int(record.get("temporal_tokens", 0)) for record in records} == {450}
            and all(
                Path(str(record.get("file", ""))).is_file()
                and record.get("sha256") == sha256(Path(str(record["file"])))
                for record in records
            )
        )
    except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
        return False


def profile_matches_checkpoint(
    profile_path: Path, config: Path, checkpoint: Path
) -> bool:
    if not profile_path.is_file() or not config.is_file() or not checkpoint.is_file():
        return False
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        identity = profile.get("artifact_identity", {})
        protocol = profile.get("eval_protocol", {})
        counts = profile.get("module_counts", {})
        return (
            identity.get("checkpoint_sha256") == sha256(checkpoint)
            and identity.get("config_sha256") == sha256(config)
            and int(profile.get("samples", 0)) == 100
            and int(profile.get("bit_trace_records", 0)) == 12
            and protocol.get("resolution") == [480, 640]
            and protocol.get("crop") is None
            and protocol.get("window_size") == [2, 15, 15]
            and int(protocol.get("tokens_per_window", 0)) == 450
            and counts.get("ATLIFTernaryPSN") == 105
            and counts.get("ShiftmaxAttention") == 12
        )
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return False


def validate_profile_trace_audit(
    profile_path: Path,
    trace: Path,
    audit_path: Path,
    config: Path,
    checkpoint: Path,
) -> dict[str, str]:
    if not trace_matches_checkpoint(trace, config, checkpoint):
        raise RuntimeError("H67 profile trace/checkpoint/config identity mismatch")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile_identity = profile.get("artifact_identity", {})
    profile_protocol = profile.get("eval_protocol", {})
    profile_counts = profile.get("module_counts", {})
    profile_checks = {
        "checkpoint": profile_identity.get("checkpoint_sha256") == sha256(checkpoint),
        "config": profile_identity.get("config_sha256") == sha256(config),
        "samples": int(profile.get("samples", 0)) == 100,
        "resolution": profile_protocol.get("resolution") == [480, 640],
        "crop": profile_protocol.get("crop") is None,
        "window": profile_protocol.get("window_size") == [2, 15, 15],
        "tokens": int(profile_protocol.get("tokens_per_window", 0)) == 450,
        "atlif": int(profile_counts.get("ATLIFTernaryPSN", 0)) == 105,
        "shiftmax": int(profile_counts.get("ShiftmaxAttention", 0)) == 12,
        "trace_records": int(profile.get("bit_trace_records", 0)) == 12,
    }
    failed = [name for name, passed in profile_checks.items() if not passed]
    if failed:
        raise RuntimeError(f"H67 profile contract failed: {failed}")

    manifest_path = trace / "manifest.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit_checks = {
        "status": audit.get("status") == "PASS",
        "manifest": Path(str(audit.get("source_manifest", ""))).resolve()
        == manifest_path.resolve(),
        "four_stages": audit.get("coverage", {}).get("four_stage_complete") is True,
        "stages": audit.get("coverage", {}).get("stages") == [0, 1, 2, 3],
        "records": len(audit.get("records") or []) == 12,
        "record_sha": all(
            record.get("sha256_ok") is True for record in (audit.get("records") or [])
        ),
    }
    failed = [name for name, passed in audit_checks.items() if not passed]
    if failed:
        raise RuntimeError(f"H67 trace audit contract failed: {failed}")
    return {
        "hardware_order_config": str(config),
        "profile": str(profile_path),
        "trace_manifest": str(manifest_path),
        "trace_audit": str(audit_path),
    }


def final_matches_rank1(epoch: int, checkpoint: Path) -> bool:
    if not FINAL.is_file():
        return False
    try:
        final = json.loads(FINAL.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if (
        final.get("status") != "PASS"
        or int(final.get("rank1_epoch", -1)) != epoch
        or Path(str(final.get("checkpoint", ""))).resolve() != checkpoint.resolve()
        or final.get("scope") != FINAL_SCOPE
    ):
        return False
    score_path = Path(
        str(final.get("rtl_report") or final.get("reused_ep30_checkpoint_bound_report") or "")
    )
    atlif_path = Path(
        str(final.get("atlif_rtl_report") or final.get("reused_ep30_atlif_checkpoint_bound_report") or "")
    )
    projection_path = Path(
        str(
            final.get("projection_rtl_report")
            or final.get("reused_ep30_projection_checkpoint_bound_report")
            or ""
        )
    )
    config_path = Path(str(final.get("hardware_order_config", "")))
    profile_path = Path(str(final.get("profile", "")))
    trace_path = Path(str(final.get("trace_manifest", ""))).parent
    audit_path = Path(str(final.get("trace_audit", "")))
    if not all(
        path.is_file()
        for path in (
            score_path,
            atlif_path,
            projection_path,
            config_path,
            profile_path,
            trace_path / "manifest.json",
            audit_path,
        )
    ):
        return False
    try:
        validate_profile_trace_audit(
            profile_path, trace_path, audit_path, config_path, checkpoint
        )
        validate_report(score_path, epoch, checkpoint, config_path)
        validate_projection_report(projection_path, epoch, checkpoint, config_path)
        atlif = json.loads(atlif_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, RuntimeError):
        return False
    return (
        atlif.get("status") == "PASS"
        and atlif.get("checkpoint_identity", {}).get("checkpoint_sha256")
        == sha256(checkpoint)
        and atlif.get("checkpoint_identity", {}).get("config_sha256")
        == sha256(config_path)
    )


def main() -> int:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            record("EXIT another post-convergence H67 watcher owns the lock")
            return 0
        wait_for_convergence()
        epoch = best_epoch()
        checkpoint = RUN / f"checkpoint_epoch{epoch}.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if final_matches_rank1(epoch, checkpoint):
            record("REUSE completed post-convergence H67 rank-1 evidence binding")
            return 0

        if epoch == 30:
            profile_evidence = validate_profile_trace_audit(
                EP30_PROFILE, EP30_TRACE, EP30_AUDIT, EP30_CONFIG, checkpoint
            )
            report = validate_report(EP30_RTL, epoch, checkpoint, EP30_CONFIG)
            atlif_report = json.loads(EP30_ATLIF_RTL.read_text(encoding="utf-8"))
            validate_projection_report(
                EP30_PROJECTION_RTL, epoch, checkpoint, EP30_CONFIG
            )
            if (
                atlif_report.get("status") != "PASS"
                or atlif_report.get("checkpoint_identity", {}).get("checkpoint_sha256")
                != sha256(checkpoint)
                or atlif_report.get("checkpoint_identity", {}).get("config_sha256")
                != sha256(EP30_CONFIG)
            ):
                raise RuntimeError("H67 ep30 ATLIF RTL report/config/checkpoint SHA mismatch")
            final = {
                "status": "PASS",
                "rank1_epoch": epoch,
                "checkpoint": str(checkpoint),
                **profile_evidence,
                "reused_ep30_checkpoint_bound_report": str(EP30_RTL),
                "reused_ep30_atlif_checkpoint_bound_report": str(EP30_ATLIF_RTL),
                "reused_ep30_projection_checkpoint_bound_report": str(EP30_PROJECTION_RTL),
                "scope": report["scope"],
            }
            FINAL.write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
            record("ALL COMPLETE H67 post-convergence rank-1=ep30 evidence reused")
            return 0

        _, hardware_config = make_deploy_configs("H67", SOURCE_CONFIG)
        tag = f"h67_fullres_ep{epoch}_postconvergence_t450_20260805"
        profile = HW_ROOT / f"results/{tag}_profile100"
        trace = HW_ROOT / f"results/{tag}_all12_bit_trace"
        audit = HW_ROOT / f"results/{tag}_all12_bit_trace_audit"
        vectors = HW_ROOT / f"tb_h67/vectors/{tag}"
        rtl_result = HW_ROOT / f"results/{tag}_score_shiftmax_rtl"
        atlif_vectors = HW_ROOT / f"tb_hitflow/vectors/{tag}_checkpoint_atlif"
        atlif_result = HW_ROOT / f"results/{tag}_checkpoint_atlif_dptme_rtl"
        projection_result = HW_ROOT / f"results/{tag}_checkpoint_projection_rtl"
        if not (
            trace_matches_checkpoint(trace, hardware_config, checkpoint)
            and profile_matches_checkpoint(
                profile / "nts11_hardware_p0_profile.json",
                hardware_config,
                checkpoint,
            )
        ):
            run(
                [
                    PYTHON,
                    "neuron_experiments/H9_bipolar_self_attention/entrypoints/profile_nts11_hardware_p0.py",
                    "--config",
                    str(hardware_config),
                    "--checkpoint",
                    str(checkpoint),
                    "--output-dir",
                    str(profile),
                    "--samples",
                    "100",
                    "--num-workers",
                    "0",
                    "--ordered-trace",
                    "--bit-trace-dir",
                    str(trace),
                    "--bit-trace-samples",
                    "1",
                    "--bit-trace-windows",
                    "1",
                    "--bit-trace-all-blocks",
                ],
                f"H67 ep{epoch} post-convergence profile100/all12 trace",
            )
        run(
            [
                PYTHON,
                "hw_autoresearch_nts07/scripts/audit_h67_bit_trace.py",
                "--manifest",
                str(trace / "manifest.json"),
                "--output-dir",
                str(audit),
                "--require-four-stages",
                "--require-records",
                "12",
            ],
            f"H67 ep{epoch} post-convergence all12 trace audit",
        )
        run(
            ["bash", "hw_autoresearch_nts07/sim_h67/run_h67_checkpoint_row_trace_checks.sh"],
            f"H67 ep{epoch} post-convergence checkpoint-bound T450 RTL",
            extra_env={
                "TRACE_MANIFEST": str(trace / "manifest.json"),
                "VECTOR_DIR": str(vectors),
                "RESULT_DIR": str(rtl_result),
                "PYTHON": PYTHON,
            },
        )
        report_path = rtl_result / "report.json"
        report = validate_report(report_path, epoch, checkpoint, hardware_config)
        run(
            [
                PYTHON,
                "hw_autoresearch_nts07/scripts/generate_checkpoint_atlif_dptme_vectors.py",
                "--config",
                str(hardware_config),
                "--checkpoint",
                str(checkpoint),
                "--output-dir",
                str(atlif_vectors),
                "--sample-index",
                "0",
            ],
            f"H67 ep{epoch} checkpoint-bound ATLIF fixed-point vectors",
        )
        run(
            ["bash", "hw_autoresearch_nts07/sim_hitflow/run_checkpoint_atlif_dptme_checks.sh"],
            f"H67 ep{epoch} checkpoint-bound ATLIF DP-TME RTL",
            extra_env={"VECTOR_DIR": str(atlif_vectors), "RESULT_DIR": str(atlif_result)},
        )
        atlif_report_path = atlif_result / "report.json"
        atlif_report = json.loads(atlif_report_path.read_text(encoding="utf-8"))
        if (
            atlif_report.get("status") != "PASS"
            or atlif_report.get("checkpoint_identity", {}).get("checkpoint_sha256")
            != sha256(checkpoint)
            or atlif_report.get("checkpoint_identity", {}).get("config_sha256")
            != sha256(hardware_config)
        ):
            raise RuntimeError(f"H67 ep{epoch} ATLIF RTL report/config/checkpoint SHA mismatch")
        run(
            ["bash", "hw_autoresearch_nts07/sim_hitflow/run_gatestack_dctf96_real_trace_checks.sh"],
            f"H67 ep{epoch} checkpoint-bound all12 real-weight projection RTL",
            extra_env={
                "SOURCE_MANIFEST": str(trace / "manifest.json"),
                "RESULT_DIR": str(projection_result),
                "PYTHON": PYTHON,
            },
        )
        projection_report_path = projection_result / "report.json"
        validate_projection_report(
            projection_report_path, epoch, checkpoint, hardware_config
        )
        profile_evidence = validate_profile_trace_audit(
            profile / "nts11_hardware_p0_profile.json",
            trace,
            audit / "audit.json",
            hardware_config,
            checkpoint,
        )
        final = {
            "status": "PASS",
            "rank1_epoch": epoch,
            "checkpoint": str(checkpoint),
            **profile_evidence,
            "rtl_report": str(report_path),
            "atlif_rtl_report": str(atlif_report_path),
            "projection_rtl_report": str(projection_report_path),
            "scope": FINAL_SCOPE,
        }
        FINAL.write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
        record(f"ALL COMPLETE H67 post-convergence rank-1=ep{epoch} T450 evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
