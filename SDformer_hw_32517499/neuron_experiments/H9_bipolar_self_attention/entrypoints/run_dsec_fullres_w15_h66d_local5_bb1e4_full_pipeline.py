"""Run the fair Local-5 fullres train/eval/deploy/profile pipeline."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import yaml

from run_dsec_fullres_paper_w15_deploy_followup import (
    file_sha256,
    make_deploy_configs,
    parse_profile,
    run_or_reuse_eval,
)


REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
CONFIG = EXP / "configs/generated/dsec_fullres_w15_H66d_local5_bb1e4_ft30.yml"
SOURCE = (
    EXP
    / "results/h66d_allbinary_all12_lr_ttx_w720_fastlr_full30_bs8_full30_20260712_setsid/checkpoint_epoch29.pth"
)
H67_DEPLOY = (
    EXP
    / "results/dsec_fullres_w15_H67_crop_bb1e4_resume30_20260804/ep30_deploy_summary.md"
)
ROOT = EXP / "results/dsec_fullres_w15_H66d_local5_bb1e4_ft30_20260805"
STATUS = ROOT / "status.log"
REDESIGN = REPO / "neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md"
EVAL_EPOCHS = (9, 14, 19, 24, 29)
RESUME_EPOCHS = (9, 19, 29)
PYTHON = Path(sys.executable)
PROFILE_WRAPPER = REPO / "hw_autoresearch_nts07/scripts/run_local5_bb1e4_postg0_profile.py"
PROFILE_ACCEPTANCE = (
    REPO
    / "hw_autoresearch_nts07/results/local5_fullres_bb1e4_postg0_acceptance_20260805/acceptance.json"
)
HARDWARE_CONFIG = EXP / (
    "configs/generated/dsec_fullres_w15_H66d_local5_bb1e4_ft30_"
    "hardware_order_q7q17_deploy.yml"
)
DYADIC_CONFIG = EXP / (
    "configs/generated/dsec_fullres_w15_H66d_local5_bb1e4_ft30_"
    "dyadic_q7q17_deploy.yml"
)


def record(message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    with STATUS.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "SDFORMER_USE_MLFLOW": "0",
            "SDFORMER_MLFLOW_MODEL_LOGGING": "0",
            "SDFORMER_SNN_BACKEND": "cupy",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    return env


def run(command: list[str], log: Path, label: str) -> None:
    record(f"START {label}: {' '.join(command)}")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            cwd=REPO,
            env=environment(),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    record(f"END {label}: exit_code={result.returncode}")
    if result.returncode:
        raise RuntimeError(f"{label} failed; see {log}")


def wait_for_h67_deploy() -> None:
    while not H67_DEPLOY.is_file():
        record("WAIT H67 ep30 deploy evaluation")
        time.sleep(60)
    record("RELEASE H67 ep30 deploy evaluation complete")


def wait_for_profile_acceptance(
    checkpoint: Path | None = None,
    poll_seconds: int = 60,
    timeout_hours: float = 12.0,
) -> None:
    deadline = time.monotonic() + timeout_hours * 3600
    while True:
        if PROFILE_ACCEPTANCE.is_file():
            if checkpoint is None:
                record("RELEASE canonical Local-5 post-G0 acceptance artifact")
                return
            try:
                validate_profile_acceptance(checkpoint)
            except (
                FileNotFoundError,
                json.JSONDecodeError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as error:
                record(f"WAIT stale Local-5 post-G0 acceptance: {error}")
            else:
                record("RELEASE canonical Local-5 post-G0 acceptance")
                return
        if time.monotonic() >= deadline:
            raise TimeoutError("Local-5 post-G0 acceptance did not pass")
        record("WAIT canonical Local-5 post-G0 acceptance")
        time.sleep(poll_seconds)


def latest_resumable_checkpoint() -> Path | None:
    """Return the newest model checkpoint with a matching optimizer state."""
    candidates: list[tuple[int, Path]] = []
    for state_path in ROOT.glob("checkpoint_epoch*_state_dict.pth"):
        match = re.fullmatch(r"checkpoint_epoch(\d+)_state_dict\.pth", state_path.name)
        if not match:
            continue
        epoch = int(match.group(1))
        checkpoint = ROOT / f"checkpoint_epoch{epoch}.pth"
        if checkpoint.is_file():
            candidates.append((epoch, checkpoint))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def validate_training_contract() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    runtime = config.get("runtime", {})
    checks = {
        "full resolution": config.get("loader", {}).get("resolution") == [480, 640],
        "no crop": config.get("loader", {}).get("crop") is None,
        "T2x15x15 window": config.get("swin_transformer", {}).get("window_size") == [2, 15, 15],
        "30 epochs": config.get("loader", {}).get("n_epochs") == 30,
        "batch2": config.get("loader", {}).get("batch_size") == 2,
        "BN no_running": config.get("test", {}).get("bn_policy") == "no_running",
        "model save epochs": tuple(runtime.get("force_save_epochs", ())) == EVAL_EPOCHS,
        "paired state epochs": tuple(runtime.get("state_save_epochs", ())) == RESUME_EPOCHS,
        "force-only saving": runtime.get("save_only_force_epochs") is True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Local-5 training contract failed: {', '.join(failed)}")
    record("PASS Local-5 training contract: fullres/window/batch/BN/save-resume epochs")


def validate_checkpoint_contract() -> None:
    missing = [
        str(ROOT / f"checkpoint_epoch{epoch}.pth")
        for epoch in EVAL_EPOCHS
        if not (ROOT / f"checkpoint_epoch{epoch}.pth").is_file()
    ]
    missing.extend(
        str(ROOT / f"checkpoint_epoch{epoch}_state_dict.pth")
        for epoch in RESUME_EPOCHS
        if not (ROOT / f"checkpoint_epoch{epoch}_state_dict.pth").is_file()
    )
    if missing:
        raise RuntimeError("Local-5 checkpoint contract missing: " + ", ".join(missing))
    record("PASS Local-5 checkpoint contract: eval models5/paired states3")


def validate_training_load_audit() -> None:
    log = (ROOT / "train.log").read_text(encoding="utf-8", errors="replace")
    audits = re.findall(
        r"\[H9\] load audit: checkpoint_overlay_keys=(\d+), missing=(\d+), unexpected=(\d+)",
        log,
    )
    if not audits:
        raise RuntimeError("Local-5 training log has no checkpoint load audit")
    overlay, missing, unexpected = (int(value) for value in audits[-1])
    checks = {
        "overlay210": overlay == 210,
        "missing0": missing == 0,
        "unexpected0": unexpected == 0,
        "ATLIF105": "installed ATLIFTernaryPSN before load: 105 modules" in log,
        "Shiftmax12": "installed attention before load: 12 modules" in log,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("Local-5 load audit failed: " + ", ".join(failed))
    record(
        "PASS Local-5 load audit: overlay210/missing0/unexpected0/"
        "ATLIF105/Shiftmax12"
    )


@lru_cache(maxsize=None)
def _checkpoint_sha256(path: Path, size: int, mtime_ns: int) -> str:
    del size, mtime_ns
    return file_sha256(path)


def checkpoint_sha256(path: Path) -> str:
    resolved = path.resolve()
    stat = resolved.stat()
    return _checkpoint_sha256(resolved, stat.st_size, stat.st_mtime_ns)


def validate_eval_profile_contract(
    profile: Path, checkpoint: Path, config_path: Path = CONFIG
) -> None:
    raw = json.loads(profile.read_text(encoding="utf-8"))
    audit = raw.get("checkpoint_load_audit") or {}
    counts = raw.get("module_counts") or {}
    protocol = raw.get("eval_protocol") or {}
    metric_contract = raw.get("metric_contract") or {}
    aggregation_audit = raw.get("metric_aggregation_audit") or {}
    frame_equal = aggregation_audit.get("frame_equal_mean") or {}
    pixel_global = aggregation_audit.get("pixel_global_mean") or {}
    sequence_balanced = aggregation_audit.get("sequence_balanced_mean") or {}
    metrics = raw.get("metrics") or {}
    validation_file_list = raw.get("validation_file_list") or {}
    validation_file = Path(str(validation_file_list.get("path", "")))
    identity = raw.get("artifact_identity") or {}
    checks = {
        "resolution480x640": protocol.get("resolution") == [480, 640],
        "crop null": protocol.get("crop") is None,
        "window2x15x15": protocol.get("window_size") == [2, 15, 15],
        "BN no_running": protocol.get("bn_policy") == "no_running",
        "eval batch1": int(protocol.get("eval_batch_size", 0)) == 1,
        "AAE-2D contract": metric_contract.get("AAE")
        == "legacy_2d_direction_angle_degrees_between_uv",
        "AE-3D contract": metric_contract.get("AAE_Benchmark")
        == "middlebury_barron_3d_angle_degrees_between_normalized_uv1",
        "DSEC Fl contract": metric_contract.get("DSEC_Fl")
        == "percentage_epe_gt_3px_and_gt_5pct_of_ground_truth_flow_magnitude",
        "frame-mean aggregation": metric_contract.get("aggregation")
        == "masked_mean_per_frame_then_equal_mean_over_validation_frames",
        "local validation population": metric_contract.get("population")
        == "local_DSEC_valid_file_list_not_official_hidden_test",
        "samples825": int(raw.get("samples", 0)) == 825,
        "aggregation schema": aggregation_audit.get("schema")
        == "flow_metric_aggregation_audit_v1",
        "aggregation frames825": int(aggregation_audit.get("frame_count", 0)) == 825,
        "aggregation sequences18": int(aggregation_audit.get("sequence_count", 0)) == 18,
        "aggregation valid pixels": float(aggregation_audit.get("valid_pixels", 0)) > 0,
        "aggregation per-sequence18": len(aggregation_audit.get("per_sequence") or {}) == 18,
        "aggregation matches production": all(
            key in metrics
            and key in frame_equal
            and abs(float(metrics[key]) - float(frame_equal[key])) <= 1.0e-5
            for key in ("AEE", "AAE", "AAE_Benchmark", "DSEC_Fl")
        ),
        "aggregation modes complete": all(
            key in pixel_global and key in sequence_balanced
            for key in ("AEE", "AAE", "AAE_Benchmark", "DSEC_Fl")
        ),
        "validation list exists": validation_file.is_file(),
        "validation list SHA": validation_file.is_file()
        and validation_file_list.get("sha256") == file_sha256(validation_file),
        "overlay210": audit.get("checkpoint_overlay_keys") == 210
        and audit.get("model_overlay_keys") == 210,
        "missing0": audit.get("missing_count") == 0,
        "unexpected0": audit.get("unexpected_count") == 0,
        "ATLIF105": counts.get("ATLIFTernaryPSN") == 105,
        "Shiftmax12": counts.get("ShiftmaxAttention") == 12,
        "checkpoint path": identity.get("checkpoint_path")
        == str(checkpoint.resolve()),
        "checkpoint SHA256": identity.get("checkpoint_sha256")
        == checkpoint_sha256(checkpoint),
        "config path": identity.get("config_path") == str(config_path.resolve()),
        "config SHA256": identity.get("config_sha256") == file_sha256(config_path),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            f"Local-5 eval profile contract failed for {profile}: {', '.join(failed)}"
        )


def validate_profile_acceptance(checkpoint: Path) -> None:
    if not PROFILE_ACCEPTANCE.is_file():
        raise FileNotFoundError(PROFILE_ACCEPTANCE)
    acceptance = json.loads(PROFILE_ACCEPTANCE.read_text(encoding="utf-8"))
    manifest_path = Path(str(acceptance.get("manifest", "")))
    identity_path = Path(str(acceptance.get("run_identity", "")))
    if not manifest_path.is_file() or not identity_path.is_file():
        raise FileNotFoundError(
            f"accepted ordered manifest/run identity missing: {manifest_path} {identity_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    semantics = acceptance.get("threshold_training_semantics") or {}
    required_checks = (
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
    checkpoint_sha = checkpoint_sha256(checkpoint)
    acceptance_checks = acceptance.get("checks") or {}
    checks = {
        "schema": acceptance.get("schema") == "local5_post_g0_acceptance_v1",
        "accepted": acceptance.get("accepted") is True,
        "samples100": int(acceptance.get("samples", 0)) == 100,
        "blocks12": int(acceptance.get("blocks", 0)) == 12,
        "all acceptance checks": all(
            acceptance_checks.get(name) is True for name in required_checks
        ),
        "manifest path": manifest_path.resolve()
        == Path(str(acceptance.get("manifest", ""))).resolve(),
        "manifest SHA": acceptance.get("manifest_sha256")
        == file_sha256(manifest_path),
        "run identity path": identity_path.resolve()
        == Path(str(acceptance.get("run_identity", ""))).resolve(),
        "run identity SHA": acceptance.get("run_identity_sha256")
        == file_sha256(identity_path),
        "run identity schema": identity.get("schema") == "local5_post_g0_run_identity_v3",
        "identity checkpoint path": Path(str(identity.get("checkpoint", ""))).resolve()
        == checkpoint.resolve(),
        "identity checkpoint SHA": identity.get("checkpoint_sha256") == checkpoint_sha,
        "identity config path": Path(str(identity.get("config", ""))).resolve()
        == HARDWARE_CONFIG.resolve(),
        "identity config exists": HARDWARE_CONFIG.is_file(),
        "identity config SHA": HARDWARE_CONFIG.is_file()
        and identity.get("config_sha256") == file_sha256(HARDWARE_CONFIG),
        "checkpoint binding": Path(str(manifest.get("checkpoint", ""))).resolve()
        == checkpoint.resolve(),
        "manifest checkpoint SHA": manifest.get("checkpoint_sha256") == checkpoint_sha,
        "manifest run identity SHA": manifest.get("run_identity_file_sha256")
        == file_sha256(identity_path),
        "official ATLIF": semantics.get("threshold_modes") == ["official_atlif"],
        "homeostatic boundary1224": semantics.get("homeostatic_freeze_after_step")
        == 1224,
        "optimizer threshold gradients active": semantics.get(
            "optimizer_gradient_freeze_enabled"
        )
        is False,
        "checkpoint-static threshold": semantics.get("inference_threshold_source")
        == "checkpoint_static_parameter",
        "acceptance threshold gate": acceptance_checks.get(
            "threshold_training_deployment_semantics"
        ) is True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("Local-5 post-G0 acceptance failed: " + ", ".join(failed))
    record("PASS Local-5 post-G0 acceptance: checkpoint/samples/blocks/threshold semantics")


def best_epoch() -> int:
    ranking = ROOT / "profile_ranking_valid825.md"
    for line in ranking.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\|\s*1\s*\|\s*(\d+)\s*\|", line)
        if match:
            return int(match.group(1))
    raise RuntimeError("cannot parse Local-5 rank-1 epoch")


def validate_deploy_summary_contract(checkpoint: Path) -> None:
    summary_path = ROOT / "deploy_summary.json"
    value = json.loads(summary_path.read_text(encoding="utf-8"))
    epoch = best_epoch()
    checks = {
        "best epoch": int(value.get("best_epoch", -1)) == epoch,
        "checkpoint path": Path(str(value.get("checkpoint", ""))).resolve()
        == checkpoint.resolve(),
    }
    required_metrics = ("AEE", "AAE", "AAE_Benchmark", "DSEC_Fl", "total_spikes_g")
    for name in ("float", "dyadic", "hardware_order"):
        metrics = value.get(name) or {}
        checks[f"{name} metrics"] = all(metric in metrics for metric in required_metrics)
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("Local-5 deploy summary contract failed: " + ", ".join(failed))

    profile_specs = {
        "dyadic": (Path(str(value.get("dyadic_profile", ""))), DYADIC_CONFIG),
        "hardware_order": (
            Path(str(value.get("hardware_profile", ""))),
            HARDWARE_CONFIG,
        ),
    }
    for name, (profile, config) in profile_specs.items():
        validate_eval_profile_contract(profile, checkpoint, config)
        parsed = parse_profile(profile)
        for metric in required_metrics:
            if abs(float(value[name][metric]) - float(parsed[metric])) > 1e-9:
                raise RuntimeError(
                    f"Local-5 deploy summary/profile metric drift: {name} {metric}"
                )
    record("PASS Local-5 deploy summary/profile contract including DSEC Fl")


def standard_profiles_reusable() -> bool:
    if not (ROOT / "profile_ranking_valid825.md").is_file():
        return False
    try:
        for epoch in EVAL_EPOCHS:
            validate_eval_profile_contract(
                ROOT / f"standard_valid825/epoch{epoch}/spike_profile.json",
                ROOT / f"checkpoint_epoch{epoch}.pth",
            )
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return False
    return True


def append_protocol() -> None:
    marker = "DSEC_FULLRES_W15_H66D_LOCAL5_BB1E4_PIPELINE_20260805"
    if marker in REDESIGN.read_text(encoding="utf-8"):
        return
    with REDESIGN.open("a", encoding="utf-8") as handle:
        handle.write("\n\n### H66d Local-5 fullres bb1e4 公平重跑（2026-08-05）\n\n")
        handle.write(f"<!-- {marker} -->\n")
        handle.write(
            "- 旧 Local-5 fullres 使用 backbone/norm LR `2e-6/1e-6`，不能与已修复的 H67 比较；"
            "本轮改为同一 bb1e4 optimizer，并用 milestones13/20 复现 H67 实际执行的"
            "ep0--12/13--19/20--29 LR 轨迹；step1224 后只停止 homeostatic threshold "
            "update，optimizer threshold gradient 仍开启。\n"
        )
        handle.write(
            "- 固定 480x640、window2x15x15、batch2、BN no_running、30 epochs；"
            "从 Local-5 crop/full30 rank-1 ep29 初始化。\n"
        )
        handle.write(
            "- 在 ep9/19/29 保存 model+optimizer/scheduler/scaler 成对状态；流水线重启时"
            "自动选择最新成对 checkpoint 并使用 `--resume 1`，避免中断后从头训练。\n"
        )
        handle.write(
            "- 评估 ep9/14/19/24/29 standard valid825；rank-1 再跑 dyadic Q7/Q1.7、"
            "hardware-order，以及100样本 T450 post-G0 ordered profile/replay/acceptance。\n"
        )
        handle.write(f"- config：`{CONFIG.relative_to(REPO)}`。\n")
        handle.write(f"- status：`{STATUS.relative_to(REPO)}`。\n")


def append_results(summary: Path) -> None:
    marker = "DSEC_FULLRES_W15_H66D_LOCAL5_BB1E4_RESULT_20260805"
    if marker in REDESIGN.read_text(encoding="utf-8"):
        return
    with REDESIGN.open("a", encoding="utf-8") as handle:
        handle.write("\n\n### H66d Local-5 fullres bb1e4 结果\n\n")
        handle.write(f"<!-- {marker} -->\n\n")
        handle.write((ROOT / "profile_ranking_valid825.md").read_text(encoding="utf-8"))
        handle.write("\n")
        handle.write(summary.read_text(encoding="utf-8"))


def main() -> int:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    ROOT.mkdir(parents=True, exist_ok=True)
    wait_for_h67_deploy()
    run(
        [str(PYTHON), str(EXP / "entrypoints/make_dsec_fullres_w15_h66d_local5_bb1e4_config.py")],
        ROOT / "config.log",
        "generate Local-5 bb1e4 config",
    )
    validate_training_contract()
    append_protocol()

    profile_process = subprocess.Popen(
        [str(PYTHON), "-u", str(PROFILE_WRAPPER)],
        cwd=REPO / "hw_autoresearch_nts07",
        env=environment(),
        stdout=(ROOT / "postg0_profile_watcher.log").open("a", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )
    record(f"START post-G0 watcher pid={profile_process.pid}")

    if not (ROOT / "checkpoint_epoch29.pth").is_file():
        resume_checkpoint = latest_resumable_checkpoint()
        train_command = [
            str(PYTHON), "-u", str(EXP / "entrypoints/train.py"),
            "--config", str(CONFIG),
            "--prev_runid", str(resume_checkpoint or SOURCE),
            "--save_path", str(ROOT / "checkpoint_epoch{}.pth"),
            "--finetune", "1",
        ]
        if resume_checkpoint is not None:
            train_command.extend(["--resume", "1"])
            record(f"RESUME Local-5 from {resume_checkpoint.name} with paired state")
        run(
            train_command,
            ROOT / "train.log",
            "Local-5 bb1e4 fullres train30",
        )

    validate_training_load_audit()
    validate_checkpoint_contract()

    ranking = ROOT / "profile_ranking_valid825.md"
    if not standard_profiles_reusable():
        record("REBUILD Local-5 standard valid825: missing or stale profile contract")
        epoch_args = [item for epoch in EVAL_EPOCHS for item in ("--epoch", str(epoch))]
        run(
            [
                str(PYTHON), "-u", str(EXP / "entrypoints/run_h9_standard_valid825_eval.py"),
                "--config", str(CONFIG), "--run-dir", str(ROOT), "--ranking-mode", "aee",
                *epoch_args,
            ],
            ROOT / "valid825.log",
            "Local-5 bb1e4 valid825",
        )
    for eval_epoch in EVAL_EPOCHS:
        validate_eval_profile_contract(
            ROOT / f"standard_valid825/epoch{eval_epoch}/spike_profile.json",
            ROOT / f"checkpoint_epoch{eval_epoch}.pth",
        )
    record("PASS Local-5 standard valid825 profiles: protocol/load/modules/checkpoint SHA")

    epoch = best_epoch()
    checkpoint = ROOT / f"checkpoint_epoch{epoch}.pth"
    dyadic_config, hardware_config = make_deploy_configs("H66d", CONFIG)
    dyadic_profile, dyadic = run_or_reuse_eval(
        f"Local-5 ep{epoch} dyadic valid825", dyadic_config, checkpoint,
        ROOT / f"deploy_valid825/dyadic_q7q17/epoch{epoch}",
    )
    hardware_profile, hardware = run_or_reuse_eval(
        f"Local-5 ep{epoch} hardware-order valid825", hardware_config, checkpoint,
        ROOT / f"deploy_valid825/hardware_order_q7q17/epoch{epoch}",
    )
    validate_eval_profile_contract(dyadic_profile, checkpoint, dyadic_config)
    validate_eval_profile_contract(hardware_profile, checkpoint, hardware_config)
    record("PASS Local-5 deploy profiles: protocol/load/modules/checkpoint+config SHA")
    floating = parse_profile(ROOT / f"standard_valid825/epoch{epoch}/spike_profile.json")
    result = {
        "scope": "attention_core_hardware_order_numeric_not_full_network_rtl_exact",
        "best_epoch": epoch,
        "checkpoint": str(checkpoint),
        "float": floating,
        "dyadic": dyadic,
        "hardware_order": hardware,
        "dyadic_profile": str(dyadic_profile),
        "hardware_profile": str(hardware_profile),
    }
    (ROOT / "deploy_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = ROOT / "deploy_summary.md"
    lines = [
        "# Local-5 bb1e4 deploy summary", "",
        "Scope: attention-core hardware-order numeric; full T450 SV zero-mismatch is a separate sign-off.", "",
        "| path | AEE | AAE benchmark | DSEC Fl(%) | spikes(G) |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, metrics in (("float", floating), ("dyadic", dyadic), ("hardware-order", hardware)):
        lines.append(
            f"| {label} | {metrics['AEE']:.4f} | "
            f"{metrics['AAE_Benchmark']:.4f} | {metrics['DSEC_Fl']:.4f} | "
            f"{metrics['total_spikes_g']:.4f} |"
        )
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    validate_deploy_summary_contract(checkpoint)
    append_results(summary)

    record("ALL COMPLETE fullres deploy followup H67 reference and H66d bb1e4")
    profile_exit = profile_process.wait()
    record(f"END post-G0 watcher exit_code={profile_exit}")
    if profile_exit:
        record(
            "RECOVER initial post-G0 child exited; canonical external producer/"
            "checkpoint-bound acceptance remains authoritative"
        )
    try:
        validate_profile_acceptance(checkpoint)
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        run(
            [str(PYTHON), "-u", str(PROFILE_WRAPPER)],
            ROOT / "postg0_profile_recovery.log",
            "recover or join canonical Local-5 post-G0 producer",
        )
    wait_for_profile_acceptance(checkpoint)
    validate_profile_acceptance(checkpoint)
    record("ALL COMPLETE Local-5 bb1e4 train/eval/deploy/post-G0 pipeline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
