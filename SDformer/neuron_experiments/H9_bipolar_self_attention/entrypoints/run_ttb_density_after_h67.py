"""Profile true token-time bundles for frozen TTX and H67 before H69."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from run_h60_family_deploy_eval import make_deploy_config, parse_profile, run_eval


REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
GEN = EXP / "configs/generated"
RESULTS = EXP / "results"
PY = Path(sys.executable)
PREV_STATUS = RESULTS / "h67_early_deploy_after_h68_status.log"
STATUS = RESULTS / "ttb_density_after_h67_status.log"
SUMMARY_JSON = RESULTS / "ttb_true_density_ttx_h67_h68_profile100.json"
SUMMARY_MD = RESULTS / "ttb_true_density_ttx_h67_h68_profile100.md"
H68_RUN = RESULTS / "h68_allbinary_all12_castling_ttx_aux050_toep20_w720_fastlr_full30_bs8_full30_20260711_setsid"
H68_DEPLOY_SUMMARY = H68_RUN / "h68_epoch19_dyadic_int8_valid825.json"
REDESIGN = REPO / "neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md"
PORTFOLIO = REPO / "neuron_autoresearch/DATE_IDEA_PORTFOLIO_20260712.md"

CASES = (
    (
        "TTX第2轮二进制定点部署",
        GEN / "date11full_ttx_dyadic_txonly_all12_deploy_int8.yml",
        RESULTS / "date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8_bs8_20260629_154937_setsid/checkpoint_epoch2.pth",
        RESULTS / "ttx_ep2_true_ttb_profile100_20260712",
    ),
    (
        "H67第19轮二进制定点部署",
        GEN / "h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30_dyadic_int8_deploy.yml",
        RESULTS / "h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30_bs8_full30_20260711_setsid/checkpoint_epoch19.pth",
        RESULTS / "h67_ep19_true_ttb_profile100_20260712",
    ),
    (
        "H68第19轮二进制定点部署",
        GEN / "h68_allbinary_all12_castling_ttx_deploy_full30_dyadic_int8_deploy.yml",
        H68_RUN / "checkpoint_epoch19.pth",
        RESULTS / "h68_ep19_true_ttb_profile100_20260713",
    ),
)


def record(message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    with STATUS.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def wait_previous() -> None:
    marker = "ALL COMPLETE H67 EARLY DEPLOY:"
    while not PREV_STATUS.exists() or marker not in PREV_STATUS.read_text(encoding="utf-8", errors="ignore"):
        record(f"WAIT H67 early deploy: {PREV_STATUS}")
        time.sleep(600)


def run_profile(label: str, config: Path, checkpoint: Path, output: Path) -> Path:
    profile = output / "nts11_hardware_p0_profile.json"
    if profile.exists() and "token_time_bundles" in profile.read_text(encoding="utf-8", errors="ignore"):
        return profile
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "SDFORMER_USE_MLFLOW": "0",
        "SDFORMER_MLFLOW_MODEL_LOGGING": "0",
        "SDFORMER_SNN_BACKEND": "cupy",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    command = [
        str(PY), "-u", str(EXP / "entrypoints/profile_nts11_hardware_p0.py"),
        "--config", str(config), "--checkpoint", str(checkpoint),
        "--output-dir", str(output), "--samples", "100", "--num-workers", "0",
    ]
    record(f"START {label}: {' '.join(command)}")
    with (output / "profile.log").open("a", encoding="utf-8") as handle:
        proc = subprocess.run(command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)
    record(f"END {label}: exit_code={proc.returncode}")
    if proc.returncode != 0:
        raise RuntimeError(f"TTB profile failed: {output / 'profile.log'}")
    return profile


def run_h68_deploy() -> None:
    source = GEN / "h68_allbinary_all12_castling_ttx_deploy_full30.yml"
    deploy = make_deploy_config(source)
    checkpoint = H68_RUN / "checkpoint_epoch19.pth"
    output = H68_RUN / "standard_dyadic_int8_valid825" / "epoch19"
    record(f"START H68 epoch19 dyadic deploy: config={deploy} checkpoint={checkpoint}")
    run_eval(deploy, checkpoint, output)
    row = {
        "candidate": "H68 Castling-trained deployment H60",
        "epoch": 19,
        "config": str(deploy),
        "checkpoint": str(checkpoint),
        "profile": str(output / "spike_profile.json"),
        **parse_profile(output / "spike_profile.json"),
    }
    H68_DEPLOY_SUMMARY.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    record(f"END H68 epoch19 dyadic deploy: {H68_DEPLOY_SUMMARY}")


def write(rows: list[dict]) -> None:
    SUMMARY_JSON.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# 真实Token-Time Bundle百样本统计", "",
        "Bundle布局为T=2×连续空间token×32通道。`活跃1--K`只是路由容量扫描，不代表已经实现等价跳过。", "",
        "| 模型 | 每bundle空间token | Q或K活性密度 | 全空 | K全零 | 无K运动 | 活跃1--4 | 活跃1--8 | 活跃1--16 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['token_bundle']} | {row['activity_density']:.6%} | "
            f"{row['empty_ratio']:.6%} | {row['kzero_ratio']:.6%} | {row['motion_zero_ratio']:.6%} | "
            f"{row['active_1_4_ratio']:.6%} | {row['active_1_8_ratio']:.6%} | "
            f"{row['active_1_16_ratio']:.6%} |"
        )
    lines += [
        "", "`全空`本身不能删除silent/silent对Shiftmax分母的贡献。逐位等价跳过仅限于已证明的"
        "Delta score复用，以及K全零时的value/投影门控。",
    ]
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    marker = "TRUE_TTB_TTX_H67_PROFILE100"
    for document in (REDESIGN, PORTFOLIO):
        if marker in document.read_text(encoding="utf-8"):
            continue
        with document.open("a", encoding="utf-8") as handle:
            handle.write("\n\n## True TTB profile100 自动结果\n\n")
            handle.write(f"<!-- {marker} -->\n")
            handle.write(f"- artifact: `{SUMMARY_MD.relative_to(REPO)}`\n\n")
            for line in lines[4:]:
                handle.write(line + "\n")


def main() -> int:
    wait_previous()
    if not H68_DEPLOY_SUMMARY.exists():
        run_h68_deploy()
    rows = []
    for label, config, checkpoint, output in CASES:
        profile = run_profile(label, config, checkpoint, output)
        data = json.loads(profile.read_text(encoding="utf-8"))
        for row in data["summary"]["token_time_bundles"]:
            rows.append({"model": label, "profile": str(profile), **row})
    write(rows)
    record(f"ALL COMPLETE TRUE TTB: {SUMMARY_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
