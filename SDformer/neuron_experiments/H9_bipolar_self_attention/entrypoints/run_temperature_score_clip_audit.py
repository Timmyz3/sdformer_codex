"""Profile deployment score clipping for fixed/event dyadic temperature candidates."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from run_h60_family_deploy_eval import best_epoch, make_deploy_config


REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
GEN = EXP / "configs/generated"
RESULTS = EXP / "results"
PY = Path(sys.executable)
STATUS = RESULTS / "temperature_score_clip_audit_status.log"
SUMMARY_JSON = RESULTS / "temperature_score_clip_profile20_20260713.json"
SUMMARY_MD = RESULTS / "temperature_score_clip_profile20_20260713.md"
REDESIGN = REPO / "neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md"
HARDWARE_DOC = REPO / "hw_autoresearch_nts07/docs/46_TTB真实分布周期模型与综合协议.md"


def record(message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    with STATUS.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def cases() -> list[dict]:
    h69_runs = sorted(RESULTS.glob("h69_allbinary_all12_dyadic_temperature_ttx_x*_w720_fastlr_full30_bs8_full30_20260711_setsid"))
    if len(h69_runs) != 1:
        raise RuntimeError(f"expected one promoted H69 run, found {h69_runs}")
    h69 = h69_runs[0]
    h69_name = h69.name.split("_bs8_full30_", 1)[0]
    h70_name = "h70_allbinary_all12_event_selective_ttx_maxshift3_w720_fastlr_full30"
    h70 = RESULTS / f"{h70_name}_bs8_full30_20260711_setsid"
    return [
        {"id": "H69", "name": h69_name, "config": GEN / f"{h69_name}.yml", "run": h69},
        {"id": "H70", "name": h70_name, "config": GEN / f"{h70_name}.yml", "run": h70},
    ]


def complete(profile: Path) -> bool:
    if not profile.exists():
        return False
    try:
        quant = json.loads(profile.read_text(encoding="utf-8"))["summary"]["score_quantization"]
        return int(quant["total"]) > 0
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def audit_log(log: Path) -> None:
    text = log.read_text(encoding="utf-8", errors="ignore")
    required = (
        r"installed ATLIF modules: 105",
        r"installed H60/Shiftmax modules: 12",
        r"load audit: checkpoint_overlay_keys=210, model_overlay_keys=210, missing=0, unexpected=0",
        r"processed 20/20",
    )
    missing = [pattern for pattern in required if re.search(pattern, text) is None]
    if missing:
        raise RuntimeError(f"temperature score-clip audit failed ({missing}): {log}")


def run_case(case: dict) -> dict:
    ranking = case["run"] / "profile_ranking_valid825.md"
    epoch = best_epoch(ranking)
    checkpoint = case["run"] / f"checkpoint_epoch{epoch}.pth"
    deploy = make_deploy_config(case["config"])
    output = RESULTS / f"{case['name']}_temperature_score_clip_profile20"
    profile = output / "nts11_hardware_p0_profile.json"
    log = output / "profile.log"
    if not complete(profile):
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
            "--config", str(deploy), "--checkpoint", str(checkpoint),
            "--output-dir", str(output), "--samples", "20", "--num-workers", "0",
        ]
        record(f"START {case['id']} temperature score-clip profile20: {' '.join(command)}")
        with log.open("a", encoding="utf-8") as handle:
            proc = subprocess.run(command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)
        record(f"END {case['id']} temperature score-clip profile20: exit_code={proc.returncode}")
        if proc.returncode != 0 or not complete(profile):
            raise RuntimeError(f"temperature score-clip profile failed: {log}")
    else:
        record(f"REUSE complete {case['id']} temperature score-clip profile20: {profile}")
    audit_log(log)
    data = json.loads(profile.read_text(encoding="utf-8"))
    return {
        "id": case["id"],
        "epoch": epoch,
        "config": str(deploy),
        "checkpoint": str(checkpoint),
        "profile": str(profile),
        **data["summary"]["score_quantization"],
    }


def write(rows: list[dict]) -> None:
    SUMMARY_JSON.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# H69/H70 deployment score-clipping profile20",
        "",
        "| candidate | best epoch | score elements | clip low | clip high | clip ratio |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['id']} | {row['epoch']} | {row['total']} | {row['clip_low']} | "
            f"{row['clip_high']} | {row['clip_ratio']:.6%} |"
        )
    lines += [
        "",
        "裁剪按量化前 score 严格小于 -2 或大于 2 计数，边界值不计入；该表用于判断固定/动态左移是否需要扩大 score 位宽，不替代 valid825 精度。",
    ]
    body = "\n".join(lines) + "\n"
    SUMMARY_MD.write_text(body, encoding="utf-8")
    marker = "H69_H70_TEMPERATURE_SCORE_CLIP_PROFILE20_20260713"
    for document in (REDESIGN, HARDWARE_DOC):
        if marker in document.read_text(encoding="utf-8"):
            continue
        with document.open("a", encoding="utf-8") as handle:
            handle.write("\n\n## H69/H70 deployment score-clipping profile20 自动结果\n\n")
            handle.write(f"<!-- {marker} -->\n")
            handle.write(f"- artifact: `{SUMMARY_MD.relative_to(REPO)}`\n\n")
            handle.write("\n".join(lines[2:]) + "\n")


def main() -> int:
    rows = [run_case(case) for case in cases()]
    write(rows)
    record(f"ALL COMPLETE H69/H70 TEMPERATURE SCORE CLIP: {SUMMARY_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
