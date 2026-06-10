"""H56a autopilot: sweep → rank → promote best to full30.

Phase 1: λ sweep (4 configs, slowbb, tr=0.05, 360-step + valid40)
  → find best λ

Phase 2: LR sweep (4 configs, best λ, tr=0.05, 360-step + valid40)
  → find best LR strategy

Phase 3: target_rate confirm (2 configs, best λ+LR, 360-step + valid40)
  → confirm tr=0.05 vs 0.07

Phase 4: Promote best config to full30

Composite score (lower is better):
  score = AEE + 0.035 * AAE + 0.25 * max(0, SOPs - 3.20)

Gates: AEE < 2.1, AAE < 12.0, SOPs < 3.8G, worst_pos_neg_ratio < 500
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

EXP_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = EXP_ROOT / "results"
CONFIG_DIR = EXP_ROOT / "configs" / "generated"
BASELINE_CKPT = (
    "/root/private_data/work/sdformer_codex/SDformer/experiments/checkpoints/"
    "bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth"
)
TRAIN_ENTRY = str(EXP_ROOT / "entrypoints" / "train.py")
PROFILE_SCRIPT = str(EXP_ROOT / "entrypoints" / "profile_sops.py")
CONDA_PREFIX = (
    "export SDFORMER_USE_MLFLOW=0 && "
    "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
    "source /opt/conda/etc/profile.d/conda.sh && "
    "conda activate sdformerflow"
)
AUTOPILOT_DIR = RESULTS_ROOT / f"h56a_autopilot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# ── Sweep definitions ──
PHASE1_LAMBDAS = [0.3, 0.5, 0.8, 1.0]
PHASE1_LR = "slowbb"
PHASE1_TR = 0.05

PHASE2_LRS = ["slowbb", "fast", "warm", "fast_warm"]
# best_lambda filled after phase 1

CORE_CFG = {
    "bsa_attention": {
        "mode": "sc_agree_disagree_shiftmax",
        "deadzone_epsilon": 0.0,
        "confidence_enabled": False,
        "k_consistency_mod": False,
        "consensus_score_norm": "head_dim",
    }
}


def _config_name(lam: float, lr: str, tr: float, suffix: str = "steps360") -> str:
    return f"h56a_swp_l{int(lam*10):02d}_{lr}_tr{int(tr*100):02d}_{suffix}"


def _config_path(name: str) -> Path:
    return CONFIG_DIR / f"{name}.yml"


def run(cmd: str, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(
        f"{CONDA_PREFIX} && {cmd}",
        shell=True,
        executable="/bin/bash",
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(EXP_ROOT),
    )


def composite_score(aee: float, aae: float, sops: float) -> float:
    return aee + 0.035 * aae + 0.25 * max(0.0, sops - 3.20)


def profile_checkpoint(checkpoint: str, run_dir: str, config_path: str, num_samples: int = 40) -> dict[str, Any] | None:
    cmd = (
        f"python {PROFILE_SCRIPT} "
        f"--checkpoint {checkpoint} "
        f"--config {config_path} "
        f"--output-dir {run_dir} "
        f"--split valid "
        f"--num-samples {num_samples} "
        f"--batch-size 4 "
        f"--num-workers 4"
    )
    result = run(cmd, timeout=1200)
    sops_path = Path(run_dir) / "sops_summary.json"
    if sops_path.exists():
        with open(sops_path) as f:
            return json.load(f)
    print(f"  [PROFILE_FAIL] stderr: {result.stderr[-500:]}")
    return None


def train_one(config_name: str, run_dir: str) -> bool:
    config_path = _config_path(config_name)
    if not config_path.exists():
        print(f"  [SKIP] config {config_name} not found")
        return False
    save_path = str(Path(run_dir) / "checkpoint_epoch{}.pth")
    cmd = (
        f"python -u {TRAIN_ENTRY} "
        f"--config {config_path} "
        f"--prev_runid {BASELINE_CKPT} "
        f"--save_path {save_path}"
    )
    print(f"  [TRAIN] {config_name}")
    result = run(cmd, timeout=1200)
    if result.returncode != 0:
        print(f"  [FAIL] {config_name}: {result.stderr[-200:]}")
        return False
    return True


def run_phase(
    name: str,
    configs: list[str],
    sweep_dir: Path,
) -> list[dict[str, Any]]:
    """Train each config for 360 steps, profile valid40, return ranked results."""
    results: list[dict[str, Any]] = []
    for i, config_name in enumerate(configs):
        print(f"\n[{name}] {i+1}/{len(configs)}: {config_name}")
        run_dir = sweep_dir / config_name
        run_dir.mkdir(parents=True, exist_ok=True)

        if not train_one(config_name, str(run_dir)):
            continue

        ckpt = run_dir / "checkpoint_epoch0.pth"
        if not ckpt.exists():
            print(f"  [SKIP] no checkpoint for {config_name}")
            continue

        profile = profile_checkpoint(str(ckpt), str(run_dir), str(_config_path(config_name)), num_samples=40)
        if profile is None:
            print(f"  [SKIP] no profile for {config_name}")
            continue

        aee = float(profile.get("AEE", 99.0))
        aae = float(profile.get("AAE", 99.0))
        sops = float(profile.get("total_sops", 9.0))
        firing = float(profile.get("total_firing_rate", 0.0))
        score = composite_score(aee, aae, sops)

        print(f"  AEE={aee:.4f} AAE={aae:.4f} SOPs={sops:.4f}G score={score:.4f}")
        results.append(
            {
                "config": config_name,
                "AEE": aee,
                "AAE": aae,
                "SOPs": sops,
                "firing": firing,
                "score": score,
                "run_dir": str(run_dir),
            }
        )
    results.sort(key=lambda r: r["score"])
    return results


def main() -> int:
    print(f"=== H56a autopilot ===")
    print(f"Results dir: {AUTOPILOT_DIR}")
    AUTOPILOT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: λ sweep ──
    print("\n── Phase 1: λ sweep (slowbb, tr=0.05) ──")
    phase1_configs = [
        _config_name(lam, PHASE1_LR, PHASE1_TR) for lam in PHASE1_LAMBDAS
    ]
    phase1_dir = AUTOPILOT_DIR / "phase1_lambda_sweep"
    phase1_dir.mkdir(parents=True, exist_ok=True)
    phase1_results = run_phase("Phase1-λ", phase1_configs, phase1_dir)

    if not phase1_results:
        print("[FATAL] Phase 1 produced no results")
        return 1

    best_lam = float(phase1_results[0]["config"].split("_l")[1][:2]) / 10.0
    print(f"\nBest λ = {best_lam} (score={phase1_results[0]['score']:.4f})")

    # ── Phase 2: LR sweep ──
    print(f"\n── Phase 2: LR sweep (λ={best_lam}, tr=0.05) ──")
    phase2_configs = [
        _config_name(best_lam, lr, PHASE1_TR) for lr in PHASE2_LRS
    ]
    phase2_dir = AUTOPILOT_DIR / "phase2_lr_sweep"
    phase2_dir.mkdir(parents=True, exist_ok=True)
    phase2_results = run_phase("Phase2-LR", phase2_configs, phase2_dir)

    if not phase2_results:
        print("[FATAL] Phase 2 produced no results")
        return 1

    best_lr = phase2_results[0]["config"].split("_")[3]
    print(f"Best LR = {best_lr} (score={phase2_results[0]['score']:.4f})")

    # ── Phase 3: target_rate confirm ──
    print(f"\n── Phase 3: target_rate confirm (λ={best_lam}, LR={best_lr}) ──")
    phase3_configs = [
        _config_name(best_lam, best_lr, tr) for tr in [0.05, 0.07]
    ]
    phase3_dir = AUTOPILOT_DIR / "phase3_target_rate"
    phase3_dir.mkdir(parents=True, exist_ok=True)
    phase3_results = run_phase("Phase3-tr", phase3_configs, phase3_dir)

    # ── Phase 4: Promote best to full30 ──
    all_results = phase1_results + phase2_results + (phase3_results or [])
    all_results.sort(key=lambda r: r["score"])

    print("\n── Full ranking ──")
    for i, r in enumerate(all_results[:10]):
        print(
            f"  {i+1}. {r['config']}: "
            f"AEE={r['AEE']:.4f} AAE={r['AAE']:.4f} "
            f"SOPs={r['SOPs']:.4f}G score={r['score']:.4f}"
        )

    best = all_results[0]
    best_full_config = best["config"].replace("_steps360", "_full30")
    print(f"\n── Promote: {best_full_config} ──")

    full_config_path = _config_path(best_full_config)
    if not full_config_path.exists():
        print(f"[FATAL] Full config {best_full_config} not found")
        return 1

    full_dir = RESULTS_ROOT / best_full_config.replace(".yml", "")
    full_dir.mkdir(parents=True, exist_ok=True)

    save_path = str(full_dir / "checkpoint_epoch{}.pth")
    log_path = full_dir / "train.log"
    cmd = (
        f"nohup python -u {TRAIN_ENTRY} "
        f"--config {full_config_path} "
        f"--prev_runid {BASELINE_CKPT} "
        f"--save_path {save_path} "
        f"> {log_path} 2>&1 &"
    )
    print(f"  Launch: {cmd}")
    subprocess.Popen(
        f"{CONDA_PREFIX} && {cmd}",
        shell=True,
        executable="/bin/bash",
        cwd=str(EXP_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Write summary
    summary_path = AUTOPILOT_DIR / "autopilot_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "best_config": best_full_config,
                "best_score": best["score"],
                "best_AEE": best["AEE"],
                "best_AAE": best["AAE"],
                "best_SOPs": best["SOPs"],
                "best_lam": best_lam,
                "best_lr": best_lr,
                "phase1_results": phase1_results,
                "phase2_results": phase2_results,
                "phase3_results": phase3_results,
            },
            f,
            indent=2,
        )
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
