#!/usr/bin/env python3
"""Finite, fail-closed GPU queue: four paired trials and standard evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml

REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
ROOT = EXP / "results/algorithm_refresh_20260906"
CONFIGS = EXP / "configs/generated/algorithm_refresh_20260906"
PARENT = EXP / "results/dsec_c12_alpha0125_ep29_resume5_20260830/checkpoint_epoch34.pth"
PARENT_SHA = "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48"
BASE = EXP / "configs/generated/dsec_c12_alpha0125_ep29_resume5_20260830.yml"
MODES = ("control", "augment", "distill", "delay")
MANIFEST = CONFIGS / "manifest.json"
TRAIN = EXP / "entrypoints/train_algorithm_refresh.py"
EVAL = EXP / "entrypoints/run_h9_standard_valid825_eval.py"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def log(message):
    line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
    print(line, flush=True)
    with (ROOT / "status.log").open("a") as stream:
        stream.write(line + "\n")


def materialize():
    if MANIFEST.exists():
        raise FileExistsError("manifest exists; refusing to rewrite experiment identities")
    if sha(PARENT) != PARENT_SHA:
        raise RuntimeError("ep34 parent identity mismatch")
    CONFIGS.mkdir(parents=True, exist_ok=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    identities = {str(PARENT): PARENT_SHA, str(BASE): sha(BASE)}
    for mode in MODES:
        for smoke in (False, True):
            cfg = yaml.safe_load(BASE.read_text())
            cfg["experiment"] = f"refresh_{mode}" + ("_smoke" if smoke else "_ft5")
            cfg["loader"]["n_epochs"] = 1 if smoke else 5
            cfg["loader"]["n_workers"] = 8
            cfg["loader"]["prefetch_factor"] = 1
            cfg["loader"]["crop"] = None
            # The parent is already after both Multistep decays. Use its final
            # LR scale in every fresh-optimizer branch, not the original peak LR.
            cfg["optimizer"]["lr"] = 1e-6
            cfg["optimizer"]["milestones"] = [3]
            groups = cfg["optimizer"]["param_groups"]
            groups.update(backbone_lr=1e-6, norm_lr=1e-6, neuron_lr=5e-7,
                          threshold_lr=5e-8)
            cfg["runtime"] = {
                "seed": 0, "allow_tf32": True, "cudnn_benchmark": True,
                "snn_backend": "cupy", "epoch_offset": 35,
                "max_train_steps": 8 if smoke else 0,
                "skip_save": False, "skip_state_save": bool(smoke),
                "save_only_force_epochs": True,
                "force_save_epochs": [0] if smoke else [2, 4],
                "state_save_epochs": [4], "use_mlflow_model_logging": False,
                "physical_batch": 2, "gradient_accumulation": 1,
                "full_resolution_protocol": "refresh_ep34_paired_fresh_optimizer_480x640_w2x15x15",
                "parent_checkpoint": str(PARENT), "parent_checkpoint_sha256": PARENT_SHA,
                "resume_protocol": "fresh_optimizer_from_ep34_not_true_optimizer_resume",
            }
            cfg["algorithm_refresh"] = {
                "mode": mode, "drop_rate": 0.05, "sample_probability": 0.5,
                "distill_weight": 0.1, "teacher_max_epe": 1.0,
                "warmup_steps": 1 if smoke else 200,
            }
            cfg["note"] = (
                "September paired train-only trial. Dense C12 inference export; "
                "not a new RTL-exact checkpoint. Frozen teacher uses clean input, "
                "no_running BN, batch1. No official DSEC test."
            )
            path = CONFIGS / (mode + ("_smoke" if smoke else "") + ".yml")
            if path.exists():
                raise FileExistsError(path)
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            identities[str(path)] = sha(path)
    sources = [Path(__file__), TRAIN, EVAL, EXP / "entrypoints/train.py",
               REPO / "third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py"]
    # Pin overlay sources consulted by loading, forward, loss and optimization.
    sources += list((EXP / "overlay/models/STSwinNet_SNN").glob("*.py"))
    sources += list((EXP / "overlay/models/STSwinNet_SNN/atlif_ternary_psn").glob("*.py"))
    for path in sources:
        identities[str(path)] = sha(path)
    write_json(MANIFEST, {"schema": "algorithm_refresh_queue_v1", "identities": identities,
                         "modes": MODES, "epochs": [37, 39], "parent": str(PARENT),
                         "status": "prepared_not_launched"})
    print(MANIFEST)


def verify_identity():
    manifest = json.loads(MANIFEST.read_text())
    for path, expected in manifest["identities"].items():
        if sha(path) != expected:
            raise RuntimeError(f"source/config/checkpoint changed: {path}")


def wait_idle():
    for _ in range(2):
        while True:
            result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
                                     "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True, check=True)
            if not result.stdout.strip():
                break
            log("WAIT GPU busy; no existing process will be interrupted")
            time.sleep(60)
        time.sleep(5)


def environment():
    env = os.environ.copy()
    env.update(SDFORMER_USE_MLFLOW="0", SDFORMER_MLFLOW_MODEL_LOGGING="0",
               SDFORMER_SNN_BACKEND="cupy", PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    return env


def run_once(command, folder, phase):
    verify_identity()
    folder.mkdir(parents=True, exist_ok=True)
    done = folder / (phase + ".done.json")
    if done.exists():
        record = json.loads(done.read_text())
        if record["command"] != command or record["manifest_sha256"] != sha(MANIFEST):
            raise RuntimeError("completed job identity mismatch")
        return
    if shutil.disk_usage(REPO).free < 8 * 1024**3:
        raise RuntimeError("less than 8 GiB free; no automatic deletion")
    wait_idle()
    attempt = folder / (phase + ".attempt.json")
    with attempt.open("x") as stream:
        json.dump({"command": command, "manifest_sha256": sha(MANIFEST)}, stream)
    log(f"START {phase} {folder.name}")
    with (folder / (phase + ".log")).open("x") as stream:
        child = subprocess.Popen(command, cwd=REPO, env=environment(),
                                 stdout=stream, stderr=subprocess.STDOUT)
        write_json(ROOT / "current.json", {"phase": phase, "folder": str(folder),
                   "pid": child.pid, "command": command, "status": "running"})
        code = child.wait()
    if code:
        raise RuntimeError(f"{folder.name}/{phase} exit={code}; no automatic retry")
    if phase == "train":
        # Scan lines without loading a large progress log into RAM.
        required = {"checkpoint_overlay_keys=210, missing=0, unexpected=0",
                    "installed ATLIFTernaryPSN before load: 105 modules",
                    "installed attention before load: 12 modules"}
        with (folder / "train.log").open(errors="replace") as stream:
            for line in stream:
                required = {marker for marker in required if marker not in line}
        if required:
            raise RuntimeError(f"training load audit missing: {required}")
    write_json(done, {"command": command, "manifest_sha256": sha(MANIFEST), "exit_code": code})
    log(f"PASS {phase} {folder.name}")


def train_command(mode, smoke):
    name = mode + ("_smoke" if smoke else "")
    folder = ROOT / name
    return [sys.executable, "-u", str(TRAIN), "--config", str(CONFIGS / (name + ".yml")),
            "--prev_runid", str(PARENT), "--save_path", str(folder / "checkpoint_epoch{}.pth"),
            "--finetune", "1"], folder


def evaluate(mode):
    from run_dsec_c12_alpha0125_resume5_20260830 import parse_profile
    folder = ROOT / mode
    command = [sys.executable, "-u", str(EVAL), "--config", str(CONFIGS / (mode + ".yml")),
               "--run-dir", str(folder), "--ranking-mode", "aee", "--epoch", "37", "--epoch", "39"]
    run_once(command, folder, "valid825")
    return [{"epoch": epoch, **parse_profile(folder / "standard_valid825" /
             f"epoch{epoch}" / "spike_profile.json")} for epoch in (37, 39)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--smoke", choices=MODES)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        materialize()
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if args.smoke:
                command, folder = train_command(args.smoke, True)
                run_once(command, folder, "train")
                return
            if not args.run:
                parser.error("choose --prepare, --smoke MODE, or --run")
            if not (ROOT / "smoke_acceptance.json").exists():
                raise RuntimeError("GPU smoke acceptance is required before formal queue")
            acceptance = json.loads((ROOT / "smoke_acceptance.json").read_text())
            if acceptance.get("manifest_sha256") != sha(MANIFEST) or not acceptance.get("accepted"):
                raise RuntimeError("smoke acceptance identity mismatch")
            rows = {}
            for mode in MODES:
                command, folder = train_command(mode, False)
                run_once(command, folder, "train")
                rows[mode] = evaluate(mode)
                write_json(ROOT / "summary.json", rows)
            write_json(ROOT / "current.json", {"status": "complete", "modes": MODES})
            log("ALL COMPLETE; inspect paired metrics, do not auto-promote a checkpoint")
        except Exception as error:
            write_json(ROOT / "current.json", {"status": "failed", "error": repr(error)})
            log(f"FAILED {error}")
            raise


if __name__ == "__main__":
    main()
