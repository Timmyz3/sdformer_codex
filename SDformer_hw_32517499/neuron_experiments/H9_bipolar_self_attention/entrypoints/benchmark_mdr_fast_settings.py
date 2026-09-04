#!/usr/bin/env python3
"""Benchmark MDR training throughput with short, isolated runs."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[3]
UPSTREAM = REPO / "third_party" / "SDformerFlow"
BASE_CONFIG = REPO / "configs" / "generated" / "train_mdr_baseline_mvsec_route.yml"
TMP_DIR = REPO / "neuron_experiments" / "H9_bipolar_self_attention" / "results" / "mdr_fast_bench_configs"


CANDIDATES = [
    {"name": "paper_bs4_w4_amp0", "batch_size": 4, "n_workers": 4, "use_amp": False},
    {"name": "paper_bs4_w8_amp0", "batch_size": 4, "n_workers": 8, "use_amp": False},
    {"name": "bs8_w8_amp0", "batch_size": 8, "n_workers": 8, "use_amp": False},
    {"name": "bs12_w8_amp0", "batch_size": 12, "n_workers": 8, "use_amp": False},
    {"name": "bs16_w8_amp0", "batch_size": 16, "n_workers": 8, "use_amp": False},
    {"name": "bs16_w4_amp0", "batch_size": 16, "n_workers": 4, "use_amp": False},
    {"name": "bs16_w12_amp0", "batch_size": 16, "n_workers": 12, "use_amp": False},
    {"name": "bs16_w8_amp1", "batch_size": 16, "n_workers": 8, "use_amp": True},
    {"name": "bs24_w8_amp0", "batch_size": 24, "n_workers": 8, "use_amp": False},
    {"name": "bs24_w8_amp1", "batch_size": 24, "n_workers": 8, "use_amp": True},
    {"name": "bs32_w8_amp0", "batch_size": 32, "n_workers": 8, "use_amp": False},
    {"name": "bs32_w8_amp1", "batch_size": 32, "n_workers": 8, "use_amp": True},
]


def write_config(candidate: dict) -> Path:
    with BASE_CONFIG.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["experiment"] = f"sdformer_mdr_fastbench_{candidate['name']}"
    config["loader"]["n_epochs"] = 1
    config["loader"]["batch_size"] = candidate["batch_size"]
    config["loader"]["n_workers"] = candidate["n_workers"]
    config["optimizer"]["use_amp"] = candidate["use_amp"]
    config["vis"]["enabled"] = False
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    path = TMP_DIR / f"{candidate['name']}.yml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return path


def run_candidate(candidate: dict, python_bin: str, max_batches: int) -> dict:
    config_path = write_config(candidate)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "SDFORMER_SNN_BACKEND": "torch",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "KMP_DUPLICATE_LIB_OK": "TRUE",
            "SDFORMER_MDR_DETECT_ANOMALY": "0",
            "SDFORMER_MDR_MAX_TRAIN_BATCHES": str(max_batches),
            "SDFORMER_MDR_SKIP_VALIDATION": "1",
        }
    )
    cmd = [
        python_bin,
        "train_mdr_supervised_SNN.py",
        "--config",
        os.path.relpath(config_path, UPSTREAM),
        "--path_mlflow",
        "file:///root/private_data/sdformer_mlflow_fastbench",
    ]
    started = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=UPSTREAM,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    elapsed = time.monotonic() - started
    output = proc.stdout
    oom = "out of memory" in output.lower() or "cuda error" in output.lower()
    stopped = re.search(r"stopping epoch after (\d+) train batches", output)
    loop_speed = re.search(r"train_samples_per_s=([0-9.]+)", output)
    batches = int(stopped.group(1)) if stopped else 0
    return {
        **candidate,
        "returncode": proc.returncode,
        "elapsed_s": elapsed,
        "batches": batches,
        "samples": batches * candidate["batch_size"],
        "samples_per_s": (batches * candidate["batch_size"] / elapsed) if elapsed > 0 else 0.0,
        "loop_samples_per_s": float(loop_speed.group(1)) if loop_speed else 0.0,
        "oom": oom,
        "config": str(config_path),
        "tail": "\n".join(output.splitlines()[-30:]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-bin", default="/opt/conda/envs/sdformerflow/bin/python")
    parser.add_argument("--max-batches", type=int, default=120)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    candidates = CANDIDATES
    if args.only:
        selected = set(args.only)
        candidates = [item for item in candidates if item["name"] in selected]

    results = []
    for candidate in candidates:
        print(f"[bench] start {candidate['name']}", flush=True)
        result = run_candidate(candidate, args.python_bin, args.max_batches)
        results.append(result)
        print(
            "[bench] "
            f"{result['name']} rc={result['returncode']} "
            f"batches={result['batches']} loop_samples/s={result['loop_samples_per_s']:.2f} "
            f"wall_samples/s={result['samples_per_s']:.2f} "
            f"elapsed={result['elapsed_s']:.1f}s oom={result['oom']}",
            flush=True,
        )
        if result["returncode"] != 0:
            print(result["tail"], flush=True)

    print("\n| candidate | batch | workers | amp | rc | batches | loop samples/s | wall samples/s | elapsed_s |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for result in sorted(results, key=lambda item: item["loop_samples_per_s"], reverse=True):
        print(
            f"| {result['name']} | {result['batch_size']} | {result['n_workers']} | "
            f"{int(result['use_amp'])} | {result['returncode']} | {result['batches']} | "
            f"{result['loop_samples_per_s']:.2f} | {result['samples_per_s']:.2f} | "
            f"{result['elapsed_s']:.1f} |"
        )
    return 0 if all(result["returncode"] == 0 for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
