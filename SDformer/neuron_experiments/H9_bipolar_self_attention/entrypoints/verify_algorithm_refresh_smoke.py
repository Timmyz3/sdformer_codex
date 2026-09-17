#!/usr/bin/env python3
"""Admit the finite queue only after real train/export/eval smoke checks."""
import json
import math
from pathlib import Path
import subprocess
import sys

import torch

import run_algorithm_refresh_20260906 as queue


def main():
    torch.set_num_threads(4)
    queue.verify_identity()
    parent = torch.load(queue.PARENT, map_location="cpu", weights_only=False)["model_state_dict"]
    evidence = {}
    for mode in ("distill", "delay"):
        folder = queue.ROOT / (mode + "_smoke")
        done = json.loads((folder / "train.done.json").read_text())
        if done["manifest_sha256"] != queue.sha(queue.MANIFEST):
            raise RuntimeError("stale smoke training")
        ckpt = folder / "checkpoint_epoch35.pth"
        state = torch.load(ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
        if state.keys() != parent.keys():
            raise RuntimeError("export key set is not the original C12 schema")
        if not all(state[k].shape == parent[k].shape and torch.isfinite(v).all()
                   for k, v in state.items()):
            raise RuntimeError("export shape/finite audit failed")
        stats = []
        with (folder / "train.log").open(errors="replace") as stream:
            for line in stream:
                if "[REFRESH] {" in line:
                    stats.append(json.loads(line.split("[REFRESH] ", 1)[1]))
        if mode == "distill" and not any(
                row.get("kd_pixels", 0) > 0 and row.get("kd_raw", 0) > 0 for row in stats):
            raise RuntimeError("distillation path produced no learning signal")
        if mode == "delay" and not any(row.get("delay_coefficient_absmax", 0) > 0 for row in stats):
            raise RuntimeError("delay coefficients did not update")
        out = folder / "eval_smoke4"
        out.mkdir(exist_ok=True)
        command = [sys.executable, "-u", "eval_DSEC_flow_SNN.py", "--config",
                   str(queue.CONFIGS / (mode + "_smoke.yml")), "--checkpoint", str(ckpt),
                   "--path_results", str(out), "--max-samples", "4", "--mode", "valid"]
        if not (out / "spike_profile.json").exists():
            queue.wait_idle()
            with (out / "attempt.json").open("x") as stream:
                json.dump({"command": command, "checkpoint_sha256": queue.sha(ckpt)}, stream)
            with (out / "eval.log").open("x") as stream:
                subprocess.run(command, cwd=queue.REPO / "third_party/SDformerFlow",
                               env=queue.environment(), stdout=stream,
                               stderr=subprocess.STDOUT, check=True)
        profile = json.loads((out / "spike_profile.json").read_text())
        identity = profile["artifact_identity"]
        if (identity["checkpoint_sha256"] != queue.sha(ckpt)
                or identity["config_sha256"] != queue.sha(queue.CONFIGS / (mode + "_smoke.yml"))):
            raise RuntimeError("cannot reuse a smoke profile with different identity")
        counts, audit = profile["module_counts"], profile["checkpoint_load_audit"]
        if (profile["samples"] != 4 or counts["ATLIFTernaryPSN"] != 105
                or counts["ShiftmaxAttention"] != 12 or audit["checkpoint_overlay_keys"] != 210
                or audit["missing_count"] != 0 or audit["unexpected_count"] != 0
                or not math.isfinite(float(profile["metrics"]["AEE"]))):
            raise RuntimeError("standard entrypoint smoke audit failed")
        evidence[mode] = {"checkpoint_sha256": queue.sha(ckpt),
                          "profile_sha256": queue.sha(out / "spike_profile.json"),
                          "profile": str(out / "spike_profile.json"),
                          "samples": 4, "last_training_stats": stats[-1]}
        print(f"PASS {mode}: dense export, live gradients, standard load, 4-sample forward", flush=True)
    queue.write_json(queue.ROOT / "smoke_acceptance.json", {
        "accepted": True, "manifest_sha256": queue.sha(queue.MANIFEST),
        "verifier_sha256": queue.sha(Path(__file__)), "evidence": evidence,
        "claim_boundary": "train/export/load smoke only; not accuracy or RTL evidence"})


if __name__ == "__main__":
    main()
