#!/usr/bin/env python3
"""Fail-closed result audit for H67 Motion versus H81 no-motion control."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time


REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
GEN = EXP / "configs/generated"
RESULTS = EXP / "results"
H81_ROOT = RESULTS / "dsec_fullres_w15_H81_nomotion_bb1e4_ft40_20260811"
H81_CONFIG = GEN / "dsec_fullres_w15_H81_nomotion_bb1e4_ft40.yml"
H67_ROOT = RESULTS / "dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_20260805"
H67_CONFIG = GEN / "dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40.yml"
H67_EPOCH = 35
FAIRNESS = REPO / "neuron_autoresearch/H67_H81_TRAINING_FAIRNESS_20260812.json"
OUTPUT = REPO / "neuron_autoresearch/H67_H81_NOMOTION_RESULT_20260812.json"
OUTPUT_MD = OUTPUT.with_suffix(".md")
REDESIGN = REPO / "neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md"
RANK_RE = re.compile(r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|")
EXPECTED_LIST_SHA = "7f3dc2800653e12caca10379c51ee8e8988aaf6bb80c391224a454a5879325d0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binding(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256(path), "size_bytes": path.stat().st_size}


def parse_ranking(path: Path) -> list[dict[str, int]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = RANK_RE.match(line)
        if match:
            rows.append({"rank": int(match.group(1)), "epoch": int(match.group(2))})
    if not rows or [row["rank"] for row in rows] != list(range(1, len(rows) + 1)):
        raise RuntimeError(f"invalid ranking: {path}")
    return rows


def profile(root: Path, config: Path, epoch: int) -> dict[str, object]:
    checkpoint = root / f"checkpoint_epoch{epoch}.pth"
    path = root / f"standard_valid825/epoch{epoch}/spike_profile.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    protocol = raw.get("eval_protocol") or {}
    identity = raw.get("artifact_identity") or {}
    load = raw.get("checkpoint_load_audit") or {}
    counts = raw.get("module_counts") or {}
    metrics = raw.get("metrics") or {}
    validation = raw.get("validation_file_list") or {}
    aggregation = raw.get("metric_aggregation_audit") or {}
    checks = {
        "resolution": protocol.get("resolution") == [480, 640],
        "crop": protocol.get("crop") is None,
        "window": protocol.get("window_size") == [2, 15, 15],
        "bn": protocol.get("bn_policy") == "no_running",
        "batch": int(protocol.get("eval_batch_size", 0)) == 1,
        "samples825": int(raw.get("samples", 0)) == 825,
        "sequences18": int(aggregation.get("sequence_count", 0)) == 18,
        "validation SHA": validation.get("sha256") == EXPECTED_LIST_SHA,
        "checkpoint path": Path(identity.get("checkpoint_path", "")).resolve() == checkpoint.resolve(),
        "checkpoint SHA": identity.get("checkpoint_sha256") == sha256(checkpoint),
        "config path": Path(identity.get("config_path", "")).resolve() == config.resolve(),
        "config SHA": identity.get("config_sha256") == sha256(config),
        "overlay210": load.get("checkpoint_overlay_keys") == 210 and load.get("model_overlay_keys") == 210,
        "missing0": load.get("missing_count") == 0,
        "unexpected0": load.get("unexpected_count") == 0,
        "ATLIF105": counts.get("ATLIFTernaryPSN") == 105,
        "Shiftmax12": counts.get("ShiftmaxAttention") == 12,
        "metrics": all(key in metrics for key in ("AEE", "AAE", "AAE_Benchmark", "DSEC_Fl")),
        "finite": all(math.isfinite(float(metrics[key])) for key in ("AEE", "AAE", "AAE_Benchmark", "DSEC_Fl")),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"profile contract failed {path}: {failed}")
    return {
        "epoch": epoch,
        "AEE": float(metrics["AEE"]),
        "AAE_2D": float(metrics["AAE"]),
        "AE_3D": float(metrics["AAE_Benchmark"]),
        "DSEC_Fl": float(metrics["DSEC_Fl"]),
        "total_spikes_g": float(raw["total_spikes"]) / 1e9,
        "energy_proxy_uj": float(raw["energy_uj"]),
        "checkpoint": binding(checkpoint),
        "profile": binding(path),
        "checks": checks,
    }


def change(reference: float, candidate: float) -> float:
    return 100.0 * (candidate - reference) / reference


def run() -> dict[str, object]:
    fairness = json.loads(FAIRNESS.read_text(encoding="utf-8"))
    if fairness.get("status") != "PASS_RECIPE_LEVEL_CONTROL_NOT_STEP_PAIRED":
        raise RuntimeError("fairness receipt not qualified")
    h81_ranking_path = H81_ROOT / "profile_ranking_valid825.md"
    ranking = parse_ranking(h81_ranking_path)
    observed = {row["epoch"] for row in ranking}
    if observed != {29, 34, 39}:
        raise RuntimeError(f"H81 ranking epochs mismatch: {observed}")
    h81_epoch = ranking[0]["epoch"]
    h67 = profile(H67_ROOT, H67_CONFIG, H67_EPOCH)
    h81 = profile(H81_ROOT, H81_CONFIG, h81_epoch)
    metrics = ("AEE", "AAE_2D", "AE_3D", "DSEC_Fl", "total_spikes_g", "energy_proxy_uj")
    delta = {name: change(float(h81[name]), float(h67[name])) for name in metrics}
    result = {
        "schema": "h67_h81_nomotion_result_audit_v1",
        "status": "PASS_PROTOCOL_AND_IDENTITY",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": (
            "negative percentage means H67 Motion is lower/better than H81 no-motion. "
            "H67 paper checkpoint is the later plus10 ep35; H81 is an uninterrupted "
            "40-epoch fullres run. This is not a same-optimizer-step ablation."
        ),
        "fairness_scope": "recipe_level_control_not_step_paired",
        "h67_motion": h67,
        "h81_no_motion": h81,
        "H67_change_pct_vs_H81": delta,
        "motion_improves_AEE": float(h67["AEE"]) < float(h81["AEE"]),
        "h81_convergence": (
            "not_plateaued"
            if h81_epoch == 39
            else "operationally_plateaued_or_overfit"
        ),
        "ranking": binding(h81_ranking_path),
        "fairness_receipt": binding(FAIRNESS),
        "claim_boundary": (
            "algorithm recipe-level no-motion control; no H81 hardware provenance; "
            "no step-paired causal claim; H67 ep35 includes a plus10 continuation that "
            "H81 does not pair"
        ),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=300)
    args = parser.parse_args()
    ranking = H81_ROOT / "profile_ranking_valid825.md"
    while not ranking.is_file():
        if not args.wait:
            raise FileNotFoundError(ranking)
        print(f"WAIT {ranking}", flush=True)
        time.sleep(args.poll_seconds)
    result = run()
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    h67 = result["h67_motion"]
    h81 = result["h81_no_motion"]
    delta = result["H67_change_pct_vs_H81"]
    OUTPUT_MD.write_text(
        "\n".join(
            (
                "# H67 Motion versus H81 no-motion control",
                "",
                f"Status: `{result['status']}`; H81 convergence: `{result['h81_convergence']}`.",
                "",
                "| route | epoch | AEE | AAE-2D | AE-3D | Fl(%) | spikes(G) | energy proxy(uJ) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
                f"| H67 Motion | {h67['epoch']} | {h67['AEE']:.6f} | {h67['AAE_2D']:.6f} | {h67['AE_3D']:.6f} | {h67['DSEC_Fl']:.4f} | {h67['total_spikes_g']:.4f} | {h67['energy_proxy_uj']:.2f} |",
                f"| H81 no-motion | {h81['epoch']} | {h81['AEE']:.6f} | {h81['AAE_2D']:.6f} | {h81['AE_3D']:.6f} | {h81['DSEC_Fl']:.4f} | {h81['total_spikes_g']:.4f} | {h81['energy_proxy_uj']:.2f} |",
                "",
                f"H67 AEE change versus H81: `{delta['AEE']:+.3f}%`; negative is better.",
                "",
                "This is a same-parent/seed/recipe control, not a bit-exact step-paired training trajectory. H81 has no inherited hardware provenance.",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    marker = "H67_H81_NOMOTION_FINAL_RESULT_20260812"
    for path in (REDESIGN,):
        text = path.read_text(encoding="utf-8")
        if marker in text:
            continue
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                "\n\n"
                f"<!-- {marker} -->\n\n"
                "### H67 Motion vs H81 no-motion 最终控制\n\n"
                f"- H67 Motion ep{h67['epoch']}: AEE=`{h67['AEE']:.6f}`，"
                f"AAE-2D=`{h67['AAE_2D']:.6f}`，AE-3D=`{h67['AE_3D']:.6f}`，"
                f"spikes=`{h67['total_spikes_g']:.4f}G`。\n"
                f"- H81 no-motion ep{h81['epoch']}: AEE=`{h81['AEE']:.6f}`，"
                f"AAE-2D=`{h81['AAE_2D']:.6f}`，AE-3D=`{h81['AE_3D']:.6f}`，"
                f"spikes=`{h81['total_spikes_g']:.4f}G`。\n"
                f"- H67 AEE 相对 H81 变化=`{delta['AEE']:+.3f}%`（负值为更好）；"
                f"H81 收敛判定=`{result['h81_convergence']}`。"
                "该证据是 recipe-level control，不是 step-paired bit-exact 训练。\n"
                f"- 机器审计：`{OUTPUT.relative_to(REPO)}`。\n"
            )
    print(f"PASS H67/H81 result audit: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
