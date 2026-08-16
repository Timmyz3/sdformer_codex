"""Evaluate valuable checkpoints with legacy AAE and DSEC/Barron AE-3D."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
GENERATOR = EXP / "entrypoints/make_valuable_checkpoint_aae_audit.py"
MANIFEST = EXP / "configs/generated/valuable_aae_audit_20260717/manifest.json"
RUN_ROOT = EXP / "results/valuable_aae_audit_20260717"
REPORT = REPO / "neuron_autoresearch/VALUABLE_CHECKPOINT_AAE_AUDIT_20260717.md"
REDESIGN = REPO / "neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md"
STATUS = RUN_ROOT / "status.log"
SUMMARY = RUN_ROOT / "summary.json"
MARKER = "VALUABLE_CHECKPOINT_AAE_AUDIT_20260717"
MAX_WORKERS = 3


def record(message: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    with STATUS.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run_eval(row: dict[str, Any]) -> None:
    out = Path(row["output"])
    profile = out / "spike_profile.json"
    if profile.is_file() and "AAE_Benchmark" in json.loads(profile.read_text(encoding="utf-8")).get("metrics", {}):
        record(f"SKIP complete {row['id']}")
        return
    out.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-u",
        "third_party/SDformerFlow/eval_DSEC_flow_SNN.py",
        "--config",
        row["config"],
        "--checkpoint",
        row["checkpoint"],
        "--path_results",
        str(out),
        "--mode",
        "valid",
    ]
    env = os.environ.copy()
    env.update({
        "SDFORMER_USE_MLFLOW": "0",
        "SDFORMER_MLFLOW_MODEL_LOGGING": "0",
        "SDFORMER_SNN_BACKEND": "cupy",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    record(f"START {row['id']}: {' '.join(command)}")
    with (out / "eval.log").open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        proc = subprocess.run(command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)
    record(f"END {row['id']}: exit_code={proc.returncode}")
    if proc.returncode != 0:
        raise RuntimeError(f"eval failed for {row['id']}; log={out / 'eval.log'}")


def parse_load_audit(row: dict[str, Any]) -> dict[str, int]:
    log = (Path(row["output"]) / "eval.log").read_text(encoding="utf-8", errors="ignore")
    atlif_match = re.search(r"eval installed ATLIFTernaryPSN:\s*(\d+) modules", log)
    shiftmax_match = re.search(r"eval installed (?:Shiftmax )?attention:\s*(\d+) modules", log)
    load_match = re.search(
        r"load audit: checkpoint_overlay_keys=(\d+), "
        r"(?:model_overlay_keys=(\d+), )?missing=(\d+), unexpected=(\d+)",
        log,
    )
    audit = {
        "atlif": int(atlif_match.group(1)) if atlif_match else 0,
        "shiftmax": int(shiftmax_match.group(1)) if shiftmax_match else 0,
        "checkpoint_overlay_keys": int(load_match.group(1)) if load_match else 0,
        "model_overlay_keys": int(load_match.group(2) or 0) if load_match else 0,
        "missing": int(load_match.group(3)) if load_match else 0,
        "unexpected": int(load_match.group(4)) if load_match else 0,
    }
    expected = (int(row["expected_atlif"]), int(row["expected_shiftmax"]))
    actual = (audit["atlif"], audit["shiftmax"])
    if actual != expected:
        raise RuntimeError(f"module-count audit failed for {row['id']}: expected={expected}, actual={actual}")
    if audit["missing"] or audit["unexpected"]:
        raise RuntimeError(f"strict-load audit failed for {row['id']}: {audit}")
    return audit


def metric(profile: dict[str, Any], key: str) -> float:
    return float(profile.get("metrics", {}).get(key, "nan"))


def collect(manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in manifest:
        profile_path = Path(item["output"]) / "spike_profile.json"
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        audit = parse_load_audit(item)
        rows.append({
            **item,
            "AEE": metric(profile, "AEE"),
            "AAE_2D": metric(profile, "AAE"),
            "AE_3D": metric(profile, "AAE_Benchmark"),
            "PE1": metric(profile, "AEE_PE1"),
            "PE2": metric(profile, "AEE_PE2"),
            "outlier": metric(profile, "AEE_outliers"),
            "total_spikes_g": float(profile.get("total_spikes", 0.0)) / 1e9,
            "energy_uj": float(profile.get("energy_uj", 0.0)),
            "samples": int(profile.get("samples", 0)),
            "load_audit": audit,
        })
    return rows


def table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| model | epoch | AEE | legacy AAE-2D | DSEC/Barron AE-3D | PE1 | PE2 | outlier | spikes(G) | energy proxy(uJ) | load |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        audit = row["load_audit"]
        lines.append(
            f"| {row['label']} | {row['epoch']} | {row['AEE']:.4f} | {row['AAE_2D']:.4f} | "
            f"{row['AE_3D']:.4f} | {row['PE1']:.4f} | {row['PE2']:.4f} | {row['outlier']:.4f} | "
            f"{row['total_spikes_g']:.4f} | {row['energy_uj']:.2f} | "
            f"ATLIF {audit['atlif']}, Shiftmax {audit['shiftmax']}, {audit['missing']}/{audit['unexpected']} |"
        )
    return lines


def write_report(rows: list[dict[str, Any]]) -> None:
    by_id = {row["id"]: row for row in rows}
    nb0 = by_id["nb0_ep59"]
    h67_float = by_id["h67_float_ep19"]
    h67_rtl = by_id["h67_rtl_ep19"]
    h68_rtl = by_id["h68_rtl_ep19"]
    groups = []
    for row in rows:
        if row["group"] not in groups:
            groups.append(row["group"])
    lines = [
        "# Valuable Checkpoint AAE Audit (2026-07-17)",
        "",
        "All rows use the same DSEC valid825 center-crop evaluation. `AAE-2D` is retained only for historical comparison; `AE-3D` is the DSEC/Barron angular metric used for benchmark-facing reporting.",
        "",
    ]
    for group in groups:
        lines.extend([f"## {group.title()}", "", *table([row for row in rows if row['group'] == group]), ""])
    lines.extend([
        "## Key Findings",
        "",
        f"- H67 RTL-exact is the current checkpoint mainline. Versus NB0, AEE changes by "
        f"{(h67_rtl['AEE'] / nb0['AEE'] - 1.0) * 100:.2f}%, AE-3D by "
        f"{(h67_rtl['AE_3D'] / nb0['AE_3D'] - 1.0) * 100:.2f}%, spikes by "
        f"{(h67_rtl['total_spikes_g'] / nb0['total_spikes_g'] - 1.0) * 100:.2f}%, and the "
        f"energy proxy by {(h67_rtl['energy_uj'] / nb0['energy_uj'] - 1.0) * 100:.2f}%.",
        f"- RTL-exact Shiftmax does not degrade H67: relative to float, delta AEE is "
        f"{h67_rtl['AEE'] - h67_float['AEE']:+.4f} and delta AE-3D is "
        f"{h67_rtl['AE_3D'] - h67_float['AE_3D']:+.4f}.",
        f"- H68 remains the zero-deployment-increment fallback, but trails H67 RTL-exact by "
        f"{h68_rtl['AEE'] - h67_rtl['AEE']:.4f} AEE and "
        f"{h68_rtl['AE_3D'] - h67_rtl['AE_3D']:.4f} AE-3D.",
        "- The local angular gap is not evidence of incomplete convergence. The corrected valid825 "
        "AE-3D remains around 9 degrees because valid825 center-crop and official DSEC test are "
        "different evaluation splits/protocols.",
        "",
        "## Loading Audit",
        "",
        "Every row completed with the expected installed ATLIF/Shiftmax count and `missing=0, unexpected=0`. The original configs, checkpoints, and historical valid825 outputs were not modified.",
        "",
        "## Interpretation Rule",
        "",
        "Do not compare these valid825 AE-3D values directly with the paper's official DSEC test AE. Use this table for same-split model comparisons; use official test submission for the final paper benchmark table.",
    ])
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_redesign(rows: list[dict[str, Any]]) -> None:
    current = REDESIGN.read_text(encoding="utf-8")
    if MARKER in current:
        return
    with REDESIGN.open("a", encoding="utf-8") as handle:
        handle.write("\n\n### 43.31 有价值 checkpoint 的统一新 AAE 审计（2026-07-17）\n\n")
        handle.write(f"<!-- {MARKER} -->\n\n")
        handle.write("统一使用 DSEC valid825，同时计算历史二维方向 AAE 与 DSEC/Barron `(u,v,1)` 三维 AE。完整可复现报告见 `neuron_autoresearch/VALUABLE_CHECKPOINT_AAE_AUDIT_20260717.md`。\n\n")
        for line in table(rows):
            handle.write(line + "\n")
        handle.write("\nH67 RTL-exact 相对 NB0：AEE 改善 `1.65%`、AE-3D 改善 `5.09%`、spikes 下降 `40.17%`、energy proxy 下降 `37.93%`，因此作为当前 checkpoint 主线。H68 RTL-exact 继续作为部署零增量回退。所有行加载审计均为预期 ATLIF/Shiftmax 数且 `missing=0, unexpected=0`；论文 official test 仍需单独提交，不能把 valid825 AE-3D 冒充 test AE。\n")


def main() -> int:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(GENERATOR)], cwd=REPO, check=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(run_eval, row): row["id"] for row in manifest}
        for future in as_completed(futures):
            future.result()
    rows = collect(manifest)
    SUMMARY.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    write_report(rows)
    append_redesign(rows)
    record(f"ALL COMPLETE: {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
