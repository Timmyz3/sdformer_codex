#!/usr/bin/env python3
"""Attach frozen valid825 density quartiles to four-line per-frame AEE/Fl/spikes.

Writes Table G from same-checkpoint rank-1 models. Hardware columns stay read-only.
Does not overwrite existing standard_valid825 spike_profile.json files.
"""

from __future__ import annotations

from datetime import datetime, timezone
import csv
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
RESULTS = EXP / "results"
OUT = RESULTS / "dsec_density_table_g_four_line_20260817"
POPULATION = REPO / "neuron_autoresearch/DSEC_VALID825_DENSITY_POPULATION_20260813.json"
TABLE_JSON = REPO / "neuron_autoresearch/DSEC_DENSITY_QUARTILE_TABLE_G_20260817.json"
TABLE_MD = TABLE_JSON.with_suffix(".md")
FRAME_LIST = OUT / "selected_frames.txt"
STATUS = OUT / "watcher.log"
LOCK = Path("/tmp/sdformer_density_table_g_four_line.lock")
PY = Path("/opt/conda/envs/sdformerflow/bin/python")
EVAL = REPO / "third_party/SDformerFlow/eval_DSEC_flow_SNN.py"

LINES = [
    {
        "id": "NB0",
        "label": "NB0 PSN",
        "epoch": 29,
        "config": EXP / "configs/generated/dsec_fullres_w15_NB0_equal_plus10_ep40.yml",
        "checkpoint": RESULTS
        / "dsec_fullres_w15_NB0_equal_plus10_ep40_20260805/checkpoint_epoch29.pth",
        "reference_aee": 1.4453525361147794,
        "reference_fl": 7.9322528587103776,
        "reference_spikes_g": 126.115607423,
    },
    {
        "id": "H81",
        "label": "H81 TTX no-motion",
        "epoch": 29,
        "config": EXP / "configs/generated/dsec_fullres_w15_H81_nomotion_bb1e4_ft40.yml",
        "checkpoint": RESULTS
        / "dsec_fullres_w15_H81_nomotion_bb1e4_ft40_20260811/checkpoint_epoch29.pth",
        "reference_aee": 1.3305970512014447,
        "reference_fl": 6.430976935979286,
        "reference_spikes_g": 80.902381753,
    },
    {
        "id": "H67",
        "label": "H67 Motion-TTX",
        "epoch": 35,
        "config": EXP / "configs/generated/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40.yml",
        "checkpoint": RESULTS
        / "dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_20260805/checkpoint_epoch35.pth",
        "reference_aee": 1.3296776408860178,
        "reference_fl": 6.427881411941348,
        "reference_spikes_g": 82.110742384,
    },
    {
        "id": "Local5",
        "label": "Local5 TTX",
        "epoch": 44,
        "config": EXP
        / "configs/generated/dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50.yml",
        "checkpoint": RESULTS
        / "dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50_20260812/checkpoint_epoch44.pth",
        "reference_aee": 1.2818928577683188,
        "reference_fl": 6.021021248107319,
        "reference_spikes_g": 85.237646886,
    },
]


def record(message: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
    print(line, flush=True)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with STATUS.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def freeze_selected_frames(population: dict) -> list[str]:
    by_q: dict[str, list[dict]] = {"Q1": [], "Q2": [], "Q3": [], "Q4": []}
    for frame in population["frames"]:
        by_q[frame["quartile"]].append(frame)
    selected: list[str] = []
    for quartile, picker in (
        ("Q1", lambda rows: min(rows, key=lambda r: r["voxel_l1"])),
        ("Q1", lambda rows: sorted(rows, key=lambda r: r["voxel_l1"])[len(rows) // 2]),
        ("Q4", lambda rows: sorted(rows, key=lambda r: r["voxel_l1"])[len(rows) // 2]),
        ("Q4", lambda rows: max(rows, key=lambda r: r["voxel_l1"])),
    ):
        chosen = picker(by_q[quartile])
        if chosen["file"] not in selected:
            selected.append(chosen["file"])
    FRAME_LIST.parent.mkdir(parents=True, exist_ok=True)
    FRAME_LIST.write_text("\n".join(selected) + "\n", encoding="utf-8")
    return selected


def csv_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return len(rows) == 825


def eval_line(line: dict, *, max_samples: int = 0) -> Path:
    out_dir = OUT / line["id"].lower()
    if max_samples:
        out_dir = OUT / f"{line['id'].lower()}_smoke"
    csv_path = out_dir / "per_frame.csv"
    log_path = out_dir / "eval.log"
    frames_dir = out_dir / "selected_frames"
    if csv_complete(csv_path) and not max_samples:
        record(f"REUSE {line['id']} {csv_path}")
        return csv_path
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["SDFORMER_USE_MLFLOW"] = "0"
    env["SDFORMER_MLFLOW_MODEL_LOGGING"] = "0"
    env["SDFORMER_SNN_BACKEND"] = "cupy"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    command = [
        str(PY),
        "-u",
        str(EVAL),
        "--config",
        str(line["config"]),
        "--checkpoint",
        str(line["checkpoint"]),
        "--path_results",
        str(out_dir),
        "--mode",
        "valid",
        "--dump-per-frame",
        str(csv_path),
        "--dump-selected-frames-dir",
        str(frames_dir),
        "--dump-frame-list",
        str(FRAME_LIST),
    ]
    if max_samples:
        command.extend(["--max-samples", str(max_samples)])
    record("START " + " ".join(command))
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        proc = subprocess.run(
            command,
            cwd=REPO,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        handle.write(f"\n[density-table-g] exit_code={proc.returncode}\n")
    if proc.returncode:
        raise RuntimeError(f"{line['id']} eval failed; log={log_path}")
    if not csv_complete(csv_path) and not max_samples:
        raise RuntimeError(f"{line['id']} per-frame CSV is not 825 rows: {csv_path}")
    record(f"END {line['id']} exit_code={proc.returncode}")
    return csv_path


def load_frames(csv_path: Path) -> dict[str, dict]:
    rows = {}
    with csv_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[row["file"]] = {
                "file": row["file"],
                "sequence": row["sequence"],
                "valid_pixels": float(row["valid_pixels"]),
                "AEE": float(row["AEE"]),
                "AAE": float(row["AAE"]),
                "AAE_Benchmark": float(row["AAE_Benchmark"]),
                "DSEC_Fl": float(row["DSEC_Fl"]),
                "gt_flow_mag": float(row["gt_flow_mag"]),
                "spikes": float(row["spikes"]),
                "elements": float(row["elements"]),
            }
    return rows


def mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def summarize_line(line: dict, frames: dict[str, dict], population: dict) -> dict:
    quartiles = {"Q1": [], "Q2": [], "Q3": [], "Q4": []}
    missing = []
    for item in population["frames"]:
        row = frames.get(item["file"])
        if row is None:
            missing.append(item["file"])
            continue
        quartiles[item["quartile"]].append(row)
    if missing:
        raise RuntimeError(f"{line['id']} missing {len(missing)} population files")
    all_rows = [row for bucket in quartiles.values() for row in bucket]
    observed_aee = mean([row["AEE"] for row in all_rows])
    observed_fl = mean([row["DSEC_Fl"] for row in all_rows])
    observed_spikes_g = sum(row["spikes"] for row in all_rows) / 1e9
    payload = {
        "id": line["id"],
        "label": line["label"],
        "epoch": line["epoch"],
        "checkpoint": str(line["checkpoint"]),
        "config": str(line["config"]),
        "n_frames": len(all_rows),
        "observed_frame_equal": {
            "AEE": observed_aee,
            "DSEC_Fl": observed_fl,
            "total_spikes_g": observed_spikes_g,
        },
        "reference": {
            "AEE": line["reference_aee"],
            "DSEC_Fl": line["reference_fl"],
            "total_spikes_g": line["reference_spikes_g"],
        },
        "delta_vs_reference": {
            "AEE": observed_aee - line["reference_aee"],
            "DSEC_Fl": observed_fl - line["reference_fl"],
            "total_spikes_g": observed_spikes_g - line["reference_spikes_g"],
        },
        "quartiles": {},
    }
    for name in ("Q1", "Q2", "Q3", "Q4"):
        rows = quartiles[name]
        payload["quartiles"][name] = {
            "frames": len(rows),
            "AEE": mean([row["AEE"] for row in rows]),
            "DSEC_Fl": mean([row["DSEC_Fl"] for row in rows]),
            "spikes_per_frame": mean([row["spikes"] for row in rows]),
            "AEE_median": float(statistics.median([row["AEE"] for row in rows])),
            "valid_pixels": sum(row["valid_pixels"] for row in rows),
        }
    return payload


def write_table(population: dict, line_summaries: list[dict]) -> None:
    cuts = population["cuts"]
    counts = population["quartile_counts"]
    rows = []
    for summary in line_summaries:
        for quartile, bounds in (
            ("Q1", f"voxel-L1 <= {cuts['q25']:.1f}"),
            ("Q2", f"voxel-L1 <= {cuts['q50']:.1f}"),
            ("Q3", f"voxel-L1 <= {cuts['q75']:.1f}"),
            ("Q4", f"voxel-L1 > {cuts['q75']:.1f}"),
        ):
            cell = summary["quartiles"][quartile]
            rows.append(
                {
                    "dataset": "DSEC valid825",
                    "method": summary["label"],
                    "density_quartile": f"{quartile} {bounds}",
                    "frames": cell["frames"],
                    "AEE": cell["AEE"],
                    "Fl": cell["DSEC_Fl"],
                    "spikes_per_frame": cell["spikes_per_frame"],
                    "active_relations": "hardware read-only",
                    "memo_hit_rate": "hardware read-only",
                    "cycles_per_frame": "hardware read-only",
                }
            )
    payload = {
        "schema": "dsec_density_quartile_table_g_v1",
        "status": "PASS_AEE_ATTACHED",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "population": str(POPULATION),
        "cuts": cuts,
        "quartile_counts": counts,
        "lines": line_summaries,
        "table_g": rows,
        "notes": [
            "Quartile cuts were frozen before seeing per-frame AEE.",
            "AEE/Fl are frame-equal means inside each frozen quartile.",
            "spikes_per_frame is the spike-profiler delta for that frame.",
            "Hardware cycle/traffic columns remain read-only.",
        ],
    }
    TABLE_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# DSEC valid825 density Table G",
        "",
        f"Status: `{payload['status']}`; frames=`825`; density=`voxel L1`.",
        "",
        "| Dataset | Method | Density quartile | Frames | AEE | Fl | Spikes/frame | Active relations | Memo hit rate | Cycles/frame |",
        "|---|---|---|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {method} | {density_quartile} | {frames} | {AEE:.4f} | {Fl:.4f} | {spikes_per_frame:.3e} | {active_relations} | {memo_hit_rate} | {cycles_per_frame} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Cuts stay frozen from `DSEC_VALID825_DENSITY_POPULATION_20260813.json`.",
            "Hardware columns are not filled from this algorithm eval.",
            "",
        ]
    )
    TABLE_MD.write_text("\n".join(lines), encoding="utf-8")
    record(f"WROTE {TABLE_JSON}")


def main() -> int:
    smoke = "--smoke" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("density Table G runner already active", flush=True)
            return 0
        if not PY.is_file():
            raise FileNotFoundError(PY)
        population = json.loads(POPULATION.read_text(encoding="utf-8"))
        selected = freeze_selected_frames(population)
        record(f"selected qualitative frames: {selected}")
        if smoke:
            eval_line(LINES[2], max_samples=2)
            record("SMOKE complete")
            return 0
        summaries = []
        for line in LINES:
            csv_path = eval_line(line)
            frames = load_frames(csv_path)
            summary = summarize_line(line, frames, population)
            record(
                f"{line['id']} observed AEE={summary['observed_frame_equal']['AEE']:.6f} "
                f"delta={summary['delta_vs_reference']['AEE']:+.6e} "
                f"spikesG={summary['observed_frame_equal']['total_spikes_g']:.4f}"
            )
            if abs(summary["delta_vs_reference"]["AEE"]) > 5e-4:
                raise RuntimeError(f"{line['id']} AEE drifted from rank-1 reference")
            summaries.append(summary)
        write_table(population, summaries)
        qual = Path(__file__).resolve().parent / "plot_date_qualitative_density_frames_20260817.py"
        if qual.is_file():
            subprocess.run([str(PY), "-u", str(qual)], cwd=REPO, check=False)
        record("ALL COMPLETE Table G attached")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
