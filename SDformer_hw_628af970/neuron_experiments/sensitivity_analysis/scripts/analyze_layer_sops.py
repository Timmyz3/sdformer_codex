"""Analyze baseline layer spike/SOP concentration.

This script consumes the outputs produced by tools/profile_sops.py and writes a
ranked layer table plus a short Markdown report. It intentionally stays outside
third_party/SDformerFlow.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def format_gops(value: float) -> str:
    return f"{value / 1e9:.4f}G"


def stage_name(layer: str) -> str:
    if ".encoders." in layer:
        return "encoder"
    if ".decoders." in layer:
        return "decoder"
    if "patch_embed" in layer:
        return "patch_embed"
    if "blocks." in layer:
        return "transformer_block"
    if "sttmultires_unet" in layer:
        return "sttmultires_unet_other"
    return layer.split(".")[0]


def substage_name(layer: str) -> str:
    parts = layer.split(".")
    if "encoders" in parts:
        idx = parts.index("encoders")
        if idx + 1 < len(parts):
            return f"encoder.{parts[idx + 1]}"
    if "decoders" in parts:
        idx = parts.index("decoders")
        if idx + 1 < len(parts):
            return f"decoder.{parts[idx + 1]}"
    if "blocks" in parts:
        idx = parts.index("blocks")
        if idx + 1 < len(parts):
            return f"block.{parts[idx + 1]}"
    return stage_name(layer)


def read_layers(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def enrich_rows(rows: list[dict[str, str]], summary: dict) -> list[dict[str, float | int | str]]:
    dense_ops = float(summary["dense_ops"])
    total_elements = int(summary["total_elements"])
    enriched = []
    for row in rows:
        spikes = int(row["spikes"])
        elements = int(row["elements"])
        layer_sops = dense_ops * spikes / total_elements if total_elements else 0.0
        dense_proxy = dense_ops * elements / total_elements if total_elements else 0.0
        enriched.append(
            {
                "layer": row["layer"],
                "stage": stage_name(row["layer"]),
                "substage": substage_name(row["layer"]),
                "calls": int(row["calls"]),
                "spikes": spikes,
                "elements": elements,
                "firing_rate": float(row["firing_rate"]),
                "mean_call_firing_rate": float(row["mean_call_firing_rate"]),
                "dense_proxy_ops": dense_proxy,
                "sops_proxy": layer_sops,
            }
        )
    enriched.sort(key=lambda item: float(item["sops_proxy"]), reverse=True)
    total_sops = sum(float(item["sops_proxy"]) for item in enriched)
    cumulative = 0.0
    for rank, item in enumerate(enriched, start=1):
        cumulative += float(item["sops_proxy"])
        item["rank"] = rank
        item["sops_pct"] = float(item["sops_proxy"]) / total_sops if total_sops else 0.0
        item["cumulative_sops_pct"] = cumulative / total_sops if total_sops else 0.0
    return enriched


def group_rows(rows: list[dict[str, float | int | str]], key: str) -> list[dict[str, float | int | str]]:
    grouped: dict[str, dict[str, float | int | str]] = defaultdict(
        lambda: {"layers": 0, "spikes": 0, "elements": 0, "sops_proxy": 0.0}
    )
    for row in rows:
        name = str(row[key])
        item = grouped[name]
        item["name"] = name
        item["layers"] = int(item["layers"]) + 1
        item["spikes"] = int(item["spikes"]) + int(row["spikes"])
        item["elements"] = int(item["elements"]) + int(row["elements"])
        item["sops_proxy"] = float(item["sops_proxy"]) + float(row["sops_proxy"])
    total_sops = sum(float(item["sops_proxy"]) for item in grouped.values())
    result = []
    for item in grouped.values():
        elements = int(item["elements"])
        item["firing_rate"] = int(item["spikes"]) / elements if elements else 0.0
        item["sops_pct"] = float(item["sops_proxy"]) / total_sops if total_sops else 0.0
        result.append(item)
    result.sort(key=lambda item: float(item["sops_proxy"]), reverse=True)
    return result


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "layer",
        "stage",
        "substage",
        "calls",
        "spikes",
        "elements",
        "firing_rate",
        "mean_call_firing_rate",
        "dense_proxy_ops",
        "sops_proxy",
        "sops_pct",
        "cumulative_sops_pct",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_group_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    fieldnames = ["name", "layers", "spikes", "elements", "firing_rate", "sops_proxy", "sops_pct"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: list[dict[str, float | int | str]], limit: int) -> list[str]:
    lines = ["| rank | layer | firing | SOPs proxy | SOPs % | cumulative % |", "| ---: | --- | ---: | ---: | ---: | ---: |"]
    for row in rows[:limit]:
        lines.append(
            "| {rank} | `{layer}` | {firing:.5f} | {sops} | {pct:.2%} | {cum:.2%} |".format(
                rank=row["rank"],
                layer=row["layer"],
                firing=float(row["firing_rate"]),
                sops=format_gops(float(row["sops_proxy"])),
                pct=float(row["sops_pct"]),
                cum=float(row["cumulative_sops_pct"]),
            )
        )
    return lines


def group_md_table(rows: list[dict[str, float | int | str]]) -> list[str]:
    lines = ["| group | layers | firing | SOPs proxy | SOPs % |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(
            "| `{name}` | {layers} | {firing:.5f} | {sops} | {pct:.2%} |".format(
                name=row["name"],
                layers=row["layers"],
                firing=float(row["firing_rate"]),
                sops=format_gops(float(row["sops_proxy"])),
                pct=float(row["sops_pct"]),
            )
        )
    return lines


def top_k_line(rows: list[dict[str, float | int | str]], k: int, total_sops: float) -> str:
    kept = rows[:k]
    sops = sum(float(row["sops_proxy"]) for row in kept)
    return f"| top {k} layers | {format_gops(sops)} | {sops / total_sops:.2%} |"


def write_report(
    path: Path,
    rows: list[dict[str, float | int | str]],
    stage_rows: list[dict[str, float | int | str]],
    substage_rows: list[dict[str, float | int | str]],
    summary: dict,
    ranked_csv: Path,
    stage_csv: Path,
    substage_csv: Path,
) -> None:
    total_sops = sum(float(row["sops_proxy"]) for row in rows)
    lines = [
        "# PSN Baseline Layer Sensitivity",
        "",
        "This report ranks baseline spiking layers by spike/SOP contribution using the same global SOP proxy as `tools/profile_sops.py`:",
        "",
        "`layer_sops_proxy = dense_ops * layer_spikes / total_elements`",
        "",
        "It is a contribution sensitivity report, not an accuracy ablation yet.",
        "",
        "## Baseline",
        "",
        "| item | value |",
        "| --- | ---: |",
        f"| samples | {summary['samples']} |",
        f"| AEE | {summary['metrics']['AEE']:.4f} |",
        f"| AAE | {summary['metrics']['AAE']:.4f} |",
        f"| firing | {summary['global_firing_rate']:.5f} |",
        f"| total SOPs | {summary['estimated_total_sops_human']} |",
        f"| profiled layers | {summary['profiled_layers']} |",
        "",
        "## SOP Concentration",
        "",
        "| target set | SOPs proxy | share |",
        "| --- | ---: | ---: |",
        top_k_line(rows, 10, total_sops),
        top_k_line(rows, 20, total_sops),
        top_k_line(rows, 40, total_sops),
        "",
        "## Stage Summary",
        "",
        *group_md_table(stage_rows),
        "",
        "## Substage Summary",
        "",
        *group_md_table(substage_rows[:20]),
        "",
        "## Top Layers",
        "",
        *md_table(rows, 30),
        "",
        "## Candidate Target Sets",
        "",
        "For the first partial sparsity experiment, use target layers with high SOP share and avoid changing every spiking node.",
        "",
        "| set | layers | reason |",
        "| --- | ---: | --- |",
        "| G1-top10 | 10 | smallest intervention; tests whether the hottest layers are compressible |",
        "| G1-top20 | 20 | stronger SOP target while still avoiding blanket replacement |",
        "| G1-decoder-hot | variable | decoder layers have high firing and direct reconstruction impact, so use after top10/top20 probe |",
        "",
        "Artifacts:",
        "",
        f"- ranked layers: `{ranked_csv}`",
        f"- stage summary: `{stage_csv}`",
        f"- substage summary: `{substage_csv}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--layers", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text())
    rows = enrich_rows(read_layers(args.layers), summary)
    stage_rows = group_rows(rows, "stage")
    substage_rows = group_rows(rows, "substage")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranked_csv = args.output_dir / "baseline_epoch59_valid40_ranked_layers.csv"
    stage_csv = args.output_dir / "baseline_epoch59_valid40_stage_summary.csv"
    substage_csv = args.output_dir / "baseline_epoch59_valid40_substage_summary.csv"
    report_md = args.output_dir / "baseline_epoch59_valid40_sensitivity.md"

    write_csv(ranked_csv, rows)
    write_group_csv(stage_csv, stage_rows)
    write_group_csv(substage_csv, substage_rows)
    write_report(report_md, rows, stage_rows, substage_rows, summary, ranked_csv, stage_csv, substage_csv)

    print(f"ranked_layers: {ranked_csv}")
    print(f"stage_summary: {stage_csv}")
    print(f"substage_summary: {substage_csv}")
    print(f"report: {report_md}")


if __name__ == "__main__":
    main()
