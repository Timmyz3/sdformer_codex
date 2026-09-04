#!/usr/bin/env python3
"""Post-process profile100 runs into (a) full-encoder per-operator activity
share tables (P1-4 Amdahl input for the hw side) and (b) 12-block same-window
tables (P1-5), for both DATE lines: H67 ep35 and Local5 ep44.

Inputs (must exist before running):
  results/h67_fullres_ep35_t450_profile100_20260818/   (P0-3, bit trace on)
  results/local5_fullres_ep44_t450_profile100_20260818/ (P1-4)

Outputs:
  results/full_encoder_amdahl_12block_20260818/
    h67_ep35/operator_share.csv|.md   per-category activity share (encoder + decoder)
    h67_ep35/blocks12_same_window.csv|.md
    local5_ep44/... same
    summary.json
    SUMMARY.md

Evidence tier: [prof] (pure post-processing of profiler outputs, no GPU).
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
RESULTS = EXP / "results"
H67_DIR = RESULTS / "h67_fullres_ep35_t450_profile100_20260818"
LOCAL5_DIR = RESULTS / "local5_fullres_ep44_t450_profile100_20260818"
OUT = RESULTS / "full_encoder_amdahl_12block_20260818"

# Operator -> category classification (by module path fragments)
def classify(name: str, operator: str) -> str:
    n = name
    if "patch_embed" in n:
        if "resblocks" in n or "residual_encoding" in n:
            return "patch_embed_residual"
        return "patch_embed_conv"
    if "attn" in n and "linear_q" in n:
        return "attn_q"
    if "attn" in n and "linear_k" in n:
        return "attn_k"
    if "attn" in n and "proj" in n:
        return "attn_proj"
    if "mlp" in n:
        return "ffn_mlp"
    if "downsample" in n:
        return "downsample"
    if "decoders" in n or "preds" in n:
        return "decoder"
    if "resblocks" in n:
        return "bottleneck_res"
    return "other"


def build_operator_share(line: Path) -> dict:
    rows = list(csv.DictReader((line / "operator_runtime.csv").open()))
    cats: dict[str, dict] = {}
    total_macs_proxy = 0.0
    for r in rows:
        c = classify(r["name"], r["operator"])
        d = cats.setdefault(c, {"calls": 0, "macs_proxy": 0.0, "input_active": 0.0,
                                "input_elements": 0, "ops": []})
        d["calls"] += int(r["calls"])
        d["macs_proxy"] += float(r["activity_weighted_macs_proxy"])
        d["input_active"] += float(r["input_active"])
        d["input_elements"] += int(r["input_elements"])
        d["ops"].append(r["name"])
        total_macs_proxy += float(r["activity_weighted_macs_proxy"])
    order = ["attn_q", "attn_k", "attn_proj", "ffn_mlp", "downsample",
             "patch_embed_conv", "patch_embed_residual", "bottleneck_res",
             "decoder", "other"]
    out = []
    for c in order:
        if c not in cats:
            continue
        d = cats[c]
        out.append({
            "category": c,
            "calls": d["calls"],
            "macs_proxy": d["macs_proxy"],
            "macs_proxy_share": d["macs_proxy"] / total_macs_proxy if total_macs_proxy else 0.0,
            "input_active": d["input_active"],
            "input_elements": d["input_elements"],
            "op_count": len(d["ops"]),
        })
    return {"categories": out, "total_macs_proxy": total_macs_proxy}


def build_atlif_share(line: Path) -> dict:
    rows = list(csv.DictReader((line / "atlif_activity.csv").open()))
    total_active = sum(int(r["active"]) for r in rows)
    by_stage: dict[str, dict] = {}
    for r in rows:
        n = r["name"]
        if "encoders" in n:
            st = "encoder"
        elif "decoders" in n or "preds" in n:
            st = "decoder"
        elif "resblocks" in n:
            st = "bottleneck"
        else:
            st = "other"
        d = by_stage.setdefault(st, {"active": 0, "modules": 0, "margin_le_1_128": 0})
        d["active"] += int(r["active"])
        d["modules"] += 1
        d["margin_le_1_128"] += int(r["margin_abs_le_1_128"])
    stages = []
    for st in ("encoder", "bottleneck", "decoder", "other"):
        if st not in by_stage:
            continue
        d = by_stage[st]
        stages.append({"stage": st, "spikes": d["active"],
                       "spike_share": d["active"] / total_active if total_active else 0.0,
                       "modules": d["modules"],
                       "margin_abs_le_1_128": d["margin_le_1_128"]})
    return {"stages": stages, "total_spikes": total_active}


def build_blocks12(line: Path) -> list[dict]:
    rows = list(csv.DictReader((line / "h60_by_block.csv").open()))
    if not rows:
        return []
    order = sorted(rows, key=lambda r: (int(r["group"].split(".")[0][1:]),
                                        int(r["group"].split(".")[1][1:])))
    sel = ["group", "calls", "q_active_density", "k_active_density",
           "qk_temporal_update_density", "zaf_active_entries_mean",
           "zaf_fold_classes_mean", "zaf_kzero_token_ratio", "gate_entropy_mean",
           "sc_mean", "sc_std", "score_clip_ratio", "top1_mass_mean",
           "top4_mass_mean", "ttb1_empty_ratio", "ttb2_empty_ratio"]
    out = []
    for r in order:
        out.append({k: (float(r[k]) if k not in ("group", "calls") else r[k]) for k in sel})
    return out


def build_blocks12_operator(line: Path) -> list[dict]:
    """Per-block same-window activity from operator_runtime.csv (for lines whose
    attention mode is not h60, e.g. Local5 binary_axnor_local5_shiftmax, where
    the H60 recorder has no data)."""
    rows = list(csv.DictReader((line / "operator_runtime.csv").open()))
    per_block: dict[str, dict] = {}
    for r in rows:
        name = r["name"]
        if ".attn." not in name or "swin_blocks" not in name:
            continue
        block = name.split("swin_blocks.")[1].split(".")[0]
        stage = name.split("layers.")[1].split(".")[0]
        key = f"S{stage}.B{block}"
        d = per_block.setdefault(key, {"calls": 0, "attn_macs_proxy": 0.0,
                                       "attn_input_active": 0.0, "attn_input_elements": 0})
        d["calls"] += int(r["calls"])
        d["attn_macs_proxy"] += float(r["activity_weighted_macs_proxy"])
        d["attn_input_active"] += float(r["input_active"])
        d["attn_input_elements"] += int(r["input_elements"])
    order = sorted(per_block, key=lambda k: (int(k.split(".")[0][1:]), int(k.split(".")[1][1:])))
    out = []
    for key in order:
        d = per_block[key]
        out.append({
            "group": key, "calls": d["calls"],
            "attn_macs_proxy": d["attn_macs_proxy"],
            "attn_input_active": d["attn_input_active"],
            "attn_input_elements": d["attn_input_elements"],
            "attn_active_density": d["attn_input_active"] / d["attn_input_elements"] if d["attn_input_elements"] else 0.0,
        })
    return out


def fmt_md_blocks12(rows: list[dict]) -> str:
    head = ["block", "calls", "q_active", "k_active", "qk_update", "zaf_entries",
            "zaf_fold", "kzero", "entropy", "sc", "clip_ratio", "top1", "ttb2_empty"]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---:" for _ in head) + "|"]
    for r in rows:
        vals = [r["group"], r["calls"],
                f"{r['q_active_density']:.4f}", f"{r['k_active_density']:.4f}",
                f"{r['qk_temporal_update_density']:.4f}",
                f"{r['zaf_active_entries_mean']:.1f}", f"{r['zaf_fold_classes_mean']:.2f}",
                f"{r['zaf_kzero_token_ratio']:.4f}", f"{r['gate_entropy_mean']:.4f}",
                f"{r['sc_mean']:.4f}", f"{r['score_clip_ratio']:.2e}",
                f"{r['top1_mass_mean']:.4f}", f"{r['ttb2_empty_ratio']:.4f}"]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> int:
    for name, d in (("h67_ep35", H67_DIR), ("local5_ep44", LOCAL5_DIR)):
        if not (d / "operator_runtime.csv").exists():
            print(f"[SKIP] {name}: {d}/operator_runtime.csv missing (profile100 not done)")
            return 2
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"schema": "full_encoder_amdahl_12block_v1",
               "timestamp_utc": datetime.now(timezone.utc).isoformat(),
               "lines": {}}
    md_parts = ["# Full-encoder Amdahl input + 12-block same-window (2026-08-18)",
                "",
                "来源：P0-3（H67 ep35 profile100，含 bit trace）与 P1-4（Local5 ep44 profile100）。",
                "证据分档：[prof]。"]
    for key, d in (("h67_ep35", H67_DIR), ("local5_ep44", LOCAL5_DIR)):
        op = build_operator_share(d)
        at = build_atlif_share(d)
        b12 = build_blocks12(d)
        b12op = build_blocks12_operator(d)
        sub = OUT / key
        sub.mkdir(parents=True, exist_ok=True)
        # operator share CSV
        with (sub / "operator_share.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["category", "calls", "macs_proxy",
                                               "macs_proxy_share", "input_active",
                                               "input_elements", "op_count"])
            w.writeheader()
            w.writerows(op["categories"])
        # atlif share CSV
        with (sub / "atlif_stage_share.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["stage", "spikes", "spike_share",
                                               "modules", "margin_abs_le_1_128"])
            w.writeheader()
            w.writerows(at["stages"])
        # blocks12 CSV (H60 path when available; operator-level fallback otherwise)
        if b12:
            with (sub / "blocks12_same_window.csv").open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(b12[0].keys()))
                w.writeheader()
                w.writerows(b12)
        if b12op:
            with (sub / "blocks12_operator.csv").open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(b12op[0].keys()))
                w.writeheader()
                w.writerows(b12op)
        # markdown
        rows = ["", f"## {key}", "",
                "### Per-operator activity share (activity-weighted MACs proxy)",
                "", "| category | calls | macs_proxy | share | input_active |", "|---|---|---:|---:|---:|"]
        for c in op["categories"]:
            rows.append(f"| {c['category']} | {c['calls']} | {c['macs_proxy']:.6e} | "
                        f"{c['macs_proxy_share'] * 100:.2f}% | {c['input_active']:.6e} |")
        rows.append(f"\ntotal activity-weighted MACs proxy: {op['total_macs_proxy']:.6e}\n")
        rows += ["### ATLIF stage spike share", "",
                 "| stage | spikes | share | modules | margin_abs_le_1_128 |", "|---|---|---:|---:|---:|"]
        for s in at["stages"]:
            rows.append(f"| {s['stage']} | {s['spikes']} | {s['spike_share'] * 100:.2f}% | "
                        f"{s['modules']} | {s['margin_abs_le_1_128']} |")
        if b12:
            rows += ["", "### 12-block same-window H60 stats ([2,15,15], 1 window/sample)", "",
                     fmt_md_blocks12(b12)]
        if b12op:
            rows += ["", "### 12-block same-window operator activity (attn q/k/proj linear, per block)", "",
                     "| block | calls | attn_macs_proxy | attn_active_density |", "|---|---|---:|---:|"]
            for r in b12op:
                rows.append(f"| {r['group']} | {r['calls']} | {r['attn_macs_proxy']:.6e} | "
                            f"{r['attn_active_density']:.4f} |")
        if not b12 and not b12op:
            rows.append("\n(该线无 12-block 同窗结构数据)")
        (sub / "blocks12_same_window.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
        md_parts += rows
        summary["lines"][key] = {
            "operator_share": str(sub / "operator_share.csv"),
            "atlif_stage_share": str(sub / "atlif_stage_share.csv"),
            "h60_calls": (b12[0]["calls"] if b12 else 0),
            "blocks12_same_window": str(sub / "blocks12_same_window.csv") if b12 else None,
            "blocks12_operator": str(sub / "blocks12_operator.csv") if b12op else None,
            "total_macs_proxy": op["total_macs_proxy"],
            "atlif_total_spikes": at["total_spikes"],
        }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "SUMMARY.md").write_text("\n".join(md_parts) + "\n", encoding="utf-8")
    print(f"[DONE] {OUT / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
