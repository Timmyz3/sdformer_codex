#!/usr/bin/env python3
"""Segment-1 autoresearch: literature-inspired knobs on ultimate baseline."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PERF = ROOT / "scripts" / "nts07_perf_model.py"
JSONL = ROOT / "autoresearch.jsonl"
DASH = ROOT / "autoresearch-dashboard.md"
CFG_DIR = ROOT / "scripts" / "configs"
DOCS = ROOT / "docs" / "12_文献启发_autoresearch.md"

ULTIMATE = {
    "skip_empty_windows": 1,
    "pe_mac": 256,
    "tx_sc_parallel": 1,
    "window_sram_kb": 256,
    "weight_buffer_kb": 128,
}

EXPERIMENTS = [
    ("lit_baseline", "终极组合（文献轮锚点）", {}),
    ("lit_unified_encode", "统一 ATLIF 编码器（共享比较器）", {"unified_atlif_encode": 1}),
    ("lit_bishop_ttb2", "Bishop TTB depth-2 打包", {"bishop_ttb_depth": 2}),
    ("lit_firefly_pop64", "FireFly 风格 popcount×64", {"firefly_popcount_par": 64}),
    ("lit_encode_lanes16", "共享编码 16 lane 宽发射", {"unified_atlif_encode": 1, "shared_encode_lanes": 16}),
    ("lit_combo_unified_ttb", "统一编码 + TTB-2", {"unified_atlif_encode": 1, "bishop_ttb_depth": 2}),
    ("lit_combo_full", "文献终极组合", {
        "unified_atlif_encode": 1,
        "bishop_ttb_depth": 2,
        "firefly_popcount_par": 64,
        "shared_encode_lanes": 16,
    }),
]


def run_one(cfg: dict) -> dict:
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CFG_DIR / "_tmp_lit.json"
    tmp.write_text(json.dumps(cfg), encoding="utf-8")
    out = subprocess.check_output(
        [sys.executable, str(PERF), "--config", str(tmp), "--json"], text=True
    )
    return json.loads(out)["metrics"]


def main() -> int:
    base_run = 11
    results = []
    best_m = 1e9
    best_name = ""
    best_cfg = {}

    for i, (key, desc, overrides) in enumerate(EXPERIMENTS, 1):
        cfg = {**ULTIMATE, **overrides}
        (CFG_DIR / f"{key}.json").write_text(
            json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        m = run_one(cfg)
        better = m["effective_energy_mj"] < best_m - 1e-6 or (
            abs(m["effective_energy_mj"] - best_m) < 0.01
            and m.get("area_mm2", 9) < best_cfg.get("_area", 9)
        )
        status = "keep" if better else "discard"
        if better:
            best_m = m["effective_energy_mj"]
            best_name = desc
            best_cfg = {**cfg, "_area": m.get("area_mm2", 2.85)}
        results.append(
            {
                "run": base_run + i,
                "metric": m["effective_energy_mj"],
                "metrics": m,
                "status": status,
                "description": desc,
                "config": cfg,
            }
        )

    existing = []
    if JSONL.exists():
        for line in JSONL.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing.append(line)

    for r in results:
        existing.append(
            json.dumps(
                {
                    "run": r["run"],
                    "commit": "autores",
                    "metric": r["metric"],
                    "metrics": r["metrics"],
                    "status": r["status"],
                    "description": r["description"],
                    "config": r["config"],
                    "timestamp": int(time.time()),
                    "segment": 1,
                },
                ensure_ascii=False,
            )
        )
    JSONL.write_text("\n".join(existing) + "\n", encoding="utf-8")

    anchor = results[0]["metric"]
    rows = [
        "# 文献启发 Autoresearch（Segment 1）",
        "",
        f"**锚点：** 终极组合 = {anchor:.2f} mJ",
        f"**最优：** {best_name} = {best_m:.2f} mJ（面积 {best_cfg.get('_area', 2.85):.2f} mm²）",
        "",
        "| # | 能耗(mJ) | FPS | SRAM | 面积(mm²) | 状态 | 描述 |",
        "|---|---------|-----|------|-----------|------|------|",
    ]
    for r in results:
        m = r["metrics"]
        rows.append(
            f"| {r['run']} | {r['metric']:.2f} | {m['fps_at_500mhz']:.1f} | "
            f"{m['sram_kb']:.0f} | {m.get('area_mm2', 2.85):.2f} | {r['status']} | {r['description']} |"
        )
    doc_body = "\n".join(rows) + "\n"
    DOCS.write_text(doc_body, encoding="utf-8")

    dash_lines = DASH.read_text(encoding="utf-8").splitlines() if DASH.exists() else []
    dash_lines.append("")
    dash_lines.append("## Segment 1：文献启发（在终极组合上叠加）")
    dash_lines.append("")
    for r in results:
        m = r["metrics"]
        dash_lines.append(
            f"- Run {r['run']}: {r['metric']:.2f} mJ, {m['fps_at_500mhz']:.0f} FPS, "
            f"{r['status']} — {r['description']}"
        )
    DASH.write_text("\n".join(dash_lines) + "\n", encoding="utf-8")

    export = {k: v for k, v in best_cfg.items() if not k.startswith("_")}
    (CFG_DIR / "best_config_lit.json").write_text(
        json.dumps(export, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"文献轮最优：{best_name} = {best_m:.2f} mJ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())