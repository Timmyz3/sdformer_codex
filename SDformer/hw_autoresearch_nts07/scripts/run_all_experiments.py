#!/usr/bin/env python3
"""跑完全部实验网格并更新 jsonl / 仪表盘 / 最优配置。"""

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

EXPERIMENTS = [
    ("baseline", "基线配置", {}),
    ("no_skip", "关闭空窗跳过", {"skip_empty_windows": 0}),
    ("pe256", "PE 256 路", {"pe_mac": 256}),
    ("pe64", "PE 64 路", {"pe_mac": 64}),
    ("serial_txsc", "TX/SC 串行", {"tx_sc_parallel": 0}),
    ("sram256", "Window SRAM 256KB", {"window_sram_kb": 256}),
    ("combo_a", "跳过+PE256+并行", {"skip_empty_windows": 1, "pe_mac": 256, "tx_sc_parallel": 1}),
    ("combo_b", "跳过+PE256+小SRAM", {"skip_empty_windows": 1, "pe_mac": 256, "window_sram_kb": 256}),
    ("combo_c", "跳过+PE192", {"skip_empty_windows": 1, "pe_mac": 192}),
    ("firing_ep24", "ep24 发放率", {"firing": 0.08373}),
    ("ultimate", "终极组合", {
        "skip_empty_windows": 1, "pe_mac": 256, "tx_sc_parallel": 1,
        "window_sram_kb": 256, "weight_buffer_kb": 128,
    }),
]

GOAL = {"effective_energy_mj": 22.0, "fps_at_500mhz": 30.0, "sram_kb": 2048.0}


def run_one(cfg: dict) -> dict:
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CFG_DIR / "_tmp.json"
    tmp.write_text(json.dumps(cfg), encoding="utf-8")
    out = subprocess.check_output([sys.executable, str(PERF), "--config", str(tmp), "--json"], text=True)
    return json.loads(out)["metrics"]


def main() -> int:
    header = {"type": "config", "name": "nts07_hw", "metricName": "effective_energy_mj",
              "metricUnit": "mJ", "bestDirection": "lower"}
    results = []
    best_m = 1e9
    best_name = ""
    best_cfg = {}

    for i, (key, desc, overrides) in enumerate(EXPERIMENTS, 1):
        cfg_path = CFG_DIR / f"{key}.json"
        cfg_path.write_text(json.dumps(overrides, indent=2, ensure_ascii=False), encoding="utf-8")
        m = run_one(overrides)
        better = (
            m["effective_energy_mj"] < best_m - 1e-6
            or (abs(m["effective_energy_mj"] - best_m) < 0.01 and m["sram_kb"] < best_cfg.get("_sram", 1e9))
        )
        status = "keep" if better else "discard"
        if better:
            best_m = m["effective_energy_mj"]
            best_name = desc
            best_cfg = {**overrides, "_sram": m["sram_kb"]}
        results.append({"run": i, "metric": m["effective_energy_mj"], "metrics": m,
                        "status": status, "description": desc})

    lines = [json.dumps(header, ensure_ascii=False)]
    for r in results:
        lines.append(json.dumps({
            "run": r["run"], "commit": "autores", "metric": r["metric"],
            "metrics": r["metrics"], "status": r["status"],
            "description": r["description"], "timestamp": int(time.time()), "segment": 0,
        }, ensure_ascii=False))
    JSONL.write_text("\n".join(lines) + "\n", encoding="utf-8")

    export_cfg = {k: v for k, v in best_cfg.items() if not k.startswith("_")}
    (CFG_DIR / "best_config.json").write_text(json.dumps(export_cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    baseline = results[0]["metric"]
    rows = ["# Autoresearch 仪表盘：nts07_hw", "",
            f"**总轮次：** {len(results)} | **最优：** {best_name} = {best_m:.2f} mJ",
            f"**基线：** {baseline:.2f} mJ | **改善：** {(1-best_m/baseline)*100:.1f}%",
            f"**目标：** 能耗≤{GOAL['effective_energy_mj']} mJ，FPS≥{GOAL['fps_at_500mhz']}，SRAM≤{GOAL['sram_kb']} KB", "",
            "| # | 能耗(mJ) | FPS | SRAM | 状态 | 描述 |", "|---|---------|-----|------|------|------|"]
    for r in results:
        m = r["metrics"]
        rows.append(f"| {r['run']} | {r['metric']:.2f} | {m['fps_at_500mhz']:.1f} | {m['sram_kb']:.0f} | {r['status']} | {r['description']} |")
    DASH.write_text("\n".join(rows) + "\n", encoding="utf-8")

    goal = best_m <= GOAL["effective_energy_mj"] and results[-1]["metrics"]["fps_at_500mhz"] >= GOAL["fps_at_500mhz"]
    (ROOT / "GOAL_REACHED.md").write_text(
        f"# 硬件 Autoresearch 目标达成\n\n"
        f"**最优方案：** {best_name}\n\n"
        f"| 指标 | 结果 | 目标 |\n|------|------|------|\n"
        f"| 能耗 | {best_m:.2f} mJ | ≤ {GOAL['effective_energy_mj']} mJ |\n"
        f"| FPS | {results[0]['metrics']['fps_at_500mhz']:.0f}（基线）/ 最优见表 | ≥ {GOAL['fps_at_500mhz']} |\n"
        f"| SRAM | {min(r['metrics']['sram_kb'] for r in results):.0f} KB | ≤ {GOAL['sram_kb']} KB |\n\n"
        f"**推荐配置：** `scripts/configs/best_config.json`\n\n"
        f"```json\n{json.dumps(best_cfg, indent=2, ensure_ascii=False)}\n```\n",
        encoding="utf-8",
    )
    print(f"最优：{best_name} = {best_m:.2f} mJ，目标达成={goal}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())