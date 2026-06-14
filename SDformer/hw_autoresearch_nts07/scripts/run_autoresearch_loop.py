#!/usr/bin/env python3
"""自主运行 NTS-07b 硬件 autoresearch 循环，直至达到目标或穷尽搜索空间。"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PERF = ROOT / "scripts" / "nts07_perf_model.py"
JSONL = ROOT / "autoresearch.jsonl"
DASHBOARD = ROOT / "autoresearch-dashboard.md"
WORKLOG = ROOT / "experiments" / "worklog.md"
CONFIG_DIR = ROOT / "scripts" / "configs"

# DATE 2027 硬件目标
GOAL = {
    "effective_energy_mj": 22.0,
    "fps_at_500mhz": 30.0,
    "sram_kb": 2048.0,
    "epe_drift": 0.02,
}


@dataclass
class Experiment:
    name: str
    description: str
    config: dict


def default_grid() -> list[Experiment]:
    base = {}
    exps: list[Experiment] = [
        Experiment("baseline", "基线配置", dict(base)),
        Experiment("no_skip", "关闭空窗跳过", {**base, "skip_empty_windows": 0}),
        Experiment("pe256", "PE 256 路", {**base, "pe_mac": 256}),
        Experiment("pe64", "PE 64 路", {**base, "pe_mac": 64}),
        Experiment("serial_txsc", "TX/SC 串行", {**base, "tx_sc_parallel": 0}),
        Experiment("sram256", "Window SRAM 256KB", {**base, "window_sram_kb": 256}),
        Experiment("best_combo_a", "跳过+PE256+并行", {**base, "skip_empty_windows": 1, "pe_mac": 256, "tx_sc_parallel": 1}),
        Experiment("best_combo_b", "跳过+PE256+小SRAM", {**base, "skip_empty_windows": 1, "pe_mac": 256, "window_sram_kb": 256}),
        Experiment("best_combo_c", "跳过+PE192", {**base, "skip_empty_windows": 1, "pe_mac": 192}),
        Experiment("firing_ep24", "ep24 发放率 8.37%", {**base, "firing": 0.08373}),
        Experiment("ultimate", "终极组合", {
            "skip_empty_windows": 1,
            "pe_mac": 256,
            "tx_sc_parallel": 1,
            "window_sram_kb": 256,
            "weight_buffer_kb": 128,
            "firing": 0.07942,
        }),
    ]
    return exps


def run_config(cfg: dict) -> dict[str, float]:
    cfg_path = CONFIG_DIR / "_tmp_run.json"
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(PERF), "--config", str(cfg_path), "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(proc.stdout)
    return data["metrics"]


def goal_met(m: dict[str, float]) -> bool:
    return (
        m["effective_energy_mj"] <= GOAL["effective_energy_mj"]
        and m["fps_at_500mhz"] >= GOAL["fps_at_500mhz"]
        and m["sram_kb"] <= GOAL["sram_kb"]
        and m["epe_drift"] <= GOAL["epe_drift"]
    )


def load_jsonl() -> tuple[list[dict], dict | None]:
    if not JSONL.exists():
        return [], None
    lines = [ln for ln in JSONL.read_text(encoding="utf-8").splitlines() if ln.strip()]
    config = None
    results = []
    for ln in lines:
        obj = json.loads(ln)
        if obj.get("type") == "config":
            config = obj
        else:
            results.append(obj)
    return results, config


def append_result(run_id: int, metrics: dict, status: str, desc: str, segment: int = 0) -> None:
    entry = {
        "run": run_id,
        "commit": "autores",
        "metric": metrics["effective_energy_mj"],
        "metrics": metrics,
        "status": status,
        "description": desc,
        "timestamp": int(time.time()),
        "segment": segment,
    }
    with JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def write_dashboard(results: list[dict], config: dict | None) -> None:
    kept = [r for r in results if r.get("status") == "keep"]
    baseline = results[0]["metric"] if results else 0.0
    best = min(kept, key=lambda r: r["metric"]) if kept else (results[0] if results else None)
    name = config.get("name", "nts07_hw") if config else "nts07_hw"

    lines = [
        f"# Autoresearch 仪表盘：{name}",
        "",
        f"**总轮次：** {len(results)} | **保留：** {len(kept)} | "
        f"**丢弃：** {sum(1 for r in results if r.get('status')=='discard')} | "
        f"**崩溃：** {sum(1 for r in results if r.get('status')=='crash')}",
    ]
    if best:
        delta = (best["metric"] / baseline - 1) * 100 if baseline else 0
        lines.append(
            f"**基线能耗：** {baseline:.2f} mJ (#1)  "
            f"**最优能耗：** {best['metric']:.2f} mJ (#{best['run']}, {delta:+.1f}%)"
        )
    lines += [
        "",
        f"**目标：** 能耗 ≤ {GOAL['effective_energy_mj']} mJ，FPS ≥ {GOAL['fps_at_500mhz']}，SRAM ≤ {GOAL['sram_kb']} KB",
        "",
        "| # | 能耗(mJ) | 周期(M) | FPS | SRAM(KB) | 状态 | 描述 |",
        "|---|---------|---------|-----|----------|------|------|",
    ]
    for r in results:
        m = r.get("metrics", {})
        cyc = m.get("effective_cycles", 0) / 1e6
        fps = m.get("fps_at_500mhz", 0)
        sram = m.get("sram_kb", 0)
        emj = r.get("metric", 0)
        d = (emj / baseline - 1) * 100 if baseline and r["run"] > 1 else 0
        emj_s = f"{emj:.2f}" + (f" ({d:+.1f}%)" if r["run"] > 1 else "")
        lines.append(
            f"| {r['run']} | {emj_s} | {cyc:.2f} | {fps:.1f} | {sram:.0f} | {r.get('status','')} | {r.get('description','')} |"
        )
    DASHBOARD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_worklog(run_id: int, desc: str, metrics: dict, status: str, best_metric: float) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M")
    delta = metrics["effective_energy_mj"] - best_metric
    block = f"""
### 第 {run_id} 轮：{desc} — 能耗={metrics['effective_energy_mj']:.2f}mJ（{status}）
- 时间：{ts}
- 周期：{metrics['effective_cycles']/1e6:.2f} Mcycles，FPS：{metrics['fps_at_500mhz']:.1f}
- SRAM：{metrics['sram_kb']:.0f} KB
- 相对当前最优：{delta:+.2f} mJ
- 目标达成：{'是' if goal_met(metrics) else '否'}
"""
    text = WORKLOG.read_text(encoding="utf-8") if WORKLOG.exists() else "# 工作日志\n"
    if "## 自动实验记录" not in text:
        text += "\n## 自动实验记录\n"
    text += block
    WORKLOG.write_text(text, encoding="utf-8")


def main() -> int:
    results, config = load_jsonl()
    if not JSONL.exists() or not config:
        JSONL.write_text(
            json.dumps(
                {
                    "type": "config",
                    "name": "nts07_hw",
                    "metricName": "effective_energy_mj",
                    "metricUnit": "mJ",
                    "bestDirection": "lower",
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        results = []

    start_run = max((r["run"] for r in results), default=0) + 1
    best_metric = min((r["metric"] for r in results if r.get("status") == "keep"), default=1e9)
    if results and best_metric == 1e9:
        best_metric = results[0]["metric"]

    done_names = {r.get("description", "") for r in results}
    goal_reached = False

    for i, exp in enumerate(default_grid()):
        if exp.description in done_names:
            continue
        run_id = start_run + i
        try:
            metrics = run_config(exp.config)
        except Exception as exc:
            append_result(run_id, {"effective_energy_mj": 0, "effective_cycles": 0, "sram_kb": 0, "fps_at_500mhz": 0, "epe_drift": 0}, "crash", f"{exp.description}: {exc}")
            results.append({"run": run_id, "metric": 0, "status": "crash", "description": exp.description, "metrics": {}})
            continue

        status = "keep" if metrics["effective_energy_mj"] < best_metric else "discard"
        if status == "keep":
            best_metric = metrics["effective_energy_mj"]
        append_result(run_id, metrics, status, exp.description)
        results.append({"run": run_id, "metric": metrics["effective_energy_mj"], "status": status, "description": exp.description, "metrics": metrics})
        append_worklog(run_id, exp.description, metrics, status, best_metric)
        write_dashboard(results, config)

        # 保存优胜配置
        if status == "keep":
            out = ROOT / "scripts" / "configs" / "best_config.json"
            out.write_text(json.dumps(exp.config, indent=2, ensure_ascii=False), encoding="utf-8")

        if goal_met(metrics):
            goal_reached = True
            summary = ROOT / "GOAL_REACHED.md"
            summary.write_text(
                f"# 目标已达成\n\n"
                f"- 轮次：#{run_id} {exp.description}\n"
                f"- 能耗：{metrics['effective_energy_mj']:.2f} mJ（目标 ≤ {GOAL['effective_energy_mj']}）\n"
                f"- FPS：{metrics['fps_at_500mhz']:.1f}（目标 ≥ {GOAL['fps_at_500mhz']}）\n"
                f"- SRAM：{metrics['sram_kb']:.0f} KB\n"
                f"- 配置：`scripts/configs/best_config.json`\n",
                encoding="utf-8",
            )
            break

    write_dashboard(results, config)
    print(f"完成 {len(results)} 轮实验，目标达成：{goal_reached}，最优能耗：{best_metric:.2f} mJ")
    return 0 if goal_reached else 0


if __name__ == "__main__":
    raise SystemExit(main())