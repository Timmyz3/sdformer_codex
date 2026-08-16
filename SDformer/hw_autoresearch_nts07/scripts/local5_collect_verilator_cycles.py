#!/usr/bin/env python3
"""Collect Local5 Verilator cycle sniffs into one equal-lane style table."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build_local5" / "parity"
OUT = ROOT / "results" / "local5_equal_lane_ledger_20260727"


def parse_log(path: Path) -> dict:
    if not path.exists():
        return {"missing": str(path)}
    text = path.read_text(errors="ignore")
    rec: dict = {"log": str(path)}
    m = re.findall(r"CYCLES\s+(\d+)\s+CMDS\s+(\d+)\s+DESTS\s+(\d+)", text)
    if m:
        c, cmds, d = map(int, m[-1])
        rec.update({"cycles": c, "cmds": cmds, "dests": d})
    wins = re.findall(
        r"WINDOW\s+(\d+)\s+CYCLES\s+(\d+)\s+CMDS\s+(\d+)\s+DESTS\s+(\d+)(?:\s+CONFLICTS\s+(\d+))?",
        text,
    )
    if wins:
        rec["windows"] = [
            {
                "id": int(a),
                "cycles": int(b),
                "cmds": int(c),
                "dests": int(d),
                "conflicts": int(e) if e else 0,
            }
            for a, b, c, d, e in wins
        ]
        cyc = [w["cycles"] for w in rec["windows"]]
        rec["mean_cycles"] = sum(cyc) // len(cyc)
        rec["min_cycles"] = min(cyc)
        rec["max_cycles"] = max(cyc)
    sm = re.search(
        r"SUMMARY mean_cycles=(\d+) min_cycles=(\d+) max_cycles=(\d+) row_tokens=(\d+)",
        text,
    )
    if sm:
        rec["summary"] = {
            "mean_cycles": int(sm.group(1)),
            "min_cycles": int(sm.group(2)),
            "max_cycles": int(sm.group(3)),
            "row_tokens": int(sm.group(4)),
        }
    if "PASS" in text:
        rec["pass"] = True
    return rec


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    reports = {
        "window4": parse_log(BUILD / "window_sim.log"),
        "window16": parse_log(BUILD / "w16_sim.log"),
        "linebuf_window8x3": parse_log(BUILD / "lbw_sim.log"),
        "motion_equal96_reference": {
            "Central96": 59853,
            "DCTF96_1C": 62264,
            "DCTF96_2C": 53910,
            "note": "Different workload — H67 sample0/window0 only",
        },
    }
    # scale proxies: cycles/dest
    for k in ("window4", "window16"):
        r = reports[k]
        if "cycles" in r and r.get("dests"):
            r["cycles_per_dest"] = r["cycles"] / r["dests"]
            r["cmds_per_dest"] = r["cmds"] / r["dests"]

    path = OUT / "verilator_cycle_table.json"
    path.write_text(json.dumps(reports, indent=2) + "\n")

    md = ["# Local5 Verilator 周期表（对齐冲刺）\n\n"]
    md.append("| 配置 | dests | cycles | cmds | cycles/dest | 备注 |\n|---|---:|---:|---:|---:|---|\n")
    if "cycles" in reports["window4"]:
        r = reports["window4"]
        md.append(
            f"| direct window4 | {r['dests']} | {r['cycles']} | {r['cmds']} | {r['cycles_per_dest']:.1f} | banklocal |\n"
        )
    if "cycles" in reports["window16"]:
        r = reports["window16"]
        md.append(
            f"| direct window16 | {r['dests']} | {r['cycles']} | {r['cmds']} | {r['cycles_per_dest']:.1f} | banklocal |\n"
        )
    if "summary" in reports["linebuf_window8x3"]:
        s = reports["linebuf_window8x3"]["summary"]
        md.append(
            f"| linebuf 8x3win | 8×3 | mean {s['mean_cycles']} "
            f"[{s['min_cycles']},{s['max_cycles']}] | — | {s['mean_cycles']/8:.1f} | 3-bank+conflicts |\n"
        )
    md.append("\n## Motion equal96 参考（不同 workload）\n\n")
    md.append("| 结构 | cycles |\n|---|---:|\n")
    md.append("| Central96 | 59853 |\n| DCTF96-1C | 62264 |\n| DCTF96-2C | 53910 |\n")
    md.append("\n**隔离**：未改 Codex Motion 源码。\n")
    (OUT / "verilator_cycle_table.md").write_text("".join(md))
    print(OUT / "verilator_cycle_table.md")
    print(json.dumps(reports, indent=2)[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
