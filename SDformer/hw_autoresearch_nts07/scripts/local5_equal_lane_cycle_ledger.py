#!/usr/bin/env python3
"""Equal-lane style cycle/work ledger for Local5 configs vs Motion equal96.

Local5-only analysis script. Does not modify Motion RTL.
Evidence: [模型] + [prof] + optional Verilator cycle sniffs; not DC PPA.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "local5_equal_lane_ledger_20260727"


def ceil_div(a: float, b: float) -> int:
    return int(math.ceil(a / b)) if b else 0


def load_profile():
    p = (
        ROOT
        / "results"
        / "local5_hardware_profile_preG0_profile100_20260726"
        / "local5_hardware_features.json"
    )
    return json.loads(p.read_text())["summary"]


def model_cycles(name: str, terms: int, edges: int, dests: int, cfg: dict) -> dict:
    """RTL-shaped sequential model (matches current engines)."""
    # frontend: 1 anchor + probes + 1 compute + edge emits per dest
    avg_deg = edges / max(dests, 1)
    probes = max(edges - dests, 0)
    front = dests + ceil_div(probes, cfg["probe_ports"]) + dests + edges
    # term scan
    term_c = ceil_div(terms * cfg["term_scan_tax"], cfg["term_issue"])
    # proj
    proj = ceil_div(terms, cfg["proj_banks"] * cfg["proj_ipc"])
    mem = ceil_div(terms, cfg["proj_banks"]) * cfg["sram_lat"]
    total = front + term_c + max(proj, mem)
    return {
        "name": name,
        "terms": terms,
        "front_cycles": front,
        "term_cycles": term_c,
        "proj_cycles": proj,
        "mem_cycles": mem,
        "total_cycles": total,
        "cfg": cfg,
    }


def try_verilator_window_cycles() -> dict | None:
    """Parse last CYCLES line from window TB log if present."""
    log = ROOT / "build_local5" / "parity" / "window_sim.log"
    if not log.exists():
        return None
    text = log.read_text(errors="ignore")
    m = re.findall(r"CYCLES\s+(\d+)\s+CMDS\s+(\d+)\s+DESTS\s+(\d+)", text)
    if not m:
        return None
    c, cmds, dests = map(int, m[-1])
    return {"verilator_cycles": c, "cmds": cmds, "dests": dests, "source": str(log)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    s = load_profile()
    edges = int(s["valid_edges"])
    naive = int(s["naive_active_edge_products"])
    mfep = int(s["mfep_multicast_terms"])
    offset = int(s["offset_multicast_terms"])
    unsafe = int(s["unsafe_set_multicast_terms"])
    dests = int(s.get("token_heads", edges // 5))

    base_cfg = {
        "probe_ports": 1,
        "term_scan_tax": 1.0,
        "term_issue": 1,
        "proj_banks": 1,
        "proj_ipc": 1,
        "sram_lat": 2,
    }
    parallel_cfg = {
        **base_cfg,
        "probe_ports": 4,
        "term_issue": 4,
        "proj_banks": 3,
        "proj_ipc": 2,
    }

    configs = [
        model_cycles("L5_naive_edge_serial", naive, edges, dests, base_cfg),
        model_cycles("L5_offset_serial", offset, edges, dests, base_cfg),
        model_cycles("L5_unsafe_set_serial", unsafe, edges, dests, base_cfg),
        model_cycles("L5_mfep_serial", mfep, edges, dests, base_cfg),
        model_cycles("L5_mfep_parallel_3bank", mfep, edges, dests, parallel_cfg),
    ]

    motion_equal96 = {
        "Central96": 59853,
        "Independent32x3": 59945,
        "DCTF96_1C": 62264,
        "DCTF96_2C": 53910,
        "note": "H67 sample0/window0 only — not same workload as Local5 profile100 totals",
        "source": "results/gatestack_equal96_dctf2c_20260722",
    }

    # Per-window normalized Local5 model: scale profile totals by windows
    # token_heads ≈ destinations across all samples; windows ≈ token_heads/162
    windows = max(1, dests // 162)
    per_window = []
    for c in configs:
        pw = dict(c)
        pw["total_cycles_per_window_proxy"] = ceil_div(c["total_cycles"], windows)
        pw["terms_per_window_proxy"] = ceil_div(c["terms"], windows)
        per_window.append(pw)

    vlog = try_verilator_window_cycles()

    report = {
        "schema": "local5_equal_lane_ledger_v1",
        "evidence": "[模型]+[prof]; Verilator sniff optional; Motion equal96 is different workload",
        "codex_isolation": True,
        "profile_totals": {
            "valid_edges": edges,
            "naive_products": naive,
            "mfep_terms": mfep,
            "offset_terms": offset,
            "unsafe_set_terms": unsafe,
            "destinations_token_heads": dests,
            "windows_proxy": windows,
        },
        "local5_configs_total": configs,
        "local5_configs_per_window_proxy": per_window,
        "motion_equal96_reference": motion_equal96,
        "verilator_window_sniff": vlog,
        "alignment_checklist": {
            "L1_leaf_rtl": True,
            "L2_row_context": True,
            "L3_mfep_term": True,
            "L4_projection_local": True,
            "L4_dctf_bridge": True,
            "L4_line_buffer": True,
            "L4_stt_descriptor": True,
            "L4_window_top": True,
            "L5_equal96_same_sample": False,
            "L5_postG0_trace": False,
            "L6_control_plane": "partial_stt",
            "L8_open_mapping": "partial",
            "L9_dc_ppa": False,
        },
    }

    (OUT / "ledger.json").write_text(json.dumps(report, indent=2) + "\n")

    md = []
    md.append("# Local5 Equal-Lane 风格周期账本\n\n")
    md.append("**隔离**：仅 Local5 新代码 + profile；未改 Motion RTL。\n\n")
    md.append("## Motion equal96 参考（不同 workload）\n\n")
    md.append("| 结构 | cycles |\n|---|---:|\n")
    for k, v in motion_equal96.items():
        if k in ("note", "source"):
            continue
        md.append(f"| {k} | {v} |\n")
    md.append(f"\n> {motion_equal96['note']}\n\n")
    md.append("## Local5 配置模型（profile100 绝对量）\n\n")
    md.append("| config | terms | total_cycles | per_window_proxy |\n|---|---:|---:|---:|\n")
    for c, pw in zip(configs, per_window):
        md.append(
            f"| {c['name']} | {c['terms']} | {c['total_cycles']} | {pw['total_cycles_per_window_proxy']} |\n"
        )
    md.append("\n## Verilator 窗口嗅探\n\n")
    md.append(f"```\n{json.dumps(vlog, indent=2)}\n```\n")
    md.append("\n## 与 Motion 进度对齐检查表\n\n")
    for k, v in report["alignment_checklist"].items():
        md.append(f"- `{k}`: **{v}**\n")
    (OUT / "ledger.md").write_text("".join(md))
    print(OUT / "ledger.md")
    print(json.dumps(report["alignment_checklist"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
