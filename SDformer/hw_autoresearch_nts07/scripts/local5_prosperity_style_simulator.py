#!/usr/bin/env python3
"""Local5 / Motion dual-line Prosperity-style component cycle simulator.

IMPORTANT
---------
- Does NOT modify Codex Motion RTL (`rtl_hitflow`, `rtl_h67`, `rtl_delta`).
- Local5 RTL lives only under `rtl_local5/` + `tb_local5/` + `sim_local5/`.
- Prosperity (https://github.com/dubcyfor3/Prosperity) is used for its
  **evaluation structure** (Stats, compute/mem/preprocess split, max-overlap
  total cycles, multi-baseline table). Its CUDA SNN kernels and paper power
  constants are **not** claimed as Local5 PPA.

Evidence tier: [模型] unless fields come from profile100 [prof] or Verilator [RTL].
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
PROSPERITY_SIM = ROOT / "third_party" / "Prosperity" / "simulator"
OUT = ROOT / "results" / "local5_prosperity_style_sim_20260727"

# Prefer official Prosperity Stats class when importable.
_StatsImpl = None
if PROSPERITY_SIM.is_dir():
    sys.path.insert(0, str(PROSPERITY_SIM))
    try:
        from utils import Stats as _ProsperityStats  # type: ignore

        _StatsImpl = _ProsperityStats
    except Exception:
        _StatsImpl = None


def ceil_div(a: float, b: float) -> int:
    return int(math.ceil(float(a) / float(b))) if b else 0


class LocalStats:
    """Fallback Stats compatible with Prosperity fields."""

    def __init__(self) -> None:
        self.total_cycles = 0
        self.mem_stall_cycles = 0
        self.compute_cycles = 0
        self.preprocess_stall_cycles = 0
        self.num_ops = 0
        self.mem_namespace = ["dram", "g_act", "g_wgt", "g_psum", "linebuf"]
        self.reads = {k: 0 for k in self.mem_namespace}
        self.writes = {k: 0 for k in self.mem_namespace}
        self.component = ""
        self.notes: List[str] = []


def make_stats() -> object:
    if _StatsImpl is not None:
        s = _StatsImpl()
        # extend namespace for Local5 line buffer accounting
        if "linebuf" not in s.mem_namespace:
            s.mem_namespace = list(s.mem_namespace) + ["linebuf"]
            s.reads["linebuf"] = 0
            s.writes["linebuf"] = 0
        return s
    return LocalStats()


def finalize_total(s: object, overlap: str = "max") -> object:
    """Prosperity-style: total = compute + max(0, mem - compute) + preprocess
    or sequential sum depending on overlap mode.
    """
    c = int(getattr(s, "compute_cycles", 0))
    m = int(getattr(s, "mem_stall_cycles", 0))
    p = int(getattr(s, "preprocess_stall_cycles", 0))
    if overlap == "max":
        # steady-state overlap of mem behind compute; initial/preprocess add
        s.total_cycles = p + c + max(0, m - c)
    elif overlap == "sum":
        s.total_cycles = p + c + m
    else:
        s.total_cycles = p + max(c, m)
    return s


@dataclass
class ArchConfig:
    name: str
    # frontend
    score_lanes: int = 32  # parallel bit lanes for axnor/TARE
    probe_ports: int = 1  # sequential PROBE in current RTL
    shiftmax_parallel: int = 5
    # MFEP / term
    term_scan_lanes_per_cycle: int = 1  # current MFEP scans lane×gate
    mfep_issue_width: int = 1
    # projection
    proj_banks: int = 3
    proj_cmds_per_cycle: int = 1
    weight_bits: int = 8
    acc_bits: int = 32
    # memory
    sram_rd_latency: int = 2
    linebuf_ports: int = 1
    freq_mhz: float = 500.0  # Prosperity default assumption for time conversion only
    note: str = ""


@dataclass
class WorkloadSlice:
    """Absolute work units for one attention projection context slice."""

    name: str
    destinations: int
    avg_degree: float
    valid_edges: int
    naive_active_products: int
    mfep_terms: int
    offset_terms: int
    unsafe_set_terms: int
    k_source_reads: int  # source-resident K word reads
    k_query_major_reads: int
    source: str


def load_local5_workload() -> WorkloadSlice:
    feat = (
        ROOT
        / "results"
        / "local5_hardware_profile_preG0_profile100_20260726"
        / "local5_hardware_features.json"
    )
    d = json.loads(feat.read_text())
    summary = d["summary"]
    valid_edges = int(summary["valid_edges"])
    token_heads = int(summary.get("token_heads", max(1, valid_edges // 5)))
    return WorkloadSlice(
        name="local5_preG0_profile100_totals",
        destinations=162,
        avg_degree=valid_edges / max(1, token_heads),
        valid_edges=valid_edges,
        naive_active_products=int(summary["naive_active_edge_products"]),
        mfep_terms=int(summary["mfep_multicast_terms"]),
        offset_terms=int(summary["offset_multicast_terms"]),
        unsafe_set_terms=int(summary["unsafe_set_multicast_terms"]),
        k_source_reads=int(summary["source_resident_k_lane_reads"]),
        k_query_major_reads=int(summary["query_major_k_lane_reads"]),
        source=str(feat),
    )


def load_motion_workload_proxy() -> WorkloadSlice:
    """Motion absolute work from dual decision / known GateStack scale."""
    dual_path = ROOT / "results" / "local5_h67_dual_profile_decision_20260726" / "local5_h67_dual_profile_decision.json"
    dual = json.loads(dual_path.read_text())
    h67 = dual.get("h67", {})
    # Use documented absolute active-K / terms when present
    return WorkloadSlice(
        name="motion_h67_profile100_proxy",
        destinations=162,
        avg_degree=float(h67.get("avg_active_tokens", 40)),
        valid_edges=int(h67.get("active_tokens_total", 108_864_000)),  # self-like scale
        naive_active_products=int(h67.get("projection_terms", 7_101_034)),
        mfep_terms=int(h67.get("nmf_terms", 7_101_034)),  # set terms ~ NMF
        offset_terms=0,
        unsafe_set_terms=int(h67.get("nmf_terms", 7_101_034)),
        k_source_reads=int(h67.get("active_k_reads", 36_507_347)),
        k_query_major_reads=int(h67.get("active_k_reads", 36_507_347)),
        source=str(dual_path),
    )


def sim_local5_row_frontend(wl: WorkloadSlice, cfg: ArchConfig) -> object:
    """Model ANCHOR_LOAD + PROBE + Shiftmax5 path matching current RTL shape."""
    s = make_stats()
    s.component = "local5_row_context"
    # each destination: 1 anchor + (degree-1) probes + 1 compute + degree emits
    # profile totals use valid_edges as edge emissions
    n_dest = max(1, wl.valid_edges // max(1, int(round(wl.avg_degree))))
    # Prefer edge-based accounting to stay absolute
    n_edge = wl.valid_edges
    n_dest = ceil_div(n_edge, max(wl.avg_degree, 1.0))

    preprocess = n_dest * 1  # ANCHOR_LOAD
    probe = n_edge - n_dest  # non-self probes approx
    if probe < 0:
        probe = 0
    probe_cycles = ceil_div(probe, cfg.probe_ports)
    compute = n_dest * 1  # ST_COMPUTE / Shiftmax5 comb → 1 cycle retire model
    emit = ceil_div(n_edge, 1)

    s.preprocess_stall_cycles = preprocess
    s.compute_cycles = probe_cycles + compute + emit
    # line-buffer: each dest reads Q + self K + neighbor K from 3-row buffer
    s.reads["linebuf"] = n_dest + n_edge  # Q once + K per edge
    s.reads["g_act"] = 0
    s.mem_stall_cycles = s.reads["linebuf"] * cfg.sram_rd_latency // max(cfg.linebuf_ports, 1)
    s.num_ops = n_edge * cfg.score_lanes
    finalize_total(s, "max")
    return s


def sim_mfep_builder(wl: WorkloadSlice, cfg: ArchConfig, term_count: int, mode: str) -> object:
    s = make_stats()
    s.component = f"local5_term_{mode}"
    # collect edges into builder then scan lane×unique_gates
    collect = wl.valid_edges
    # scan cost: for MFEP, terms are already unique; scanning empty slots modeled as
    # 32 * uniq_gates_per_dest * n_dest, approximate with term_count * fill factor
    if mode == "naive_edge_product":
        issue = wl.naive_active_products
        scan = collect
    elif mode == "offset":
        issue = term_count
        scan = term_count * 2  # direction dimension tax
    elif mode == "unsafe_set":
        issue = term_count
        scan = term_count
    else:  # mfep
        issue = term_count
        scan = term_count + collect  # build unique + emit

    s.preprocess_stall_cycles = ceil_div(collect, 1)
    s.compute_cycles = ceil_div(scan, cfg.term_scan_lanes_per_cycle)
    s.num_ops = issue
    s.writes["g_act"] = issue  # term IR traffic proxy (bits later)
    s.mem_stall_cycles = 0
    finalize_total(s, "sum")
    return s


def sim_banklocal_proj(wl: WorkloadSlice, cfg: ArchConfig, term_count: int, mode: str) -> object:
    s = make_stats()
    s.component = f"local5_proj_{mode}"
    # each term issues 1 cmd; OUT dim folded into bank cycle (model OUT tile = 1 for relative)
    cmds = term_count
    s.compute_cycles = ceil_div(cmds, cfg.proj_cmds_per_cycle * max(cfg.proj_banks, 1))
    # weight reads: one W[lane, :] per term (lane-stationary ideal would reduce; model 1)
    s.reads["g_wgt"] = cmds
    s.writes["g_psum"] = cmds
    s.mem_stall_cycles = (s.reads["g_wgt"] + s.writes["g_psum"]) * cfg.sram_rd_latency // (
        2 * max(cfg.proj_banks, 1)
    )
    s.num_ops = cmds
    finalize_total(s, "max")
    return s


def sim_motion_proxy(wl: WorkloadSlice, cfg: ArchConfig) -> Dict[str, object]:
    """Coarse Motion SCS+NMF+DCTF proxy for dual-line table only."""
    front = make_stats()
    front.component = "motion_scs_proxy"
    # active tokens load + class fold + emit
    front.preprocess_stall_cycles = 162  # clear/load proxy per window * many windows absorbed in totals
    # scale: use valid_edges as active-ish events for motion proxy
    front.compute_cycles = wl.naive_active_products // max(cfg.score_lanes, 1)
    front.reads["g_act"] = wl.k_source_reads
    front.mem_stall_cycles = front.reads["g_act"] * cfg.sram_rd_latency // 2
    finalize_total(front, "max")

    term = make_stats()
    term.component = "motion_nmf_proxy"
    term.compute_cycles = wl.mfep_terms  # NMF terms
    term.num_ops = wl.mfep_terms
    finalize_total(term, "sum")

    proj = sim_banklocal_proj(wl, cfg, wl.mfep_terms, "dctf_set")
    proj.component = "motion_dctf_proxy"
    return {"frontend": front, "term": term, "proj": proj}


def stack_total(parts: List[object]) -> object:
    t = make_stats()
    t.component = "stack"
    for p in parts:
        t.compute_cycles += int(getattr(p, "compute_cycles", 0))
        t.mem_stall_cycles += int(getattr(p, "mem_stall_cycles", 0))
        t.preprocess_stall_cycles += int(getattr(p, "preprocess_stall_cycles", 0))
        t.num_ops += int(getattr(p, "num_ops", 0))
        for k in t.reads:
            t.reads[k] = t.reads.get(k, 0) + int(getattr(p, "reads", {}).get(k, 0))
            t.writes[k] = t.writes.get(k, 0) + int(getattr(p, "writes", {}).get(k, 0))
    # sequential components (score then term then proj) — no full pipeline yet
    finalize_total(t, "sum")
    return t


def stats_to_dict(s: object) -> dict:
    return {
        "component": getattr(s, "component", ""),
        "total_cycles": int(getattr(s, "total_cycles", 0)),
        "compute_cycles": int(getattr(s, "compute_cycles", 0)),
        "mem_stall_cycles": int(getattr(s, "mem_stall_cycles", 0)),
        "preprocess_stall_cycles": int(getattr(s, "preprocess_stall_cycles", 0)),
        "num_ops": int(getattr(s, "num_ops", 0)),
        "reads": dict(getattr(s, "reads", {})),
        "writes": dict(getattr(s, "writes", {})),
        "time_s_at_500MHz": int(getattr(s, "total_cycles", 0)) / (500e6),
    }


def run_local5_ablation(wl: WorkloadSlice, cfg: ArchConfig) -> dict:
    front = sim_local5_row_frontend(wl, cfg)
    modes = {
        "naive_edge_product": wl.naive_active_products,
        "offset_separated": wl.offset_terms if wl.offset_terms else int(wl.mfep_terms * 1.56),
        "unsafe_set_or": wl.unsafe_set_terms if wl.unsafe_set_terms else int(wl.mfep_terms * 0.63),
        "mfep_multiset": wl.mfep_terms,
    }
    out = {"frontend": stats_to_dict(front), "modes": {}}
    for mode, terms in modes.items():
        term_s = sim_mfep_builder(wl, cfg, terms, mode)
        proj_s = sim_banklocal_proj(wl, cfg, terms, mode)
        stack = stack_total([front, term_s, proj_s])
        out["modes"][mode] = {
            "term_count": terms,
            "term": stats_to_dict(term_s),
            "proj": stats_to_dict(proj_s),
            "stack": stats_to_dict(stack),
            "correctness": (
                "exact"
                if mode in ("naive_edge_product", "offset_separated", "mfep_multiset")
                else "UNSAFE_loses_multiplicity"
            ),
        }
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = ArchConfig(
        name="local5_rtl_shaped_v1",
        note="Matches current local5_row_context + mfep + banklocal RTL issue widths",
    )
    wl = load_local5_workload()
    ablation = run_local5_ablation(wl, cfg)

    # dual-line coarse motion proxy (for packaging table only)
    motion_wl = WorkloadSlice(
        name="motion_proxy_from_doc157_absolutes",
        destinations=162,
        avg_degree=40.0,
        valid_edges=108_864_000,
        naive_active_products=7_101_034,
        mfep_terms=7_101_034,
        offset_terms=0,
        unsafe_set_terms=7_101_034,
        k_source_reads=36_507_347,
        k_query_major_reads=36_507_347,
        source="docs/157 absolute work table",
    )
    motion_parts = sim_motion_proxy(motion_wl, cfg)
    motion_stack = stack_total(list(motion_parts.values()))

    prosperity_meta = {
        "repo": "https://github.com/dubcyfor3/Prosperity",
        "local_path": str(PROSPERITY_SIM),
        "stats_import": _StatsImpl is not None,
        "used_for": [
            "Stats field schema",
            "compute / mem_stall / preprocess split",
            "max-overlap total cycle composition",
            "multi-baseline ablation table structure",
        ],
        "not_used_as": [
            "Local5 bit-exact score model",
            "DATE DC area/power numbers",
            "Prosperity paper mW constants as our PPA",
            "CUDA product-sparsity kernel on stencil edges",
        ],
        "phi_simulator": "No public Phi open-source simulator found; Phi used as format baseline idea only",
    }

    report = {
        "schema": "local5_prosperity_style_sim_v1",
        "evidence_tier": "[模型]+[prof workload]; not [RTL cycle equal96]; not [DC]",
        "codex_isolation": {
            "motion_rtl_dirs_unmodified": ["rtl_hitflow", "rtl_h67", "rtl_delta"],
            "local5_only_dirs": ["rtl_local5", "tb_local5", "sim_local5"],
            "reuse": "May instantiate Codex modules from Local5 tops later; this sim does not edit them",
        },
        "prosperity": prosperity_meta,
        "arch_config": asdict(cfg),
        "workload_local5": asdict(wl),
        "local5_ablation": ablation,
        "motion_proxy_stack": stats_to_dict(motion_stack),
        "relative": {},
    }

    mfep_cyc = ablation["modes"]["mfep_multiset"]["stack"]["total_cycles"]
    naive_cyc = ablation["modes"]["naive_edge_product"]["stack"]["total_cycles"]
    report["relative"] = {
        "mfep_vs_naive_stack_cycle_ratio": mfep_cyc / naive_cyc if naive_cyc else None,
        "mfep_term_vs_naive_product": wl.mfep_terms / wl.naive_active_products,
        "mfep_stack_cycles": mfep_cyc,
        "naive_stack_cycles": naive_cyc,
        "motion_proxy_stack_cycles": int(motion_stack.total_cycles),
        "warning": (
            "Local5 and Motion workload slices are NOT equal-lane normalized here; "
            "do not rank DATE winners from this ratio alone."
        ),
    }

    (OUT / "sim_report.json").write_text(json.dumps(report, indent=2) + "\n")

    # Markdown
    lines = []
    lines.append("# Local5 Prosperity 风格组件周期仿真\n\n")
    lines.append("**证据档**：`[模型]` + profile100 `[prof]`；**不是** DC / equal96 RTL 周期。\n\n")
    lines.append("## 0. 与 Codex 代码隔离\n\n")
    lines.append("- **未修改** `rtl_hitflow/`、`rtl_h67/`、`rtl_delta/` 中任何 Codex Motion 模块。\n")
    lines.append("- Local5 仅新增 `rtl_local5/*`、`tb_local5/*`、`sim_local5/*`。\n")
    lines.append("- Codex 叶模块 `local5_axnor/shiftmax5/stencil`（7-25）保持只读复用。\n")
    lines.append("- TARE dual-mode 可由 Local5 top **实例化调用**，本仿真不改其源文件。\n\n")
    lines.append("## 1. Prosperity 用了什么 / 没用什么\n\n")
    lines.append(f"- 本地克隆：`third_party/Prosperity`（官方仓库）。\n")
    lines.append(f"- Stats 导入成功：`{prosperity_meta['stats_import']}`\n")
    lines.append("- 借用：Stats 分账、compute/mem/preprocess、max 重叠、多基线消融表。\n")
    lines.append("- **不借用**：SNN CUDA kernel 结果、论文 mW 常数、28nm DC 脚本（官方未开源）。\n")
    lines.append("- Phi：**无公开仿真器**；只作 pattern+residual 格式基线 idea。\n\n")
    lines.append("## 2. Local5 消融（stack cycles，RTL-shaped issue width）\n\n")
    lines.append("| mode | term_count | stack_cycles | correctness |\n|---|---:|---:|---|\n")
    for mode, rec in ablation["modes"].items():
        lines.append(
            f"| {mode} | {rec['term_count']} | {rec['stack']['total_cycles']} | {rec['correctness']} |\n"
        )
    lines.append("\n")
    lines.append(
        f"- MFEP / naive product 计数比：`{wl.mfep_terms / wl.naive_active_products:.6f}`\n"
    )
    lines.append(
        f"- MFEP / naive **stack cycle** 比：`{report['relative']['mfep_vs_naive_stack_cycle_ratio']:.6f}` "
        f"（含 frontend，模型）\n"
    )
    lines.append("\n## 3. 复现\n\n```bash\n")
    lines.append("python3 scripts/local5_prosperity_style_simulator.py\n")
    lines.append("./sim_local5/run_local5_parity_checks.sh\n```\n")
    (OUT / "sim_report.md").write_text("".join(lines))

    print(OUT / "sim_report.md")
    print(json.dumps(report["relative"], indent=2))
    print("prosperity_stats_import", prosperity_meta["stats_import"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
