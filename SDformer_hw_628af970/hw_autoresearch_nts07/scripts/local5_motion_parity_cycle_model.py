#!/usr/bin/env python3
"""Dual-line Local5 vs Motion functional-chain parity + cycle/work model.

This is a [模型]/[prof] ledger, not DC PPA or full equal96 real-trace cycles.
It reuses existing dual-profile numbers and estimates the new Local5 RTL chain:
  ANCHOR_LOAD + PROBE + Shiftmax5 + MFEP + bank-local cmd
against Motion SCS + NMF/G1 + DCTF term path structure.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "local5_motion_parity_20260727"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    dual = load_json(
        ROOT
        / "results"
        / "local5_h67_dual_profile_decision_20260726"
        / "local5_h67_dual_profile_decision.json"
    )
    motion_equal96 = ROOT / "results" / "gatestack_equal96_dctf2c_20260722"
    motion_cycles = {
        "central96": 59853,
        "dctf_1c": 62264,
        "dctf_2c": 53910,
        "acc32_match": 233280,
        "note": "real-trace equal-lane GateStack/DCTF (existing Motion evidence)",
    }

    local5 = dual["local5"]
    motion = dual.get("h67", dual.get("motion", {}))

    # Functional chain completeness (honest gates)
    chain = {
        "motion": {
            "score": "h67_motionxor + SCS row engine [RTL+trace]",
            "gate": "SCS Shiftmax/class fold [RTL+trace]",
            "term": "NMF/G1 builder [RTL+trace]",
            "proj": "DCTF-1C/2C bank-local [RTL+equal96 cycles]",
            "shared_residual": "TARE-4 dual-mode [RTL+trace]",
            "completion_score": 0.95,
        },
        "local5_before": {
            "score": "leaf axnor + TARE single edge [RTL synthetic]",
            "gate": "leaf Shiftmax5 [RTL comb]",
            "term": "profile-only MFEP [prof]",
            "proj": "missing",
            "shared_residual": "TARE-4 dual-mode Local5 mode [RTL synthetic]",
            "completion_score": 0.35,
        },
        "local5_after": {
            "score": "row-context ANCHOR_LOAD+PROBE [RTL verilator]",
            "gate": "Shiftmax5 in row retire [RTL verilator]",
            "term": "MFEP multiset builder [RTL verilator]",
            "proj": "MFEP->DCTF cmd adapter + banklocal Acc top [RTL+yosys partial]",
            "shared_residual": "TARE-4 dual-mode (score substrate) [RTL]",
            "completion_score": 0.78,
            "remaining": [
                "post-G0 ordered real-trace profile",
                "full DCTF-96 fabric multiset planes (not lightweight banklocal)",
                "multi-destination window scheduler / line-buffer SRAM",
                "equal-lane cycle table vs Motion equal96 on same samples",
                "DC/STA/SAIF PPA",
            ],
        },
    }

    # Work models from dual profile
    l5_valid_edges = int(local5["valid_edges"])
    l5_mfep_ratio = float(local5["mfep_term_ratio_pre_g0"])
    l5_mfep_terms = int(round(l5_valid_edges * float(local5.get("offset_term_ratio_pre_g0", 0)) * 0))  # placeholder
    # Prefer absolute-ish numbers from dual decision if present
    # Reconstruct from ratios in summary when abs missing
    naive_products = None
    mfep_terms = None
    # Use stage-aggregated fields if available via features json
    feat_path = Path(dual["local5_source"])
    if not feat_path.exists():
        # symlink tree may point at sdformer_codex path
        alt = ROOT / "results" / "local5_hardware_profile_preG0_profile100_20260726" / "local5_hardware_features.json"
        feat_path = alt
    if feat_path.exists():
        feat = load_json(feat_path)
        totals = feat.get("totals") or feat.get("summary") or feat
        if "mfep_multicast_terms" in totals:
            mfep_terms = int(totals["mfep_multicast_terms"])
        if "naive_active_edge_products" in totals:
            naive_products = int(totals["naive_active_edge_products"])

    if mfep_terms is None:
        mfep_terms = 0
        model_note = "mfep_terms missing from profile"
    else:
        model_note = "mfep_terms from local5_hardware_features absolute"

    # Per-destination RTL cycle model (from design of new engines)
    # cycles ≈ 1 anchor + n_probe + 1 compute + n_edge_emit + 1 build
    #         + (n_uniq_gates * 32 scan slots, empty skipped in 1c each)
    #         + n_terms cmd + 1 done
    # Synthetic TB mean from N_VEC=8 score_gate_term (515 cmds / 8 ≈ 64.4 terms/dest)
    synth = {
        "vectors": 8,
        "cmds_total": 515,
        "cmds_per_dest_mean": 515 / 8,
        "row_context_edges_checked": 96,
        "row_context_rows": 24,
        "mfep_checked_terms": 44,
        "verilator": "PASS row + mfep + score_gate_term_top",
    }

    # Idea fusion map (ECTP/paper mechanisms) with evidence tier
    ideas = [
        {
            "idea": "Prosperity-inspired static residual anchor",
            "local5": "ANCHOR_LOAD self + PROBE neighbors; TARE residual core shared",
            "tier": "[RTL partial] edge TARE + row-context direct score",
        },
        {
            "idea": "Bishop-inspired density stratification",
            "local5": "TARE ZERO/SPARSE/DENSE classifier shared with Motion",
            "tier": "[RTL] classifier; STT descriptor not yet",
        },
        {
            "idea": "FireFly-T multi-lane extract",
            "local5": "inside TARE LIST4 path",
            "tier": "[RTL] shared leaf",
        },
        {
            "idea": "FLAT/FuseMax-like materialization-free projection",
            "local5": "MFEP multiset terms -> banklocal Acc without A[T,C]",
            "tier": "[RTL prototype] lightweight Acc, not full DCTF-96",
        },
        {
            "idea": "LoAS/source-stationary organization",
            "local5": "row-context + MFEP dest-centric now; line-buffer TBD",
            "tier": "[设计/部分RTL]",
        },
        {
            "idea": "SpAtten-like exact skip cascade",
            "local5": "zero-gate product skip in MFEP",
            "tier": "[RTL] product skip only",
        },
        {
            "idea": "ECTP framing (diffs/ablations, not PADE laundering)",
            "local5": "dual-line chain table + term-count vs naive products",
            "tier": "[文档/模型] packaging",
        },
        {
            "idea": "DCTF multiset multiplicity planes",
            "local5": "cmd_multiplicity sideband x1..x5",
            "tier": "[RTL adapter] not full fabric planes",
        },
    ]

    report = {
        "schema": "local5_motion_parity_v1",
        "date": "2026-07-27",
        "chain_completion": chain,
        "motion_equal96_cycles": motion_cycles,
        "local5_profile_preG0": {
            "samples": local5.get("samples"),
            "valid_edges": local5.get("valid_edges"),
            "delta_lane_density": local5.get("delta_lane_density"),
            "exact_k_edge_ratio": local5.get("exact_k_edge_ratio"),
            "mfep_term_ratio_pre_g0": local5.get("mfep_term_ratio_pre_g0"),
            "mfep_term_count_reduction_pre_g0": local5.get(
                "mfep_term_count_reduction_pre_g0"
            ),
            "topology_k_read_reduction": local5.get("topology_k_read_reduction"),
        },
        "work_model": {
            "naive_products": naive_products,
            "mfep_terms": mfep_terms,
            "note": model_note,
            "evidence": "[prof]/[模型] — not RTL cycle equal96",
        },
        "rtl_synth_smoke": synth,
        "idea_fusion": ideas,
        "parity_verdict": {
            "functional_chain_closed": True,
            "equal_to_motion_evidence_depth": False,
            "reason": (
                "Local5 now has score->gate->term->proj RTL chain with Verilator "
                "bit-checks, but lacks Motion-grade real-trace equal96 cycles, "
                "full DCTF fabric, and DC PPA."
            ),
            "local5_completion_before": 0.35,
            "local5_completion_after": 0.78,
            "motion_completion": 0.95,
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "parity_report.json").write_text(json.dumps(report, indent=2) + "\n")

    md = []
    md.append("# Local5 ↔ Motion 硬件完成度追平报告（功能链）\n")
    md.append("**日期**：2026-07-27\n")
    md.append("**证据边界**：Verilator 功能链 PASS；preG0 profile；**不是** DC PPA / equal96 实迹周期表。\n")
    md.append("\n## 1. 完成度对照\n")
    md.append("| 环节 | Motion | Local5 之前 | Local5 现在 |\n|---|---|---|---|\n")
    md.append(
        f"| Score | {chain['motion']['score']} | {chain['local5_before']['score']} | {chain['local5_after']['score']} |\n"
    )
    md.append(
        f"| Gate | {chain['motion']['gate']} | {chain['local5_before']['gate']} | {chain['local5_after']['gate']} |\n"
    )
    md.append(
        f"| Term | {chain['motion']['term']} | {chain['local5_before']['term']} | {chain['local5_after']['term']} |\n"
    )
    md.append(
        f"| Proj | {chain['motion']['proj']} | {chain['local5_before']['proj']} | {chain['local5_after']['proj']} |\n"
    )
    md.append(
        f"| 完成度(自评) | {chain['motion']['completion_score']} | {chain['local5_before']['completion_score']} | {chain['local5_after']['completion_score']} |\n"
    )
    md.append("\n## 2. 仿真冒烟\n")
    md.append(f"- row_context: {synth['row_context_edges_checked']} edges / {synth['row_context_rows']} rows PASS\n")
    md.append(f"- mfep_term_builder: {synth['mfep_checked_terms']} terms PASS\n")
    md.append(
        f"- score_gate_term_top: {synth['cmds_total']} cmds / {synth['vectors']} dest, mean {synth['cmds_per_dest_mean']:.2f} cmds/dest PASS\n"
    )
    md.append("\n## 3. Motion 已有 equal96 周期（对照，非 Local5 新测）\n")
    md.append(f"- Central96: {motion_cycles['central96']}\n")
    md.append(f"- DCTF-1C: {motion_cycles['dctf_1c']}\n")
    md.append(f"- DCTF-2C: {motion_cycles['dctf_2c']}\n")
    md.append(f"- acc32 match: {motion_cycles['acc32_match']}\n")
    md.append("\n## 4. Local5 preG0 工作量机会（[prof]）\n")
    md.append(f"- valid_edges: {local5.get('valid_edges')}\n")
    md.append(f"- MFEP term ratio: {local5.get('mfep_term_ratio_pre_g0')}\n")
    md.append(
        f"- MFEP term-count reduction: {local5.get('mfep_term_count_reduction_pre_g0')}\n"
    )
    md.append(
        f"- topology K-read reduction: {local5.get('topology_k_read_reduction')}\n"
    )
    md.append("\n## 5. Idea 融合与证据档\n")
    md.append("| Idea | Local5 落地 | 证据档 |\n|---|---|---|\n")
    for it in ideas:
        md.append(f"| {it['idea']} | {it['local5']} | {it['tier']} |\n")
    md.append("\n## 6. 诚实结论\n")
    md.append(
        "- **功能链已闭合**到与 Motion 同构的 score→gate→term→proj，Local5 自评完成度约 0.35→0.78。\n"
    )
    md.append(
        "- **证据深度尚未追平** Motion：缺 post-G0 实迹、equal96 周期表、完整 DCTF multiset fabric、DC/STA。\n"
    )
    md.append(
        "- 不得把 preG0 MFEP 92.7% term 压缩写成 accelerator speedup。\n"
    )
    md.append("\n## 7. 复现\n")
    md.append("```bash\n")
    md.append("./sim_local5/run_local5_parity_checks.sh\n")
    md.append("python3 scripts/local5_motion_parity_cycle_model.py\n")
    md.append("```\n")
    (OUT / "parity_report.md").write_text("".join(md))
    print(OUT / "parity_report.md")
    print(json.dumps(report["parity_verdict"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
