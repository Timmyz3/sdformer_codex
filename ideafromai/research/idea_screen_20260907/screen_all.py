#!/usr/bin/env python3.12
"""S1 screen of every idea-catalog candidate. Proposals + sealed receipts, not paper RTL."""
from __future__ import annotations

import json
import hashlib
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path("/home/zhumd/work")
OUT = Path("/home/zhumd/work/ideafromai/research/idea_screen_20260907")
HW = ROOT / "sdformer_codex/SDformer/hw_autoresearch_nts07"
IDEA = ROOT / "ideafromai/research"
QK_DIR = (
    HW
    / "system_handoff/received/h67_ep35_system_trace_handoff_20260821"
    / "h67_ep35_system_trace_handoff_20260821/trace_qk_100sample_12block"
)
AEE_EP34 = 1.199514
AEE_FLOOR = 1.259
AEE_SDFORMER = 1.5848


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def loadj(p: Path):
    return json.loads(p.read_text())


def gate_aee(aee: float | None) -> str:
    if aee is None:
        return "NO_AEE"
    if aee <= AEE_FLOOR and aee < AEE_SDFORMER:
        return "PASS_NEW_GATE"
    return "FAIL_NEW_GATE"


def unpack_qk(npz_path: Path):
    z = np.load(npz_path)
    q_shape = tuple(int(x) for x in z["q_shape"])
    k_shape = tuple(int(x) for x in z["k_shape"])
    q = np.unpackbits(z["q_bits_packed"]).reshape(q_shape)
    k = np.unpackbits(z["k_bits_packed"]).reshape(k_shape)
    return q.astype(np.uint8), k.astype(np.uint8)


def attn_stats(q: np.ndarray, k: np.ndarray) -> dict:
    # (T, ..., C) with T=2
    assert q.shape[0] == 2 and k.shape[0] == 2
    c = q.shape[-1]
    q0, q1 = q[0], q[1]
    k0, k1 = k[0], k[1]
    tok = int(np.prod(q0.shape[:-1]))
    q0r, q1r = q0.reshape(tok, c), q1.reshape(tok, c)
    k0r, k1r = k0.reshape(tok, c), k1.reshape(tok, c)
    k_zero = (k0r.sum(1) == 0) & (k1r.sum(1) == 0)
    k0_zero = k0r.sum(1) == 0
    k1_zero = k1r.sum(1) == 0
    q_dirty = (q0r != q1r).any(1)
    k_dirty = (k0r != k1r).any(1)
    dirty = q_dirty | k_dirty
    # Motion-XOR terms vs temporal peer (token-aligned T=2)
    overlap = (q1r & k1r).sum(1)
    motion = (k1r ^ k0r).sum(1)
    same_zero = ((q1r == 0) & (k1r == 0)).sum(1)
    # SDSA-style work: sum_i Q then scale K — count nonzeros
    q_pop = q1r.sum(1)
    # dual-spike skip: Q and K both all-zero at a timestep
    both0 = (q1r.sum(1) == 0) & (k1r.sum(1) == 0)
    score_leaf_needed = dirty & (~k_zero)
    return {
        "tokens": tok,
        "k_zero_both_t": int(k_zero.sum()),
        "k0_zero": int(k0_zero.sum()),
        "k1_zero": int(k1_zero.sum()),
        "q_dirty": int(q_dirty.sum()),
        "k_dirty": int(k_dirty.sum()),
        "dirty_or": int(dirty.sum()),
        "score_leaf_needed": int(score_leaf_needed.sum()),
        "both_qk_zero_t1": int(both0.sum()),
        "overlap_mean": float(overlap.mean()),
        "motion_xor_mean": float(motion.mean()),
        "same_zero_mean": float(same_zero.mean()),
        "q_pop_t1_mean": float(q_pop.mean()),
        "q_nnz": int(q1r.sum() + q0r.sum()),
        "k_nnz": int(k1r.sum() + k0r.sum()),
        "bits": int(q.size),
    }


def screen_attention() -> dict:
    files = sorted(QK_DIR.glob("sample*_S*_B*_attn.npz"))
    acc = defaultdict(int)
    float_acc = defaultdict(float)
    n = 0
    per_stage = defaultdict(lambda: defaultdict(int))
    for p in files:
        q, k = unpack_qk(p)
        s = attn_stats(q, k)
        n += 1
        for k_, v in s.items():
            if isinstance(v, float):
                float_acc[k_] += v
            else:
                acc[k_] += v
        # sample12_S2_B3
        name = p.stem  # sample0_S0_B0_attn
        parts = name.split("_")
        stage = parts[1]
        for k_ in ("tokens", "k_zero_both_t", "dirty_or", "score_leaf_needed"):
            per_stage[stage][k_] += s[k_]
    tok = acc["tokens"]
    out = {
        "identity": "H67 Q/K packed traces, 100 samples x 12 blocks",
        "checkpoint_in_trace": "ep35_path_NOT_ep34",
        "n_files": n,
        "tokens": tok,
        "k_zero_both_t_frac": acc["k_zero_both_t"] / tok,
        "k0_zero_frac": acc["k0_zero"] / tok,
        "k1_zero_frac": acc["k1_zero"] / tok,
        "dirty_or_frac": acc["dirty_or"] / tok,
        "score_leaf_needed_frac": acc["score_leaf_needed"] / tok,
        "ideal_skip_if_kzero_or_clean": 1.0 - acc["score_leaf_needed"] / tok,
        "both_qk_zero_t1_frac": acc["both_qk_zero_t1"] / tok,
        "q_bit_density": acc["q_nnz"] / acc["bits"],
        "k_bit_density": acc["k_nnz"] / acc["bits"],
        "overlap_mean": float_acc["overlap_mean"] / n,
        "motion_xor_mean": float_acc["motion_xor_mean"] / n,
        "same_zero_mean": float_acc["same_zero_mean"] / n,
        "per_stage": {
            st: {
                "tokens": v["tokens"],
                "k_zero_frac": v["k_zero_both_t"] / v["tokens"],
                "dirty_frac": v["dirty_or"] / v["tokens"],
                "leaf_needed_frac": v["score_leaf_needed"] / v["tokens"],
            }
            for st, v in sorted(per_stage.items())
        },
        "qk_dir_sha256_manifest": sha256(QK_DIR / "manifest.json") if (QK_DIR / "manifest.json").exists() else None,
    }
    return out


def tsbg_summary():
    p = HW / "results/tsbg_ep34_same_io_b2_b4_b8_quickkill_r1_20260902/result.json"
    d = loadj(p)
    rows = [r for r in d["rows"] if r.get("scope_type") == "all" and r.get("scope") == "FC1_FC2"]
    by = {}
    for r in rows:
        by[r["bundle"]] = {
            "conservative_serialized_speedup": r["conservative_serialized_speedup"],
            "roofline_speedup": r["roofline_speedup"],
            "weight_fetch_ratio": r["weight_fetch_ratio"],
            "weight_byte_reduction": r["weight_byte_reduction"],
            "cycle_gate_ge_1p15": r["cycle_gate_ge_1p15"],
            "energy_branch_weight_reduction_ge_30pct": r["energy_branch_weight_reduction_ge_30pct"],
        }
    return {"path": str(p), "checkpoint_sha_prefix": d["identity"]["checkpoint_sha256"][:8], "by_bundle": by, "decisions": d["decisions"]}


def s2_summary():
    p = HW / "results/m1713_ep34_s2_fc_patch_zero_cost_upper_bound_fastkill_r1_20260901/result.json"
    d = loadj(p)
    return {
        "path": str(p),
        "decision": d["decision"],
        "family_upper_bounds": [
            {
                "object": x["object"],
                "zero_cost_complete_elimination_upper_bound": x["zero_cost_complete_elimination_upper_bound"],
                "min_frac_remaining_to_drop_for_1p15x": x["minimum_fraction_of_family_remaining_work_to_drop_for_1p15x"],
                "direct_no_go_below_1p15": x["direct_no_go_below_1p15"],
            }
            for x in d["family_upper_bounds"]
        ],
    }


def paft_summary():
    p = HW / "results/m247_paft_vs_control_paired_valid825_r1_20260825/m247_paft_vs_control_paired_valid825_r1.json"
    d = loadj(p)
    hw = d["hardware_decision"]
    # running-BN PAFT vs control: improvement percent. Need absolute AEE if present
    abs_aee = None
    for key in ("paft_running_aee", "running_aee", "aee"):
        if key in hw:
            abs_aee = hw[key]
    # later tau=1 destructive number from known sealed text
    tau1 = {
        "note": "selective tau=1 on PAFT-ep4 running-BN identity, not Motion C12 ep34",
        "control_aee": 1.46915,
        "candidate_aee": 1.49848,
        "delta": 0.02933,
        "new_gate": gate_aee(1.49848),
    }
    running_gain = hw["paft_running_aee_improvement_percent"]
    return {
        "m247_path": str(p),
        "m247_running_bn_aee_improvement_percent": running_gain,
        "m247_identity": "PAFT ep4, NOT ep34",
        "m247_new_gate_if_control_is_1p47": gate_aee(1.46915 * (1 - running_gain / 100.0)),
        "tau1_valid825": tau1,
        "headline_admitted_in_receipt": d["admission"]["paft_accuracy_headline"],
    }


def c1_full_layer():
    p = IDEA / "complete_transfer_20260907/c1_full_layer_r1.json"
    d = loadj(p)
    points = d["points"]
    out = []
    for pt in points:
        modes = {}
        for name, m in pt["modes"].items():
            c = m["official_outer_counts"]
            modes[name] = {
                "total_cycles": c["total_cycles"],
                "compute_cycles": c["compute_cycles"],
                "mem_stall_cycles": c["mem_stall_cycles"],
            }
        if "bit" in modes and "product" in modes:
            b, pr = modes["bit"], modes["product"]
            modes["product_over_bit_total"] = b["total_cycles"] / pr["total_cycles"] if pr["total_cycles"] else None
            modes["mem_stall_frac_bit"] = b["mem_stall_cycles"] / b["total_cycles"] if b["total_cycles"] else None
        out.append({"config": pt["config"]["id"], "modes": modes})
    return {"path": str(p), "n_points": len(points), "first_two": out[:2], "status": d["status"]}


def c1_parent_promote():
    p = IDEA / "c1_retained_parent_promotion_20260907/result_r1.json"
    d = loadj(p)
    return {"path": str(p), "status": d.get("status"), "excerpt_keys": list(d)[:20]}


def c2_bank_and_equiv():
    bank = loadj(IDEA / "complete_transfer_20260907/c2_bank_mode_r1.json")
    eq = loadj(IDEA / "complete_transfer_20260907/c2_equivalence_r1.json")
    return {
        "bank_status": bank.get("status"),
        "bank_keys": list(bank)[:15],
        "eq_status": eq.get("status"),
        "eq_keys": list(eq)[:15],
    }


def source_order():
    d = loadj(IDEA / "fusion_delivery_20260907/source_order_r1.json")
    layers = []
    for L in d["layers"]:
        item = {"module": L["module"], "slots": {}}
        for slot, pt in L["points"].items():
            item["slots"][slot] = {
                "adds_over_direct_FTP": pt["adds_over_direct_FTP"],
                "adds_over_full_flat_groups": pt["adds_over_full_flat_groups"],
                "add_reduction_vs_FTP_pct": (1 - pt["adds_over_direct_FTP"]) * 100,
            }
        layers.append(item)
    return {"status": d["status"], "layers": layers}


def c2_rtl():
    p = IDEA / "c2_temporal_shared_protocol_20260907/records/functional_r2/result.json"
    d = loadj(p)
    return {"path": str(p), "status": d.get("status"), "keys": list(d)[:20]}


def late_theta():
    p = IDEA / "hardware_mechanisms_20260906/records/functional_r3/result.json"
    d = loadj(p)
    return {"path": str(p), "status": d.get("status"), "keys": list(d)[:20]}


def prosperity_fusion():
    p = IDEA / "prosperity_fusion_20260906/same_cohort_strong_baseline.json"
    if not p.exists():
        return {"missing": str(p)}
    d = loadj(p)
    return {"path": str(p), "keys": list(d)[:20], "status": d.get("status")}


def card(cid, pitch, stage, status, numbers, next_action, aee=None, notes=None):
    return {
        "id": cid,
        "pitch": pitch,
        "s1_status": status,
        "highest_stage_reached": stage,
        "aee_gate": gate_aee(aee),
        "aee": aee,
        "numbers": numbers,
        "next": next_action,
        "notes": notes,
    }


def main():
    attn = screen_attention()
    tsbg = tsbg_summary()
    s2 = s2_summary()
    paft = paft_summary()
    c1 = c1_full_layer()
    promo = c1_parent_promote()
    c2be = c2_bank_and_equiv()
    so = source_order()
    rtl_c2 = c2_rtl()
    theta = late_theta()
    pf = prosperity_fusion()

    cards = []
    # A cluster
    cards.append(card("A1", "Prosperity complete-chain copy onto H67", "S1", "SEALED_REPLAY",
                      {"c1_full_layer": c1, "prosperity_fusion": pf},
                      "Optional: dual-port parent SRAM model; do not call it a new mechanism until vs official run_fc same-resource"))
    cards.append(card("A2", "In-register parent promotion", "S1", "SEALED_REPLAY",
                      {"result": promo}, "Only retry if layout/ports change; +1.58% add was the increment"))
    cards.append(card("A3", "Phi PAFT hardware-aware sparsity training", "S3_OLD_IDENTITY", "SEALED_REPLAY",
                      paft, "Must retrain on ep34; PAFT-ep4 running AEE ~1.47 already fails 1.259 floor",
                      aee=1.46915))
    b8 = tsbg["by_bundle"].get(8, {})
    cards.append(card("A4", "TSBG weight-row broadcast B2/B4/B8", "S1", "SEALED_REPLAY_EP34",
                      tsbg, "CPU premodel GO at 1.15 cycle gate; RTL still separate. Under new contract this is retry-not-killed."))
    cards.append(card("A5", "Lossy S2 block skip", "S1", "SEALED_REPLAY_EP34",
                      s2, "FC2 zero-cost UB<1.15 NO-GO; FC1/patch only if drop remaining work 39.5%/22.8% AND paired AEE"))
    cards.append(card("A6", "C2 temporal shared partial-sum RTL", "S4_FUNC", "SEALED_RTL_FUNC",
                      {"rtl": rtl_c2, "source_order_add_reduction": so},
                      "Wire bank-return + persistent Y; compare same-resource vs direct FTP"))
    cards.append(card("A7", "Per-bank one 10-bit mode", "S1", "SEALED_REPLAY",
                      {"bank": c2be["bank_status"]},
                      "Remap capture address==bank or stop claiming physical request equality"))
    cards.append(card("A8", "Late-known BN theta packet", "S4_FUNC", "SEALED_RTL_FUNC",
                      {"rtl": theta}, "Feed real BN/PSN intervals; measure restore rate"))
    cards.append(card("A9", "Motion residual default state", "S0", "NO_RTL_NO_AEE",
                      {}, "Needs DSEC train; skip until AEE probe"))

    # B cluster from live QK
    cards.append(card("B1", "Motion-XOR triple-popcount score ALU", "S1", "NEW_QK_CENSUS_EP35",
                      {"overlap_mean": attn["overlap_mean"], "motion_xor_mean": attn["motion_xor_mean"],
                       "same_zero_mean": attn["same_zero_mean"], "q_bit_density": attn["q_bit_density"],
                       "k_bit_density": attn["k_bit_density"]},
                      "Need ep34 QK census before paper table; leaf object viable"))
    cards.append(card("B2", "Dirty-lane skip of score leaf", "S1", "NEW_QK_CENSUS_EP35",
                      {"dirty_or_frac": attn["dirty_or_frac"],
                       "score_leaf_needed_frac": attn["score_leaf_needed_frac"],
                       "ideal_skip": attn["ideal_skip_if_kzero_or_clean"],
                       "per_stage": attn["per_stage"]},
                      "Row-level Shiftmax denom still unmeasured; do not treat token skip as row skip"))
    cards.append(card("B3", "K=0 skip score and V", "S1", "NEW_QK_CENSUS_EP35",
                      {"k_zero_both_t_frac": attn["k_zero_both_t_frac"],
                       "k0_zero_frac": attn["k0_zero_frac"],
                       "k1_zero_frac": attn["k1_zero_frac"]},
                      "Lossless if K-as-V holds; combine with B2"))
    cards.append(card("B4", "Replace Motion-XOR with linear QK/SDSA in window", "S0", "NEEDS_OVERLAY_FINETUNE",
                      {}, "S2: one stage2 block swap, 10-frame AEE from ep34"))
    cards.append(card("B5", "STSA spatiotemporal spiking attention", "S0", "NEEDS_OVERLAY_FINETUNE",
                      {"tw": 2}, "T_w=2 may be too short; S2 after kernel swap"))
    cards.append(card("B6", "SLI local path + SSA fusion", "S0", "NEEDS_OVERLAY_FINETUNE",
                      {}, "Add depthwise beside SSA; A800 only after S2"))
    cards.append(card("B7", "SQKFormer IECA+MABN", "S0", "CONTRACT_CONFLICT_RISK",
                      {}, "Align with PSN/ATLIF BN before any train"))
    cards.append(card("B8", "LRF-SSA neural-dynamics attention", "S0", "NEEDS_OVERLAY_FINETUNE",
                      {}, "Plugin on ep34; S2 10-frame"))
    cards.append(card("B9", "FireFly-T binary engine retargeted to Motion-XOR", "S1", "MAPPED_FROM_QK_CENSUS",
                      {"three_pop_means": [attn["overlap_mean"], attn["same_zero_mean"], attn["motion_xor_mean"]]},
                      "AND-PopCount is only overlap term; must add K_peer XOR"))
    cards.append(card("B10", "Dual-spike skip attention datapath", "S1", "NEW_QK_CENSUS_EP35",
                      {"both_qk_zero_t1_frac": attn["both_qk_zero_t1_frac"],
                       "q_bit_density": attn["q_bit_density"],
                       "k_bit_density": attn["k_bit_density"]},
                      "Q already very sparse; dual-spike skip is almost Q-skip"))

    cards.append(card("C1", "Mixed-horizon ATLIF island", "S0", "NEEDS_ISLAND_SPEC",
                      {"t_attn": 2, "t_neuron": 10}, "Reuse C3 coverage RTL as rate converter"))
    cards.append(card("C2", "Learned / dynamic T", "S0", "NEEDS_TRAIN",
                      {}, "AEE-sensitive; S2 with T=1/2/4 on subset"))
    cards.append(card("C3", "Spiking Patches tokenizer", "S0", "NEEDS_RETRAIN_FROM_SDFORMERFLOW",
                      {}, "Changes input identity"))
    cards.append(card("C4", "Fold theta into W vs explicit payload", "S1", "IDENTITY_LOCKED_AS_THETA_G",
                      {"z": "theta*g", "ep34_aee": AEE_EP34},
                      "Changing contract requires S3 AEE", aee=AEE_EP34))

    cards.append(card("D1", "LoAS mixed-T FTP join", "S1", "SEALED_CPU_REF",
                      {"source_order": so}, "Complete T FTP is the strong baseline; candidate must beat it same-resource"))
    cards.append(card("D2", "RSR++ shared reduction", "S1", "SEALED_CPU_REF",
                      {"fc2_signatures": str(IDEA / "fusion_delivery_20260907/fc2_signatures_r1.json")},
                      "Keep as comparator, not auto title"))
    cards.append(card("D3", "Decoder ConvTranspose sparse island", "S1", "PARTIAL_EP34_SHARDS",
                      {"shard_dir_glob": "results/m1681_ep34_decoder_d0_shard_*"},
                      "Need complete decoder Table-A before system claim"))
    cards.append(card("D4", "Dual-side weight+activation sparsity", "S0", "NEEDS_RETRAIN",
                      {"old_nm_audit": "FP32 weights have no exact zero blocks"},
                      "Retrain prune+4bit then S3"))
    cards.append(card("D5", "One fabric, many operators", "S0", "ARCHITECTURE_OPTION",
                      {}, "Only after A1/D1/D3 share a parent-forest ISA"))

    cards.append(card("E1", "SDSA vs Motion-XOR ablation", "S0", "NEEDS_OVERLAY_FINETUNE",
                      {}, "S2 10-frame"))
    cards.append(card("E2", "QKFormer replace stage2 only", "S0", "NEEDS_OVERLAY_FINETUNE",
                      {"stage2_blocks": 6}, "Cheapest network swap"))
    cards.append(card("E3", "Spike-driven Transformer v3 blocks", "S0", "NEEDS_OVERLAY_FINETUNE",
                      {}, "Keep mul-free attention"))
    cards.append(card("E4", "Retile window / Tw", "S0", "NEEDS_RETRAIN_FROM_SDFORMERFLOW",
                      {}, "Geometry change"))

    report = {
        "date": str(date.today()),
        "gates": {"aee_floor": AEE_FLOOR, "sdformer": AEE_SDFORMER, "ep34": AEE_EP34},
        "attention_census": attn,
        "cards": cards,
        "claim_boundary": [
            "S1/S4-func only except PAFT which had old valid825 on a different checkpoint",
            "QK census is ep35-path 100-sample packed traces, not sealed ep34",
            "No component speedups multiplied",
            "No winner selected",
        ],
    }
    (OUT / "scoreboard.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    lines = [
        "# S1 全候选筛查记分板",
        "",
        f"日期 {report['date']}。不选赢家。QK 普查身份=**ep35 路径 100 样本**，不是 ep34。",
        "",
        "| ID | 最高阶 | 状态 | 关键数字 | 下一步 |",
        "|---|---|---|---|---|",
    ]
    for c in cards:
        num = json.dumps(c["numbers"], ensure_ascii=False)
        if len(num) > 180:
            num = num[:180] + "…"
        lines.append(f"| {c['id']} | {c['highest_stage_reached']} | {c['s1_status']} | `{num}` | {c['next']} |")
    lines += [
        "",
        "## Attention census (ep35 QK, 100×12)",
        "",
        f"- tokens={attn['tokens']}",
        f"- K 双时间全零 {attn['k_zero_both_t_frac']:.4f}",
        f"- dirty(Q或K变化) {attn['dirty_or_frac']:.4f}",
        f"- 仍需打分叶 {attn['score_leaf_needed_frac']:.4f} → 理想跳过 {attn['ideal_skip_if_kzero_or_clean']:.4f}",
        f"- t1 上 Q且K 全零 {attn['both_qk_zero_t1_frac']:.4f}",
        f"- Q bit 密度 {attn['q_bit_density']:.5f}  K bit 密度 {attn['k_bit_density']:.5f}",
        f"- overlap/motion/same_zero 均值 {attn['overlap_mean']:.3f} / {attn['motion_xor_mean']:.3f} / {attn['same_zero_mean']:.3f}",
        "",
        "### per stage",
        json.dumps(attn["per_stage"], indent=2),
        "",
        "## 新精度门对旧 PAFT",
        "",
        "PAFT-ep4 running-BN 对照 AEE≈1.47 **已经高于 1.259**。要套 A3 必须在 **ep34** 上重训，不能引用 1.47 身份。",
        "",
        "## TSBG ep34 CPU premodel",
        json.dumps(tsbg["by_bundle"], indent=2),
    ]
    (OUT / "SCOREBOARD.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"n_cards": len(cards), "attn_tokens": attn["tokens"], "out": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
