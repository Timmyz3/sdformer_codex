"""Legacy H66d Local-5 dyadic + hardware-order numeric valid825 pipeline.

Runs against rank-1 float checkpoint (epoch 29 by default / ranking).
Does not modify training configs or old H67 assets.

The filename and historical output filenames retain ``rtl_exact`` for
compatibility. This script evaluates the attention-core Python numeric model;
it is not a full-network or full-resolution SystemVerilog replay.
"""

from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path

import yaml

from run_h60_family_deploy_eval import best_epoch, parse_profile, run_eval


REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
GEN = EXP / "configs/generated"
RESULTS = EXP / "results"
HW_RESULTS = REPO / "hw_autoresearch_nts07/results"
REDESIGN = REPO / "neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md"
HW_DOC = REPO / "hw_autoresearch_nts07/docs/76_H66d_Local5主线定点与RTL签核.md"

SOURCE_CFG = GEN / "h66d_allbinary_all12_lr_ttx_w720_fastlr_full30.yml"
RUN_DIR = RESULTS / "h66d_allbinary_all12_lr_ttx_w720_fastlr_full30_bs8_full30_20260712_setsid"
FLOAT_RANKING = RUN_DIR / "profile_ranking_valid825.md"


def make_dyadic_config(source: Path) -> Path:
    config = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    deploy = deepcopy(config)
    deploy["experiment"] = source.stem + "_dyadic_int8_deploy"
    deploy["bsa_attention"].update(
        {
            "alpha0": 1.0 / 64.0,
            "castling_matrix_aux_weight": 0.0,
            "castling_matrix_aux_end_step": 0,
            "binary_motion_xor_alpha": 0.0,
            "hardware_quant_enabled": True,
            "hardware_rtl_shiftmax_enabled": False,
            "hardware_mu_pow2_shift": 0,
            "hardware_score_step": 1.0 / 128.0,
            "hardware_score_min": -2.0,
            "hardware_score_max": 2.0,
            "hardware_gate_step": 1.0 / 128.0,
            "hardware_gate_min": 0.0,
            "hardware_gate_max": 2.0,
            "hardware_mask_invalid_candidates": True,
        }
    )
    deploy.setdefault("runtime", {})["deployment_contract"] = {
        "scope": "attention_core_numeric",
        "score_quantization": "Q7_step_2^-7",
        "shiftmax": "float_exp2",
        "gate_quantization": "Q1.7_RNE",
        "invalid_candidate_mask": True,
        "full_network_fixed_point": False,
        "systemverilog_replay": False,
    }
    deploy["note"] = (
        "H66d Local-5 frozen deploy: alpha0=1/64, INT8 Q7 score / Q1.7 gate, "
        "float 2^x Shiftmax (software dyadic). mode=binary_axnor_local5_shiftmax."
    )
    path = GEN / f"{deploy['experiment']}.yml"
    path.write_text(yaml.safe_dump(deploy, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def make_rtl_config(dyadic: Path) -> Path:
    config = yaml.safe_load(dyadic.read_text(encoding="utf-8")) or {}
    deploy = deepcopy(config)
    deploy["experiment"] = dyadic.stem + "_rtl_exact"
    attention = deploy["bsa_attention"]
    attention["hardware_rtl_shiftmax_enabled"] = True
    attention["hardware_quant_enabled"] = True
    attention["hardware_mask_invalid_candidates"] = True
    deploy.setdefault("runtime", {})["deployment_contract"] = {
        "scope": "attention_core_hardware_order_numeric",
        "score_quantization": "Q7_step_2^-7",
        "shiftmax": "Q8_LUT_integer_rowsum_ceil_pow2",
        "gate_quantization": "Q1.7_RNE",
        "invalid_candidate_mask": True,
        "full_network_fixed_point": False,
        "systemverilog_replay": False,
    }
    deploy["note"] = (
        "H66d Local-5 attention-core hardware-order numeric deploy: Q7 score, "
        "16-entry Q8 exp2 LUT Shiftmax, "
        "integer row sum, ceil-pow2 normalize, Q1.7 RNE gate, [0,2] sat. "
        "Stencil: self+4-axial with true invalid-candidate exclusion. This is "
        "not a full-network or full-resolution SystemVerilog replay."
    )
    path = GEN / f"{deploy['experiment']}.yml"
    path.write_text(yaml.safe_dump(deploy, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def write_case_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_redesign(epoch: int, dyadic: dict, rtl: dict) -> None:
    marker = "H66D_LOCAL5_DEPLOY_RTL_VALID825_20260725"
    text = REDESIGN.read_text(encoding="utf-8") if REDESIGN.exists() else ""
    if marker in text:
        return
    with REDESIGN.open("a", encoding="utf-8") as handle:
        handle.write("\n\n### 43.36 H66d Local-5 dyadic + RTL-exact valid825（2026-07-25）\n\n")
        handle.write(f"<!-- {marker} -->\n\n")
        handle.write(f"- checkpoint: epoch{epoch}\n")
        handle.write(
            f"- dyadic: AEE {dyadic['AEE']:.4f} / AAE {dyadic['AAE']:.4f} / "
            f"spikes {dyadic['total_spikes_g']:.4f}G / energy {dyadic['spike_energy_proxy_uj']:.2f}uJ\n"
        )
        handle.write(
            f"- RTL-exact: AEE {rtl['AEE']:.4f} / AAE {rtl['AAE']:.4f} / "
            f"spikes {rtl['total_spikes_g']:.4f}G / energy {rtl['spike_energy_proxy_uj']:.2f}uJ\n"
        )
        handle.write(
            f"- vs dyadic: AEE {rtl['AEE'] - dyadic['AEE']:+.4f}, "
            f"AAE {rtl['AAE'] - dyadic['AAE']:+.4f}\n"
        )
        handle.write(
            "- gate: AEE degradation vs dyadic ≤ 0.02 for deploy freeze; "
            "software precision mainline remains H66d only if float+dyadic beat H67 dyadic 1.4626.\n"
        )


def write_hw_doc(epoch: int, dyadic: dict, rtl: dict) -> None:
    lines = [
        "# H66d Local-5 主线定点与 RTL 签核",
        "",
        f"**日期**：2026-07-25  ",
        f"**checkpoint**：epoch{epoch}  ",
        f"**mode**：`binary_axnor_local5_shiftmax`",
        "",
        "## 1. 部署图",
        "",
        "```text",
        "binary ATLIF -> Local-5 stencil (self+N/S/E/W)",
        "  alpha-XNOR score (alpha0=1/64)",
        "  Q7 quant -> Shiftmax5 (float 2^x or RTL LUT)",
        "  Q1.7 gate -> sum_j gate_j * K_j",
        "```",
        "",
        "无效边界候选 mask 到 score_min（定点）或 -1e4（训练浮点）。",
        "",
        "## 2. valid825 结果",
        "",
        "| 路径 | AEE | AAE | spikes(G) | energy_proxy(uJ) |",
        "|---|---:|---:|---:|---:|",
        f"| float rank-1 (ep{epoch}) | 1.4432 | 9.4012 | 27.0403 | 23976.31 |",
        f"| dyadic INT8 | {dyadic['AEE']:.4f} | {dyadic['AAE']:.4f} | {dyadic['total_spikes_g']:.4f} | {dyadic['spike_energy_proxy_uj']:.2f} |",
        f"| RTL-exact Shiftmax | {rtl['AEE']:.4f} | {rtl['AAE']:.4f} | {rtl['total_spikes_g']:.4f} | {rtl['spike_energy_proxy_uj']:.2f} |",
        "",
        f"| RTL − dyadic AEE | {rtl['AEE'] - dyadic['AEE']:+.4f} |",
        "",
        "## 3. 与 H67 对照（同协议）",
        "",
        "| 方法 | float AEE | dyadic AEE | RTL AEE |",
        "|---|---:|---:|---:|",
        "| H67 Motion-XOR | 1.4671 | 1.4626 | 1.4627 |",
        f"| H66d Local-5 | 1.4432 | {dyadic['AEE']:.4f} | {rtl['AEE']:.4f} |",
        "",
        "## 4. 主线判定",
        "",
        "- **软件精度主线**：float 已是 H66d；若 dyadic 仍优于 H67 dyadic 1.4626，则部署精度主线切 H66d。",
        "- **硬件主线**：需 Stencil-5 row engine RTL（见 `rtl_local5/`），不可复用 H67 Motion-XOR top 冒充。",
        "- spike energy 仍为 proxy，不含 halo/gather/Shiftmax5 控制。",
        "",
        "## 5. 产物路径",
        "",
        f"- dyadic json：`{RUN_DIR / f'h66d_epoch{epoch}_dyadic_int8_valid825.json'}`",
        f"- RTL json：`{HW_RESULTS / 'h66d_local5_rtl_exact_valid825.json'}`",
        "",
    ]
    HW_DOC.parent.mkdir(parents=True, exist_ok=True)
    HW_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    print(
        "[scope] Legacy rtl_exact filenames mean attention-core hardware-order "
        "numeric evaluation, not full-network/SystemVerilog RTL-exact.",
        flush=True,
    )
    if not SOURCE_CFG.exists():
        raise FileNotFoundError(SOURCE_CFG)
    if not FLOAT_RANKING.exists():
        raise FileNotFoundError(FLOAT_RANKING)
    epoch = best_epoch(FLOAT_RANKING)
    ckpt = RUN_DIR / f"checkpoint_epoch{epoch}.pth"
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    dyadic_cfg = make_dyadic_config(SOURCE_CFG)
    rtl_cfg = make_rtl_config(dyadic_cfg)

    dyadic_out = RUN_DIR / f"standard_dyadic_int8_valid825/epoch{epoch}"
    rtl_out = RUN_DIR / f"rtl_exact_valid825/epoch{epoch}"

    t0 = time.time()
    print(f"[H66d] dyadic INT8 valid825 epoch{epoch}", flush=True)
    run_eval(dyadic_cfg, ckpt, dyadic_out)
    dyadic = parse_profile(dyadic_out / "spike_profile.json")
    dyadic_json = RUN_DIR / f"h66d_epoch{epoch}_dyadic_int8_valid825.json"
    write_case_json(
        dyadic_json,
        {
            "candidate": "H66d Local-5 TTX",
            "epoch": epoch,
            "config": str(dyadic_cfg),
            "checkpoint": str(ckpt),
            "profile": str(dyadic_out / "spike_profile.json"),
            **dyadic,
            "seconds": time.time() - t0,
            "energy_scope": "spike_activity_proxy_only_excludes_overlay_attention_control_reduction_and_memory",
        },
    )
    print(json.dumps(dyadic, indent=2), flush=True)

    t1 = time.time()
    print(f"[H66d] RTL-exact valid825 epoch{epoch}", flush=True)
    run_eval(rtl_cfg, ckpt, rtl_out)
    rtl = parse_profile(rtl_out / "spike_profile.json")
    HW_RESULTS.mkdir(parents=True, exist_ok=True)
    write_case_json(
        HW_RESULTS / "h66d_local5_rtl_exact_valid825.json",
        {
            "状态": "完成",
            "结果": [
                {
                    "名称": "H66d Local-5",
                    "epoch": epoch,
                    "配置": str(rtl_cfg),
                    "checkpoint": str(ckpt),
                    "profile": str(rtl_out / "spike_profile.json"),
                    "耗时秒": time.time() - t1,
                    "RTL逐位结果": rtl,
                    "原浮点Shiftmax量化结果": {
                        "AEE": dyadic["AEE"],
                        "AAE": dyadic["AAE"],
                        "total_spikes_g": dyadic["total_spikes_g"],
                        "spike_energy_proxy_uj": dyadic["spike_energy_proxy_uj"],
                    },
                    "AEE变化": rtl["AEE"] - dyadic["AEE"],
                    "AAE变化": rtl["AAE"] - dyadic["AAE"],
                    "spikes变化G": rtl["total_spikes_g"] - dyadic["total_spikes_g"],
                }
            ],
        },
    )
    md = [
        "# H66d Local-5 RTL-exact valid825",
        "",
        "| 候选 | RTL AEE | vs dyadic AEE | RTL AAE | vs dyadic AAE | spikes(G) |",
        "|---|---:|---:|---:|---:|---:|",
        f"| H66d Local-5 | {rtl['AEE']:.4f} | {rtl['AEE'] - dyadic['AEE']:+.4f} | "
        f"{rtl['AAE']:.4f} | {rtl['AAE'] - dyadic['AAE']:+.4f} | {rtl['total_spikes_g']:.4f} |",
        "",
        "判定：AEE 退化 ≤ 0.02 可冻结当前 LUT/量化网格进入 RTL/DC 前设计。",
        "",
    ]
    (HW_RESULTS / "h66d_local5_rtl_exact_valid825.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(rtl, indent=2), flush=True)

    write_hw_doc(epoch, dyadic, rtl)
    append_redesign(epoch, dyadic, rtl)
    print("DONE", HW_DOC, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
