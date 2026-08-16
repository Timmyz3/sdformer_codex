"""Legacy H67/H68 attention-core hardware-order numeric valid825 evaluation.

The filename and historical result paths retain ``rtl_exact`` for compatibility.
The evaluator runs a Python integer/LUT attention path; it does not establish
full-network SystemVerilog equivalence.
"""

from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path

import yaml

from run_h60_family_deploy_eval import parse_profile, run_eval


REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
GEN = EXP / "configs/generated"
RESULTS = EXP / "results"
HW_RESULTS = REPO / "hw_autoresearch_nts07/results"


CASES = (
    {
        "名称": "H67 Motion-XOR TTX",
        "配置": GEN / "h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30_dyadic_int8_deploy.yml",
        "运行目录": RESULTS / "h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30_bs8_full30_20260711_setsid",
        "原部署结果": RESULTS / "h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30_bs8_full30_20260711_setsid/h67_epoch19_dyadic_int8_valid825.json",
    },
    {
        "名称": "H68 Castling训练、TTX部署",
        "配置": GEN / "h68_allbinary_all12_castling_ttx_deploy_full30_dyadic_int8_deploy.yml",
        "运行目录": RESULTS / "h68_allbinary_all12_castling_ttx_aux050_toep20_w720_fastlr_full30_bs8_full30_20260711_setsid",
        "原部署结果": RESULTS / "h68_allbinary_all12_castling_ttx_aux050_toep20_w720_fastlr_full30_bs8_full30_20260711_setsid/h68_epoch19_dyadic_int8_valid825.json",
    },
)


def make_rtl_config(source: Path) -> Path:
    config = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    deploy = deepcopy(config)
    deploy["experiment"] = source.stem + "_rtl_exact"
    attention = deploy["bsa_attention"]
    attention["hardware_rtl_shiftmax_enabled"] = True
    attention["hardware_quant_enabled"] = True
    attention["hardware_score_step"] = 1.0 / 128.0
    attention["hardware_score_min"] = -2.0
    attention["hardware_score_max"] = 2.0
    attention["hardware_gate_step"] = 1.0 / 128.0
    attention["hardware_gate_min"] = 0.0
    attention["hardware_gate_max"] = 2.0
    deploy.setdefault("runtime", {})["deployment_contract"] = {
        "scope": "attention_core_hardware_order_numeric",
        "score_quantization": "Q7_step_2^-7",
        "shiftmax": "Q8_LUT_integer_rowsum_ceil_pow2",
        "gate_quantization": "Q1.7_RNE",
        "full_network_fixed_point": False,
        "systemverilog_replay": False,
    }
    deploy["note"] = (
        "Attention-core hardware-order numeric验证：score先量化到Q7；使用16项"
        "Q8 exp2 LUT、整数行和、上取整二次幂归一化和Q1.7最近偶数舍入；"
        "Gate饱和到[0,2]。不是全网SystemVerilog RTL-exact。"
    )
    path = GEN / f"{deploy['experiment']}.yml"
    path.write_text(yaml.safe_dump(deploy, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def old_metrics(path: Path) -> dict[str, float]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        "AEE": float(raw["AEE"]),
        "AAE": float(raw["AAE"]),
        "total_spikes_g": float(raw["total_spikes_g"]),
        "spike_energy_proxy_uj": float(raw["spike_energy_proxy_uj"]),
    }


def main() -> int:
    print(
        "[scope] Legacy rtl_exact filenames mean attention-core hardware-order "
        "numeric evaluation, not full-network/SystemVerilog RTL-exact.",
        flush=True,
    )
    rows = []
    for case in CASES:
        config = make_rtl_config(case["配置"])
        checkpoint = case["运行目录"] / "checkpoint_epoch19.pth"
        output = case["运行目录"] / "rtl_exact_valid825/epoch19"
        start = time.time()
        run_eval(config, checkpoint, output)
        exact = parse_profile(output / "spike_profile.json")
        old = old_metrics(case["原部署结果"])
        rows.append(
            {
                "名称": case["名称"],
                "epoch": 19,
                "配置": str(config),
                "checkpoint": str(checkpoint),
                "profile": str(output / "spike_profile.json"),
                "耗时秒": time.time() - start,
                "RTL逐位结果": exact,
                "原浮点Shiftmax量化结果": old,
                "AEE变化": exact["AEE"] - old["AEE"],
                "AAE变化": exact["AAE"] - old["AAE"],
                "spikes变化G": exact["total_spikes_g"] - old["total_spikes_g"],
            }
        )

    HW_RESULTS.mkdir(parents=True, exist_ok=True)
    json_path = HW_RESULTS / "h67_h68_rtl_exact_valid825.json"
    md_path = HW_RESULTS / "h67_h68_rtl_exact_valid825.md"
    json_path.write_text(json.dumps({"状态": "完成", "结果": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# H67/H68 RTL逐位一致valid825",
        "",
        "## 评估口径",
        "",
        "本结果不再使用浮点 `2^x` Shiftmax。推理路径严格复现当前RTL：原始score的Q7量化、"
        "16项Q8 exp2 LUT、整数行和、上取整二次幂归一化、Q1.7最近偶数舍入和[0,2]饱和。",
        "",
        "| 候选 | AEE | 相对原部署AEE | AAE | 相对原部署AAE | spikes(G) | 相对原部署spikes(G) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        exact = row["RTL逐位结果"]
        lines.append(
            f"| {row['名称']} | {exact['AEE']:.4f} | {row['AEE变化']:+.4f} | "
            f"{exact['AAE']:.4f} | {row['AAE变化']:+.4f} | {exact['total_spikes_g']:.4f} | "
            f"{row['spikes变化G']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## 判定规则",
            "",
            "- 若RTL逐位模型相对原部署模型的AEE退化不超过0.02，可继续以当前LUT进入DC前冻结。",
            "- 若退化超过0.02，应先搜索LUT位宽/节点或进行硬件感知微调，不能用原浮点结果替代。",
            "- H68部署图必须保持无Castling矩阵分支；它只能作为训练期软硬件协同贡献。",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
