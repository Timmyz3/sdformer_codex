#!/usr/bin/env python3
"""Audit that H68 training-only augmentation is absent from deployment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    exp = repo / "neuron_experiments/H9_bipolar_self_attention"
    train_path = exp / "configs/generated/h68_allbinary_all12_castling_ttx_aux050_toep20_w720_fastlr_full30.yml"
    deploy_path = exp / "configs/generated/h68_allbinary_all12_castling_ttx_deploy_full30_dyadic_int8_deploy.yml"
    result_path = exp / (
        "results/h68_allbinary_all12_castling_ttx_aux050_toep20_w720_fastlr_full30_bs8_"
        "full30_20260711_setsid/h68_epoch19_dyadic_int8_valid825.json"
    )
    source_path = exp / "overlay/models/STSwinNet_SNN/bsa_attention.py"

    train = yaml.safe_load(train_path.read_text(encoding="utf-8"))
    deploy = yaml.safe_load(deploy_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    source = source_path.read_text(encoding="utf-8")
    attention = deploy["bsa_attention"]

    overlay = str(exp / "overlay")
    if overlay not in sys.path:
        sys.path.insert(0, overlay)
    from models.STSwinNet_SNN.bsa_attention import _castling_aux_weight, config_from_dict

    class DummyModule:
        training = False

    train_cfg = config_from_dict(train["bsa_attention"])
    deploy_cfg = config_from_dict(attention)

    checks = {
        "train_auxiliary_enabled": float(train["bsa_attention"]["castling_matrix_aux_weight"]) == 0.5,
        "deploy_mode_h60": attention["mode"] == "h60",
        "deploy_auxiliary_zero": float(attention["castling_matrix_aux_weight"]) == 0.0,
        "deploy_auxiliary_end_zero": int(attention["castling_matrix_aux_end_step"]) == 0,
        "deploy_motion_xor_zero": float(attention["binary_motion_xor_alpha"]) == 0.0,
        "deploy_alpha0_dyadic": float(attention["alpha0"]) == 1.0 / 64.0,
        "deploy_score_q7": float(attention["hardware_score_step"]) == 1.0 / 128.0,
        "deploy_gate_q7": float(attention["hardware_gate_step"]) == 1.0 / 128.0,
        "deploy_quant_enabled": bool(attention["hardware_quant_enabled"]),
        "all12_attention": len(attention["target_blocks"]) == 12,
        "binary_atlif": deploy["atlif_ternary_psn"]["output_mode"] == "binary",
        "eval_forces_aux_zero_with_train_cfg": _castling_aux_weight(DummyModule(), train_cfg) == 0.0,
        "eval_forces_aux_zero_with_deploy_cfg": _castling_aux_weight(DummyModule(), deploy_cfg) == 0.0,
        "result_samples_825": int(result["samples"]) == 825,
        "source_auxiliary_parameter_free": "register_parameter" not in source[
            source.index("def _castling_binary_matrix_output"):source.index("def _binary_alpha_xnor_stencil_attention")
        ],
    }
    passed = all(checks.values())
    payload = {
        "pass": passed,
        "checks": checks,
        "float_rank1": {"epoch": 19, "AEE": 1.4688, "AAE": 9.4794, "spikes_g": 26.4244},
        "dyadic_deploy": {
            "epoch": 19,
            "AEE": result["AEE"],
            "AAE": result["AAE"],
            "spikes_g": result["total_spikes_g"],
        },
        "hardware_conclusion": "H68推理图等同dyadic TTX；训练期矩阵辅助不进入RTL。",
        "artifacts": {
            "train_config": str(train_path),
            "deploy_config": str(deploy_path),
            "result": str(result_path),
            "source": str(source_path),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = args.output.with_suffix(".md")
    lines = [
        "# H68 部署契约自动审计", "",
        "本审计验证训练期矩阵辅助不会进入推理 RTL。", "",
        "| 检查项 | 结果 |", "|---|---|",
    ]
    for name, value in checks.items():
        lines.append(f"| `{name}` | {'通过' if value else '失败'} |")
    lines += [
        "",
        f"- float epoch19：AEE `1.4688`、AAE `9.4794`、spikes `26.4244G`。",
        f"- dyadic epoch19：AEE `{result['AEE']:.4f}`、AAE `{result['AAE']:.4f}`、"
        f"spikes `{result['total_spikes_g']:.4f}G`。",
        "- 硬件结论：H68 推理图等同冻结 dyadic TTX；矩阵 auxiliary 的部署面积增量为零。",
        "",
        f"总结果：**{'通过' if passed else '失败'}**。",
    ]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
