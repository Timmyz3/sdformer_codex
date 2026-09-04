#!/usr/bin/env python3
"""审计H67/H68的ATLIF安装覆盖、实际调用覆盖和无carrier分支。"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
sys.path[:0] = [
    str(EXP / "overlay"),
    str(REPO / "third_party/SDformerFlow"),
    str(REPO),
]

import torch  # noqa: E402
from models.STSwinNet_SNN.bsa_attention import register_shiftmax_pickle_compat  # noqa: E402


CHECKPOINT = EXP / (
    "results/h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30_bs8_full30_"
    "20260711_setsid/checkpoint_epoch19.pth"
)
PROFILES = {
    "H67": EXP / "results/h67_ep19_true_ttb_profile100_20260712/atlif_activity.csv",
    "H68": EXP / "results/h68_ep19_true_ttb_profile100_20260713/atlif_activity.csv",
}
BSA_SOURCE = EXP / "overlay/models/STSwinNet_SNN/bsa_attention.py"
SWIN_SOURCE = REPO / "third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py"


def called_modules(path: Path) -> set[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["name"] for row in csv.DictReader(handle)}


def main() -> int:
    register_shiftmax_pickle_compat()
    model = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    installed = {
        name
        for name, module in model.named_modules()
        if module.__class__.__name__ == "ATLIFTernaryPSN"
    }
    temporal_steps = {
        name: int(module.T)
        for name, module in model.named_modules()
        if module.__class__.__name__ == "ATLIFTernaryPSN"
    }
    del model

    source = BSA_SOURCE.read_text(encoding="utf-8")
    start = source.index('elif cfg.mode in {"h60", "tx_sc_k_mag_no_carrier_shiftmax"}:')
    end = source.index("\n    elif cfg.mode in {", start + 1)
    h60_branch = source[start:end]
    branch_checks = {
        "H60分支不调用sn2_q": "sn2_q" not in h60_branch,
        "H60分支采用gate乘K": "attn = k_orig.mul(gate)" in h60_branch,
        "attn_sn结果不进入projection": "attn = self.attn_sn(x)\n    x = self.proj(x)" in source,
    }
    swin_source = SWIN_SOURCE.read_text(encoding="utf-8")
    branch_checks["正常Swin路径只消费attention第一返回值"] = (
        "attn_windows, attn_score = self.attn(x_windows, mask=attn_mask)" in swin_source
        and "if return_attention:\n            return attn_score" in swin_source
    )

    cases = []
    for name, profile in PROFILES.items():
        called = called_modules(profile)
        uncalled = sorted(installed - called)
        unexpected = sorted(called - installed)
        dead_called = sorted(path for path in called if path.endswith(".attn.attn_sn.spiking_neuron"))
        live_called = called - set(dead_called)
        cases.append(
            {
                "设计": name,
                "安装模块": len(installed),
                "实际调用模块": len(called),
                "未调用模块": uncalled,
                "调用但未安装": unexpected,
                "动态调用但部署结果死亡": dead_called,
                "固定部署功能活跃模块": len(live_called),
                "功能活跃T2": sum(temporal_steps[path] == 2 for path in live_called),
                "功能活跃T10": sum(temporal_steps[path] == 10 for path in live_called),
                "未调用均为sn2_q": all(
                    path.endswith(".attn.sn2_q.spiking_neuron") for path in uncalled
                ),
            }
        )

    passed = (
        len(installed) == 105
        and all(item["实际调用模块"] == 93 for item in cases)
        and all(len(item["未调用模块"]) == 12 for item in cases)
        and all(len(item["动态调用但部署结果死亡"]) == 12 for item in cases)
        and all(item["固定部署功能活跃模块"] == 81 for item in cases)
        and all(item["功能活跃T2"] == 36 and item["功能活跃T10"] == 45 for item in cases)
        and all(item["未调用均为sn2_q"] for item in cases)
        and all(not item["调用但未安装"] for item in cases)
        and all(branch_checks.values())
    )
    result = {
        "状态": "通过" if passed else "失败",
        "checkpoint": str(CHECKPOINT),
        "分支静态检查": branch_checks,
        "结果": cases,
    }
    output_json = ROOT / "results/h67_h68_atlif_module_coverage.json"
    output_md = output_json.with_suffix(".md")
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    missing = cases[0]["未调用模块"] if cases else []
    lines = [
        "# H67/H68 ATLIF模块覆盖审计",
        "",
        f"- 状态：**{result['状态']}**。",
        "- 安装覆盖：105个one-sided binary ATLIF wrapper。",
        "- H67实际forward覆盖：93个；H68实际forward覆盖：93个。",
        "- 每个attention block的 `attn_sn` 虽被调用，但其结果不进入正常projection/Swin输出，共12个部署死结果。",
        "- 固定H67/H68正常推理的功能活跃ATLIF为81个：T=2共36个，T=10共45个。",
        "- 两条线未调用的12个模块集合完全一致，全部是每个attention block的 `sn2_q` 原carrier神经元。",
        "- H60部署分支静态检查确认不调用 `sn2_q`，输出为 `gate*K`。",
        "",
        "## 未调用模块",
        "",
    ]
    lines.extend(f"- `{name}`" for name in missing)
    lines.extend(
        [
            "",
            "## 硬件口径",
            "",
            "固定H67/H68部署核只需要调度81个功能活跃ATLIF逻辑点。软件forward仍会动态调用93点，其中12个 `attn_sn` 只产生未被正常推理路径消费的第二返回值；固定部署可以做死代码删除。不能把未执行的12个carrier wrapper或结果死亡的12个 `attn_sn` 计入固定核运行周期和动态功耗。若芯片要求兼容原QKFormer或return_attention调试路径，才保留相应配置。",
            "",
            "105是软件转换覆盖，93是PyTorch动态调用，81是固定正常推理功能活跃口径。三者必须在论文中分列，不能混写。",
        ]
    )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_md)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
