#!/usr/bin/env python3
"""Build the stage0-local5 experiment config (h12s0_local5_ternary_ep5.yml).

单变量设计（相对 h12cal_ternary_ws4_ep5）：仅 stage0 的 attention 从 h60 全窗
换成 binary_axnor_local5_shiftmax（5-lane stencil + Shiftmax over 5 候选），
stage1-3 保持 h60。其余（三元神经元、θ 标定、H12 修复 backward、warm start、
LR、epoch 数）与 h12cal 完全一致。

- 主安装段 bsa_attention 用 target_blocks 精确覆盖 stage1-3（stage0 不碰，
  避免双重安装）。
- bsa_attention_stage0 参数抄自 local5 参考配置
  dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50_hardware_order_q7q17_deploy.yml
  （ep44 AEE 1.2819 那条线的 bsa_attention 段），入口会强制 stage_selection=stage0。
"""

from __future__ import annotations

import yaml
from pathlib import Path

H9 = Path("/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention")
H12 = Path("/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H12_atlif_gradfix")

SRC_CFG = H9 / "configs/generated/t49_ternary_ws4_ep5.yml"
LOCAL5_REF = H9 / "configs/generated/dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50_hardware_order_q7q17_deploy.yml"
OUT_CFG = H12 / "configs/generated/h12s0_local5_ternary_ep5.yml"

# stage1-3 共 10 个 attention 模块（depths [2,2,6,2]）
STAGE123_BLOCKS = [
    "1:0", "1:1",
    "2:0", "2:1", "2:2", "2:3", "2:4", "2:5",
    "3:0", "3:1",
]


def main() -> None:
    cfg = yaml.safe_load(SRC_CFG.read_text(encoding="utf-8"))
    ref = yaml.safe_load(LOCAL5_REF.read_text(encoding="utf-8"))

    # 主段：stage1-3 保持 h60
    main_bsa = dict(cfg["bsa_attention"])
    main_bsa["target_blocks"] = list(STAGE123_BLOCKS)
    main_bsa.pop("stage_selection", None)
    cfg["bsa_attention"] = main_bsa

    # stage0 段：local5 stencil + Shiftmax（参数抄 local5 参考线）
    s0 = dict(ref["bsa_attention"])
    s0.pop("target_blocks", None)
    s0.pop("stage_selection", None)
    cfg["bsa_attention_stage0"] = s0

    OUT_CFG.parent.mkdir(parents=True, exist_ok=True)
    OUT_CFG.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"[h12s0] wrote {OUT_CFG}")
    print(f"[h12s0] main targets: {main_bsa['target_blocks']}")
    print(f"[h12s0] stage0 mode: {s0['mode']} alpha0={s0.get('alpha0')}")


if __name__ == "__main__":
    main()
