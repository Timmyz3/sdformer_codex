#!/usr/bin/env python3
"""D2 (h89/motion_sw12_overlap) 训练配置生成器：short 验证 + fullres 模板。

从现网 Motion 族配置模板（dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40.yml，
窗口 (2,15,15)、bs2、seed 0）派生两套配置，全部差异集中在 bsa_attention
块与运行期参数；模型/数据/优化器结构不动（纯算子消融口径）：

  ft5_short  dsec_fullres_w15_H89_motion_sw12_overlap_ft5_short_20260819.yml
             短验证：n_epochs=5，续训起点 = Motion 锚点 checkpoint_epoch35.pth
             （AEE 1.3297@ep35，--prev_runid 由 launcher 传入）
  ft40       dsec_fullres_w15_H89_motion_sw12_overlap_ft40.yml
             fullres 模板：n_epochs=40，force_save_epochs [34,39]（对齐锚点 ep35）

D2 契约关键字段：
  mode: h89 / motion_sw12_overlap（stride-12/窗口-15 重叠滑窗 + 滚动分母）
  binary_motion_xor_alpha: 0.0（运动项由算子内规范 16·m̄ 承担，不双重计数）
  sw12_window_size: 15 / sw12_stride: 12（合同钉死）
  sw12_num_steps: 10（= SNN num_steps，10 % 2 == 0 -> 5 个两切片窗行）
  sw12_batch: 2（训练 bs2 的 batch 维分解偏好；评测 bs1 自动回退）

对比口径（写入 NOTE，launcher 与合同验证用）：
  1. Motion 锚点 1.3297@ep35 基于 stride=15 稠密非重叠分窗，与 D2 重叠滑窗
     的 token 覆盖/窗口数（520 -> 825）不再同口径，AEE 数值不可直接比较；
  2. 合同验证以 h89 内部退化为准：stride=15（mult 全 1、窗口数不增）作为
     稠密非重叠基线，pass 条件 = AEE(stride12) ≤ AEE(stride15)·1.02；
  3. 滚动分母逐位精确（J1）与门守恒（J3）为算子级硬约束，已在 CPU 单测
     （tests/test_motion_sw12_overlap_*.py）逐位验证，训练不改算子。
  4. lr 5e-5（D1 教训：contract change 使用更保守学习率，不沿用模板 1e-4）。

输出 manifest（dsec_fullres_w15_H89_motion_sw12_overlap_manifest.json）记录
两配置与模板、锚点的 sha256。本脚本只写 configs/generated/，不训练不评测。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]
GENERATED = EXP / "configs/generated"
TEMPLATE = GENERATED / "dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40.yml"
ANCHOR = (
    EXP
    / "results/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_20260805/checkpoint_epoch35.pth"
)
SHORT_NAME = "dsec_fullres_w15_H89_motion_sw12_overlap_ft5_short_20260819"
FT40_NAME = "dsec_fullres_w15_H89_motion_sw12_overlap_ft40"
MANIFEST = GENERATED / "dsec_fullres_w15_H89_motion_sw12_overlap_manifest.json"

NOTE = (
    "D2 cross-window semantics (h89): stride-12/窗口-15 overlap sliding window "
    "with rolling denominator (Z_{i+1} = Z_i - leave + enter, 16-bit chunk "
    "int64, bitwise-equal to full recompute, J1). Window partitioning happens "
    "inside the operator; Swin window (2,15,15) and all model params unchanged "
    "(pure-operator ablation). 36% tokens have overlap multiplicity 2; 90-token "
    "shared bands per window edge; window count 520 -> 825 (+58.7%); per-window "
    "incremental exp 450 -> 270. Motion-XOR alpha must stay 0 (canonical 16*m̄ "
    "already inside the operator). COMPARISON NOTE: the Motion anchor "
    "(AEE 1.3297@ep35) used dense stride-15 partitioning and is NOT directly "
    "comparable to overlap-window AEE; contract baseline = h89 stride-15 "
    "degradation (dense non-overlap), pass if AEE(stride12) <= 1.02 * "
    "AEE(stride15); J1 bitwise exactness and J3 gate conservation are verified "
    "in CPU unit tests. lr 5e-5 (conservative for contract change, per D1 "
    "lesson)."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive(name: str, n_epochs: int, force_save_epochs: list[int]) -> Path:
    cfg = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    cfg["experiment"] = name
    # D2 算子块：纯追加字段 + mode 切换 + 运动不双重计数
    cfg["bsa_attention"]["mode"] = "h89"
    cfg["bsa_attention"]["binary_motion_xor_alpha"] = 0.0
    cfg["bsa_attention"]["sw12_window_size"] = 15
    cfg["bsa_attention"]["sw12_stride"] = 12
    cfg["bsa_attention"]["sw12_num_steps"] = 10
    cfg["bsa_attention"]["sw12_batch"] = 2
    # 运行期：short/fullres 预算 + 存点
    cfg["loader"]["n_epochs"] = n_epochs
    cfg["runtime"]["force_save_epochs"] = force_save_epochs
    cfg["runtime"]["state_save_epochs"] = [force_save_epochs[-1]]
    cfg["runtime"]["full_resolution_protocol"] = (
        "paper_480x640_window2x15x15_lr_rescue"
    )
    cfg["runtime"]["seed"] = 0
    # 学习率：D1 教训 —— contract change 用 5e-5，不沿用模板 1e-4
    cfg["optimizer"]["lr"] = 5.0e-05
    cfg["optimizer"]["param_groups"]["backbone_lr"] = 5.0e-05
    cfg["optimizer"]["param_groups"]["neuron_lr"] = 5.0e-05
    # 清掉 H67 equal+10 续训专用字段（D2 起点 = ep35 锚点，launcher 用
    # --prev_runid + --finetune 1 装载；不再叠加 equal+10 延长预算）
    for key in (
        "resume_protocol",
        "convergence_extension",
        "convergence_extension_lr",
        "resume_rng_scope",
        "resume_source_budget",
        "resume_source_checkpoint_label",
    ):
        cfg["runtime"].pop(key, None)
    cfg["note"] = NOTE
    out = GENERATED / f"{name}.yml"
    out.write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return out


def main() -> int:
    if not TEMPLATE.is_file():
        raise FileNotFoundError(f"template missing: {TEMPLATE}")
    if not ANCHOR.is_file():
        raise FileNotFoundError(f"anchor checkpoint missing: {ANCHOR}")
    short = derive(SHORT_NAME, n_epochs=5, force_save_epochs=[4])
    ft40 = derive(FT40_NAME, n_epochs=40, force_save_epochs=[34, 39])
    manifest = {
        "schema": "d2_motion_sw12_overlap_configs_v1",
        "generated_utc": None,  # launcher 冻结时补
        "template": {
            "path": str(TEMPLATE),
            "sha256": sha256(TEMPLATE),
        },
        "anchor": {
            "checkpoint": str(ANCHOR),
            "label": "Motion AEE 1.3297@ep35 (NOT directly comparable: "
            "stride-15 dense windowing vs overlap windows)",
            "sha256": sha256(ANCHOR),
        },
        "configs": {
            short.stem: {"path": str(short), "sha256": sha256(short)},
            ft40.stem: {"path": str(ft40), "sha256": sha256(ft40)},
        },
    }
    MANIFEST.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {short}")
    print(f"wrote {ft40}")
    print(f"wrote {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
