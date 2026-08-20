#!/usr/bin/env python3
"""B2 (h87b/motion_t4_pad_quotient) 训练配置生成器：short 验证配置 + manifest。

从 D1 short 配置（dsec_fullres_w15_H87_motion_t5_quotient_ft5_short_20260818.yml，
B1 调参后的现网短验证配置：lr 2.5e-5、seed 0、bs2、force_save [4]、续训
Motion ep35 锚点）派生 B2 short 配置，全部差异集中在 bsa_attention 算子块：

  dsec_fullres_w15_H87B_motion_t4_pad_quotient_ft5_short_20260819.yml

B2 契约关键字段（D1_VARIANT_SEARCH_20260819.md §4.1，plan B 预案）：
  mode: h87b / motion_t4_pad_quotient（T=4+pad wildcard 时间商）
  binary_motion_xor_alpha: 0.0（运动项由算子内规范 16·m̄ 承担，不双重计数）
  temporal_quotient_steps: 10（= SNN num_steps；10 % 4 == 2 -> 2 个 pad 槽）
  temporal_quotient_len: 4（合同钉死四元组长度；steps%len!=0 时启用 pad 掩码）
  temporal_quotient_batch: 2（训练 bs2 的 batch 维分解偏好；评测 bs1 自动回退）

位账（逐边模型）：E[独立门]/位置 = 3 + 7·(1−p̄)（第三组按 len-2 计，
pad 槽 wildcard 不参与商组）-> −61.4%。训练超参与 B1 同源（lr 2.5e-5 冷重启
剂量），全部模型/数据/优化器结构与 D1 完全一致（纯算子消融口径）。

输出 manifest（dsec_fullres_w15_H87B_motion_t4_pad_quotient_manifest.json）
记录派生源配置与 B2 配置的 sha256。本脚本只写 configs/generated/，
不训练不评测。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]
GENERATED = EXP / "configs/generated"
SOURCE = GENERATED / "dsec_fullres_w15_H87_motion_t5_quotient_ft5_short_20260818.yml"
ANCHOR = (
    EXP
    / "results/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_20260805/checkpoint_epoch35.pth"
)
SHORT_NAME = "dsec_fullres_w15_H87B_motion_t4_pad_quotient_ft5_short_20260819"
MANIFEST = GENERATED / "dsec_fullres_w15_H87B_motion_t4_pad_quotient_manifest.json"

NOTE = (
    "B2 Motion T=4 + pad wildcard time quotient (h87b, plan B for D1): "
    "num_steps=10 -> 3 quadruple groups (0,1,2,3)/(4,5,6,7)/(8,9,pad,pad); "
    "pad slots do not participate in the quotient (wildcard mask: no run-length "
    "contribution, no fused score, skipped in broadcast; last group accounted "
    "as len-2). Real-slot fused form bitwise identical to D1 "
    "s_t = min(RNE16(64*o_t + sz_t + 16*m̄_t), 162); 7/9 edge coverage; "
    "bit-budget -61.4%. Window (2,15,15) and all model params unchanged vs "
    "Motion/D1; pure-operator ablation resumed from Motion anchor "
    "checkpoint_epoch35.pth (AEE 1.3297@ep35). lr 2.5e-5 cold restart "
    "(B1 dose, inherited from the D1 short config). Motion-XOR alpha must "
    "stay 0 (canonical 16*m̄ already inside the operator)."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive(name: str) -> Path:
    cfg = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    cfg["experiment"] = name
    # B2 算子块：mode 切换 + T=4 分组（余数 -> pad 槽 wildcard 掩码）
    cfg["bsa_attention"]["mode"] = "h87b"
    cfg["bsa_attention"]["binary_motion_xor_alpha"] = 0.0
    cfg["bsa_attention"]["temporal_quotient_steps"] = 10
    cfg["bsa_attention"]["temporal_quotient_len"] = 4
    cfg["bsa_attention"]["temporal_quotient_batch"] = 2
    # 运行期：short 预算（seed 0 / bs2 / force_save [4] 由 D1 源配置继承）
    cfg["runtime"]["force_save_epochs"] = [4]
    cfg["runtime"]["state_save_epochs"] = [4]
    cfg["runtime"]["seed"] = 0
    cfg["note"] = NOTE
    out = GENERATED / f"{name}.yml"
    out.write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return out


def main() -> int:
    if not SOURCE.is_file():
        raise FileNotFoundError(f"D1 short source config missing: {SOURCE}")
    if not ANCHOR.is_file():
        raise FileNotFoundError(f"anchor checkpoint missing: {ANCHOR}")
    short = derive(SHORT_NAME)
    manifest = {
        "schema": "b2_motion_t4_pad_configs_v1",
        "generated_utc": None,  # launcher 冻结时补
        "source": {
            "path": str(SOURCE),
            "label": "D1 short ft5 (B1-tuned, lr 2.5e-5)",
            "sha256": sha256(SOURCE),
        },
        "anchor": {
            "checkpoint": str(ANCHOR),
            "label": "Motion AEE 1.3297@ep35",
            "sha256": sha256(ANCHOR),
        },
        "configs": {
            short.stem: {"path": str(short), "sha256": sha256(short)},
        },
    }
    MANIFEST.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {short}")
    print(f"wrote {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
