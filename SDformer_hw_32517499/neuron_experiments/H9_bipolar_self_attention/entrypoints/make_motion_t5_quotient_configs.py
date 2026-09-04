#!/usr/bin/env python3
"""D1 (h87/motion_t5_quotient) 训练配置生成器：short 验证 + fullres 模板。

从现网 Motion 族配置模板（dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40.yml，
窗口 (2,15,15)、bs2、seed 0）派生两套配置，全部差异集中在 bsa_attention
块与运行期参数；模型/数据/优化器结构不动（纯算子消融口径）：

  ft5_short  dsec_fullres_w15_H87_motion_t5_quotient_ft5_short_20260818.yml
             短验证：n_epochs=5，续训起点 = Motion 锚点 checkpoint_epoch35.pth
             （AEE 1.3297@ep35，--prev_runid 由 launcher 传入）
  ft40       dsec_fullres_w15_H87_motion_t5_quotient_ft40.yml
             fullres 模板：n_epochs=40，force_save_epochs [34,39]（对齐锚点 ep35）

h87 契约关键字段：
  mode: h87 / motion_t5_quotient（Motion T>2 时间商，规范融合式）
  binary_motion_xor_alpha: 0.0（运动项由算子内规范 16·m̄ 承担，不双重计数）
  temporal_quotient_steps: 10（= SNN num_steps，10 % 5 == 0 -> 2 组五元组）
  temporal_quotient_len: 5（合同钉死五元组长度）
  temporal_quotient_batch: 2（训练 bs2 的 batch 维分解偏好；评测 bs1 自动回退）

输出 manifest（dsec_fullres_w15_H87_motion_t5_quotient_manifest.json）记录
两配置与模板的 sha256。本脚本只写 configs/generated/，不训练不评测。
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
SHORT_NAME = "dsec_fullres_w15_H87_motion_t5_quotient_ft5_short_20260818"
FT40_NAME = "dsec_fullres_w15_H87_motion_t5_quotient_ft40"
MANIFEST = GENERATED / "dsec_fullres_w15_H87_motion_t5_quotient_manifest.json"

NOTE = (
    "D1 Motion T>2 time quotient (h87): per-slot canonical fused score "
    "s_t = min(RNE16(64*o_t + sz_t + 16*m̄_t), 162) over T=5 quintuple windows "
    "(num_steps=10 -> 2 groups), run-length broadcast gates (eq=0.979 -> "
    "1.084/5 independent gates, -78.3%). Window (2,15,15) and all model params "
    "unchanged vs Motion; pure-operator ablation resumed from Motion anchor "
    "checkpoint_epoch35.pth (AEE 1.3297@ep35). Motion-XOR alpha must stay 0 "
    "(canonical 16*m̄ already inside the operator)."
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
    # D1 算子块：纯追加字段 + mode 切换 + 运动不双重计数
    cfg["bsa_attention"]["mode"] = "h87"
    cfg["bsa_attention"]["binary_motion_xor_alpha"] = 0.0
    cfg["bsa_attention"]["temporal_quotient_steps"] = 10
    cfg["bsa_attention"]["temporal_quotient_len"] = 5
    cfg["bsa_attention"]["temporal_quotient_batch"] = 2
    # 运行期：short/fullres 预算 + 存点
    cfg["loader"]["n_epochs"] = n_epochs
    cfg["runtime"]["force_save_epochs"] = force_save_epochs
    cfg["runtime"]["state_save_epochs"] = [force_save_epochs[-1]]
    cfg["runtime"]["full_resolution_protocol"] = (
        "paper_480x640_window2x15x15_lr_rescue"
    )
    cfg["runtime"]["seed"] = 0
    # 清掉 H67 equal+10 续训专用字段（D1 起点 = ep35 锚点，launcher 用
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
        "schema": "d1_motion_t5_quotient_configs_v1",
        "generated_utc": None,  # launcher 冻结时补
        "template": {
            "path": str(TEMPLATE),
            "sha256": sha256(TEMPLATE),
        },
        "anchor": {
            "checkpoint": str(ANCHOR),
            "label": "Motion AEE 1.3297@ep35",
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
