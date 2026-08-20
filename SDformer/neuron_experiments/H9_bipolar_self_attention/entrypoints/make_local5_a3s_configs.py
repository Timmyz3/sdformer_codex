#!/usr/bin/env python3
"""D3 (h88/local5_a3s) 训练配置生成器：short 验证 + fullres 模板。

从现网 Local5 配置模板（dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50.yml，
窗口 (2,15,15)、bs2、seed 0、alpha0 0.015625）派生两套配置，全部差异集中在
bsa_attention 块与运行期参数；模型/数据/优化器结构不动（纯算子消融口径）：

  ft5_short  dsec_fullres_w15_H88_local5_a3s_ft5_short_20260818.yml
             短验证：n_epochs=5，续训起点 = Local5 锚点 checkpoint_epoch44.pth
             （AEE 1.2819@ep44，--prev_runid 由 launcher 传入）
  ft40       dsec_fullres_w15_H88_local5_a3s_ft40.yml
             fullres 模板：n_epochs=40，force_save_epochs [34,39]

h88 契约关键字段（合同草案 D3）：
  mode: local5_a3s（别名 binary_axnor_local5_a3s_shiftmax / h88）
  a3s_delta_bins: 8（Δ = 1/16 = 8 个 1/128 网格档；0 = Δ=0 恒等锚点）
  a3s_delta_warmup_steps: 1224（Δ 注入式渐增：约 1 个 epoch（threshold 冻结
     步数）内从 0 线性升至满档；起调 Δ=0 档与现网 Local5 逐位等价，K1 锚点）
  binary_motion_xor_alpha: 0.0（与现网 Local5 同纪律，静默忽略，保持位稳定）

续训起点决策：采用 **ep44 锚点**（valid825 AEE 1.2819@ep44，与 D1 用 Motion
ep35 锚点同纪律——续训起点 = 锚点 checkpoint）。ep39 是 equal+20 阶段的原
resume 源，保留为 launcher 备选（--prev_runid 可替换），默认冻结 ep44。

输出 manifest（dsec_fullres_w15_H88_local5_a3s_manifest.json）记录两配置与
模板、锚点的 sha256。本脚本只写 configs/generated/，不训练不评测。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]
GENERATED = EXP / "configs/generated"
TEMPLATE = GENERATED / "dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50.yml"
ANCHOR = (
    EXP
    / "results/dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50_20260812/checkpoint_epoch44.pth"
)
SHORT_NAME = "dsec_fullres_w15_H88_local5_a3s_ft5_short_20260818"
FT40_NAME = "dsec_fullres_w15_H88_local5_a3s_ft40"
MANIFEST = GENERATED / "dsec_fullres_w15_H88_local5_a3s_manifest.json"

NOTE = (
    "D3 axis-aligned anisotropic stencil (h88/local5_a3s): Local5 5-lane "
    "stencil scores plus direction-field offset ±Δ (Δ=1/16 = 8 Q7 1/128 grid "
    "bins; aligned lane +Δ, orthogonal −Δ, self 0). Direction field = 3x3 "
    "temporal XOR-gradient argmax (2bit/pixel, fixed bitmap, no gradient); "
    "ident-K unique gate splits from 1 group to 3 offset classes. Δ=0 is "
    "bit-identical to the deployed Local5 gate (K1 anchor); Δ ramps from 0 "
    "over a3s_delta_warmup_steps (injectable training). Window (2,15,15) and "
    "all model params unchanged vs Local5; pure-operator ablation resumed "
    "from Local5 anchor checkpoint_epoch44.pth (valid825 AEE 1.2819@ep44)."
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
    # D3 算子块：mode 切换 + A3S 字段（Δ 满档 8 档 = 1/16，注入式渐增 1 epoch）
    cfg["bsa_attention"]["mode"] = "local5_a3s"
    cfg["bsa_attention"]["a3s_delta_bins"] = 8
    cfg["bsa_attention"]["a3s_delta_warmup_steps"] = 1224
    cfg["bsa_attention"]["binary_motion_xor_alpha"] = 0.0
    # 运行期：short/fullres 预算 + 存点 + seed 0（红线）
    cfg["loader"]["n_epochs"] = n_epochs
    cfg["runtime"]["force_save_epochs"] = force_save_epochs
    cfg["runtime"]["state_save_epochs"] = [force_save_epochs[-1]]
    cfg["runtime"]["full_resolution_protocol"] = (
        "paper_480x640_window2x15x15_local5_a3s_ft40"
    )
    cfg["runtime"]["seed"] = 0
    # 清掉 Local5 equal+20 续训专用字段（D3 起点 = ep44 锚点，launcher 用
    # --prev_runid + --finetune 1 装载；不再叠加 equal+20 延长预算）
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
        "schema": "d3_local5_a3s_configs_v1",
        "generated_utc": None,  # launcher 冻结时补
        "template": {
            "path": str(TEMPLATE),
            "sha256": sha256(TEMPLATE),
        },
        "anchor": {
            "checkpoint": str(ANCHOR),
            "label": "Local5 valid825 AEE 1.2819@ep44",
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
