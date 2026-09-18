#!/usr/bin/env python3
"""T53: h60 注意力「极性旋钮」零样本矩阵（在 sd5ai 上跑，纯评测不训练）。

动机（T50/T51 查清的账）：h60 走的 `_ternary_alpha_xnor_token_scores`
（bsa_attention.py:1736）**本来就是按有符号三值写的**：
    q_event = _ternary_sign_ste(...)      # 硬 {-1,0,+1}
    k_event = _ternary_sign_ste(...)
    opposite = (q_event == -k_event) & q_active & k_active
    score = same_nonzero + alpha0*same_zero - mismatch_penalty*opposite
                            - single_active_penalty*single_active
但两件事同时把极性打掉了：
  1. 神经元输出 `official_atlif` ∈ {0,+θ} ⇒ `opposite` **恒为空集**（死代码）；
  2. 即便不是空集，源配置 `mismatch_penalty: 0.0` ⇒ 系数也是 0（双重关闭）。
另外 `binary_motion_xor_alpha: 0.125` 走的 `_binary_temporal_k_xor_popcount`
用 `_binary_event_ste = x.gt(0)`（bsa_attention.py:3143），**把负事件压成 0**。

⇒ 此前所有三值 arm（含 §4.2 的 LR 扫描四臂）都跑在
   `binary_motion_xor_alpha=0.125` + `mismatch_penalty=0.0` 上，
   **极性机制是关着的** ⇒ 那些 arm 只量到了三值的代价，没量到它的收益。

本脚本用**零样本**（权重复用 ep34，不训练）一次问清两件事：
  A. 二值下把 `mismatch_penalty` 打开 → 应当逐位不变（`opposite` 在二值下是空集）
     ⇒ 这是「`opposite` 确实是死代码」的直接证明。
  B. 三值下打开 `mismatch_penalty`，随 S 扫描 → 极性份额 ∝ 1/S，同时 SOPS 代价 ∝ 1/S
     ⇒ 极性与能耗**共用同一个旋钮**，必须按 Pareto 曲线记账，没有免费的极性。

**单变量纪律**：只翻 `mismatch_penalty`（`-β·opposite` 的系数）。
`binary_motion_xor_alpha` 保持 0.125 不动 —— 该 motion 项已在本次修复里由
`_binary_event_ste`(x.gt(0)) 换成 `_ternary_sign_ste`（bsa_attention.py:1807），
它**现在自己就区分极性**，且 0.125 是调过的先验；把它归零会引入第二个变量。

输出：每个 arm 一份 valid825（`metrics.AEE` / `metrics.DSEC_Fl` / `global_firing_rate`
/ `total_spikes` / `energy_uj`），落在 results/t53_polarity/<arm>/standard_valid825/epoch34/。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path("/root/private_data/work/sdformer_codex/SDformer")
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
SRC_CFG = (REPO / "hw_autoresearch_nts07/system_handoff/incoming/m2041_ep34_quant_binding_inputs/"
           "dsec_c12_alpha0125_ep29_resume5_20260830.yml")
SRC_RUN = EXP / "results/dsec_c12_alpha0125_ep29_resume5_20260830"
OUT_ROOT = EXP / "results/t53_polarity"
GEN_CFGS = EXP / "configs/generated"
EPOCH = 34
PY = "/opt/conda/envs/sdformerflow/bin/python"

# 源配置的极性相关值（原样，用于 reference arm）
SRC_MOTION_ALPHA = 0.125
SRC_MISMATCH_PENALTY = 0.0
# 唯一的自变量：`mismatch_penalty` 是 `-β·opposite` 的系数，源配置 0.0 ⇒
# 即便 opposite 非空也被乘成 0（"双重关闭"的第二重）。
# **刻意不动 `binary_motion_xor_alpha`**：该 motion 项已在本轮修复里从
# `_binary_event_ste`(x.gt(0)) 换成 `_ternary_sign_ste`，它现在自己就区分极性
# （|Δ| ∈ {0,1,2} = 同极性/一侧静默/反极性），且 0.125 是调过的先验。
# 把它一并归零会引入第二个变量，破坏「只翻 polarity 系数」的归因。
POL_MISMATCH_PENALTY = 1.0

# arm 名 → (binary?, negative_threshold_scale, mismatch_penalty, 说明)
#
# ⚠️ 预期值的修正（重要，2026-09-17 推 T53 前）：
# motion 项的修复（`_binary_event_ste`(x.gt(0)) → `_ternary_sign_ste`，:1807）
# **在二值下是前向逐位中性的**：二值 ATLIF 输出 ∈ {0,+θ} 严格非负 ⇒
# `sign(x) ≡ gt(0)(x)`（x>0→1=1, x=0→0=0；没有 x<0）⇒ bin_* 两臂仍应
# 精确复现锚点 1.19951 / 5.31336。
# **但三值下 k_orig 可以取负（−θ 或 −S·θ）** ⇒ 同一个修复会改变 motion 项
# ⇒ 历史零样本数（S=8: 1.42507 / S=4: 8.263 / S=2: 17.16，均带 gt(0) motion 项）
# **不再可复现**。这不是 bug，是自变量多了一个；下面 `ter8_ref` 的预期已相应改成
# 「复现旧值 or 偏离，偏离量 = motion 项修复在三值下的单独贡献」。
ARMS: dict[str, tuple[bool, float | None, float, str]] = {
    "bin_ref":      (True,  None, SRC_MISMATCH_PENALTY,
                     "二值锚点复现（应 == AEE 1.19951 / DSEC_Fl 5.31336；motion 修复在二值下逐位中性）"),
    "bin_mp_on":    (True,  None, POL_MISMATCH_PENALTY,
                     "A：二值 + mismatch_penalty 开 ⇒ 应仍 == 锚点（证明 opposite 在二值下是空集）"),
    "ter8_ref":     (False, 8.0,  SRC_MISMATCH_PENALTY,
                     "三值 S=8 + mp=0：与旧零样本 1.42507 的差 = motion 修复在三值下的单独贡献"),
    "ter8_mp_on":   (False, 8.0,  POL_MISMATCH_PENALTY,
                     "B：三值 S=8（FR≈锚点）+ opposite 项上电；与 ter8_ref 的差 = opposite 的单独贡献"),
    "ter4_mp_on":   (False, 4.0,  POL_MISMATCH_PENALTY,
                     "B：三值 S=4 + opposite 项上电（负脉冲占 ~17%）"),
    "ter2_mp_on":   (False, 2.0,  POL_MISMATCH_PENALTY,
                     "B：三值 S=2 + opposite 项上电（极性份额最大，能耗代价也最大）"),
}


def build_config(binary: bool, scale: float | None, mismatch_penalty: float, arm: str) -> dict:
    cfg = yaml.safe_load(SRC_CFG.read_text(encoding="utf-8"))

    a = cfg["atlif_ternary_psn"]
    if binary:
        # 源配置本来就是 binary + official_atlif，一个键都不动
        if scale is not None:
            raise ValueError("binary arm 不能带 negative_threshold_scale")
    else:
        for d in (a, *(a.get("target_groups") or ())):
            d["output_mode"] = "ternary"
            d["threshold_mode"] = "asymmetric_scale"
        a["negative_threshold_scale"] = float(scale)

    b = cfg["bsa_attention"]
    b["binary_motion_xor_alpha"] = float(SRC_MOTION_ALPHA)   # 全程不动（单变量）
    b["mismatch_penalty"] = float(mismatch_penalty)

    cfg["experiment"] = "t53_" + arm
    cfg["note"] = ("T53 h60 polarity zero-shot matrix; %s; motion_alpha=%g (unchanged) "
                   "mismatch_penalty=%g; no training"
                   % ("binary" if binary else "ternary S=%g" % scale,
                      SRC_MOTION_ALPHA, mismatch_penalty))
    return cfg


def main() -> int:
    wanted = sys.argv[1:] or list(ARMS)
    unknown = [w for w in wanted if w not in ARMS]
    if unknown:
        print("unknown arms: %s\nknown: %s" % (unknown, list(ARMS)), file=sys.stderr)
        return 2

    GEN_CFGS.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "SDFORMER_USE_MLFLOW": "0", "SDFORMER_MLFLOW_MODEL_LOGGING": "0",
           "SDFORMER_SNN_BACKEND": "cupy", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

    for arm in wanted:
        binary, scale, mp, why = ARMS[arm]
        out_cfg = GEN_CFGS / ("t53_%s.yml" % arm)
        out_cfg.write_text(yaml.safe_dump(build_config(binary, scale, mp, arm), sort_keys=False),
                           encoding="utf-8")

        run_dir = OUT_ROOT / arm
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt = run_dir / ("checkpoint_epoch%d.pth" % EPOCH)
        if not ckpt.exists():
            shutil.copy2(SRC_RUN / ("checkpoint_epoch%d.pth" % EPOCH), ckpt)

        out_dir = run_dir / "standard_valid825" / ("epoch%d" % EPOCH)
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [PY, "-u", "third_party/SDformerFlow/eval_DSEC_flow_SNN.py",
               "--config", str(out_cfg), "--checkpoint", str(ckpt),
               "--path_results", str(out_dir), "--mode", "valid"]
        print("[T53] ==== %s :: %s" % (arm, why), flush=True)
        print("[T53] launching %s" % arm, flush=True)
        with open(out_dir / "eval.log", "w", encoding="utf-8") as log:
            rc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
        print("[T53] %s rc=%d" % (arm, rc), flush=True)

    print("[T53] ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
