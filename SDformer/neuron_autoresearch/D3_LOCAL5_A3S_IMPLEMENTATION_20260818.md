# D3 各向异性 stencil（h88/local5_a3s, A3S）实现说明

日期：2026-08-18。算法实现 agent 交付，对应
`CLAUDE_OPERATOR_CONTRACT_DRAFTS_20260818.md` 的 D3（Local5 线，P0-2）与
`CLAUDE_ALGORITHM_CONTRACT_QUEUE_20260818.md` 的 Local5 线验收。
实现完全保持 Swin 分窗 (2,15,15) 与全部模型参数不动（checkpoint 与 Local5
ep44 锚点直接兼容，valid825 对比口径不变，锚点 AEE 1.2819@ep44），纯算子消融。

红线遵守：未启动任何训练/评测/GPU；bsa_attention.py 为纯追加
（0 删除行，见 §6）；未改 H82/H86/H67/Local5/h87 任何既有路径
（含 D1 刚加的 h87/motion_t5_quotient，逐函数源码不变确认）。

---

## 1. 算子模式（纯追加）

新 mode：`local5_a3s` / 别名 `binary_axnor_local5_a3s_shiftmax` / `h88`
（bsa_attention.py 追加区内）。

A3S 门路径 = 现网 Local5 门路径 + 方向场偏移 ±Δ：

    （Local5 同式）scores = (same_spike + alpha0·same_silent) / head_dim
                 → Q7 量化 → shiftmax → 门量化 → attn
    （A3S 插入，在 Q7 量化之前）
        m = mean_t popcount(K_t ⊕ K_{t+1})              时域 XOR 梯度
        dirs = argmax_E,W,N,S |roll(m) − m|             方向场（2bit/pixel）
        offset[lane] = +Δ 若 dirs == lane 轴码，否则 −Δ（self lane 恒 0）
        scores += offset                                （Δ = 8 × 1/128 档）

关键恒等（K1，硬约束）：**Δ=0 档不触碰 scores**（`if delta_bins > 0` 才加
偏移），其余算术与 `_binary_alpha_xnor_stencil_attention`
（temporal_pair=False, spatial_cross=True, motion=0）逐式一致 ——
float 路径与 hardware Q7 路径均与现网 Local5 门逐位 `torch.equal`
（算子级 10 组随机平面 + forward 级逐位对比，见 §5）。这是可注入式训练的
锚点：起调档即现网恒等，loss 起点等于 Local5 续训起点，无塌陷路径。

方向场是固定位图（新存储对象，2bit/pixel，450bit/窗 <1% 存储增量），
`.detach()` 无梯度；Δ 是固定参数（对齐/正交双权重槽，无需训练）。
Δ 注入式渐增：`a3s_delta_warmup_steps > 0` 时读 `module._h9_global_step`，
从 0 线性升至 `a3s_delta_bins` 满档（short 配置 1224 步 ≈ 1 epoch，即
threshold 冻结步数；前 1 epoch Δ=0 恒等档 → 平滑注入）。

## 2. 新增函数（bsa_attention.py 追加区，4 个）

1. `_d3_axis_field(q_orig, k_orig)`：3×3 时域 XOR 梯度 argmax 方向场
   `[B, H, H, W]` 码（E=0/W=1/N=2/S=3），与 check_d3 的 `axis_field`
   逐式一致（uint8 位异或 popcount → 时间平均 → E/W/N/S 各向
   `|roll(m)−m|` → argmax）。K 布局 [B, H, T*N, D]（t-major token 序，
   与 `_binary_temporal_k_xor_popcount` 同式）。
2. `_d3_effective_delta_bins(cfg, profile_module)`：Δ 注入式渐增调度
   （warmup<=0 立即满档；warmup>0 时 step/round 线性斜坡）。
3. `_d3_a3s_offset(scores, dirs, delta_bins)`：对齐 lane +Δ / 正交 −Δ /
   self 0 的分数偏移 `[B, H, T*N, 5]`（lane 序 = self, N, S, W, E 与现网
   stencil 同序；两时间切片共享同一方向场位图，tile 展开）。
4. `_binary_axnor_local5_a3s_attention(q_orig, k_orig, cfg,
   profile_module=None)`：主算子。算术顺序逐式复刻现网 Local5（含
   matrix_diag_bias、无效候选掩码、rtl_shiftmax 双路径、preserve_mean、
   门量化）；唯一差异 = §1 的方向场偏移。返回
   `(attn, row_sum, gate, a3s_stats)`。

## 3. 挂载账本（forward 验证用，均 .detach()）

- `_h9_a3s_direction_field`：方向场位图 [B*, H, H, W]（2bit/pixel）。
- `_h9_a3s_delta_bins`：本步实际生效档（调度后；0 = 恒等档）。
- `_h9_a3s_axis_frac_ew`：E/W 轴方向场占比（K3 语义账，移动条 76-85%）。
- `_h9_a3s_winner_hit_rate`：对齐 lane 的量化分数 argmax 命中率
  （K4 诚实指标，仅运动承载像素；草案实测 91.2% vs 基线 0.0%）。

## 4. 实现期发现

1. 方向场的时间平均语义：check_d3 的 `axis_field` 对 (T−1) 个时域差平面做
   `mean`，T=2 时退化为单平面 —— 实现保留通用 `mean(dim=2)`，T=2 与
   参考实现逐位一致（测试交叉验证）。T=2 由合同窗钉死，但代码对 T>2 也
   是同一式（可扩展）。
2. Δ 作用于**归一化后**分数（/head_dim 后），与 check_d3 的 `a3s_gate`
   一致（K2 在归一化分数单位上验证：8 档 == 1/16 精确位移）。
3. 现网 Local5 分支"静默忽略 binary_motion_xor_alpha"的纪律对 A3S 同样
   成立：A3S 函数内 motion 恒 0，模板继承 motion alpha 时保持位稳定，
   不需要（也不允许）在 A3S 中加运动项。
4. 单测抓到 2 个测试侧问题（非算子 bug）：`make_planes` 默认 b=2 导致
   方向场 tile 对比的 batch 假设错误；gate 全元素均值 = 行和均值/5
   （5-lane，Local5 同款）而非相等（h87 是 1-lane 所以 D1 测试相等）。
   算子侧 0 bug。

## 5. 测试（CPU，unittest，全绿 28 例）

- `tests/test_local5_a3s_scores.py`（15 例）：算子级 K1-K5 ——
  K1 Δ=0 与现网 Local5 gate/attn/row_sum 逐位一致（10 组随机平面 × float
  路径 + hardware Q7 路径 + warmup step0 起调档）；K2 Δ=1/16 == 8 档与
  Q7 量化 commute（200 组 clamp 外）+ 偏移符号语义（对齐 +Δ/正交 −Δ/
  self 0，两时间切片共享位图）；K3 移动条 E/W 轴占比 ≥50% + 与 check_d3
  参考 `axis_field` 逐位交叉；K4 对齐 lane winner 命中率 Δ=8 显著 > 基线
  （基线 <50%）；K5 ident-K 目的地全部分裂为 3 偏移类 {self 0, +Δ, −Δ}；
  另含 Δ 注入式渐增调度 4 例。
- `tests/test_local5_a3s_forward.py`（13 例）：注入式
  `_qk_shiftmax_gate_forward`（Identity 子模块 + 二值 x，窗口 (2,15,15)，
  B*=20 行）端到端 —— F1 形状/归一化、F2 **forward 级 K1**：
  local5_a3s(Δ=0) 与 h66_lr 现网 forward 的 x_out 与 attn 逐位 equal
  （同 seed 输入）、F3 A3S 账本挂载、F4 warmup 调度（step0=恒等档/
  半程 4 档/满档 8 档）、F5 回归（h66_lr/h87/h82/compat 均运行）、
  F6 STE 梯度非空（CPU）。
- 既有回归：`tests/test_motion_t5_quotient_*.py`（D1，29 例）全绿。

## 6. 回归与 diff 纪律

diff vs 原始文件（/tmp/bsa_attention_pristine_20260818_d3.py）：
**0 删除行 / 322 追加行**。config 纯追加字段：`a3s_delta_bins`（默认 0）、
`a3s_delta_warmup_steps`（默认 0），含 config_from_dict 对应解析。
h82/h86/h67/Local5/h87 既有函数源码逐字节不变（diff 无 `<` 行佐证）。

## 7. 配置（configs/generated/，manifest 已记录 sha256）

- `dsec_fullres_w15_H88_local5_a3s_ft5_short_20260818.yml`：short 验证。
  n_epochs=5、bs 2、**seed 0**（红线）、force_save [4]；mode local5_a3s、
  a3s_delta_bins 8、a3s_delta_warmup_steps 1224、binary_motion_xor_alpha
  0.0；窗口 [2,15,15]；alpha0 0.015625 与 preserve_mean false 沿用 Local5
  模板；续训起点 = **Local5 ep44 锚点**（--prev_runid + --finetune 1；
  ep39 为 equal+20 原 resume 源，保留为 launcher 备选，默认冻结 ep44）。
- `dsec_fullres_w15_H88_local5_a3s_ft40.yml`：fullres 模板。n_epochs=40、
  force_save [34,39]。
- `dsec_fullres_w15_H88_local5_a3s_manifest.json`：模板/锚点/两配置 sha256。

## 8. 启动步骤（GPU 空闲后，由队列协调者执行）

    python3 entrypoints/run_local5_a3s_short_20260818.py

launcher 纪律（仿 run_motion_t5_quotient_short_20260818.py）：fcntl 锁
/tmp/sdformer_h88_local5_a3s.lock + status.log + SHA 冻结合同（算子 py /
配置 / 本说明 / 锚点 checkpoint）+ train.py subprocess。前置检查：锚点
ep44 存在、配置存在、**GPU 空闲**（nvidia-smi memory.used < 4096MiB 且
无已知训练/评测进程；忙时记录并退出 0，不自动排队）。通过标准（合同验证
实验 1）：loss 曲线与 Local5 基线同量级（不塌、不减半退化），step-1k 后
单调下降；随后 fullres ft40 对比 valid825 锚点 Local5 1.2819@ep44
（AEE ≤ 1.2819·1.01 或下降）。
