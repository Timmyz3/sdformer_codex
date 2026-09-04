# D1 Motion T>2 时间商（h87/motion_t5_quotient）实现说明

日期：2026-08-18。算法实现 agent 交付，对应
`CLAUDE_OPERATOR_CONTRACT_DRAFTS_20260818.md` 的 D1（P0-1）与
`CLAUDE_ALGORITHM_CONTRACT_QUEUE_20260818.md` 的 Local5 线验收。
实现选项采用草案第 (b) 项：**不动 Swin 分窗 (2,15,15)，T=5 分组在算子内
跨窗完成**——checkpoint 与 Motion ep35 锚点直接兼容，valid825 对比口径不变
（锚点 AEE 1.3297@ep35），纯算子消融。

红线遵守：未启动任何训练/评测/GPU；bsa_attention.py 为纯追加
（0 删除行，见 §6）；未改 H82/H86/H67/Local5 任何既有路径。

---

## 1. 算子模式（纯追加）

新 mode：`motion_t5_quotient` / 别名 `h87`（bsa_attention.py 追加区内）。

每时间槽 t 的分数冻结为规范融合式（I1 实锤：拆解式在 RNE 平局商奇偶翻转处
差 1 档，全域 2.74%，故唯一规范为融合式，硬件实现与部署必须同式）：

    q_t=popcount(Q_t), k_t=popcount(K_t), o_t=popcount(Q_t&K_t)
    sz_t = 32 - q_t - k_t + o_t                     （容斥界约束域内）
    m̄_t = popcount(K_{t-1} ⊕ K_t)                    （运动边，窗口内逐位置）
    s_t = min(RNE16(64·o_t + sz_t + 16·m̄_t), 162)   （Q7 网格 [0,162]）

时间结构：SNN num_steps=10 → n_pairs=5 个两切片窗行 → 2 组 T=5 五元组
（10 % 5 == 0，无 pad）。槽位 t 采用运动边 (t−1, t)；组内首槽复用组内第 1
条边（与 H67 pair 的"同一运动边喂两个槽位"一致，I4）。跨组边 (4,5) 不可见
（I7：8/9 边覆盖，合同值）。运动项嵌入规范融合式，故
`binary_motion_xor_alpha != 0` 直接抛 RuntimeError（运动不双重计数，F4）。

## 2. 新增函数（bsa_attention.py 追加区，3 个）

1. `_rne16_div_pow2_ste(numerator, denominator=16)`：RNE16，int64 逐位精确
   （分子 ≤ 2592 << 2^53），STE 直通梯度。round 后在 int64 上
   `floor-div` + `bitwise_and(1)` 判奇偶（float 位运算会报错，torch 2.4.1
   亦无 `torch.divmod`）。
2. `_d1_decompose_temporal_batch(batch_total, n_pairs, cfg)`：把 attention
   batch 维分解为 (B, n_sw)（B* = B × n_pairs × n_sw）。候选 n_sw ∈
   (1376, 352, 88, 24, 6, 2)（w15 全分辨率族各 stage 空间窗数），首个整除解
   采用；`temporal_quotient_batch` 是偏好而非覆盖——评测 bs1 时 batch 变化，
   自动回退（batch 分解 5 例在 LayoutAndBoundaryTests 验证）。
3. `_binary_t5_quotient_token_scores(q_orig, k_orig, cfg)`：跨窗时间槽分组
   （行序 row=(b·n_pairs+wd)·n_sw+s，wd-major，window_partition_v2 固定），
   每槽 (o_t, sz_t, m̄_t) → 融合式分数 [B*, H, num_steps, N]，再按 token
   布局 (t_local, n) ← 槽 2·wd(idx)+t_local 写回 [B*, H, 2N, 1]。

## 3. 挂载账本（forward 验证用，均 .detach()）

- `_h9_d1_rle_stats`：mean_runs_per_position / independent_gate_ratio
  （eq=0.979 时 1.084/5，−78.3%，I6）/ eq_edge_rate / batch_decomposition
  （short 配置下 (2,5,2)）。
- `_h9_d1_slot_scores` / `_h9_d1_slot_overlap` / `_h9_d1_slot_remainder`：
  [B*, H, 10, N]，remainder == scores % 4（I2 槽位分解 r ∈ {0,1,2}；
  I5 反解 (s−m̄)→(o,r) 在物理域内唯一，0.00% 退化）。

## 4. 实现期发现（算子级测试抓到 2 个真实 bug）

1. `q_event[tb % 2]` 取错维：q_event 布局 [B*, H, 2, N, D]，时间维是第 3
   维，应 `q_event[:, :, tb % 2]`（原写法 batch=2 时静默取错行、>2 时越界）。
2. `_rne16_div_pow2_ste` 的 floor 商未转 int64，`bitwise_and` 对 float 报错。
两处均在追加区内修复；其余失败均为测试侧几何/断言错误（详见测试文件注释）。

## 5. 测试（CPU，unittest，全绿）

- `tests/test_motion_t5_quotient_scores.py`（17 例）：算子级——I1 规范融合式
  逐位（含 slot-1 因 k_alt 的 (o,sz) 单独算）、I2 槽位分解（m=0 构造窗验证
  无 s%4==3 + I5 反解 r∈{0,1,2} 恒成立）、I3 网格位移不变、I4 T=5 均匀边剖面
  ≡ H67 T=2（m=2 配对翻转构造）、I5 商可逆（RQTB 记录→分数双向）、I6 RLE
  账、I7 边覆盖（跨组边不得入槽）、布局与 batch 分解、STE 梯度。
- `tests/test_motion_t5_quotient_forward.py`（12 例）：注入式
  `_qk_shiftmax_gate_forward`（Identity 子模块 + 二值 x，窗口 (2,15,15)，
  B*=20 行 = 2×5×2）端到端——F1 形状（x [20,450,32] / attn [2,20,15,15,32]）、
  F2 gate 归一化（shiftmax 行和 ∈ (0.5,1] 与 h60 锚点同一约定；preserve_mean
  ×n_tokens 后 gate_mean == row_sum_mean）、F3 RLE 账与槽位视图挂载、
  F4 运动 alpha 抛 RuntimeError / steps=8 抛 ValueError、F5 h60/h82/compat
  回归、F6 STE 反向梯度非空。

## 6. 回归与 diff 纪律

diff vs 原始文件（/tmp/bsa_attention_pristine_20260818.py）：
**0 删除行 / 252 追加行**。config 纯追加字段：`temporal_quotient_steps`
（默认 0）、`temporal_quotient_len`（合同钉死 5）、`temporal_quotient_batch`
（0=自动），含 config_from_dict 对应解析。

## 7. 配置（configs/generated/，manifest 已记录 sha256）

- `dsec_fullres_w15_H87_motion_t5_quotient_ft5_short_20260818.yml`：short
  验证。n_epochs=5、bs 2、seed 0、force_save [4]；mode h87、
  binary_motion_xor_alpha 0.0、temporal_quotient_steps/len/batch = 10/5/2；
  窗口 [2,15,15]；num_steps 10；续训起点 = Motion ep35 锚点
  （--prev_runid + --finetune 1）。
- `dsec_fullres_w15_H87_motion_t5_quotient_ft40.yml`：fullres 模板。
  n_epochs=40、force_save [34,39]。
- `dsec_fullres_w15_H87_motion_t5_quotient_manifest.json`：模板/锚点/两配置
  sha256。

## 8. 启动步骤（GPU 空闲后，由队列协调者执行）

    python3 entrypoints/run_motion_t5_quotient_short_20260818.py

launcher 纪律（仿 run_h82_class_major_ttx_ft15_20260817.py）：fcntl 锁
/tmp/sdformer_h87_motion_t5_quotient.lock + status.log + SHA 冻结合同
（算子 py / 配置 / 本说明）+ train.py subprocess。前置检查：锚点 ep35 存在、
配置存在、**GPU 空闲**（nvidia-smi memory.used < 4096MiB 且无已知训练/评测
进程；忙时记录并退出 0，不自动排队）。通过标准（合同验证实验 1）：loss 曲线
与 T=2 基线同量级（不塌、不减半退化），step-1k 后单调下降；随后 fullres ft40
对比 valid825 锚点 Motion 1.3297@ep35（AEE ≤ 1.3297·1.01 或下降）。
