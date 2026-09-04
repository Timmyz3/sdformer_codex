# B2 Motion T=4+pad wildcard 时间商（h87b/motion_t4_pad_quotient）实现说明

日期：2026-08-19。算法实现 agent 交付，对应
`D1_VARIANT_SEARCH_20260819.md` §4.1 的 plan B 预案（B2：T=4+pad12）——
D1 合同（h87/motion_t5_quotient）训练失败即无等待切换的备选，本交付为
纯 CPU 实现（算子 + 配置 + CPU 单测 + launcher），不训练、不评测、不碰 GPU
（GPU 被 D1 训练占用，实测 memory.used = 57 GiB，launcher 启动即退出 0）。

红线遵守：bsa_attention.py 纯追加（0 删除行，见 §6）；未动 h87/h88/h89
及更早路径；未改 D1 配置/测试/launcher 任何既有文件；未删文件。

---

## 1. 实现选择 (a)：独立新 mode + 独立新函数

选择 **方案 a（`motion_t4_pad_quotient` / 别名 `h87b` 独立函数
`_binary_t4_pad_quotient_token_scores`）**，不扩展现有 h87 函数。理由：

1. **与既有单测零冲突**：D1 测试 `test_validation_errors` 明确断言
   `temporal_quotient_len=4` 在 h87 下必须抛 ValueError（h87 合同钉死 5）。
   方案 b（h87 加 wildcard 分支）必须改这段校验，直接破坏既有测试；
   方案 a 完全不动 h87 函数体。
2. **diff 风险最低**：h87 的 I1-I7 路径逐字节不动，B2 仅共享只读助手
   （`_rne16_div_pow2_ste` / `_d1_decompose_temporal_batch` /
   `_binary_event_ste` / `_qkformer_token_q`）。
3. **语义隔离**：B2 的 RLE 账口径与 D1 不同（见 §3），混入同一函数会
   产生口径歧义；独立函数在注释与返回字段上各自钉死。

## 2. 算子（bsa_attention.py 追加区，1 个函数 + 1 个 forward 分支）

`_binary_t4_pad_quotient_token_scores(q_orig, k_orig, cfg)`：

- 分组：num_steps=10 → 3 组 T=4：(0,1,2,3)、(4,5,6,7)、(8,9,pad,pad)，
  末组 2 个 pad 槽（pad = len − steps%len，自动派生）。校验：
  `temporal_quotient_len==4` 钉死；`steps%len==0` 抛 ValueError
  （整除归 h87）；steps 必须偶（T=2 窗）。
- **pad 槽 wildcard 掩码**：pad 槽不参与商组——不进 slot 融合式（无
  (o,sz,m̄) 统计）、不贡献 run-length 统计（不产生 eq 边、不产生 run
  断点，wildcard 合并；(pad,pad) 恒等）、广播时按掩码跳过。实现为
  组布局 `grouped [B*,H,3,4,N]` + `valid` 掩码（末组 2、3 位 False）；
  `eq_edge` 只在两端皆真实时计入；`pad_mask`（True=pad）随 slot_views
  挂载（forward 验证用）。
- **真实槽融合式与 D1 逐位一致**（I1）：`s_t = min(RNE16(64·o_t + sz_t +
  16·m̄_t), 162)`；组内首槽采用组内第 1 条边（I4，末组两槽共享边 (8,9)）；
  跨组边 (3,4)/(7,8) 不可见（I7：7/9 边覆盖，7 条真实边；
  (8,9) 仍是 within-pair 边 eq 0.9808）。
- 布局与 batch 分解与 D1 同族：行序 row=(b·n_pairs+wd)·n_sw+s、
  `_d1_decompose_temporal_batch`（batch 偏好 2，评测 bs1 自动回退）、
  token 布局写回 (t_local, n) ← 槽 2·wd(idx)+t_local。

## 3. 位账与 RLE 账口径（与 D1 的唯一口径差异，契约 §4.1）

D1 的 `mean_runs_per_position` 是每 (组,位置) 均值（ratio 分母 = len=5）；
B2 的 pad 槽无门数可言，故 `mean_runs_per_position` = **每位置 10 槽序列
总独立门数** = Σ_g (1 + Σ_{组内真实边}(1−eq)) = 3 + 7·(1−p̄)（第三组按
len-2 计，E[runs] = 1+(1−eq_8,9)；全组口径 −64.6% 不采用——pad 不得与
真实槽合并）。`independent_gate_ratio = mean/10`、
`broadcast_saving = 1 − mean/10`、`eq_edge_rate` 只数 7 条真实边。
逐边模型：p̄≈0.879 → E[门]≈3.85 → **−61.5%（合同 −61.4%）**。
rle_stats 另含 n_groups=3、group_lengths=(4,4,2)、pad_slots=2、
coverage_edges=7、batch_decomposition。

## 4. forward 挂载账（mode h87b 分支，均 .detach()）

`_h9_b2_rle_stats` / `_h9_b2_slot_scores` / `_h9_b2_slot_overlap` /
`_h9_b2_slot_remainder`（[B*,H,10,N] 仅真实槽）/ `_h9_b2_pad_mask`
（[B*,H,3,4,N]，True=pad）/ `_h9_b2_grouped_runs`（[B*,H,3,N]）。
运动不双重计数：`binary_motion_xor_alpha != 0` 抛 RuntimeError（同 D1）。

## 5. 测试（CPU，unittest，全绿）

- `tests/test_motion_t4_pad_quotient_scores.py`（18 例）：P1 pad 掩码恒等式
  （pad_mask 形状与内容、跨组边 (3,4)/(7,8) 不得影响任何真实槽分数、
  wildcard 广播账全同分 → 每位置 3 门 saving 70%（对照 D1 80%）、末组
  E[runs]=1+(1−eq_8,9)、总门恒等式 3+7(1−p̄)）；P2 真实槽与 D1 逐位一致
  （槽 0..3 与 9 随机数据逐位相等；m=0 退化路径全部 10 槽逐位相等；
  槽 8 按 B2 组首槽约定用边 (8,9) 而非 D1 的 (7,8)——预期差异）；P3
  I7 7/9 覆盖（XOR=2 只出现在 {0,1,2,4,5,6,8}）；P4 I1 融合式逐位 /
  I2 无 s%4==3 / I5 反解唯一（真实槽）；P5 布局写回、batch 分解、
  校验（len≠4 / steps%4==0 / steps 奇 / T≠2 / 分解失败 / STE 梯度）。
- `tests/test_motion_t4_pad_quotient_forward.py`（16 例）：注入式
  `_qk_shiftmax_gate_forward`（(2,15,15) 窗、B*=20、分解 (2,5,2)）——
  F1 形状 / F2 gate 归一化 / F3 RLE 账与 pad_mask 挂载 / F4 配置校验
  （alpha、steps=8、len=5 抛错）/ F5 回归（h87、h88 Δ=0 锚点档、
  h89 sw12、h60、h82、compat）/ F6 STE 梯度。
- 回归：既有 h87（scores 17 例 + forward 12 例）、h88（local5_a3s
  scores+forward）、h89（sw12 scores+forward）全部不破（§6 逐套重跑）。

## 6. diff 纪律与回归

diff vs 原始文件（/tmp/bsa_attention_pristine_20260818.py）：
**0 删除行**。追加：B2 算子函数 + 区块注释（约 200 行）+ forward 分支 1 处
（h87 与 h88 分支之间）。未改 h87/h88/h89 及更早路径、未改既有配置/测试/
launcher。既有测试逐套重跑全绿。

## 7. 配置（configs/generated/，manifest 已记录 sha256）

- 生成器 `entrypoints/make_motion_t4_pad_configs.py`：从 D1 short 配置
  （B1 调参版，lr 2.5e-5）派生，差异仅算子块——mode h87b、
  temporal_quotient_len 4（steps 10 / batch 2 不变）。继承：seed 0、
  bs2、ft5（n_epochs 5）、force_save [4]、lr 2.5e-5（backbone 同）、
  Motion ep35 锚点续训（--prev_runid + --finetune 1）、窗口 [2,15,15]。
- `dsec_fullres_w15_H87B_motion_t4_pad_quotient_ft5_short_20260819.yml`：
  short 验证（唯一配置；fullres ft40 待 short 通过后按同模板派生）。
- `dsec_fullres_w15_H87B_motion_t4_pad_quotient_manifest.json`：
  源配置（D1 short）/锚点/B2 配置 sha256。

## 8. 启动步骤（GPU 空闲后，由队列协调者执行）

    python3 entrypoints/run_motion_t4_pad_short_20260819.py

launcher（`run_motion_t4_pad_short_20260819.py`，仿 D1）纪律：fcntl 锁
/tmp/sdformer_h87b_motion_t4_pad.lock + status.log + SHA 冻结合同（算子 py /
配置 / 本说明）+ train.py subprocess（--prev_runid 锚点 + --finetune 1）。
前置检查：配置存在、锚点存在、**GPU 空闲**（memory.used < 4096MiB 且无
已知训练/评测进程，KNOWN_BUSY_PIDS 含 D1 launcher 名；忙时记录并退出 0，
不自动排队）。通过标准（B2 预案 §4.1）：loss 与 T=2 基线同量级
（不塌、不减半退化）、step-1k 后单调下降；位账 −61.4% 与 I 系列恒等式
在真实槽上由本套 CPU 单测冻结。B3 分层记账（−61.1% 确定性版）为位账
写作资产，无需训练（变体搜索文档 §4.3）。
