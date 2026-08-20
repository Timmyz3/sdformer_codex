# DATE 主线（Motion / Local5）算法实验缺口清单

日期：2026-08-18。执行 agent：Claude（算法侧）。GPU：A800 80GB 空闲（H86 已停）。
用户决策锚点：CLAUDE_LINE_DECISION_20260818.md——算法锁定 Motion（H67）与 Local5，
DATE 主线二者选一；禁止 H82/H86 训练/评测；不改 overlay bsa_attention.py；seed 0；
评测走 eval_DSEC_flow_SNN.py --mode valid（batch=1）。

## 0. 审计基线（2026-08-18）

| 线 | valid825 AEE | spikes G | MVSEC day2-scratch | 同 ckpt RTL |
|---|---:|---:|---|---|
| Motion/H67 ep35 | 1.3297 | 82.11 | 四序列全过门（唯一） | 有（ep35 同 ckpt，hw 侧） |
| Local5 ep44 | 1.2819 | 85.24 | IF1 不过门 | 有（ep44 重绑，2026-08-15 PASS） |

磁盘 overlay `bsa_attention.py` SHA=`66d0a339…`（H83–H86 叠代版，未改动）。
H67/Local5 的 h60/Local5 数值路径与 8-10/8-15 产收据时同一磁盘版本（H82 分析已证
h83–h86 为纯增量），评测前在 provenance 里记录 SHA。

已核实收口（无需再动）：
- 四线账本 `DATE_FOUR_LINE_LEDGER_20260817.json`（4 线 rank-1 + 预算行，profile SHA 绑定）
- Table G `DSEC_DENSITY_QUARTILE_TABLE_G_20260817.json`（status=`PASS_AEE_ATTACHED`）
- 图 `figures/date_four_line_20260817/`
- Local5 ep44 hardware-order 数值（deploy_valid825/hardware_order_q7q17/epoch44，
  2026-08-15 14:12，AEE 1.2804，config SHA 078bb517 + ckpt SHA 19820bec）已存在

## 1. 缺口清单

### P0-1 H67 fullres ep35 hardware-order / dyadic 数值 valid825（DATE 主线锚点）【待执行】

- 做什么：`dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_{hardware_order,dyadic}_q7q17_deploy.yml`
  × `checkpoint_epoch{35,40}.pth`，输出 `deploy_valid825/{hardware_order_q7q17,dyadic_q7q17}/epoch{35,40}/`
  （ep35 是主线锚点，ep40 补同线曲线）。
- 为什么 DATE 需要：DATE 主线锚点 ep35 缺硬件序数值评估（只有 float standard_valid825）；
  Local5 ep44 已有同协议配套，Motion 若当主线不能只有"RTL 有、算法数值链缺"。硬件侧
  "同 ckpt 数值链"是 docs/439 完整度口径（不加创新分，但缺了会被审稿人点名）。
- 参考范式：Local5 ep44 deploy_valid825 q7q17 先例（2026-08-15 重绑后重评）。
- GPU 时长：4 × ~17 min ≈ 70 min。
- 优先级：**P0**。

### P0-2 MVSEC 四序列门核实与 Local5/H81 差距落档 【核实完毕，写档】

- 做什么：用四个 `mvsec_cicc_*_full` / `fixed800` 的 mvsec_summary.json 逐序列核对
  预注册门"四测试序列 AEE 均 < NB0 对应序列"，量化失败项。
- 核实结果（full-sequence AEE；fixed800 同序）：
  - NB0：OD1 0.8450 / IF1 1.5998 / IF2 2.7536 / IF3 2.1106（参考）
  - H67：0.8201 / 1.5868 / 2.6258 / 2.0357 → **四序列全过（唯一）**
  - Local5：0.8414 / **1.6282** / 2.6679 / 2.0669 → **IF1 不过（+1.76%）**
  - H81：0.8205 / **1.6248** / 2.6670 / 2.0581 → **IF1 不过（+1.56%）**
  - fixed800 口径一致：H67 唯一全过；Local5 IF1 1.6259 vs NB0 1.5977（+1.77%）。
- 补救结论：Local5 不重训（scratch 重训 ≥2 天/次且无通过保证，用户已锁定两条线不新开）；
  该门如实保留为 Motion 主线支柱之一，Local5 以"等预算 DSEC 精度 + Q1 密度分层"赢面表述。
- GPU 时长：0。
- 优先级：**P0**。

### P0-3 Motion 100 样本 raw Q/K bit trace 导出（ep35，硬件侧"多样本真实 INT8 全通道"的算法侧一半）【待执行】

- 做什么：`profile_nts11_hardware_p0.py --config <ep35 hardware_order deploy> --checkpoint ep35
  --samples 100 --ordered-trace --bit-trace-dir <dir> --bit-trace-samples 100
  --bit-trace-windows 1 --bit-trace-all-blocks`。工具链 SHA 与 8-15 multisample10 完全一致
  （profiler 5f21c8d7 / trace_writer 75c91340，已核对）。
- 为什么 DATE 需要：hw docs/399 §6.1 明列"Motion 多样本真实 RTL：100-sample profile 没有原始
  Q/K bit trace；算法侧释放 GPU 后应导出分层 raw trace，再跑 Fixed2S/RQTB2S 真 RTL"。
  现只有 10 样本（PASS_SEALED_COMPONENT_RTL），100 样本可把验证从 10/10 推到 100 样本区间。
- 边界：只导出 trace + aggregate profile，不写 RTL（RTL 回放归硬件侧）；产出证据分档 [模型]+[prof]。
- GPU 时长：预估 2–4 h（10 样本 ≈ 96 MB NPZ；100 样本 ≈ 0.9–1 GB）。>2 h，启动前写状态文件。
- 优先级：**P0**（唯一长任务，排在两个短评测之后）。

### P1-4 full-encoder Amdahl 的算法侧输入：per-operator 全网活动占比（双线）【待执行】

- 做什么：对 H67 ep35 与 Local5 ep44 各跑一次 `profile_nts11_hardware_p0.py --samples 100
  --ordered-trace`（无 bit trace），产出 per-operator/per-stage/per-block 活动表
  （operator_runtime / h60_by_block / h60_by_stage / activation_records），并显式把
  attention 之外的开销（ATLIF、FFN、residual、decoder、DMA/存储）按活动占比列出，
  供硬件侧计算整网 Amdahl 的优化占比 f。
- 为什么 DATE 需要：docs/439"full-encoder Amdahl（现有模型 Motion 1.1997×、Local5 tile
  1.1165×，不是整网）"——算法侧可补的是 f 的真实测量输入；docs/399 §6 仍缺工作 #3。
- 参考范式：现有 `h67_fullres_ep30_t450_profile100_20260805`（ep30 crop）同结构搬到 ep35/ep44。
- GPU 时长：2 × ~30–60 min。
- 优先级：**P1**。

### P1-5 12-block 同窗评估表（双线）【随 P1-4 产物后处理】

- 做什么：从 P1-4 的 h60_by_block.csv 生成 12-block 同窗表（12 个 block 同一 window [2,15,15]
  下的 spikes/活动/量化统计），H67 ep35 与 Local5 ep44 各一张；Local5 无既有同结构产物。
- 为什么 DATE 需要：docs/439 完整度缺口之一（不加创新分）。
- GPU 时长：0（后处理）。
- 优先级：**P1**。

### P2-6 官方 DSEC test 提交 writer（mode=test）【已实现，待 smoke/全量/注册】

- 做什么：`entrypoints/eval_DSEC_test_submission.py`（独立新文件，不改
  eval_DSEC_flow_SNN.py/overlay）：test split loader（416 samples）+ 与 valid deploy
  相同的模型/overlay/BN 路径 + 推理，输出 DSEC 提交格式 PNG-FI
  （v=round(flow*128+2^15)，3ch uint16，`{seq}/{file_index:06d}.png`）。
- 为什么 DATE 需要：AAE 差距（本地 ~5.5–5.9 三聚合 vs 官方 4.871）只有官方提交可闭合；
  NB0_AAE_GAP_CLOSURE_20260812 已冻结"本地聚合不能闭合"结论。
- 前置：**2026-08-18 已核实官方 test 数据完全本地化**——test_split_seq.csv（416 样本 /
  7 序列）、event_tensors/10bins/left/<seq>/（416 个全齐，与 split 逐项匹配）、
  test_forward_optical_flow_timestamps/<seq>.csv（官方 file_index）均在
  data/Datasets/DSEC/saved_flow_data/，无需官网注册下载。
- 剩余：smoke（3–5 样本，随 P1-4 后 GPU 空档跑）；全量推理 ~17–20 min GPU；官方注册提交。
- 优先级：**P2**。

### P2-7 Local5 ep44 多 output tile / cross-head 数值扩展支持 【待硬件侧需求】

- 做什么：若硬件侧推进 docs/399 §6.2（QS+rolling 前端接 HxH cross-head shell），算法侧补
  对应 population 数值锚。
- GPU 时长：视需求，单次 ~30–60 min。
- 优先级：**P2**。

## 2. 状态跟踪

| ID | 缺口 | 优先级 | 状态 | 产物 |
|---|---|---|---|---|
| P0-1 | H67 ep35/40 hardware-order+dyadic 数值 | P0 | **已完成**（2026-08-18） | results/.../deploy_valid825/q7q17/* + h67_fullres_ep35_ep40_deploy_q7q17_valid825_20260818.{json,md} |
| | | | ep35 HW-order AEE 1.3287（float 1.3297，无损）；ep40 1.3358 确认 rank-1=ep35；dyadic ep35 1.3279 == QF7 1.327912 交叉验证；load audit 已补录（210/210, missing=0） | |
| P0-2 | MVSEC 四序列门核实 + Local5 差距 | P0 | **已完成**（2026-08-18） | results/mvsec_four_sequence_gate_verify_20260818.{json,md} |
| P0-3 | 100 样本 raw Q/K trace（ep35） | P0 | **已完成**（2026-08-18 18:29，1h45m） | h67_fullres_ep35_t450_profile100_20260818/（RECEIPT）+ hw_autoresearch_nts07/results/h67_ep35_multisample100_t450_real_rtl_bit_trace/（1200 npz, 958 MB, manifest SHA 2bb0dc3e） |
| P1-4 | full-encoder 活动占比（双线） | P1 | **已完成**（2026-08-18；Local5 ep44 shadow 运行 19:04 落档） | local5_fullres_ep44_t450_profile100_20260818/（RECEIPT）+ full_encoder_amdahl_12block_20260818/ |
| P1-5 | 12-block 同窗表 | P1 | **已完成**（2026-08-18；H67 完整 h60 12-block 表；Local5 无 H60 结构，operator 级表补位） | full_encoder_amdahl_12block_20260818/（RECEIPT） |
| P2-6 | 官方 DSEC test writer | P2 | **算法侧完成**（2026-08-18 19:32 UTC）：数据本地核实、writer 实现、数值正确性核实（与权威 eval 逐样本一致，~10x 为口径错误非缺陷）、全量 416/416 PNG（~5 min GPU）落档；**官方注册 + zip 上传为外部动作**，返回官方 AEE 后回填 | entrypoints/eval_DSEC_test_submission.py + results/dsec_test_submission_full_20260818/（RECEIPT） |
| P2-7 | Local5 cross-head 数值 | P2 | 待需求 | 视需求 |

### 纪律事件记录（2026-08-18）

- **18:31 UTC：磁盘 overlay `bsa_attention.py` 被并发 D1（T>2 时间商算子实现）agent 修改**
  （SHA 66d0a339 → a8e94f56；D1 自己在改前保存了 pristine 副本
  `/tmp/bsa_attention_pristine_20260818.py`，SHA=66d0a339 已核对）。
- 处置：P0-1/P0-3 已在 16:44 前完成，收据绑定 66d0a339 不受影响。
- P1-4（Local5 ep44 profile100）改为 **shadow run**：字节一致拷贝的 profiler/trace_writer
  （SHA 5f21c8d7/75c91340 不变）从 /tmp/p1_4_shadow 运行，其 overlay/ 为 pristine 66d0a339；
  磁盘 overlay 未触碰（D1 的工作文件），summary.json 中记录 shadow_run 原因与 SHA。
- 后续所有 DATE 线运行（评测/profile）必须显式校验 overlay SHA；若磁盘 overlay 再次变化，
  一律走 shadow 方案，不许还原/覆盖磁盘文件。

## 3. 纪律自检（执行前）

- 不启动 H82/H86 任何训练/评测；不改 overlay bsa_attention.py（记录 SHA 66d0a339…）；
- 新运行 seed 0 / 不混表；评测 batch=1、mode valid（硬件序配置为 deploy 定量化路径，同 Local5 先例）；
- 一次一个 GPU 任务；长任务（>2 h）先写状态文件；
- 证据分档：P0-1/P0-3 数值与 trace 为 [prof]/[模型]，RTL 回放 [rtl] 归硬件侧。
