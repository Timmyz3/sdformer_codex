# Motion/H67 quotient-file 侧车：端口契约冻结（证据 ④）

日期：2026-08-18。证据等级：`[模型]`（文档冻结；TESC 端口清单来自 `rtl_h67/` 只读引用，`[rtl]`）。红线：纯 CPU、不写 RTL、不改任何现有文件、`docs/359`/selector/生产 RTL/194436Z 不动。
依据：`CLAUDE_MOTION_SIDECAR_DESIGN_20260818.md` §5 前置 4 与 §2.3（措辞边界/计数器口径）；`rtl_h67/h67_temporal_quotient_shiftmax_gate_top.sv`、`h67_temporal_quotient_scs_frontend.sv`、`h67_temporal_weighted_scs_directory.sv`、`h67_temporal_score_quotient.sv`、`h67_sync_dual_bank_k_store.sv`、`h67_banked_active_descriptor_store.sv`、`h67_row_qmax_denominator_certificate.sv`（只读）。

## 0. 冻结结论一句话

侧车以**目录层替换**方式插入 `h67_temporal_quotient_scs_frontend` 内部的 directory 物化层：quotient 边界（输入）与 class/active/row_max 边界（输出）**信号名、位宽、握手逐位不变**；现网 top（`h67_temporal_quotient_shiftmax_gate_top`）及下游 exp LUT / gate quant / K store / emit 一个都不动；8 项 perf 计数器同名同宽同语义，侧车**不重复计数**；default-off = 目录实例参数 `SIDECAR_DIRECTORY=0`，现网实例树原封（bit-exact by construction，无运行时 mux）。

## 1. TESC 现有接口盘点（rtl_h67 只读）

### 1.1 TESC 顶层：`h67_temporal_quotient_shiftmax_gate_top`

| 组 | 信号 | 方向 | 位宽 | 说明 |
|---|---|---|---|---|
| 控制 | clk_core / rst_core | in | 1/1 | 全局 |
| 控制 | window_start / window_seal | in | 1/1 | 窗口边界 |
| 控制 | cfg_preserve_mean | in | 1 | 0/1 两档（miter 各一遍） |
| 控制 | cfg_threshold_q8 | in | 8 | threshold 截断 |
| 输入 | pair_valid / pair_ready | in/out | 1/1 | 握手 |
| 输入 | pair_id | in | 8（PAIR_ID_W=$clog2(225)） | scan 序 0..224 |
| 输入 | q_pair / k_pair | in | 64/64 | 低 32=t0，高 32=t1 |
| 输出 | seal_ready / window_done | out | 1/1 | 窗口协议 |
| 输出 | out_valid / out_ready | out/in | 1/1 | token 握手 |
| 输出 | out_last | out | 1 | 窗口末 token |
| 输出 | out_token_id | out | 9（TOKEN_W=$clog2(451)） | 2p/2p+1 |
| 输出 | out_k_bits | out | 32 | gated-K 展开 |
| 输出 | out_gate_q17 | out | 9 | Q1.7 gate |
| 输出 | out_threshold_q8 | out | 8 | 原样透传 |
| 错误 | protocol_error | out | 1 | fail-closed 汇聚 |
| perf | 8 × perf_* | out | 8×32 | 见 §3 |

### 1.2 内部边界（侧车替换点）

```
h67_temporal_score_quotient ──quotient 边界──▶ h67_temporal_quotient_scs_frontend 内
  （u_quotient, 现网不动）                     h67_temporal_weighted_scs_directory
                                                  │ class 流 / active 流 / row_max_q7
                                                  ▼
                                      h67_temporal_quotient_shiftmax_gate_top
                                      （exp LUT / row_sum / gate quant / emit，现网不动）
```

- **quotient 边界（侧车输入）** = 现网 `h67_temporal_weighted_scs_directory` 的 in_* 端口（frontend 内连线：`quotient_valid/quotient_ready/quotient_pair_id/quotient_score_q7/quotient_temporal_mask/quotient_active_mask`），源自 `h67_temporal_score_quotient` 输出（out_valid/out_ready/out_pair_id[7:0]/out_score_q7[15:0] signed/out_temporal_mask[1:0]/out_active_mask[1:0]）。
- **目录输出（侧车输出）** = 现网目录的 class/active/row_max 端口（`h67_temporal_quotient_shiftmax_gate_top` 的 row_sum 累加器与 emit 协议检查直接消费）。
- **K 边界** = `h67_sync_dual_bank_k_store`（现网不动）：k_pair 写（fetch 期）、active k_mask 驱动的 read_req_valid[1:0] 双 bank 读（expand 期）。
- **宏参照** = `h67_banked_active_descriptor_store`（MEMORY_IMPL=1 的 fakeram45_256x32 绑定与 32-bit 代理宏 padding 惯例）；`h67_row_qmax_denominator_certificate`（证书端口模式：row_load_start/load_accept/load_pair_id → certificate_valid/certificate_pass/denominator_shift[5:0]/accepted_pairs/protocol_error）。

## 2. 侧车目录信号级契约（与现网 `h67_temporal_weighted_scs_directory` 端口逐位相同）

| 信号 | 方向 | 位宽 | 握手/语义 | 与现网一致性 |
|---|---|---|---|---|
| window_start / window_seal | in | 1/1 | 窗口边界 | 相同 |
| window_ready / window_done | out | 1/1 | seal 能力/窗口完成 | 相同（现网：window_ready = IDLE\|DONE；window_done = ST_DONE） |
| in_valid / in_ready | in/out | 1/1 | quotient 边界 valid/ready，1 条/拍 | 相同（现网：in_ready = ST_BUILD） |
| in_pair_id | in | 8 | 0..224 | 相同 |
| in_score_q7 | in | 16 signed | Q7 分数 | 相同 |
| in_temporal_mask | in | 2 | 11=common / 01,10=split | 相同 |
| in_active_mask | in | 2 | K0/K1 active（= k_mask 同义） | 相同 |
| class_valid / class_ready | out/in | 1/1 | class 相位；现网 top class_ready=1 | 相同 |
| class_score | out | 8（CLASS_W=$clog2(163)） | 0..162 | 相同 |
| class_multiplicity | out | 9（COUNT_W=$clog2(451)） | **count 文件输出**（Mode A：Σpopcount；Mode B：Σ1），保 top 的 row_sum bit-exact | 数值相同（golden 已证 Z_A==Z_C7，672,000 行 0 mismatch） |
| class_last | out | 1 | 末 class 一拍 | 相同（现网：仅剩 1 个 present bit） |
| active_valid / active_ready | out/in | 1/1 | active 相位；现网 top active_ready=!emit_valid_q | 相同 |
| active_pair_id | out | 8 | 0..224 | 相同 |
| active_score_q7 | out | 16 signed | 由 descriptor 流读回 | 相同（bit-exact 目标） |
| active_temporal_mask | out | 2 | pair_last + 对内位置恢复 | 相同 |
| active_k_mask | out | 2 | descriptor record 的 k_mask | 相同 |
| active_last | out | 1 | 末 descriptor 一拍 | 相同 |
| row_max_q7 | out | 16 signed | SCS max；certificate row_max 同值 | 相同（golden：row_max 逐位一致） |
| protocol_error | out | 1 | 含容量/协议违例（§③ fail-closed） | 语义超集（新增 descriptor 写越界、bitmap 越界） |
| perf_quotient_descriptors | out | 32 | D = 2P − E | 相同 |
| perf_original_tokens | out | 32 | 2P = 450 | 相同 |
| perf_active_entries | out | 32 | D | 相同 |

**相位契约（文本级）**：class 相位先于 active 相位，严格互斥（现网 ST_CLASS→ST_ACTIVE 顺序）；现网 top 的 `active_valid && !class_phase_done_q → protocol_error` 检查原样保留；`seal_ready = directory 空闲 && 前端排空 && !quotient_valid`（现网 frontend 逻辑）；`window_done = frontend_done && !emit_valid_q`（现网）。

**黄金对账**：`report.json` 的 mode_a 11 个对照项（stored equal receipt、D=2P−E、pair_last=225、popcount 恒 2P、token 序、Z_A==Z_C7、gate 逐类一致、公平包 138 行 equal/gate/K/token 序）全 0 mismatch（672,000 行 PASS）——本契约的数值面已被 golden 锚定；硬件 bit-exact 仍需 ⑤ 后 §4.4 miter。

## 3. 8 项 perf 复用（逐项，侧车不重复计数）

| # | 计数器 | 语义 | 现网来源 | 侧车处理 |
|---|---|---|---|---|
| 1 | perf_pairs | 收对计数 | quotient front（u_quotient） | 侧车不碰（frontend 原样） |
| 2 | perf_quotient_descriptors | D = 2P−E | 目录 | **同名同宽同语义**；侧车目录内 D 计数为协议/容量检查（fail-closed），不另导出口 |
| 3 | perf_original_tokens | 2P=450 | 目录 | 同上（常量语义，侧车目录透传同值） |
| 4 | perf_active_entries | D | 目录 | 同上 |
| 5 | perf_equal_pairs | E | quotient front（u_quotient） | 侧车不碰 |
| 6 | perf_class_transactions | C | top（class 流消费计数） | 侧车不碰（top 原样） |
| 7 | perf_exp_transactions | = class_transactions + active_entries | top 组合 | 侧车不碰 |
| 8 | perf_emitted_tokens | 输出 token 数 | top（emit） | 侧车不碰 |

- **不重复计数声明**：8 个名字 = 单一事实源；侧车目录内部计数器（D、C、count 文件、bitmap 扫描进度）只服务容量/protocol 检查与相位控制，**一律不导出**，不新增任何 perf 名/缩写（docs/359 措辞边界：不新增贡献名）。
- **SAIF 同口径**：5 项由 frontend/top 原样产生（侧车不参与），3 项由侧车目录产生但语义与现网一致 ⇒ 与现网 VCD 同口径（设计文档 §2.3 计数器口径）。

## 4. default-off 插入点

- **开关位置**：`h67_temporal_quotient_scs_frontend` 内 directory 实例处，参数 `SIDECAR_DIRECTORY`（0=现网 `h67_temporal_weighted_scs_directory`，1=sidecar 目录），generate/参数化实例选择。
- **bit-exact 保证**：SIDECAR_DIRECTORY=0 时实例树与现网逐叶相同（elaboration 时静态选择，netlist 中不存在禁用分支，**无运行时 mux、无时序/能量扰动**）——default-off 与现网 C7 路径 bit 级一致是**构造性质**，非仿真结果。
- **共享不动**：K store（h67_sync_dual_bank_k_store）、slot FIFO、exp LUT、gate quant、emit 状态机、out 边界、window 协议、threshold/preserve_mean 语义——全部原样（设计文档 §2.1 的 4 条 bit-exact 约束逐条保持）。
- **先例**：现网 `h67_temporal_slot_shiftmax_sync_k_2s_top` 已有 `QUOTIENT_ENABLE/MSSB5_SCORE_FRONT/MEMORY_IMPL` 参数化惯例；侧车沿用同款（参数化实例选择，不改任何现有 RTL 文件）。
- **开关外开一个口**：`descriptor_issue_enable` 类门控语义（现网 2s top 已有）不属侧车对象，侧车不新增门控信号；若 ⑤ 需要关闭侧车做能量对照，用 SIDECAR_DIRECTORY=0 即可，无需额外开关。

## 5. 与 docs/359 主表的关系（只读约束）

- 封存列全部不动：主锚点 `1.1865×（112589→94891）`、held-out ep30 `1.1850×（111807→94348）`（禁止与主锚平均）、机制账本 `equal 28001/31050`、`slot 34099/62100 = −45.09%`、密行 `20604/23625 / 26646/47250 = −43.61%`（occupancy-gated，不进主表合成）。
- 侧车**不新增贡献名、不新增缩写**；论文只能作为"同一 quotients 流的目录物理实现变体"，在 **energy 腿**下呈现（设计文档 §2.3 措辞边界）；周期腿 +0.7% 持平（450+C+D=696 vs 691）如实陈述为 non-negative，不写新周期表、不与 energy 合成 EDP。
- 证据①+② 已复算三方账本一致（62100/34099/28001/31050）且未触碰 docs/359 文件本身；本契约同样只读引用。

## 6. 剩余前置清单（交叉核对设计文档 §6 的 5 项）

| # | 前置证据 | 状态 |
|---|---|---|
| 1 | CPU golden 参考 + 现网账本对齐（Mode A 全量 0 mismatch + one-vote 差异包） | **已完成**（证据 ①+②） |
| 2 | descriptor p95 口径对账（0.578 vs 0.589 归因、逐 stage 分布） | **已完成**（证据 ②，口径定稿 0.589） |
| 3 | 容量与 spill 语义冻结、宏粒度映射、同端口记账表 | **已完成**（`CLAUDE_MOTION_SIDECAR_CAPACITY_PORTS_20260818.md`，证据 ③；含对账项 #1 count 文件、#2 certificate 位宽） |
| 4 | 端口契约冻结（TESC 边界、8 项 perf、default-off） | **本文档（证据 ④）完成** |
| 5 | SAIF/PPA 同宏基线（Fixed2S/RQTB2S/sidecar 三方，fakeram45+Nangate45，TT/SS/FF，5 ns；组件动态 ≥15%、面积 ≤+10%、Fmax ≥−5%） | **唯一剩余项**，新思机器 |

**写 sidecar RTL 的前置 5 项只剩 ⑤**。⑤ 执行时把证据 ③ 的 §2.4/§3.4（count 文件对象集合、同端口记账表、fail-closed）与本契约的 §2/§3/§4（信号契约、8 项 perf 单源、SIDECAR_DIRECTORY 参数点）作为冻结输入；miter（设计文档 §4.4）的公平包/反压/seed 规格不受本契约影响。

本文件不修改任何现有文件；只新增本 md。证据分档 `[模型]`（端口清单挂靠 `[rtl]` 只读引用）。
