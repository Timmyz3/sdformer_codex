# Motion/H67 quotient-file 侧车：容量/spill 语义与宏粒度映射冻结（证据 ③）

日期：2026-08-18。证据等级：`[模型]`（文档冻结，数值链挂靠证据 ①+② golden 与现网 `[rtl]` 只读端口）。红线：纯 CPU、不写 RTL、不改任何现有文件、`docs/359`/selector/生产 RTL/194436Z 不动。
依据：`CLAUDE_MOTION_SIDECAR_DESIGN_20260818.md` §5 前置 3 与 §1.2/§1.4/§4.2；证据 ①+②（`results/motion_sidecar_golden_evidence_20260818/`）。

## 0. 冻结结论一句话

descriptor 流物理容量 = 合法域 D≤2P=450 条（11 bit/条，4,950 bit，fakeram45_256x32 单宏 2 条/字）；occupancy bitmap 容量 = 163 bin（合法域）；spill 语义 = **fail-closed（protocol_error）**，不降级 C7；并冻结一枚设计文档账目缺失的 **per-class multiplicity count 文件（163×9 = 1,467 bit）**——它是 Mode A 与现网 top 的 class 流契约 bit-exact 的充要对象（详见 §2.5 对账项）。

## 1. 冻结依据与口径

- 容量按合法域不按 profile（设计文档 §1.2；docs/389 §4：四码表被全量 max=63 证伪的教训）——写 RTL 的 SRAM 尺寸依据是 `D ≤ 2P = 450`、`C ≤ 163`（MAX_SCORE=162 → 163 bins），不是 p95。
- workload 锚定（门限/能量腿用，冻结 ep35 全量 672,000 行，证据 ② canonical 口径）：`D p95=265`、`D mean=230.66`、`D max=435`（S3）、`C p95=16`、`C mean=5.1143`、`C max=46`。p95 是"workload 代表性"门，不是容量门。
- 所有 D/C 上界均为**构造不变量**而非统计外推：225 个 pair 每个至多发射 2 条 descriptor ⇒ D ≤ 450；score 域 0..162 ⇒ C ≤ 163。现网 `MAX_DESCRIPTORS = 2*PAIRS` 参数与 pair_seen 重复检查同源。

## 2. 容量冻结

### 2.1 三对象物理容量

| 对象 | 字段 | 位宽 | 物理容量 | 合法域上界 | 依据 |
|---|---|---:|---:|---|---|
| occupancy bitmap | 每窗口 occupied class bitmap | **163 bit**（Motion）/ 513 bit（H82 兼容口径） | 163 bin | C ≤ 163 = MAX_SCORE+1 | 设计文档 §1.1 |
| quotient descriptor 流 | `{class_id[7:0], k_mask[1:0], pair_last[0]}` | **11 bit/条**（H82 口径 12 bit） | 450 条 × 11 bit = **4,950 bit** | D ≤ 2P = 450（构造：225 pair × ≤2 条） | 设计文档 §1.1/§1.4 |
| denominator certificate | `row_max[8:0] + denom_shift[4:0]` | **14 bit** | 1 份 | — | 设计文档 §1.1（denom_shift 端口宽度按现网 `h67_row_qmax_denominator_certificate` 取 [5:0]，账目 5 bit 是有效值位宽，17<32 一致，见 §5 对账项 #2） |

### 2.2 descriptor 流 SRAM 打包（细化设计文档 §1.4 的"单宏可容纳"）

- 450 条 × 11 bit = 4,950 bit < fakeram45_256x32 的 8,192 bit ⇒ **单宏**，但深度 450 > 256，需**每字 2 条打包**：descriptor 逻辑地址 = pair_id（0..224 的 1–2 条/cycle），物理字 = `pair_id[7:1]`（225 字 ≤ 256），字内 slot = `pair_id[0]`。split pair 的两条落在同一字 → 每 pair 至多 **1 次宏写事务**（split 满字、common 半字），1RW 单口成立。
- 字 = `{entry0[10:0], entry1[10:0], 10'b0}`（22 bit 有效 + 10 bit padding）。padding 位**保守计入整字写活动**，与现网 `h67_banked_active_descriptor_store` MEMORY_IMPL=1 的 32-bit 代理宏惯例（padding 随低位数据写、整字活动计入）完全一致。
- 宏族：与现网 K store 同族 `fakeram45_256x32`（`rtl_h67/fakeram45_256x32_bb.sv`）；现网参照 `h67_banked_active_descriptor_store`（DEPTH=450, DATA_W=28，双 bank 各 225，MEMORY_IMPL=1 绑定 2× fakeram45_256x32）。侧车以"单宏 2 条/字"替代现网双 bank 结构，端口模型相同（1RW、ce_in/we_in/w_mask_in/addr_in/wd_in/rd_out）。

### 2.3 spill 语义定稿：**fail-closed，不降级 C7**

建议与理由（写入 RTL 前置合同）：

1. **D ≤ 450 是证明级不变量**：225 pair × ≤2 条/对。任何 D>450 必然是协议违例（重复 pair_id、越界 pair_id），不可能由合法 workload 触发。现网已对同类违例 fail-closed（`h67_temporal_quotient_shiftmax_gate_top` 的 pair_legal/pair_seen_q + `protocol_error`；frontend 的 `quotient_descriptors != perf_quotient_descriptors` 终态检查；`h67_banked_active_descriptor_store` 的越界/冲突 protocol_error）。docs/261 已有 ATLIF fail-closed 签核先例。
2. **降级 C7 会破坏 default-off bit-exact 合同**：降级路径意味着"合法输入下与现网数值一致、非法输入下不一致"的双重语义，miter 无法一次性证明，且会掩盖协议违例——这正是 docs/389 禁止的"后验 profile 拟合 exact"的硬件版。
3. **降级开关本身扰动能量对比**：C7 路径 mux 加入数据通路，SAIF 腿（⑤）的同端口公平性被污染。
4. 唯一需防的"容量溢出"是**实现缺陷**（如 SRAM 地址截断），fail-closed 恰好暴露它；RTL 侧加 `protocol_error` 于 descriptor 写地址 ≥450 或 bitmap 越界（score>162）。

**代价为零**：合法域下 spill 不可能发生；p95 门（0.589 ≤ 0.60）只约束"workload 代表性"，不约束正确性（设计文档 §5 杀 3(d)）。

### 2.4 对账项 #1（关键发现）：Mode A 需要一枚账目缺失对象——per-class multiplicity count 文件

**现象**：现网 top `h67_temporal_quotient_shiftmax_gate_top` 的 `row_sum_q8_q` 由 **class 流**喂入：`row_sum += exp(class_delta) × class_multiplicity`（`class_multiplicity[8:0]`，class 相位先于 active 相位，`class_phase_done` 后 active 才被允许）。Mode A 的 bit-exact 分母 `Z_A = Σ_class exp(c−rm)·mult(c)`（golden 已证 672,000 行 0 mismatch）要求 sidecar 目录发射的 class 流携带与现网 `class_hist` 完全一致的 multiplicity。设计文档 §1.3 的 `S_qg/S_do` 账目不含任何 multiplicity 对象（§1.1 还写"无 multiplicity 字段"），但：

- **class 流契约（class_multiplicity[8:0]）是 TESC top 的原生输入**，top 不能被改动（default-off 要求现网 RTL 原封）；
- 逐 descriptor 的 multiplicity（common=2 / split=1，由 pair_last 与对内位置恢复）只能在**再扫一遍 descriptor 流**时获得（D 条/class 级扫描 → C×D 读，不可行）或**在 build 期累加**；
- 设计文档自己的周期模型 `450+C+desc`（+0.7% 持平）只允许 class 相位 = C 拍——意味着 multiplicity 必须已在 build 期物化。

**冻结**：per-class count 文件 **163 × 9 bit = 1,467 bit**（宽度 = 现网 `class_hist` 同款，`COUNT_W = $clog2(2·PAIRS+1) = 9`；最大值 450 < 512 恰好够）。build 期每 descriptor 增量 = `popcount(temporal_mask)`（Mode A，复刻现网 class_hist 语义）或 `1`（Mode B one-vote，`Z_B = Σ_class count(c)·exp(c)` 逐位一致）；模式位选择增量。class 相位按 bitmap 升序扫描，每 present class 发射 `{class_score, class_multiplicity=count[c], class_last}`——**同一个目录数据通路、同一份 class 流契约同时服务 Mode A/B**。

**诚实重述（存储位口径，p95 保守点 (16,265)）**：

| 账 | pair-gather 强基线 | 侧车（设计账） | 侧车（+count 文件） | 差值（vs 基线） |
|---|---:|---:|---:|---:|
| quotient-gate | 4,132 | 3,222 | **4,689** | **+13.5%（≥20% 腿 FAIL）** |
| denom-only | 4,132 | 3,092 | **4,559** | **+10.3%（≥20% 腿 FAIL）** |
| 现网 C7 物化（上下文，非强基线） | — | — | 7,957 | **−41.1%** |

- 含 count 文件后，**存储位口径的"相对 fused pair-gather 降 ≥20%"腿在 p95 点不成立**（三腿门中 C 腿 16≤192 ✓、desc 腿 0.589≤0.60 ✓、state 腿由 −23.4%/−26.5% 改为 +13.5%/+10.3% ✗）。合法域最坏行 (163,450)：qg' = 8,047（+47.5% vs 5,455）、do' = 6,429（+17.8%）。H82 兼容口径更甚（count 文件 513×9=4,617 bit）。
- **不后验改门**：state 腿按设计文档 §4.2 的运作腿重新锚定——能量腿是 **SAIF 实测组件动态 ≥15%**（⑤），其载体是**写活动位宽差**而非静态容量：descriptor 流写活动 1 次宏写/对（32-bit 字）vs 现网 active pair store 28-bit×≤2 写/对（56 bit）+ token-gate 物化写（450×9）+ exp 事务（现网已 −22.04% VCD 观测，docs/263 §3.3）；count 文件写活动与现网 class_hist 写活动同构（每 descriptor 1 次 RMW），两侧对消。**⑤ 的 SAIF 腿预期仍可过 ≥15%，但 ⑤ 必须把 count 文件的活动同时计入两侧，不能只报目录差分**（设计文档 §4.3 记账要求同义）。
- 论文措辞落点：存储对象相对**现网 C7 物化 −41%**（上下文列）与**写活动位宽差**（energy 腿）为诚实口径；"state 相对 pair-gather −23.4%"标注为含遗漏对象的历史口径，写 RTL 前作废。

这个对账项正是"写 RTL 前冻结"要抓的东西（评分循环判据：不后验改门、诚实披露、容量按合法域）。

## 3. 宏粒度映射冻结

### 3.1 三对象 → 宏映射

| 对象 | 容量 | 宏映射 | 理由 | 现网/仓库参照 |
|---|---|---|---|---|
| occupancy bitmap | 163 bit（H82：513 bit） | **flop 寄存器文件**（非深宏；1 写端口按位置位 + 1 扫描读端口） | 单窗口生命周期、C≤163、置位写活动是能量腿卖点；现网目录的 `class_present_q[MAX_SCORE:0]` 同为 flop 向量，端口模型公平 | `h67_temporal_weighted_scs_directory` 的 class_present_q；`h67_pair_bitmap_metadata_builder` 的 `active_pair_mem[0:PAIRS-1]` |
| quotient descriptor 流 | 450×11 bit = 4,950 bit | **fakeram45_256x32 单宏**（225 字 × 2 条/字，10 bit padding，1RW） | 4,950 < 8,192；宏族与 K store 同族；打包使每 pair ≤1 写事务，1RW 成立 | `rtl_h67/fakeram45_45_256x32_bb.sv`（K store、`h67_banked_active_descriptor_store` MEMORY_IMPL=1）；`rtl_qfit/qfit_fakeram45_relation_bank_450`（450 深度双宏覆盖惯例） |
| denominator certificate | 14 bit | **flop 寄存器**（`row_max[8:0]` + `denom_shift[5:0]` + certificate_valid/pass） | 单实例标量；复刻现网证书模块端口模式 | `h67_row_qmax_denominator_certificate`（certificate_valid/certificate_pass/denominator_shift/accepted_pairs/protocol_error 模式） |

命名提案（RTL 写手可调，宏族绑定冻结）：`h67_quotient_sidecar_occupancy_bitmap`、`h67_quotient_sidecar_descriptor_stream`、`h67_quotient_sidecar_denominator_certificate`、`h67_quotient_sidecar_directory`（目录主状态机）。前缀沿用 h67_ 惯例（参照 h67_zkqi_*、h67_mssb5_* 的机制前缀法），不新增论文缩写（docs/359 措辞边界：侧车不新增贡献名、不新增缩写）。

### 3.2 与仓库大宏的对照（为什么侧车对象不需要 FCSR/TCFM5 级结构）

| 参照宏 | 结构 | 侧车对象差异 |
|---|---|---|
| FCSR ring（docs/208） | (32+5×9)×3×15 = 3,465 bit 三行 gate ring，**跨窗口驻留** | 侧车三对象全部**单窗口生命周期**（window_start 复位），无驻留/无跨头，无 ring 移位 |
| TCFM5（rtl_qfit/qfit_tcfm5_acc_bank） | 5 × 90×1024 1R1W 向量 Acc bank，持续 RMW | 侧车目录 build/expand 相位互斥 ⇒ 每对象 1RW 足够，无需 1R1W |
| Local5 relation bank（qfit_fakeram45_relation_bank_450） | 2× fakeram45_256x32 覆盖 450 深度 | 侧车 descriptor 流用单宏 2 条/字覆盖 450 深度，端口模型相同 |

### 3.3 宏接口草案（与 TESC 交接点）

**h67_quotient_sidecar_occupancy_bitmap（163-bit flop RF）**
- 写：`set_valid + set_class[7:0]`（build 期，1 拍 1 位）；读：`scan_valid/scan_ready + scan_class[7:0] + scan_set`（class 相位，升序扫描）。
- 交接点：写侧来自 quotient 边界（descriptor 的 class_id）；读侧驱动 class 流。

**h67_quotient_sidecar_descriptor_stream（fakeram45_256x32 绑定）**
- 1RW；写：`write_valid + write_pair_id[7:0] + write_desc0[10:0] + write_desc1_valid + write_desc1[10:0]`（每 pair ≤1 写事务，2 条/字）；读：`read_valid/read_ready + read_pair_id[7:0] + read_desc[10:0]`（expand 期，pair 序 1 条/拍）。
- 交接点：写侧 = quotient 边界（`quotient_valid/quotient_ready/quotient_pair_id/quotient_score_q7/quotient_temporal_mask/quotient_active_mask`，即现网 `h67_temporal_weighted_scs_directory` 的 in_* 端口）；读侧 = active 流（active_pair_id/active_score_q7/active_temporal_mask/active_k_mask/active_last）。

**h67_quotient_sidecar_denominator_certificate（14-bit flop）**
- 输入：`row_load_start/load_accept/load_pair_id` + Q 计数（复刻现网 `h67_row_qmax_denominator_certificate` 模式）；输出：`certificate_valid/certificate_pass/denominator_shift[5:0]/row_max[8:0]`。
- 交接点：row_max 同时驱动现网 top 的 `row_max_q7`（SCS max 逐位一致，golden 已证）；denom-only 变体在 expand 按 class_id 重算 exp2（expand_exp2 = D 次/窗，设计文档 §1.3）。

### 3.4 同端口记账表（⑤ 执行口径）

| 项 | RQTB2S（现网，docs/264） | sidecar | 记账要求 |
|---|---|---|---|
| descriptor 存储 | 450×28 banked（2× fakeram45_256x32） | 450×11 单宏 2 条/字（1× fakeram45_256x32） | 宏面积、写活动按 32-bit 整字口径两侧一致 |
| K store / FIFO / emit 后端 | h67_sync_dual_bank_k_store + slot FIFO + Shiftmax | **同一实例，不动** | 不许只报目录差分（设计文档 §4.3） |
| 逻辑库 | Nangate45（docs/264 同流程，5 ns） | 同 | TT 25°C 标称 + SS/FF 敏感 |
| 反压/激励 | 同一 LFSR 公平 replay + 确定性反压（docs/264 §3.1） | 同 | 同 seed、同 depth32 |

## 4. 每窗口生命周期状态机草案（文本级，不写 RTL）

五个阶段，对象状态迁移表：

| 阶段 | 触发/条件 | 动作 | 对象状态迁移 |
|---|---|---|---|
| **IDLE** | rst / window_done | 等 window_start | 全部对象可复位 |
| **FETCH**（fetch→score） | window_start | pair_valid/pair_ready 逐对收 225 对；`h67_temporal_score_quotient` 算 Q7 分数；K pair 写 `h67_sync_dual_bank_k_store`（现网不动）；pair_seen/pair_legal 检查 | K store：写（fetch 期）；bitmap/descriptor/count：window_start 清零 |
| **BUILD**（quotient build） | quotient 边界（1–2 条/对） | 每 descriptor：descriptor 流写（addr=pair_id，≤1 宏写事务/对）；occupancy bitmap 置位；count[c] += popcount(temporal_mask)（Mode A）/1（Mode B）；running row_max；D 计数；seal 时校验 D = 2P − E（golden 0 mismatch 链） | bitmap：置位；descriptor 流：写；count：RMW；row_max：running max |
| **CLASS**（score） | window_seal → seal_ready（现网逻辑：directory 空闲 && 前端排空） | 按 bitmap 升序扫 C 拍；每 present class 发射 `{class_score, class_multiplicity=count[c], class_last}`；**row_sum 在此 settle（class_last 一拍后 class_phase_done）**；certificate 出 row_max/denom_shift | bitmap：扫描读；count：读；certificate：valid |
| **EXPAND** | class_phase_done | 按 pair 序读 descriptor 流（1 条/拍）：由 pair_last+对内位置恢复 temporal membership（common=2/split=1）；k_mask 选 K0/K1 双 bank（现网 h67_sync_dual_bank_k_store，read_req_valid[1:0]）；gate = quant(exp(class_delta), row_sum, n_tokens=450, preserve_mean)；t0→t1 发 token（out_valid/out_token_id/out_k_bits/out_gate_q17/out_threshold_q8/out_last）；denom-only 变体按 class_id 重算 exp2 | descriptor 流：读；K store：读；emit 状态机（现网不动） |
| **RETIRE** | active_last → 末 token → out_last | window_done = frontend_done && !emit_valid（现网）；8 项 perf 计数锁存（window_start 复位） | 全部对象可由下窗口 window_start 复位 |

**冻结不变量（miter 前置，文本级）**：
1. **row_sum 先于首 token gate settle**（bit-exact 的数值前提；count 文件方案下 = class 相位 C 拍内 settle，周期模型 450+C+D 成立）。
2. class_phase_done 先于 active 相位（现网 top 的 `active_valid && !class_phase_done_q → protocol_error` 检查原样保留）。
3. seal 时 D = 2P − E、成员 popcount 恒 2P、pair_last=1 计数恒 225（golden 0 mismatch 三项）。
4. D ≤ 450、C ≤ 163 由 fail-closed 强制（§2.3），非统计外推。
5. token 序：每 pair 先 t0（token_id=2p）后 t1（2p+1），pair 扫描序（现网输出边界逐位一致）。

## 5. 与设计文档对账

| 设计文档条目 | 冻结结果 |
|---|---|
| §1.2 容量按合法域（450/163），不按 profile | ✓ 冻结，且 D≤450 为构造不变量 |
| §1.4 "450×11 = 4,950 bit → fakeram45_256x32 单宏" | ✓ 细化：2 条/字打包（225 字），每 pair ≤1 写事务，1RW 成立 |
| §1.3 S_qg/S_do 账目 | **对账项 #1**：须加 per-class count 文件 163×9=1,467 bit；state 腿存储位口径重述见 §2.4（≥20% 腿 FAIL，能量腿重新锚定 SAIF ≥15%） |
| §2.1 "无 multiplicity 字段" | 收紧：mult 信息必须物化（count 文件）才能保 class 流契约与 top 原封；"无 multiplicity 字段"仅指 descriptor record 不含 mult |
| §5 杀 3(d) "容量是合法域（450/163），spill 无损" | ✓ spill 语义定稿 fail-closed（无损=合法域下永不 spill；溢出=协议违例，fail-closed 暴露） |
| §1.1 denom_shift[5] | **对账项 #2**：端口宽度按现网证书惯例取 [5:0]（6 bit）；账目 5 bit 是有效值位宽（17<32），不冲突 |
| §4.2 同宏三方可比性 | ✓ 记账表 §3.4；⑤ 必须把 count 文件活动计入两侧 |

## 6. 剩余前置清单（交叉核对设计文档 §6 的 5 项）

| # | 前置证据 | 状态 |
|---|---|---|
| 1 | CPU golden 参考 + 现网账本对齐（Mode A 全量 0 mismatch + one-vote 差异包） | **已完成**（证据 ①+②，golden.py/report.json/report.md） |
| 2 | descriptor p95 口径对账（0.578 vs 0.589 归因、逐 stage 分布、C/D 联合） | **已完成**（证据 ②，口径定稿 0.589） |
| 3 | 容量与 spill 语义冻结、宏粒度映射、同端口记账表 | **本文档（证据 ③）完成** |
| 4 | 端口契约冻结（TESC 边界、8 项 perf、default-off） | **`CLAUDE_MOTION_SIDECAR_PORT_CONTRACT_20260818.md`（证据 ④）完成** |
| 5 | SAIF/PPA 同宏基线（Fixed2S/RQTB2S/sidecar 三方，fakeram45+Nangate45，TT/SS/FF，5 ns） | **唯一剩余项**，新思机器（docs/433：DC/STA/SAIF/PTPX 在另一台有目标库服务器） |

**剩余前置清单 = 仅 ⑤**。写 sidecar RTL 前需 ⑤ 门执行（组件动态能量 ≥15% 同端口、逻辑+宏面积 ≤+10%、Fmax ≥−5%，docs/445/433 合同）；⑤ 执行时把 §2.4 的 count 文件与 §3.4 记账表作为对象集合（两侧同口径），并复核 §2.4 的存储位重述在论文口径的落点。

本文件不修改任何现有文件；只新增本 md。证据分档 `[模型]`。
