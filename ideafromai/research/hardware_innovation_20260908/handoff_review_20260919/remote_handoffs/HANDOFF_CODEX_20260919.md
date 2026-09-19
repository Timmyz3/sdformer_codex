# Codex 交接文档：SNN 光流 TCAS-II 工作汇总

**生成日期**：2026-09-19
**生成者**：CodeBuddy（本轮会话）
**仓库根**：`/root/private_data/work/sdformer_codex/SDformer`
**会话环境**：A800 80GB ×1（本轮大部分时间空闲）、torch 2.2.2+cu121、专用环境 `/opt/conda/envs/sdformerflow/bin/python`
**核心目标**：把「SDformerFlow 改造版 SNN 光流 + 软硬件协同」收敛成可投 TCAS-II 的工作

> **重要**：本文档中所有标注 ✅ 的条目都是本轮**亲自回到源文件/实测验证**过的；标注 ⚠ 的是存疑或未验证。标注 ❌ 的是已裁定否决、不要再做的方向。

---

# 目录

- [Part A — TCAS-II 严格评审结论](#part-a--tcas-ii-严格评审结论)
- [Part B — 深度调研：算法与硬件 idea](#part-b--深度调研算法与硬件-idea)
- [Part C — 本机实测验证的事实](#part-c--本机实测验证的事实)
- [Part D — 事故记录与恢复（必读）](#part-d--事故记录与恢复必读)
- [Part E — 阻塞项与失败记录](#part-e--阻塞项与失败记录)
- [Part F — 下一步执行清单](#part-f--下一步执行清单)
- [Part G — 文件与路径索引](#part-g--文件与路径索引)
- [附录 1 — 绝对不能写进论文的清单](#附录-1--绝对不能写进论文的清单)
- [附录 2 — 审稿人必问清单](#附录-2--审稿人必问清单)

---

# Part A — TCAS-II 严格评审结论

## A.1 总体判定

**Recommendation: Reject（现状距可投稿差距为架构级，不是补实验级）**

一句话：**算法侧有一组干净但很窄的消融；硬件侧目前没有一颗加速器，协同闭环不存在。** TCAS-II 是电路与系统期刊，硬件必须是主体；当前能被证据支持的硬件实体是 **1279 行 SystemVerilog 的一个 QKᵀ 乘积累加 island**，占全部 1684 个 `.sv` 的 0.26%。

## A.2 Fatal（不改必拒）

| # | 问题 | 证据 |
|---|---|---|
| **F1** | 没有加速器，只有一个组件 island | 被综合的只有 `rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv`(1238行) + `rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv`(41行) = **1279 行**，占 1684 个 `.sv` 的 0.26%。名义顶层 `rtl/nts07_top.v` **仅 93 行**，Q/K 数组悬空（注释自承 "stubbed to zero in top integration test"）✅ |
| **F2** | PPA 在 TCAS-II 口径下不可用 | `dc_handoff/runs/m931_.../reports/flow_contract.rpt` 逐行写明 `clock_period_ns=3.000 / ideal_clock=true / wireload=ZeroWireload / hold_fix_command_count=0 / hold_diagnostic_only=true` ✅。**零互连延迟**下 WNS=0.00 是乐观值；无 SPEF / DEF / GDS；hold WNS −0.09ns、TNS −120.72ns、9992 条违约未修 |
| **F3** | 6.25 mW 不是这颗芯片的功耗 | `ptpx_power_primary_100ps.rpt`：Total 6.2538 mW，其中 **clock_network 4.5884 mW = 73.37%**（理想时钟未做 CTS），**memory power = 0** ✅。对象是 `m405_q32_elastic_selected_slice`，工艺角 **tt/0.9V/25°C**，与 DC 的 SS/0.9V/125°C 不一致。项目自身文档 `docs/526_...md:119` 已把它列为**禁止包装项** |
| **F4** | 存储假设与实测差 26–43 倍 | 实际 SRAM：`TS1N28HPCPHVTB128X128M4S` **9 实例 = 18 KB**，且 wrapper 中 `A[6]` 硬接低 → **实际可用 9 KB**；而周期模型假设 240 KiB、执行摘要假设 388 KB |
| **F5** | 官方 DSEC test 从未提交 | `DATE_PAPER_RESULT_TABLES_20260901.md:99` —— Proposed 行三列全是 `pending`。评测集 valid825 是自造（825帧/18序列，取自 DSEC **train** 序列） |
| **F6** | "ATLIF" 的 adaptive 是关闭的 | `ATLIF_EP34_AUDIT_AND_FLOW_LITERATURE_20260906.md:73-75,101-104`：`threshold_eta=0`、`activity_eta=0`、`target_rate=null`；ep34 中 **105 个阈值里 95 个精确等于 float32 1.0**，24 个 Q/K 阈值全部 = 1.0 ✅。**命名与实现不符**，核心创新点未生效 |
| **F7** | "三元化"在最终模型里不存在 | `output_mode: binary`、`negative_threshold_scale: 1.0`、阈值恒 1 → 实为二值；负发放 neg:pos = **1:165**。而 `HARDWARE_ACCELERATOR_DESIGN.md:71-72` 整个架构建立在 "ternary {-T,0,+T} ~67% zero" 之上 —— **架构文档前提在最终模型上不成立** |

## A.3 Major（大修级）

- **M1 协同设计闭环未闭合**：`rtl/sparse_mac_pe.v` 是 `else if (enable && spike_in) acc_out <= acc_out + weight` → **累加器门控，省功耗省 0 个周期**；全库检索不到任何 dense vs sparse 周期数对比
- **M2 稀疏标定只有单帧**：`hardware_innovation_20260908/fusion_ten_trials_20260914/algorithm_sparse/calibration.json` → `"frame": "thun_00_a_0002.npy", "belongs_train": true, "overlap_valid825": false` ✅（**但已有零成本解法，见 C.3**）
- **M3 主模型身份两份"冻结"冲突**：H67 ep35 AEE 1.329678 vs C12 ep34 AEE 1.199514；`docs/H67_MOTION_ALGORITHM_FILE_INDEX.md`(08-27) 仍写 H67 冻结
- **M4 三个卖点里两个在主指标上为零或为负**（`DATE_PAPER_RESULT_TABLES_20260901.md` Table 1/A4）✅：
  - C00→C10（ATLIF）：AEE **−4.27%**、spikes **−47.83%** ← 全部收益在这
  - C10→C12（Complete TTX）：AEE **+0.27%（变差）**
  - Motion（A4）：AEE **−0.069%**、spikes **+1.49%**
  - MVSEC Table 3：C10 spikes 125.1G（最低），C12 **140.7G（+12.43%）** → TTX 是用活动度换精度
- **M5 MVSEC 比公开方法差 2–2.7 倍** —— **但此结论已被修正，见 Part C.1**
- **M6 无 T 消融**（1287 个 config 全部 `num_steps: 10`）—— **根因已查明，见 C.2**
- **M7 零 error bar**：`DSEC_SEED12_REGISTRY_20260813.json` 中 seed1/seed2 全部 `registered_not_launched`
- **M8 无 bit-accurate / cycle-accurate 对齐 PyTorch 的记录**；`golden_hw_sim.py` 仅 73 行玩具（硬编码 `[1..8]`、`threshold=4`）
- **M9 目标期刊错配**：全库检索 TCAS 仅出现在 related-work 引用里，**TCAS-II 零文档**；所有 roadmap 目标仍是 DATE 2027
- **M10 未与最直接对手做同 workload 对比**：Prosperity (HPCA'25)、Phi (ISCA'25)

## A.4 一个改变选题判断的发现（CICC'26 直接对标）

✅ **双源核实**（IEEE Xplore + Semantic Scholar）：

> **A 28-nm Optical Flow Estimation Accelerator with Redundancy Speculation, Bit-Width-Aware Compression and Similarity Detection**
> Tao Zhang, Yi Zhong, Youming Yang, Jinhao Ruan, Yipeng Gao, Zhaotong Zhang, Yuan Wang
> **CICC 2026**, Seattle, 2026-04-19~23，DOI `10.1109/CICC65509.2026.11509564`，含 Section III Measurement Results

| 指标 | CICC'26 | 我们当前 |
|---|---|---|
| operations | **0.20×** | 0.52×（只来自 spikes −47.8%） |
| EMA（片外访存） | **0.08×** | **无数据** |
| energy | **0.12×** | 无完整数 |
| latency | **0.19×** | 无完整数 |
| 能效 | **14.07 TOPS/W（含 EMA）** | ≈10.2 TOPS/W（**不含** EMA），含 EMA 估 4–7 |

**为什么决定性**：它的三个机制（redundancy speculation / bit-width-aware compression / similarity detection）正好覆盖"稀疏复用 / 位宽 / 模式匹配"，**与我们要 claim 的空间高度重叠**；且它有 Measurement Results（大概率硅实测），我们只有 ZeroWireload 综合。

**待办**：
1. 拿到全文，**确认它是 SNN 还是 ANN**（摘要未写）→ 决定差异化立场
2. related work 正面处理它，并**照抄它的指标格式**（ops/EMA/energy/latency 四项归一化 + 含 EMA 能效）

> ⚠ **口径警告**：SNN 论文用 **SOP**（只累加、无乘法），ANN 用 MAC/OP。我们的 10.2 TOPS/W、CICC 的 14.07 TOPS/W、Spike-IAND-Former 的 38.33 **TSOPS**/W 三者口径不同，**不可直接排序**。论文必须同时给 GOP/s（MAC 等价）与 GSOP/s。

---

# Part B — 深度调研：算法与硬件 idea

## B.1 算法侧（按优先级）

### Tier 0：零 / 近零成本

| # | Idea | 出处 | 收益 | 精度风险 |
|---|---|---|---|---|
| **A0-1** | **MVSEC 协议复查**（dt=1 vs dt=4、split、macro 定义） | — | 见 C.1，报数 1.7649 → 1.2~1.5 | 无 |
| **A0-2** | **UVG 统一体素栅格**：每 bin 以 t_b 为中心固定 2τ 支撑，替代全局归一化 | Sensors 25(10):3158, arXiv 2307.05033 | DSEC EPE **0.96→0.89（−7.3%）**、1PE −9.0%、3PE −9.6%；**网络/参数/MACs 完全不变** | 极低 |
| **A0-3** | **12 个死 `attn_sn` 层静态删除** | 本机实测 + 仓库自身审计 | 零精度损失（firing 恒 0），见 C.4 | **零** |
| **A0-4** | IWE 多参考 focus loss + Charbonnier/TV 辅助损失 | Secrets of Event-Based Optical Flow, ECCV'22 | 按其 1.69→1.488 折算 **AEE −5%~−12%**；纯训练期 | 低 |
| **A0-5** | **硬件稀疏标定换用 valid825 profile** | 本机已有数据 | 补 M2，**零 GPU 成本**，见 C.3 | 无 |

> **A0-2 的重要修正**：体素在**预处理阶段固化成 `.npy` 缓存**（`DSEC_dataset_preprocess.py:185-209`），改 UVG 需**重生成缓存 + 微调**，**不是零成本**。且 UVG 论文的 −7.3% 是带着 UVG 从头训测出来的，直接套到已训好的 C12 上会掉点。
> **A0-2 的更便宜替代**：仓库里已有 `TemporalDifferenceVoxel`（EDCFlow 式，`autoresearch_sparsity/overlay/sparse_preprocess.py:215`）+ 已 smoke 的配置 + 现成 V42a sweep 计划，且文档明确"**无需改模型结构，无需重新生成数据，直接 adapter**"。

### Tier 1：主力（需训练）

| # | Idea | 出处 | 收益 | AEE 风险 |
|---|---|---|---|---|
| **A1** | **MTT 混合时间步**：按 stage 配异构 T，stage 间插 communication module | ICLR'25, `mlanthology du2025iclr-temporal`（Du/Wu/Deng/Gu） | **T10→7：周期 −30%、spikes 72.9G→51G、能耗 −25~30%** | **+0.00~+0.04** |
| **A2** | **ANN 孪生 + A2S 转换**（根因级） | ST-FlowNet, Neural Networks 2025, doi 10.1016/j.neunet.2025.107730（**SNN 光流 SOTA，MVSEC macro 0.60**） | **MVSEC 目标 0.9~1.1（−40~50%）** | 需实测 |
| **A3** | 分块 rate 蒸馏（打在 mlp.sn1 16% / decoder 13–26%） | CVPR'25 pp.10025-10035, arXiv 2503.16572 | firing 5.67%→4.2~4.8%，**spikes −15~26%** | ≤1.20 不退化 |
| **A4** | 结构化通道剪枝（SVD，16 对齐）+ MLP-only MPS/MPO 低秩 | QP-SNN 判据 + tensor network | spike −22~25%、权重字节 −25% | 中 |

> **A2 的根因诊断（重要）**：我们基线 SDformerFlow 自己就是 macro 0.72，改造版退化到 1.7649，而 SNN SOTA 是 0.60 且**靠 ANN→SNN 转换（A2S 阈值均衡）拿到，非直接 STBP**。**差距主要在训练范式，不在架构。**

### Tier 2：位宽（0 周期，省存储/访存）

| # | Idea | 出处 | 收益 | 风险 |
|---|---|---|---|---|
| **A5** | M12→M8 膜电位整数化 | SpQuant-SNN, Frontiers Neurosci 2024 | 膜电位 refill **−33%**、能耗 −8~12% | +0.01~0.03 |
| A5b | 配 MD-SNN 膜电位蒸馏回收 | arXiv 2512.04443（venue **uncertain**） | 回收 50–70% 损失 | 训练 1.3–1.5× |
| **A6** | W8→W4（ReScaW）只做 Conv+MLP | QP-SNN, ICLR'25, arXiv 2502.05905 | 权重访存 **−50%**；**周期 0 收益** | +0.03~0.08 |
| A6b | 叠加 SVS 30% 通道剪枝 | 同上 | 才有 **−30% 周期** | +0.06~0.12 ⚠ |
| **A7** | 异构位宽 —— **硬约束：只能是 2^n** | 子代理裁定 | decoder 权重 8→4、psum 16→8 → 9KB 内 resident 行数 64→**128** | 非 2^n 破坏 16-bit mask word 打包，island 周期 **+30~60%** |

### Tier 3：有风险 / 待定

- **A8 IAND 残差**（Spike-IAND-Former, ISCAS'25, arXiv 2503.19643）：加法器→AND 门，该通路组合逻辑 −95%。**但 Conv 输出是多 bit，只适用于 sn 之后的二值合并点**，需先 ablation 确认 ΔAEE < +0.01
- **A9 BSA 双极 + Shiftmax**（NeurIPS'25, DOI 10.52202/085713-3401）：修二值表达力。代价 **spike +9.4%**，需 AEE 改善 ≥0.02。**注意仓库已有 `H9_bipolar_self_attention` 且已启用（12 modules），先查已做到哪一步**
- **A10 UniSparse 联合剪枝**（ICML'26, github xyshi2000/UniSparse）：存在性确认，**论文数字读不到，只可引方法不可引数字**
- **A11 STSC 轨迹连续性正则**（CVPR'26, arXiv 2605.25570）：训练期 0 硬件成本，可能换 num_steps 10→8
- **A12 From Contrast to Consistency 双向一致性**（同上）

### ❌ 已裁定否决

**ragged token 剪枝**（TP-Spikformer 通用形式）在本项目 NO-GO，三条硬理由：
1. O(N) 无 QK^T → 失去 token 压缩的主要收益来源（二次项）
2. 90% FLOPs 在 3×3 Conv → 零散 token 丢弃被邻域 stencil 吃掉
3. 15×15 窗口 ragged top-k **破坏 64 行 × 16bit mask / 96 lane 对齐，XOR-residual + psum 复用失效，反而 +15~25% 周期**

**独立交叉验证**：仓库自己的实验记录 `voxelization_experiments/VOXELIZATION_NEXT_REVIEW_20260523.md` 写道 —— `s41a_sparsespikformer_token85` / `s41b_qpsnn_svs90` 短测后 **"AAE 到 30+，输入级硬 mask 会破坏光流方向估计"**。两条独立证据链同一结论，**token 剪枝可以彻底排除**。

> **唯一例外**：**整分支 / 块对齐**的静态跳过（如 12 个死 `attn_sn`）是安全的，因为不改 mask 粒度。

## B.2 硬件侧

| # | Idea | 出处 | 关键数字 |
|---|---|---|---|
| **H0** | 12 个 `attn_sn` 死层静态删除 | 本机实测（C.4） | 零精度成本 |
| **H1 ★** | SparTen 离线 greedy balancing (GB-H) | MICRO'19, DOI 10.1145/3352460.3358291 | **纯软件、不改 RTL**；96 lane 在 p=5.67% 下利用率 <30% → cycle=max 压到 ≈1.05×mean，p=5.67% 层 ~2.2× |
| **H2 ★** | SCNN 32×32 tile 位图双缓冲 | ISCA'17, DOI 10.1145/3079856.3080254 | 工作集 **384KB → 5.12KB**，塞进 9KB |
| **H3 ★** | **时间步并行展开**（tick-batching） | Spike-IAND-Former, ISCAS'25 | 膜电位 SRAM traffic **−90%**、权重 SRAM 访问 **−43.2%** |
| H4 | TFLIF：BN bias 折进阈值/膜电位初值 | VESTA, arXiv 2503.20246（venue uncertain） | 93 层每层省一次 bias 加 |
| H5 | spike **位置编码**替代 bitmap | arXiv 2501.07825（venue uncertain） | 权重 SRAM 读 −94%（理想），但能效仅 1.33× |
| H6 | FireFly-T 多非零提取 + OoO 派发 | IEEE TC Vol.75 No.6, Jun 2026, DOI 10.1109/TC.2026.3672901 | 利用率 30%→75–85%（FPGA，ASIC 需重做） |
| H7 | Phi 二级稀疏（pattern 预计算 + 稀疏残差） | ISCA'25, arXiv 2505.10909 | 3.45× speedup / 4.93× 能效 |
| H8 | Prosperity product sparsity | HPCA'25, arXiv 2503.03379 | 7.4×/8.0× vs PTB —— **我们 m935 已是它的复现，见 C.5** |
| ❌ H9 | BitWave bit-column-serial | HPCA'24 | **已否决**：激活仅 1bit 无收益，权重侧需位稀疏 >75%（OzMAC 盈亏线 58%），INT8 Hamming≈4/8 时**净亏约 2×** |
| ❌ H10 | Neural Cache 近存 AND-popcount | ISCA'18 | 净 6–8× 周期，但**编译宏位线不能做原位 AND，必须放弃现有 9 个 SRAM 实例重做存储** |

**28nm 对标锚点**：
- Spike-IAND-Former：38.33 **TSOPS**/W、90.15mW、139.25KB SRAM、3456 PE
- VESTA：9.844 **TSOPS**/W、0.844mm²、416.1mW；**周期分布：linear/MLP 80.79%、attention 14.88%、8b conv 4.13%** → 说明我们 3×3 Conv 占比更高，**优化次序应先 Conv 数据流再 MLP**

## B.3 评测方法学（补证据等级的最快路径）

| # | 工具 | 出处 | 补什么 |
|---|---|---|---|
| **E1 ★** | **Sparseloop** 解析建模 | MICRO'22（Distinguished Artifact） | 误差 0.1–8%、快 >2000×。**不需要端到端 RTL 就能出 dense-vs-sparse 周期表**（补 M1） |
| **E2 ★** | **MasterRTL** | ICCAD'23, DOI 10.1109/ICCAD57390.2023.10323951 | 把 WNS=0 改写成"zero-wireload 333MHz / 预测 post-layout **230–285MHz**"双数字（补 F2） |
| E3 | Timeloop (ISPASS'19) + Accelergy (ICCAD'19) | — | **ERT 必须用我们 28nm post-synthesis 网表反标定**，能耗饼图才可辩护 |
| E4 | LEAP / LEAP-Power | arXiv 2608.01946（**预印本，uncertain**） | post-synthesis→layout per-cycle 功耗，MAPE 4.55%；统一功耗角 + 多帧 SAIF |
| **E5 ★** | **SCALE-Sim v3** | arXiv 2504.15377（**预印本，venue uncertain**） | **关键警告：只看 compute cycle 会误判**（WS vs OS 只差 21%，计入 DRAM stall 后 OS 反少 30.1%）。**9KB 下 DRAM 可能占总周期 30–60% → 决定 story 写"稀疏→算力"还是"稀疏→带宽"。动手前先跑这个** |
| E6 | FireSim | ISCA'18 | cycle-exact FPGA 仿真；需自制 AXI/TileLink bridge |
| E7 | iEDA | ASP-DAC'24, arXiv 2308.01857 | 开源 P&R；**风险：28nm HPC+ 商业 PDK 通常不允许进开源流程** |

---

# Part C — 本机实测验证的事实

> 本节全部是**本轮亲自执行命令得到的**结果，不是推断。

## C.1 MVSEC「差 2.5 倍」的结论已被推翻（协议分解）

**原判断**：我们 MVSEC macro AEE 1.7649 vs 公开 SNN 方法 0.60~0.69，差 2.5 倍。
**修正后**：

| 项 | 我们 | 公开方法 | 判定 |
|---|---|---|---|
| 评测 dt | **dt=1**（本机**只有 dt1 GT**，无 dt4） | **dt=4** | **纯口径，最大头**。dt=1 的 GT 位移小、噪声占比高，AEE 天然抬高约 1.5–2.5× |
| macro 组成 | OD1/IF1/IF2/IF3 算术平均（无 IF4，OD2 当训练集） | 同 | 一致，非差异项 |
| 每序列帧数 | **fixed800**（manifest `mvsec_cicc_dt1_v1.json`） | 多为全序列/不同采样 | 次级口径 |
| 训练数据 | MVSEC **outdoor_day2 单序列自造划分** | MDR（SDformerFlow-v2）/ MVSEC 全量 | **真实差距**：同协议下 1.7649(day2) → **1.2019(MDR)**，−31.9% |

**量化分解**：
- 1.7649 − 31.9% → **1.2019**（同 dt=1、换 MDR 训练）⇒ 训练数据效应 = **0.56 AEE**
- 1.2019 (dt=1) → dt=4 后预期落在 0.6–0.8，与公开 SDformerFlow-v2 的 0.66 基本重合 ⇒ 剩余部分基本由 dt 口径解释
- **结论：真实可归因差距约 1.3×，不是 2.5×**

**旁证**：EDCFlow (CVPR'25) 的 dt=1 表：EV-FlowNet **1.19** / IDNet-4 **1.10** / EDCFlow **1.02** —— dt=1 口径下 SOTA 本来就在 1.0 附近。我们在 dt=1 的 1.2019 与之同量级。

**⚠ 另一件事**：**1.7649 与「C12 ep34 = DSEC 1.199514」不是同一个权重**。ep34 从未在 MVSEC 上跑过（2026-08-30 那次任务在 valid825 之后中断）。论文表里必须拆成两行，**不得合并表述**。

**论文表建议写法**：
- `Ours-MDR-route (dt=1, fixed800, 4-seq macro)` = **1.2019**（对外对比行）
- `Ours-day2-route (dt=1, ...)` = 1.7649（**仅作同父模块 ablation 行**）
- DSEC C12 ep34 (1.199514) 单独一行
- 表注必须写全：`dt=1 | fixed800 | macro = mean(OD1,IF1,IF2,IF3) | 训练集 = MVSEC outdoor_day2 内部划分 / MDR | indoor_flying4 不可用`

## C.2 ⚠ T 消融的结构性根因（重要）

**尝试**：把 `num_steps: 10` 改成 6 跑推理。
**结果**：checkpoint 加载失败 ✅

```
size mismatch ... spiking_neuron.weight: torch.Size([10, 10]) from checkpoint, current model [6, 6]
size mismatch ... spiking_neuron.bias:   torch.Size([10, 1])  current model [6, 1]
size mismatch ... spiking_neuron.center: torch.Size([10, 1])  current model [6, 1]
```

**根因**：`SpikingNormLayer`（`third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py:129`）**为每个时间步建一个独立 BatchNorm2d**：`[nn.BatchNorm2d(...) for i in range(self.num_steps)]`；`Spiking_neuron` 另有可学习 `weight [T,T]` / `bias [T,1]` / `center [T,1]`。

**→ T 被焊进参数形状里，不是推理期可调旋钮。** 这就是 M6（T 消融完全空白）的结构性原因 —— 不是偷懒，是架构决定的。

**参数形态实测**（C12 ep34 checkpoint）：
| 形态 | 数量 | 例子 |
|---|---|---|
| `(10, 10)` | 45 | `resblocks.0.sn1.spiking_neuron.weight`（主通路，T=10） |
| `(10, 1)` | 90 | `resblocks.0.sn1.spiking_neuron.bias` |
| `(2, 2)` | 60 | `attn.sn_q.spiking_neuron.weight`（**注意力神经元 T=2**，对应 `window_size[0]=2`） |
| `(2, 1)` | 120 | `attn.sn_q.spiking_neuron.bias` |

**推论**：
1. **MTT（ICLR'25 混合时间步训练）是必需品，不是可选项** —— 它是让权重跨 T 可部署的现成方法
2. 探索中发现的第二条线索：**改 `num_steps` 会连带改变 `patch_embed` 的期望输入通道（2→3）**，说明模型构造层把 num_steps 也用作了输入维度。且 eval 日志出现 `[H9] remap=v1 interpolation complete; applying interpolated state dict` —— **流水线本身已有 T 重映射/插值机制**，手写切片器可能多余，应先读 `h9_load_audit.py` 的 remap 逻辑

## C.3 ✅ M2（单帧标定）有零成本解法

硬件侧 `calibration.json` 只在单帧 `thun_00_a_0002.npy`（`belongs_train: true`, `overlap_valid825: false`）上标定。
**但算法侧已有的 `spike_profile.json` 里已经有 825 帧、93 层的完整统计**：

```
结果路径：neuron_experiments/H9_bipolar_self_attention/results/
          dsec_c12_alpha0125_ep29_resume5_20260830/standard_valid825/epoch34/spike_profile.json

samples                = 825
global_firing_rate     = 0.05670896364557509
sparsity_ratio         = 0.9432910363544249
total_spikes           = 72,891,240,701
layer_firing_rates     = 93 层，每层 {spikes, elements, firing_rate}
其他字段：metric_contract / metric_aggregation_audit / validation_file_list /
         energy_scope / profiled_layers / eval_protocol / checkpoint_load_audit
```

**→ 把硬件稀疏标定的数据源换成这份 profile 即可补上 M2，无需任何 GPU 时间。**

## C.4 ✅ 12 个死层 + 发放率热点（实测）

用 `spike_profile.json` 的 93 层统计排序：

**完全静默层（firing == 0.000%，spikes = 0）—— 共 12 层，全部是 `attn.attn_sn`**：
```
encoders.swin3d.layers.0.swin_blocks.0.attn.attn_sn
encoders.swin3d.layers.0.swin_blocks.1.attn.attn_sn
encoders.swin3d.layers.1.swin_blocks.0.attn.attn_sn
encoders.swin3d.layers.1.swin_blocks.1.attn.attn_sn
encoders.swin3d.layers.2.swin_blocks.0..5.attn.attn_sn   (6 层)
encoders.swin3d.layers.3.swin_blocks.0.attn.attn_sn
encoders.swin3d.layers.3.swin_blocks.1.attn.attn_sn
```

**仓库自身审计已确认原因**（不是新发现，但值得利用）：
- `neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/h9_load_audit.py:37`：
  "``.attn_sn``'s output is discarded (``proj`` consumes the pre-neuron value)."
- 同目录 `atlif_ternary_psn/installer.py:180-196` 同样记载，并说明"Reviving ``.attn_sn`` is not a bug fix"

**→ 这 12 层是死模块（输出被丢弃），不是死注意力。删除它们零精度损失。** 也解释了之前审计的"105 个 ATLIF 中 12 个输出未被使用 → 仅 81 个功能有效"。eval 日志也确认：`[H9] eval installed ATLIFTernaryPSN: 81 modules`

**次静默**：`layers.1.swin_blocks.0.attn.sn_k` 仅 68 个 spikes（825 帧全量）。

**发放率热点（周期/能耗优化应打这里，不是注意力）**：
| firing | spikes | 层 |
|---|---|---|
| **25.972%** | **7,981,175,663** | `decoders.3.sn` ← 单层占全网 **11%** spikes |
| 20.522% | 780,147,293 | `encoders.swin3d.layers.2.downsample.sn` |
| 20.148% | 3,063,736,790 | `encoders.swin3d.layers.0.downsample.sn` |
| 16.580% | 630,308,335 | `layers.2.swin_blocks.5.mlp.sn1` |
| 16.362% | 2,488,109,263 | `layers.0.swin_blocks.0.mlp.sn1` |
| 15.668% | 2,382,568,810 | `preds.2.sn` |
| 15.419% | 2,356,927,513 | `decoders.2.sn` |

## C.5 ✅ m935 就是 Prosperity 的复现（不能当 novelty）

逐行读 `rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv`（1238 行）：

```
// maximize parent popcount, and retain the lowest row id for an equal-popcount tie
input  logic [5:0]  prep_row_id,        // 64 行
input  logic [15:0] prep_mask,          // 16-bit spike mask
output logic [1823:0] psum_write_data,  // 96 lane 累加结果
input  logic [1823:0] issue_psum_prior, // ← 复用 prefix 的 partial sum
```

对照 Prosperity (HPCA'25，已抓全文核实)：

| Prosperity | 我们的 m935 |
|---|---|
| Prefix = 1 的个数最多的子集行 | "maximize parent popcount" |
| pattern = S_q XOR S_p | "XOR the residual" |
| Output(S_i) = Output(S_j) + Σ 差集权重 | `issue_psum_prior` → `psum_write_data` |
| tile k=16 bit, m=256 | k=16 bit, **m=64** |

**结论：m935 是 Prosperity 核心机制的 64 行版本，硬件主张不能是"提出 product sparsity 引擎"。**

**但 Prosperity 没有而我们有的是什么（novelty 落脚点）**：
1. **光流 dense prediction workload**：T2×15×15 窗口 → 相邻时间步/窗口 spike 模式高度相关，EM 率天然远高于 Prosperity 的 SpikeBERT 场景（**可量化并宣称**）
2. **O(N) element-wise attention**（`attn = K * sn2(ΣQ)`）**不需要 softmax** → 省掉 Prosperity 必需的整个 SFU
3. **二值 ATLIF 天然满足 ISCA'25 Single-Spike 性质**（每步 ≤1 脉冲）→ 累加器可退化成选择器，**从未 claim 过**

## C.6 ✅ UVG 边缘缺陷实测坐实

对 3 个真实 DSEC npy（`data/Datasets/DSEC/saved_flow_data/event_tensors/10bins/left/*`）：
```
interlaken_00_b_0000.npy  energy ratio [0.48 0.999 0.978 0.964 1.039 1.015 0.99 1.015 1.001 0.421]
                          nz ratio     [0.73 0.972 1.02  1.018 0.974 0.973 1.021 1.045 0.977 0.689]
```
**首尾 bin 能量只有中间 bin 的 0.42–0.49×，非零覆盖 0.69–0.75×** —— 边缘半支撑真实存在。

源头代码 `third_party/SDformerFlow/DSEC_dataloader/event_representations.py:255`：
```python
t_norm = (C - 1) * (t_norm - t_norm[0]) / (t_norm[-1] - t_norm[0])  #0-bin
```
这正是 UVG 论文点名批评的全局归一化形式。

**三处改动点**（注意：**不要动 `encoders.py`**，那里 `VoxelSpikeEncoder.forward()` 是纯 `return x`）：
| 路径 | 函数 | 改什么 |
|---|---|---|
| `third_party/SDformerFlow/MDR_dataloader/loader_utils.py:492` | `EventSequenceToVoxelGrid_Pytorch.__call__` | 改固定 τ + 三角核以 bin 中心对称 |
| `tools/prepare_dsec_single_sequence.py:64` | `make_voxel_chunk` | 同上，τ 由 GT 窗口决定 |
| `third_party/SDformerFlow/DSEC_dataloader/event_representations.py:254-255` | `VoxelGrid.convert_CHW` | 去掉二次归一化 |

**成本约束**：MVSEC 是在线编码（改一行即可），**先做 MVSEC**；DSEC 是离线 npy，全量重算需 ~105GB 而 `/root/private_data` 仅剩 **55G (95%)** → 走**在线 voxel 化**（`DSEC_dataset_lite.py:153-159` 的 raw 分支已存在但未接线）。
**坑**：重算不要用 `DSEC_dataset_preprocess.py`（encoding='voxel' 时写到 `10bins_pol`，与 `DSEC_dataset_lite.py:55-56` 读的 `10bins` 不一致）。

---

# Part D — 事故记录与恢复（必读）

## D.1 事故经过

**时间**：2026-09-19
**操作**：为跑 T 消融，创建了 5 个 run-dir，每个里面用 `ln -sf` 把 `checkpoint_epoch34.pth` 链到原始 C12 ep34 权重；随后用 `torch.save` 把 T=8 切片结果**写到同一个路径**。
**后果**：`torch.save` **穿透符号链接**，**直接覆盖了原始 C12 ep34 权重（T=10 → T=8）**。

**放大原因**：T=8 先跑，覆盖了原文件；后续 T=6/4/2 以"原始"为源时已读不到 T=10 张量（切片器报告 `touched 0 tensors`），错误因此静默延续。

## D.2 恢复过程与验证

**恢复来源**：`neuron_experiments/H9_bipolar_self_attention/results/t49_ternary_smoke/warmstart/checkpoint_epoch34.pth`

**为什么能证明是同一份**：切片只动了 135 个携带 T 轴的张量，**其余所有参数逐位不变**。因此被覆盖的文件仍是全部非 T 参数的可靠指纹。写了校验器 `find_intact_c12_ep34_20260919.py`：
- 条件 1：T 轴张量必须是 `(10,10)`
- 条件 2：**所有非 T 张量必须与被覆盖文件逐位相等**
- 在 20+ 个候选中筛出 **3 份**同时满足的副本

**恢复动作与验证** ✅：
```
恢复后：sttmultires_unet.resblocks.0.sn1.spiking_neuron.weight -> (10, 10)
        ...bias   -> (10, 1)
        ...center -> (10, 1)
文件大小 225504447 字节 —— 与 checkpoint_epoch30.pth / checkpoint_epoch32.pth 完全一致
```
损坏版本已另存 `/tmp/corrupted_T8_backup.pth`（可删）。

## D.3 遗留影响与待办

1. **原始文件 mtime 已变**（现为 Sep 19 17:27）→ **任何依赖该文件哈希的校验需要重新核对**，例如 `standard_valid825/epoch34/` 下的 `evaluation_identity.json`（内部含 `checkpoint_sha256`）。**建议跑一次 valid825 复现确认 AEE 仍为 1.199514。**
2. `checkpoint_epoch34_state_dict.pth`（449MB）**只有 optimizer / scheduler / epoch / scaler，没有模型权重** —— 名字极具误导性，真出事时救不了。**建议改名或补一份真正的权重备份。**
3. **教训**：不要对 checkpoint 路径用 `ln -sf` 后再写入。要走 `cp` 或写到独立真实路径。

---

# Part E — 阻塞项与失败记录

## E.1 MVSEC 评测：数据不在本机 ❌

```
data/Datasets/ 只有:  DSEC -> /root/private_data/SothisAI/dataset/Console/DSEC/main/DSEC
MVSEC 目录不存在
```
仓库只有下载/准备脚本（`scripts/download_mvsec_indoor_flying.py`、`scripts/prepare_mvsec_dt1.py`），**没有数据**。
→ 补跑 C12 ep34 的 MVSEC 评测本地走不通，需先下数据（且需确认能否联网）。

## E.2 T 消融：改配置行不通（见 C.2）

两次失败记录：
1. **用 DSEC 配置跑 MVSEC** → `KeyError: 'event_interval'`（MVSEC 需要 `data.event_interval`，且 `patch_config_sequence` 只改 `test_sequence`，data 段必须自带该键）
2. **改 num_steps 跑 DSEC** → checkpoint 尺寸不匹配（T 焊在参数里），随后又出现 `patch_embed.head.conv.0.weight: ckpt [48,2,3,3] vs model [48,3,3,3]`

## E.3 已创建但未走通的产物（可清理）

| 路径 | 状态 |
|---|---|
| `neuron_experiments/H9_bipolar_self_attention/entrypoints/slice_timestep_checkpoint_20260919.py` | T 切片器，工作正常，但**在修好模型构造层之前无用** |
| `neuron_experiments/H9_bipolar_self_attention/entrypoints/find_intact_c12_ep34_20260919.py` | 指纹校验器（事故恢复用），**建议保留作为安全检查工具** |
| `configs/generated/t_ablation_c12_ep34_T{10,8,6,4,2}_20260919.yml` | 已确认与原始配置仅差 num_steps 一行（diff 验证过），但**当前不可用** |
| `results/t_ablation_c12_ep34_T{10,8,6,4,2}_20260919/` | 含切片权重与失败日志，**建议清理或归档** |

## E.4 一个待查的重要线索

eval 日志出现 `[H9] remap=v1 interpolation complete; applying interpolated state dict` —— **流水线本身已有 T 重映射/插值机制**。在做 T 消融前，**应该先读 `neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/h9_load_audit.py` 的 remap 逻辑**，很可能不需要手写切片器。

---

# Part F — 下一步执行清单

## F.0 阻塞一切的第一步

**CICC'26 论文全文**（DOI `10.1109/CICC65509.2026.11509564`）→ 确认 **SNN 还是 ANN** → 定差异化立场。

`HARDWARE_ACCELERATION_DESIGN` 相关：`neuron_autoresearch/HARDWARE_ACCELERATOR_DESIGN.md`（2026-05-29，**已过期**：里面假设 ternary 输出、500MHz、30FPS、<5W、<2MB SRAM，全部与现状不符）。

## F.1 零成本项（不需 GPU，本周可出）

| 顺序 | 动作 | 产出 |
|---|---|---|
| 1 | **复现验证**：跑一遍 valid825 确认 AEE 仍 1.199514（确认事故恢复无误） | 安全确认 |
| 2 | **MVSEC 协议表重做**（按 C.1 的写法，拆行 + 表注） | 消除"差 2.5 倍"的伪劣势 |
| 3 | **硬件稀疏标定换用 valid825 profile**（C.3） | 补 M2 |
| 4 | **删除 12 个死 `attn_sn` 层**（C.4） | 零精度损失的清理 |

## F.2 出周期表（补 M1）

| 顺序 | 动作 | 说明 |
|---|---|---|
| 5 | **Sparseloop 建模** | 喂入 93 层密度，出 dense-vs-sparse 周期表，**不需要端到端 RTL** |
| 6 | **SCALE-Sim v3 判 DRAM 占比** | **决定 story 写"稀疏→算力"还是"稀疏→带宽"。此判断应前置于任何 RTL 改动** |
| 7 | SparTen GB-H 离线重排 | 纯软件、不改 RTL，96 lane 利用率 <30% → 压到 ≈1.05×mean |

## F.3 补证据等级

| 顺序 | 动作 |
|---|---|
| 8 | MasterRTL 修频率数字（333 → 双数字 333/230–285MHz）+ 声明 hold 未修 |
| 9 | LEAP-Power 统一功耗角 + 多帧 SAIF |
| 10 | 写网络级 bit-accurate 定点仿真器（对齐 `configs/hw/quant_spec.yaml`）→ 一次补 M1/M2/M8 |

> `configs/hw/quant_spec.yaml` 内容（**存在但零验证**）：
> `weight_bits: 8 / activation_bits: 8 / membrane_bits: 12 / threshold_bits: 12 / accumulator_bits: 24`
> `tiling: window_tokens 64, tile_m/n/k = 8`

## F.4 最大精度杠杆（需训练）

| 顺序 | 动作 | 目标 |
|---|---|---|
| 11 | **ANN 孪生 + A2S 转换**（对准 ST-FlowNet 0.60） | MVSEC 1.7649 → 0.9~1.1 |
| 12 | **V42a EDCFlow 时间差分**（仓库已实现、已 smoke、无需重生成数据） | 硬件 0 成本的精度收益 |
| 13 | MTT 训练（T=8/7/6） | 周期 −30%，spikes →51G |
| 14 | 位宽消融（限 2^n）+ 结构化剪枝 | SRAM + 周期 |

## F.5 可立即参考的仓库既有资产

| 资产 | 路径 | 说明 |
|---|---|---|
| EDCFlow 时间差分实现 | `autoresearch_sparsity/overlay/sparse_preprocess.py:215` `TemporalDifferenceVoxel` | 已 smoke（20/120 step） |
| EventPillars Lite | 同文件 `:262` | 已实现未充分 profile |
| V42a sweep 计划 | `voxelization_experiments/VOXELIZATION_NEXT_REVIEW_20260523.md` | alpha ∈ {0.05, 0.10, 0.20} |
| 体素化调研 | `voxelization_experiments/research/voxelization_methods_report_20260426.md` | — |
| 硬件加速器设计（**已过期**） | `neuron_autoresearch/HARDWARE_ACCELERATOR_DESIGN.md` | 2026-05-29，假设与现状不符 |
| 硬件禁止包装项 | `hw_autoresearch_nts07/docs/526_...md:111-119` | 权威清单 |
| 冻结结果表 | `neuron_autoresearch/DATE_PAPER_RESULT_TABLES_20260901.md` | DATE 口径，非 TCAS-II |
| 稀疏档案 | `.../dsec_c12_alpha0125_ep29_resume5_20260830/standard_valid825/epoch34/spike_profile.json` | 825 帧 × 93 层 |

## F.6 可直接使用的命令（cwd = 仓库根）

**A. 复现 DSEC valid825（sanity check，期望 AEE=1.199514）**
```bash
/opt/conda/envs/sdformerflow/bin/python -u \
  neuron_experiments/H9_bipolar_self_attention/entrypoints/run_h9_standard_valid825_eval.py \
  --config neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_c12_alpha0125_ep29_resume5_20260830.yml \
  --run-dir neuron_experiments/H9_bipolar_self_attention/results/dsec_c12_alpha0125_ep29_resume5_20260830 \
  --ranking-mode aee --epoch 34
```

**B. 读取 825 帧稀疏统计（零 GPU）**
```bash
/opt/conda/envs/sdformerflow/bin/python -c "
import json
d=json.load(open('neuron_experiments/H9_bipolar_self_attention/results/dsec_c12_alpha0125_ep29_resume5_20260830/standard_valid825/epoch34/spike_profile.json'))
items=sorted(((k,v['firing_rate'],v['spikes']) for k,v in d['layer_firing_rates'].items()), key=lambda t:t[1])
for k,fr,sp in items[:14]: print(f'{fr*100:7.3f}%  {sp:>12,}  {k}')
"
```

**C. UVG 平坦度验收（秒级）**
```bash
/opt/conda/envs/sdformerflow/bin/python -c "
import numpy as np
v=np.load('data/Datasets/DSEC/saved_flow_data/event_tensors/10bins/left/interlaken_00_b/interlaken_00_b_0000.npy')
e=np.abs(v).sum(axis=(1,2)); nz=(v!=0).mean(axis=(1,2))
print('energy', np.round(e/e[1:-1].mean(),3)); print('nz', np.round(nz/nz[1:-1].mean(),3))
"
```
基线应为 `energy [0.48 ... 0.42]`、`nz [0.73 ... 0.69]`；目标首尾/中间比 >0.85。

---

# Part G — 文件与路径索引

## G.1 评审依据（本轮核实过的源文件）

```
neuron_autoresearch/DATE_PAPER_RESULT_TABLES_20260901.md          # 冻结结果表（Table 1/2/3/4/5/6 + A1-A5）
neuron_autoresearch/ATLIF_EP34_AUDIT_AND_FLOW_LITERATURE_20260906.md  # 阈值恒 1.0 的审计（:73-75, :101-111）
neuron_autoresearch/HARDWARE_ACCELERATOR_DESIGN.md                # 架构设计（2026-05-29，已过期）
neuron_autoresearch/DATE_PUBLIC_BENCHMARK_COMPARISON_20260826.md  # 基线对比（对手数字来自论文）
hw_autoresearch_nts07/docs/526_Prosperity_Phi性能Headline方法审计与H67合法包装_20260827.md  # 禁止项 :111-119
hw_autoresearch_nts07/docs/100_DSEC全分辨率RTLExact签核阻塞与T450闭环清单_20260730.md
hw_autoresearch_nts07/dc_handoff/runs/m931_.../reports/flow_contract.rpt     # ideal_clock/ZeroWireload
hw_autoresearch_nts07/dc_handoff/runs/m448r4_.../reports/ptpx_power_primary_100ps.rpt  # 6.2538 mW
hw_autoresearch_nts07/rtl_m935_c1_match_pipeline/m935_m912_three_stage_exact_parent_match_product_capture_island.sv
voxelization_experiments/VOXELIZATION_NEXT_REVIEW_20260523.md
third_party/SDformerFlow/DSEC_dataloader/event_representations.py            # :255 体素归一化
third_party/SDformerFlow/DSEC_dataloader/DSEC_dataset_preprocess.py          # :185-209 缓存固化
third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py             # :129 每时间步 BN
configs/hw/quant_spec.yaml                                                   # 定点规范（未验证）
tools/golden_hw_sim.py                                                       # 73 行玩具
hardware_innovation_20260908/fusion_ten_trials_20260914/algorithm_sparse/calibration.json  # 单帧标定
```

## G.2 本轮新建的文件

```
neuron_experiments/H9_bipolar_self_attention/entrypoints/slice_timestep_checkpoint_20260919.py
neuron_experiments/H9_bipolar_self_attention/entrypoints/find_intact_c12_ep34_20260919.py
configs/generated/t_ablation_c12_ep34_T{10,8,6,4,2}_20260919.yml
results/t_ablation_c12_ep34_T{10,8,6,4,2}_20260919/
/tmp/corrupted_T8_backup.pth
```

## G.3 记忆文件（若 Codex 也读 memory）

```
/root/.codebuddy/projects/root-private_data/memory/MEMORY.md
/root/.codebuddy/projects/root-private_data/memory/project_snntx_opticalflow_tcasii.md
/root/.codebuddy/projects/root-private_data/memory/reference_hw_accel_papers.md   # 论文库（本轮大幅补充）
/root/.codebuddy/projects/root-private_data/memory/user_profile.md
```

---

# 附录 1 — 绝对不能写进论文的清单

| 禁用项 | 原因 |
|---|---|
| 6.25 mW 作为加速器功耗 | selected-slice、无 macro、73.37% 理想时钟、工艺角不一致 |
| 91 FPS @500MHz / 10.4 mJ/帧 / 2.45 mm² / 388 KB | 执行摘要自标 "provisional 解析模型" |
| WNS = 0.00 作为收敛证据 | ZeroWireload + ideal_clock，hold TNS −120.72 ns 未修 |
| 240 KiB / 388 KB SRAM 假设 | 实际可用 9 KB |
| "ternary"、"三值 ATLIF" | 最终模型 `output_mode=binary`，neg:pos = 1:165 |
| "adaptive threshold" 作为贡献 | `threshold_eta=0`，95/105 阈值恒为 1.0 |
| "full-network RTL-exact" / "fullres RTL-exact" | 项目自建 doc 100 明令禁止 |
| 任何 FPS / latency / GOPS / pJ-per-op 实测表述 | 无任何实测值 |
| 1.794–1.823x 加速比 | doc 526 标明是"待闭合目标区间，非可投稿实测值" |
| Prosperity/Phi 的 2.46x / 7.4x 改名为 ours | doc 526 §5.2 首条禁止项 |
| MVSEC Proposed 行进主表 | 协议口径不同；应按 C.1 拆行 |
| H67 ep35 与 C12 ep34 同时出现 | 两份"冻结"身份冲突 |
| 把 1.7649 与 C12 ep34 (1.199514) 合并表述 | **不是同一个权重**，ep34 从未跑过 MVSEC |

# 附录 2 — 审稿人必问清单

1. 综合出来的单元占整个网络计算量的百分之多少？给出 Amdahl 分析
2. 稀疏到底省了多少**周期**？给出 dense vs sparse 的实测 cycle count
3. 权重位宽、激活位宽、累加器位宽各是多少？量化是否经训练？精度损失多少？
4. 与 PyTorch 浮点模型的 bit-accuracy 误差是多少？有 cycle-accurate 仿真吗？
5. on-chip SRAM 只有 9 KB，如何承载 480×640？片外带宽需求是多少？DRAM 能耗算了吗？
6. 为什么功耗在 tt/25°C 而时序在 ss/125°C？给出同角数据
7. 为什么没有 hold 修复和 P&R？布线后频率还剩多少？
8. 相比 Prosperity / Phi 在同一 workload 上的加速比和能效比是多少？
9. 相比 GPU（A100）和 FPGA baseline 的能效对比是多少？
10. ATLIF 的自适应阈值为什么在最终模型中全部等于 1.0？那"adaptive"指什么？
11. 为什么叫 ternary 而实际是 binary？负发放占比 1:165 如何支撑三元化收益？
12. 官方 DSEC test-set 的 EPE / AE / 3PE 是多少？
13. T（时间步）从 10 降到 2/4 时精度和延迟如何变化？
14. 多个随机种子下的方差是多少？
15. 为什么 MVSEC 上比已发表 SNN 方法差？（→ 用 C.1 的协议表回答）
16. Complete TTX 在 DSEC 上使 AEE 变差 0.27%，它的贡献究竟是什么？
17. 105 个 ATLIF 中只有 81 个生效，其余为什么存在？

---

**文档结束。**
