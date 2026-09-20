# 顶层文献调研 + 新 idea 提炼（r1，2026-09-17）

作者：顶层调研 worker（CPU-only）。产出目录：
`hw_autoresearch_nts07/reviews/idea_survey_2026_r1_20260917/`（本目录为本轮唯一写入点）。

**红线声明**：未运行任何 GPU 命令（`t49_ternary_ws4_ep5` PID 721081 全程未触碰）；未改任何
编号文档（含 `docs/359/362/366`）与冻结合同；未删文件；H82/H86 未重启；未写 RTL、未跑 EDA。
证据分档沿用 `[rtl]` / `[prof]` / `[模型]`；**外部文献一律标 `[lit]`**，其含义见 §0.2。

上游依据：`neuron_autoresearch/ATLIF_EP34_AUDIT_AND_FLOW_LITERATURE_20260906.md` §5（去重基线）、
`docs/433`（4.0 门槛）、`docs/524`（三贡献收口）、`docs/CLAUDE_INNOVATION_ATTACK_ROUND2_MOTION_20260818.md`
与 `_LOCAL5_`（否决清单）、`reviews/date_open_rtl_gap_mining_r1_20260827`（8 候选 fast-kill）、
`reviews/date_hw_gate_closure_r1_20260917`（H1/H2）、`reviews/local5_c1_qlabel_runpackage_r1_20260917`（LQ4/DRC/实测 FAIL）、
`docs/CLAUDE_SCORE_20260917_1709.md`（当前分数与 GPU 队列）。

---

## 0. 方法与边界

### 0.1 检索方法

- 工具：`WebSearch`（约 20 组查询，2025-09→2026-09 窗口，覆盖 CVPR/ICCV/NeurIPS/ICLR/WACV/ICLR 2026、
  ISCA/MICRO/HPCA/ICCAD/DAC/ISSCC/ASP-DAC 2025-26、arXiv 2026）；本机一手 PDF 直读 1 份
  （`docs/Zhang 等 - 2026 - A 28-nm Optical Flow Estimation Accelerator with Redundancy Speculation, Bit-Width-Aware Compression.pdf`，CICC 2026，已逐页读取）。
- **环境限制（诚实声明）**：本会话 `WebFetch` 权限被环境拒绝（4 次尝试均 Permission denied），
  因此外部文献事实均为**检索摘要/收录页级证据**，未做一手 PDF 逐句核对。凡引用处均给链接，
  标 `[lit]`；**任何 `[lit]` 数字不得直接进论文主表**，需二次一手核对后才可升级。
- **去重声明**：`ATLIF_EP34 §5` 已覆盖的 5 条（SDformerFlow TETCI 扩展版、AAAI 2026 TIDNet 系、
  ST-FlowNet、Aq-FireNet、STIRFlow）本报告**不作为新发现**；本文其余条目均为本轮新检索所得。
  其中 STIRFlow 有状态更新：已见 IEEE Xplore 正式出版记录（TIP，doc 11668994），不再是"仅作者声明"。

### 0.2 证据分档

| 档 | 含义（本轮） |
|---|---|
| `[rtl]`/`[prof]`/`[模型]` | 沿用项目惯例：已有 RTL/封存 profiler/本机只读复算 |
| `[lit]` | 外部文献，检索摘要级；链接已给，未经一手全文核对 |
| `[lit-一手]` | 本机可直读的一手原件（仅 CICC 2026 PDF 一项） |

---

## 1. 调研结果（四个板块）

### 1.1 (a) SNN / 事件相机光流 2025-2026：新进展

| 工作 | venue/年份 | 机制要点 | 对本地意义 |
|---|---|---|---|
| STIRFlow | IEEE TIP 2026（正式出版，doc 11668994）[lit] | SNN 特征编码器 + **时间迭代精化**；MVSEC 上比 E-RAFT/TMA 计算量低 17×/33× 且精度相当 | §5 已列，状态更新为"已出版"；"迭代精化"与 M1 相关 |
| EmFlow | Carleton MASc thesis 2026 [lit] | 硬件导向 SNN：稀疏卷积、延迟上采样、1-bit spike feature map、有限持久膜状态；Kria KV260 + GenX320，40 FPS@3.7-4.0 W；HFlow320 2.39 px、MVSEC 1.77 px、DSEC 9.43 px（25 ms 窗） | 说明"1-bit 特征图 + 持久膜状态"在 FPGA 上可行；DSEC 数字与我们不同口径，**不得混比** |
| Few-Spikes Neuron (FSN) flow | Springer 2026-06 [lit] | 首个 FSN 光流：1 ms 极短窗、7×2DFSN 层、MVSEC outdoor EPE ≤0.45 px、<3 spikes/pixel、相对 10 ms LIF 节能 8× | "极短时间窗"路线；提示时间窗长度是精度-能耗的显式旋钮（与 M7 相关） |
| Spiking Patches | IROS 2026 / arXiv 2510.26614 [lit] | 异步事件 tokenizer：patch 在活动超阈值时"发放"，带 refractory period；相对 voxel/frame 推理快 3.4×/10.4×，O(n) 复杂度 | 即"事件 tokenizer"类，**与本地 ESDA/patch event-token NO-GO 同族**（见 §4） |
| Neural Events | arXiv 2606.19835 (2026-06) [lit] | 异步离散自编码器把事件流 re-tokenize 成"neural events"，事件率降 2×，~140 KFLOPs/event | 同上，属输入 token 化路线 |
| SDformerFlow TETCI | 已覆盖（§5） | — | 去重基线 |
| STSC-Flow（非 SNN） | CVPR 2026 [lit] | 自监督时空结构一致性；DSEC EPE 0.663（比 BFlow 低 11.6%） | 竞争基线锚点；SNN 线不能宣称 DSEC 精度 SOTA |
| Directional Selective Filters | Pattern Recognition 2026-08 [lit] | 运动能量/Gabor + 方向选择 LIF 群体向量解码，紧凑可解释 | 小体量 baseline 参考 |
| Flow-Guided UAV Tracking | IEEE T-AES 2026 [lit] | 脉冲视网膜事件生成 + 事件-帧融合光流 | 事件生成侧，非本栈 |
| 综述 | Pattern Recognition 2026 [lit] | "Attention Mechanisms in SNNs: From neural encoding to hardware implementation"：强调 dense matmul/softmax 与 event-driven sparsity 不兼容 | 支撑"spike 原生稀疏注意力"叙事 |

**结论（(a)）**：除已覆盖的 5 条外，2025-26 的 SNN 光流新工作全部是**卷积/GRU/混合**路线或**输入
tokenizer** 路线；**仍然没有第二个独立发布的 spiking-Transformer 光流 backbone**（与 §5 结论一致，
本轮复核未推翻）。新的可用线索集中在"迭代精化 + 时间窗长度旋钮 + 1-bit 特征图"三点。

### 1.2 (b) 高效稀疏注意力与动态计算（CVPR/ICCV/NeurIPS/ICLR/WACV 2025-26）

| 工作 | venue | 机制 | 关键数字 |
|---|---|---|---|
| **TP-Spikformer** | ICLR 2026（arXiv 2603.00527）[lit] | **training-free**、插在每个 block 前的剪枝插件：IRToP 判据 = 空间显著性（token 与邻域均值余弦不相似度）+ 时间变化（相邻 timestep 差分）；IR-Arc = 未信息 token **block 级 early-stop，保留原值不删除** | 计算量最多降 ~48%，精度只掉 0.5-1.5%；跨 Spikformer/QKFormer/SDT-V1/V3，且验证到**事件目标跟踪** |
| **C-STEP** | IEEE 2026-04（doc 11539306）[lit] | LightSoftmax 置信度 → **时间维 exit**；早时刻引导的通道剪枝；共享 spike 分解 | synaptic ops 最多降 65.4% |
| **STEG-AIW** | WACV 2026 [lit] | 残差级 STEG 门抑制非显著激活 + AIW 逐步累积证据→halting 概率，**样本级早停** | timestep 减少 34-88%；CIFAR10-DVS 82.50%@5.68 步 |
| Edge-RecViT | CVPR 2026 [lit] | 边缘感知 ranker + 共享递归层，冗余 token 提前退出 | FLOPs −30.5%，参数 86M→23.21M |
| SPOT | CVPR 2026 Findings [lit] | attention 计算前预测冗余 token | ~40% 效率增益 |
| QuietPrune | CVPR 2026 [lit] | 查询引导的**早期**（ViT 内）2×2 组剪枝 + 聚合 | prefill 延迟 −19.0% |
| HoliTom | NeurIPS 2025 [lit] | 全局冗余感知时间分段 + 时空合并 | >90% token 削减，6.9% FLOPs |
| MeToM | CVPR 2026 [lit] | 用 codec 残差/GoP 大小做逐帧 token 预算 | 2.65× 加速 |
| LIPAR | arXiv 2603.05811 [lit] | training-free **跳过多帧重复 latent patch 的重算**，RoPE 感知近似 + attention 恢复 | 1.45× FPS、显存 −29%、warp error 优于 token merging |
| ORBIS | arXiv 2605.22015 [lit] | **输出引导** token reduction（用上一 timestep 输出相似度建索引对，corr 0.953 vs 输入引导 0.782）+ **专用匹配加速器** | token 削减 ~2× AsymRnR，4.5× 加速、能量 −79.3% vs A100 |
| HyperVAttention | arXiv 2607.03012 [lit] | 时空聚类稀疏注意力；**聚类分配跨 step 稳定**（step10 后仅 26-28% 变化）；硬件对齐的 cluster merging | 1.8×/2.3× 延迟下降 |
| 动态 token 计算综述 | IEEE 2026 [lit] | 分类学 + 警告：**理论节省常因选择开销/硬件不支持而落空** | 与 "Sparsity Tax" 互相印证 |

**结论（(b)）**：2025-26 的共识是"选择/预测开销本身就是大头"（PADE：8-bit 预测器可占 63% 功耗；
综述：selection overheads 吃掉理论收益），所以主流已转向 **无预测器**（PADE）、**输出引导**
（ORBIS/LIPAR）、**训练免费 + 保留语义的早停**（TP-Spikformer IR-Arc）。这对本地"不许新造
通用 matcher/预测器"的纪律是正面证据。

### 1.3 (c) SNN 加速器与神经形态硬件（ISCA/MICRO/HPCA/ICCAD/DAC/ISSCC/ASP-DAC 2025-26）

| 工作 | venue | 机制 | 数字 |
|---|---|---|---|
| **PADE** | HPCA 2026 [lit] | **无预测器**稀疏注意力加速：bit-serial enable stage-fusion（BUI-GF 逐 bit-round 不确定区间守卫过滤 + BS-OOE 乱序 + ISTA 交织 tiling） | 对比 H100 7.43× 加速、31.1× 能效；预测器功耗问题（8-bit 下 >63%）是其立论 |
| **Sparse by Command** | MICRO 2026（arXiv 2607.22038；artifact Zenodo 21503283）[lit] | 任务命令（推理前已知、持续数百帧）→ **门控 MLP（<0.12% 参数）预测 per-tile 二值执行掩码**；ISA 指令带 per-tile bitmask 字段；tile manager 见 0 bit 一拍跳过、不取 DMA；**掩码跨层传播**（被掩码的输出 tile 下层的输入组也不载入） | FLOPs −66~76%、延迟 9.12→3.74-4.44 ms（−51~59%）、263→108-128 mJ；GPU 上反而慢 22% |
| Sparsity Tax | arXiv 2607.22790 [lit] | **bitmap 直接当 clock-enable**：SRAM 行/寄存器/ALU 的时钟门控；R7 保存未来 16 条指令的 skip/execute 位；SIMT 变体加 per-PE 地址生成 + 游程编码稀疏权重 | 量化"稀疏税"：元数据存储与指令开销在低稀疏度下**使吞吐下降** |
| ME-MoD | TVLSI 2026 [lit] | 首个 MoD（mixture-of-depths）ViT 加速器：token 重排/顺序记录模块 + LN-Routing 融合 + token-stationary 层融合数据流 | 1.62× 加速、外部访存 −46.5%、能量 −45.2%、23.6 TOPS/W |
| PacViT | IEEE 2026-05 [lit] | 边缘 ViT 加速器：**由 attention 矩阵自身**做零开销动态剪枝（NDP）+ Sliding Cache Attention | 5.58 mJ/frame、9.31 ms、对比 V100 403× 能效、Top-1 仅掉 2% |
| 双稀疏自注意力核 | TVLSI 2026（Zhou 等）[lit] | **神经元抑制机制**：识别膜电位极负、确定不会发放的神经元并旁路其计算（neuron + spike 双稀疏） | 事件驱动自注意力核 |
| JSSC 2025 3-D 计算阵列 | JSSC 60(3) 2025 [lit] | 多 timestep 并行 3-D 阵列（权重复用）+ 并行非零数据 fetcher + 多模式调度器（SCONV / spiking Q/K/V 生成 / SSA） | 0.078 pJ/SOP、40nm、spiking transformer 77.6% ImageNet |
| Xpikeformer | TVLSI 33(6) 2025 [lit] | 混合模拟-数字：AIMC 做 FF/FC + 随机脉冲注意力（SSA）引擎 | 对 SOTA 数字 ANN-transformer 加速器 13× 节能；对 SOTA SNN transformer 的数字 ASIC 投影 1.9× |
| **SPARTA** | ICCAD 2025（doc 11240724）[lit] | RL（PPO）动态 token 跳过 + spike-aware token prediction（按权重分析预判非活跃 token）+ ReRAM-CIM + **token 路由/脉冲注意力数字引擎** | vs GPU 543.1×、vs COMPASS 10.2×；能效 308×/5.2×；精度损失 ≤1.1% |
| ASTER | arXiv 2511.06770 (2025-11) [lit] | 内存中心 CIM spiking-transformer 加速器；**推理期层跳过 + timestep 缩减**用贝叶斯优化协同搜索 | vs Jetson Orin Nano ~467× 能耗下降、vs 前代 PIM 1.86× |
| SpikeRAM | ISSCC 2026（DOI 10.1109/ISSCC49663.2026.11409002）[lit] | 事件驱动 spiking 近存/存内处理器 + 神经形态传感器 + 终身片上学习；48.1 pW/synapse/bit；464M synapses @ 8.28 mW | ISSCC 类 |
| GALSNP | ISSCC 2026 SRP [lit] | 40nm 1.01 pJ/SOP GALS 神经形态处理器，混合 ANN-SNN，无梯度片上学习 | ISSCC 类 |
| ExSpike | arXiv 2606.20414v2 [lit] | 全事件架构：direct coding + 事件驱动卷积/池化/FC；Sparse Core 把有效事件地址抽进 AER FIFO 触发事件驱动 PE 核 | 与本地 AER-NoC NO-GO 相邻 |
| ASP-DAC 2026 SNN 协同设计 | ASP-DAC 2026 [lit] | 稀疏编码 + **zero-skipping 加速器** | spike 数 −88%，相对 rate/temporal 编码加速器节能 88%/89%、吞吐 4.5×/26.8× |
| DSP/边缘 SNN 若干 | ISCAS 2026 / IEEE 2026 [lit] | RISC-V + 事件驱动核、spike-time sorter 跳非信息事件；BLC 位级松散耦合 + **混合早停**（LeNet-5 cycle −28.31%）；零检测 power-gating（估计执行周期 −89.03%） | 均为"跳过零/无效"类 |
| DAC 2026 检索结果 | DAC 2026 [lit] | 未见 SNN 专用；稀疏类为 GNN/LLM 方向（TAG、BSGCN、PiHG、PEACE、SparseE、FineMoE） | "动态稀疏执行与数据流架构不匹配"（FineMoE）是被反复引用的痛点 |

**结论（(c)）**：SNN 加速器的 2026 主线是 **CIM/PIM + token 级跳过 + timestep 缩减**；
**"跳过"已经开始以"ISA/描述符字段 + 一跳跳过 + 跨层传播"的形态固化（Sparse by Command）**，
以及"bitmap 即 clock-enable"的固化（Sparsity Tax）。这两条是给本地硬件线最可迁移的**执行对象级**范式。

### 1.4 (d) 软硬件协同的稀疏/跳过执行 + 生态位检查

| 工作 | venue | 机制 | 备注 |
|---|---|---|---|
| **CICC 2026 28-nm 光流加速器**（Zhang 等，北大） | CICC 2026；本机 PDF 直读 `[lit-一手]` + IEEE doc 11509564 | U-Net 混合 SNN-ANN 光流；① Dense-Channel-First Speculation：按通道密度 θ_c 在 PE 线内**就地推测** MaxPool/ReLU 冗余（MVSEC 上 MaxPool+ReLU 冗余降 73.8%）；② Bit-Width-Aware Compression（bit 级变宽压缩）；③ **Deep-Level Skip（DLS）**：当前层为 Level0 且相似度 > θ_s 时**只算 Level 0**（浅 U-Net），否则全 U-Net | 0.20×/0.08× 操作量、0.12×/0.19× 能耗、35.80 TOPS/W（含 MAC 14.07）；**"相似度→跳整层"已有硅级先例**，是本报告 M1 的最强外部锚 |
| ORBIS / LIPAR / HyperVAttention | 见 §1.2 | 输出引导 + 专用匹配器 + 聚类合并 | 说明"跳过决策"可以做成硬件块而非预测器 |
| SENTRY | GLSVLSI 2026（DOI 10.1145/3787109.3816393）[lit] | 近传感器分层空间推理：粗粒度 spike-rate 门**丢弃低 ROI 活动帧**；区域活动与全局背景率比较抑制弥散噪声 | +64% 精度、FP 率 −57%；输入侧门控范式 |
| SPIF | MDPI Remote Sensing 18(15) 2026 [lit] | 逐事件可靠性分数（邻域时空极性支持 + 像素发放史），低于阈值的事件被丢弃 | 事件级过滤，非区域跳过 |
| ETHEREAL | arXiv 2609.15241 [lit] | 事件驱动 GNN 处理器：3D 稀疏不规则访存用 8-way 组相联时空 cache（命中 58% → 读 EMA −2.4×），spline 卷积 | 事件图类，非本栈 |
| PacViT / ME-MoD / Sparse by Command / Sparsity Tax | 见 §1.3 | 均为软硬件协同跳过 | — |

### 1.5 本地基线事实（决定哪些方向必然死，必须与 idea 一起读）

| 事实 | 数字 | 来源 |
|---|---|---|
| Amdahl 冻结包络 | patch embed `32.15%`、ATLIF `20.64%`、FC1 `19.08%`、4 层 Conv `12.84%`、FC2 `6.68%`、attention `0.59%` | `[prof]` agent A 引 `620,302,905` |
| H67 中间 output-site 空置 | `0.1117%`（**禁止从传感器稀疏推断网络中间稀疏**） | `[prof]` gap-mining §4 |
| 跨帧 local-vs-temporal 逐行选择 | 仅约 `2.7%` source-work | `[prof]` 同上 |
| ATLIF delta/early-stop 的自然收益 | issue reduction `0.0676%` | `[prof]` 同上 |
| DATE 硬门 | decoder-complete、同资源、非重叠、全网络 ≥`1.10x`（首选 `1.15x`）；当前 Table A = 0 行 | `docs/524`、m628 |
| 算法侧现状 | C12 ep34 FP32 AEE `1.199514`；95/105 阈值恒 1、阈值自适应禁用；t49 ternary 不稳定 | `[prof]`/`[模型]` score #10、ATLIF 审计 |

**由此得到的筛选律（本轮全部 idea 的第一判据）**：
> 任何依赖 **"现有中间特征里天然就有的零/冗余"** 的跳过，都已被本地实测判死（0.1117% / 2.7% / 0.0676%）。
> 能活的只有两类：**(A) 契约层新稀疏**——由训练/合同引入、本来不存在的稀疏（学习门控、每 tile 变深度、
> 时间早停、支持域语义）；**(B) 阶段级对象切换**——跳过的是"整层/整槽/整 tile/整 token 行"的**执行对象**，
> 而不是逐操作数的零。下面 8 个候选全部按此设计。

---

## 2. 候选 idea（8 个）

统一格式：来源 / 一句话机制 / 插入位置 / 硬件侧对象变化（对齐 docs/433 双腿）/ 收益方向 /
与本地否决清单关系 / 最小验证路径（CPU 先行 + GPU 队列）。

### M1. CGRD — 收敛门控的逐 tile 精化深度（Convergence-Gated Refinement Depth，带精确证书）

1. **来源**：CICC 2026 28-nm 光流加速器 DLS（本机 PDF `[lit-一手]`，相似度 > θ_s 只算 Level 0；
   0.20×/0.08× 操作、0.12×/0.19× 能耗）；DPFlow（CVPR 2025，金字塔层数随分辨率自适应）；
   STIRFlow（TIP 2026，SNN + 时间迭代精化）；STEG-AIW（WACV 2026，证据累积→halting）；
   C-STEP（2026，时间 exit，synaptic ops −65.4%）；block-skipping early-exit for segmentation（PerCom Workshops 2026）。
   链接：`https://ieeexplore.ieee.org/document/11509564`、`https://ieeexplore.ieee.org/document/11668994`、
   `https://openaccess.thecvf.com/content/WACV2026/html/Saju_STEG-AIW_Spatio-Temporal_Gating_and_Adaptive-Timestep_Inference_for_Efficient_Spiking_Neural_WACV_2026_paper.html`、
   `https://www.semanticscholar.org/paper/0762be96f775a4aafdbc8fbd5ae57091ee78c32d`。
2. **机制一句话**：每个空间 tile 的**精化/解码层执行到该 tile 的整数流增量（residual）≤ ε 就停**，
   深度由"精确整数证书"决定并落成 per-tile 深度图，而不是全图跑满。
3. **插入位置**：`sttmultires_unet.decoders.0/1`、`preds.0/1`（以及 `encoders.swin3d.layers.2/3` 的深层
   ——stage2/3 恰是 H81 口径下同分结构最弱、信息最少的两级，`89.39%/96.78%` equal ratio）；证书
   比较器挂在 decoder 流增量寄存器旁，控制面走既有 typed phase/epoch 面（M221）。
4. **硬件侧对应点**：**新存储对象** = per-tile 深度图（2-3 bit × tile 数）+ 流增量寄存器（tile 级，小）；
   **新执行对象** = "level-ℓ service enable" 由描述符流携带、门控 decoder/精化流水线的 issue；
   证书本体是整数幅值比较（**exact**，可复现、可 RTL miter）。文档措辞必须写"depth-certificate 对象"，
   不得写成"又一个稀疏 matcher"。
5. **预期收益方向**：延迟/能耗 ∝ 被跳过的深层份额（外部硅级先例 0.20× 操作量）；面积增量 <1 tile 寄存器；
   精度由 ε 事后标定控制。
6. **与本地否决清单关系**：**不是** V1 跨窗 quotient（无跨窗状态、无目录持久）；**不是** gap-mining 的
   "跨帧 warp/delta tile cache"（无跨帧 feature cache、无 warp；状态是**单次推理内**的收敛深度）；
   **不是**"attention prune/score bin"（不在 attention，且跳的是整层）。
   仍需自证：其收益不得被"dense 基线本来就快"解释（须给 frozen-trace 上界 + 同资源对照）。
7. **最小验证路径**：CPU 先行：① 用冻结 H67 profile 算 decoder/深层各级的 cycle 占比与"若 ε 生效的
   上界"（[模型]）；② 写 ε-证书的数学可逆性检查（收敛单调、证书无假停）。GPU 队列：③ 冻结 checkpoint
   上**免训练标定** ε 扫一层（无梯度、只前向）；④ 若 loss 形态可接受再进 short 训练；判据 = valid825
   AEE 相对锚点（Motion ep35 `1.3297`）`≤×1.01`，且 cycle 上界 ≥10%。**副产品：强制补齐 decoder trace（G1 缺口）。**

### M2. RCM — Regime-Commanded tile Mask（模式指令化 tile 掩码，描述符携带）

1. **来源**：Sparse by Command（MICRO 2026，arXiv 2607.22038，Zenodo 21503283）——门控 MLP <0.12%
   参数、per-tile 二值掩码、ISA 携带 bitmask、一拍跳过、跨层掩码传播、FLOPs −66~76%、延迟 −51~59%；
   ME-MoD（TVLSI 2026，token 重排 + 层融合，EMA −46.5%）；Sparsity Tax（arXiv 2607.22790，bitmap 即
   clock-enable + 稀疏税必须计价）。链接：`https://arxiv.org/abs/2607.22038`、`https://zenodo.org/records/21503283`、
   `https://ieeexplore.ieee.org/document/11625947`、`https://arxiv.org/html/2607.22790v1`。
2. **机制一句话**：把"**事件流先验/运动 regime**"当成 Sparse-by-Command 里的"command"（推理前已知、
   跨帧稳定），用小门控网预测 **FC1/FC2/Conv tile 级执行掩码**，掩码作为**一等对象**进描述符并跨层传播。
3. **插入位置**：typed descriptor 头（C2 的 source descriptor 家族）新增**掩码字段类**；消费端 =
   M519 FC2 端点、M490/M491/M495/M499 的 8-bank 前端、M498 的 PWT/Scratch 提交面；掩码产生的
   "被跳过输出 tile"要在下一层输入装载侧同样不出请求（跨层传播，与 MICRO 论文同构）。
4. **硬件侧对应点**：**新存储对象** = 执行掩码寄存器组（数百 bit 级）+ 跨层掩码台账（ledger）；
   **新执行对象** = tile manager 的"读掩码→一拍跳过、不发 DMA、不 commit"状态机；**无数据 cache、
   无 router、无 reorder**（主动避开 V5 与 AER/NoE 禁令的措辞）。
5. **预期收益方向**：周期/能耗 ∝ 被掩码 tile 份额（外部 FPGA 原型 −51~59% 延迟）；逻辑面积增量小
   （掩码寄存器 + 一跳控制器）；风险在**学习到的掩码是否真的省 cycle**——必须走 MICRO 论文同款的
   "GPU 上掩码反而变慢"对照，证明只有本硬件形态能兑现。
6. **与本地否决清单关系**：**不是**"近似剪枝"（433 封的是"现有算子下的近似剪枝不产生新物化对象"，
   本项新增**掩码对象 + 跳过状态机 + 跨层台账**三件物化）；**不是** V5 第四 matcher/bitmap decoder/
   event router/delta cache/NoC bridge/reorder adapter（逐条对照：无匹配器、无事件路由、无缓存副本、
   掩码静态携带不重排）；与 H1/H2 任务（时间窗口对象）正交，可叠加。
7. **最小验证路径**：CPU 先行：① 从冻结 trace 统计 per-tile 工作分布与"掩码可达上界"（[模型]）；
   ② 把掩码字段加入 descriptor 规格草案（只写文本，不动 RTL/编号文档）。GPU 队列：③ 训练 <0.12%
   参数的门控 MLP（与 backbone 联合，三阶段 soft→hard 掩码，照 MICRO 论文配方）；④ valid825 对比锚点；
   ⑤ 之后才谈 RTL（一次同端口周期/能量模型）。

### M3. TAP — Token 早停 + 保持（spiking 专用 IR-Arc 语义进描述符流）

1. **来源**：TP-Spikformer（ICLR 2026，arXiv 2603.00527）——training-free、IRToP（邻域均值余弦不相似度 +
   相邻 timestep 差分，**全是局部算子**）、IR-Arc（未信息 token **块级早停但保留原值**），计算 −~48%、
   精度 −0.5~1.5%，且验证到**事件目标跟踪**；QuietPrune（CVPR 2026，早期剪枝优于晚期）；
   动态 token 综述（IEEE 2026，警告选择开销）。链接：`https://iclr.cc/virtual/2026/poster/10010064`、
   `https://arxiv.org/abs/2603.00527`、`https://openaccess.thecvf.com/content/CVPR2026/html/Gao_QuietPrune_Query-Guided_Early_Token_Pruning_for_Vision-Language_Models_CVPR_2026_paper.html`。
2. **机制一句话**：用**局部时空判据**给每个 token 算 enable 位；enable=0 的 token 跳过 SSA/MLP 但**原地保持**
   （write-skip + hold），而不是删除——保持语义被写进算子合同。
3. **插入位置**：spiking Swin block 前端（`encoders.swin3d.layers.*.swin_blocks.*`）与其 `mlp.sn1`
   （FC1/FC2 合计 `38.5%`）；enable 位搭既有 typed descriptor/bundle 顺风车；**RQTB 目录不写、不改**
   （避免与 389/445 的 score-front CSE 封禁擦边）。
4. **硬件侧对应点**：**新存储对象** = per-slot hold 使能位 + hold 寄存器组（或"保持不写"的写跳过）；
   **新执行对象** = 块级 early-stop 通路：FC1/FC2 的 issue 行选择与 ATLIF rank service 的 site 使能
   吃同一张 enable 位图（**bitmap 直接当时钟/issue enable**，Sparsity Tax 范式），并且这张位图本身
   是描述符的一个字段（Sparse-by-Command 范式）。
5. **预期收益方向**：FC1/FC2/ATLIF 三块合计 `59.7%` 的份额按 enable-off 比例下降；面积近乎零增量；
   精度风险是本组最高（training-free 迁移到光流需实测）。
6. **与本地否决清单关系**：**不是**"attention prune/score bin"（那份只有 `0.5894%`，且此处跳过的是
   整个 block 的 MLP/FFN，主要命中 FC1/FC2）；**不是** ESDA/patch event-token（不改输入 tokenizer、
   不引入 coordinate metadata）；**必须反驳**"中间 output-site 只有 0.1117% 空"（我们的稀疏来自
   **判据 + 训练**，不是天然零）——这要在报告里明写，否则会被 gap-mining §7 的"极高事实风险"红线击中。
7. **最小验证路径**：CPU 先行：① 用 m1458 ATLIF 活动捕获 + 现成 per-site/per-call 记录，复算
   "IRToP 判据在 H67 特征上的 enable-off 率"的**代理界**（[模型]，注意 m1458 是 activity 不是 per-token
   feature，先做判据本身的可算性验证）；② 写 hold 语义的等价性检查（前向一致性：被 hold 的 token 必须
   在下一 block 的 attention 中仍可被读到）。GPU 队列：③ training-free 探针（无梯度）；④ short 训练 → valid825。

### M4. BSF-PM — 精确父匹配的逐 bit-round 守卫过滤（PADE 式无预测器融合，exact）

1. **来源**：PADE（HPCA 2026，doc 11408448 / arXiv 2512.14322）——无预测器；BUI-GF 逐 bit-round
   不确定区间守卫过滤（保证正确性）、BS-OOE、ISTA；预测器 8-bit 下可占 >63% 功耗是其立论；
   7.43×/31.1× vs H100；Sanger/DOTA/SOFA 是其对照。链接：`https://2026.hpca-conf.org/details/hpca-2026-main-conference/3/`、
   `https://www.alphaxiv.org/abs/2512.14322`。
2. **机制一句话**：把"守卫"**融进**现有 exact 父-积匹配流水的 bit-round 里：每轮用不确定区间**精确**排除
   不可能闭合的候选对，**不新增独立预测器对象**（这正是 PADE 的核心洞见）。
3. **插入位置**：M935 三级 exact 父匹配链 / M363/M348 的 banked q128 signed residual matcher 前端
   （C2 的 source-parent 搜索侧）；若放 RQTB score 前，只能按"附录级"处理（gap-mining 已把 attention
   prune/score bin 判为附录）。
4. **硬件侧对应点**：**新存储对象** = per-lane 2-bit 不确定区间寄存器（替代"预测器表"）；
   **新执行对象** = 匹配流水内的 guard 阶段（区间比较 + 该 bit-round 的 lane clock-enable）；
   与 C2 现有 bank/port 合同兼容（区间寄存器很小，无需新缓存）。
5. **预期收益方向**：exact 搜索的周期下降（外部同级工作 7× 级，但我们必须在冻结 ledger 上自测）；
   **精度风险为零**（守卫保证 exactness），这是本组唯一"精度零风险"候选，适合做 DATE 的 soundness 加分项。
6. **与本地否决清单关系**：**不是**"第四种 matcher"（这是**既有 exact 匹配流水内的阶段融合**，
   不是新匹配语义）；**不是** ATLIF delta/early-stop（对象不同：匹配搜索 vs 神经元发放）；
   符合"exact 文化"（守卫带证明边界），与 524 的 capture gap 叙事天然同页。
7. **最小验证路径**：CPU 先行：① 对冻结 51.84M 行账本做 bit-round 粒度重放，算 guard 的**理论对削率**
   与 worst-case（[模型]，纯 CPU，零依赖）；② 写 guard 的区间代数证明草图（1 页）。后续：仅当对削率
   过同端口门才进 RTL 草图与 EDA 队列。

### M5. SIP — 神经元抑制预测旁路（signed inhibit pre-state，"neuron + spike" 双稀疏）

1. **来源**：TVLSI 2026 双稀疏自注意力核（Zhou 等，`https://www.semanticscholar.org/paper/7f53c7a66bf47ce6584b2f367b0f61bcc4611de8`，
   "膜电位极负→确定不发放→旁路其计算"）；Sparsity Tax（bitmap clock-enable）；L-SPINE（arXiv 2604.03626，
   2/4/8-bit 可重构 SIMD SNN 数据通路）。链接：`https://arxiv.org/html/2604.03626v1`。
2. **机制一句话**：在 C3 相位解耦神经元服务上，给每个 site 增加**有符号预充/抑制状态**，预测"本 rank 不发"，
   预测为抑制的 site 直接旁路其累加/提交，并把抑制判定做成一种**新的描述符类别**。
3. **插入位置**：ATLIF rank service（`docs/524` C3，M273/M289/M518 家族）；消费端 = Acc24 上下文
   与 parent 提交面（C1/C2）。
4. **硬件侧对应点**：**新存储对象** = per-site 1-2 bit 有符号抑制寄存器（相对 membrane 已有存储是
   小增量）；**新执行对象** = "inhibit token" 描述符类 + 旁路通路（跳过该 site 的贡献与 commit）。
5. **预期收益方向**：能耗/周期按被旁路 site 份额下降；但**收益是三者中最不确定的**。
6. **与本地否决清单关系**：**与已判死的"ATLIF delta/early-stop engine"（issue reduction `0.0676%`）
   高度相邻**——区别在于本项不是"跳零操作数"，而是"**带训练合同的有符号预状态**预测不发放"，
   属契约层新稀疏（§1.5 的 A 类）。**必须先证明新合同能产生可测的旁路率**，否则按 same-file 判死。
   另：当前训练侧阈值自适应禁用、95/105 阈值恒 1、t49 ternary 不稳定（score #10 + ATLIF 审计）→
   本项显式依赖训练侧先修好（H12 梯度修复队列）。
7. **最小验证路径**：CPU 先行：① 用 m1458 活动捕获算"若存在有符号预状态，可预测-抑制的 site/rank 比例"
   （[模型]）；② 与 0.0676% 的门做并列比较，**不过则直接降为附录**。GPU 队列：③ 带抑制正则的 short 训练。

### M6. MWM — 带重数（multiplicity）的 token 合并合同

1. **来源**：HoliTom（NeurIPS 2025，>90% token 削减）、MeToM（CVPR 2026，codec 元数据引导逐帧预算）、
   InfoMerge（2026，二阶时间冗余指纹）、TM-Adapter（WACV 2026）；本地 H81 已验证可复用的
   **multiplicity-aware normalization** 对象（`docs/433` §2）。链接：`https://proceedings.neurips.cc/paper_files/paper/2025/hash/c573258c38d0a3919d8c1364053c45df-Abstract-Conference.html`、
   `https://www.openaccess.thecvf.com/content/CVPR2026/html/Wu_MeToM_Metadata-Guided_Token_Merging_for_Efficient_Video_LLMs_CVPR_2026_paper.html`。
2. **机制一句话**：把时空冗余 token **合并成一个代表 + 显式重数字段**，下游归一化用重数感知平均
   （H81 已有精确整数版本），从而 token 行数下降而**语义不丢**（相对剪枝的卖点）。
3. **插入位置**：patch embedding 之后、block 0 之前的分区器；以及 stage 之间的下采样处；
   重数字段进 descriptor（与 RQTB 的 multiplicity 语义同源）。
4. **硬件侧对应点**：**新存储对象** = 合并索引图（小）+ descriptor 重数字段；**新执行对象** =
   合并/解合并单元 + 按合并后 token 数收缩的 issue 宽度（FC1/FC2 行数与 ATLIF site 数随之减少）。
5. **预期收益方向**：token 行数 → 线性收益（覆盖 patch/FC/ATLIF 大份额）；**风险**：合并本身有损，
   合同必须规定精确重数语义（整数计数）与可训练判据，否则精度过不了锚点。
6. **与本地否决清单关系**：**不是** reorder adapter（不是布局重排，是"消元 + 重数"）；
   **不是** V5 的 bitmap decoder/event router；**不是**"近似剪枝"（重数使归一化保持精确）；
   与 M3 互补（M3 保持、M6 合并），二者选一或串行需 DSE。
7. **最小验证路径**：CPU 先行：① 在冻结捕获上算每窗合并率与重数分布，估收益上界；② 重数语义的
   整数等价检查（对齐 H81 normalization）。GPU：③ ToMe 式免训练探针；④ short 训练 → valid825。

### M7. TPS — 时间槽早停（证据累积 halting 证书，逐 tile 可变时间深度）

1. **来源**：STEG-AIW（WACV 2026，timestep −34~88%）、C-STEP（2026，temporal exit，synaptic ops −65.4%）、
   ASTER（arXiv 2511.06770，layer skipping + timestep reduction 用 BO 协同搜索）、BLC SNN 加速器
   （IEEE 2026-05 doc 11511848，混合早停 cycle −28.31%）。链接：`https://arxiv.org/abs/2511.06770`、
   `https://ieeexplore.ieee.org/document/11511848`。
2. **机制一句话**：时间槽循环对每个 tile/token **证据饱和即停**（证书：累积证据 ≥ 阈值或增量 < ε），
   halt 位图按槽下发，被停槽不再推进 slot FIFO / 不再服务 rank。
3. **插入位置**：时间 slot FIFO + shiftmax + ATLIF service（Motion T=10 / Fixed2S T=2 的槽语义）；
   halt 位图与 M2 的掩码字段同族（可共用一个描述符字段类）。
4. **硬件侧对应点**：**新存储对象** = per-tile 证据寄存器 + halt 位图；**新执行对象** =
   槽推进控制器（读 halt → 冻结槽状态、跳 service）。
5. **预期收益方向**：周期 ∝ 实际执行槽数（外部 34-88%）；**与 LQ4/TLQ-5 的区别必须写清**：那两者
   保留全部槽、只做记录/广播；本项**减少槽数**。
6. **与本地否决清单关系**：**不是** V1 跨窗（无跨窗状态）；**不是** TLQ-5/LQ4（对象不同：槽**计数**
   vs 槽**记录**）；**与"ATLIF delta/early-stop"判死理由（0.0676%）相邻**——区别是跳过的是**整槽**
   （ATLIF 占 `20.64%`，整槽跳的份额远大于操作数级），但必须先给冻结 trace 的槽级活动分布作上界。
   与 D1/B2 训练线（T>2 合同）**强耦合**，必须双线协调，避免与 D1 争夺同一训练身份。
7. **最小验证路径**：CPU 先行：① 从冻结 profile 算槽级活动饱和分布与"可停槽"上界（[模型]）；
   ② 与 LQ4/TLQ-5 做同 trace 对比表（谁是 D1 的更强继承者）。GPU：③ halting loss short 训练；
   ④ valid825。**排在 D1 训练线之后**，不并行抢卡。

### M8. SENS — 事件支持域契约（flow + support + hold 语义；输入侧稀疏、输出侧保持）

1. **来源**：CICC 2026 光流加速器（相似度门控 + 层级跳过，本机 PDF）、SENTRY（GLSVLSI 2026，区域活动
   对比全局背景率）、SPIF（MDPI 2026，事件可靠性过滤）。链接：`https://dl.acm.org/doi/10.1145/3787109.3816393`、
   `https://www.mdpi.com/2072-4292/18/15/2551`。
2. **机制一句话**：输出契约改为 `{flow, support, confidence}`：只在**事件支持度超过全局背景率**的 tile 上
   计算 flow，未支持 tile **保持**上一帧该 tile 的流值（仅输出域保持，非特征缓存）。
3. **插入位置**：voxelizer/输入前端产 support 位图（来自事件流，**真实稀疏**）；patch embedding 与
   后续 tile scheduler 消费它做 tile 级跳过；输出头写 hold。
4. **硬件侧对应点**：**新存储对象** = support 位图寄存器组 + 输出域 hold 缓冲（tile 数 × 2 通道，
   极小的**流场**保持，**不是** feature cache）；**新执行对象** = 由 support 位图门控的 tile issue
   （与 M2 共用控制器）。
5. **预期收益方向**：低活动序列上 patch/encoder/decoder 的整 tile 跳过；精度由 hold 语义与
   support 阈值控制（须多序列验证）。
6. **与本地否决清单关系**：**必须与"跨帧 warp/delta tile cache"（NO-GO）严格切割**：本项
   **不缓存特征、不做 warp、不做 delta**，只保持低精度输出流场；也**不触碰** gap-mining §4 的
   "patch event-token engine"（不改 tokenizer、不加 coordinate metadata）。
   同时正面回应"禁止从传感器稀疏推断网络中间稀疏"：我们的跳过决策只用**输入侧真实稀疏**，
   不假设中间特征稀疏。
7. **最小验证路径**：CPU 先行：① 用 DSEC 事件数据算 per-tile 支持度分布与"可跳 tile 份额"（[模型]）；
   ② hold 语义的误差传播上界（静态分析）。GPU：③ 带 support/hold 目标的 fine-tune；④ **多序列**
   （≥3 条）valid825 + 低活动序列专项。

---

## 3. 排序与推荐

| 排名 | ID | 新算子合同 | 存储对象 | 执行对象 | 外部锚强度 | 精度风险 | 与目标线冲突 | 综合 |
|---|---|---|---|---|---|---|---|---|
| 1 | **M1 CGRD** | 逐 tile 收敛深度 + 精确证书 | per-tile 深度图/增量寄存器 | level-enable issue 门 | **强**（CICC 2026 硅级先例） | 低-中（ε 事后标定） | 无 | **A** |
| 2 | **M2 RCM** | regime 指令化 tile 掩码 | 掩码寄存器 + 跨层台账 | tile manager 一拍跳过 | **强**（MICRO 2026 + FPGA 实测） | 中（需联合训练） | 无（与 H1/H2 正交） | **A** |
| 3 | **M3 TAP** | token 早停 + 保持语义 | hold 使能位 + hold 寄存器 | block 级 early-stop + 行使能 issue | **强**（ICLR 2026，训练免费，事件任务验证） | 中-高（迁移未证） | 需与 RQTB 目录解耦 | **A−** |
| 4 | **M4 BSF-PM** | exact 守卫融合（无新预测器） | 区间寄存器 | bit-round 守卫阶段 | 中（HPCA 2026，工作负载不同） | **零**（exact） | 无 | **B+** |
| 5 | **M7 TPS** | 时间槽 halting 证书 | 证据寄存器 + halt 位图 | 槽推进控制器 | 中（WACV/C-STEP/ASTER） | 中-高（SNN 时间语义） | **与 D1/B2 训练线抢身份** | **B** |
| 6 | **M6 MWM** | 合并 + 精确重数 | 索引图 + 重数字段 | 合并/解合并 + 收缩 issue | 中（NeurIPS/CVPR 2026） | 高（有损） | 无 | **B** |
| 7 | **M8 SENS** | 支持域 + hold 输出语义 | support 位图 + 流场 hold | support 门控 tile issue | 中（CICC/SENTRY） | 中（hold 语义） | 与 NO-GO 相邻，须切割 | **B−** |
| 8 | **M5 SIP** | 有符号抑制预状态 | 抑制寄存器 | inhibit 描述符类 + 旁路 | 中（TVLSI 2026） | 中 | **与 ATLIF early-stop 判死相邻** | **C+** |

**若只做三个（供 DATE 4.0 + 硬门同时推进）**：**M1 + M2 + M3**。
- M1 直击 decoder（补齐 G1 缺口）且外部硅级先例最强；
- M2 是唯一"掩码进 ISA/描述符 + 跨层传播"的**指令级对象变化**，且与本地 typed descriptor 天然同构；
- M3 是唯一"spiking-native + training-free 起点"的候选，允许先用零训练成本试错。
M4 作为**并列第四**（精确零风险），适合在任何训练资源紧张时先做 CPU 侧对削率验证。

**统一硬件公共载体（三条 M 共用，建议一次说清，避免被当成三个 matcher）**：
"enable 位图作为一等 issue 操作数"——位图进描述符字段 → 直接作 FC/ATLIF/decoder 的
clock/issue enable（Sparsity Tax 范式），**只有一套掩码寄存器 + 一套跳过状态机**，
三个候选复用同一个载体对象。

---

## 4. 明确不适用 / 应保持封死的方向（对新检索结果的裁决）

| 方向 | 外部新证据 | 本地裁决维持理由 |
|---|---|---|
| 跨窗/跨帧 quotient 或类集复用（含 ORBIS 式输出引导复用落到 RQTB） | ORBIS corr 0.953、LIPAR 1.45× | ROUND2 V1：非重叠 tile、邻接增量仅 +0.17、目录写活动 <1% |
| 前帧 feature delta/warp tile cache（DeltaCNN/MotionDeltaCNN 系） | DeltaCNN 官方 RTL/论文 | gap-mining：自然 source-work 仅 2.7%、SRAM 税未定价 |
| Patch event-token / submanifold 引擎（ESDA-SNE 系） | Spiking Patches（IROS 2026）3.4×/10.4× | gap-mining：H67 空 output-site 0.1117%；强 baseline 已跳零 |
| AER/NoC bundle router（ELSA/ActiveN 系） | ExSpike AER FIFO；SPARTA token routing | gap-mining：无 NoC/first-response ledger；C2 已 bundle |
| Reorder-in-reduction adapter（FEATHER 系） | ME-MoD token 重排仅可作实现借鉴 | gap-mining：无 layout-reorder 账；M490/M499 已是 adapter |
| Attention prune / score bin | TP-Spikformer、SPARTA RL-TS | attention 仅 0.5894%；gap-mining 判"附录，不开主线" |
| 第四种通用 sparse matcher | PADE、Sparsity Tax bitmap gating | 433/V5：换名不换对象；对 DATE 硬门零贡献 |
| 侧车 quotient-file（Motion） | — | agent A 本轮裁决 SHELVE（state 腿 +13.5%/+10.3% ✗） |
| Local5 C1 跨 pair 统计平面 | — | **已由本轮实测判 FAIL**（G4 23.42% < 60%）；转 implementation-only |
| A3S / D3 | — | docs/448 NO_GO_AS_HARDWARE_ACCELERATOR |

---

## 5. 最小验证路径汇总（严格 CPU 先行）

**CPU 可先行（本机，零 GPU、零 EDA；建议按此顺序）**：
1. M1-S1：decoder/深层 cycle 占比与 ε 上界（读冻结 profile）+ 证书可逆性检查 → `[模型]` JSON。
2. M4-S1：bit-round 守卫对削率重放（读冻结账本）→ `[模型]` JSON（成本最低、风险最低，建议最先做）。
3. M2-S1：per-tile 工作分布 + 掩码可达上界 → `[模型]` JSON。
4. M3-S1：IRToP 判据在 H67 现有捕获上的 enable-off 代理界 + hold 语义等价性检查。
5. M5-S1：抑制旁路率上界，与 0.0676% 门并列比较（不过即降附录）。
6. M7-S1：槽级饱和分布 + 与 LQ4/TLQ-5 的同 trace 对比表。
7. M8-S1：DSEC 事件数据的 per-tile 支持度分布（纯数据侧，可与 GPU 无关）。

**GPU 队列（排在 `t49_ternary_ws4_ep5` 之后，遵守一次一个任务）**：
① M1 免训练 ε 标定（前向）→ ② M3 training-free 探针 → ③ M2 门控 MLP 联合 short → ④ 各自 valid825
（判据统一：相对锚点 `≤×1.01`，且周期上界 ≥10% 或组件能量 ≥15%）。**均不与 D1/B2 并行抢卡**；
M7 排 D1 之后。所有 GPU 任务需用户批准后按既有 run-card 纪律执行。

**EDA 队列**：本轮不新增；M4 若过 CPU 门，其 RTL 只允许并入既有 exact 匹配线（不新开模块线）。

---

## 6. 证据与可复现性

| 项 | 等级 | 说明 |
|---|---|---|
| 外部文献事实（§1 全部条目） | `[lit]` | 检索摘要级；WebFetch 被环境拒绝，未做一手全文核对；链接齐全 |
| CICC 2026 28-nm 光流加速器细节 | `[lit-一手]` | 本机 PDF 直读（题目/架构/三机制/0.20×/0.08×/0.12×/0.19×/35.80 TOPS/W 等） |
| 本地基线事实（§1.5） | `[prof]`/`[模型]` | 引自 docs/524、gap-mining、score #10、ATLIF 审计，未复算 |
| 8 个候选 | 提案 | 均未写 RTL、未训练；收益方向为预测，非结果 |
| 去重边界 | 声明 | ATLIF_EP34 §5 的 5 条不作为新发现；STIRFlow 状态更新为"已见正式出版" |

**本轮未做**：任何 GPU 命令、任何 EDA、任何 RTL、任何训练；未修改 `docs/359/362/366` 或任何编号文档；
未触碰冻结合同、closure、results；未删除文件；H82/H86 未重启。除本目录外零写入。

## 附录 A：本目录产物

| 文件 | 说明 |
|---|---|
| `review.md` | 本文（SHA256 见 `review_artifact_sha256.json`，避免自引用漂移） |
| `candidates.json` | 8 个候选的机器可读清单（字段对应 §2 七项；SHA256 见 `review_artifact_sha256.json`） |
| `review_artifact_sha256.json` | 上述两件的 SHA256 回执（本目录产物集的权威校验值） |
