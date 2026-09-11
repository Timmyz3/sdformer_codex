# C1/C2 创新逐文件复审与重构判断

2026-09-06 · 唯一目标：TCAS-II · 研究审阅，未形成投稿准入

**原始 idea 包已完成 42/42 文件精读，另复审上一轮 Codex 研究入口；合计 620 条定位记录，包含重复提案、文献背景、合同和工程史料。原包中没有一条能直接升为当前主创新。** 这不是说不能借鉴，而是需要把“借来的机制”改成一个能在我们的真实合同下成立、且有独立电路价值的执行方法。只换任务、名称、位宽或广播顺序不够。

目前最值得继续推进的是 **C2 浅层 FC1 的二值源统计先行**：尝试用输入共发放计数直接形成当前动态 BN 的统计，省去为了统计而生成第一遍完整宽隐层。两名未参与提案的独立评审均给主机制研究潜力 **6/10**。它已经改变计算对象，但专用电路、同资源净收益和算术身份仍未闭合。**C1 重构目前没有通过这一级筛选；不能为凑 C1+C2 双贡献而保留弱机制。** 本轮没有“稳 accept”或已可投稿的候选。

## 先看独立评审结论

评分 N/F/H/T 分别表示新意、冻结身份适配、硬件可行性、TCAS-II 主机制研究潜力，各 0–10；不是录用概率。T≤3：淘汰或实现背景；T4–5：有条件重构；T6：值得做有界验证；T≥7 仍须有电路证据。来源文件中的作者自评不沿用。冻结 FP 适配与另建整数部署的适配分开记录，不取平均。

| 本轮重构候选 | 真正改变什么 | 独立 N/F/H/T | 决策与主要阻碍 |
|---|---|---|---|
| V1 · C1 在线虚拟双源基 | 父节点不再限于真实输入行；载入激活时构造少量共享基，一个输出可消费多个基 | 4/8/5/4 | 暂不升主线。SumMerge 已有虚拟交集 DAG；只剩在线构造与有限端口服务可做电路增量 |
| V2 · C2 二值源统计先行 | 动态 BN 首遍统计从宽 FC1 输出转移到二值输入共发放，跨输出通道复用 | 6/5/5/6；另一评审 5/3/6/6 | 第一优先，仅浅层两个 FC1。密 Gram、宽收缩和输入重放必须胜过强融合／重算基线；整数精确不等于冻结 FP 精确 |
| V3 · 动态 BN 消费者精化 | 一个 token 的精化通过全域统计，帮助判定其他未执行 token 的二值结果 | 6/5/2/4 | 有执行依赖上的差别，但宽锚点、松区间及反复全域检查可能吞掉收益 |
| V4 · 共享投影阈值区间执行 | 对共享投影的可达二值模式，直接选择 FC2 聚合向量 | 3/0/5/4 | 原 ep34 不可直接用；DT-SCNN 已覆盖共享投影与多阈值核心。只有区间到下层聚合的增量待证 |
| V5 · C1 位移相关统计与边缘修正 | 用卷积核相关式形成统计，以精确边缘修正恢复零填充语义 | 5/3/1/2 | 淘汰当前 C1 挂载。真实卷积为 768 通道，计数和宽收缩过大；迁往高分辨率 patch 是另一工作负载 |

完整原卡见 [candidate_cards_v1.md](/home/zhumd/work/sdformer_codex/ideafromai/research/innovation_first_audit_20260906/candidate_cards_v1.md)。该文件保留送审快照，后续位宽和先验纠正以下文及独立评审为准。独立评审：[V1/V3](/home/zhumd/work/sdformer_codex/ideafromai/research/innovation_first_audit_20260906/records/review_v1_v3_by_c2.json)、[V2 评审一](/home/zhumd/work/sdformer_codex/ideafromai/research/innovation_first_audit_20260906/records/review_v2_by_c1.json)、[V2/V4 评审二](/home/zhumd/work/sdformer_codex/ideafromai/research/innovation_first_audit_20260906/records/review_v4_v2_by_new.json)、[V5](/home/zhumd/work/sdformer_codex/ideafromai/research/innovation_first_audit_20260906/records/review_v5_c1_spatial_moments.json)。

## 原始 idea 包哪些可以借，哪些不能直接迁

下表合并重复出现的机制族；后面的 620 条台账仍保留每个原文件的具体位置、原机制、最近先验、冻结适配、实际增量、代价、迁移方式、四维评分和结论；20 条非机制工程记录的评分为“不适用”。文献清单中的每篇论文并不自动变成一个新提案。

| 原包机制族与典型位置 | 可借的内容 | 当前不能成立的部分 | 迁移判断 |
|---|---|---|---|
| OP-STW、DPP-Skip、光流方向唤醒；Card A、ANN/光流/Round 1–2 | 因果预测、漏判后回退、tile 依赖传播 | 当前 flow 不能免费先于产生它的推理；当前 U-Net 没有 RAFT 相关体积或迭代残差。非零 tile 跳过通常改函数 | 原版不采纳。另算法须 AEE；精确版须算前证明输出不变 |
| HBG-RP、ADP、幅度／脉冲双路径；Card B、SNN、ATLIF 合同 | 类型化接口、符号与极值协议可作实现底座 | ep34 出口是静态 θ 缩放的二值；没有逐事件 int8 ATLIF 载荷。对错误身份设计的双幅度 PE 无法当冻结贡献 | 原版淘汰。若坚持实值载荷，必须另建模型 |
| MSBC、结构稀疏、级联精度、LUT 权重类；ANN/NPU | 可借训练约束与精度自适应接口 | 量化码重复、分组和 bit-skip 已有强先验；改变精度／N:M 会改变模型，不能借无损基线数值 | 算法候选或实现，不单独当新 C1/C2 |
| ECP-QKV、FACT/Bishop 式算前判定 | 在昂贵投影之前建立证书，而非算完再决定跳过 | 动态 BN 的全域依赖、满秩 PSN 的正负时间系数不能忽略；经验预测不等于精确证书 | 保留重构抓手，形成 V3；目前证书代价高 |
| Δ-MaskPipe、MW-DeltaBuf、视频差分执行 | 只处理变化，并保留非线性状态与依赖 | 局部输入不变不能保证动态 BN 后输出不变；光流运动补偿、稠密锚点、边缘和回退要计费 | 可重构，不允许把静态 BN 视频电路原封套入 |
| SS-FSA、Flash-SDSA、MX3P、脏分数；Grok46、Card C | H67 原生算子融合；K 时间对端驻留；精确分数失效条件 | SDSA 不是 H67 公式；Q7 部署不是冻结 FP；AND-only 缺静默和 Motion 项，不能作同功能基线 | 可做独立注意力叶，原提案 T4–5；仍需 ep34 份额和脏率门 |
| 选择性共享部分和；Codex 独立包 | 按目的掩码构造公共中间和，再选择性分发 | Mailman/RSR++/CSE 已有代数；在线构造、编码、取数和生命周期不能免费 | 原包 T5；进一步形成 V1，但独立审稿降至 T4 |
| 端口、权重驻留、上下文队列、广播；NPU、microarch、plans | 是合法强底座，也可用于公平对照 | 加任务条件、换广播顺序、增加缓存不自动构成主创新 | 降为实现及消融项；上一轮父槽重算和双权重槽同样降级 |
| 膜／权重融合 CIM、10T SRAM、多行读；CIM/Round 3 | 可借近存状态与局部服务思路 | TS1N28 1RW 不提供这些端口和位线操作；冻结 PSN 也不是 IF/LIF reset | 原版淘汰，不包装成数字 28 nm 结果 |
| 混 T、首次发放早停、膜态驻留 | T=10 神经元与 T=2 注意力窗要分开核算 | 满秩 T×T PSN 不能任意截时间；C3 没有公平加速比 | 状态服务／覆盖，不能凑第三条加速 |
| 事件栈、解码插零、外部光流芯片 | 输入组织、数据搬运和系统边界可借 | 当前解码器已铺满执行 lane；现成空间局部性与插零跳过不能重命名为本工作原语 | 不作为新岛主贡献 |
| OpenROAD 18 行评分板、握手收据 | 用作隔离骨架和工程状态记录 | 无合法本工艺宏与闭环，不能引用为 ASIC PPA；骨架条目不是 18 项创新 | 无主机制评分，PPA 准入为 0 |

α-XNOR、Prosperity、SumMerge、RSR++、UCNN、BNFF、DeltaCNN、DT-SCNN 等先验的访问状态在各条记录旁列明；未取得全文的 FACT、Comperity、OPN 等不写成“已排除覆盖”。公开机制可以借，但必须明确承认其已有部分。

## 为什么 V2 比旧 C2 更值得继续

冻结评估实际采用当前域动态 BN。MLP 路径为 `sn1 → FC1 → BN1 → sn2 → FC2 → BN2`；`sn2` 是完整时间矩阵 PSN。当前 FC1 无偏置。传统强实现可以把统计融入 FC1，再保存原始隐层；也可丢弃隐层，统计就绪后重放二值输入、重算 FC1。已有 BN 融合与本仓旧物化消除都应包含在对照中。源码定位：[MLP](/home/zhumd/work/sdformer_codex/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py:165)、[评估 no_running 设置](/home/zhumd/work/sdformer_codex/SDformer/third_party/SDformerFlow/eval_DSEC_flow_SNN.py:198)、[满秩 PSN](/home/zhumd/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/atlif_ternary_psn/atlif_ternary_psn.py:343)。

设完整 BN 域二值矩阵为 `S[N,C]`，静态有效权重为 `W[C,H]`，`Y=SW`。可以先计算：

```
k_i    = Σ_n S_ni
G_ij   = Σ_n S_ni S_nj        （G_ii = k_i）
sum_h  = Σ_i w_ih k_i
sumsq_h = Σ_i w_ih² k_i + 2 Σ_(i<j) w_ih w_jh G_ij
```

**候选因果路径：** 完整二值源 → k/G → 通道矩统计 → 统计封存 → 二值源重放 → 一次正常 FC1 → BN1 → 完整 PSN → 二值消费者。它没有消除全域屏障，也不依赖阈值早停成立。初步电路边界只到 `count/sum/sumsq`，后续 BN／PSN 不计成已测收益。

均值与协方差的线性传播是已知数学；创新只能落在“源统计如何比首遍宽输出执行便宜，以及如何按有限端口实现”上。[Analytic Variance Propagation](https://arxiv.org/html/1803.10560v1) 已给矩传播关系；[BNFF，MLSys 2019](https://mlsys.org/Conferences/2019/doc/2019/18.pdf) 已融合前层与统计，仍生成前层数值。这两者都需要明确引用。

只看输入各通道发放次数不够。例如 `S=[10,01,11,00]` 与 `S=[11,11,00,00]` 都有 `k=[2,2]`，取 `w=[2,−1]` 时，两者平方和分别为 6 和 2。完整交叉共发放项决定结果，不能按 16 源块各自统计后相加冒充全 C 统计。

### 已有捕获给出的机会与反证

以下为已有 ep34 捕获元数据和本轮固定 `sample_id=0` 的 CPU 源统计。不是新模型 profile、VCS 周期、PPA 或 AEE。原捕获仍有独立结果审计门，CRC 和计数恒等式通过只证明本次源解析及统计自洽。

| 层级 | FC1 的 N / C / H | N/C | 上三角 G 裸载荷 | 每调用完整二次型收缩项 |
|---|---|---:|---:|---:|
| stage0，2 块 | 192000 / 96 / 384 | 2000 | 10,476 B | 1,787,904 |
| stage1，2 块 | 48000 / 192 / 768 | 250 | 37,056 B | 14,229,504 |
| stage2，6 块 | 12000 / 384 / 1536 | 31.25 | 129,360 B | 113,541,120 |
| stage3，2 块 | 3000 / 768 / 3072 | 3.90625 | 442,944 B | 907,149,312 |

这里 N 已包括 T=10。stage0 的真实空间形状是 120×160；不使用交接示意图中的更大尺寸替代封存捕获。容量是数学裸载荷，未含端口、宏利用率、权重、输入位图或缓冲。

| 预选样本 0 的浅层 FC1 | 每行平均活跃源 | 非对角 G 非零比例 | 不含对角的源对计数更新 | 加入对角后的更新数 |
|---|---:|---:|---:|---:|
| stage0 block0 | 16.9435 | 100% | 35,101,829 | 38,354,975 |
| stage0 block1 | 14.7469 | 100% | 25,728,665 | 28,560,075 |

样本 0 中全部 12 个 FC1 的非对角 G 非零比例为 **98.6236%–100%**。因此本轮直接否决“G 也稀疏，可以跳掉多数收缩”的变体。输入稀疏仅影响如何形成 G，不代表 G 本身稀疏。

浅层一份二值源裸位图为 2.304 MB，一份 Acc24 隐层为 221.184 MB；这些是张量载荷，不能写成需要这么大片上 SRAM，也不能直接写成减少的外存流量。若预存所有上三角 `w_i*w_j`，仅 signed16 系数表就需 3.576 MB，不能当免费常量。若把交叉项因子 2 预折进去，极值 32768 需要 signed17；也可保留 16 位乘积，在宽累加端左移。

完整统计脚本与结果：[screen_source_moments.py](/home/zhumd/work/sdformer_codex/ideafromai/research/innovation_first_audit_20260906/scripts/screen_source_moments.py)、[v2_sample0_moments.json](/home/zhumd/work/sdformer_codex/ideafromai/research/innovation_first_audit_20260906/records/v2_sample0_moments.json)、[元数据推导](/home/zhumd/work/sdformer_codex/ideafromai/research/innovation_first_audit_20260906/records/v2_existing_geometry.json)。校验包含原载荷 SHA、选中帧 CRC／顺序／形状／二值性、对角计数、总二阶矩恒等式及固定子矩阵独立整数参考；另一代理复核了脚本和直方图恒等式，没有假称第二次独立捕获。

### 电路必须真正解决的问题

1. **形成 G：** 稀疏源对更新需要计数器读改写和热点处理；位平面 AND-popcount 需要转置缓冲、源重读和批量更新。单计数器串行更新不能用“二值很省”遮盖。
2. **宽收缩：** G 是多位计数，权重乘积也是多位，不能把收缩当普通二值条件加。片上保存全系数表与运行时生成系数是两种都要计费的选择。
3. **同资源对照：** 取“FC1＋统计融合＋保存 raw”与“FC1＋统计融合＋丢弃后重算”两者较优。逻辑输出块宽度不能代替物理执行 lane 数；生产 C2 前端的真实 slice／bank 接口要单独核。
4. **整数与冻结 FP 分开：** 实数恒等式不保证 cuDNN／TF32 归约逐位等价。最小统计组件可以采用共同精确整数合同；完整网络采用它需要明确的新部署与 AEE。Acc24 不缩位，sum/sumsq 另扩宽。

送审后的定向排雷又核了 9 篇主要来源，其中 6 篇取得相关原文。**二值 Gram 的核心也不是新原语**：[BISMO](https://arxiv.org/html/1806.08862) 已有位打包 AND-popcount 和局部累积，[Bishop，ISCA 2025](https://arxiv.org/html/2505.12281) 已有二值对积与多位结果驻留。将它们用于 `SᵀS` 是本轮迁移推断，不能把应用名称当发明。V2 的余项是完整“源统计替换输出统计遍”的电路及其同资源收益。OPN、ACBN、FlexAcc 的全文缺口仍未关闭；8 次定向检索未发现已核直接命中，不证明首创。详见 [定向先验审阅](/home/zhumd/work/sdformer_codex/ideafromai/research/innovation_first_audit_20260906/records/v2_targeted_prior_audit.json)。

### 已具体化的计数微结构草案

对浅层 C=96，可分成三个 32 源块，共六个上三角块；每次输入 64 个 token 的位平面。用七个 64 位 AND-popcount 形成一组增量，把七个 18 位 G 计数打包进一个 128 位宏字，整体读改写。为让同字计数共享一根源列，按源行对齐：三个对角块共 `3×Σ_(r=1..32)ceil(r/7)=270` 字，三个非对角块共 `3×32×ceil(32/7)=480` 字，合计 **750 字、12,000 B**，含行尾填充。它比 10,476 B 的无填充裸计数多，换取较简单的源读出组织。

计数容量上可映射到六个 128×128 宏，共 12 KiB：两个逻辑 bank 各三个宏、各用 375/384 字。另需单份 768 B 位平面缓冲，双份为 1536 B；六宏剩余 288 B 装不下，故 12 KiB 不是整岛存储。按字奇偶交错读写仍须满足实际读延迟、popcount 流水延迟和写回相位；连续逐字读时，读到写的延迟为奇数拍才自然落在不同 bank，停顿／跳过后仍须显式避让。下一 64-token 块重用同址前要完成上次写回；完整 750 字连续扫描可以在满足依赖时不逐块排空，完整 BN 域尾必须排空。**这只是容量与端口调度草案，没有“六宏已映射通过”或“一字一拍实测”。** 旧九宏适配器按宽度切片且地址存在限制，不能直接引用其闭环为新六宏深度级联背书。

比较时要给稀疏计数基线同样的宏字打包／更新合并能力，并纳入 BISMO/Bishop 类局部累积，而非拿每个源对单独访问宏的弱基线。七个 popcount、转置缓冲与后续宽收缩均是新增资源。生产 C2 的 16-lane slice、八源 bank 和逻辑输出块也不能混成“一拍 96 个物理输出 lane”。这些约束使草案可被反驳和核算，但打包与 bank 交错本身仍不足以提高创新评分。

[这条微结构的独立评审](/home/zhumd/work/sdformer_codex/ideafromai/research/innovation_first_audit_20260906/records/review_v2_packed_counter_by_c1.json) 给打包／交错本身 T2，整体 V2 仍维持 T6。评审针对明确给出的结构消息，未冒称审过尚未完成的 RTL 或全部设计。

另已归档 [设计者的微结构草案](/home/zhumd/work/sdformer_codex/ideafromai/research/innovation_first_audit_20260906/records/v2_circuit_refinement.json)，含稠密／稀疏计数、运行时权重对生成、宽收缩及输出统计强基线的参数公式。设计者未给自己独立评分；宽乘法器复用和完整端口排程仍为未闭合项。协调者补正了输入传输公式：连续位流可跨 token 打包时按总位数收费，不能直接套逐 token 对齐的拍数。

## C1 这轮为什么仍未过关

V1 确实比只找真实父行更完整，但 [SumMerge，ICS 2021](https://cwfletcher.github.io/content/research/2021.ics.summerge.paper.pdf) 已构造非原始行的递归共享交集。相对 [Prosperity，HPCA 2025](https://arxiv.org/html/2503.03379v1) 的缺口，不等于相对全部先验的缺口。[RSR++](https://arxiv.org/html/2411.06360v3) 也不能削弱成仅合并完整相同签名的基线。V1 只能继续寻找在线、有界构造与单口服务的净增量；当前独立 T4 不足以升主机制。

卡片中的四份 96-lane 双 INT8 基为 432 B，只在输入确为 signed INT8 时成立。生产 C1 源接口实际为 96×12 位；任意两个 12 位源求和一般要 13 位，四基裸载荷应为 624 B，不能直接说沿用 12 位接口便能保真。这是送审后补出的位宽门，不是已实现设计。

V5 试图把 V2 的统计思路进一步改为卷积的有限位移相关，并通过边缘精确修正保留零填充函数。数学可成立，但封存瓶颈卷积为 **768→768、15×20、T=10**；96 是执行宽度。对称合并后仍需 7,373,184 个相关计数，全驻留 12 位裸载荷约 11.06 MB；宽收缩约 56.63 亿项，边缘修正覆盖 22% 输出位置。原 Acc24 隐层载荷仅 6.912 MB。这些代价足以拒绝当前 C1 挂载。形状来自 [ep34 operator_runtime.json](/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/results/m1458_m1434_motion_ep34_live93_unified_hardware_capture_s40_r1_20260831/operator_runtime.json)，另有权重导出形状交叉核验；均不作为周期准入。

高分辨率 patch 残差卷积是 96 通道，统计方向可能重新有空间，但那是另一挂载；不能把换层后的机会算成原 C1 的同工作负载优势。V5 已保留否决记录，防止以后又按错误 96 通道重启。

## 新岛和改模型方案的具体边界

Motion-XOR 可以继续做 H67 的数字映射，但原 Card C 还不能直接作为强创新。现有 Q7 叶的共静默、Motion 系数与冻结浮点配置不同；两种算术身份必须分开。AND-only 省掉了功能项，不是合法同功能对照。即使某 K 为零，其分数仍会影响 Shiftmax 的行最大值和分母；只能证明自身 value 贡献为零，不能顺带删分数。相关核对位于 [bsa_attention.py](/home/zhumd/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/bsa_attention.py:1765) 和 [Q7 叶](/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/rtl_h67/h67_motionxor_score_q7.sv:18)。旧约 0.6% 注意力份额仍不是 ep34 新测的系统许可。

V4 也没有保住最初较高的作者预期。[DT-SCNN，2024](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2024.1418115/full) 已用共享投影／膜值与多个阈值产生多组二值特征。因此只有“可达模式直接消费 FC2 聚合”还可作为电路问题。必须在同一个共享投影新模型上比较独立阈值＋原稀疏 FC2，不能把模型绑权收益全记给新电路。m=4、T=10 时跨时间模式最多可达 16 种，普通 m+1 表容量假设不可靠；表的位宽也比原 INT8 向量大。

## 下一步的最小证据门

优先推进 V2 的浅层统计电路，避免再次围绕旧 C1/C2 模型倍率打磨。

| 门 | 必须拿到的证据 | 否决条件 |
|---|---|---|
| 数值合同 | 同域二值源、同 INT8 有效权重、未饱和 Acc24 点积及扩宽 count/sum/sumsq；独立大整数参考 | 需要忽略非对角或采用近似才能成立，却继续声称无损 |
| 电路组织 | 1RW 的 G 更新／转置方案、系数生成与宽收缩、源重放和全域完成状态 | 只有二次型公式；所有收益来自旧融合或免费 SRAM 端口 |
| 同资源判断 | 完整费用下与强融合保存／重算取优比较；物理 lane、bank、宏与带宽一致 | 控制、计数、收缩和源重读吞掉收益，无清楚能量优势 |
| 工作负载 | 浅层两个 FC1 的预声明多场景、低中高活跃及最坏共发放；源证据独立审计 | 只挑单样本、忽略高活跃或跨块交叉项 |
| 可发表电路结果 | 同工作负载 VCS＋DC/PT＋Formality 闭环，合法存储宏、setup/hold 与面积／能量 | CPU 项数或裸容量被改写为 RTL 速度／PPA |
| 网络表述 | 若声称部署于 ep34，另明确算术与 valid825 AEE；否则只写统计组件 | 偷用冻结 FP 精度、整网 FPS、组件倍率相乘 |

[TCAS-II 官方指南](https://ieee-cas.org/publication/TCAS-II/guidelines-author) 要求显著新结果及清楚的先验差异。五页中正文最多 4.5 页，最后半页即最后整栏留给参考文献。当前适合收窄成一个经过验证的统计电路问题，不能靠堆四个组件得到强录用判断。

## 覆盖、复核和研究状态

原始入口、Grok Bot 各轮、Grok46 全包、Card A/B/C、独立假说、合同、microarch 和 plans 均有文件 SHA 与逐条定位。旧 MANIFEST 漏列的 Round 2–4 文件已由实际磁盘枚举补入覆盖。后面台账保留不同文件对同一机制的不同假设及评分，没有把它们合成一座加速器。

本轮原始文件精读与候选交叉评审已完成；新电路主创新目标仍未达到投稿准入。现有源统计与新卡片均为研究证据，`RTL_SPEEDUP_ADMISSION=0`、`PPA_ADMISSION=0`。未修改主稿贡献句、生产 RTL、docs/359 或 H81。原目录入口只更新研究优先级，并保留审阅前快照与哈希；不改写旧实验结果。


## 逐文件逐条审阅

重复机制保留各文件出现位置；记录数不代表独立创新数量。N=新意，F=冻结适配，H=实现可行性，T=TCAS-II主机制潜力；均为0–10。空分表示入口或工程史料。


### HANDOFF_NEXT_AGENT_20260905.md

精读完成；SHA256 `de13311a3e0c7dd03a8c13d1696117d368dad08f9a26dab588653e116219674c`。

全文384行精读；身份、准入和范围合同，不新增独立点子；C1/C2/CardA/B/C/共享部分和重复机制在所属原文件逐条审。接手HEAD为历史值；ep35 capture与ep34当前身份不得混成同次实测。


### INDEX.json

精读完成；SHA256 `c0b5bf254881f99757b59327cfeb08551bf93282e2cc961d2653d95296357f8f`。

全文；机器入口含上一轮重算/暂存优先级，已被用户新指令否定。待汇总后新增本轮入口，不修改历史数值。 审阅后仅修订研究入口；本条 SHA 对应已保留的审阅前快照，当前入口 SHA 另记。


### MANIFEST.txt

精读完成；SHA256 `37a777ce688f57b16654bf27565ff53ebf7e566a39b76ff5b9aa826c72f224fc`。

全文；当前清单遗漏若干R2–R4文件，本次42文件inventory按磁盘枚举补全。 审阅后仅修订研究入口；本条 SHA 对应已保留的审阅前快照，当前入口 SHA 另记。


### MANIFEST_GROKBOT.txt

精读完成；SHA256 `e0d5f6d3c525237496579295b525d40a7962f9b5770fd6ceff19067468a2c761`。

全文；/workspace为历史打包路径，规范根始终/home/zhumd/work/sdformer_codex/ideafromai。


### README.md

精读完成；SHA256 `a36a87c481c23b7e1ceb986d4fc9fcd9a78b59d5fe44ade846533572c6b5ec72`。

全文；上一轮有收益不等于有主创新，旧优先级应降为实现探索；捕获93条与ep34的对应须回具体证据，不能凭入口一句话完成证明。 审阅后仅修订研究入口；本条 SHA 对应已保留的审阅前快照，当前入口 SHA 另记。


### README_GROKBOT.md

精读完成；SHA256 `57b9c0bc2666ebdb7476723ab0c702137f3a9daa8e888408801fceb5cbacf763`。

全文；目录导航，R1–R4包各文件独立审。


### README_TREE_GROKBOT.md

精读完成；SHA256 `2ef5b387b7246826dd6d7ddcc4090da1d808b699682cb84b146dd0d30883ec35`。

全文；隔离树导航和复制边界，不授权合并。原模块清单在microarch及各卡复审。


### codex_cards/CARD_A_OP_STW.md

精读完成；SHA256 `153716592427d0009538002889f4478e76115018bd97d4faabb716a9e862ae28`。


#### CARD-A · OP-STW / DPP-Skip

定位：22–46；48–53。N/F/H/T：**1 / 2 / 8 / 1**。

- **原机制：** 64 tile 并行比较 abs(flow_cur-flow_prev)>TH_W 或 event_cnt>TH_E，注册 wake。

- **最近先验与访问状态：** [Liu et al., An Ultra-High Performance and Scalable Optical Flow Hardware Accelerator Based on FPGA for Autonomous Driving, TCAS-I 2025](https://ieeexplore.ieee.org/document/11030859/)；本轮 IEEE 原始摘要已核：方向预测、可配置 PE、金字塔管线；未取得全文

- **冻结适配：** 低；ep34 无前置当前 flow 预测器，跳过非零 tile 改变函数。

- **真正增量：** 卡片落实的电路仅减法、绝对值、比较与或门；没有方向估计、预测、旧结果恢复或精确跳过证明。

- **代价与反证：** flow_cur 如是当前待算输出便形成因果环；scalar flow无法表示二维方向；8-bit有符号相减范围[-255,255]应先扩到9 bit，不能8 bit减后取abs；TH_W有符号参数需非负合同；没有ready/backpressure定义；三例TB不能验证这些边界或AEE。

- **迁移方式：** 可当独立活动比较器骨架；输入改为因果可得的预测/已验证脏位，明确二维流语义、9-bit差分与握手。不能靠定向TB通过宣称新C1已成立。

- **结论：** 原案不能入冻结主线；可作为算法重构入口


### codex_cards/CARD_B_HBG_RP.md

精读完成；SHA256 `d2aa283ace8a772d24b2825a60975cd643b53dd2059545d32df0e2ec9e557cbd`。


#### CardB.goal · HBG-RP packetizer goal/interface

定位：L11–39。N/F/H/T：**1 / 0 / 9 / 1**。

- **原机制：** 把幅值拆为非零门控 g 与 int8 实值 p，g 控制时钟/请求。

- **最近先验与访问状态：** [Activity Pruning for Efficient Spiking Neural Networks, Bu/Shi/Yu, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/fa12d67b5939c37ea8a4659c31a08d2c-Paper-Conference.pdf)；一手 PDF §3.2 式10–11及阈值吸收段已核；OpenReview 被拦后由 NeurIPS 正式库取得

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。

- **真正增量：** 冻结身份下退化为已有 active 位与权重加法；没有新增幅值原语。

- **代价与反证：** eps=1 严格 > 会删 ±1；量化/截断与门控均需新 AEE。常量 θ 若≤eps 甚至整层静默；最小负数 abs 必须9位。

- **迁移方式：** 冻结版本删除 p，只作为普通二值协议/门控基线；实值版本需重训并另建身份。

- **结论：** 否决为主机制；禁止套 ep34


#### CardB.abs · absolute threshold gate

定位：L40–44。N/F/H/T：**1 / 0 / 9 / 1**。

- **原机制：** 计算 abs(amp)>EPS，gate 决定 p 与 clock-enable。

- **最近先验与访问状态：** [Activity Pruning for Efficient Spiking Neural Networks, Bu/Shi/Yu, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/fa12d67b5939c37ea8a4659c31a08d2c-Paper-Conference.pdf)；一手 PDF §3.2 式10–11及阈值吸收段已核；OpenReview 被拦后由 NeurIPS 正式库取得

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。

- **真正增量：** 冻结身份下退化为已有 active 位与权重加法；没有新增幅值原语。

- **代价与反证：** int8 -128 的绝对值需扩到9位；EPS有符号可为负，参数合法域需明确；EPS=1抑制±1且不等于零跳过。

- **迁移方式：** 冻结版本删除 p，只作为普通二值协议/门控基线；实值版本需重训并另建身份。

- **结论：** 否决为主机制；禁止套 ep34


#### CardB.handshake · gp_valid / pe_clk_en handshake

定位：L43–46。N/F/H/T：**0 / 10 / 10 / 0**。

- **原机制：** 注册有效与输出数据/使能对齐。

- **最近先验与访问状态：** 通用硬件性能计数器/验证方法；非论文机制

- **冻结适配：** 可用于冻结图，但必须绑定同身份、同工作负载。

- **真正增量：** 验证基础设施，不是电路贡献。

- **代价与反证：** 组合g与注册gp_valid若混周期会错配；clock enable不能替代安全ICG实现、reset和流控。

- **迁移方式：** 用于筛查和审计，不计新机制。

- **结论：** 保留工程必需项


#### CardB.TB.1 · amp=0

定位：L48。N/F/H/T：**0 / 10 / 10 / 0**。

- **原机制：** 计数、CSV、可关闭消融与对照接口。

- **最近先验与访问状态：** 通用硬件性能计数器/验证方法；非论文机制

- **冻结适配：** 可用于冻结图，但必须绑定同身份、同工作负载。

- **真正增量：** 验证基础设施，不是电路贡献。

- **代价与反证：** 仅这些case不足：缺-128、极端EPS、连续包/valid气泡/复位相位；更不证明ep34身份或PPA。

- **迁移方式：** 用于筛查和审计，不计新机制。

- **结论：** 保留工程必需项


#### CardB.TB.2 · amp=±1

定位：L49。N/F/H/T：**0 / 10 / 10 / 0**。

- **原机制：** 计数、CSV、可关闭消融与对照接口。

- **最近先验与访问状态：** 通用硬件性能计数器/验证方法；非论文机制

- **冻结适配：** 可用于冻结图，但必须绑定同身份、同工作负载。

- **真正增量：** 验证基础设施，不是电路贡献。

- **代价与反证：** 仅这些case不足：缺-128、极端EPS、连续包/valid气泡/复位相位；更不证明ep34身份或PPA。

- **迁移方式：** 用于筛查和审计，不计新机制。

- **结论：** 保留工程必需项


#### CardB.TB.3 · amp=±2

定位：L50。N/F/H/T：**0 / 10 / 10 / 0**。

- **原机制：** 计数、CSV、可关闭消融与对照接口。

- **最近先验与访问状态：** 通用硬件性能计数器/验证方法；非论文机制

- **冻结适配：** 可用于冻结图，但必须绑定同身份、同工作负载。

- **真正增量：** 验证基础设施，不是电路贡献。

- **代价与反证：** 仅这些case不足：缺-128、极端EPS、连续包/valid气泡/复位相位；更不证明ep34身份或PPA。

- **迁移方式：** 用于筛查和审计，不计新机制。

- **结论：** 保留工程必需项


#### CardB.TB.4 · invalid

定位：L51。N/F/H/T：**0 / 10 / 10 / 0**。

- **原机制：** 计数、CSV、可关闭消融与对照接口。

- **最近先验与访问状态：** 通用硬件性能计数器/验证方法；非论文机制

- **冻结适配：** 可用于冻结图，但必须绑定同身份、同工作负载。

- **真正增量：** 验证基础设施，不是电路贡献。

- **代价与反证：** 仅这些case不足：缺-128、极端EPS、连续包/valid气泡/复位相位；更不证明ep34身份或PPA。

- **迁移方式：** 用于筛查和审计，不计新机制。

- **结论：** 保留工程必需项


### codex_cards/CARD_C_MX3P_DIRTY_SCORE.md

精读完成；SHA256 `daa9295ce64e5b8537cd3308941d4e4870b368f5221d23a041b530ad2e8e3e1f`。


#### CARDC-ATTENTION · MX3P + dirty Motion-XOR attention island

定位：8–37。N/F/H/T：**3 / 6 / 7 / 4**。

- **原机制：** 完整Motion-XOR score＋peer供给＋严格score memo＋Kzero value服务，作为同一attention岛

- **最近先验与访问状态：** [FireFly-T](https://arxiv.org/abs/2505.12771)；PRIMARY_FULLTEXT_RELEVANT_SECTIONS_READ_IN_PRECEDING_RESEARCH; FPGA attention/transpose，未复核期刊年份

- **冻结适配：** 算子结构适配；原文Q7 /64、/4不是冻结FP的0.02、0.125

- **真正增量：** 增加既有位运算项与peer供给；尚无超过直接实现的机制

- **代价与反证：** 行13/21混Q7与冻结；行16/36 AND-only功能不同；行23分母依赖待定义；三popcount+缓存本身新意薄

- **迁移方式：** 重写函数合同并选完整MX基线，先证明分母与peer/窗口依赖，再考虑1RW映射

- **结论：** DRAFT_REQUIRES_REWRITE


#### CARDC-NEURON · Mixed-horizon binary ATLIF / membrane firewall

定位：39–41。N/F/H/T：**2 / 6 / 6 / 3**。

- **原机制：** 神经元内部保存T状态，仅向外发送二值fire

- **最近先验与访问状态：** [Chen/Chang Spike-IAND-Former](https://arxiv.org/abs/2503.19643)；NOT_ACCESSED_THIS_AUDIT; T≤4 unroll 来自本地摘录

- **冻结适配：** 出口二值正确；原文leak/reset错误，实际fullrank A*x+b

- **真正增量：** 类型/速率转换；尚未给新神经元执行机制

- **代价与反证：** T10不能从T≤4 LIF muxunroll推导廉价；内部非经典递推膜

- **迁移方式：** 重写PSN完整时间矩阵服务，二值边界为事实非创新

- **结论：** SUPPORT_ONLY_CORRECT_SEMANTICS


### codex_cards/README_GROKBOT.md

精读完成；SHA256 `97f876662ebd94c4163cc84ba8118845f535c108877f2df216906efd6bd4641e`。

全文；卡片导航，未来C–G代号没有对应完整新机制，不当已形成idea。


### codex_independent_20260905/01_review_and_selective_shared_reduction.md

精读完成；SHA256 `54a732727884920d4825469bf247b0e0814520d217aa95f299739f4dd41c0d8e`。


#### IND01-C1 · C1 exact-subset product capture

定位：11–21。N/F/H/T：**1 / 8 / 8 / 2**。

- **原机制：** 完整行parent+源残差复用

- **最近先验与访问状态：** [Prosperity, HPCA 2025](https://arxiv.org/html/2503.03379v1)；NOT_ACCESSED_THIS_SUBAUDIT; 由本地记录及主代理负责全文核；不得假定无限容量

- **冻结适配：** 现有部署底座合法，非新方向

- **真正增量：** 只有1RW/有限服务实现差异待证明

- **代价与反证：** 不能称Prosperity无限缓存；当前倍率包含继承机制

- **迁移方式：** 作为强对照与底座

- **结论：** BASELINE_ONLY


#### IND01-C2 · C2 TSBG / Gustavson broadcast

定位：23–29。N/F/H/T：**1 / 9 / 8 / 2**。

- **原机制：** group-major共享weight row并更新独立目的

- **最近先验与访问状态：** [Eyeriss v2, JETCAS 2019](https://people.csail.mit.edu/emer/media/papers/2019.04.jetcas.eyeriss_v2.pdf)；NOT_ACCESSED_THIS_AUDIT; 本地作者全文链接

- **冻结适配：** 适配自然+1二值流，协议sign定向覆盖

- **真正增量：** 实现前请求抑制/共享控制，未形成新计算图

- **代价与反证：** 广播/loop交换先验强；低复用能量反向；不能用单K1

- **迁移方式：** 作为公平基线；新贡献要减少归约或证明免算

- **结论：** BASELINE_ONLY


#### IND01-SR · 动态公共部分和选择性构造与广播

定位：35–85;94–142;144–176。N/F/H/T：**4 / 6 / 5 / 5**。

- **原机制：** 按B个目的token列签名分组，先归约多source共享权重和，再更新独立acc；只对多source多消费者且净收益组物化

- **最近先验与访问状态：** [The Mailman algorithm, Liberty and Zucker](https://cs.yale.edu/homes/el327/papers/mailmanAlgorithm.pdf)；PRIMARY_FULLTEXT_ALGORITHM_READ; A=UP、有限字母列编码与预处理已核

- **冻结适配：** 二值输入和同weight身份适配；跨非线性不适配；整数重排不保证ep34 FP位等价

- **真正增量：** 可能新增在线有限组的选择与退休，使未出现为输入行的公共组合也可共享

- **代价与反证：** Mailman代数已有；Phi/Comperity是强近邻；K8已可并行归约，少加法不等于少周期；构造/路由/Acc24溢出计费

- **迁移方式：** 用公平TSBG与K8归约作同宏同端口强对照；不堆叠多组件

- **结论：** RESEARCH_HYPOTHESIS_NOT_ADMITTED


#### IND01-SIGNED · 完整signed目的签名扩展

定位：86–92。N/F/H/T：**2 / 3 / 5 / 1**。

- **原机制：** B4三值签名最多80种，研究相反签名规范化

- **最近先验与访问状态：** [The Mailman algorithm, Liberty and Zucker](https://cs.yale.edu/homes/el327/papers/mailmanAlgorithm.pdf)；PRIMARY_FULLTEXT_ALGORITHM_READ; A=UP、有限字母列编码与预处理已核

- **冻结适配：** 自然ATLIF非零为+1；不用合成负号制造收益

- **真正增量：** 可选协议覆盖

- **代价与反证：** 3^B签名、取负溢出和目的独立性

- **迁移方式：** 先binary主路径，有真实signed需求再扩

- **结论：** DEFER_NO_NATIVE_MOTIVATION


### codex_independent_20260905/02_literature_and_alternative_ideas.md

精读完成；SHA256 `87e184feb17cbd00d73f122d92e884c511fe9727225b43470e2958646e671079`。


#### IND02-PAT · 小容量精确模式结果表

定位：13–23。N/F/H/T：**2 / 6 / 6 / 3**。

- **原机制：** xW=pW+(x−p)W，同weight tile模式表加双向残差

- **最近先验与访问状态：** [Phi, ISCA 2025](https://arxiv.org/html/2505.10909v1)；PRIMARY_FULLTEXT_SECTIONS_READ; §2.4/3.1，分区模式表、双向校正、无模式回退已核

- **冻结适配：** 整数部署可精确，FP重排需另证

- **真正增量：** 小2/4/8表和在线构造或减存储，参数变小不是机制

- **代价与反证：** Phi已分区、双向校正、无模式回退；校准/表读/建表/weightID成本；Comperity未全文排除

- **迁移方式：** 作为强对照或找到新生命周期机制后再提

- **结论：** PRIOR_DENSE_SUPPORT_ONLY


#### IND02-EBPC · 按bank独立解码无损权重存储

定位：25–37。N/F/H/T：**3 / 6 / 6 / 3**。

- **原机制：** 小块随机寻址压缩，按请求bank解码，raw回退

- **最近先验与访问状态：** [Extended Bit-Plane Compression, JETCAS 2019](https://arxiv.org/html/1908.11645v2)；NOT_ACCESSED_THIS_AUDIT; 官方 RTL https://github.com/pulp-platform/stream-ebpc 已在本地列出

- **冻结适配：** 适配冻结派生INT8流，非ep34 FP训练算术

- **真正增量：** 稀疏随机读与大块压缩冲突可研究

- **代价与反证：** EBPC主要feature maps；目录、宽decoder、读放大、宏阶梯计费

- **迁移方式：** 配套存储候选，需超过强随机访问压缩baseline

- **结论：** SUPPORT_OR_WEAK_MAIN


#### IND02-ACC · 融合signed归约—累加器

定位：39–49。N/F/H/T：**2 / 8 / 8 / 2**。

- **原机制：** 符号修正并入多操作数压缩/归约，窄增量后更新Acc24

- **最近先验与访问状态：** [Parallel Accurate Minifloat MACCs for Neural Network Inference on Versal FPGAs](https://researchportal.tuni.fi/files/135714845/Parallel_Accurate_Minifloat_MACCs_for_Neural_Network_Inference_on_Versal_FPGAs.pdf)；NOT_ACCESSED_THIS_AUDIT; 本地作者全文链接

- **冻结适配：** 八个±INT8增量12bit安全；自然+1动机弱

- **真正增量：** 或减少中间扩展/切换，属实现优化

- **代价与反证：** 综合器可能已优化；setup/hold相互影响；不能只比串行链

- **迁移方式：** 作为PE底座，报告真实面积频率能耗

- **结论：** SUPPORT_ONLY


#### IND02-WIN · 有限窗口独立token重组

定位：51–61。N/F/H/T：**3 / 7 / 6 / 3**。

- **原机制：** 已到达8/16描述符中选相关B4，按原ID提交

- **最近先验与访问状态：** [Gamma, ASPLOS 2021](https://www.guowei.zone/files/2021.gamma.asplos.pdf)；NOT_ACCESSED_THIS_AUDIT; 本地作者全文链接

- **冻结适配：** 同算子独立token内调序可保图，不能跨BN/PSN依赖

- **真正增量：** 有限窗口选择共享伙伴

- **代价与反证：** affinity/reuse-distance已有；等待、容量、尾包、公平性、输出重排计费

- **迁移方式：** 可作SR上游，不能重复记TSBG读复用收益

- **结论：** SUPPORT_ONLY


#### IND02-LIT01 · C1 exact-subset product capture

定位：67。N/F/H/T：**1 / 8 / 8 / 2**。

- **原机制：** 完整行parent+源残差复用

- **最近先验与访问状态：** [Prosperity, HPCA 2025](https://arxiv.org/html/2503.03379v1)；NOT_ACCESSED_THIS_SUBAUDIT; 由本地记录及主代理负责全文核；不得假定无限容量

- **冻结适配：** 现有部署底座合法，非新方向

- **真正增量：** 只有1RW/有限服务实现差异待证明

- **代价与反证：** 不能称Prosperity无限缓存；当前倍率包含继承机制

- **迁移方式：** 作为强对照与底座

- **结论：** BASELINE_ONLY


#### IND02-LIT02 · FEATHER计算与重排

定位：68。N/F/H/T：**2 / 6 / 4 / 2**。

- **原机制：** 归约过程中重排

- **最近先验与访问状态：** [FEATHER, ISCA 2024](https://arxiv.org/abs/2405.13170)；NOT_ACCESSED_THIS_AUDIT; 本地引文

- **冻结适配：** 适配自然+1二值流，协议sign定向覆盖

- **真正增量：** 当前未证有transpose瓶颈

- **代价与反证：** 整阵列改造大，官方RTL非即成PPA

- **迁移方式：** 作为公平基线；新贡献要减少归约或证明免算

- **结论：** SUPPORT_ONLY


#### IND02-LIT03 · ELSA elastic first-response

定位：69。N/F/H/T：**2 / 2 / 5 / 2**。

- **原机制：** token逐层弹性先到先算

- **最近先验与访问状态：** [ELSA, ISCA 2026](https://arxiv.org/html/2605.20802v1)；NOT_ACCESSED_THIS_SUBAUDIT; 动态 BN 屏障由本地源码独立确认

- **冻结适配：** 24 dynamic BN需整域统计；fullrankPSN需完整T输入

- **真正增量：** 仅局部队列打包可保留

- **代价与反证：** 分类first-correct不能变成denseflow完整结果；无全网闭环

- **迁移方式：** 只保局部ready/valid与BAER，不写弹性网络加速

- **结论：** REJECT_SYSTEM_CLAIM


#### IND02-LIT04 · SpikeX空间时间weight sharing

定位：70。N/F/H/T：**1 / 9 / 8 / 2**。

- **原机制：** group-major共享weight row并更新独立目的

- **最近先验与访问状态：** [SpikeX](https://arxiv.org/html/2505.12292v1)；NOT_ACCESSED_THIS_AUDIT; 本地引文

- **冻结适配：** 适配自然+1二值流，协议sign定向覆盖

- **真正增量：** 实现前请求抑制/共享控制，未形成新计算图

- **代价与反证：** 广播/loop交换先验强；低复用能量反向；不能用单K1

- **迁移方式：** 作为公平基线；新贡献要减少归约或证明免算

- **结论：** BASELINE_ONLY


#### IND02-LIT05 · Eyeriss v2互连

定位：71。N/F/H/T：**1 / 9 / 8 / 2**。

- **原机制：** group-major共享weight row并更新独立目的

- **最近先验与访问状态：** [Eyeriss v2, JETCAS 2019](https://people.csail.mit.edu/emer/media/papers/2019.04.jetcas.eyeriss_v2.pdf)；NOT_ACCESSED_THIS_AUDIT; 本地作者全文链接

- **冻结适配：** 适配自然+1二值流，协议sign定向覆盖

- **真正增量：** 实现前请求抑制/共享控制，未形成新计算图

- **代价与反证：** 广播/loop交换先验强；低复用能量反向；不能用单K1

- **迁移方式：** 作为公平基线；新贡献要减少归约或证明免算

- **结论：** BASELINE_ONLY


#### IND02-LIT06 · Sparseloop成本模型

定位：72。N/F/H/T：**0 / 9 / 8 / 0**。

- **原机制：** 稀疏优化分类与成本建模

- **最近先验与访问状态：** [Sparseloop, MICRO 2022](https://sparseloop.mit.edu/documents/2022-micro-sparseloop.pdf)；NOT_ACCESSED_THIS_AUDIT; 模型不是 RTL 证据

- **冻结适配：** 分析方法适配，非执行机制

- **真正增量：** 无硬件创新

- **代价与反证：** 不能替代同workload VCS/DC/PT/FM

- **迁移方式：** 用于立项kill-gate

- **结论：** EVALUATION_METHOD


#### IND02-LIT07 · UCNN重复权重复用

定位：73。N/F/H/T：**2 / 5 / 5 / 2**。

- **原机制：** 相同权重共享计算

- **最近先验与访问状态：** [UCNN, ISCA 2018](https://www.kartikhegde.net/media/UCNN_ISCA.pdf)；NOT_ACCESSED_THIS_AUDIT; 本地作者全文链接

- **冻结适配：** 固定权重可核重复；binary源乘本来是加法

- **真正增量：** 除乘法收益不能照搬

- **代价与反证：** 权重重复与模式收益未知；不新增训练约束

- **迁移方式：** 核跨归约共享，强对照SR

- **结论：** PRIOR_REFERENCE


#### IND02-LIT08 · LoAS FTP mixed-T fiber join

定位：74。N/F/H/T：**2 / 6 / 5 / 3**。

- **原机制：** 将T位向量压成非静默fiber并inner-join

- **最近先验与访问状态：** [LoAS, MICRO 2024](https://arxiv.org/abs/2407.14073)；PRIMARY_FULLTEXT_RELEVANT_SECTIONS_READ_IN_PRECEDING_RESEARCH; LIF reset 与 temporal fibers；不等同 PSN

- **冻结适配：** 二值T2/T10可分开封包；不能照搬LoAS LIF/reset或稀疏W

- **真正增量：** 不同fiber宽度和Motion项服务

- **代价与反证：** 共静默不能从删掉的坐标丢失；索引/merge开销与1RW端口未核

- **迁移方式：** 只作配套索引组织；强对照直接bitset/popcount

- **结论：** SUPPORT_ONLY


#### IND02-LIT09 · Inter-frame dirty-tile encoder skip

定位：75。N/F/H/T：**3 / 2 / 4 / 3**。

- **原机制：** 无新事件的tile复用上次深层输出

- **最近先验与访问状态：** [DeltaCNN, Parger et al., CVPR 2022](https://arxiv.org/html/2203.03996v2)；PRIMARY_FULLTEXT_SECTIONS_READ; §3.1、§4、§8.4，静态融合 BN 与累计 FP 误差已核

- **冻结适配：** 原始输入不变不保证含BN/窗口/PSN依赖的输出不变

- **真正增量：** 普通CBinfer/delta推到网络

- **代价与反证：** no_running BN造成全域耦合；卷积halo/跨窗/跨时依赖；FP长期误差

- **迁移方式：** 必须新全局消费者证书；参见 BNREFINE，不直接免算

- **结论：** REJECT_SIMPLE_VERSION


#### IND02-LIT10 · 4bit / dual-side W sparsity

定位：76。N/F/H/T：**2 / 0 / 5 / 1**。

- **原机制：** 剪权/降精度换更稀疏PE服务

- **最近先验与访问状态：** [BBS/BitVert](https://arxiv.org/abs/2409.05227)；NOT_ACCESSED_THIS_AUDIT; 本地引文

- **冻结适配：** 不适配冻结quant=false

- **真正增量：** 新训练/部署Pareto，不是无损硬件

- **代价与反证：** 现有dense W不能套97%稀疏；无重训练不等于无损

- **迁移方式：** 另行明确模型身份才可重训，此轮不迁移

- **结论：** NEW_MODEL_ONLY


### codex_independent_20260905/README.md

精读完成；SHA256 `15e1dfb6c3dc4aa66ddace3e9ddfd8e46ca20e8db53aa08bdb5bee48686a3182`。


#### INDREADME-SR · 动态公共部分和选择性构造与广播

定位：11;17–18。N/F/H/T：**4 / 6 / 5 / 5**。

- **原机制：** 按B个目的token列签名分组，先归约多source共享权重和，再更新独立acc；只对多source多消费者且净收益组物化

- **最近先验与访问状态：** [The Mailman algorithm, Liberty and Zucker](https://cs.yale.edu/homes/el327/papers/mailmanAlgorithm.pdf)；PRIMARY_FULLTEXT_ALGORITHM_READ; A=UP、有限字母列编码与预处理已核

- **冻结适配：** 二值输入和同weight身份适配；跨非线性不适配；整数重排不保证ep34 FP位等价

- **真正增量：** 可能新增在线有限组的选择与退休，使未出现为输入行的公共组合也可共享

- **代价与反证：** Mailman代数已有；Phi/Comperity是强近邻；K8已可并行归约，少加法不等于少周期；构造/路由/Acc24溢出计费

- **迁移方式：** 用公平TSBG与K8归约作同宏同端口强对照；不堆叠多组件

- **结论：** RESEARCH_HYPOTHESIS_NOT_ADMITTED


### contracts/ATLIF_contract_r1_grokbot.md

精读完成；SHA256 `ea261d77d90bbdac5384f88f05b52872f6e7109038fdfac0813edf56f8561a5f`。


#### contract.field.1 · int8 amp

定位：L11。N/F/H/T：**1 / 0 / 9 / 1**。

- **原机制：** 把幅值拆为非零门控 g 与 int8 实值 p，g 控制时钟/请求。

- **最近先验与访问状态：** [Activity Pruning for Efficient Spiking Neural Networks, Bu/Shi/Yu, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/fa12d67b5939c37ea8a4659c31a08d2c-Paper-Conference.pdf)；一手 PDF §3.2 式10–11及阈值吸收段已核；OpenReview 被拦后由 NeurIPS 正式库取得

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。

- **真正增量：** 冻结身份下退化为已有 active 位与权重加法；没有新增幅值原语。

- **代价与反证：** eps=1 严格 > 会删 ±1；量化/截断与门控均需新 AEE。常量 θ 若≤eps 甚至整层静默；最小负数 abs 必须9位。

- **迁移方式：** 冻结版本删除 p，只作为普通二值协议/门控基线；实值版本需重训并另建身份。

- **结论：** 否决为主机制；禁止套 ep34


#### contract.field.2 · eps=1

定位：L12。N/F/H/T：**1 / 0 / 9 / 1**。

- **原机制：** 把幅值拆为非零门控 g 与 int8 实值 p，g 控制时钟/请求。

- **最近先验与访问状态：** [Activity Pruning for Efficient Spiking Neural Networks, Bu/Shi/Yu, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/fa12d67b5939c37ea8a4659c31a08d2c-Paper-Conference.pdf)；一手 PDF §3.2 式10–11及阈值吸收段已核；OpenReview 被拦后由 NeurIPS 正式库取得

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。

- **真正增量：** 冻结身份下退化为已有 active 位与权重加法；没有新增幅值原语。

- **代价与反证：** eps=1 严格 > 会删 ±1；量化/截断与门控均需新 AEE。常量 θ 若≤eps 甚至整层静默；最小负数 abs 必须9位。

- **迁移方式：** 冻结版本删除 p，只作为普通二值协议/门控基线；实值版本需重训并另建身份。

- **结论：** 否决为主机制；禁止套 ep34


#### contract.field.3 · NO θ absorption

定位：L13。N/F/H/T：**1 / 0 / 9 / 1**。

- **原机制：** 把幅值拆为非零门控 g 与 int8 实值 p，g 控制时钟/请求。

- **最近先验与访问状态：** [Activity Pruning for Efficient Spiking Neural Networks, Bu/Shi/Yu, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/fa12d67b5939c37ea8a4659c31a08d2c-Paper-Conference.pdf)；一手 PDF §3.2 式10–11及阈值吸收段已核；OpenReview 被拦后由 NeurIPS 正式库取得

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。

- **真正增量：** 冻结身份下退化为已有 active 位与权重加法；没有新增幅值原语。

- **代价与反证：** eps=1 严格 > 会删 ±1；量化/截断与门控均需新 AEE。常量 θ 若≤eps 甚至整层静默；最小负数 abs 必须9位。

- **迁移方式：** 冻结版本删除 p，只作为普通二值协议/门控基线；实值版本需重训并另建身份。

- **结论：** 否决为主机制；禁止套 ep34


#### contract.field.4 · hard gate

定位：L14。N/F/H/T：**1 / 0 / 9 / 1**。

- **原机制：** 把幅值拆为非零门控 g 与 int8 实值 p，g 控制时钟/请求。

- **最近先验与访问状态：** [Activity Pruning for Efficient Spiking Neural Networks, Bu/Shi/Yu, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/fa12d67b5939c37ea8a4659c31a08d2c-Paper-Conference.pdf)；一手 PDF §3.2 式10–11及阈值吸收段已核；OpenReview 被拦后由 NeurIPS 正式库取得

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。

- **真正增量：** 冻结身份下退化为已有 active 位与权重加法；没有新增幅值原语。

- **代价与反证：** eps=1 严格 > 会删 ±1；量化/截断与门控均需新 AEE。常量 θ 若≤eps 甚至整层静默；最小负数 abs 必须9位。

- **迁移方式：** 冻结版本删除 p，只作为普通二值协议/门控基线；实值版本需重训并另建身份。

- **结论：** 否决为主机制；禁止套 ep34


#### contract.field.5 · trunc/sat payload

定位：L15。N/F/H/T：**1 / 0 / 9 / 1**。

- **原机制：** 把幅值拆为非零门控 g 与 int8 实值 p，g 控制时钟/请求。

- **最近先验与访问状态：** [Activity Pruning for Efficient Spiking Neural Networks, Bu/Shi/Yu, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/fa12d67b5939c37ea8a4659c31a08d2c-Paper-Conference.pdf)；一手 PDF §3.2 式10–11及阈值吸收段已核；OpenReview 被拦后由 NeurIPS 正式库取得

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。

- **真正增量：** 冻结身份下退化为已有 active 位与权重加法；没有新增幅值原语。

- **代价与反证：** eps=1 严格 > 会删 ±1；量化/截断与门控均需新 AEE。常量 θ 若≤eps 甚至整层静默；最小负数 abs 必须9位。

- **迁移方式：** 冻结版本删除 p，只作为普通二值协议/门控基线；实值版本需重训并另建身份。

- **结论：** 否决为主机制；禁止套 ep34


#### contract.field.6 · no soft gate

定位：L16。N/F/H/T：**1 / 0 / 9 / 1**。

- **原机制：** 把幅值拆为非零门控 g 与 int8 实值 p，g 控制时钟/请求。

- **最近先验与访问状态：** [Activity Pruning for Efficient Spiking Neural Networks, Bu/Shi/Yu, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/fa12d67b5939c37ea8a4659c31a08d2c-Paper-Conference.pdf)；一手 PDF §3.2 式10–11及阈值吸收段已核；OpenReview 被拦后由 NeurIPS 正式库取得

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。

- **真正增量：** 冻结身份下退化为已有 active 位与权重加法；没有新增幅值原语。

- **代价与反证：** eps=1 严格 > 会删 ±1；量化/截断与门控均需新 AEE。常量 θ 若≤eps 甚至整层静默；最小负数 abs 必须9位。

- **迁移方式：** 冻结版本删除 p，只作为普通二值协议/门控基线；实值版本需重训并另建身份。

- **结论：** 否决为主机制；禁止套 ep34


#### contract.packet · atlif_gp_t

定位：L18–24。N/F/H/T：**1 / 0 / 9 / 1**。

- **原机制：** 把幅值拆为非零门控 g 与 int8 实值 p，g 控制时钟/请求。

- **最近先验与访问状态：** [Activity Pruning for Efficient Spiking Neural Networks, Bu/Shi/Yu, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/fa12d67b5939c37ea8a4659c31a08d2c-Paper-Conference.pdf)；一手 PDF §3.2 式10–11及阈值吸收段已核；OpenReview 被拦后由 NeurIPS 正式库取得

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。

- **真正增量：** 冻结身份下退化为已有 active 位与权重加法；没有新增幅值原语。

- **代价与反证：** eps=1 严格 > 会删 ±1；量化/截断与门控均需新 AEE。常量 θ 若≤eps 甚至整层静默；最小负数 abs 必须9位。

- **迁移方式：** 冻结版本删除 p，只作为普通二值协议/门控基线；实值版本需重训并另建身份。

- **结论：** 否决为主机制；禁止套 ep34


#### contract.claim · absorbable downgrade clause

定位：L26–28。N/F/H/T：**1 / 0 / 9 / 1**。

- **原机制：** 把幅值拆为非零门控 g 与 int8 实值 p，g 控制时钟/请求。

- **最近先验与访问状态：** [Activity Pruning for Efficient Spiking Neural Networks, Bu/Shi/Yu, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/fa12d67b5939c37ea8a4659c31a08d2c-Paper-Conference.pdf)；一手 PDF §3.2 式10–11及阈值吸收段已核；OpenReview 被拦后由 NeurIPS 正式库取得

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。

- **真正增量：** 文档本身承认可吸收时应降级；ep34正满足该条件。

- **代价与反证：** eps=1 严格 > 会删 ±1；量化/截断与门控均需新 AEE。常量 θ 若≤eps 甚至整层静默；最小负数 abs 必须9位。

- **迁移方式：** 冻结版本删除 p，只作为普通二值协议/门控基线；实值版本需重训并另建身份。

- **结论：** 按本条直接降级为基线


### contracts/ablation_ladder_grokbot.md

精读完成；SHA256 `870256a3d9bfcd1da8e876fd3a54edc812a2c0183e46927c5d8541a94c09b0fe`。


#### ladder.C1.1 · Exact capture wrapper

定位：L4。N/F/H/T：**1 / 4 / 8 / 1**。

- **原机制：** 在既有有限容量父行捕获岛入口加wake/match/ROI条件。

- **最近先验与访问状态：** [Prosperity HPCA 2025; current C1 1RW capture](https://arxiv.org/html/2503.03379v1)；本地C1合同与原文机制已核；历史周期模型不得转RTL

- **冻结适配：** 原C1求值可作为精确对照；新增谓词若预测则破坏精确。

- **真正增量：** 外围使能包装；不是新的乘积复用。

- **代价与反证：** 门不成立的输入若复用/填充将改变图；全部mask是否足以保真没有证明。

- **迁移方式：** 保留原岛与强父行策略作比较；不能作为新标题。

- **结论：** 旧执行底座


#### ladder.C1.2 · Dual-side bitmap sparsity

定位：L5。N/F/H/T：**1 / 6 / 8 / 1**。

- **原机制：** 位图压缩/解码并跳过激活与权重零值。

- **最近先验与访问状态：** [FireFly-S; SCNN ISCA 2017](https://arxiv.org/abs/1708.04485)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。 激活侧适配；冻结权重并未授权剪枝。

- **真正增量：** row_live/bit-skip 与既有 TSBG 已覆盖基本收益。

- **代价与反证：** 权重稀疏训练另身份；索引、负载与随机 bank 冲突收费。

- **迁移方式：** 作为强稀疏基线，不凭双侧命名立项。

- **结论：** 仅基线


#### ladder.C1.3 · OP-STW

定位：L6。N/F/H/T：**3 / 1 / 5 / 2**。

- **原机制：** 用上一时刻 coarse flow/差分和事件计数预测 tile wake，睡眠 tile 复用输出。

- **最近先验与访问状态：** [MotionDeltaCNN ICCV 2023; motion-gated vision SoCs](https://openaccess.thecvf.com/content/ICCV2023/html/Parger_MotionDeltaCNN_Sparse_CNN_Inference_of_Frame_Differences_in_Moving_Camera_ICCV_2023_paper.html)；CVF 一手摘要和 PDF 方法段已读

- **冻结适配：** H67 无现成 coarse-flow 迭代支路；no_running 动态 BN 的全域均值/方差使局部不变不能直接推出消费者不变。

- **真正增量：** 加 motion 名称不足；若以严格消费者证书取代启发式 wake 才形成新执行问题。

- **代价与反证：** 当前帧 flow 不能提前免费获得；睡眠输出未必精确，必须计预测器、warp、缓存与 AEE。

- **迁移方式：** 原版只可有损新评价；精确重构必须全域 BN 依赖闭合。

- **结论：** 原版不作为冻结主线


#### ladder.C1.4 · PRRC

定位：L7。N/F/H/T：**2 / 0 / 6 / 1**。

- **原机制：** 粗到细金字塔预算/ROI，预算耗尽后复用或传播填充。

- **最近先验与访问状态：** [MotionDeltaCNN; optical-flow coarse-to-fine methods](https://openaccess.thecvf.com/content/ICCV2023/html/Parger_MotionDeltaCNN_Sparse_CNN_Inference_of_Frame_Differences_in_Moving_Camera_ICCV_2023_paper.html)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** 冻结 U-Net 非迭代金字塔搜索；不含预算耗尽语义。

- **真正增量：** 计数器与 ROI 不是新原语；须有算法可证明的停止条件。

- **代价与反证：** 强制 REUSE/PROP 本质近似，打 overflow 标志不能使输出精确。

- **迁移方式：** 不套 ep34；新算法需训练/valid825。

- **结论：** 原版否决


#### ladder.C1.5 · OGEC

定位：L8。N/F/H/T：**3 / 0 / 5 / 1**。

- **原机制：** 以前后向一致性判遮挡，匹配路径精算，未匹配邻域传播。

- **最近先验与访问状态：** [MotionDeltaCNN; occlusion-aware optical-flow algorithms](https://openaccess.thecvf.com/content/ICCV2023/html/Parger_MotionDeltaCNN_Sparse_CNN_Inference_of_Frame_Differences_in_Moving_Camera_ICCV_2023_paper.html)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** 冻结图无免费反向 flow 或邻域填充替代神经层。

- **真正增量：** 稀疏门控加光流谓词；尚无独特正确性原语。

- **代价与反证：** 遮挡处往往最难，传播可能损伤 AEE；额外反向推理/检测必须收费。

- **迁移方式：** 只作有损新评价，不叫 exact path 加速。

- **结论：** 原版否决


#### ladder.C2.1 · MFBD

定位：L11。N/F/H/T：**1 / 6 / 7 / 1**。

- **原机制：** 同权重行向相邻时间/运动上下文广播。

- **最近先验与访问状态：** [ELSA ISCA 2026; Eyeriss; existing TSBG](https://arxiv.org/html/2605.20802v1)；ELSA 一手 HTML 相关 dataflow 已读

- **冻结适配：** 若每 consumer 保持真实 token/时步身份可合法。

- **真正增量：** 现有 TSBG 已在 mem_req 前按 source-group 共享取权，剩余只是分组选择。

- **代价与反证：** 不能重复计权重请求收益；运动假设不存在，广播网络与上下文容量不能免费增加。

- **迁移方式：** 必须改变实际计算内容才重立项；现版降为 TSBG 对照。

- **结论：** 仅既有机制换名


#### ladder.C2.2 · HBG-RP

定位：L12。N/F/H/T：**1 / 0 / 9 / 1**。

- **原机制：** 把幅值拆为非零门控 g 与 int8 实值 p，g 控制时钟/请求。

- **最近先验与访问状态：** [Activity Pruning for Efficient Spiking Neural Networks, Bu/Shi/Yu, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/fa12d67b5939c37ea8a4659c31a08d2c-Paper-Conference.pdf)；一手 PDF §3.2 式10–11及阈值吸收段已核；OpenReview 被拦后由 NeurIPS 正式库取得

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。

- **真正增量：** 冻结身份下退化为已有 active 位与权重加法；没有新增幅值原语。

- **代价与反证：** eps=1 严格 > 会删 ±1；量化/截断与门控均需新 AEE。常量 θ 若≤eps 甚至整层静默；最小负数 abs 必须9位。

- **迁移方式：** 冻结版本删除 p，只作为普通二值协议/门控基线；实值版本需重训并另建身份。

- **结论：** 否决为主机制；禁止套 ep34


#### ladder.C2.3 · ADP-MAC

定位：L13。N/F/H/T：**2 / 2 / 6 / 2**。

- **原机制：** 激活与权重逐位跳零，空闲 slot donation。

- **最近先验与访问状态：** [BitFusion MICRO 2018; FireFly-v2](https://arxiv.org/abs/1712.01507)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。 激活只有1位，双侧幅值位稀疏前提消失。

- **真正增量：** bit-serial/slot donation 已有；不能重新命名为光流原语。

- **代价与反证：** 零位译码、移位、交换网、符号扩展与最坏精度时间都收费。

- **迁移方式：** 只保留权重位串行作为同资源对照；无新意暂不重做。

- **结论：** 不作主机制


#### ladder.C2.4 · ARM-Acc or MFBD

定位：L14。N/F/H/T：**3 / 0 / 5 / 1**。

- **原机制：** K 个运动方向假设独立累加，按证据赢家提交。

- **最近先验与访问状态：** Multi-hypothesis optical-flow algorithms; generic speculative accumulation；原文件给出来源；本轮未独立取得全文

- **冻结适配：** 冻结 H67 没有 K=4/8 搜索假设或赢家提交算子。

- **真正增量：** 4 个 consumer 是独立 token，不可改称竞争假设。

- **代价与反证：** 新增假设计算、状态与选择改变模型；不能将输家丢弃说成精确。

- **迁移方式：** 先有新算法与 AEE 才可考虑；当前剔除。

- **结论：** 冻结下否决


#### ladder.C2.5 · SP-Gate

定位：L15。N/F/H/T：**2 / 1 / 6 / 2**。

- **原机制：** 按 attention mass 选择 hot token 并抑制 cold token 的后续 FC 计算。

- **最近先验与访问状态：** [SparseVideoGen; token pruning / early-exit](https://arxiv.org/abs/2502.01776)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** 冻结 attention 的 gate 不是 token 整体无贡献证明；FC/BN/残差仍有消费者。

- **真正增量：** 普通分数剪枝；没有精确停止证书。

- **代价与反证：** 剪 token 改动态 BN 统计及稠密光流输出；分数读取/排序成本。

- **迁移方式：** 仅新 AEE Pareto；精确版必须完整依赖界。

- **结论：** 冻结无损路线否决


### microarch/05_C1star_C2star_microarch_sketches.md

精读完成；SHA256 `6cb8b5c59c7c6bd8f7ec830e7334ae8a62c255d3ae4fe98b18e70623cbbcf268`。


#### microarch.M1 · OP-STW

定位：L43–48。N/F/H/T：**3 / 1 / 5 / 2**。

- **原机制：** 用上一时刻 coarse flow/差分和事件计数预测 tile wake，睡眠 tile 复用输出。

- **最近先验与访问状态：** [MotionDeltaCNN ICCV 2023; motion-gated vision SoCs](https://openaccess.thecvf.com/content/ICCV2023/html/Parger_MotionDeltaCNN_Sparse_CNN_Inference_of_Frame_Differences_in_Moving_Camera_ICCV_2023_paper.html)；CVF 一手摘要和 PDF 方法段已读

- **冻结适配：** H67 无现成 coarse-flow 迭代支路；no_running 动态 BN 的全域均值/方差使局部不变不能直接推出消费者不变。

- **真正增量：** 加 motion 名称不足；若以严格消费者证书取代启发式 wake 才形成新执行问题。

- **代价与反证：** 当前帧 flow 不能提前免费获得；睡眠输出未必精确，必须计预测器、warp、缓存与 AEE。

- **迁移方式：** 原版只可有损新评价；精确重构必须全域 BN 依赖闭合。

- **结论：** 原版不作为冻结主线


#### microarch.M2 · PRRC

定位：L50–54。N/F/H/T：**2 / 0 / 6 / 1**。

- **原机制：** 粗到细金字塔预算/ROI，预算耗尽后复用或传播填充。

- **最近先验与访问状态：** [MotionDeltaCNN; optical-flow coarse-to-fine methods](https://openaccess.thecvf.com/content/ICCV2023/html/Parger_MotionDeltaCNN_Sparse_CNN_Inference_of_Frame_Differences_in_Moving_Camera_ICCV_2023_paper.html)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** 冻结 U-Net 非迭代金字塔搜索；不含预算耗尽语义。

- **真正增量：** 计数器与 ROI 不是新原语；须有算法可证明的停止条件。

- **代价与反证：** 强制 REUSE/PROP 本质近似，打 overflow 标志不能使输出精确。

- **迁移方式：** 不套 ep34；新算法需训练/valid825。

- **结论：** 原版否决


#### microarch.M3 · OGEC

定位：L56–60。N/F/H/T：**3 / 0 / 5 / 1**。

- **原机制：** 以前后向一致性判遮挡，匹配路径精算，未匹配邻域传播。

- **最近先验与访问状态：** [MotionDeltaCNN; occlusion-aware optical-flow algorithms](https://openaccess.thecvf.com/content/ICCV2023/html/Parger_MotionDeltaCNN_Sparse_CNN_Inference_of_Frame_Differences_in_Moving_Camera_ICCV_2023_paper.html)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** 冻结图无免费反向 flow 或邻域填充替代神经层。

- **真正增量：** 稀疏门控加光流谓词；尚无独特正确性原语。

- **代价与反证：** 遮挡处往往最难，传播可能损伤 AEE；额外反向推理/检测必须收费。

- **迁移方式：** 只作有损新评价，不叫 exact path 加速。

- **结论：** 原版否决


#### microarch.M4 · Exact capture wrapper

定位：L62–63。N/F/H/T：**1 / 4 / 8 / 1**。

- **原机制：** 在既有有限容量父行捕获岛入口加wake/match/ROI条件。

- **最近先验与访问状态：** [Prosperity HPCA 2025; current C1 1RW capture](https://arxiv.org/html/2503.03379v1)；本地C1合同与原文机制已核；历史周期模型不得转RTL

- **冻结适配：** 原C1求值可作为精确对照；新增谓词若预测则破坏精确。

- **真正增量：** 外围使能包装；不是新的乘积复用。

- **代价与反证：** 门不成立的输入若复用/填充将改变图；全部mask是否足以保真没有证明。

- **迁移方式：** 保留原岛与强父行策略作比较；不能作为新标题。

- **结论：** 旧执行底座


#### microarch.M5 · 统计/消融基础设施

定位：L65–66。N/F/H/T：**0 / 10 / 10 / 0**。

- **原机制：** 计数、CSV、可关闭消融与对照接口。

- **最近先验与访问状态：** 通用硬件性能计数器/验证方法；非论文机制

- **冻结适配：** 可用于冻结图，但必须绑定同身份、同工作负载。

- **真正增量：** 验证基础设施，不是电路贡献。

- **代价与反证：** TB 通过不等于VCS+DC/PT+Formality准入；有损参数与无损数字分开。

- **迁移方式：** 用于筛查和审计，不计新机制。

- **结论：** 保留工程必需项


#### microarch.N1 · HBG-RP

定位：L109–112。N/F/H/T：**1 / 0 / 9 / 1**。

- **原机制：** 把幅值拆为非零门控 g 与 int8 实值 p，g 控制时钟/请求。

- **最近先验与访问状态：** [Activity Pruning for Efficient Spiking Neural Networks, Bu/Shi/Yu, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/fa12d67b5939c37ea8a4659c31a08d2c-Paper-Conference.pdf)；一手 PDF §3.2 式10–11及阈值吸收段已核；OpenReview 被拦后由 NeurIPS 正式库取得

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。

- **真正增量：** 冻结身份下退化为已有 active 位与权重加法；没有新增幅值原语。

- **代价与反证：** eps=1 严格 > 会删 ±1；量化/截断与门控均需新 AEE。常量 θ 若≤eps 甚至整层静默；最小负数 abs 必须9位。

- **迁移方式：** 冻结版本删除 p，只作为普通二值协议/门控基线；实值版本需重训并另建身份。

- **结论：** 否决为主机制；禁止套 ep34


#### microarch.N2 · ADP-MAC

定位：L114–118。N/F/H/T：**2 / 2 / 6 / 2**。

- **原机制：** 激活与权重逐位跳零，空闲 slot donation。

- **最近先验与访问状态：** [BitFusion MICRO 2018; FireFly-v2](https://arxiv.org/abs/1712.01507)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。 激活只有1位，双侧幅值位稀疏前提消失。

- **真正增量：** bit-serial/slot donation 已有；不能重新命名为光流原语。

- **代价与反证：** 零位译码、移位、交换网、符号扩展与最坏精度时间都收费。

- **迁移方式：** 只保留权重位串行作为同资源对照；无新意暂不重做。

- **结论：** 不作主机制


#### microarch.N3a · ARM-Acc

定位：L120–122。N/F/H/T：**3 / 0 / 5 / 1**。

- **原机制：** K 个运动方向假设独立累加，按证据赢家提交。

- **最近先验与访问状态：** Multi-hypothesis optical-flow algorithms; generic speculative accumulation；原文件给出来源；本轮未独立取得全文

- **冻结适配：** 冻结 H67 没有 K=4/8 搜索假设或赢家提交算子。

- **真正增量：** 4 个 consumer 是独立 token，不可改称竞争假设。

- **代价与反证：** 新增假设计算、状态与选择改变模型；不能将输家丢弃说成精确。

- **迁移方式：** 先有新算法与 AEE 才可考虑；当前剔除。

- **结论：** 冻结下否决


#### microarch.N3b · MFBD

定位：L124–126。N/F/H/T：**1 / 6 / 7 / 1**。

- **原机制：** 同权重行向相邻时间/运动上下文广播。

- **最近先验与访问状态：** [ELSA ISCA 2026; Eyeriss; existing TSBG](https://arxiv.org/html/2605.20802v1)；ELSA 一手 HTML 相关 dataflow 已读

- **冻结适配：** 若每 consumer 保持真实 token/时步身份可合法。

- **真正增量：** 现有 TSBG 已在 mem_req 前按 source-group 共享取权，剩余只是分组选择。

- **代价与反证：** 不能重复计权重请求收益；运动假设不存在，广播网络与上下文容量不能免费增加。

- **迁移方式：** 必须改变实际计算内容才重立项；现版降为 TSBG 对照。

- **结论：** 仅既有机制换名


#### microarch.N4 · SP-Gate

定位：L128–129。N/F/H/T：**2 / 1 / 6 / 2**。

- **原机制：** 按 attention mass 选择 hot token 并抑制 cold token 的后续 FC 计算。

- **最近先验与访问状态：** [SparseVideoGen; token pruning / early-exit](https://arxiv.org/abs/2502.01776)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** 冻结 attention 的 gate 不是 token 整体无贡献证明；FC/BN/残差仍有消费者。

- **真正增量：** 普通分数剪枝；没有精确停止证书。

- **代价与反证：** 剪 token 改动态 BN 统计及稠密光流输出；分数读取/排序成本。

- **迁移方式：** 仅新 AEE Pareto；精确版必须完整依赖界。

- **结论：** 冻结无损路线否决


#### microarch.N5 · 统计/消融基础设施

定位：L131–132。N/F/H/T：**0 / 10 / 10 / 0**。

- **原机制：** 计数、CSV、可关闭消融与对照接口。

- **最近先验与访问状态：** 通用硬件性能计数器/验证方法；非论文机制

- **冻结适配：** 可用于冻结图，但必须绑定同身份、同工作负载。

- **真正增量：** 验证基础设施，不是电路贡献。

- **代价与反证：** TB 通过不等于VCS+DC/PT+Formality准入；有损参数与无损数字分开。

- **迁移方式：** 用于筛查和审计，不计新机制。

- **结论：** 保留工程必需项


#### microarch.C1.ladder.1 · Exact capture wrapper

定位：L69。N/F/H/T：**1 / 4 / 8 / 1**。

- **原机制：** 在既有有限容量父行捕获岛入口加wake/match/ROI条件。

- **最近先验与访问状态：** [Prosperity HPCA 2025; current C1 1RW capture](https://arxiv.org/html/2503.03379v1)；本地C1合同与原文机制已核；历史周期模型不得转RTL

- **冻结适配：** 原C1求值可作为精确对照；新增谓词若预测则破坏精确。

- **真正增量：** 外围使能包装；不是新的乘积复用。

- **代价与反证：** 门不成立的输入若复用/填充将改变图；全部mask是否足以保真没有证明。

- **迁移方式：** 保留原岛与强父行策略作比较；不能作为新标题。

- **结论：** 旧执行底座


#### microarch.C1.ladder.2 · Dual-side bitmap sparsity

定位：L70。N/F/H/T：**1 / 6 / 8 / 1**。

- **原机制：** 位图压缩/解码并跳过激活与权重零值。

- **最近先验与访问状态：** [FireFly-S; SCNN ISCA 2017](https://arxiv.org/abs/1708.04485)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。 激活侧适配；冻结权重并未授权剪枝。

- **真正增量：** row_live/bit-skip 与既有 TSBG 已覆盖基本收益。

- **代价与反证：** 权重稀疏训练另身份；索引、负载与随机 bank 冲突收费。

- **迁移方式：** 作为强稀疏基线，不凭双侧命名立项。

- **结论：** 仅基线


#### microarch.C1.ladder.3 · OP-STW

定位：L71。N/F/H/T：**3 / 1 / 5 / 2**。

- **原机制：** 用上一时刻 coarse flow/差分和事件计数预测 tile wake，睡眠 tile 复用输出。

- **最近先验与访问状态：** [MotionDeltaCNN ICCV 2023; motion-gated vision SoCs](https://openaccess.thecvf.com/content/ICCV2023/html/Parger_MotionDeltaCNN_Sparse_CNN_Inference_of_Frame_Differences_in_Moving_Camera_ICCV_2023_paper.html)；CVF 一手摘要和 PDF 方法段已读

- **冻结适配：** H67 无现成 coarse-flow 迭代支路；no_running 动态 BN 的全域均值/方差使局部不变不能直接推出消费者不变。

- **真正增量：** 加 motion 名称不足；若以严格消费者证书取代启发式 wake 才形成新执行问题。

- **代价与反证：** 当前帧 flow 不能提前免费获得；睡眠输出未必精确，必须计预测器、warp、缓存与 AEE。

- **迁移方式：** 原版只可有损新评价；精确重构必须全域 BN 依赖闭合。

- **结论：** 原版不作为冻结主线


#### microarch.C1.ladder.4 · PRRC

定位：L72。N/F/H/T：**2 / 0 / 6 / 1**。

- **原机制：** 粗到细金字塔预算/ROI，预算耗尽后复用或传播填充。

- **最近先验与访问状态：** [MotionDeltaCNN; optical-flow coarse-to-fine methods](https://openaccess.thecvf.com/content/ICCV2023/html/Parger_MotionDeltaCNN_Sparse_CNN_Inference_of_Frame_Differences_in_Moving_Camera_ICCV_2023_paper.html)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** 冻结 U-Net 非迭代金字塔搜索；不含预算耗尽语义。

- **真正增量：** 计数器与 ROI 不是新原语；须有算法可证明的停止条件。

- **代价与反证：** 强制 REUSE/PROP 本质近似，打 overflow 标志不能使输出精确。

- **迁移方式：** 不套 ep34；新算法需训练/valid825。

- **结论：** 原版否决


#### microarch.C1.ladder.5 · OGEC

定位：L73。N/F/H/T：**3 / 0 / 5 / 1**。

- **原机制：** 以前后向一致性判遮挡，匹配路径精算，未匹配邻域传播。

- **最近先验与访问状态：** [MotionDeltaCNN; occlusion-aware optical-flow algorithms](https://openaccess.thecvf.com/content/ICCV2023/html/Parger_MotionDeltaCNN_Sparse_CNN_Inference_of_Frame_Differences_in_Moving_Camera_ICCV_2023_paper.html)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** 冻结图无免费反向 flow 或邻域填充替代神经层。

- **真正增量：** 稀疏门控加光流谓词；尚无独特正确性原语。

- **代价与反证：** 遮挡处往往最难，传播可能损伤 AEE；额外反向推理/检测必须收费。

- **迁移方式：** 只作有损新评价，不叫 exact path 加速。

- **结论：** 原版否决


#### microarch.C2.ladder.1 · MFBD

定位：L135。N/F/H/T：**1 / 6 / 7 / 1**。

- **原机制：** 同权重行向相邻时间/运动上下文广播。

- **最近先验与访问状态：** [ELSA ISCA 2026; Eyeriss; existing TSBG](https://arxiv.org/html/2605.20802v1)；ELSA 一手 HTML 相关 dataflow 已读

- **冻结适配：** 若每 consumer 保持真实 token/时步身份可合法。

- **真正增量：** 现有 TSBG 已在 mem_req 前按 source-group 共享取权，剩余只是分组选择。

- **代价与反证：** 不能重复计权重请求收益；运动假设不存在，广播网络与上下文容量不能免费增加。

- **迁移方式：** 必须改变实际计算内容才重立项；现版降为 TSBG 对照。

- **结论：** 仅既有机制换名


#### microarch.C2.ladder.2 · HBG-RP

定位：L136。N/F/H/T：**1 / 0 / 9 / 1**。

- **原机制：** 把幅值拆为非零门控 g 与 int8 实值 p，g 控制时钟/请求。

- **最近先验与访问状态：** [Activity Pruning for Efficient Spiking Neural Networks, Bu/Shi/Yu, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/fa12d67b5939c37ea8a4659c31a08d2c-Paper-Conference.pdf)；一手 PDF §3.2 式10–11及阈值吸收段已核；OpenReview 被拦后由 NeurIPS 正式库取得

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。

- **真正增量：** 冻结身份下退化为已有 active 位与权重加法；没有新增幅值原语。

- **代价与反证：** eps=1 严格 > 会删 ±1；量化/截断与门控均需新 AEE。常量 θ 若≤eps 甚至整层静默；最小负数 abs 必须9位。

- **迁移方式：** 冻结版本删除 p，只作为普通二值协议/门控基线；实值版本需重训并另建身份。

- **结论：** 否决为主机制；禁止套 ep34


#### microarch.C2.ladder.3 · ADP-MAC

定位：L137。N/F/H/T：**2 / 2 / 6 / 2**。

- **原机制：** 激活与权重逐位跳零，空闲 slot donation。

- **最近先验与访问状态：** [BitFusion MICRO 2018; FireFly-v2](https://arxiv.org/abs/1712.01507)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。 激活只有1位，双侧幅值位稀疏前提消失。

- **真正增量：** bit-serial/slot donation 已有；不能重新命名为光流原语。

- **代价与反证：** 零位译码、移位、交换网、符号扩展与最坏精度时间都收费。

- **迁移方式：** 只保留权重位串行作为同资源对照；无新意暂不重做。

- **结论：** 不作主机制


#### microarch.C2.ladder.4 · ARM-Acc

定位：L138。N/F/H/T：**3 / 0 / 5 / 1**。

- **原机制：** K 个运动方向假设独立累加，按证据赢家提交。

- **最近先验与访问状态：** Multi-hypothesis optical-flow algorithms; generic speculative accumulation；原文件给出来源；本轮未独立取得全文

- **冻结适配：** 冻结 H67 没有 K=4/8 搜索假设或赢家提交算子。

- **真正增量：** 4 个 consumer 是独立 token，不可改称竞争假设。

- **代价与反证：** 新增假设计算、状态与选择改变模型；不能将输家丢弃说成精确。

- **迁移方式：** 先有新算法与 AEE 才可考虑；当前剔除。

- **结论：** 冻结下否决


#### microarch.C2.ladder.5 · SP-Gate

定位：L139。N/F/H/T：**2 / 1 / 6 / 2**。

- **原机制：** 按 attention mass 选择 hot token 并抑制 cold token 的后续 FC 计算。

- **最近先验与访问状态：** [SparseVideoGen; token pruning / early-exit](https://arxiv.org/abs/2502.01776)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** 冻结 attention 的 gate 不是 token 整体无贡献证明；FC/BN/残差仍有消费者。

- **真正增量：** 普通分数剪枝；没有精确停止证书。

- **代价与反证：** 剪 token 改动态 BN 统计及稠密光流输出；分数读取/排序成本。

- **迁移方式：** 仅新 AEE Pareto；精确版必须完整依赖界。

- **结论：** 冻结无损路线否决


#### microarch.R1 · R1 OP-STW + HBG + OGEC/ARM

定位：L171–174。N/F/H/T：**1 / 0 / 9 / 1**。

- **原机制：** 把幅值拆为非零门控 g 与 int8 实值 p，g 控制时钟/请求。

- **最近先验与访问状态：** [Activity Pruning for Efficient Spiking Neural Networks, Bu/Shi/Yu, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/fa12d67b5939c37ea8a4659c31a08d2c-Paper-Conference.pdf)；一手 PDF §3.2 式10–11及阈值吸收段已核；OpenReview 被拦后由 NeurIPS 正式库取得

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。

- **真正增量：** 组合未修正任何失配；不能以多组件弥补机制缺口。

- **代价与反证：** eps=1 严格 > 会删 ±1；量化/截断与门控均需新 AEE。常量 θ 若≤eps 甚至整层静默；最小负数 abs 必须9位。

- **迁移方式：** 冻结版本删除 p，只作为普通二值协议/门控基线；实值版本需重训并另建身份。

- **结论：** 否决为主机制；禁止套 ep34


#### microarch.R2 · R2 HBG + ADP + ARM

定位：L176–177。N/F/H/T：**2 / 2 / 6 / 2**。

- **原机制：** 激活与权重逐位跳零，空闲 slot donation。

- **最近先验与访问状态：** [BitFusion MICRO 2018; FireFly-v2](https://arxiv.org/abs/1712.01507)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。 激活只有1位，双侧幅值位稀疏前提消失。

- **真正增量：** bit-serial/slot donation 已有；不能重新命名为光流原语。

- **代价与反证：** 零位译码、移位、交换网、符号扩展与最坏精度时间都收费。

- **迁移方式：** 只保留权重位串行作为同资源对照；无新意暂不重做。

- **结论：** 不作主机制


#### microarch.minimum.C1 · minimum OP-STW

定位：L180。N/F/H/T：**3 / 1 / 5 / 2**。

- **原机制：** 用上一时刻 coarse flow/差分和事件计数预测 tile wake，睡眠 tile 复用输出。

- **最近先验与访问状态：** [MotionDeltaCNN ICCV 2023; motion-gated vision SoCs](https://openaccess.thecvf.com/content/ICCV2023/html/Parger_MotionDeltaCNN_Sparse_CNN_Inference_of_Frame_Differences_in_Moving_Camera_ICCV_2023_paper.html)；CVF 一手摘要和 PDF 方法段已读

- **冻结适配：** H67 无现成 coarse-flow 迭代支路；no_running 动态 BN 的全域均值/方差使局部不变不能直接推出消费者不变。

- **真正增量：** 加 motion 名称不足；若以严格消费者证书取代启发式 wake 才形成新执行问题。

- **代价与反证：** 当前帧 flow 不能提前免费获得；睡眠输出未必精确，必须计预测器、warp、缓存与 AEE。

- **迁移方式：** 原版只可有损新评价；精确重构必须全域 BN 依赖闭合。

- **结论：** 原版不作为冻结主线


#### microarch.minimum.C2 · minimum HBG-RP

定位：L181–182。N/F/H/T：**1 / 0 / 9 / 1**。

- **原机制：** 把幅值拆为非零门控 g 与 int8 实值 p，g 控制时钟/请求。

- **最近先验与访问状态：** [Activity Pruning for Efficient Spiking Neural Networks, Bu/Shi/Yu, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/fa12d67b5939c37ea8a4659c31a08d2c-Paper-Conference.pdf)；一手 PDF §3.2 式10–11及阈值吸收段已核；OpenReview 被拦后由 NeurIPS 正式库取得

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。

- **真正增量：** 冻结身份下退化为已有 active 位与权重加法；没有新增幅值原语。

- **代价与反证：** ‘已能发表’无证据；4–6人周与代码可实现不构成期刊录用依据。

- **迁移方式：** 冻结版本删除 p，只作为普通二值协议/门控基线；实值版本需重训并另建身份。

- **结论：** 否决为主机制；禁止套 ep34


#### microarch.infrastructure.profiling · profiling

定位：L187。N/F/H/T：**0 / 10 / 10 / 0**。

- **原机制：** 计数、CSV、可关闭消融与对照接口。

- **最近先验与访问状态：** 通用硬件性能计数器/验证方法；非论文机制

- **冻结适配：** 可用于冻结图，但必须绑定同身份、同工作负载。

- **真正增量：** 验证基础设施，不是电路贡献。

- **代价与反证：** TB 通过不等于VCS+DC/PT+Formality准入；有损参数与无损数字分开。

- **迁移方式：** 用于筛查和审计，不计新机制。

- **结论：** 保留工程必需项


#### microarch.infrastructure.CSV · CSV

定位：L188。N/F/H/T：**0 / 10 / 10 / 0**。

- **原机制：** 计数、CSV、可关闭消融与对照接口。

- **最近先验与访问状态：** 通用硬件性能计数器/验证方法；非论文机制

- **冻结适配：** 可用于冻结图，但必须绑定同身份、同工作负载。

- **真正增量：** 验证基础设施，不是电路贡献。

- **代价与反证：** TB 通过不等于VCS+DC/PT+Formality准入；有损参数与无损数字分开。

- **迁移方式：** 用于筛查和审计，不计新机制。

- **结论：** 保留工程必需项


#### microarch.infrastructure.isolated baseline · isolated baseline

定位：L189。N/F/H/T：**0 / 10 / 10 / 0**。

- **原机制：** 计数、CSV、可关闭消融与对照接口。

- **最近先验与访问状态：** 通用硬件性能计数器/验证方法；非论文机制

- **冻结适配：** 可用于冻结图，但必须绑定同身份、同工作负载。

- **真正增量：** 验证基础设施，不是电路贡献。

- **代价与反证：** TB 通过不等于VCS+DC/PT+Formality准入；有损参数与无损数字分开。

- **迁移方式：** 用于筛查和审计，不计新机制。

- **结论：** 保留工程必需项


#### microarch.infrastructure.cost estimate · cost estimate

定位：L190。N/F/H/T：**0 / 10 / 10 / 0**。

- **原机制：** 计数、CSV、可关闭消融与对照接口。

- **最近先验与访问状态：** 通用硬件性能计数器/验证方法；非论文机制

- **冻结适配：** 可用于冻结图，但必须绑定同身份、同工作负载。

- **真正增量：** 验证基础设施，不是电路贡献。

- **代价与反证：** TB 通过不等于VCS+DC/PT+Formality准入；有损参数与无损数字分开。

- **迁移方式：** 用于筛查和审计，不计新机制。

- **结论：** 保留工程必需项


#### microarch.timeline · workload estimate consistency

定位：L75–85,L141–152。N/F/H/T：**0 / 0 / 0 / 0**。

- **原机制：** 计数、CSV、可关闭消融与对照接口。

- **最近先验与访问状态：** 通用硬件性能计数器/验证方法；非论文机制

- **冻结适配：** 可用于冻结图，但必须绑定同身份、同工作负载。

- **真正增量：** 验证基础设施，不是电路贡献。

- **代价与反证：** 8.5–12.5人周被写成一人2–3周，12–17人周写成3–4周，内部单位不一致；不可作为工期承诺。

- **迁移方式：** 用于筛查和审计，不计新机制。

- **结论：** 纠正计划口径


### plans/06_R1R2_full_plan_codex_ready.md

精读完成；SHA256 `6191555913b616a02707f749b4eda231846e170d826a4942965c5ef233febd1d`。


#### plan.scope.C1.1 · OP-STW

定位：L13。N/F/H/T：**3 / 1 / 5 / 2**。

- **原机制：** 用上一时刻 coarse flow/差分和事件计数预测 tile wake，睡眠 tile 复用输出。

- **最近先验与访问状态：** [MotionDeltaCNN ICCV 2023; motion-gated vision SoCs](https://openaccess.thecvf.com/content/ICCV2023/html/Parger_MotionDeltaCNN_Sparse_CNN_Inference_of_Frame_Differences_in_Moving_Camera_ICCV_2023_paper.html)；CVF 一手摘要和 PDF 方法段已读

- **冻结适配：** H67 无现成 coarse-flow 迭代支路；no_running 动态 BN 的全域均值/方差使局部不变不能直接推出消费者不变。

- **真正增量：** 加 motion 名称不足；若以严格消费者证书取代启发式 wake 才形成新执行问题。

- **代价与反证：** 当前帧 flow 不能提前免费获得；睡眠输出未必精确，必须计预测器、warp、缓存与 AEE。

- **迁移方式：** 原版只可有损新评价；精确重构必须全域 BN 依赖闭合。

- **结论：** 原版不作为冻结主线


#### plan.scope.C1.2 · PRRC

定位：L14。N/F/H/T：**2 / 0 / 6 / 1**。

- **原机制：** 粗到细金字塔预算/ROI，预算耗尽后复用或传播填充。

- **最近先验与访问状态：** [MotionDeltaCNN; optical-flow coarse-to-fine methods](https://openaccess.thecvf.com/content/ICCV2023/html/Parger_MotionDeltaCNN_Sparse_CNN_Inference_of_Frame_Differences_in_Moving_Camera_ICCV_2023_paper.html)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** 冻结 U-Net 非迭代金字塔搜索；不含预算耗尽语义。

- **真正增量：** 计数器与 ROI 不是新原语；须有算法可证明的停止条件。

- **代价与反证：** 强制 REUSE/PROP 本质近似，打 overflow 标志不能使输出精确。

- **迁移方式：** 不套 ep34；新算法需训练/valid825。

- **结论：** 原版否决


#### plan.scope.C1.3 · OGEC

定位：L15。N/F/H/T：**3 / 0 / 5 / 1**。

- **原机制：** 以前后向一致性判遮挡，匹配路径精算，未匹配邻域传播。

- **最近先验与访问状态：** [MotionDeltaCNN; occlusion-aware optical-flow algorithms](https://openaccess.thecvf.com/content/ICCV2023/html/Parger_MotionDeltaCNN_Sparse_CNN_Inference_of_Frame_Differences_in_Moving_Camera_ICCV_2023_paper.html)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** 冻结图无免费反向 flow 或邻域填充替代神经层。

- **真正增量：** 稀疏门控加光流谓词；尚无独特正确性原语。

- **代价与反证：** 遮挡处往往最难，传播可能损伤 AEE；额外反向推理/检测必须收费。

- **迁移方式：** 只作有损新评价，不叫 exact path 加速。

- **结论：** 原版否决


#### plan.scope.C1.4 · Exact capture wrapper

定位：L16。N/F/H/T：**1 / 4 / 8 / 1**。

- **原机制：** 在既有有限容量父行捕获岛入口加wake/match/ROI条件。

- **最近先验与访问状态：** [Prosperity HPCA 2025; current C1 1RW capture](https://arxiv.org/html/2503.03379v1)；本地C1合同与原文机制已核；历史周期模型不得转RTL

- **冻结适配：** 原C1求值可作为精确对照；新增谓词若预测则破坏精确。

- **真正增量：** 外围使能包装；不是新的乘积复用。

- **代价与反证：** 门不成立的输入若复用/填充将改变图；全部mask是否足以保真没有证明。

- **迁移方式：** 保留原岛与强父行策略作比较；不能作为新标题。

- **结论：** 旧执行底座


#### plan.scope.C2.1 · HBG-RP

定位：L19。N/F/H/T：**1 / 0 / 9 / 1**。

- **原机制：** 把幅值拆为非零门控 g 与 int8 实值 p，g 控制时钟/请求。

- **最近先验与访问状态：** [Activity Pruning for Efficient Spiking Neural Networks, Bu/Shi/Yu, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/fa12d67b5939c37ea8a4659c31a08d2c-Paper-Conference.pdf)；一手 PDF §3.2 式10–11及阈值吸收段已核；OpenReview 被拦后由 NeurIPS 正式库取得

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。

- **真正增量：** 冻结身份下退化为已有 active 位与权重加法；没有新增幅值原语。

- **代价与反证：** eps=1 严格 > 会删 ±1；量化/截断与门控均需新 AEE。常量 θ 若≤eps 甚至整层静默；最小负数 abs 必须9位。

- **迁移方式：** 冻结版本删除 p，只作为普通二值协议/门控基线；实值版本需重训并另建身份。

- **结论：** 否决为主机制；禁止套 ep34


#### plan.scope.C2.2 · ADP-MAC

定位：L20。N/F/H/T：**2 / 2 / 6 / 2**。

- **原机制：** 激活与权重逐位跳零，空闲 slot donation。

- **最近先验与访问状态：** [BitFusion MICRO 2018; FireFly-v2](https://arxiv.org/abs/1712.01507)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。 激活只有1位，双侧幅值位稀疏前提消失。

- **真正增量：** bit-serial/slot donation 已有；不能重新命名为光流原语。

- **代价与反证：** 零位译码、移位、交换网、符号扩展与最坏精度时间都收费。

- **迁移方式：** 只保留权重位串行作为同资源对照；无新意暂不重做。

- **结论：** 不作主机制


#### plan.scope.C2.3 · ARM-Acc 与 MFBD

定位：L21。N/F/H/T：**3 / 0 / 5 / 1**。

- **原机制：** K 个运动方向假设独立累加，按证据赢家提交。

- **最近先验与访问状态：** Multi-hypothesis optical-flow algorithms; generic speculative accumulation；原文件给出来源；本轮未独立取得全文

- **冻结适配：** 冻结 H67 没有 K=4/8 搜索假设或赢家提交算子。

- **真正增量：** 4 个 consumer 是独立 token，不可改称竞争假设。

- **代价与反证：** 新增假设计算、状态与选择改变模型；不能将输家丢弃说成精确。

- **迁移方式：** 先有新算法与 AEE 才可考虑；当前剔除。

- **结论：** 冻结下否决


#### plan.scope.C2.4 · SP-Gate

定位：L22。N/F/H/T：**2 / 1 / 6 / 2**。

- **原机制：** 按 attention mass 选择 hot token 并抑制 cold token 的后续 FC 计算。

- **最近先验与访问状态：** [SparseVideoGen; token pruning / early-exit](https://arxiv.org/abs/2502.01776)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** 冻结 attention 的 gate 不是 token 整体无贡献证明；FC/BN/残差仍有消费者。

- **真正增量：** 普通分数剪枝；没有精确停止证书。

- **代价与反证：** 剪 token 改动态 BN 统计及稠密光流输出；分数读取/排序成本。

- **迁移方式：** 仅新 AEE Pareto；精确版必须完整依赖界。

- **结论：** 冻结无损路线否决


#### plan.scope.C2.5 · 统计/消融基础设施

定位：L23。N/F/H/T：**0 / 10 / 10 / 0**。

- **原机制：** 计数、CSV、可关闭消融与对照接口。

- **最近先验与访问状态：** 通用硬件性能计数器/验证方法；非论文机制

- **冻结适配：** 可用于冻结图，但必须绑定同身份、同工作负载。

- **真正增量：** 验证基础设施，不是电路贡献。

- **代价与反证：** TB 通过不等于VCS+DC/PT+Formality准入；有损参数与无损数字分开。

- **迁移方式：** 用于筛查和审计，不计新机制。

- **结论：** 保留工程必需项


#### plan.CardA · OP-STW

定位：L98–100。N/F/H/T：**3 / 1 / 5 / 2**。

- **原机制：** 用上一时刻 coarse flow/差分和事件计数预测 tile wake，睡眠 tile 复用输出。

- **最近先验与访问状态：** [MotionDeltaCNN ICCV 2023; motion-gated vision SoCs](https://openaccess.thecvf.com/content/ICCV2023/html/Parger_MotionDeltaCNN_Sparse_CNN_Inference_of_Frame_Differences_in_Moving_Camera_ICCV_2023_paper.html)；CVF 一手摘要和 PDF 方法段已读

- **冻结适配：** H67 无现成 coarse-flow 迭代支路；no_running 动态 BN 的全域均值/方差使局部不变不能直接推出消费者不变。

- **真正增量：** 加 motion 名称不足；若以严格消费者证书取代启发式 wake 才形成新执行问题。

- **代价与反证：** 当前帧 flow 不能提前免费获得；睡眠输出未必精确，必须计预测器、warp、缓存与 AEE。

- **迁移方式：** 原版只可有损新评价；精确重构必须全域 BN 依赖闭合。

- **结论：** 原版不作为冻结主线


#### plan.CardB · HBG-RP

定位：L102–104。N/F/H/T：**1 / 0 / 9 / 1**。

- **原机制：** 把幅值拆为非零门控 g 与 int8 实值 p，g 控制时钟/请求。

- **最近先验与访问状态：** [Activity Pruning for Efficient Spiking Neural Networks, Bu/Shi/Yu, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/fa12d67b5939c37ea8a4659c31a08d2c-Paper-Conference.pdf)；一手 PDF §3.2 式10–11及阈值吸收段已核；OpenReview 被拦后由 NeurIPS 正式库取得

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。

- **真正增量：** 冻结身份下退化为已有 active 位与权重加法；没有新增幅值原语。

- **代价与反证：** eps=1 严格 > 会删 ±1；量化/截断与门控均需新 AEE。常量 θ 若≤eps 甚至整层静默；最小负数 abs 必须9位。

- **迁移方式：** 冻结版本删除 p，只作为普通二值协议/门控基线；实值版本需重训并另建身份。

- **结论：** 否决为主机制；禁止套 ep34


#### plan.CardC · PRRC + OGEC

定位：L106–107。N/F/H/T：**2 / 0 / 6 / 1**。

- **原机制：** 粗到细金字塔预算/ROI，预算耗尽后复用或传播填充。

- **最近先验与访问状态：** [MotionDeltaCNN; optical-flow coarse-to-fine methods](https://openaccess.thecvf.com/content/ICCV2023/html/Parger_MotionDeltaCNN_Sparse_CNN_Inference_of_Frame_Differences_in_Moving_Camera_ICCV_2023_paper.html)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** 冻结 U-Net 非迭代金字塔搜索；不含预算耗尽语义。

- **真正增量：** 计数器与 ROI 不是新原语；须有算法可证明的停止条件。

- **代价与反证：** 强制 REUSE/PROP 本质近似，打 overflow 标志不能使输出精确。

- **迁移方式：** 不套 ep34；新算法需训练/valid825。

- **结论：** 原版否决


#### plan.CardD · ADP-MAC

定位：L109–110。N/F/H/T：**2 / 2 / 6 / 2**。

- **原机制：** 激活与权重逐位跳零，空闲 slot donation。

- **最近先验与访问状态：** [BitFusion MICRO 2018; FireFly-v2](https://arxiv.org/abs/1712.01507)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。 激活只有1位，双侧幅值位稀疏前提消失。

- **真正增量：** bit-serial/slot donation 已有；不能重新命名为光流原语。

- **代价与反证：** 零位译码、移位、交换网、符号扩展与最坏精度时间都收费。

- **迁移方式：** 只保留权重位串行作为同资源对照；无新意暂不重做。

- **结论：** 不作主机制


#### plan.CardE · ARM-Acc

定位：L112–113。N/F/H/T：**3 / 0 / 5 / 1**。

- **原机制：** K 个运动方向假设独立累加，按证据赢家提交。

- **最近先验与访问状态：** Multi-hypothesis optical-flow algorithms; generic speculative accumulation；原文件给出来源；本轮未独立取得全文

- **冻结适配：** 冻结 H67 没有 K=4/8 搜索假设或赢家提交算子。

- **真正增量：** 4 个 consumer 是独立 token，不可改称竞争假设。

- **代价与反证：** 新增假设计算、状态与选择改变模型；不能将输家丢弃说成精确。

- **迁移方式：** 先有新算法与 AEE 才可考虑；当前剔除。

- **结论：** 冻结下否决


#### plan.CardF · MFBD + SP-Gate

定位：L115–116。N/F/H/T：**1 / 6 / 7 / 1**。

- **原机制：** 同权重行向相邻时间/运动上下文广播。

- **最近先验与访问状态：** [ELSA ISCA 2026; Eyeriss; existing TSBG](https://arxiv.org/html/2605.20802v1)；ELSA 一手 HTML 相关 dataflow 已读

- **冻结适配：** 若每 consumer 保持真实 token/时步身份可合法。

- **真正增量：** 现有 TSBG 已在 mem_req 前按 source-group 共享取权，剩余只是分组选择。

- **代价与反证：** 不能重复计权重请求收益；运动假设不存在，广播网络与上下文容量不能免费增加。

- **迁移方式：** 必须改变实际计算内容才重立项；现版降为 TSBG 对照。

- **结论：** 仅既有机制换名


#### plan.CardG · 统计/消融基础设施

定位：L118–119。N/F/H/T：**0 / 10 / 10 / 0**。

- **原机制：** 计数、CSV、可关闭消融与对照接口。

- **最近先验与访问状态：** 通用硬件性能计数器/验证方法；非论文机制

- **冻结适配：** 可用于冻结图，但必须绑定同身份、同工作负载。

- **真正增量：** 验证基础设施，不是电路贡献。

- **代价与反证：** TB 通过不等于VCS+DC/PT+Formality准入；有损参数与无损数字分开。

- **迁移方式：** 用于筛查和审计，不计新机制。

- **结论：** 保留工程必需项


#### plan.contract.1 · amp位宽

定位：L129。N/F/H/T：**1 / 0 / 9 / 1**。

- **原机制：** 把幅值拆为非零门控 g 与 int8 实值 p，g 控制时钟/请求。

- **最近先验与访问状态：** [Activity Pruning for Efficient Spiking Neural Networks, Bu/Shi/Yu, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/fa12d67b5939c37ea8a4659c31a08d2c-Paper-Conference.pdf)；一手 PDF §3.2 式10–11及阈值吸收段已核；OpenReview 被拦后由 NeurIPS 正式库取得

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。

- **真正增量：** 冻结身份下退化为已有 active 位与权重加法；没有新增幅值原语。

- **代价与反证：** 当前冻结事实已明确；不能用草案默认值覆盖，也不能要求用户选错误二选一来制造创新。

- **迁移方式：** 冻结版本删除 p，只作为普通二值协议/门控基线；实值版本需重训并另建身份。

- **结论：** 否决为主机制；禁止套 ep34


#### plan.contract.2 · eps

定位：L130。N/F/H/T：**1 / 0 / 9 / 1**。

- **原机制：** 把幅值拆为非零门控 g 与 int8 实值 p，g 控制时钟/请求。

- **最近先验与访问状态：** [Activity Pruning for Efficient Spiking Neural Networks, Bu/Shi/Yu, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/fa12d67b5939c37ea8a4659c31a08d2c-Paper-Conference.pdf)；一手 PDF §3.2 式10–11及阈值吸收段已核；OpenReview 被拦后由 NeurIPS 正式库取得

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。

- **真正增量：** 冻结身份下退化为已有 active 位与权重加法；没有新增幅值原语。

- **代价与反证：** 当前冻结事实已明确；不能用草案默认值覆盖，也不能要求用户选错误二选一来制造创新。

- **迁移方式：** 冻结版本删除 p，只作为普通二值协议/门控基线；实值版本需重训并另建身份。

- **结论：** 否决为主机制；禁止套 ep34


#### plan.contract.3 · 不可吸进W

定位：L131。N/F/H/T：**1 / 0 / 9 / 1**。

- **原机制：** 把幅值拆为非零门控 g 与 int8 实值 p，g 控制时钟/请求。

- **最近先验与访问状态：** [Activity Pruning for Efficient Spiking Neural Networks, Bu/Shi/Yu, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/fa12d67b5939c37ea8a4659c31a08d2c-Paper-Conference.pdf)；一手 PDF §3.2 式10–11及阈值吸收段已核；OpenReview 被拦后由 NeurIPS 正式库取得

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。

- **真正增量：** 冻结身份下退化为已有 active 位与权重加法；没有新增幅值原语。

- **代价与反证：** 当前冻结事实已明确；不能用草案默认值覆盖，也不能要求用户选错误二选一来制造创新。

- **迁移方式：** 冻结版本删除 p，只作为普通二值协议/门控基线；实值版本需重训并另建身份。

- **结论：** 否决为主机制；禁止套 ep34


#### plan.contract.4 · 硬/soft gate

定位：L132。N/F/H/T：**1 / 0 / 9 / 1**。

- **原机制：** 把幅值拆为非零门控 g 与 int8 实值 p，g 控制时钟/请求。

- **最近先验与访问状态：** [Activity Pruning for Efficient Spiking Neural Networks, Bu/Shi/Yu, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/fa12d67b5939c37ea8a4659c31a08d2c-Paper-Conference.pdf)；一手 PDF §3.2 式10–11及阈值吸收段已核；OpenReview 被拦后由 NeurIPS 正式库取得

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。

- **真正增量：** 冻结身份下退化为已有 active 位与权重加法；没有新增幅值原语。

- **代价与反证：** 当前冻结事实已明确；不能用草案默认值覆盖，也不能要求用户选错误二选一来制造创新。

- **迁移方式：** 冻结版本删除 p，只作为普通二值协议/门控基线；实值版本需重训并另建身份。

- **结论：** 否决为主机制；禁止套 ep34


#### plan.contract.5 · ep34统计对齐

定位：L133。N/F/H/T：**0 / 10 / 10 / 0**。

- **原机制：** 计数、CSV、可关闭消融与对照接口。

- **最近先验与访问状态：** 通用硬件性能计数器/验证方法；非论文机制

- **冻结适配：** 可用于冻结图，但必须绑定同身份、同工作负载。

- **真正增量：** 验证基础设施，不是电路贡献。

- **代价与反证：** 当前冻结事实已明确；不能用草案默认值覆盖，也不能要求用户选错误二选一来制造创新。

- **迁移方式：** 用于筛查和审计，不计新机制。

- **结论：** 保留工程必需项


#### plan.schedule · Codex 0.35–0.5 工期系数

定位：L32–59。N/F/H/T：**0 / 0 / 0 / 0**。

- **原机制：** 计数、CSV、可关闭消融与对照接口。

- **最近先验与访问状态：** 通用硬件性能计数器/验证方法；非论文机制

- **冻结适配：** 可用于冻结图，但必须绑定同身份、同工作负载。

- **真正增量：** 验证基础设施，不是电路贡献。

- **代价与反证：** 经验系数无本工程证据；EDA/物理/算法验证不会因写代码提速自动通过。

- **迁移方式：** 用于筛查和审计，不计新机制。

- **结论：** 不作为交付保证


### research/00_seed_notes.md

精读完成；SHA256 `9301ebea4e98b6e9d8cd2c2028463b71455963881300e03bb708a238b65fbdf1`。

全文20行；10篇/组文献入口不是10项新机制，按其4条迁移动机复审，具体原文在对应专题分工覆盖。


#### ROOT-SEED-01 · 光流方向预测/时序相干迁移

定位：5–9,20。N/F/H/T：**2 / 3 / 6 / 2**。

- **原机制：** 以方向/时序预测引导spike或token调度

- **最近先验与访问状态：** 对应原始机制记录；本条为入口/工程证据；本地全文

- **冻结适配：** 以二值 ep34/no_running BN/full-rank PSN 为准

- **真正增量：** 只换任务名不足；若只预取且保留所有真实请求可无损，但只是常见预测调度

- **代价与反证：** 当前flow不可先于当前网络免费取得；预测漏唤醒无精确纠正会改输出

- **迁移方式：** source-only论文清单见03逐条；保留可验证的无损预取作实现

- **结论：** 实现或背景，不作主创新


#### ROOT-SEED-02 · 多比特SNN执行器迁移

定位：11–16。N/F/H/T：**2 / 2 / 5 / 2**。

- **原机制：** SpiDR/L-SPINE/Mega/量化脉冲Transformer等作相关工作入口

- **最近先验与访问状态：** 对应原始机制记录；本条为入口/工程证据；本地全文

- **冻结适配：** 以二值 ep34/no_running BN/full-rank PSN 为准

- **真正增量：** 多比特膜状态不等于ATLIF出口多幅值；权重可变精度与源二值分开

- **代价与反证：** 源载荷前提冲突，模拟CIM不属于本轮工艺合同

- **迁移方式：** 每篇具体迁移见02/10/13；可借位图解码、控制，不借模拟载荷身份

- **结论：** 实现或背景，不作主创新


#### ROOT-SEED-03 · SNN-Transformer光流组合是空白

定位：18。N/F/H/T：**1 / 9 / 8 / 1**。

- **原机制：** 将未常见的任务/网络组合视作创新空间

- **最近先验与访问状态：** 对应原始机制记录；本条为入口/工程证据；本地全文

- **冻结适配：** 以二值 ep34/no_running BN/full-rank PSN 为准

- **真正增量：** 首次任务组合需要新的电路问题与实测增量支撑

- **代价与反证：** 没有同机制强基线，不能从少见推稳录用

- **迁移方式：** 仅问题背景

- **结论：** 实现或背景，不作主创新


#### ROOT-SEED-04 · 实值ATLIF幅值保留数据通路

定位：19。N/F/H/T：**1 / 0 / 5 / 1**。

- **原机制：** 以ATLIF幅值不能折进权重为动机

- **最近先验与访问状态：** 对应原始机制记录；本条为入口/工程证据；本地全文

- **冻结适配：** 以二值 ep34/no_running BN/full-rank PSN 为准

- **真正增量：** 冻结出口为{0,θ}，此动机不成立

- **代价与反证：** 与冻结身份冲突；仅换名字无效

- **迁移方式：** 若另训多幅值需新AEE轨迹，当前淘汰

- **结论：** 实现或背景，不作主创新


### research/01_ann_sparsity_mechanisms.md

精读完成；SHA256 `92abf869ee0d3ece75940d84fe5b6ef47a014f995abe3a5006863f69cf248931`。


#### ANN01 · OP-STW / DPP-Skip

定位：23–47；365–366；404。N/F/H/T：**3 / 1 / 5 / 2**。

- **原机制：** 方向或流残差决定 tile 是否执行。

- **最近先验与访问状态：** [Liu et al., An Ultra-High Performance and Scalable Optical Flow Hardware Accelerator Based on FPGA for Autonomous Driving, TCAS-I 2025](https://ieeexplore.ieee.org/document/11030859/)；本轮 IEEE 原始摘要已核：方向预测、可配置 PE、金字塔管线；未取得全文

- **冻结适配：** 低；ep34 无前置当前 flow 预测器，跳过非零 tile 改变函数。

- **真正增量：** 增加任务相关门控条件；仅阈值比较未构成执行原语。

- **代价与反证：** 当前 flow 用于跳过产生它的推理存在因果环；漏唤醒、边界依赖、旧结果读写未收费。

- **迁移方式：** 改成来自前一帧或便宜独立前级的因果预测；有损另开 AEE，精确版需要输出不变证明。

- **结论：** 原案不能入冻结主线；可作为算法重构入口


#### ANN02 · ADP-MAC / 双边 bit particle

定位：51–76；367；398。N/F/H/T：**2 / 0 / 5 / 1**。

- **原机制：** 权重和 ATLIF 幅度双边 bit-skip、slot donation。

- **最近先验与访问状态：** [BBS: Bi-directional Bit-level Sparsity for Deep Learning Acceleration, MICRO 2024](https://arxiv.org/abs/2409.05227)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** 不适配；ATLIF 出口为二值，静态 theta 可折权重。

- **真正增量：** 多比特幅度数据通路是假定新增，不是已有算法独特性。

- **代价与反证：** 增加 bit-serial 延迟和粒度调度；自然源侧没有第二组幅度位可跳。

- **迁移方式：** 仅迁到 gate×weight 的真实多比特门控路径且重定合同；不能用 ATLIF 名义。

- **结论：** 淘汰 ATLIF 主线版本


#### ANN03 · MSBC / MSBC-Cascade

定位：80–104；366；401。N/F/H/T：**3 / 2 / 6 / 3**。

- **原机制：** motion saliency×attention mass 级联裁 token/head/time bundle。

- **最近先验与访问状态：** [SpAtten: Efficient Sparse Attention Architecture with Cascade Token and Head Pruning, HPCA 2021](https://arxiv.org/abs/2012.09852)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** 需新算法评价；固定密集光流输出不能直接丢 token。

- **真正增量：** 只把已有重要性函数换输入语义；新硬件未明确。

- **代价与反证：** top-k、选择器、被删 token 恢复与误差传播；小流速不代表低重要性。

- **迁移方式：** 研究可证明结果不变的门控界；若有损须按遮挡和边界分层 AEE。

- **结论：** 保留算法候选，不当已成立电路贡献


#### ANN04 · SEPA / DLZS predictive attention

定位：108–131；368；400。N/F/H/T：**3 / 2 / 4 / 3**。

- **原机制：** LZC/log 包络预测再正式注意力，跨阶段排序融合。

- **最近先验与访问状态：** [SOFA: A Compute-Memory Optimized Sparsity Accelerator via Cross-Stage Coordinated Tiling](https://arxiv.org/abs/2407.10416)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** 低；二值无多比特包络，H67 也非 SDSA。

- **真正增量：** 增加近似预测阶段，基本结构来自 SOFA。

- **代价与反证：** 预测可能比三 popcount 正式分数还贵；必须处理共静默和时间对端项。

- **迁移方式：** 直接用 Motion-XOR 的计数上下界替换包络，研究精确早终止而非新命名。

- **结论：** 有条件重构


#### ANN05 · PR-HSS / hierarchy sparse reshape

定位：135–158；368；399。N/F/H/T：**2 / 1 / 5 / 2**。

- **原机制：** 每金字塔层选择分层 N:M 模式。

- **最近先验与访问状态：** [HighLight: Efficient and Flexible DNN Acceleration with Hierarchical Structured Sparsity, MICRO 2023](https://arxiv.org/abs/2305.12718)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** 当前权重未训练成这些结构，运行配置不能创造结构稀疏。

- **真正增量：** 层配置与既有 HSS 映射；未改变数学执行内容。

- **代价与反证：** 需剪枝/重训与 AEE；填零密化、交集、负载损失。

- **迁移方式：** 仅作为新训练 Pareto 的硬件底座。

- **结论：** 不作冻结主机制


#### ANN06 · FPTE / flow-pregated experts

定位：162–187；366；406。N/F/H/T：**2 / 0 / 4 / 1**。

- **原机制：** 前帧运动先路由下一专家并重叠权重预取。

- **最近先验与访问状态：** [Pre-gated MoE: Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference, ISCA 2024](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/05/isca24_pregated_moe_camera_ready.pdf)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** ep34 无 MoE、方向专家或此路由接口。

- **真正增量：** 新增一套模型后套既有 pre-gate；名称不能赋予专家语义。

- **代价与反证：** 训练、专家存储、路由误判、预取和回退；无原始 fetch 瓶颈证据。

- **迁移方式：** 只留远期新模型；不为电路短文扩成 MoE。

- **结论：** 淘汰本轮主线


#### ANN07 · FC-PP / progressive precision

定位：191–217；367；402/405。N/F/H/T：**3 / 2 / 5 / 3**。

- **原机制：** 置信度决定先算 MSB、后补 LSB 或异常值路径。

- **最近先验与访问状态：** [SpAtten: Efficient Sparse Attention Architecture with Cascade Token and Head Pruning, HPCA 2021](https://arxiv.org/abs/2012.09852)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** ATLIF 本身无幅度位；真实权重/门控可以研究独立部署量化。

- **真正增量：** 任务置信度接入已有渐进精度；暂缺新的进位/界限电路。

- **代价与反证：** 部分和要能恢复、额外重读、误差校准；不能保证只需低位。

- **迁移方式：** 转为 Acc/门控结果的区间证书；结果确定即停止剩余位或剩余项。

- **结论：** 保留精确证书重构抓手


#### ANN08 · SS-FSA / Flash-SDSA fusion

定位：221–245；367。N/F/H/T：**4 / 6 / 7 / 5**。

- **原机制：** 分数、归一化和加权输出融合，避免中间图写回。

- **最近先验与访问状态：** [FuseMax: Leveraging Extended Einsums to Optimize Attention Accelerator Design, MICRO 2024](https://arxiv.org/html/2406.10491)；本轮原文融合与映射章节已核；one-pass cascade 与最大融合均为已有机制

- **冻结适配：** H67 可借融合思想；原文 SDSA/V/LIF 后处理与 gate⊙K 不同。

- **真正增量：** 原案 score-stationary 已有；实际空间在 H67 三项分数—门控—K 消费的专属生命周期。

- **代价与反证：** 必须保持分母、舍入、零值语义；Q7/Shiftmax 属部署候选而非冻结训练算术；旧岛不收费的流量不能算收益。

- **迁移方式：** 重写因果图，以 K=V 与两时间对端为精确不变量；匹配 unfused/强融合基线。

- **结论：** 值得算法原生重构，原名本身非创新


#### ANN09 · AAPT / amplitude-aware tick batching

定位：249–274；368；407。N/F/H/T：**2 / 2 / 5 / 2**。

- **原机制：** 并行展开时间步，共享权重并减少膜状态访存。

- **最近先验与访问状态：** [Hardware Efficient Accelerator for Spiking Transformer With Reconfigurable Parallel Time Step Computing, ISCAS 2025](https://arxiv.org/abs/2503.19643)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** 真实时间为 T10 神经元/T2 窗口；非多比特输出 T4/8。

- **真正增量：** 并行时间展开已有，幅度与 wake 不足以刷新机制。

- **代价与反证：** 复制时间 lane 面积、PSN 时间混合、权重扇出与状态提交；不能宣称膜完全消失。

- **迁移方式：** 只比较精确 T10 状态服务与 T2 成对分数，作为配套底座。

- **结论：** 配套实现，非标题贡献


#### ANN10 · SG-SHMAC / control-variate MAC

定位：278–302。N/F/H/T：**2 / 1 / 5 / 1**。

- **原机制：** 近似乘法后抵消均值误差，显著区走精确路。

- **最近先验与访问状态：** [Control Variate Approximation for DNN Accelerators, DAC 2021](https://arxiv.org/abs/2102.09642)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** C1 二值×INT8 是条件加法，目标乘法器不存在。

- **真正增量：** 新模式门控来自 OF 显著度，核心算术来自先验。

- **代价与反证：** 均值无偏不保证单样本 AEE，更不保证阈值发放不变。

- **迁移方式：** 若迁真实 gate×weight 需误差界和部署 AEE；不为 ATLIF 增一般 MAC。

- **结论：** 淘汰 C1 原案


#### ANN11 · RMSP / residual membrane skip port

定位：306–330。N/F/H/T：**2 / 3 / 5 / 2**。

- **原机制：** 残差保留膜域并直接经端口合并。

- **最近先验与访问状态：** [Hardware Efficient Accelerator for Spiking Transformer With Reconfigurable Parallel Time Step Computing, ISCAS 2025](https://arxiv.org/abs/2503.19643)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** ep34 是 U-Net，未有 RAFT 式跨迭代膜残差环；IAND 改残差函数。

- **真正增量：** 片上残差端口是常规算子融合。

- **代价与反证：** 多端口/缓存代价；不能把神经元状态与跨帧旧激活混同。

- **迁移方式：** 按真实 residual consumer 仅做精确融合。

- **结论：** 配套实现，不作主机制


#### ANN12 · DFED / event differential dispatcher

定位：334–358；404。N/F/H/T：**2 / 4 / 6 / 2**。

- **原机制：** 以 spike/残差惊奇触发工作，按密度换 bitmap/CSR。

- **最近先验与访问状态：** [Multiply-and-Fire: An Event-Driven Neural Network Accelerator](https://arxiv.org/abs/2204.09797)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** 天然二值可编码；以流阈值丢非零描述符为有损。

- **真正增量：** 编码切换和事件控制均属普通基线，任务阈值附加。

- **代价与反证：** 编码器、模式标志、转换和回退需收费；event=0 不推出网络输出不变。

- **迁移方式：** 保留类型化准确脏事件；必须证明实际少算而非只少发空描述符。

- **结论：** 编码作基线，差分另重构


#### ANN13 · entropy / residual block early exit

定位：403。N/F/H/T：**2 / 2 / 4 / 2**。

- **原机制：** 低熵或小残差时结束块/网络。

- **最近先验与访问状态：** [Tambe et al., entropy-based early exit / mixed-precision predication, ISSCC 2023](https://sld.cs.columbia.edu/pubs/tambe_isscc23.pdf)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** 不存在已校准早退头；T10 是确定推理展开。

- **真正增量：** 已有早退判据迁任务。

- **代价与反证：** 可能绕过时间依赖和解码；阈值无全网误差保证。

- **迁移方式：** 新算法支线配训练/校准及 AEE，勿称无损。

- **结论：** 不作冻结主线


#### ANN-PACK1 · C1* residual/eager/wake联合包

定位：362–371；439–440。N/F/H/T：**2 / 1 / 3 / 1**。

- **原机制：** 将方向、残差、遮挡与预测器并列接到C1。

- **最近先验与访问状态：** [Liu et al., An Ultra-High Performance and Scalable Optical Flow Hardware Accelerator Based on FPGA for Autonomous Driving, TCAS-I 2025](https://ieeexplore.ieee.org/document/11030859/)；本轮 IEEE 原始摘要已核：方向预测、可配置 PE、金字塔管线；未取得全文

- **冻结适配：** 多个原算法不存在的反馈边；不是ep34原位实现。

- **真正增量：** 组件堆叠不能自动增加单机制新意。

- **代价与反证：** 成本和误差耦合，TCAS-II 4.5页无法支撑每个新算法。

- **迁移方式：** 拆出一个确实少执行的原语，其余最多底座。

- **结论：** 否决整包贡献叙事


#### ANN-PACK2 · C2* dual-rail/motion/sparse联合包

定位：367；371；439–441。N/F/H/T：**1 / 0 / 3 / 0**。

- **原机制：** 实值双轨、运动context、分型稀疏及Mask-Add堆叠。

- **最近先验与访问状态：** [Xu et al., Bishop: Sparsified Bundling Spiking Transformers on Heterogeneous Cores with Error-Constrained Pruning, ISCA 2025](https://arxiv.org/html/2505.12281)；本轮原文 §3/§5.1 已核：TTB、二值 QK 点积界、BSA／ECP-aware 训练均为已有内容

- **冻结适配：** 实值载荷和hyp身份冲突；SDSA与H67混用。

- **真正增量：** 多数是已知路由或标签换名。

- **代价与反证：** 多后端面积/训练/误差未闭，不能相乘收益。

- **迁移方式：** 去掉虚构载荷与hyp，围绕一个真实消费者不变量重写。

- **结论：** 否决整包


### research/02_snn_atlif_realvalued_mechanisms.md

精读完成；SHA256 `2a0c873cb843d6149cc9c35cd65c3d9ef1d62df25dd6e6646625761bcb5c0cb5`。


#### 02.identity.canonical · Canonical AT-LIF {0,θ}

定位：L13。N/F/H/T：**0 / 10 / 10 / 0**。

- **原机制：** 静态阈值缩放二值输出，可吸入下层权重。

- **最近先验与访问状态：** [Activity Pruning for Efficient Spiking Neural Networks, Bu/Shi/Yu, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/fa12d67b5939c37ea8a4659c31a08d2c-Paper-Conference.pdf)；一手 PDF §3.2 式10–11及阈值吸收段已核；OpenReview 被拦后由 NeurIPS 正式库取得

- **冻结适配：** 与冻结二值出口匹配；PSN 内部动力学另由 fullrank A 定义。

- **真正增量：** 这是身份事实，不是本电路新贡献。

- **代价与反证：** 不能把 θ 称逐事件实值载荷。

- **迁移方式：** 所有 C2 候选须以此为入口合同。

- **结论：** 冻结合同保留


#### 02.identity.user · 所谓 user real-valued ATLIF

定位：L14,L19–21。N/F/H/T：**1 / 0 / 9 / 1**。

- **原机制：** 把幅值拆为非零门控 g 与 int8 实值 p，g 控制时钟/请求。

- **最近先验与访问状态：** [Activity Pruning for Efficient Spiking Neural Networks, Bu/Shi/Yu, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/fa12d67b5939c37ea8a4659c31a08d2c-Paper-Conference.pdf)；一手 PDF §3.2 式10–11及阈值吸收段已核；OpenReview 被拦后由 NeurIPS 正式库取得

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。

- **真正增量：** 冻结身份下退化为已有 active 位与权重加法；没有新增幅值原语。

- **代价与反证：** eps=1 严格 > 会删 ±1；量化/截断与门控均需新 AEE。常量 θ 若≤eps 甚至整层静默；最小负数 abs 必须9位。

- **迁移方式：** 冻结版本删除 p，只作为普通二值协议/门控基线；实值版本需重训并另建身份。

- **结论：** 否决为主机制；禁止套 ep34


#### 02.identity.multibit · MBLIF

定位：L15。N/F/H/T：**2 / 4 / 7 / 2**。

- **原机制：** 多位值按位展开移加，Cin/Cout/空间/时间并行。

- **最近先验与访问状态：** [FireFly v2, IEEE TCAD 2024](https://arxiv.org/abs/2309.16158)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** 多位脉冲另一个神经元模型，非 ep34。

- **真正增量：** 四维铺排和 bit-serial 均是已有机制。

- **代价与反证：** 需按 full-T PSN 存储与真实非脉冲算子计费；原 FPGA 资源不可换算 ASIC。

- **迁移方式：** 可保留覆盖非脉冲算子与常规混精度对照。

- **结论：** 工程候选；非主机制


#### 02.identity.ternary · Ternary/MLF

定位：L16。N/F/H/T：**2 / 0 / 4 / 1**。

- **原机制：** 动态换神经编码以减少时步。

- **最近先验与访问状态：** SpinalFlow; STELLAR；原文件给出来源；本轮未独立取得全文

- **冻结适配：** 冻结自然出口不是 ±θ 三值；sign 桥不能当三值捕获。

- **真正增量：** 编码切换须新模型训练；不是无损调度。

- **代价与反证：** PSN 跨时步语义丢失；需另 AEE/Pareto。

- **迁移方式：** 除非重训另线，否则排除。

- **结论：** 不迁移


#### 02.identity.graded · Loihi 2 graded temporal aggregation

定位：L17。N/F/H/T：**1 / 0 / 9 / 1**。

- **原机制：** 把幅值拆为非零门控 g 与 int8 实值 p，g 控制时钟/请求。

- **最近先验与访问状态：** Loihi 2 graded spikes（原文件泛指 CLANE）；仅原文件；未核具体论文

- **冻结适配：** Loihi 可支持 graded 不证明 ep34 采用 graded。

- **真正增量：** 冻结身份下退化为已有 active 位与权重加法；没有新增幅值原语。

- **代价与反证：** eps=1 严格 > 会删 ±1；量化/截断与门控均需新 AEE。常量 θ 若≤eps 甚至整层静默；最小负数 abs 必须9位。

- **迁移方式：** 冻结版本删除 p，只作为普通二值协议/门控基线；实值版本需重训并另建身份。

- **结论：** 否决为主机制；禁止套 ep34


#### 02.1.1.1 · SpinalFlow

定位：L31。N/F/H/T：**2 / 6 / 6 / 2**。

- **原机制：** 排序压缩事件列、按输出连续执行以减少状态存储。

- **最近先验与访问状态：** SpinalFlow: An Architecture and Dataflow Tailored for Spiking Neural Networks, ISCA 2020；原文件给出来源；本轮未独立取得全文

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。 事件压缩可用；神经元服务不能照搬。

- **真正增量：** 按固定T位束组织输入可作为实现基线，‘改成光流’不构成增量。

- **代价与反证：** 冻结 PSN 为完整 T×T 线性映射（T=2/10），不是按漏电、复位递推的 LIF。 压缩排序/解码和源重读收费，不能称实值破坏二值路线。

- **迁移方式：** 只借输入组织；对 full-T 数据到齐与现有 K8 吞吐比较。

- **结论：** 保留工程参考


#### 02.1.1.2 · SATO

定位：L32。N/F/H/T：**2 / 3 / 6 / 2**。

- **原机制：** 并行各时步累加，搜索加法树恢复 LIF 放电时刻，桶排序均衡。

- **最近先验与访问状态：** [SATO: Spiking Neural Network Acceleration via Temporal-Oriented Dataflow and Architecture, DAC 2022](https://mxhx7199.github.io/files/%5BDAC-2022%5DSATO_preprint.pdf)；一手 PDF 已访问，摘要与神经元搜索结构核对

- **冻结适配：** 冻结 PSN 为完整 T×T 线性映射（T=2/10），不是按漏电、复位递推的 LIF。

- **真正增量：** 时间并行/桶调度可用；原搜索树不实现冻结 A 矩阵。

- **代价与反证：** 不能用前缀漏电或首次过阈值替代 fullrank PSN；额外排序队列。

- **迁移方式：** 保留时步向量接口，PSN 用精确完整矩阵服务。

- **结论：** 原树否决；数据流仅工程参考


#### 02.1.1.3 · STELLAR

定位：L33。N/F/H/T：**2 / 3 / 5 / 2**。

- **原机制：** Few-Spikes 神经元/训练配合时空行驻留与窗口并行。

- **最近先验与访问状态：** [STELLAR: Energy-Efficient and Low-Latency SNN Algorithm and Hardware Co-Design with Spatiotemporal Computation, HPCA 2024](https://www.proceedings.com/content/074/074054webtoc.pdf)；正式会议目录核题名；论文全文本轮未取得，机制判断基于本地描述与冻结合同

- **冻结适配：** 冻结 PSN 为完整 T×T 线性映射（T=2/10），不是按漏电、复位递推的 LIF。 冻结无 Few-Spikes 训练身份。

- **真正增量：** 时空 tile 驻留已有；替换神经元属于新模型。

- **代价与反证：** FSBP 神经元/训练需另建身份；本轮未取得原文全文，性能不引用。

- **迁移方式：** 只借 stRS 比较数据复用；神经元路线另作 AEE。

- **结论：** 当前不立项；来源题名待核


#### 02.1.2.1 · FireFly

定位：L39。N/F/H/T：**1 / 8 / 8 / 1**。

- **原机制：** 二值脉冲选权重并累加，FPGA DSP 复用。

- **最近先验与访问状态：** FireFly, IEEE TVLSI 2023（原文件所列）；原文件给出来源；本轮未独立取得全文

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。 正好适合二值乘加。

- **真正增量：** 转 CMOS 后是普通 AND/mux/add。

- **代价与反证：** LUT/DSP 倍率不能搬入 28 nm；无幅值乘法需求。

- **迁移方式：** 作为 bit-skip 加法基础实现。

- **结论：** 仅基线


#### 02.1.2.2 · FireFly v2

定位：L40。N/F/H/T：**2 / 4 / 7 / 2**。

- **原机制：** 多位值按位展开移加，Cin/Cout/空间/时间并行。

- **最近先验与访问状态：** [FireFly v2, IEEE TCAD 2024](https://arxiv.org/abs/2309.16158)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。 多位支路针对真实非脉冲算子，不是 ATLIF 出口。

- **真正增量：** 四维铺排和 bit-serial 均是已有机制。

- **代价与反证：** 需按 full-T PSN 存储与真实非脉冲算子计费；原 FPGA 资源不可换算 ASIC。

- **迁移方式：** 可保留覆盖非脉冲算子与常规混精度对照。

- **结论：** 工程候选；非主机制


#### 02.1.2.3 · FireFly-S

定位：L41。N/F/H/T：**1 / 6 / 8 / 1**。

- **原机制：** 位图压缩/解码并跳过激活与权重零值。

- **最近先验与访问状态：** [FireFly-S; SCNN ISCA 2017](https://arxiv.org/abs/1708.04485)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。 激活侧适配；冻结权重并未授权剪枝。

- **真正增量：** row_live/bit-skip 与既有 TSBG 已覆盖基本收益。

- **代价与反证：** 权重稀疏训练另身份；索引、负载与随机 bank 冲突收费。

- **迁移方式：** 作为强稀疏基线，不凭双侧命名立项。

- **结论：** 仅基线


#### 02.1.2.4 · FireFly-T

定位：L42。N/F/H/T：**2 / 6 / 7 / 3**。

- **原机制：** AND-popcount/位图稀疏译码实现二值注意力。

- **最近先验与访问状态：** [FireFly-T](https://arxiv.org/abs/2505.12771)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。 Q/K 二值可用；H67 是 Motion-XOR、共静默与 gated-K。

- **真正增量：** 冻结公式的准确叶映射有应用价值；AND-popcount 本身不新。

- **代价与反证：** 不能只套 SDSA QKᵀV；门控乘权重仍在；FPGA LUT6 不等于 CMOS 电路贡献。

- **迁移方式：** 只作为 Motion-XOR 同算术强基线；新意须来自受证脏更新服务。

- **结论：** 有界重构参考


#### 02.1.2.5 · Lee & Li spatiotemporal systolic

定位：L43。N/F/H/T：**1 / 8 / 8 / 1**。

- **原机制：** 保持权重或累加器驻留并铺排时空维。

- **最近先验与访问状态：** [Eyeriss, JSSC 2017; Lee & Li ICCD 2020](https://doi.org/10.1109/JSSC.2016.2616357)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** 冻结 PSN 为完整 T×T 线性映射（T=2/10），不是按漏电、复位递推的 LIF。 完整时间维铺排可合法实现。

- **真正增量：** 映射选择，不是新计算语义。

- **代价与反证：** 至少保留全 T 输入/中间向量；不能免费无状态。

- **迁移方式：** 纳入硬件基线，不作为标题贡献。

- **结论：** 仅工程


#### 02.1.3.1 · ESSA

定位：L49。N/F/H/T：**2 / 7 / 6 / 2**。

- **原机制：** 脉冲压缩与可合并 fan-in。

- **最近先验与访问状态：** ESSA, IEEE TVLSI 2022（原文件所列）；原文件给出来源；本轮未独立取得全文

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。

- **真正增量：** 公共归约/压缩已存在；须与 CSE、UCNN 比较。

- **代价与反证：** 真实同源权重身份、图构造和部分和存储不能免费。

- **迁移方式：** 只用精确源重合建立候选；若仍广播或暂存则淘汰。

- **结论：** 有限参考


#### 02.1.3.2 · Skydiver

定位：L50。N/F/H/T：**1 / 8 / 7 / 1**。

- **原机制：** 按事件负载分桶、分派或窃取任务。

- **最近先验与访问状态：** [Skydiver, TCAD 2022; SATO DAC 2022](https://mxhx7199.github.io/files/%5BDAC-2022%5DSATO_preprint.pdf)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。

- **真正增量：** 负载均衡常规；上下文身份本身不是新原语。

- **代价与反证：** 队列、交叉路由与 K8 同带宽约束；可能只掩盖 bank 冲突。

- **迁移方式：** 加入 TSBG 强基线，以端到端服务衡量。

- **结论：** 仅工程


#### 02.1.3.3 · COMPASS

定位：L51。N/F/H/T：**2 / 0 / 1 / 0**。

- **原机制：** 阵列内点积、模拟神经元或定制存储计算。

- **最近先验与访问状态：** COMPASS（原文件所列，准确论文题名按原文另核）；本轮未独立取得全文；定制/模拟CIM失配依据本地宏合同与用户硬约束

- **冻结适配：** 不符合普通 foundry 1RW 数字 28 nm 合同。

- **真正增量：** 没有可直接迁移的存储计算原语。

- **代价与反证：** 无对应宏、ADC/器件与物理验证；用户明确禁止模拟 CIM。

- **迁移方式：** 只借测量纪律；不立 CIM 方案。

- **结论：** 否决


#### 02.1.3.4 · Seneca SCDQ

定位：L52。N/F/H/T：**1 / 1 / 7 / 1**。

- **原机制：** 循环队列携带事件、延迟及幅值。

- **最近先验与访问状态：** [Shared Circular Delay Queue, Seneca](https://arxiv.org/abs/2404.10597)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** 冻结图没有可学习突触延迟算子；ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。

- **真正增量：** 增加 payload 或延迟字段不是贡献。

- **代价与反证：** 队列与位宽扩大；不能把 PSN 时间矩阵说成 delay queue。

- **迁移方式：** 仅在出现真实延迟算法新身份时考虑。

- **结论：** 不迁移


#### 02.1.3.5 · 82 nW clock-free UED SNN wakeup

定位：L53。N/F/H/T：**2 / 0 / 1 / 0**。

- **原机制：** 阵列内点积、模拟神经元或定制存储计算。

- **最近先验与访问状态：** 82 nW clock-free UED SNN wakeup（原文件所列，准确论文题名按原文另核）；本轮未独立取得全文；定制/模拟CIM失配依据本地宏合同与用户硬约束

- **冻结适配：** 不符合普通 foundry 1RW 数字 28 nm 合同。

- **真正增量：** 没有可直接迁移的存储计算原语。

- **代价与反证：** 无对应宏、ADC/器件与物理验证；用户明确禁止模拟 CIM。

- **迁移方式：** 只借测量纪律；不立 CIM 方案。

- **结论：** 否决


#### 02.1.4.1 · SpikeSim / SpikeFlow

定位：L59。N/F/H/T：**2 / 0 / 1 / 0**。

- **原机制：** 阵列内点积、模拟神经元或定制存储计算。

- **最近先验与访问状态：** SpikeSim / SpikeFlow（原文件所列，准确论文题名按原文另核）；本轮未独立取得全文；定制/模拟CIM失配依据本地宏合同与用户硬约束

- **冻结适配：** 不符合普通 foundry 1RW 数字 28 nm 合同。

- **真正增量：** 没有可直接迁移的存储计算原语。

- **代价与反证：** 无对应宏、ADC/器件与物理验证；用户明确禁止模拟 CIM。

- **迁移方式：** 只借测量纪律；不立 CIM 方案。

- **结论：** 否决


#### 02.1.4.2 · SpikeTA

定位：L60。N/F/H/T：**2 / 6 / 6 / 2**。

- **原机制：** 加法树、深度感知缓冲和分流引擎实现 Transformer SNN。

- **最近先验与访问状态：** SpikeTA, IEEE TCAD 2025（原文件所列）；原文件给出来源；本轮未独立取得全文

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。 二值路径不因所谓实值而失效。

- **真正增量：** 面向 H67 的公式接入不自动产生新加法树。

- **代价与反证：** 缺一手全文，不能沿用 first FPGA 宣称；还需动态 BN 服务。

- **迁移方式：** 工程强基线候选，先核原文。

- **结论：** 工程参考


#### 02.1.4.3 · Xpikeformer

定位：L61。N/F/H/T：**2 / 0 / 1 / 0**。

- **原机制：** 阵列内点积、模拟神经元或定制存储计算。

- **最近先验与访问状态：** [Xpikeformer（原文件所列，准确论文题名按原文另核）](https://arxiv.org/abs/2408.08794)；本轮未独立取得全文；定制/模拟CIM失配依据本地宏合同与用户硬约束

- **冻结适配：** 不符合普通 foundry 1RW 数字 28 nm 合同。

- **真正增量：** 没有可直接迁移的存储计算原语。

- **代价与反证：** 无对应宏、ADC/器件与物理验证；用户明确禁止模拟 CIM。

- **迁移方式：** 只借测量纪律；不立 CIM 方案。

- **结论：** 否决


#### 02.1.4.4 · Sparse Spike-driven Transformer

定位：L62。N/F/H/T：**1 / 8 / 8 / 1**。

- **原机制：** 编码活动位置再读取权重做稀疏加法。

- **最近先验与访问状态：** [A sparse spike-driven Transformer accelerator（原文件 arXiv 项）](https://arxiv.org/abs/2501.07825)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。

- **真正增量：** 与已有 C2 descriptor+K8+TSBG 同类。

- **代价与反证：** 源组读共享不能再算第二遍收益。

- **迁移方式：** 作为译码/带宽基线。

- **结论：** 仅基线


#### 02.1.4.5 · L-SPINE

定位：L63。N/F/H/T：**2 / 2 / 6 / 2**。

- **原机制：** 统一 2/4/8 位 SIMD 移加。

- **最近先验与访问状态：** [L-SPINE（原文件 arXiv 项）](https://arxiv.org/abs/2604.03626)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。 幅值并不需要多精度。

- **真正增量：** 可能服务权重或未量化数值算子；不是 ATLIF 特有。

- **代价与反证：** 题名/文献内容未独核；需真实量化 AEE 与位串行时间收费。

- **迁移方式：** 多精度单元作为对照，不作主机制。

- **结论：** 原适配否决


#### 02.1.4.6 · SNNIM

定位：L64。N/F/H/T：**2 / 0 / 1 / 0**。

- **原机制：** 阵列内点积、模拟神经元或定制存储计算。

- **最近先验与访问状态：** SNNIM（原文件所列，准确论文题名按原文另核）；本轮未独立取得全文；定制/模拟CIM失配依据本地宏合同与用户硬约束

- **冻结适配：** 不符合普通 foundry 1RW 数字 28 nm 合同。

- **真正增量：** 没有可直接迁移的存储计算原语。

- **代价与反证：** 无对应宏、ADC/器件与物理验证；用户明确禁止模拟 CIM。

- **迁移方式：** 只借测量纪律；不立 CIM 方案。

- **结论：** 否决


#### 02.1.4.7 · DYNAP-SE2

定位：L65。N/F/H/T：**2 / 0 / 1 / 0**。

- **原机制：** 阵列内点积、模拟神经元或定制存储计算。

- **最近先验与访问状态：** DYNAP-SE2（原文件所列，准确论文题名按原文另核）；本轮未独立取得全文；定制/模拟CIM失配依据本地宏合同与用户硬约束

- **冻结适配：** 不符合普通 foundry 1RW 数字 28 nm 合同。

- **真正增量：** 没有可直接迁移的存储计算原语。

- **代价与反证：** 无对应宏、ADC/器件与物理验证；用户明确禁止模拟 CIM。

- **迁移方式：** 只借测量纪律；不立 CIM 方案。

- **结论：** 否决


#### 02.1.5.1 · AND/mask/popcount

定位：L69。N/F/H/T：**2 / 6 / 7 / 3**。

- **原机制：** AND-popcount/位图稀疏译码实现二值注意力。

- **最近先验与访问状态：** [FireFly-T](https://arxiv.org/abs/2505.12771)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。 Q/K 二值可用；H67 是 Motion-XOR、共静默与 gated-K。

- **真正增量：** 冻结公式的准确叶映射有应用价值；AND-popcount 本身不新。

- **代价与反证：** 不能只套 SDSA QKᵀV；门控乘权重仍在；FPGA LUT6 不等于 CMOS 电路贡献。

- **迁移方式：** 只作为 Motion-XOR 同算术强基线；新意须来自受证脏更新服务。

- **结论：** 有界重构参考


#### 02.1.5.2 · chronological binary merge

定位：L70。N/F/H/T：**2 / 6 / 6 / 2**。

- **原机制：** 排序压缩事件列、按输出连续执行以减少状态存储。

- **最近先验与访问状态：** SpinalFlow: An Architecture and Dataflow Tailored for Spiking Neural Networks, ISCA 2020；原文件给出来源；本轮未独立取得全文

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。 事件压缩可用；神经元服务不能照搬。

- **真正增量：** 按固定T位束组织输入可作为实现基线，‘改成光流’不构成增量。

- **代价与反证：** 冻结 PSN 为完整 T×T 线性映射（T=2/10），不是按漏电、复位递推的 LIF。 压缩排序/解码和源重读收费，不能称实值破坏二值路线。

- **迁移方式：** 只借输入组织；对 full-T 数据到齐与现有 K8 吞吐比较。

- **结论：** 保留工程参考


#### 02.1.5.3 · adder-search tree

定位：L71。N/F/H/T：**2 / 3 / 6 / 2**。

- **原机制：** 并行各时步累加，搜索加法树恢复 LIF 放电时刻，桶排序均衡。

- **最近先验与访问状态：** [SATO: Spiking Neural Network Acceleration via Temporal-Oriented Dataflow and Architecture, DAC 2022](https://mxhx7199.github.io/files/%5BDAC-2022%5DSATO_preprint.pdf)；一手 PDF 已访问，摘要与神经元搜索结构核对

- **冻结适配：** 冻结 PSN 为完整 T×T 线性映射（T=2/10），不是按漏电、复位递推的 LIF。

- **真正增量：** 时间并行/桶调度可用；原搜索树不实现冻结 A 矩阵。

- **代价与反证：** 不能用前缀漏电或首次过阈值替代 fullrank PSN；额外排序队列。

- **迁移方式：** 保留时步向量接口，PSN 用精确完整矩阵服务。

- **结论：** 原树否决；数据流仅工程参考


#### 02.1.5.4 · 1-bit spike NoC

定位：L72。N/F/H/T：**1 / 8 / 8 / 1**。

- **原机制：** 编码活动位置再读取权重做稀疏加法。

- **最近先验与访问状态：** [A sparse spike-driven Transformer accelerator（原文件 arXiv 项）](https://arxiv.org/abs/2501.07825)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。

- **真正增量：** 与已有 C2 descriptor+K8+TSBG 同类。

- **代价与反证：** 源组读共享不能再算第二遍收益。

- **迁移方式：** 作为译码/带宽基线。

- **结论：** 仅基线


#### 02.1.5.5 · spike as mux

定位：L73。N/F/H/T：**1 / 8 / 8 / 1**。

- **原机制：** 二值脉冲选权重并累加，FPGA DSP 复用。

- **最近先验与访问状态：** FireFly, IEEE TVLSI 2023（原文件所列）；原文件给出来源；本轮未独立取得全文

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。 正好适合二值乘加。

- **真正增量：** 转 CMOS 后是普通 AND/mux/add。

- **代价与反证：** LUT/DSP 倍率不能搬入 28 nm；无幅值乘法需求。

- **迁移方式：** 作为 bit-skip 加法基础实现。

- **结论：** 仅基线


#### 02.1.5.6 · θ absorption called invalid

定位：L74。N/F/H/T：**1 / 0 / 9 / 1**。

- **原机制：** 把幅值拆为非零门控 g 与 int8 实值 p，g 控制时钟/请求。

- **最近先验与访问状态：** [Activity Pruning for Efficient Spiking Neural Networks, Bu/Shi/Yu, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/fa12d67b5939c37ea8a4659c31a08d2c-Paper-Conference.pdf)；一手 PDF §3.2 式10–11及阈值吸收段已核；OpenReview 被拦后由 NeurIPS 正式库取得

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。

- **真正增量：** 冻结身份下退化为已有 active 位与权重加法；没有新增幅值原语。

- **代价与反证：** eps=1 严格 > 会删 ±1；量化/截断与门控均需新 AEE。常量 θ 若≤eps 甚至整层静默；最小负数 abs 必须9位。

- **迁移方式：** 冻结版本删除 p，只作为普通二值协议/门控基线；实值版本需重训并另建身份。

- **结论：** 否决为主机制；禁止套 ep34


#### 02.1.6.1 · bit-decompose

定位：L78。N/F/H/T：**2 / 4 / 7 / 2**。

- **原机制：** 多位值按位展开移加，Cin/Cout/空间/时间并行。

- **最近先验与访问状态：** [FireFly v2, IEEE TCAD 2024](https://arxiv.org/abs/2309.16158)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。 多位支路针对真实非脉冲算子，不是 ATLIF 出口。

- **真正增量：** 四维铺排和 bit-serial 均是已有机制。

- **代价与反证：** 需按 full-T PSN 存储与真实非脉冲算子计费；原 FPGA 资源不可换算 ASIC。

- **迁移方式：** 可保留覆盖非脉冲算子与常规混精度对照。

- **结论：** 工程候选；非主机制


#### 02.1.6.2 · OS/stRS/4D

定位：L79。N/F/H/T：**1 / 8 / 8 / 1**。

- **原机制：** 保持权重或累加器驻留并铺排时空维。

- **最近先验与访问状态：** [Eyeriss, JSSC 2017; Lee & Li ICCD 2020](https://doi.org/10.1109/JSSC.2016.2616357)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** 冻结 PSN 为完整 T×T 线性映射（T=2/10），不是按漏电、复位递推的 LIF。 完整时间维铺排可合法实现。

- **真正增量：** 映射选择，不是新计算语义。

- **代价与反证：** 至少保留全 T 输入/中间向量；不能免费无状态。

- **迁移方式：** 纳入硬件基线，不作为标题贡献。

- **结论：** 仅工程


#### 02.1.6.3 · on-the-fly residual Vmem cache

定位：L80。N/F/H/T：**3 / 2 / 5 / 2**。

- **原机制：** 跨窗口或帧缓存膜/中间结果，按相似输入复用。

- **最近先验与访问状态：** [DeltaCNN CVPR 2022](https://arxiv.org/html/2203.03996v2)；一手 HTML 已访问；动态 BN 反证来自本地合同

- **冻结适配：** no_running 动态 BN 的全域均值/方差使局部不变不能直接推出消费者不变。 无 RAFT 循环状态；窗口重叠不等于同一上下文。

- **真正增量：** 只有精确输入、权重和归一化身份同时匹配才可复用。

- **代价与反证：** 锚点宽张量、版本、地址及比较流量；PSN 不等于 LIF 残留膜。

- **迁移方式：** 收窄到 BN 前纯线性叶，或改全域统计证书；原缓存不直接迁移。

- **结论：** 需大改


#### 02.1.6.4 · widened SCDQ

定位：L81。N/F/H/T：**1 / 1 / 7 / 1**。

- **原机制：** 循环队列携带事件、延迟及幅值。

- **最近先验与访问状态：** [Shared Circular Delay Queue, Seneca](https://arxiv.org/abs/2404.10597)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** 冻结图没有可学习突触延迟算子；ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。

- **真正增量：** 增加 payload 或延迟字段不是贡献。

- **代价与反证：** 队列与位宽扩大；不能把 PSN 时间矩阵说成 delay queue。

- **迁移方式：** 仅在出现真实延迟算法新身份时考虑。

- **结论：** 不迁移


#### 02.1.6.5 · dual-side sparsity

定位：L82。N/F/H/T：**1 / 6 / 8 / 1**。

- **原机制：** 位图压缩/解码并跳过激活与权重零值。

- **最近先验与访问状态：** [FireFly-S; SCNN ISCA 2017](https://arxiv.org/abs/1708.04485)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。 激活侧适配；冻结权重并未授权剪枝。

- **真正增量：** row_live/bit-skip 与既有 TSBG 已覆盖基本收益。

- **代价与反证：** 权重稀疏训练另身份；索引、负载与随机 bank 冲突收费。

- **迁移方式：** 作为强稀疏基线，不凭双侧命名立项。

- **结论：** 仅基线


#### 02.1.6.6 · bucket/worker balance

定位：L83。N/F/H/T：**1 / 8 / 7 / 1**。

- **原机制：** 按事件负载分桶、分派或窃取任务。

- **最近先验与访问状态：** [Skydiver, TCAD 2022; SATO DAC 2022](https://mxhx7199.github.io/files/%5BDAC-2022%5DSATO_preprint.pdf)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。

- **真正增量：** 负载均衡常规；上下文身份本身不是新原语。

- **代价与反证：** 队列、交叉路由与 K8 同带宽约束；可能只掩盖 bank 冲突。

- **迁移方式：** 加入 TSBG 强基线，以端到端服务衡量。

- **结论：** 仅工程


#### 02.1.6.7 · PSN GEMM

定位：L84。N/F/H/T：**1 / 10 / 8 / 2**。

- **原机制：** 完整 T×T 矩阵乘输入向量后逐位阈值。

- **最近先验与访问状态：** [Parallel Spiking Neurons with High Efficiency and Ability to Learn Long-term Dependencies, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/a834ac3dfdb90da54292c2c932c997cc-Paper-Conference.pdf)；一手 PDF §3.2–3.3，式9–12已核

- **冻结适配：** 冻结 PSN 为完整 T×T 线性映射（T=2/10），不是按漏电、复位递推的 LIF。 完全匹配算子身份。

- **真正增量：** 把 GEMM 落电路是覆盖，不是新时间代数。

- **代价与反证：** 不能将 T 个输入求总和后再阈值；A 可正可负且非因果；C3 已有覆盖。

- **迁移方式：** 保留为其他机制的消费者合同。

- **结论：** 必需语义；不独立主打


#### 02.A1 · HBG-RP

定位：L95–100。N/F/H/T：**1 / 0 / 9 / 1**。

- **原机制：** 把幅值拆为非零门控 g 与 int8 实值 p，g 控制时钟/请求。

- **最近先验与访问状态：** [Activity Pruning for Efficient Spiking Neural Networks, Bu/Shi/Yu, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/fa12d67b5939c37ea8a4659c31a08d2c-Paper-Conference.pdf)；一手 PDF §3.2 式10–11及阈值吸收段已核；OpenReview 被拦后由 NeurIPS 正式库取得

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。

- **真正增量：** 冻结身份下退化为已有 active 位与权重加法；没有新增幅值原语。

- **代价与反证：** eps=1 严格 > 会删 ±1；量化/截断与门控均需新 AEE。常量 θ 若≤eps 甚至整层静默；最小负数 abs 必须9位。

- **迁移方式：** 冻结版本删除 p，只作为普通二值协议/门控基线；实值版本需重训并另建身份。

- **结论：** 否决为主机制；禁止套 ep34


#### 02.A2 · VGTS

定位：L102–106。N/F/H/T：**3 / 0 / 3 / 1**。

- **原机制：** 根据幅值与漏电预测未来若干时间步不发放并跳过更新。

- **最近先验与访问状态：** [SATO; SnaPEA ISCA 2018](https://cseweb.ucsd.edu/~vakhlagh/ISCA18-SnaPEA.pdf)；SnaPEA 一手 PDF 已读；精确模式与预测模式分开

- **冻结适配：** 冻结 PSN 为完整 T×T 线性映射（T=2/10），不是按漏电、复位递推的 LIF。 冻结没有此因果漏电过程。

- **真正增量：** 改为完整 T 向量的严格剩余贡献界才合法；那是另一机制。

- **代价与反证：** 未来输入可能通过负/正 A 翻转输出，当前无脉冲不证明未来静默。

- **迁移方式：** 原版否决；严格证书只作独立研究，且先解决动态 BN。

- **结论：** 原版否决


#### 02.A3 · MST-MAC

定位：L108–112。N/F/H/T：**2 / 1 / 5 / 1**。

- **原机制：** 共享指数/尾数聚合多个时间步后复用权重。

- **最近先验与访问状态：** [FireFly-v2; Mailman finite-alphabet aggregation](https://cs.yale.edu/homes/el327/papers/mailmanAlgorithm.pdf)；Mailman 原始论文已读；FireFly 原文未独核

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。 θ 已静态折入 W。

- **真正增量：** 无新的逐事件尺度可合并；有限字母聚合代数已有。

- **代价与反证：** 冻结 PSN 为完整 T×T 线性映射（T=2/10），不是按漏电、复位递推的 LIF。 时间求和丢失 A 的逐时刻系数；格式另需新量化身份。

- **迁移方式：** 只在保留完整时间模式时分析成本，不可直接合并 T。

- **结论：** 否决原式


#### 02.A4 · ReMem-Tok / residual membrane cache

定位：L114–118。N/F/H/T：**3 / 2 / 5 / 2**。

- **原机制：** 跨窗口或帧缓存膜/中间结果，按相似输入复用。

- **最近先验与访问状态：** [DeltaCNN CVPR 2022](https://arxiv.org/html/2203.03996v2)；一手 HTML 已访问；动态 BN 反证来自本地合同

- **冻结适配：** no_running 动态 BN 的全域均值/方差使局部不变不能直接推出消费者不变。 无 RAFT 循环状态；窗口重叠不等于同一上下文。

- **真正增量：** 只有精确输入、权重和归一化身份同时匹配才可复用。

- **代价与反证：** 锚点宽张量、版本、地址及比较流量；PSN 不等于 LIF 残留膜。

- **迁移方式：** 收窄到 BN 前纯线性叶，或改全域统计证书；原缓存不直接迁移。

- **结论：** 需大改


#### 02.A5 · SAG-CLK

定位：L120–123。N/F/H/T：**1 / 7 / 8 / 1**。

- **原机制：** 非零活动 OR 触发时钟/银行门控，并按幅值调压调频。

- **最近先验与访问状态：** [Eyeriss / Envision / event-driven SNN clock gating](https://doi.org/10.1109/JSSC.2016.2616357)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。 活动门控适配，幅值只有静态 θ。

- **真正增量：** 普通时钟门控不构成主机制。

- **代价与反证：** VFS 需库、电源域、唤醒与物理表征；现有 TSBG 门控低复用能量变差不能隐藏。

- **迁移方式：** 保留标准门控，按真实活动计能量。

- **结论：** 仅工程


#### 02.B1 · Amplitude-aware attention dual path

定位：L127–132。N/F/H/T：**2 / 0 / 6 / 1**。

- **原机制：** 二值 gate 控制独立实值 V/MAC 支路。

- **最近先验与访问状态：** [FireFly-T; Spike-driven Transformer](https://arxiv.org/abs/2505.12771)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。 H67 K 当 V；没有该独立实值 V 投影。

- **真正增量：** 原双轨建立在错误数据路径上。

- **代价与反证：** Q1.7 部署 gate 是另算术身份，不能解释成 ATLIF int8；gate×weight仍存在。

- **迁移方式：** 重画 H67 exact score→gate⊙K 算术，先不立双轨。

- **结论：** 否决原故事


#### 02.B2 · TTX-HW / Motion-XOR dirty

定位：L134–137。N/F/H/T：**5 / 6 / 6 / 5**。

- **原机制：** 时间对端 K XOR 形成变化指示，尝试抑制不变分数工作。

- **最近先验与访问状态：** [DeltaCNN; FireFly-T; H67 Motion-XOR local leaf](https://arxiv.org/html/2203.03996v2)；DeltaCNN 一手 HTML；H67 公式本地合同，非宣称同一算法

- **冻结适配：** 仅精确 Q/K/peer 版本不变时可复用对应分数；直接 XOR=0 不能省整个注意力。

- **真正增量：** 可研究三项分数的依赖位图、影子 K、精确脏 lane 提交。

- **代价与反证：** 还需 Q 变化、same-zero、overlap 与归一化消费者；全量影子存储和脏率，旧0.6%份额不能当新测。

- **迁移方式：** 候选留给 Card C：先 ep34 挂载统计，再仅一个 score 岛。

- **结论：** 可迁移重构；未有统计/电路准入


#### 02.B3 · PSN temporal GEMM

定位：L139–141。N/F/H/T：**1 / 10 / 8 / 2**。

- **原机制：** 完整 T×T 矩阵乘输入向量后逐位阈值。

- **最近先验与访问状态：** [Parallel Spiking Neurons with High Efficiency and Ability to Learn Long-term Dependencies, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/a834ac3dfdb90da54292c2c932c997cc-Paper-Conference.pdf)；一手 PDF §3.2–3.3，式9–12已核

- **冻结适配：** 冻结 PSN 为完整 T×T 线性映射（T=2/10），不是按漏电、复位递推的 LIF。 完全匹配算子身份。

- **真正增量：** 把 GEMM 落电路是覆盖，不是新时间代数。

- **代价与反证：** 不能将 T 个输入求总和后再阈值；A 可正可负且非因果；C3 已有覆盖。

- **迁移方式：** 保留为其他机制的消费者合同。

- **结论：** 必需语义；不独立主打


#### 02.B4 · Seneca SCDQ / widened AER

定位：L143–145。N/F/H/T：**1 / 1 / 7 / 1**。

- **原机制：** 循环队列携带事件、延迟及幅值。

- **最近先验与访问状态：** [Shared Circular Delay Queue, Seneca](https://arxiv.org/abs/2404.10597)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** 冻结图没有可学习突触延迟算子；ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。

- **真正增量：** 增加 payload 或延迟字段不是贡献。

- **代价与反证：** 队列与位宽扩大；不能把 PSN 时间矩阵说成 delay queue。

- **迁移方式：** 仅在出现真实延迟算法新身份时考虑。

- **结论：** 不迁移


#### 02.B5 · FireFly-v2 bit decomposition

定位：L147–149。N/F/H/T：**2 / 4 / 7 / 2**。

- **原机制：** 多位值按位展开移加，Cin/Cout/空间/时间并行。

- **最近先验与访问状态：** [FireFly v2, IEEE TCAD 2024](https://arxiv.org/abs/2309.16158)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。 多位支路针对真实非脉冲算子，不是 ATLIF 出口。

- **真正增量：** 四维铺排和 bit-serial 均是已有机制。

- **代价与反证：** 需按 full-T PSN 存储与真实非脉冲算子计费；原 FPGA 资源不可换算 ASIC。

- **迁移方式：** 可保留覆盖非脉冲算子与常规混精度对照。

- **结论：** 工程候选；非主机制


#### 02.C1 · Dendritic fan-in combine

定位：L155。N/F/H/T：**2 / 7 / 6 / 2**。

- **原机制：** 脉冲压缩与可合并 fan-in。

- **最近先验与访问状态：** ESSA, IEEE TVLSI 2022（原文件所列）；原文件给出来源；本轮未独立取得全文

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。

- **真正增量：** 公共归约/压缩已存在；须与 CSE、UCNN 比较。

- **代价与反证：** 真实同源权重身份、图构造和部分和存储不能免费。

- **迁移方式：** 只用精确源重合建立候选；若仍广播或暂存则淘汰。

- **结论：** 有限参考


#### 02.C2 · Bitmap compression

定位：L156。N/F/H/T：**1 / 6 / 8 / 1**。

- **原机制：** 位图压缩/解码并跳过激活与权重零值。

- **最近先验与访问状态：** [FireFly-S; SCNN ISCA 2017](https://arxiv.org/abs/1708.04485)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。 激活侧适配；冻结权重并未授权剪枝。

- **真正增量：** row_live/bit-skip 与既有 TSBG 已覆盖基本收益。

- **代价与反证：** 权重稀疏训练另身份；索引、负载与随机 bank 冲突收费。

- **迁移方式：** 作为强稀疏基线，不凭双侧命名立项。

- **结论：** 仅基线


#### 02.C3 · IMC

定位：L157。N/F/H/T：**2 / 0 / 1 / 0**。

- **原机制：** 阵列内点积、模拟神经元或定制存储计算。

- **最近先验与访问状态：** IMC（原文件所列，准确论文题名按原文另核）；本轮未独立取得全文；定制/模拟CIM失配依据本地宏合同与用户硬约束

- **冻结适配：** 不符合普通 foundry 1RW 数字 28 nm 合同。

- **真正增量：** 没有可直接迁移的存储计算原语。

- **代价与反证：** 无对应宏、ADC/器件与物理验证；用户明确禁止模拟 CIM。

- **迁移方式：** 只借测量纪律；不立 CIM 方案。

- **结论：** 否决


#### 02.C4 · STDP

定位：L158。N/F/H/T：**1 / 0 / 2 / 0**。

- **原机制：** 芯片内局部突触训练或在线规则。

- **最近先验与访问状态：** DYNAP / neuromorphic STDP literature（原文泛指）；原文件给出来源；本轮未独立取得全文

- **冻结适配：** 冻结模型采用已有监督训练，不含在线 STDP。

- **真正增量：** 增加训练算子是新任务，不是当前推理节省。

- **代价与反证：** 状态、训练准确度与器件合同均缺。

- **迁移方式：** 当前排除。

- **结论：** 不迁移


#### 02.C5 · rate/temporal switch

定位：L159。N/F/H/T：**2 / 0 / 4 / 1**。

- **原机制：** 动态换神经编码以减少时步。

- **最近先验与访问状态：** SpinalFlow; STELLAR；原文件给出来源；本轮未独立取得全文

- **冻结适配：** 冻结 T、A、训练编码固定。

- **真正增量：** 编码切换须新模型训练；不是无损调度。

- **代价与反证：** PSN 跨时步语义丢失；需另 AEE/Pareto。

- **迁移方式：** 除非重训另线，否则排除。

- **结论：** 不迁移


#### 02.3.1 · 整包 ATLIF-Aware Spikeformer OF

定位：L165–174。N/F/H/T：**1 / 0 / 9 / 1**。

- **原机制：** 把幅值拆为非零门控 g 与 int8 实值 p，g 控制时钟/请求。

- **最近先验与访问状态：** [Activity Pruning for Efficient Spiking Neural Networks, Bu/Shi/Yu, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/fa12d67b5939c37ea8a4659c31a08d2c-Paper-Conference.pdf)；一手 PDF §3.2 式10–11及阈值吸收段已核；OpenReview 被拦后由 NeurIPS 正式库取得

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。

- **真正增量：** 多个旧机制叠加，身份前提错误，不能作为一个5页机制。

- **代价与反证：** eps=1 严格 > 会删 ±1；量化/截断与门控均需新 AEE。常量 θ 若≤eps 甚至整层静默；最小负数 abs 必须9位。

- **迁移方式：** 冻结版本删除 p，只作为普通二值协议/门控基线；实值版本需重训并另建身份。

- **结论：** 否决为主机制；禁止套 ep34


#### 02.3.2.1 · event packet + exponent

定位：L178。N/F/H/T：**1 / 0 / 9 / 1**。

- **原机制：** 把幅值拆为非零门控 g 与 int8 实值 p，g 控制时钟/请求。

- **最近先验与访问状态：** [Activity Pruning for Efficient Spiking Neural Networks, Bu/Shi/Yu, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/fa12d67b5939c37ea8a4659c31a08d2c-Paper-Conference.pdf)；一手 PDF §3.2 式10–11及阈值吸收段已核；OpenReview 被拦后由 NeurIPS 正式库取得

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。

- **真正增量：** 冻结身份下退化为已有 active 位与权重加法；没有新增幅值原语。

- **代价与反证：** eps=1 严格 > 会删 ±1；量化/截断与门控均需新 AEE。常量 θ 若≤eps 甚至整层静默；最小负数 abs 必须9位。

- **迁移方式：** 冻结版本删除 p，只作为普通二值协议/门控基线；实值版本需重训并另建身份。

- **结论：** 否决为主机制；禁止套 ep34


#### 02.3.2.2 · gated MAC

定位：L179。N/F/H/T：**1 / 0 / 9 / 1**。

- **原机制：** 把幅值拆为非零门控 g 与 int8 实值 p，g 控制时钟/请求。

- **最近先验与访问状态：** [Activity Pruning for Efficient Spiking Neural Networks, Bu/Shi/Yu, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/fa12d67b5939c37ea8a4659c31a08d2c-Paper-Conference.pdf)；一手 PDF §3.2 式10–11及阈值吸收段已核；OpenReview 被拦后由 NeurIPS 正式库取得

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。

- **真正增量：** 冻结身份下退化为已有 active 位与权重加法；没有新增幅值原语。

- **代价与反证：** eps=1 严格 > 会删 ±1；量化/截断与门控均需新 AEE。常量 θ 若≤eps 甚至整层静默；最小负数 abs 必须9位。

- **迁移方式：** 冻结版本删除 p，只作为普通二值协议/门控基线；实值版本需重训并另建身份。

- **结论：** 否决为主机制；禁止套 ep34


#### 02.3.2.3 · LUT6 gate attention

定位：L180。N/F/H/T：**2 / 6 / 7 / 3**。

- **原机制：** AND-popcount/位图稀疏译码实现二值注意力。

- **最近先验与访问状态：** [FireFly-T](https://arxiv.org/abs/2505.12771)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。 Q/K 二值可用；H67 是 Motion-XOR、共静默与 gated-K。

- **真正增量：** 冻结公式的准确叶映射有应用价值；AND-popcount 本身不新。

- **代价与反证：** 不能只套 SDSA QKᵀV；门控乘权重仍在；FPGA LUT6 不等于 CMOS 电路贡献。

- **迁移方式：** 只作为 Motion-XOR 同算术强基线；新意须来自受证脏更新服务。

- **结论：** 有界重构参考


#### 02.3.2.4 · LIF dynamics + real payload

定位：L181。N/F/H/T：**3 / 0 / 3 / 1**。

- **原机制：** 根据幅值与漏电预测未来若干时间步不发放并跳过更新。

- **最近先验与访问状态：** [SATO; SnaPEA ISCA 2018](https://cseweb.ucsd.edu/~vakhlagh/ISCA18-SnaPEA.pdf)；SnaPEA 一手 PDF 已读；精确模式与预测模式分开

- **冻结适配：** 冻结 PSN 为完整 T×T 线性映射（T=2/10），不是按漏电、复位递推的 LIF。 冻结没有此因果漏电过程。

- **真正增量：** 改为完整 T 向量的严格剩余贡献界才合法；那是另一机制。

- **代价与反证：** 未来输入可能通过负/正 A 翻转输出，当前无脉冲不证明未来静默。

- **迁移方式：** 原版否决；严格证书只作独立研究，且先解决动态 BN。

- **结论：** 原版否决


#### 02.3.2.5 · XOR history gate

定位：L182。N/F/H/T：**5 / 6 / 6 / 5**。

- **原机制：** 时间对端 K XOR 形成变化指示，尝试抑制不变分数工作。

- **最近先验与访问状态：** [DeltaCNN; FireFly-T; H67 Motion-XOR local leaf](https://arxiv.org/html/2203.03996v2)；DeltaCNN 一手 HTML；H67 公式本地合同，非宣称同一算法

- **冻结适配：** 仅精确 Q/K/peer 版本不变时可复用对应分数；直接 XOR=0 不能省整个注意力。

- **真正增量：** 可研究三项分数的依赖位图、影子 K、精确脏 lane 提交。

- **代价与反证：** 还需 Q 变化、same-zero、overlap 与归一化消费者；全量影子存储和脏率，旧0.6%份额不能当新测。

- **迁移方式：** 候选留给 Card C：先 ep34 挂载统计，再仅一个 score 岛。

- **结论：** 可迁移重构；未有统计/电路准入


#### 02.4.algorithm.1 · SDformerFlow

定位：L205。N/F/H/T：**1 / 3 / 5 / 1**。

- **原机制：** 选择某算法的现有算子做硬件映射。

- **最近先验与访问状态：** [SDformerFlow](https://arxiv.org/abs/2409.04082)；原文件引用；本轮未独核原文

- **冻结适配：** 必须逐算子对照 H67；公开算法同名不代表冻结路径。

- **真正增量：** 只有应用标签/第一份映射，尚无可审查电路机制。

- **代价与反证：** ‘没见到硬件’不证明首个；不得用他人算法性能作为本岛优势。

- **迁移方式：** 先提炼瓶颈与执行变换，再形成一个电路机制。

- **结论：** 文献索引；不是机制


#### 02.4.algorithm.2 · Adaptive-SpikeNet

定位：L206。N/F/H/T：**1 / 3 / 5 / 1**。

- **原机制：** 选择某算法的现有算子做硬件映射。

- **最近先验与访问状态：** [Adaptive-SpikeNet](https://arxiv.org/abs/2209.11741)；原文件引用；本轮未独核原文

- **冻结适配：** 必须逐算子对照 H67；公开算法同名不代表冻结路径。

- **真正增量：** 只有应用标签/第一份映射，尚无可审查电路机制。

- **代价与反证：** ‘没见到硬件’不证明首个；不得用他人算法性能作为本岛优势。

- **迁移方式：** 先提炼瓶颈与执行变换，再形成一个电路机制。

- **结论：** 文献索引；不是机制


#### 02.4.algorithm.3 · Spike-FlowNet

定位：L207。N/F/H/T：**1 / 3 / 5 / 1**。

- **原机制：** 选择某算法的现有算子做硬件映射。

- **最近先验与访问状态：** Spike-FlowNet；原文件引用；本轮未独核原文

- **冻结适配：** 必须逐算子对照 H67；公开算法同名不代表冻结路径。

- **真正增量：** 只有应用标签/第一份映射，尚无可审查电路机制。

- **代价与反证：** ‘没见到硬件’不证明首个；不得用他人算法性能作为本岛优势。

- **迁移方式：** 先提炼瓶颈与执行变换，再形成一个电路机制。

- **结论：** 文献索引；不是机制


#### 02.4.algorithm.4 · Best of Both Worlds hybrid OF

定位：L208。N/F/H/T：**1 / 3 / 5 / 1**。

- **原机制：** 选择某算法的现有算子做硬件映射。

- **最近先验与访问状态：** [Best of Both Worlds hybrid OF](https://arxiv.org/abs/2306.02960)；原文件引用；本轮未独核原文

- **冻结适配：** 必须逐算子对照 H67；公开算法同名不代表冻结路径。

- **真正增量：** 只有应用标签/第一份映射，尚无可审查电路机制。

- **代价与反证：** ‘没见到硬件’不证明首个；不得用他人算法性能作为本岛优势。

- **迁移方式：** 先提炼瓶颈与执行变换，再形成一个电路机制。

- **结论：** 文献索引；不是机制


#### 02.4.algorithm.5 · Schnider analog-spike optical flow

定位：L209。N/F/H/T：**1 / 0 / 9 / 1**。

- **原机制：** 把幅值拆为非零门控 g 与 int8 实值 p，g 控制时钟/请求。

- **最近先验与访问状态：** Schnider analog-spike optical flow；原文件引用；本轮未独核原文

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。

- **真正增量：** 冻结身份下退化为已有 active 位与权重加法；没有新增幅值原语。

- **代价与反证：** eps=1 严格 > 会删 ±1；量化/截断与门控均需新 AEE。常量 θ 若≤eps 甚至整层静默；最小负数 abs 必须9位。

- **迁移方式：** 冻结版本删除 p，只作为普通二值协议/门控基线；实值版本需重训并另建身份。

- **结论：** 否决为主机制；禁止套 ep34


#### 02.4.algorithm.6 · STE-FlowNet / EVA-Flow / E-RAFT

定位：L210。N/F/H/T：**1 / 3 / 5 / 1**。

- **原机制：** 选择某算法的现有算子做硬件映射。

- **最近先验与访问状态：** STE-FlowNet / EVA-Flow / E-RAFT；原文件引用；本轮未独核原文

- **冻结适配：** 必须逐算子对照 H67；公开算法同名不代表冻结路径。

- **真正增量：** 只有应用标签/第一份映射，尚无可审查电路机制。

- **代价与反证：** ‘没见到硬件’不证明首个；不得用他人算法性能作为本岛优势。

- **迁移方式：** 先提炼瓶颈与执行变换，再形成一个电路机制。

- **结论：** 文献索引；不是机制


#### 02.4.algorithm.7 · Spike-driven Transformer

定位：L216。N/F/H/T：**2 / 6 / 7 / 3**。

- **原机制：** AND-popcount/位图稀疏译码实现二值注意力。

- **最近先验与访问状态：** Spike-driven Transformer；原文件引用；本轮未独核原文

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。 Q/K 二值可用；H67 是 Motion-XOR、共静默与 gated-K。

- **真正增量：** 冻结公式的准确叶映射有应用价值；AND-popcount 本身不新。

- **代价与反证：** 不能只套 SDSA QKᵀV；门控乘权重仍在；FPGA LUT6 不等于 CMOS 电路贡献。

- **迁移方式：** 只作为 Motion-XOR 同算术强基线；新意须来自受证脏更新服务。

- **结论：** 有界重构参考


#### 02.4.algorithm.8 · Spike-driven Transformer V2

定位：L217。N/F/H/T：**1 / 3 / 5 / 1**。

- **原机制：** 选择某算法的现有算子做硬件映射。

- **最近先验与访问状态：** Spike-driven Transformer V2；原文件引用；本轮未独核原文

- **冻结适配：** 必须逐算子对照 H67；公开算法同名不代表冻结路径。

- **真正增量：** 只有应用标签/第一份映射，尚无可审查电路机制。

- **代价与反证：** ‘没见到硬件’不证明首个；不得用他人算法性能作为本岛优势。

- **迁移方式：** 先提炼瓶颈与执行变换，再形成一个电路机制。

- **结论：** 文献索引；不是机制


#### 02.4.algorithm.9 · Spikformer

定位：L218。N/F/H/T：**1 / 3 / 5 / 1**。

- **原机制：** 选择某算法的现有算子做硬件映射。

- **最近先验与访问状态：** Spikformer；原文件引用；本轮未独核原文

- **冻结适配：** 必须逐算子对照 H67；公开算法同名不代表冻结路径。

- **真正增量：** 只有应用标签/第一份映射，尚无可审查电路机制。

- **代价与反证：** ‘没见到硬件’不证明首个；不得用他人算法性能作为本岛优势。

- **迁移方式：** 先提炼瓶颈与执行变换，再形成一个电路机制。

- **结论：** 文献索引；不是机制


#### 02.4.algorithm.10 · Spikingformer

定位：L219。N/F/H/T：**3 / 2 / 5 / 2**。

- **原机制：** 跨窗口或帧缓存膜/中间结果，按相似输入复用。

- **最近先验与访问状态：** [Spikingformer](https://arxiv.org/abs/2304.11954)；原文件引用；本轮未独核原文

- **冻结适配：** no_running 动态 BN 的全域均值/方差使局部不变不能直接推出消费者不变。 无 RAFT 循环状态；窗口重叠不等于同一上下文。

- **真正增量：** 只有精确输入、权重和归一化身份同时匹配才可复用。

- **代价与反证：** 锚点宽张量、版本、地址及比较流量；PSN 不等于 LIF 残留膜。

- **迁移方式：** 收窄到 BN 前纯线性叶，或改全域统计证书；原缓存不直接迁移。

- **结论：** 需大改


#### 02.4.algorithm.11 · PSN

定位：L220。N/F/H/T：**1 / 10 / 8 / 2**。

- **原机制：** 完整 T×T 矩阵乘输入向量后逐位阈值。

- **最近先验与访问状态：** [PSN](https://proceedings.neurips.cc/paper_files/paper/2023/file/a834ac3dfdb90da54292c2c932c997cc-Paper-Conference.pdf)；一手 PSN 原文已核

- **冻结适配：** 冻结 PSN 为完整 T×T 线性映射（T=2/10），不是按漏电、复位递推的 LIF。 完全匹配算子身份。

- **真正增量：** 把 GEMM 落电路是覆盖，不是新时间代数。

- **代价与反证：** 不能将 T 个输入求总和后再阈值；A 可正可负且非因果；C3 已有覆盖。

- **迁移方式：** 保留为其他机制的消费者合同。

- **结论：** 必需语义；不独立主打


#### 02.4.algorithm.12 · SEW-ResNet

定位：L221。N/F/H/T：**2 / 4 / 7 / 2**。

- **原机制：** 多位值按位展开移加，Cin/Cout/空间/时间并行。

- **最近先验与访问状态：** SEW-ResNet；原文件引用；本轮未独核原文

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。 多位支路针对真实非脉冲算子，不是 ATLIF 出口。

- **真正增量：** 四维铺排和 bit-serial 均是已有机制。

- **代价与反证：** 需按 full-T PSN 存储与真实非脉冲算子计费；原 FPGA 资源不可换算 ASIC。

- **迁移方式：** 可保留覆盖非脉冲算子与常规混精度对照。

- **结论：** 工程候选；非主机制


#### 02.4.algorithm.13 · AT-LIF activity pruning

定位：L222。N/F/H/T：**0 / 10 / 9 / 1**。

- **原机制：** 把幅值拆为非零门控 g 与 int8 实值 p，g 控制时钟/请求。

- **最近先验与访问状态：** [AT-LIF activity pruning](https://proceedings.neurips.cc/paper_files/paper/2025/file/fa12d67b5939c37ea8a4659c31a08d2c-Paper-Conference.pdf)；一手 PDF §3.2 式10–11及阈值吸收段已核；OpenReview 被拦后由 NeurIPS 正式库取得

- **冻结适配：** AT-LIF 二值θ输出本身与冻结适配；原文所谓不可吸收幅值反转hook不成立。

- **真正增量：** 冻结身份下退化为已有 active 位与权重加法；没有新增幅值原语。

- **代价与反证：** eps=1 严格 > 会删 ±1；量化/截断与门控均需新 AEE。常量 θ 若≤eps 甚至整层静默；最小负数 abs 必须9位。

- **迁移方式：** 冻结版本删除 p，只作为普通二值协议/门控基线；实值版本需重训并另建身份。

- **结论：** 算法身份保留，幅值硬件hook否决


#### 02.4.algorithm.14 · Multi-bit MBLIF

定位：L223。N/F/H/T：**2 / 4 / 7 / 2**。

- **原机制：** 多位值按位展开移加，Cin/Cout/空间/时间并行。

- **最近先验与访问状态：** [Multi-bit MBLIF](https://arxiv.org/abs/2407.05739)；原文件引用；本轮未独核原文

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。 多位支路针对真实非脉冲算子，不是 ATLIF 出口。

- **真正增量：** 四维铺排和 bit-serial 均是已有机制。

- **代价与反证：** 需按 full-T PSN 存储与真实非脉冲算子计费；原 FPGA 资源不可换算 ASIC。

- **迁移方式：** 可保留覆盖非脉冲算子与常规混精度对照。

- **结论：** 工程候选；非主机制


#### 02.4.algorithm.15 · Ternary Spike

定位：L224。N/F/H/T：**2 / 0 / 4 / 1**。

- **原机制：** 动态换神经编码以减少时步。

- **最近先验与访问状态：** Ternary Spike；原文件引用；本轮未独核原文

- **冻结适配：** 冻结 T、A、训练编码固定。

- **真正增量：** 编码切换须新模型训练；不是无损调度。

- **代价与反证：** PSN 跨时步语义丢失；需另 AEE/Pareto。

- **迁移方式：** 除非重训另线，否则排除。

- **结论：** 不迁移


#### 02.4.algorithm.16 · QKFormer

定位：L225。N/F/H/T：**1 / 3 / 5 / 1**。

- **原机制：** 选择某算法的现有算子做硬件映射。

- **最近先验与访问状态：** QKFormer；原文件引用；本轮未独核原文

- **冻结适配：** 必须逐算子对照 H67；公开算法同名不代表冻结路径。

- **真正增量：** 只有应用标签/第一份映射，尚无可审查电路机制。

- **代价与反证：** ‘没见到硬件’不证明首个；不得用他人算法性能作为本岛优势。

- **迁移方式：** 先提炼瓶颈与执行变换，再形成一个电路机制。

- **结论：** 文献索引；不是机制


#### 02.4.algorithm.17 · CATFormer DTLIF

定位：L226。N/F/H/T：**2 / 0 / 4 / 1**。

- **原机制：** 动态换神经编码以减少时步。

- **最近先验与访问状态：** [CATFormer DTLIF](https://arxiv.org/abs/2603.15184)；原文件引用；本轮未独核原文

- **冻结适配：** 冻结 T、A、训练编码固定。

- **真正增量：** 编码切换须新模型训练；不是无损调度。

- **代价与反证：** PSN 跨时步语义丢失；需另 AEE/Pareto。

- **迁移方式：** 除非重训另线，否则排除。

- **结论：** 不迁移


#### 02.5.1 · contribution 1 HBG

定位：L238。N/F/H/T：**1 / 0 / 9 / 1**。

- **原机制：** 把幅值拆为非零门控 g 与 int8 实值 p，g 控制时钟/请求。

- **最近先验与访问状态：** [Activity Pruning for Efficient Spiking Neural Networks, Bu/Shi/Yu, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/fa12d67b5939c37ea8a4659c31a08d2c-Paper-Conference.pdf)；一手 PDF §3.2 式10–11及阈值吸收段已核；OpenReview 被拦后由 NeurIPS 正式库取得

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。

- **真正增量：** 冻结身份下退化为已有 active 位与权重加法；没有新增幅值原语。

- **代价与反证：** eps=1 严格 > 会删 ±1；量化/截断与门控均需新 AEE。常量 θ 若≤eps 甚至整层静默；最小负数 abs 必须9位。

- **迁移方式：** 冻结版本删除 p，只作为普通二值协议/门控基线；实值版本需重训并另建身份。

- **结论：** 否决为主机制；禁止套 ep34


#### 02.5.2 · contribution 2 VGTS + ReMem

定位：L239。N/F/H/T：**3 / 0 / 3 / 1**。

- **原机制：** 根据幅值与漏电预测未来若干时间步不发放并跳过更新。

- **最近先验与访问状态：** [SATO; SnaPEA ISCA 2018](https://cseweb.ucsd.edu/~vakhlagh/ISCA18-SnaPEA.pdf)；SnaPEA 一手 PDF 已读；精确模式与预测模式分开

- **冻结适配：** 冻结 PSN 为完整 T×T 线性映射（T=2/10），不是按漏电、复位递推的 LIF。 冻结没有此因果漏电过程。

- **真正增量：** 改为完整 T 向量的严格剩余贡献界才合法；那是另一机制。

- **代价与反证：** 未来输入可能通过负/正 A 翻转输出，当前无脉冲不证明未来静默。

- **迁移方式：** 原版否决；严格证书只作独立研究，且先解决动态 BN。

- **结论：** 原版否决


#### 02.5.3 · contribution 3 amplitude attention

定位：L240。N/F/H/T：**2 / 0 / 6 / 1**。

- **原机制：** 二值 gate 控制独立实值 V/MAC 支路。

- **最近先验与访问状态：** [FireFly-T; Spike-driven Transformer](https://arxiv.org/abs/2505.12771)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。 H67 K 当 V；没有该独立实值 V 投影。

- **真正增量：** 原双轨建立在错误数据路径上。

- **代价与反证：** Q1.7 部署 gate 是另算术身份，不能解释成 ATLIF int8；gate×weight仍存在。

- **迁移方式：** 重画 H67 exact score→gate⊙K 算术，先不立双轨。

- **结论：** 否决原故事


### research/03_opticalflow_data_hw_algo.md

精读完成；SHA256 `fc95656d2c4f97bb6517857f2ce1792ecd9e3f51fb2acc363e68d333994dee61`。


#### OF-A · OP-STW / DPP-Skip

定位：94–99；237–238。N/F/H/T：**3 / 1 / 5 / 2**。

- **原机制：** 方向或流残差决定 tile 是否执行。

- **最近先验与访问状态：** [Liu et al., An Ultra-High Performance and Scalable Optical Flow Hardware Accelerator Based on FPGA for Autonomous Driving, TCAS-I 2025](https://ieeexplore.ieee.org/document/11030859/)；本轮 IEEE 原始摘要已核：方向预测、可配置 PE、金字塔管线；未取得全文

- **冻结适配：** 低；ep34 无前置当前 flow 预测器，跳过非零 tile 改变函数。

- **真正增量：** 增加任务相关门控条件；仅阈值比较未构成执行原语。

- **代价与反证：** 当前 flow 用于跳过产生它的推理存在因果环；漏唤醒、边界依赖、旧结果读写未收费。

- **迁移方式：** 改成来自前一帧或便宜独立前级的因果预测；有损另开 AEE，精确版需要输出不变证明。

- **结论：** 原案不能入冻结主线；可作为算法重构入口


#### OF-B · PRRC / pyramid residual capture

定位：101–106；239。N/F/H/T：**2 / 1 / 4 / 2**。

- **原机制：** 粗层写流残差，细层只算 warp 对齐 ROI。

- **最近先验与访问状态：** [Ling et al., FlowAcc, DATE 2022](https://doi.org/10.23919/DATE54114.2022.9774506)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** 分层 U-Net 不等于粗流驱动的迭代残差算法。

- **真正增量：** 新增粗流反馈与 ROI 算法，而非改现有存储政策。

- **代价与反证：** 原前向无所需控制边；ROI halo、warp、残差存储和恢复成本。

- **迁移方式：** 若坚持粗细残差需明确新 forward 和训练；不能复用 ep34 无损数字。

- **结论：** 原冻结适配失败


#### OF-C · OGEC / occlusion-gated exact capture

定位：108–113；240。N/F/H/T：**3 / 1 / 4 / 3**。

- **原机制：** 遮挡/未匹配 token 转廉价传播通路，匹配走精确。

- **最近先验与访问状态：** [Xu et al., GMFlow: Learning Optical Flow via Global Matching, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_GMFlow_Learning_Optical_Flow_via_Global_Matching_CVPR_2022_paper.pdf)；本轮 CVF 原始摘要／引言已核；全局匹配和传播属于其算法

- **冻结适配：** ep34 没有该传播分支；遮挡不是可删的无效像素。

- **真正增量：** 选路+新增算法分支；GMFlow 已有传播。

- **代价与反证：** 双向一致性可能需额外推理；遮挡正是光流难例，邻居填充没有等价证明。

- **迁移方式：** 独立 AEE 分支；若只预取保留全部执行则创新很薄。

- **结论：** 需算法重构，不能打 exact 标签


#### OF-D · TSP / temporal-signature product key

定位：115–120；245。N/F/H/T：**2 / 4 / 5 / 2**。

- **原机制：** 以重复 Motion-XOR/时间签名查找旧乘积。

- **最近先验与访问状态：** [Wei et al., Prosperity, HPCA 2025](https://arxiv.org/html/2503.03379v1)；本线程原文 §III-B/D 已核：排除非原行交集，保留一个原行父节点

- **冻结适配：** 仅签名相同不足以证明输入掩码和权重向量相同。

- **真正增量：** 改变查找键；仍是结果 memoization。

- **代价与反证：** popcount/XOR 有碰撞；上下文/源mask必须核对，CAM 与保存成本。

- **迁移方式：** 只允许完整源集合及权重身份的无碰撞复用；用作正确性基线。

- **结论：** 不作创新主线


#### OF-E · ARM-Acc / motion hypotheses

定位：122–127；241。N/F/H/T：**2 / 0 / 4 / 1**。

- **原机制：** Acc 槽存多方向假设并由证据选赢家。

- **最近先验与访问状态：** [Stumpp et al., hARMS: A Hardware Acceleration Architecture for Real-Time Event-Based Optical Flow, IEEE Access 2022](https://arxiv.org/html/2112.06772)；原文可访问；本轮概要核读，未逐条核实现数字

- **冻结适配：** ep34 通道/head 无显式运动假设和赢家提交语义。

- **真正增量：** 仅给 Acc 槽改标签不改执行；真正迁移需新算子。

- **代价与反证：** K 假设会增加计算/状态；hARMS 多尺度估计不等同 Transformer head。

- **迁移方式：** 只有明确定义并训练 hypothesis tensor 后再考虑共享求证算子。

- **结论：** 原案为语义换名，淘汰


#### OF-F · MFBD / motion-feature bundles

定位：129–134；237/244/246。N/F/H/T：**2 / 3 / 5 / 2**。

- **原机制：** 时间相邻运动状态束内共享权重并广播。

- **最近先验与访问状态：** [Shi et al., VideoFlow: Exploiting Temporal Cues for Multi-frame Optical Flow Estimation, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Shi_VideoFlow_Exploiting_Temporal_Cues_for_Multi-frame_Optical_Flow_Estimation_ICCV_2023_paper.html)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** ep34 无 TROF/MOP 迭代状态；普通 token×time 可打包。

- **真正增量：** 同权重广播本质未改，运动标签不增加代数机会。

- **代价与反证：** 不能借 TMA 少迭代速度；新帧上下文缓冲/身份检查。

- **迁移方式：** 抽取多目的共同表达式才改执行量；普通广播保留强基线。

- **结论：** 降为基线，不作主机制


#### OF-G · SP-Gate / attention mass

定位：136–141；238/243。N/F/H/T：**3 / 3 / 5 / 3**。

- **原机制：** 小注意力质量不发后级请求。

- **最近先验与访问状态：** [SpAtten: Efficient Sparse Attention Architecture with Cascade Token and Head Pruning, HPCA 2021](https://arxiv.org/abs/2012.09852)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** H67 分数含共静默/时间项，低 Q 发放不等于低 gate；K=V。

- **真正增量：** 非零重要性裁剪是现有稀疏注意力迁移。

- **代价与反证：** 省流量前要算分数；有损阈值改变归一化分母和输出。

- **迁移方式：** 寻找量化门控精确为零的充分条件，并收费界计算；部署合同另列。

- **结论：** 保留精确界重构


#### OF-H · EHSC / event-history core

定位：143–148；242。N/F/H/T：**3 / 0 / 4 / 2**。

- **原机制：** 事件历史环触发窗口，替代帧式体素。

- **最近先验与访问状态：** [Stumpp et al., hARMS: A Hardware Acceleration Architecture for Real-Time Event-Based Optical Flow, IEEE Access 2022](https://arxiv.org/html/2112.06772)；原文可访问；本轮概要核读，未逐条核实现数字

- **冻结适配：** 固定 T10 voxel/PSN 非在线异步网络。

- **真正增量：** 更换时间输入语义与调度机制，确实改变计算但不是原模型实现。

- **代价与反证：** 所有邻接窗口更新、时间戳状态、复位与精度训练代价。

- **迁移方式：** 远期流式算法线；禁止直接借 ep34 精度。

- **结论：** 另开模型，非本轮


#### OF-I · EESUC / early-exit spike update

定位：150–155；244/255。N/F/H/T：**2 / 2 / 4 / 2**。

- **原机制：** 小 flow/膜残差就减少后续时间步或迭代。

- **最近先验与访问状态：** [An FPGA-based Real-Time Optical Flow Accelerator for Recurrent All-Pairs Field Transforms, ISCAS 2025](https://doi.org/10.1109/ISCAS56072.2025.11043529)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** 没有 RAFT refinement 环；PSN T10/T2 不能按收敛迭代解释。

- **真正增量：** 早停算法已知，现案无可用终止不变量。

- **代价与反证：** 缺失步可能翻转未来发放；T10 固定服务不可截断。

- **迁移方式：** 只保留严格未来输入界证明的提前确定输出，非经验小 Δ。

- **结论：** 冻结版本否决；证书方向可研究


#### OF-J · HMA / Hamming propose-refine

定位：157–162；239/243/257。N/F/H/T：**3 / 3 / 5 / 3**。

- **原机制：** 低价二值匹配先给候选，再做正式注意力。

- **最近先验与访问状态：** [Ling et al., FlowAcc, DATE 2022](https://doi.org/10.23919/DATE54114.2022.9774506)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** H67 已是低价二值三项分数，不是昂贵相关体。

- **真正增量：** 二级筛选来自匹配加速器。

- **代价与反证：** 预筛可能无价格优势；不可忽略同零项导致漏候选。

- **迁移方式：** 用分数项分解的精确上界提前终止，比额外建一个 Hamming 模型更合理。

- **结论：** 重构筛选不变量后再评


#### OF-K · CRAFT-lite Q/K noise filter

定位：164–169。N/F/H/T：**2 / 1 / 4 / 1**。

- **原机制：** Q/K 前附加语义平滑降低噪声。

- **最近先验与访问状态：** [Sui et al., CRAFT: Cross-Attentional Flow Transformers for Robust Optical Flow, CVPR 2022](https://arxiv.org/abs/2203.16896)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** 会改变冻结注意力，权重需学习/评价。

- **真正增量：** 额外过滤算子，不是新的执行原语。

- **代价与反证：** 新投影/平滑增加操作；大位移好处未由 DSEC 证明。

- **迁移方式：** 仅算法对照，不移植到 C2 标成硬件创新。

- **结论：** 淘汰主机制


#### OF-L · ASNA-style moving spatial frontier

定位：171–176；238/242。N/F/H/T：**1 / 3 / 6 / 1**。

- **原机制：** 事件邻域在空间 PE 网格上跟随流向。

- **最近先验与访问状态：** [ASNA-Flow, IEEE TVLSI 2025](https://doi.org/10.1109/TVLSI.2025.3600953)；未全文核；用户冻结禁令明确排除把其空间局部性作为我们的原语

- **冻结适配：** 一般 locality 可用；未知运动驱动映射非原算法。

- **真正增量：** 空间局部性已由 OF 芯片使用；原文件没有新的通信不变量。

- **代价与反证：** 用户硬禁令；路由、边界迁移和 Acc搬运可能超过节省。

- **迁移方式：** 普通空间映射当强基线；只另查时间相似与算法融合。

- **结论：** 明确排除主创新


#### OF-M · Tri-frame bi-directional context fabric

定位：178–183；240/246。N/F/H/T：**2 / 0 / 4 / 1**。

- **原机制：** 三帧前后向上下文共享权重并做一致性筛选。

- **最近先验与访问状态：** [Shi et al., VideoFlow: Exploiting Temporal Cues for Multi-frame Optical Flow Estimation, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Shi_VideoFlow_Exploiting_Temporal_Cues_for_Multi-frame_Optical_Flow_Estimation_ICCV_2023_paper.html)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** ep34 没有三帧双向预测图；T2 窗口不等于双向流。

- **真正增量：** 新增算法任务；上下文重命名本身无创新。

- **代价与反证：** 可能等同多做一份推理并增加未来帧等待。

- **迁移方式：** 仅另开多帧模型时设计共享算子。

- **结论：** 淘汰本轮主线


#### OF-N · FlowFormer-lite cost-token capture

定位：185–190；243。N/F/H/T：**2 / 1 / 4 / 1**。

- **原机制：** 把匹配代价压成 cost token 再注意力。

- **最近先验与访问状态：** [Huang et al., FlowFormer: A Transformer Architecture for Optical Flow, ECCV 2022](https://arxiv.org/abs/2203.16194)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** ep34 U-Net 无 4D cost volume。

- **真正增量：** 引入另一网络的中间表示。

- **代价与反证：** 压缩通常有损；exact cost-token 并不自动等于 exact 原网络。

- **迁移方式：** 作为新光流模型研究，不能当 C1 原位替换。

- **结论：** 当前挂载点不存在


#### OF-O · TDE-Prior → residual transformer

定位：192–197；239/245。N/F/H/T：**4 / 0 / 4 / 3**。

- **原机制：** TDE 给粗速度，网络仅学残差。

- **最近先验与访问状态：** [TDE-3: An improved prior for optical flow computation in spiking neural networks](https://arxiv.org/html/2402.11662)；原文可访问；本轮概要核读，不能据此认定与 ep34 已集成

- **冻结适配：** 改变输入和预测目标，需训练；可完全数字实现。

- **真正增量：** 生物时间差前级和神经网络间真正新增分工。

- **代价与反证：** TDE 对纹理/遮挡的误差、时间戳存储、原始事件到体素桥及端到端成本。

- **迁移方式：** 保留长期算法硬件协同；先定义 residual forward，不宣称 ep34 可直接切换。

- **结论：** 有研究价值，但不宜当前短文主线


#### OF-PACK1 · C1* residual/eager/wake联合包

定位：203–211；323。N/F/H/T：**2 / 1 / 3 / 1**。

- **原机制：** 将方向、残差、遮挡与预测器并列接到C1。

- **最近先验与访问状态：** [Liu et al., An Ultra-High Performance and Scalable Optical Flow Hardware Accelerator Based on FPGA for Autonomous Driving, TCAS-I 2025](https://ieeexplore.ieee.org/document/11030859/)；本轮 IEEE 原始摘要已核：方向预测、可配置 PE、金字塔管线；未取得全文

- **冻结适配：** 多个原算法不存在的反馈边；不是ep34原位实现。

- **真正增量：** 组件堆叠不能自动增加单机制新意。

- **代价与反证：** 成本和误差耦合，TCAS-II 4.5页无法支撑每个新算法。

- **迁移方式：** 拆出一个确实少执行的原语，其余最多底座。

- **结论：** 否决整包贡献叙事


#### OF-PACK2 · C2* dual-rail/motion/sparse联合包

定位：213–220；324。N/F/H/T：**1 / 0 / 3 / 0**。

- **原机制：** 实值双轨、运动context、分型稀疏及Mask-Add堆叠。

- **最近先验与访问状态：** [Xu et al., Bishop: Sparsified Bundling Spiking Transformers on Heterogeneous Cores with Error-Constrained Pruning, ISCA 2025](https://arxiv.org/html/2505.12281)；本轮原文 §3/§5.1 已核：TTB、二值 QK 点积界、BSA／ECP-aware 训练均为已有内容

- **冻结适配：** 实值载荷和hyp身份冲突；SDSA与H67混用。

- **真正增量：** 多数是已知路由或标签换名。

- **代价与反证：** 多后端面积/训练/误差未闭，不能相乘收益。

- **迁移方式：** 去掉虚构载荷与hyp，围绕一个真实消费者不变量重写。

- **结论：** 否决整包


#### OF-PACK3 · TDE＋event-history联合前级

定位：222–223；325。N/F/H/T：**4 / 0 / 4 / 3**。

- **原机制：** 异步历史触发TDE粗流，网络做残差。

- **最近先验与访问状态：** [TDE-3: An improved prior for optical flow computation in spiking neural networks](https://arxiv.org/html/2402.11662)；原文可访问；本轮概要核读，不能据此认定与 ep34 已集成

- **冻结适配：** 需要改变输入时序和训练目标。

- **真正增量：** 比标签换名实质，但尚无连接合同。

- **代价与反证：** 前级精度、事件史内存、体素边界及训练成本。

- **迁移方式：** 保留长期独立课题，单个TDE stub不能证明整条链。

- **结论：** 远期，不为短文拼岛


#### OF-GEOM · epipolar / geometry optional

定位：34。N/F/H/T：**2 / 1 / 4 / 1**。

- **原机制：** 使用几何约束缩小对应搜索。

- **最近先验与访问状态：** [Xu et al., GMFlow: Learning Optical Flow via Global Matching, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_GMFlow_Learning_Optical_Flow_via_Global_Matching_CVPR_2022_paper.pdf)；本轮 CVF 原始摘要／引言已核；全局匹配和传播属于其算法

- **冻结适配：** ep34为二维光流，无已接入深度/相机姿态条件。

- **真正增量：** 外部几何先验引入，与现网无关。

- **代价与反证：** 姿态来源和错误几何会误删运动物体。

- **迁移方式：** 有额外传感器及新算法时另评。

- **结论：** 不作当前C1/C2


#### OF-STH · STH-Gate

定位：254。N/F/H/T：**3 / 2 / 4 / 3**。

- **原机制：** 按空间/时间head类型选不同稀疏pattern。

- **最近先验与访问状态：** Sparse VideoGen: Accelerating Video Diffusion Transformers with Spatial-Temporal Sparsity, ICML 2025；原分配文件未给可靠链接；本轮未全文核

- **冻结适配：** H67共12块统一Motion-XOR路径；无已定义head类型。

- **真正增量：** 将视频扩散的头分类搬到OF；未有自然分型证据。

- **代价与反证：** 分类器、稀疏模式/误差；Tw2不足以直接套长视频时间head。

- **迁移方式：** 先以真实贡献分布检验可分型性；必要时独立稀疏训练。

- **结论：** 未证假说，不按旧9分推进


#### OF-SMAM · SMAM-RP / dual-spike mask-add

定位：256。N/F/H/T：**2 / 1 / 6 / 1**。

- **原机制：** gate做Mask-Add，实值载荷做MAC。

- **最近先验与访问状态：** [Li et al., An Efficient Sparse Hardware Accelerator for Spike-Driven Transformer, 2025](https://arxiv.org/html/2501.07825)；本轮原文 III-C/D 已核：二值 Hadamard／token 累加／阈值 mask；不是 H67 Motion-XOR

- **冻结适配：** 原SMAM的token累加阈值mask不是Motion-XOR；不可吸收载荷不存在。

- **真正增量：** 两条已知路径相连；关键差异假设错误。

- **代价与反证：** 有真实multi-bit gate但不能称ATLIF幅度；多项score及归一化不能省。

- **迁移方式：** 借比较/位图交集电路，重建三计数＋K复用叶；新机制另命题。

- **结论：** 原RP版本淘汰


#### OF-HW01 · Ling et al., FlowAcc, DATE 2022

定位：41。N/F/H/T：**0 / 3 / 5 / 0**。

- **原机制：** 二值金字塔特征、Hamming匹配与层级规则化

- **最近先验与访问状态：** [Ling et al., FlowAcc, DATE 2022](https://doi.org/10.23919/DATE54114.2022.9774506)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** H67 已是低价二值三项分数，不是昂贵相关体。

- **真正增量：** 这是先验论文景观条目；文件未为该条单独提出已实现的新电路。

- **代价与反证：** 预筛可能无价格优势；不可忽略同零项导致漏候选。

- **迁移方式：** 用分数项分解的精确上界提前终止，比额外建一个 Hamming 模型更合理。

- **结论：** 先验来源；不得计入我们的创新数


#### OF-HW02 · Ling et al., Ultra-Flow, FPL 2022

定位：42。N/F/H/T：**0 / 1 / 4 / 0**。

- **原机制：** 单特征图多尺度匹配与复用

- **最近先验与访问状态：** [Ling et al., Ultra-Flow, FPL 2022](https://doi.org/10.1109/FPL57034.2022.00017)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** 分层 U-Net 不等于粗流驱动的迭代残差算法。

- **真正增量：** 这是先验论文景观条目；文件未为该条单独提出已实现的新电路。

- **代价与反证：** 原前向无所需控制边；ROI halo、warp、残差存储和恢复成本。

- **迁移方式：** 若坚持粗细残差需明确新 forward 和训练；不能复用 ep34 无损数字。

- **结论：** 先验来源；不得计入我们的创新数


#### OF-HW03 · Yan et al., JSA 136:102818, 2023

定位：43。N/F/H/T：**0 / 1 / 4 / 0**。

- **原机制：** 同家族DNN光流FPGA后续

- **最近先验与访问状态：** Yan et al., JSA 136:102818, 2023；未全文核；不得引用该条原文未验证的性能数字

- **冻结适配：** 分层 U-Net 不等于粗流驱动的迭代残差算法。

- **真正增量：** 这是先验论文景观条目；文件未为该条单独提出已实现的新电路。

- **代价与反证：** 原前向无所需控制边；ROI halo、warp、残差存储和恢复成本。

- **迁移方式：** 若坚持粗细残差需明确新 forward 和训练；不能复用 ep34 无损数字。

- **结论：** 先验来源；不得计入我们的创新数


#### OF-HW04 · An FPGA-based Real-Time Optical Flow Accelerator for Recurrent All-Pairs Field Transforms, ISCAS 2025

定位：44。N/F/H/T：**0 / 2 / 4 / 0**。

- **原机制：** RAFT迭代预测／跳过

- **最近先验与访问状态：** [An FPGA-based Real-Time Optical Flow Accelerator for Recurrent All-Pairs Field Transforms, ISCAS 2025](https://doi.org/10.1109/ISCAS56072.2025.11043529)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** 没有 RAFT refinement 环；PSN T10/T2 不能按收敛迭代解释。

- **真正增量：** 这是先验论文景观条目；文件未为该条单独提出已实现的新电路。

- **代价与反证：** 缺失步可能翻转未来发放；T10 固定服务不可截断。

- **迁移方式：** 只保留严格未来输入界证明的提前确定输出，非经验小 Δ。

- **结论：** 先验来源；不得计入我们的创新数


#### OF-HW05 · Li et al., RAFT-Lite FPGA, CAC 2022

定位：45。N/F/H/T：**0 / 2 / 4 / 0**。

- **原机制：** 压缩RAFT与卷积资源调度

- **最近先验与访问状态：** [Li et al., RAFT-Lite FPGA, CAC 2022](https://doi.org/10.1109/CAC57257.2022.10054761)；未全文核；不得引用该条原文未验证的性能数字

- **冻结适配：** 没有 RAFT refinement 环；PSN T10/T2 不能按收敛迭代解释。

- **真正增量：** 这是先验论文景观条目；文件未为该条单独提出已实现的新电路。

- **代价与反证：** 缺失步可能翻转未来发放；T10 固定服务不可截断。

- **迁移方式：** 只保留严格未来输入界证明的提前确定输出，非经验小 Δ。

- **结论：** 先验来源；不得计入我们的创新数


#### OF-HW06 · Liu et al., An Ultra-High Performance and Scalable Optical Flow Hardware Accelerator Based on FPGA for Autonomous Driving, TCAS-I 2025

定位：46。N/F/H/T：**0 / 1 / 5 / 0**。

- **原机制：** 方向预测与可重构金字塔管线

- **最近先验与访问状态：** [Liu et al., An Ultra-High Performance and Scalable Optical Flow Hardware Accelerator Based on FPGA for Autonomous Driving, TCAS-I 2025](https://ieeexplore.ieee.org/document/11030859/)；本轮 IEEE 原始摘要已核：方向预测、可配置 PE、金字塔管线；未取得全文

- **冻结适配：** 低；ep34 无前置当前 flow 预测器，跳过非零 tile 改变函数。

- **真正增量：** 这是先验论文景观条目；文件未为该条单独提出已实现的新电路。

- **代价与反证：** 当前 flow 用于跳过产生它的推理存在因果环；漏唤醒、边界依赖、旧结果读写未收费。

- **迁移方式：** 改成来自前一帧或便宜独立前级的因果预测；有损另开 AEE，精确版需要输出不变证明。

- **结论：** 先验来源；不得计入我们的创新数


#### OF-HW07 · Gong et al., multi-scale LK tracking accelerator, TCAS-I 2023

定位：47。N/F/H/T：**0 / 1 / 4 / 0**。

- **原机制：** 多尺度LK稀疏关键点跟踪

- **最近先验与访问状态：** [Gong et al., multi-scale LK tracking accelerator, TCAS-I 2023](https://doi.org/10.1109/TCSI.2023.3298969)；未全文核；不得引用该条原文未验证的性能数字

- **冻结适配：** ep34为二维光流，无已接入深度/相机姿态条件。

- **真正增量：** 这是先验论文景观条目；文件未为该条单独提出已实现的新电路。

- **代价与反证：** 姿态来源和错误几何会误删运动物体。

- **迁移方式：** 有额外传感器及新算法时另评。

- **结论：** 先验来源；不得计入我们的创新数


#### OF-HW08 · STMicroelectronics VD56G3 characterization, 2023

定位：48。N/F/H/T：**0 / 3 / 5 / 0**。

- **原机制：** 传感器内块匹配

- **最近先验与访问状态：** [STMicroelectronics VD56G3 characterization, 2023](https://arxiv.org/abs/2305.13087)；未全文核；不得引用该条原文未验证的性能数字

- **冻结适配：** H67 已是低价二值三项分数，不是昂贵相关体。

- **真正增量：** 这是先验论文景观条目；文件未为该条单独提出已实现的新电路。

- **代价与反证：** 预筛可能无价格优势；不可忽略同零项导致漏候选。

- **迁移方式：** 用分数项分解的精确上界提前终止，比额外建一个 Hamming 模型更合理。

- **结论：** 先验来源；不得计入我们的创新数


#### OF-HW09 · Stumpp et al., hARMS: A Hardware Acceleration Architecture for Real-Time Event-Based Optical Flow, IEEE Access 2022

定位：51。N/F/H/T：**0 / 0 / 4 / 0**。

- **原机制：** 事件历史与多尺度孔径鲁棒估计

- **最近先验与访问状态：** [Stumpp et al., hARMS: A Hardware Acceleration Architecture for Real-Time Event-Based Optical Flow, IEEE Access 2022](https://arxiv.org/html/2112.06772)；原文可访问；本轮概要核读，未逐条核实现数字

- **冻结适配：** 固定 T10 voxel/PSN 非在线异步网络。

- **真正增量：** 这是先验论文景观条目；文件未为该条单独提出已实现的新电路。

- **代价与反证：** 所有邻接窗口更新、时间戳状态、复位与精度训练代价。

- **迁移方式：** 远期流式算法线；禁止直接借 ep34 精度。

- **结论：** 先验来源；不得计入我们的创新数


#### OF-HW10 · ASNA-Flow, IEEE TVLSI 2025

定位：52。N/F/H/T：**0 / 3 / 6 / 0**。

- **原机制：** 事件光流空间局部稀疏ASIC

- **最近先验与访问状态：** [ASNA-Flow, IEEE TVLSI 2025](https://doi.org/10.1109/TVLSI.2025.3600953)；未全文核；用户冻结禁令明确排除把其空间局部性作为我们的原语

- **冻结适配：** 一般 locality 可用；未知运动驱动映射非原算法。

- **真正增量：** 这是先验论文景观条目；文件未为该条单独提出已实现的新电路。

- **代价与反证：** 用户硬禁令；路由、边界迁移和 Acc搬运可能超过节省。

- **迁移方式：** 普通空间映射当强基线；只另查时间相似与算法融合。

- **结论：** 先验来源；不得计入我们的创新数


#### OF-HW11 · Di Mauro et al., Kraken SoC

定位：53。N/F/H/T：**0 / 0 / 4 / 0**。

- **原机制：** 事件/帧融合SoC及稀疏SNN

- **最近先验与访问状态：** [Di Mauro et al., Kraken SoC](https://arxiv.org/abs/2209.01065)；未全文核；不得引用该条原文未验证的性能数字

- **冻结适配：** 固定 T10 voxel/PSN 非在线异步网络。

- **真正增量：** 这是先验论文景观条目；文件未为该条单独提出已实现的新电路。

- **代价与反证：** 所有邻接窗口更新、时间戳状态、复位与精度训练代价。

- **迁移方式：** 远期流式算法线；禁止直接借 ep34 精度。

- **结论：** 先验来源；不得计入我们的创新数


#### OF-HW12 · Xu et al., SENECA ANN/SNN event optical-flow comparison

定位：54。N/F/H/T：**0 / 4 / 6 / 0**。

- **原机制：** 神经形态平台ANN/SNN光流及稀疏化比较

- **最近先验与访问状态：** [Xu et al., SENECA ANN/SNN event optical-flow comparison](https://arxiv.org/abs/2407.20421)；未全文核；不得引用该条原文未验证的性能数字

- **冻结适配：** 天然二值可编码；以流阈值丢非零描述符为有损。

- **真正增量：** 这是先验论文景观条目；文件未为该条单独提出已实现的新电路。

- **代价与反证：** 编码器、模式标志、转换和回退需收费；event=0 不推出网络输出不变。

- **迁移方式：** 保留类型化准确脏事件；必须证明实际少算而非只少发空描述符。

- **结论：** 先验来源；不得计入我们的创新数


#### OF-HW13 · TDE-3: An improved prior for optical flow computation in spiking neural networks

定位：57。N/F/H/T：**0 / 0 / 4 / 0**。

- **原机制：** 第三抑制输入的时间差编码

- **最近先验与访问状态：** [TDE-3: An improved prior for optical flow computation in spiking neural networks](https://arxiv.org/html/2402.11662)；原文可访问；本轮概要核读，不能据此认定与 ep34 已集成

- **冻结适配：** 改变输入和预测目标，需训练；可完全数字实现。

- **真正增量：** 这是先验论文景观条目；文件未为该条单独提出已实现的新电路。

- **代价与反证：** TDE 对纹理/遮挡的误差、时间戳存储、原始事件到体素桥及端到端成本。

- **迁移方式：** 保留长期算法硬件协同；先定义 residual forward，不宣称 ep34 可直接切换。

- **结论：** 先验来源；不得计入我们的创新数


#### OF-HW14 · Digital TDE FPGA (原文件无唯一题名/DOI)

定位：58。N/F/H/T：**0 / 0 / 4 / 0**。

- **原机制：** 数字TDE FPGA运动前级

- **最近先验与访问状态：** Digital TDE FPGA (原文件无唯一题名/DOI)；未全文核；不得引用该条原文未验证的性能数字

- **冻结适配：** 改变输入和预测目标，需训练；可完全数字实现。

- **真正增量：** 这是先验论文景观条目；文件未为该条单独提出已实现的新电路。

- **代价与反证：** TDE 对纹理/遮挡的误差、时间戳存储、原始事件到体素桥及端到端成本。

- **迁移方式：** 保留长期算法硬件协同；先定义 residual forward，不宣称 ep34 可直接切换。

- **结论：** 先验来源；不得计入我们的创新数


#### OF-ALG01 · Teed and Deng, RAFT, ECCV 2020

定位：68。N/F/H/T：**0 / 2 / 4 / 0**。

- **原机制：** 全对相关＋ConvGRU迭代

- **最近先验与访问状态：** [Teed and Deng, RAFT, ECCV 2020](https://arxiv.org/abs/2003.12039)；未全文核；不得引用该条原文未验证的性能数字

- **冻结适配：** 没有 RAFT refinement 环；PSN T10/T2 不能按收敛迭代解释。

- **真正增量：** 这是先验论文景观条目；文件未为该条单独提出已实现的新电路。

- **代价与反证：** 缺失步可能翻转未来发放；T10 固定服务不可截断。

- **迁移方式：** 只保留严格未来输入界证明的提前确定输出，非经验小 Δ。

- **结论：** 先验来源；不得计入我们的创新数


#### OF-ALG02 · Gehrig et al., E-RAFT, 3DV 2021

定位：69。N/F/H/T：**0 / 2 / 4 / 0**。

- **原机制：** 事件体素上的RAFT

- **最近先验与访问状态：** [Gehrig et al., E-RAFT, 3DV 2021](https://arxiv.org/abs/2108.10552)；未全文核；不得引用该条原文未验证的性能数字

- **冻结适配：** 没有 RAFT refinement 环；PSN T10/T2 不能按收敛迭代解释。

- **真正增量：** 这是先验论文景观条目；文件未为该条单独提出已实现的新电路。

- **代价与反证：** 缺失步可能翻转未来发放；T10 固定服务不可截断。

- **迁移方式：** 只保留严格未来输入界证明的提前确定输出，非经验小 Δ。

- **结论：** 先验来源；不得计入我们的创新数


#### OF-ALG03 · Xu et al., GMFlow: Learning Optical Flow via Global Matching, CVPR 2022

定位：70。N/F/H/T：**0 / 1 / 4 / 0**。

- **原机制：** 全局匹配与自注意力传播

- **最近先验与访问状态：** [Xu et al., GMFlow: Learning Optical Flow via Global Matching, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_GMFlow_Learning_Optical_Flow_via_Global_Matching_CVPR_2022_paper.pdf)；本轮 CVF 原始摘要／引言已核；全局匹配和传播属于其算法

- **冻结适配：** ep34 没有该传播分支；遮挡不是可删的无效像素。

- **真正增量：** 这是先验论文景观条目；文件未为该条单独提出已实现的新电路。

- **代价与反证：** 双向一致性可能需额外推理；遮挡正是光流难例，邻居填充没有等价证明。

- **迁移方式：** 独立 AEE 分支；若只预取保留全部执行则创新很薄。

- **结论：** 先验来源；不得计入我们的创新数


#### OF-ALG04 · Sui et al., CRAFT: Cross-Attentional Flow Transformers for Robust Optical Flow, CVPR 2022

定位：71。N/F/H/T：**0 / 1 / 4 / 0**。

- **原机制：** 跨帧QK增强相关稳健性

- **最近先验与访问状态：** [Sui et al., CRAFT: Cross-Attentional Flow Transformers for Robust Optical Flow, CVPR 2022](https://arxiv.org/abs/2203.16896)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** 会改变冻结注意力，权重需学习/评价。

- **真正增量：** 这是先验论文景观条目；文件未为该条单独提出已实现的新电路。

- **代价与反证：** 新投影/平滑增加操作；大位移好处未由 DSEC 证明。

- **迁移方式：** 仅算法对照，不移植到 C2 标成硬件创新。

- **结论：** 先验来源；不得计入我们的创新数


#### OF-ALG05 · Huang et al., FlowFormer: A Transformer Architecture for Optical Flow, ECCV 2022

定位：72。N/F/H/T：**0 / 1 / 4 / 0**。

- **原机制：** 代价体token和位置查询解码

- **最近先验与访问状态：** [Huang et al., FlowFormer: A Transformer Architecture for Optical Flow, ECCV 2022](https://arxiv.org/abs/2203.16194)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** ep34 U-Net 无 4D cost volume。

- **真正增量：** 这是先验论文景观条目；文件未为该条单独提出已实现的新电路。

- **代价与反证：** 压缩通常有损；exact cost-token 并不自动等于 exact 原网络。

- **迁移方式：** 作为新光流模型研究，不能当 C1 原位替换。

- **结论：** 先验来源；不得计入我们的创新数


#### OF-ALG06 · Shi et al., VideoFlow: Exploiting Temporal Cues for Multi-frame Optical Flow Estimation, ICCV 2023

定位：73。N/F/H/T：**0 / 3 / 5 / 0**。

- **原机制：** 三帧双向和运动传播

- **最近先验与访问状态：** [Shi et al., VideoFlow: Exploiting Temporal Cues for Multi-frame Optical Flow Estimation, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Shi_VideoFlow_Exploiting_Temporal_Cues_for_Multi-frame_Optical_Flow_Estimation_ICCV_2023_paper.html)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** ep34 无 TROF/MOP 迭代状态；普通 token×time 可打包。

- **真正增量：** 这是先验论文景观条目；文件未为该条单独提出已实现的新电路。

- **代价与反证：** 不能借 TMA 少迭代速度；新帧上下文缓冲/身份检查。

- **迁移方式：** 抽取多目的共同表达式才改执行量；普通广播保留强基线。

- **结论：** 先验来源；不得计入我们的创新数


#### OF-ALG07 · Temporal Motion Aggregation for Event-Based Optical Flow, ICCV 2023

定位：74。N/F/H/T：**0 / 3 / 5 / 0**。

- **原机制：** 事件时间运动聚合

- **最近先验与访问状态：** [Temporal Motion Aggregation for Event-Based Optical Flow, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Temporal_Motion_Aggregation_for_Event-Based_Optical_Flow_ICCV_2023_paper.html)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** ep34 无 TROF/MOP 迭代状态；普通 token×time 可打包。

- **真正增量：** 这是先验论文景观条目；文件未为该条单独提出已实现的新电路。

- **代价与反证：** 不能借 TMA 少迭代速度；新帧上下文缓冲/身份检查。

- **迁移方式：** 抽取多目的共同表达式才改执行量；普通广播保留强基线。

- **结论：** 先验来源；不得计入我们的创新数


#### OF-ALG08 · Lee et al., Spike-FlowNet, ECCV 2020

定位：75。N/F/H/T：**0 / 4 / 6 / 0**。

- **原机制：** 混合SNN/ANN光流

- **最近先验与访问状态：** [Lee et al., Spike-FlowNet, ECCV 2020](https://arxiv.org/abs/2003.06696)；未全文核；不得引用该条原文未验证的性能数字

- **冻结适配：** 天然二值可编码；以流阈值丢非零描述符为有损。

- **真正增量：** 这是先验论文景观条目；文件未为该条单独提出已实现的新电路。

- **代价与反证：** 编码器、模式标志、转换和回退需收费；event=0 不推出网络输出不变。

- **迁移方式：** 保留类型化准确脏事件；必须证明实际少算而非只少发空描述符。

- **结论：** 先验来源；不得计入我们的创新数


#### OF-ALG09 · Tian and Andrade-Cetto, SDformerFlow, ICPR 2024

定位：76。N/F/H/T：**0 / 6 / 7 / 0**。

- **原机制：** 公开Swin脉冲光流骨架

- **最近先验与访问状态：** [Tian and Andrade-Cetto, SDformerFlow, ICPR 2024](https://arxiv.org/html/2409.04082)；原始论文可访问；冻结 H67 覆盖算术以用户合同为准，不能用公开 SDSA 代替

- **冻结适配：** H67 可借融合思想；原文 SDSA/V/LIF 后处理与 gate⊙K 不同。

- **真正增量：** 这是先验论文景观条目；文件未为该条单独提出已实现的新电路。

- **代价与反证：** 必须保持分母、舍入、零值语义；Q7/Shiftmax 属部署候选而非冻结训练算术；旧岛不收费的流量不能算收益。

- **迁移方式：** 重写因果图，以 K=V 与两时间对端为精确不变量；匹配 unfused/强融合基线。

- **结论：** 先验来源；不得计入我们的创新数


#### OF-ALG10 · Wu et al., IDNet (原文件未给唯一链接)

定位：77。N/F/H/T：**0 / 1 / 4 / 0**。

- **原机制：** 迭代去模糊替代相关体

- **最近先验与访问状态：** Wu et al., IDNet (原文件未给唯一链接)；未全文核；不得引用该条原文未验证的性能数字

- **冻结适配：** ep34 U-Net 无 4D cost volume。

- **真正增量：** 这是先验论文景观条目；文件未为该条单独提出已实现的新电路。

- **代价与反证：** 压缩通常有损；exact cost-token 并不自动等于 exact 原网络。

- **迁移方式：** 作为新光流模型研究，不能当 C1 原位替换。

- **结论：** 先验来源；不得计入我们的创新数


#### OF-ALG11 · TDE-3: An improved prior for optical flow computation in spiking neural networks

定位：78。N/F/H/T：**0 / 0 / 4 / 0**。

- **原机制：** 时间差运动先验

- **最近先验与访问状态：** [TDE-3: An improved prior for optical flow computation in spiking neural networks](https://arxiv.org/html/2402.11662)；原文可访问；本轮概要核读，不能据此认定与 ep34 已集成

- **冻结适配：** 改变输入和预测目标，需训练；可完全数字实现。

- **真正增量：** 这是先验论文景观条目；文件未为该条单独提出已实现的新电路。

- **代价与反证：** TDE 对纹理/遮挡的误差、时间戳存储、原始事件到体素桥及端到端成本。

- **迁移方式：** 保留长期算法硬件协同；先定义 residual forward，不宣称 ep34 可直接切换。

- **结论：** 先验来源；不得计入我们的创新数


#### OF-ALG12 · Greatorex et al., Event timing OF, CVPRF 2026 (原文件未给唯一题名)

定位：79。N/F/H/T：**0 / 0 / 4 / 0**。

- **原机制：** 精确事件时序／突触门控

- **最近先验与访问状态：** Greatorex et al., Event timing OF, CVPRF 2026 (原文件未给唯一题名)；未全文核；不得引用该条原文未验证的性能数字

- **冻结适配：** 改变输入和预测目标，需训练；可完全数字实现。

- **真正增量：** 这是先验论文景观条目；文件未为该条单独提出已实现的新电路。

- **代价与反证：** TDE 对纹理/遮挡的误差、时间戳存储、原始事件到体素桥及端到端成本。

- **迁移方式：** 保留长期算法硬件协同；先定义 residual forward，不宣称 ep34 可直接切换。

- **结论：** 先验来源；不得计入我们的创新数


### research/04_SYNTHESIS_C1_C2_REMAKE.md

精读完成；SHA256 `6eeb2b6983e5d7267c52eaa7d616a9de180b9730c754e1bbab1180922bad5350`。


#### R1-OP · OP-STW / DPP-Skip

定位：21；53。N/F/H/T：**3 / 1 / 5 / 2**。

- **原机制：** 方向或流残差决定 tile 是否执行。

- **最近先验与访问状态：** [Liu et al., An Ultra-High Performance and Scalable Optical Flow Hardware Accelerator Based on FPGA for Autonomous Driving, TCAS-I 2025](https://ieeexplore.ieee.org/document/11030859/)；本轮 IEEE 原始摘要已核：方向预测、可配置 PE、金字塔管线；未取得全文

- **冻结适配：** 低；ep34 无前置当前 flow 预测器，跳过非零 tile 改变函数。

- **真正增量：** 增加任务相关门控条件；仅阈值比较未构成执行原语。

- **代价与反证：** 当前 flow 用于跳过产生它的推理存在因果环；漏唤醒、边界依赖、旧结果读写未收费。

- **迁移方式：** 改成来自前一帧或便宜独立前级的因果预测；有损另开 AEE，精确版需要输出不变证明。

- **结论：** 原案不能入冻结主线；可作为算法重构入口


#### R1-PRRC · PRRC / pyramid residual capture

定位：22。N/F/H/T：**2 / 1 / 4 / 2**。

- **原机制：** 粗层写流残差，细层只算 warp 对齐 ROI。

- **最近先验与访问状态：** [Ling et al., FlowAcc, DATE 2022](https://doi.org/10.23919/DATE54114.2022.9774506)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** 分层 U-Net 不等于粗流驱动的迭代残差算法。

- **真正增量：** 新增粗流反馈与 ROI 算法，而非改现有存储政策。

- **代价与反证：** 原前向无所需控制边；ROI halo、warp、残差存储和恢复成本。

- **迁移方式：** 若坚持粗细残差需明确新 forward 和训练；不能复用 ep34 无损数字。

- **结论：** 原冻结适配失败


#### R1-OGEC · OGEC / occlusion-gated exact capture

定位：23；57。N/F/H/T：**3 / 1 / 4 / 3**。

- **原机制：** 遮挡/未匹配 token 转廉价传播通路，匹配走精确。

- **最近先验与访问状态：** [Xu et al., GMFlow: Learning Optical Flow via Global Matching, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_GMFlow_Learning_Optical_Flow_via_Global_Matching_CVPR_2022_paper.pdf)；本轮 CVF 原始摘要／引言已核；全局匹配和传播属于其算法

- **冻结适配：** ep34 没有该传播分支；遮挡不是可删的无效像素。

- **真正增量：** 选路+新增算法分支；GMFlow 已有传播。

- **代价与反证：** 双向一致性可能需额外推理；遮挡正是光流难例，邻居填充没有等价证明。

- **迁移方式：** 独立 AEE 分支；若只预取保留全部执行则创新很薄。

- **结论：** 需算法重构，不能打 exact 标签


#### R1-HBG · HBG-RP / gate＋real payload

定位：32；52；72。N/F/H/T：**1 / 0 / 6 / 0**。

- **原机制：** 独立二值门控制时钟/访存，幅度载荷做 MAC。

- **最近先验与访问状态：** AT-LIF, NeurIPS 2025（原 idea 的二值阈值发放对照）；官方题名链接本轮未核；二值 {0,theta}、静态阈值可吸收由用户冻结合同独立确定

- **冻结适配：** 与冻结 {0,theta} 直接冲突；theta静态，天然非零不是int8载荷。

- **真正增量：** 无法吸收的载荷是虚构的前提，门控本身已有。

- **代价与反证：** 增加幅度存储和乘法；负号协议不代表实值 ATLIF。

- **迁移方式：** 真实多比特 gate×weight 可另定名字和接口；不能偷换 ATLIF 出口。

- **结论：** 冻结主线淘汰


#### R1-ADP · ADP-MAC / 双边 bit particle

定位：33；56。N/F/H/T：**2 / 0 / 5 / 1**。

- **原机制：** 权重和 ATLIF 幅度双边 bit-skip、slot donation。

- **最近先验与访问状态：** [BBS: Bi-directional Bit-level Sparsity for Deep Learning Acceleration, MICRO 2024](https://arxiv.org/abs/2409.05227)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** 不适配；ATLIF 出口为二值，静态 theta 可折权重。

- **真正增量：** 多比特幅度数据通路是假定新增，不是已有算法独特性。

- **代价与反证：** 增加 bit-serial 延迟和粒度调度；自然源侧没有第二组幅度位可跳。

- **迁移方式：** 仅迁到 gate×weight 的真实多比特门控路径且重定合同；不能用 ATLIF 名义。

- **结论：** 淘汰 ATLIF 主线版本


#### R1-ARM · ARM-Acc / motion hypotheses

定位：34；54。N/F/H/T：**2 / 0 / 4 / 1**。

- **原机制：** Acc 槽存多方向假设并由证据选赢家。

- **最近先验与访问状态：** [Stumpp et al., hARMS: A Hardware Acceleration Architecture for Real-Time Event-Based Optical Flow, IEEE Access 2022](https://arxiv.org/html/2112.06772)；原文可访问；本轮概要核读，未逐条核实现数字

- **冻结适配：** ep34 通道/head 无显式运动假设和赢家提交语义。

- **真正增量：** 仅给 Acc 槽改标签不改执行；真正迁移需新算子。

- **代价与反证：** K 假设会增加计算/状态；hARMS 多尺度估计不等同 Transformer head。

- **迁移方式：** 只有明确定义并训练 hypothesis tensor 后再考虑共享求证算子。

- **结论：** 原案为语义换名，淘汰


#### R1-MFBD · MFBD / motion-feature bundles

定位：34；55。N/F/H/T：**2 / 3 / 5 / 2**。

- **原机制：** 时间相邻运动状态束内共享权重并广播。

- **最近先验与访问状态：** [Shi et al., VideoFlow: Exploiting Temporal Cues for Multi-frame Optical Flow Estimation, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Shi_VideoFlow_Exploiting_Temporal_Cues_for_Multi-frame_Optical_Flow_Estimation_ICCV_2023_paper.html)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** ep34 无 TROF/MOP 迭代状态；普通 token×time 可打包。

- **真正增量：** 同权重广播本质未改，运动标签不增加代数机会。

- **代价与反证：** 不能借 TMA 少迭代速度；新帧上下文缓冲/身份检查。

- **迁移方式：** 抽取多目的共同表达式才改执行量；普通广播保留强基线。

- **结论：** 降为基线，不作主机制


#### R1-SP · SP-Gate / attention mass

定位：35；61。N/F/H/T：**3 / 3 / 5 / 3**。

- **原机制：** 小注意力质量不发后级请求。

- **最近先验与访问状态：** [SpAtten: Efficient Sparse Attention Architecture with Cascade Token and Head Pruning, HPCA 2021](https://arxiv.org/abs/2012.09852)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** H67 分数含共静默/时间项，低 Q 发放不等于低 gate；K=V。

- **真正增量：** 非零重要性裁剪是现有稀疏注意力迁移。

- **代价与反证：** 省流量前要算分数；有损阈值改变归一化分母和输出。

- **迁移方式：** 寻找量化门控精确为零的充分条件，并收费界计算；部署合同另列。

- **结论：** 保留精确界重构


#### R1-FUSE · SS-FSA / Flash-SDSA fusion

定位：35。N/F/H/T：**4 / 6 / 7 / 5**。

- **原机制：** 分数、归一化和加权输出融合，避免中间图写回。

- **最近先验与访问状态：** [FuseMax: Leveraging Extended Einsums to Optimize Attention Accelerator Design, MICRO 2024](https://arxiv.org/html/2406.10491)；本轮原文融合与映射章节已核；one-pass cascade 与最大融合均为已有机制

- **冻结适配：** H67 可借融合思想；原文 SDSA/V/LIF 后处理与 gate⊙K 不同。

- **真正增量：** 原案 score-stationary 已有；实际空间在 H67 三项分数—门控—K 消费的专属生命周期。

- **代价与反证：** 必须保持分母、舍入、零值语义；Q7/Shiftmax 属部署候选而非冻结训练算术；旧岛不收费的流量不能算收益。

- **迁移方式：** 重写因果图，以 K=V 与两时间对端为精确不变量；匹配 unfused/强融合基线。

- **结论：** 值得算法原生重构，原名本身非创新


#### R1-TDE · TDE-Prior → residual transformer

定位：44；58；74。N/F/H/T：**4 / 0 / 4 / 3**。

- **原机制：** TDE 给粗速度，网络仅学残差。

- **最近先验与访问状态：** [TDE-3: An improved prior for optical flow computation in spiking neural networks](https://arxiv.org/html/2402.11662)；原文可访问；本轮概要核读，不能据此认定与 ep34 已集成

- **冻结适配：** 改变输入和预测目标，需训练；可完全数字实现。

- **真正增量：** 生物时间差前级和神经网络间真正新增分工。

- **代价与反证：** TDE 对纹理/遮挡的误差、时间戳存储、原始事件到体素桥及端到端成本。

- **迁移方式：** 保留长期算法硬件协同；先定义 residual forward，不宣称 ep34 可直接切换。

- **结论：** 有研究价值，但不宜当前短文主线


#### R1-VGTS · VGTS / value-leak timestep skipping

定位：45；59。N/F/H/T：**3 / 2 / 4 / 3**。

- **原机制：** 按幅度与泄漏预测跨过无发放时间。

- **最近先验与访问状态：** [Hardware Efficient Accelerator for Spiking Transformer With Reconfigurable Parallel Time Step Computing, ISCAS 2025](https://arxiv.org/abs/2503.19643)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** PSN学习时间耦合，不能直接套递推 LIF 泄漏；出口仍二值。

- **真正增量：** 如有严格跨步闭式状态可改变执行，但原文没有给。

- **代价与反证：** 预测错会影响后续阈值；跨帧状态复位和T10全覆盖。

- **迁移方式：** 推导固定 T10 对真实 PSN 的精确状态转换后单独评价。

- **结论：** 有条件神经元研究，不是第三加速


#### R1-REMEM · ReMem-Tok / overlapping-window membrane reuse

定位：45；60。N/F/H/T：**2 / 3 / 4 / 2**。

- **原机制：** shifted-window 重叠时沿用膜状态。

- **最近先验与访问状态：** [Hardware Efficient Accelerator for Spiking Transformer With Reconfigurable Parallel Time Step Computing, ISCAS 2025](https://arxiv.org/abs/2503.19643)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** 不同块权重/阈值/时间上下文不等价，空间重叠不证明膜可重用。

- **真正增量：** 基本是状态缓存，尚无输出恒等式。

- **代价与反证：** 错误跨block复用；影子状态和版本标签成本。

- **迁移方式：** 仅同层相同消费者的状态生命周期消除，不凭窗口重叠重用。

- **结论：** 原方案不成立


#### R1-EESUC · EESUC / early-exit spike update

定位：46。N/F/H/T：**2 / 2 / 4 / 2**。

- **原机制：** 小 flow/膜残差就减少后续时间步或迭代。

- **最近先验与访问状态：** [An FPGA-based Real-Time Optical Flow Accelerator for Recurrent All-Pairs Field Transforms, ISCAS 2025](https://doi.org/10.1109/ISCAS56072.2025.11043529)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** 没有 RAFT refinement 环；PSN T10/T2 不能按收敛迭代解释。

- **真正增量：** 早停算法已知，现案无可用终止不变量。

- **代价与反证：** 缺失步可能翻转未来发放；T10 固定服务不可截断。

- **迁移方式：** 只保留严格未来输入界证明的提前确定输出，非经验小 Δ。

- **结论：** 冻结版本否决；证书方向可研究


#### R1-MSBC · MSBC / MSBC-Cascade

定位：61。N/F/H/T：**3 / 2 / 6 / 3**。

- **原机制：** motion saliency×attention mass 级联裁 token/head/time bundle。

- **最近先验与访问状态：** [SpAtten: Efficient Sparse Attention Architecture with Cascade Token and Head Pruning, HPCA 2021](https://arxiv.org/abs/2012.09852)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** 需新算法评价；固定密集光流输出不能直接丢 token。

- **真正增量：** 只把已有重要性函数换输入语义；新硬件未明确。

- **代价与反证：** top-k、选择器、被删 token 恢复与误差传播；小流速不代表低重要性。

- **迁移方式：** 研究可证明结果不变的门控界；若有损须按遮挡和边界分层 AEE。

- **结论：** 保留算法候选，不当已成立电路贡献


#### R1-PACK1 · C1* residual/eager/wake联合包

定位：17–26；89–95。N/F/H/T：**2 / 1 / 3 / 1**。

- **原机制：** 将方向、残差、遮挡与预测器并列接到C1。

- **最近先验与访问状态：** [Liu et al., An Ultra-High Performance and Scalable Optical Flow Hardware Accelerator Based on FPGA for Autonomous Driving, TCAS-I 2025](https://ieeexplore.ieee.org/document/11030859/)；本轮 IEEE 原始摘要已核：方向预测、可配置 PE、金字塔管线；未取得全文

- **冻结适配：** 多个原算法不存在的反馈边；不是ep34原位实现。

- **真正增量：** 组件堆叠不能自动增加单机制新意。

- **代价与反证：** 成本和误差耦合，TCAS-II 4.5页无法支撑每个新算法。

- **迁移方式：** 拆出一个确实少执行的原语，其余最多底座。

- **结论：** 否决整包贡献叙事


#### R1-PACK2 · C2* dual-rail/motion/sparse联合包

定位：28–38；89–95。N/F/H/T：**1 / 0 / 3 / 0**。

- **原机制：** 实值双轨、运动context、分型稀疏及Mask-Add堆叠。

- **最近先验与访问状态：** [Xu et al., Bishop: Sparsified Bundling Spiking Transformers on Heterogeneous Cores with Error-Constrained Pruning, ISCA 2025](https://arxiv.org/html/2505.12281)；本轮原文 §3/§5.1 已核：TTB、二值 QK 点积界、BSA／ECP-aware 训练均为已有内容

- **冻结适配：** 实值载荷和hyp身份冲突；SDSA与H67混用。

- **真正增量：** 多数是已知路由或标签换名。

- **代价与反证：** 多后端面积/训练/误差未闭，不能相乘收益。

- **迁移方式：** 去掉虚构载荷与hyp，围绕一个真实消费者不变量重写。

- **结论：** 否决整包


#### R1-SOURCE01 · SDformerFlow

定位：67。N/F/H/T：**0 / 6 / 7 / 0**。

- **原机制：** 分数、归一化和加权输出融合，避免中间图写回。

- **最近先验与访问状态：** [SDformerFlow](https://arxiv.org/html/2409.04082)；原文件来源表；未将其他模型的算法性质套到冻结ep34，具体原文访问状态见同主题提案条目

- **冻结适配：** H67 可借融合思想；原文 SDSA/V/LIF 后处理与 gate⊙K 不同。

- **真正增量：** 算法来源表，不是已给出的新电路。

- **代价与反证：** 必须保持分母、舍入、零值语义；Q7/Shiftmax 属部署候选而非冻结训练算术；旧岛不收费的流量不能算收益。

- **迁移方式：** 重写因果图，以 K=V 与两时间对端为精确不变量；匹配 unfused/强融合基线。

- **结论：** 来源背景；不计主创新


#### R1-SOURCE02 · TMA, ICCV 2023

定位：68。N/F/H/T：**0 / 3 / 5 / 0**。

- **原机制：** 时间相邻运动状态束内共享权重并广播。

- **最近先验与访问状态：** TMA, ICCV 2023；原文件来源表；未将其他模型的算法性质套到冻结ep34，具体原文访问状态见同主题提案条目

- **冻结适配：** ep34 无 TROF/MOP 迭代状态；普通 token×time 可打包。

- **真正增量：** 算法来源表，不是已给出的新电路。

- **代价与反证：** 不能借 TMA 少迭代速度；新帧上下文缓冲/身份检查。

- **迁移方式：** 抽取多目的共同表达式才改执行量；普通广播保留强基线。

- **结论：** 来源背景；不计主创新


#### R1-SOURCE03 · VideoFlow, ICCV 2023

定位：69。N/F/H/T：**0 / 3 / 5 / 0**。

- **原机制：** 时间相邻运动状态束内共享权重并广播。

- **最近先验与访问状态：** VideoFlow, ICCV 2023；原文件来源表；未将其他模型的算法性质套到冻结ep34，具体原文访问状态见同主题提案条目

- **冻结适配：** ep34 无 TROF/MOP 迭代状态；普通 token×time 可打包。

- **真正增量：** 算法来源表，不是已给出的新电路。

- **代价与反证：** 不能借 TMA 少迭代速度；新帧上下文缓冲/身份检查。

- **迁移方式：** 抽取多目的共同表达式才改执行量；普通广播保留强基线。

- **结论：** 来源背景；不计主创新


#### R1-SOURCE04 · GMFlow, CVPR 2022

定位：70。N/F/H/T：**0 / 1 / 4 / 0**。

- **原机制：** 遮挡/未匹配 token 转廉价传播通路，匹配走精确。

- **最近先验与访问状态：** [GMFlow, CVPR 2022](https://arxiv.org/abs/2111.13680)；原文件来源表；未将其他模型的算法性质套到冻结ep34，具体原文访问状态见同主题提案条目

- **冻结适配：** ep34 没有该传播分支；遮挡不是可删的无效像素。

- **真正增量：** 算法来源表，不是已给出的新电路。

- **代价与反证：** 双向一致性可能需额外推理；遮挡正是光流难例，邻居填充没有等价证明。

- **迁移方式：** 独立 AEE 分支；若只预取保留全部执行则创新很薄。

- **结论：** 来源背景；不计主创新


#### R1-SOURCE05 · Spike-driven Transformer / V2 / QSD / SFA 多比特或近似发放模型族

定位：71。N/F/H/T：**0 / 0 / 6 / 0**。

- **原机制：** 独立二值门控制时钟/访存，幅度载荷做 MAC。

- **最近先验与访问状态：** Spike-driven Transformer / V2 / QSD / SFA 多比特或近似发放模型族；原文件来源表；未将其他模型的算法性质套到冻结ep34，具体原文访问状态见同主题提案条目

- **冻结适配：** 与冻结 {0,theta} 直接冲突；theta静态，天然非零不是int8载荷。

- **真正增量：** 算法来源表，不是已给出的新电路。

- **代价与反证：** 增加幅度存储和乘法；负号协议不代表实值 ATLIF。

- **迁移方式：** 真实多比特 gate×weight 可另定名字和接口；不能偷换 ATLIF 出口。

- **结论：** 来源背景；不计主创新


#### R1-SOURCE06 · AT-LIF, NeurIPS 2025

定位：72。N/F/H/T：**0 / 0 / 6 / 0**。

- **原机制：** 独立二值门控制时钟/访存，幅度载荷做 MAC。

- **最近先验与访问状态：** AT-LIF, NeurIPS 2025；原文件来源表；未将其他模型的算法性质套到冻结ep34，具体原文访问状态见同主题提案条目

- **冻结适配：** 与冻结 {0,theta} 直接冲突；theta静态，天然非零不是int8载荷。

- **真正增量：** 算法来源表，不是已给出的新电路。

- **代价与反证：** 增加幅度存储和乘法；负号协议不代表实值 ATLIF。

- **迁移方式：** 真实多比特 gate×weight 可另定名字和接口；不能偷换 ATLIF 出口。

- **结论：** 来源背景；不计主创新


#### R1-SOURCE07 · PSN: Parallel Spiking Neurons, NeurIPS 2023

定位：73。N/F/H/T：**0 / 2 / 5 / 0**。

- **原机制：** 并行展开时间步，共享权重并减少膜状态访存。

- **最近先验与访问状态：** PSN: Parallel Spiking Neurons, NeurIPS 2023；原文件来源表；未将其他模型的算法性质套到冻结ep34，具体原文访问状态见同主题提案条目

- **冻结适配：** 真实时间为 T10 神经元/T2 窗口；非多比特输出 T4/8。

- **真正增量：** 算法来源表，不是已给出的新电路。

- **代价与反证：** 复制时间 lane 面积、PSN 时间混合、权重扇出与状态提交；不能宣称膜完全消失。

- **迁移方式：** 只比较精确 T10 状态服务与 T2 成对分数，作为配套底座。

- **结论：** 来源背景；不计主创新


#### R1-SOURCE08 · TDE-3

定位：74。N/F/H/T：**0 / 0 / 4 / 0**。

- **原机制：** TDE 给粗速度，网络仅学残差。

- **最近先验与访问状态：** [TDE-3](https://arxiv.org/html/2402.11662)；原文件来源表；未将其他模型的算法性质套到冻结ep34，具体原文访问状态见同主题提案条目

- **冻结适配：** 改变输入和预测目标，需训练；可完全数字实现。

- **真正增量：** 算法来源表，不是已给出的新电路。

- **代价与反证：** TDE 对纹理/遮挡的误差、时间戳存储、原始事件到体素桥及端到端成本。

- **迁移方式：** 保留长期算法硬件协同；先定义 residual forward，不宣称 ep34 可直接切换。

- **结论：** 来源背景；不计主创新


### research/07_transformer_attention_accelerators.md

精读完成；SHA256 `b584e10bbeec11724003bbd569f0448455e88bb4add587753a339b17f9dc86ce`。


#### ATT-M1a · MSBC / MSBC-Cascade

定位：26–30；192。N/F/H/T：**3 / 2 / 6 / 3**。

- **原机制：** motion saliency×attention mass 级联裁 token/head/time bundle。

- **最近先验与访问状态：** [SpAtten: Efficient Sparse Attention Architecture with Cascade Token and Head Pruning, HPCA 2021](https://arxiv.org/abs/2012.09852)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** 需新算法评价；固定密集光流输出不能直接丢 token。

- **真正增量：** 只把已有重要性函数换输入语义；新硬件未明确。

- **代价与反证：** top-k、选择器、被删 token 恢复与误差传播；小流速不代表低重要性。

- **迁移方式：** 研究可证明结果不变的门控界；若有损须按遮挡和边界分层 AEE。

- **结论：** 保留算法候选，不当已成立电路贡献


#### ATT-M1b · FC-PP / progressive precision

定位：28–29；214。N/F/H/T：**3 / 2 / 5 / 3**。

- **原机制：** 置信度决定先算 MSB、后补 LSB 或异常值路径。

- **最近先验与访问状态：** [SpAtten: Efficient Sparse Attention Architecture with Cascade Token and Head Pruning, HPCA 2021](https://arxiv.org/abs/2012.09852)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** ATLIF 本身无幅度位；真实权重/门控可以研究独立部署量化。

- **真正增量：** 任务置信度接入已有渐进精度；暂缺新的进位/界限电路。

- **代价与反证：** 部分和要能恢复、额外重读、误差校准；不能保证只需低位。

- **迁移方式：** 转为 Acc/门控结果的区间证书；结果确定即停止剩余位或剩余项。

- **结论：** 保留精确证书重构抓手


#### ATT-M2 · Hyp-Candidate Select / A3

定位：32–36；199；226。N/F/H/T：**2 / 2 / 4 / 2**。

- **原机制：** 近似关系检索挑少量 K 假设。

- **最近先验与访问状态：** [Ham et al., A3: Accelerating Attention Mechanisms in Neural Networks with Approximation, HPCA 2020](https://arxiv.org/abs/2002.10941)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** 无 motion hypothesis tensor；可换成H67候选过滤但改变结果。

- **真正增量：** 已有候选筛选换任务输入。

- **代价与反证：** 候选搜索、阈值和恢复；错删motion项强候选。

- **迁移方式：** 用三计数分数界做精确过滤或另开 AEE。

- **结论：** 原案不作主机制


#### ATT-M3 · SS-FSA-OF / Sanger

定位：38–42；196。N/F/H/T：**2 / 3 / 5 / 3**。

- **原机制：** 低比特预测 mask，结构化pack，score stationary 执行。

- **最近先验与访问状态：** [Sanger: A Co-Design Framework for Enabling Sparse Attention using Reconfigurable Architecture, MICRO 2021](https://doi.org/10.1145/3466752.3480125)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** 无多比特ATLIF包络；H67不是一般QK softmax AV。

- **真正增量：** 预测＋pack＋驻留均已有；任务标签不新。

- **代价与反证：** 错mask、pack、score buffer/PE网络成本。

- **迁移方式：** 只借精确融合生命周期，重建H67因果图。

- **结论：** 配套融合来源


#### ATT-M4 · SRP-OGEC / relation hash

定位：44–48；198；215。N/F/H/T：**2 / 2 / 4 / 2**。

- **原机制：** 符号随机投影过滤不相关QK，转传播路径。

- **最近先验与访问状态：** [Ham et al., ELSA: Hardware-Software Co-design for Efficient, Lightweight Self-Attention, ISCA 2021](https://taejunham.github.io/data/elsa_isca21.pdf)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** 二值 H67 已轻；sign hash不能判定遮挡。

- **真正增量：** 在旧过滤器上加遮挡标签，没有因果证书。

- **代价与反证：** 误检/漏检与额外投影成本；匹配弱不等于遮挡。

- **迁移方式：** 删除“遮挡检测等价”说法，只保留带界候选过滤研究。

- **结论：** 原语义推断失败


#### ATT-M5 · Flow-DOTA Detector

定位：50–54；197；215。N/F/H/T：**3 / 2 / 4 / 3**。

- **原机制：** 小检测器根据事件/流残差裁边。

- **最近先验与访问状态：** [Qu et al., DOTA: Detect and Omit Weak Attentions for Scalable Transformer Acceleration, ASPLOS 2022](https://par.nsf.gov/servlets/purl/10357543)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** 需要训练新检测器并评 AEE，当前flow可得性不明。

- **真正增量：** 检测器输入换OF特征；核心既有。

- **代价与反证：** 计算检测器及特征来源，改变归一化和光流边界。

- **迁移方式：** 仅研究前置因果证书，或坦诚有损联合训练分支。

- **结论：** 另开算法评价


#### ATT-M6 · ECP-QKV / FACT before projection

定位：56–60；188；213。N/F/H/T：**4 / 3 / 5 / 4**。

- **原机制：** 投影前以便宜特征预测相关性并删 Q/K/V 投影。

- **最近先验与访问状态：** [Qin et al., FACT: FFN-Attention Co-optimized Transformer Architecture with Eager Correlation Prediction, ISCA 2023](https://doi.org/10.1145/3579371.3589057)；本轮 Crossref 出版者题名／作者／日期已核；ACM 403、Crossref 无摘要，具体算法未全文核

- **冻结适配：** K兼V，没有独立V；投影前无 sn_q/sn_k，也无实值ATLIF直方图。

- **真正增量：** “提前于昂贵投影”是有价值执行位置；原案只是换预测特征。

- **代价与反证：** 模型预测代价/误判；K被attention与value共同消费，裁K可能联动两路。

- **迁移方式：** 重构为二值阈值投影结果的精确确定证书，需从当前输入与静态W推导，不用未来flow。

- **结论：** 保留执行位置；FACT全文未核，不能高分定案


#### ATT-M7 · Pyramid-MRF / Energon

定位：62–66；194；214。N/F/H/T：**2 / 2 / 5 / 2**。

- **原机制：** 低位多轮筛选后高位精算。

- **最近先验与访问状态：** [Energon: Towards Efficient Acceleration of Transformers Using Dynamic Sparse Attention, TCAD 2022](https://arxiv.org/abs/2110.09310)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** Swin多尺度不是同一候选的多轮精化；自然QK已二值。

- **真正增量：** 多轮精度过滤已有。

- **代价与反证：** 二值正式分数本来便宜，重复筛选可能加工作；不能直接按stage套轮次。

- **迁移方式：** 按H67各分数项的界渐进计算而非凭空建INT2/4幅度。

- **结论：** 原案低适配


#### ATT-M8 · DynaTran / AccelTran

定位：68–72。N/F/H/T：**1 / 2 / 6 / 1**。

- **原机制：** 运行时激活剪枝配 tiled dataflow。

- **最近先验与访问状态：** [AccelTran: A Sparsity-Aware Accelerator for Dynamic Inference with Transformers](https://arxiv.org/abs/2302.14705)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** 非零阈值删除改变ep34；普通零门可精确。

- **真正增量：** 常见动态剪枝，HBG接口不存在。

- **代价与反证：** 索引/打包成本和AEE；已有稀疏比较必须强。

- **迁移方式：** 只作动态剪枝/零跳过对照。

- **结论：** 基线，不作主机制


#### ATT-M9a · Motion-TTB / motion bundle keys

定位：78–87；189；227。N/F/H/T：**1 / 4 / 6 / 2**。

- **原机制：** 把token-time改为(tile,dt,hyp)并共享权重。

- **最近先验与访问状态：** [Xu et al., Bishop: Sparsified Bundling Spiking Transformers on Heterogeneous Cores with Error-Constrained Pruning, ISCA 2025](https://arxiv.org/html/2505.12281)；本轮原文 §3/§5.1 已核：TTB、二值 QK 点积界、BSA／ECP-aware 训练均为已有内容

- **冻结适配：** token×T可用，但没有hyp；T2/T10要分开。

- **真正增量：** TTB与束内/束间权重复用已做；键改名无执行变化。

- **代价与反证：** 打包和尾部利用率；不能把Bishop倍数搬到本岛。

- **迁移方式：** 只保留TTB强基线；新的多目的公共部分和另研究。

- **结论：** 降为强基线


#### ATT-M9b · OF-ECP / error-bounded bundle prune

定位：80/83/86；189；228。N/F/H/T：**4 / 3 / 5 / 4**。

- **原机制：** 以bundle活跃计数限制注意力删边误差。

- **最近先验与访问状态：** [Xu et al., Bishop: Sparsified Bundling Spiking Transformers on Heterogeneous Cores with Error-Constrained Pruning, ISCA 2025](https://arxiv.org/html/2505.12281)；本轮原文 §3/§5.1 已核：TTB、二值 QK 点积界、BSA／ECP-aware 训练均为已有内容

- **冻结适配：** Bishop界是二值QK；H67共静默/时间项、门控分母和K=V均改变界。

- **真正增量：** 若推导新算术的可执行精确界，可能有实质增量；原文直接写AEE界不成立。

- **代价与反证：** 局部分数误差不能直接变全网AEE界；裁Q与裁K不对称。

- **迁移方式：** 优先推导门控舍入后严格零/严格相同输出证书，非把classification界改名。

- **结论：** 保留严谨重构，不接受现成AEE界


#### ATT-M9c · Bishop-style dense/sparse stratifier

定位：80/84；224。N/F/H/T：**1 / 6 / 7 / 2**。

- **原机制：** 按spike bundle密度分派密核和稀核。

- **最近先验与访问状态：** [Xu et al., Bishop: Sparsified Bundling Spiking Transformers on Heterogeneous Cores with Error-Constrained Pruning, ISCA 2025](https://arxiv.org/html/2505.12281)；本轮原文 §3/§5.1 已核：TTB、二值 QK 点积界、BSA／ECP-aware 训练均为已有内容

- **冻结适配：** 二值密度真实可测；不存在amp密度第二轴。

- **真正增量：** 异构密稀核与分层路由已做。

- **代价与反证：** 复制执行核面积与交汇状态；需要同资源公平基线。

- **迁移方式：** 作为执行底座或基线，不作命名创新。

- **结论：** 配套基线


#### ATT-M10 · SMAM-RP / dual-spike mask-add

定位：89–93；190；224。N/F/H/T：**2 / 1 / 6 / 1**。

- **原机制：** gate做Mask-Add，实值载荷做MAC。

- **最近先验与访问状态：** [Li et al., An Efficient Sparse Hardware Accelerator for Spike-Driven Transformer, 2025](https://arxiv.org/html/2501.07825)；本轮原文 III-C/D 已核：二值 Hadamard／token 累加／阈值 mask；不是 H67 Motion-XOR

- **冻结适配：** 原SMAM的token累加阈值mask不是Motion-XOR；不可吸收载荷不存在。

- **真正增量：** 两条已知路径相连；关键差异假设错误。

- **代价与反证：** 有真实multi-bit gate但不能称ATLIF幅度；多项score及归一化不能省。

- **迁移方式：** 借比较/位图交集电路，重建三计数＋K复用叶；新机制另命题。

- **结论：** 原RP版本淘汰


#### ATT-M11 · HeatFlow-Tok

定位：99–103；192；213。N/F/H/T：**3 / 2 / 5 / 3**。

- **原机制：** 逐层根据事件/残差剪并合并token。

- **最近先验与访问状态：** [HeatViT: Hardware-Efficient Adaptive Token Pruning for Vision Transformers, HPCA 2023](https://arxiv.org/abs/2211.08110)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** 密集flowhead和skip需要完整空间网格。

- **真正增量：** 既有selector换OF特征。

- **代价与反证：** 复原被删位置与位置信息、选择器训练和成本。

- **迁移方式：** 若有损另开AEE；精确版仅安全零区域可删。

- **结论：** 算法候选，非现有硬件创新


#### ATT-M12 · Polar-Flow-Attn

定位：105–109；193；215。N/F/H/T：**2 / 2 / 5 / 2**。

- **原机制：** 平滑区稀疏、遮挡区稠密固定模式，双引擎。

- **最近先验与访问状态：** [ViTCoD: Vision Transformer Acceleration via Dedicated Algorithm and Accelerator Co-Design, HPCA 2023](https://github.com/GATECH-EIC/ViTCoD)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** ep34注意力未训练成极化模式。

- **真正增量：** 已有极化策略加语义条件。

- **代价与反证：** 稀密两套资源、检测器、mask误差与AEE。

- **迁移方式：** 当新稀疏训练基线，勿当自然结构。

- **结论：** 另开训练


#### ATT-M13 · Auto-ViT-Acc mixed scheme

定位：111–115；225。N/F/H/T：**1 / 5 / 7 / 1**。

- **原机制：** 固定点/PoT混合量化及FPGA自动映射。

- **最近先验与访问状态：** [Auto-ViT-Acc: An FPGA-Aware Automatic Acceleration Framework for Vision Transformer with Mixed-Scheme Quantization, FPL 2022](https://doi.org/10.1109/FPL57034.2022.00027)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** 真实权重可另评价；冻结hardware_quant=false，不能改变部署身份不报。

- **真正增量：** 通用量化映射工具，不是新机制。

- **代价与反证：** 量化误差与硬件协议；PoT权重不令Q1.7门控自动无乘法。

- **迁移方式：** 用于部署搜索和基线。

- **结论：** 工程工具


#### ATT-M14 · SS-FSA / Flash-SDSA fusion

定位：121–129；195；214。N/F/H/T：**4 / 6 / 7 / 5**。

- **原机制：** 分数、归一化和加权输出融合，避免中间图写回。

- **最近先验与访问状态：** [FuseMax: Leveraging Extended Einsums to Optimize Attention Accelerator Design, MICRO 2024](https://arxiv.org/html/2406.10491)；本轮原文融合与映射章节已核；one-pass cascade 与最大融合均为已有机制

- **冻结适配：** H67 可借融合思想；原文 SDSA/V/LIF 后处理与 gate⊙K 不同。

- **真正增量：** 原案 score-stationary 已有；实际空间在 H67 三项分数—门控—K 消费的专属生命周期。

- **代价与反证：** 必须保持分母、舍入、零值语义；Q7/Shiftmax 属部署候选而非冻结训练算术；旧岛不收费的流量不能算收益。

- **迁移方式：** 重写因果图，以 K=V 与两时间对端为精确不变量；匹配 unfused/强融合基线。

- **结论：** 值得算法原生重构，原名本身非创新


#### ATT-M15 · Hybrid-score / Linear-payload

定位：131–135；201；224。N/F/H/T：**2 / 2 / 5 / 2**。

- **原机制：** score用log混合表示，payload在线性域。

- **最近先验与访问状态：** [H-FA: A Hybrid Floating-Point and Logarithmic Approach to Hardware Accelerated FlashAttention](https://arxiv.org/abs/2511.00295)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** H67原生分数已为计数；实值ATLIF前提错误。

- **真正增量：** 已有数制分工。

- **代价与反证：** 转换、舍入、指数/分母代价；部署数字要另标。

- **迁移方式：** 先比较计数/定点专用叶，只有更优等价编码才值得做。

- **结论：** 不作当前主机制


#### ATT-M16 · Butterfly-Coarse / SDSA-Fine

定位：137–142；200。N/F/H/T：**2 / 1 / 4 / 1**。

- **原机制：** 粗层蝶形近似注意力/FFN，细层稀疏注意力。

- **最近先验与访问状态：** [FABNet: Adaptable Butterfly Accelerator for Attention-based NNs via Hardware and Algorithm Co-design, MICRO 2022](https://arxiv.org/abs/2209.09570)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** 替换算子和网络，不是ep34映射。

- **真正增量：** 现有蝶形后端与精细后端拼接。

- **代价与反证：** 重训、两后端资源、精度及跨尺度接口。

- **迁移方式：** 新模型方向，当前短文不扩大范围。

- **结论：** 冻结不适配


#### ATT-M17 · PagedAttention analogy

定位：144–148。N/F/H/T：**0 / 1 / 5 / 0**。

- **原机制：** 分页非连续KV管理。

- **最近先验与访问状态：** [Kwon et al., PagedAttention, SOSP 2023](https://arxiv.org/abs/2309.06180)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** 固定短窗无自回归长KV和多租户。

- **真正增量：** 仅弱缓存类比。

- **代价与反证：** 页表、TLB/碎片管理白付；不是瓶颈。

- **迁移方式：** 不迁移。

- **结论：** 明确无关


#### ATT-M18 · Sparseloop taxonomy

定位：152–158。N/F/H/T：**0 / 9 / 9 / 0**。

- **原机制：** 区分representation/gating/skipping并建模。

- **最近先验与访问状态：** [Wu et al., Sparseloop, MICRO 2022](https://arxiv.org/abs/2205.05826)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** 适合标签与评价纪律，不提供RTL闭环。

- **真正增量：** 方法学引用，无新电路。

- **代价与反证：** 随机密度模型不能代替ep34轨迹和实测。

- **迁移方式：** 用于公平基线和收费表，沿用VCS/DC/PT/FM准入。

- **结论：** 方法学，不计贡献


#### ATT-M19 · ST-Online-Wake / FlightVGM

定位：164–168；213。N/F/H/T：**3 / 2 / 4 / 3**。

- **原机制：** 用视频时空相似进行在线剪枝和混合精度。

- **最近先验与访问状态：** [FlightVGM: Efficient Video Generation Model Inference with Online Sparsification and Hybrid Precision on FPGAs, FPGA 2025](https://doi.org/10.1145/3706628.3708864)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** ep34不是视频扩散过程，时间轴要重新定义。

- **真正增量：** 领域迁移；未给可执行精确更新规则。

- **代价与反证：** 时空统计、误差、压缩解压和恢复成本。

- **迁移方式：** 只用真实层间输入差分，检查时间耦合后构造精确变更传播。

- **结论：** 有条件差分重构


#### ATT-M20 · Trajectory-Reorder Attn

定位：170–174；191；226。N/F/H/T：**2 / 2 / 5 / 2**。

- **原机制：** 以轨迹/几何重排得到块结构，再稀疏量化。

- **最近先验与访问状态：** [Zhao et al., PAROAttention: Pattern-Aware ReOrdering for Efficient Sparse and Quantized Attention in Visual Generation Models, 2025](https://arxiv.org/abs/2506.16054)；本轮 arXiv 原始摘要已核；原文件将其笼统称专用 PE 加速器，全文未核，暂不承接 ASIC 实现断言

- **冻结适配：** 当前无轨迹或对极约束；位置偏置和窗口mask必须同步映射。

- **真正增量：** 全排列等价则只有调度变化；删边/量化则新算法。

- **代价与反证：** 排序/逆置换、mask一致性、轨迹来源；仅换序不能降低算术量。

- **迁移方式：** 只当结构稀疏对照；需证明真正移除哪些执行。

- **结论：** 原主创新不足


#### ATT-M21 · CW-Reuse / latent channel reuse

定位：176–180。N/F/H/T：**3 / 4 / 5 / 3**。

- **原机制：** 跨时空复用通道部分结果。

- **最近先验与访问状态：** [Miao et al., Kaleido: Algorithm-Hardware Co-Design for Video Diffusion Transformers by Exploiting Latent Space Correlations, 2026](https://arxiv.org/abs/2607.13770)；本轮 arXiv 原始摘要已核；通道部分结果复用＋硬件已存在，具体误差与实现未全文核

- **冻结适配：** 二值相同可精确复用；“相似”膜/激活不能直接代替。

- **真正增量：** 通道复用与专用硬件先验已存在。

- **代价与反证：** 比较、参考状态、warp及失配修正；全通道真实相等率未给。

- **迁移方式：** 改为精确输出不变证书或多源虚拟基；避免缓存容量调优故事。

- **结论：** 需更强执行原语


#### ATT-PACK1 · C1* residual/eager/wake联合包

定位：209–218；235。N/F/H/T：**2 / 1 / 3 / 1**。

- **原机制：** 将方向、残差、遮挡与预测器并列接到C1。

- **最近先验与访问状态：** [Liu et al., An Ultra-High Performance and Scalable Optical Flow Hardware Accelerator Based on FPGA for Autonomous Driving, TCAS-I 2025](https://ieeexplore.ieee.org/document/11030859/)；本轮 IEEE 原始摘要已核：方向预测、可配置 PE、金字塔管线；未取得全文

- **冻结适配：** 多个原算法不存在的反馈边；不是ep34原位实现。

- **真正增量：** 组件堆叠不能自动增加单机制新意。

- **代价与反证：** 成本和误差耦合，TCAS-II 4.5页无法支撑每个新算法。

- **迁移方式：** 拆出一个确实少执行的原语，其余最多底座。

- **结论：** 否决整包贡献叙事


#### ATT-PACK2 · C2* dual-rail/motion/sparse联合包

定位：220–238。N/F/H/T：**1 / 0 / 3 / 0**。

- **原机制：** 实值双轨、运动context、分型稀疏及Mask-Add堆叠。

- **最近先验与访问状态：** [Xu et al., Bishop: Sparsified Bundling Spiking Transformers on Heterogeneous Cores with Error-Constrained Pruning, ISCA 2025](https://arxiv.org/html/2505.12281)；本轮原文 §3/§5.1 已核：TTB、二值 QK 点积界、BSA／ECP-aware 训练均为已有内容

- **冻结适配：** 实值载荷和hyp身份冲突；SDSA与H67混用。

- **真正增量：** 多数是已知路由或标签换名。

- **代价与反证：** 多后端面积/训练/误差未闭，不能相乘收益。

- **迁移方式：** 去掉虚构载荷与hyp，围绕一个真实消费者不变量重写。

- **结论：** 否决整包


#### ATT-M9d · Bishop BSA bundle-sparsity training

定位：80；84。N/F/H/T：**2 / 1 / 5 / 2**。

- **原机制：** 专门训练bundle结构稀疏以使TTB空束更多。

- **最近先验与访问状态：** [Xu et al., Bishop: Sparsified Bundling Spiking Transformers on Heterogeneous Cores with Error-Constrained Pruning, ISCA 2025](https://arxiv.org/html/2505.12281)；本轮原文 §3/§5.1 已核：TTB、二值 QK 点积界、BSA／ECP-aware 训练均为已有内容

- **冻结适配：** 当前权重未训练成这些结构，运行配置不能创造结构稀疏。

- **真正增量：** 层配置与既有 HSS 映射；未改变数学执行内容。

- **代价与反证：** 需要重训；不能把ep34自然稀疏度当BSA结构稀疏，bundle统计需重新捕获。

- **迁移方式：** 仅作为新训练 Pareto 的硬件底座。

- **结论：** 不作冻结主机制


#### ATT-M16b · Performer / Nystrom / Gated DeltaNet options

定位：142。N/F/H/T：**2 / 1 / 4 / 1**。

- **原机制：** 用线性核或近似注意力替换现有算术。

- **最近先验与访问状态：** [FABNet: Adaptable Butterfly Accelerator for Attention-based NNs via Hardware and Algorithm Co-design, MICRO 2022](https://arxiv.org/abs/2209.09570)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** 替换算子和网络，不是ep34映射。

- **真正增量：** 现有蝶形后端与精细后端拼接。

- **代价与反证：** 重训、两后端资源、精度及跨尺度接口。

- **迁移方式：** 新模型方向，当前短文不扩大范围。

- **结论：** 冻结不适配


### research/08_video_temporal_sparsity_accelerators.md

精读完成；SHA256 `10c68b4f544ad471ecc679e2d4570b805263087709017a4ce416337d42d4f9b0`。


#### 08.Idea1 · STH-Gate

定位：L33–44。N/F/H/T：**2 / 1 / 6 / 2**。

- **原机制：** 在线判定 head 更重空间/时间，选择块掩码和数据布局。

- **最近先验与访问状态：** [SparseVideoGen / SparseVideoGen2](https://arxiv.org/abs/2502.01776)；原文件来源；本轮未独立取得全文

- **冻结适配：** 视频 DiT 的 attention 稀疏结构不是 H67 T=2 Motion-XOR 保真条件。

- **真正增量：** head 分类和稀疏 mask 已有；换事件统计尚无新服务原语。

- **代价与反证：** profile/sort 开销和误删分数；动态 BN 及残差仍依赖所有 token。

- **迁移方式：** 可只改变无损执行顺序；真正减少分数需新 AEE 或严谨界。

- **结论：** 原版不作为冻结主机制


#### 08.Idea2 · STA-SU

定位：L48–59。N/F/H/T：**2 / 1 / 6 / 2**。

- **原机制：** 在线稀疏化和混合精度，按时间变化更新计算。

- **最近先验与访问状态：** [FlightVGM, FPGA 2025](https://doi.org/10.1145/3706628.3708864)；原文件来源；本轮未独立取得全文

- **冻结适配：** DiT 稀疏化与 ATLIF 实值假设均不等于冻结二值入口。

- **真正增量：** 常规 sparse update unit 尚未回答动态 BN。

- **代价与反证：** threshold 稀疏化/混精度需另 AEE；不能把 sparse 操作数换名为硬件创新。

- **迁移方式：** 只留下精确位图变化检测作为基本服务。

- **结论：** 原版不作为冻结主机制


#### 08.Idea3 · CW-Reuse

定位：L63–74。N/F/H/T：**3 / 2 / 5 / 2**。

- **原机制：** 利用通道相关/重复 chunk 捕获并复用注意力中间结果。

- **最近先验与访问状态：** [Kaleido（原文件 arXiv2607.13770）](https://arxiv.org/abs/2607.13770)；原文件来源；本轮未独立取得全文

- **冻结适配：** 冻结图未见 RoPE；部分通道相似不证明整分数相同。

- **真正增量：** 精确 chunk 匹配只能是 CSE/部分和的特例。

- **代价与反证：** 比较、字典、累加再装配与索引版本成本；来源未全文核验。

- **迁移方式：** 仅在精确子表达式和合法上下文上保留；与 SumMerge/RSR++ 比。

- **结论：** 原版不作为冻结主机制


#### 08.Idea4 · DF-OS

定位：L78–89。N/F/H/T：**3 / 2 / 6 / 2**。

- **原机制：** 从帧差提取变化，保持旧 feature 并更新输出。

- **最近先验与访问状态：** [VideoTime3, IEEE LSSC 2023](https://doi.org/10.1109/LSSC.2023.3286698)；原文件来源；本轮未独立取得全文

- **冻结适配：** 事件体素并非该数值帧差；无额外 APS 支路；动态 BN 破坏直接差分传播。

- **真正增量：** 已有 delta 思路的应用映射。

- **代价与反证：** 宽历史 feature、第一帧初始化和重置成本；非静态 BN 需全域统计更新。

- **迁移方式：** 仅纯线性叶精确差分；全层需 V3 类型重构。

- **结论：** 原版不作为冻结主机制


#### 08.Idea5 · Δ-MaskPipe

定位：L93–104。N/F/H/T：**3 / 2 / 5 / 3**。

- **原机制：** 按前后帧变化传播稀疏数值差分和非线性残余。

- **最近先验与访问状态：** [DeltaCNN: End-to-End CNN Inference of Sparse Frame Differences in Videos, CVPR 2022](https://arxiv.org/html/2203.03996v2)；一手 HTML 已访问；原文 BN/bias 只在首帧应用段已核

- **冻结适配：** H67 no_running BN 使未变化 token 仍随 μ/σ 改变。

- **真正增量：** 直接迁移无新原语；动态 BN 下的合法消费者提交才有研究问题。

- **代价与反证：** 缓存宽状态、层间失效、截断精度；原文 BN 按固定仿射处理不能替代动态统计。

- **迁移方式：** 收窄到 BN 前线性叶，或改全域区间精化（高风险 V3）。

- **结论：** 原版不作为冻结主机制


#### 08.Idea6 · MW-ΔBuf

定位：L108–119。N/F/H/T：**3 / 1 / 5 / 2**。

- **原机制：** 运动对齐旧 feature，球面缓冲和补边卷积处理新区域。

- **最近先验与访问状态：** [MotionDeltaCNN, Parger et al., ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Parger_MotionDeltaCNN_Sparse_CNN_Inference_of_Frame_Differences_in_Moving_Camera_ICCV_2023_paper.html)；CVF 一手摘要、PDF 方法段已核

- **冻结适配：** H67 没有免费的运动对齐信号；动态 BN 仍是障碍。

- **真正增量：** 环形地址与补边控制已有；光流作为输入/输出不构成电路区别。

- **代价与反证：** 当前输出不能充当前置先验；warp 差错、边界和历史状态均收费。

- **迁移方式：** 只可用因果外部运动先验做新 AEE 路线；精确迁移局限线性叶。

- **结论：** 原版不作为冻结主机制


#### 08.Idea7 · TL-Exit

定位：L123–134。N/F/H/T：**2 / 0 / 5 / 1**。

- **原机制：** 按时间或层置信度提前退出，冻结剩余计算。

- **最近先验与访问状态：** [TLEE, IEEE IoT Journal 2023](https://doi.org/10.1109/JIOT.2023.3293506)；原文件来源；本轮未独立取得全文

- **冻结适配：** 分类早退不能直接替代稠密 flow head；完整 PSN 不是可截断时步。

- **真正增量：** 无当前网络原生的 residual convergence 流程。

- **代价与反证：** 需新增退出头/训练和 AEE；预测已足够不是严格输出证书。

- **迁移方式：** 原版排除；完整 T 贡献界单独研究。

- **结论：** 原版不作为冻结主机制


#### 08.Idea8 · T3D-Reuse

定位：L138–149。N/F/H/T：**1 / 7 / 7 / 1**。

- **原机制：** 时空三维块复用输入/权重，减少重读。

- **最近先验与访问状态：** [Systolic-Cube DAC 2019; Morph MICRO 2018](https://doi.org/10.1145/3316781.3317919)；原文件来源；本轮未独立取得全文

- **冻结适配：** 数据驻留可适配；T 位置相同输入不意味完整 PSN 输出相同。

- **真正增量：** 通用 tiling 已有；没有改变实际求值内容。

- **代价与反证：** 缓冲/halo/时间维栅栏成本；不可把复用容量免费扩大。

- **迁移方式：** 作为数据流强基线，暂不标题化。

- **结论：** 原版不作为冻结主机制


#### 08.Idea9 · EV-Wake

定位：L153–164。N/F/H/T：**2 / 1 / 4 / 1**。

- **原机制：** 低功耗事件/运动前端唤醒后续视觉计算。

- **最近先验与访问状态：** [DVS-CIS always-on fusion, ISCAS 2025（原文件）](https://doi.org/10.1109/ISCAS56072.2025.11043578)；原文件来源；本轮未独立取得全文

- **冻结适配：** 冻结 H67 按帧输出稠密光流；场景静默不证明可省全部输出。

- **真正增量：** AoV 系统理念已存在；本项目没有传感器/SoC全闭环。

- **代价与反证：** 待机/唤醒电源域与漏电/漏检；不能把摄像头节能冒充岛级提升。

- **迁移方式：** 另系统任务才做；当前不新增。

- **结论：** 原版不作为冻结主机制


#### 08.Idea10 · TokMerge-OF

定位：L168–179。N/F/H/T：**2 / 0 / 5 / 2**。

- **原机制：** 合并相似时空 token，再展开或传播输出。

- **最近先验与访问状态：** [Token Merging, ICLR 2023; TESTA; TempMe](https://arxiv.org/abs/2210.09461)；原文件来源；本轮未独立取得全文

- **冻结适配：** 改变冻结 token 几何、Swin 窗口和 BN 样本统计。

- **真正增量：** 通用 token merge；OF 标签不是新增原语。

- **代价与反证：** 边缘光流与遮挡 AEE、unmerge、位置编码及动态统计必须重评。

- **迁移方式：** 只作新模型 Pareto；不能与无损 C2 同表。

- **结论：** 原版不作为冻结主机制


#### 08.Idea11 · TS-Shift

定位：L183–194。N/F/H/T：**1 / 0 / 8 / 1**。

- **原机制：** 以零乘法时间通道移位替代/补充时间卷积。

- **最近先验与访问状态：** [TSM: Temporal Shift Module, ICCV 2019](https://arxiv.org/abs/1811.08383)；原文件来源；本轮未独立取得全文

- **冻结适配：** 移位是置换，不能替代冻结 fullrank T×T A。

- **真正增量：** 既有算法 operator；免费 FLOP 不代表免费数据移动。

- **代价与反证：** 换算子需重训；搬移、重排和边缘填零成本。

- **迁移方式：** 仅原模型含 shift 时映射；当前排除。

- **结论：** 原版不作为冻结主机制


#### 08.Idea12 · SinkSparse

定位：L198–209。N/F/H/T：**2 / 0 / 5 / 1**。

- **原机制：** 代表块/首帧 sink 选择稀疏 attention 支持集合。

- **最近先验与访问状态：** [RainFusion2.0; SPADE; VSA（原文件）](https://arxiv.org/abs/2505.13389)；原文件来源；本轮未独立取得全文

- **冻结适配：** 冻结 Tw=2 没有固定第一帧 sink 保真合同。

- **真正增量：** 已有稀疏注意力启发式重命名。

- **代价与反证：** 选块/采样开销、误删分数和 AEE；近期来源未全文核。

- **迁移方式：** 需要新评价；当前不主打。

- **结论：** 原版不作为冻结主机制


#### 08.landscape.1 · STH-Gate

定位：L19。N/F/H/T：**2 / 1 / 6 / 2**。

- **原机制：** 在线判定 head 更重空间/时间，选择块掩码和数据布局。

- **最近先验与访问状态：** [SparseVideoGen / SparseVideoGen2](https://arxiv.org/abs/2502.01776)；原文件来源；本轮未独立取得全文

- **冻结适配：** 视频 DiT 的 attention 稀疏结构不是 H67 T=2 Motion-XOR 保真条件。

- **真正增量：** head 分类和稀疏 mask 已有；换事件统计尚无新服务原语。

- **代价与反证：** profile/sort 开销和误删分数；动态 BN 及残差仍依赖所有 token。

- **迁移方式：** 可只改变无损执行顺序；真正减少分数需新 AEE 或严谨界。

- **结论：** 原版不作为冻结主机制


#### 08.landscape.2 · Δ-MaskPipe

定位：L20。N/F/H/T：**3 / 2 / 5 / 3**。

- **原机制：** 按前后帧变化传播稀疏数值差分和非线性残余。

- **最近先验与访问状态：** [DeltaCNN: End-to-End CNN Inference of Sparse Frame Differences in Videos, CVPR 2022](https://arxiv.org/html/2203.03996v2)；一手 HTML 已访问；原文 BN/bias 只在首帧应用段已核

- **冻结适配：** H67 no_running BN 使未变化 token 仍随 μ/σ 改变。

- **真正增量：** 直接迁移无新原语；动态 BN 下的合法消费者提交才有研究问题。

- **代价与反证：** 缓存宽状态、层间失效、截断精度；原文 BN 按固定仿射处理不能替代动态统计。

- **迁移方式：** 收窄到 BN 前线性叶，或改全域区间精化（高风险 V3）。

- **结论：** 原版不作为冻结主机制


#### 08.landscape.3 · TL-Exit

定位：L21。N/F/H/T：**2 / 0 / 5 / 1**。

- **原机制：** 按时间或层置信度提前退出，冻结剩余计算。

- **最近先验与访问状态：** [TLEE, IEEE IoT Journal 2023](https://doi.org/10.1109/JIOT.2023.3293506)；原文件来源；本轮未独立取得全文

- **冻结适配：** 分类早退不能直接替代稠密 flow head；完整 PSN 不是可截断时步。

- **真正增量：** 无当前网络原生的 residual convergence 流程。

- **代价与反证：** 需新增退出头/训练和 AEE；预测已足够不是严格输出证书。

- **迁移方式：** 原版排除；完整 T 贡献界单独研究。

- **结论：** 原版不作为冻结主机制


#### 08.landscape.4 · MW-ΔBuf

定位：L22。N/F/H/T：**3 / 1 / 5 / 2**。

- **原机制：** 运动对齐旧 feature，球面缓冲和补边卷积处理新区域。

- **最近先验与访问状态：** [MotionDeltaCNN, Parger et al., ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Parger_MotionDeltaCNN_Sparse_CNN_Inference_of_Frame_Differences_in_Moving_Camera_ICCV_2023_paper.html)；CVF 一手摘要、PDF 方法段已核

- **冻结适配：** H67 没有免费的运动对齐信号；动态 BN 仍是障碍。

- **真正增量：** 环形地址与补边控制已有；光流作为输入/输出不构成电路区别。

- **代价与反证：** 当前输出不能充当前置先验；warp 差错、边界和历史状态均收费。

- **迁移方式：** 只可用因果外部运动先验做新 AEE 路线；精确迁移局限线性叶。

- **结论：** 原版不作为冻结主机制


#### 08.landscape.5 · T3D-Reuse

定位：L23。N/F/H/T：**1 / 7 / 7 / 1**。

- **原机制：** 时空三维块复用输入/权重，减少重读。

- **最近先验与访问状态：** [Systolic-Cube DAC 2019; Morph MICRO 2018](https://doi.org/10.1145/3316781.3317919)；原文件来源；本轮未独立取得全文

- **冻结适配：** 数据驻留可适配；T 位置相同输入不意味完整 PSN 输出相同。

- **真正增量：** 通用 tiling 已有；没有改变实际求值内容。

- **代价与反证：** 缓冲/halo/时间维栅栏成本；不可把复用容量免费扩大。

- **迁移方式：** 作为数据流强基线，暂不标题化。

- **结论：** 原版不作为冻结主机制


#### 08.landscape.6 · EV-Wake

定位：L24。N/F/H/T：**2 / 1 / 4 / 1**。

- **原机制：** 低功耗事件/运动前端唤醒后续视觉计算。

- **最近先验与访问状态：** [DVS-CIS always-on fusion, ISCAS 2025（原文件）](https://doi.org/10.1109/ISCAS56072.2025.11043578)；原文件来源；本轮未独立取得全文

- **冻结适配：** 冻结 H67 按帧输出稠密光流；场景静默不证明可省全部输出。

- **真正增量：** AoV 系统理念已存在；本项目没有传感器/SoC全闭环。

- **代价与反证：** 待机/唤醒电源域与漏电/漏检；不能把摄像头节能冒充岛级提升。

- **迁移方式：** 另系统任务才做；当前不新增。

- **结论：** 原版不作为冻结主机制


#### 08.landscape.7 · TokMerge-OF

定位：L25。N/F/H/T：**2 / 0 / 5 / 2**。

- **原机制：** 合并相似时空 token，再展开或传播输出。

- **最近先验与访问状态：** [Token Merging, ICLR 2023; TESTA; TempMe](https://arxiv.org/abs/2210.09461)；原文件来源；本轮未独立取得全文

- **冻结适配：** 改变冻结 token 几何、Swin 窗口和 BN 样本统计。

- **真正增量：** 通用 token merge；OF 标签不是新增原语。

- **代价与反证：** 边缘光流与遮挡 AEE、unmerge、位置编码及动态统计必须重评。

- **迁移方式：** 只作新模型 Pareto；不能与无损 C2 同表。

- **结论：** 原版不作为冻结主机制


#### 08.upgrade.1 · OP-STW / DPP-Skip

定位：L217。N/F/H/T：**3 / 1 / 5 / 2**。

- **原机制：** 用上一时刻 coarse flow/差分和事件计数预测 tile wake，睡眠 tile 复用输出。

- **最近先验与访问状态：** [MotionDeltaCNN ICCV 2023; motion-gated vision SoCs](https://openaccess.thecvf.com/content/ICCV2023/html/Parger_MotionDeltaCNN_Sparse_CNN_Inference_of_Frame_Differences_in_Moving_Camera_ICCV_2023_paper.html)；CVF 一手摘要和 PDF 方法段已读

- **冻结适配：** H67 无现成 coarse-flow 迭代支路；no_running 动态 BN 的全域均值/方差使局部不变不能直接推出消费者不变。

- **真正增量：** 加 motion 名称不足；若以严格消费者证书取代启发式 wake 才形成新执行问题。

- **代价与反证：** 当前帧 flow 不能提前免费获得；睡眠输出未必精确，必须计预测器、warp、缓存与 AEE。

- **迁移方式：** 原版只可有损新评价；精确重构必须全域 BN 依赖闭合。

- **结论：** 原版不作为冻结主线


#### 08.upgrade.2 · PRRC

定位：L218。N/F/H/T：**2 / 0 / 6 / 1**。

- **原机制：** 粗到细金字塔预算/ROI，预算耗尽后复用或传播填充。

- **最近先验与访问状态：** [MotionDeltaCNN; optical-flow coarse-to-fine methods](https://openaccess.thecvf.com/content/ICCV2023/html/Parger_MotionDeltaCNN_Sparse_CNN_Inference_of_Frame_Differences_in_Moving_Camera_ICCV_2023_paper.html)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** 冻结 U-Net 非迭代金字塔搜索；不含预算耗尽语义。

- **真正增量：** 计数器与 ROI 不是新原语；须有算法可证明的停止条件。

- **代价与反证：** 强制 REUSE/PROP 本质近似，打 overflow 标志不能使输出精确。

- **迁移方式：** 不套 ep34；新算法需训练/valid825。

- **结论：** 原版否决


#### 08.upgrade.3 · OGEC

定位：L219。N/F/H/T：**3 / 0 / 5 / 1**。

- **原机制：** 以前后向一致性判遮挡，匹配路径精算，未匹配邻域传播。

- **最近先验与访问状态：** [MotionDeltaCNN; occlusion-aware optical-flow algorithms](https://openaccess.thecvf.com/content/ICCV2023/html/Parger_MotionDeltaCNN_Sparse_CNN_Inference_of_Frame_Differences_in_Moving_Camera_ICCV_2023_paper.html)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** 冻结图无免费反向 flow 或邻域填充替代神经层。

- **真正增量：** 稀疏门控加光流谓词；尚无独特正确性原语。

- **代价与反证：** 遮挡处往往最难，传播可能损伤 AEE；额外反向推理/检测必须收费。

- **迁移方式：** 只作有损新评价，不叫 exact path 加速。

- **结论：** 原版否决


#### 08.upgrade.4 · HBG-RP

定位：L220。N/F/H/T：**1 / 0 / 9 / 1**。

- **原机制：** 把幅值拆为非零门控 g 与 int8 实值 p，g 控制时钟/请求。

- **最近先验与访问状态：** [Activity Pruning for Efficient Spiking Neural Networks, Bu/Shi/Yu, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/fa12d67b5939c37ea8a4659c31a08d2c-Paper-Conference.pdf)；一手 PDF §3.2 式10–11及阈值吸收段已核；OpenReview 被拦后由 NeurIPS 正式库取得

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。

- **真正增量：** 冻结身份下退化为已有 active 位与权重加法；没有新增幅值原语。

- **代价与反证：** eps=1 严格 > 会删 ±1；量化/截断与门控均需新 AEE。常量 θ 若≤eps 甚至整层静默；最小负数 abs 必须9位。

- **迁移方式：** 冻结版本删除 p，只作为普通二值协议/门控基线；实值版本需重训并另建身份。

- **结论：** 否决为主机制；禁止套 ep34


#### 08.upgrade.5 · ADP-MAC

定位：L221。N/F/H/T：**2 / 2 / 6 / 2**。

- **原机制：** 激活与权重逐位跳零，空闲 slot donation。

- **最近先验与访问状态：** [BitFusion MICRO 2018; FireFly-v2](https://arxiv.org/abs/1712.01507)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。 激活只有1位，双侧幅值位稀疏前提消失。

- **真正增量：** bit-serial/slot donation 已有；不能重新命名为光流原语。

- **代价与反证：** 零位译码、移位、交换网、符号扩展与最坏精度时间都收费。

- **迁移方式：** 只保留权重位串行作为同资源对照；无新意暂不重做。

- **结论：** 不作主机制


#### 08.upgrade.6 · ARM-Acc

定位：L222。N/F/H/T：**3 / 0 / 5 / 1**。

- **原机制：** K 个运动方向假设独立累加，按证据赢家提交。

- **最近先验与访问状态：** Multi-hypothesis optical-flow algorithms; generic speculative accumulation；原文件给出来源；本轮未独立取得全文

- **冻结适配：** 冻结 H67 没有 K=4/8 搜索假设或赢家提交算子。

- **真正增量：** 4 个 consumer 是独立 token，不可改称竞争假设。

- **代价与反证：** 新增假设计算、状态与选择改变模型；不能将输家丢弃说成精确。

- **迁移方式：** 先有新算法与 AEE 才可考虑；当前剔除。

- **结论：** 冻结下否决


#### 08.upgrade.7 · MFBD

定位：L223。N/F/H/T：**1 / 6 / 7 / 1**。

- **原机制：** 同权重行向相邻时间/运动上下文广播。

- **最近先验与访问状态：** [ELSA ISCA 2026; Eyeriss; existing TSBG](https://arxiv.org/html/2605.20802v1)；ELSA 一手 HTML 相关 dataflow 已读

- **冻结适配：** 若每 consumer 保持真实 token/时步身份可合法。

- **真正增量：** 现有 TSBG 已在 mem_req 前按 source-group 共享取权，剩余只是分组选择。

- **代价与反证：** 不能重复计权重请求收益；运动假设不存在，广播网络与上下文容量不能免费增加。

- **迁移方式：** 必须改变实际计算内容才重立项；现版降为 TSBG 对照。

- **结论：** 仅既有机制换名


#### 08.upgrade.8 · SP-Gate

定位：L224。N/F/H/T：**2 / 1 / 6 / 2**。

- **原机制：** 按 attention mass 选择 hot token 并抑制 cold token 的后续 FC 计算。

- **最近先验与访问状态：** [SparseVideoGen; token pruning / early-exit](https://arxiv.org/abs/2502.01776)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** 冻结 attention 的 gate 不是 token 整体无贡献证明；FC/BN/残差仍有消费者。

- **真正增量：** 普通分数剪枝；没有精确停止证书。

- **代价与反证：** 剪 token 改动态 BN 统计及稠密光流输出；分数读取/排序成本。

- **迁移方式：** 仅新 AEE Pareto；精确版必须完整依赖界。

- **结论：** 冻结无损路线否决


#### 08.upgrade.9 · EESUC

定位：L225。N/F/H/T：**2 / 0 / 5 / 1**。

- **原机制：** 按时间或层置信度提前退出，冻结剩余计算。

- **最近先验与访问状态：** [TLEE, IEEE IoT Journal 2023](https://doi.org/10.1109/JIOT.2023.3293506)；原文件来源；本轮未独立取得全文

- **冻结适配：** 分类早退不能直接替代稠密 flow head；完整 PSN 不是可截断时步。

- **真正增量：** 无当前网络原生的 residual convergence 流程。

- **代价与反证：** 需新增退出头/训练和 AEE；预测已足够不是严格输出证书。

- **迁移方式：** 原版排除；完整 T 贡献界单独研究。

- **结论：** 原版不作为冻结主机制


#### 08.upgrade.10 · VGTS

定位：L226。N/F/H/T：**3 / 0 / 3 / 1**。

- **原机制：** 根据幅值与漏电预测未来若干时间步不发放并跳过更新。

- **最近先验与访问状态：** [SATO; SnaPEA ISCA 2018](https://cseweb.ucsd.edu/~vakhlagh/ISCA18-SnaPEA.pdf)；SnaPEA 一手 PDF 已读；精确模式与预测模式分开

- **冻结适配：** 冻结 PSN 为完整 T×T 线性映射（T=2/10），不是按漏电、复位递推的 LIF。 冻结没有此因果漏电过程。

- **真正增量：** 改为完整 T 向量的严格剩余贡献界才合法；那是另一机制。

- **代价与反证：** 未来输入可能通过负/正 A 翻转输出，当前无脉冲不证明未来静默。

- **迁移方式：** 原版否决；严格证书只作独立研究，且先解决动态 BN。

- **结论：** 原版否决


#### 08.upgrade.11 · ReMem-Tok

定位：L227。N/F/H/T：**3 / 2 / 5 / 2**。

- **原机制：** 跨窗口或帧缓存膜/中间结果，按相似输入复用。

- **最近先验与访问状态：** [DeltaCNN CVPR 2022](https://arxiv.org/html/2203.03996v2)；一手 HTML 已访问；动态 BN 反证来自本地合同

- **冻结适配：** no_running 动态 BN 的全域均值/方差使局部不变不能直接推出消费者不变。 无 RAFT 循环状态；窗口重叠不等于同一上下文。

- **真正增量：** 只有精确输入、权重和归一化身份同时匹配才可复用。

- **代价与反证：** 锚点宽张量、版本、地址及比较流量；PSN 不等于 LIF 残留膜。

- **迁移方式：** 收窄到 BN 前纯线性叶，或改全域统计证书；原缓存不直接迁移。

- **结论：** 需大改


#### 08.C1.bundle.1 · MW-ΔBuf

定位：L236。N/F/H/T：**3 / 1 / 5 / 2**。

- **原机制：** 运动对齐旧 feature，球面缓冲和补边卷积处理新区域。

- **最近先验与访问状态：** [MotionDeltaCNN, Parger et al., ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Parger_MotionDeltaCNN_Sparse_CNN_Inference_of_Frame_Differences_in_Moving_Camera_ICCV_2023_paper.html)；CVF 一手摘要、PDF 方法段已核

- **冻结适配：** H67 没有免费的运动对齐信号；动态 BN 仍是障碍。

- **真正增量：** 环形地址与补边控制已有；光流作为输入/输出不构成电路区别。

- **代价与反证：** 当前输出不能充当前置先验；warp 差错、边界和历史状态均收费。

- **迁移方式：** 只可用因果外部运动先验做新 AEE 路线；精确迁移局限线性叶。

- **结论：** 原版不作为冻结主机制


#### 08.C1.bundle.2 · Δ-MaskPipe / DF-OS

定位：L237。N/F/H/T：**3 / 2 / 5 / 3**。

- **原机制：** 按前后帧变化传播稀疏数值差分和非线性残余。

- **最近先验与访问状态：** [DeltaCNN: End-to-End CNN Inference of Sparse Frame Differences in Videos, CVPR 2022](https://arxiv.org/html/2203.03996v2)；一手 HTML 已访问；原文 BN/bias 只在首帧应用段已核

- **冻结适配：** H67 no_running BN 使未变化 token 仍随 μ/σ 改变。

- **真正增量：** 直接迁移无新原语；动态 BN 下的合法消费者提交才有研究问题。

- **代价与反证：** 缓存宽状态、层间失效、截断精度；原文 BN 按固定仿射处理不能替代动态统计。

- **迁移方式：** 收窄到 BN 前线性叶，或改全域区间精化（高风险 V3）。

- **结论：** 原版不作为冻结主机制


#### 08.C1.bundle.3 · EV-Wake

定位：L238。N/F/H/T：**2 / 1 / 4 / 1**。

- **原机制：** 低功耗事件/运动前端唤醒后续视觉计算。

- **最近先验与访问状态：** [DVS-CIS always-on fusion, ISCAS 2025（原文件）](https://doi.org/10.1109/ISCAS56072.2025.11043578)；原文件来源；本轮未独立取得全文

- **冻结适配：** 冻结 H67 按帧输出稠密光流；场景静默不证明可省全部输出。

- **真正增量：** AoV 系统理念已存在；本项目没有传感器/SoC全闭环。

- **代价与反证：** 待机/唤醒电源域与漏电/漏检；不能把摄像头节能冒充岛级提升。

- **迁移方式：** 另系统任务才做；当前不新增。

- **结论：** 原版不作为冻结主机制


#### 08.C1.bundle.4 · OP-STW / PRRC / OGEC

定位：L239。N/F/H/T：**3 / 1 / 5 / 2**。

- **原机制：** 用上一时刻 coarse flow/差分和事件计数预测 tile wake，睡眠 tile 复用输出。

- **最近先验与访问状态：** [MotionDeltaCNN ICCV 2023; motion-gated vision SoCs](https://openaccess.thecvf.com/content/ICCV2023/html/Parger_MotionDeltaCNN_Sparse_CNN_Inference_of_Frame_Differences_in_Moving_Camera_ICCV_2023_paper.html)；CVF 一手摘要和 PDF 方法段已读

- **冻结适配：** H67 无现成 coarse-flow 迭代支路；no_running 动态 BN 的全域均值/方差使局部不变不能直接推出消费者不变。

- **真正增量：** 加 motion 名称不足；若以严格消费者证书取代启发式 wake 才形成新执行问题。

- **代价与反证：** 当前帧 flow 不能提前免费获得；睡眠输出未必精确，必须计预测器、warp、缓存与 AEE。

- **迁移方式：** 原版只可有损新评价；精确重构必须全域 BN 依赖闭合。

- **结论：** 原版不作为冻结主线


#### 08.C2.bundle.1 · STH-Gate

定位：L244。N/F/H/T：**2 / 1 / 6 / 2**。

- **原机制：** 在线判定 head 更重空间/时间，选择块掩码和数据布局。

- **最近先验与访问状态：** [SparseVideoGen / SparseVideoGen2](https://arxiv.org/abs/2502.01776)；原文件来源；本轮未独立取得全文

- **冻结适配：** 视频 DiT 的 attention 稀疏结构不是 H67 T=2 Motion-XOR 保真条件。

- **真正增量：** head 分类和稀疏 mask 已有；换事件统计尚无新服务原语。 多模块组合没有额外可证明机制。

- **代价与反证：** profile/sort 开销和误删分数；动态 BN 及残差仍依赖所有 token。

- **迁移方式：** 可只改变无损执行顺序；真正减少分数需新 AEE 或严谨界。

- **结论：** 原版不作为冻结主机制


#### 08.C2.bundle.2 · SinkSparse / CW-Reuse

定位：L245。N/F/H/T：**2 / 0 / 5 / 1**。

- **原机制：** 代表块/首帧 sink 选择稀疏 attention 支持集合。

- **最近先验与访问状态：** [RainFusion2.0; SPADE; VSA（原文件）](https://arxiv.org/abs/2505.13389)；原文件来源；本轮未独立取得全文

- **冻结适配：** 冻结 Tw=2 没有固定第一帧 sink 保真合同。

- **真正增量：** 已有稀疏注意力启发式重命名。 多模块组合没有额外可证明机制。

- **代价与反证：** 选块/采样开销、误删分数和 AEE；近期来源未全文核。

- **迁移方式：** 需要新评价；当前不主打。

- **结论：** 原版不作为冻结主机制


#### 08.C2.bundle.3 · TokMerge / TS-Shift / T3D-Reuse

定位：L246。N/F/H/T：**2 / 0 / 5 / 2**。

- **原机制：** 合并相似时空 token，再展开或传播输出。

- **最近先验与访问状态：** [Token Merging, ICLR 2023; TESTA; TempMe](https://arxiv.org/abs/2210.09461)；原文件来源；本轮未独立取得全文

- **冻结适配：** 改变冻结 token 几何、Swin 窗口和 BN 样本统计。

- **真正增量：** 通用 token merge；OF 标签不是新增原语。 多模块组合没有额外可证明机制。

- **代价与反证：** 边缘光流与遮挡 AEE、unmerge、位置编码及动态统计必须重评。

- **迁移方式：** 只作新模型 Pareto；不能与无损 C2 同表。

- **结论：** 原版不作为冻结主机制


#### 08.C2.bundle.4 · HBG / ADP / ARM / MFBD / SP-Gate

定位：L247。N/F/H/T：**1 / 0 / 9 / 1**。

- **原机制：** 把幅值拆为非零门控 g 与 int8 实值 p，g 控制时钟/请求。

- **最近先验与访问状态：** [Activity Pruning for Efficient Spiking Neural Networks, Bu/Shi/Yu, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/fa12d67b5939c37ea8a4659c31a08d2c-Paper-Conference.pdf)；一手 PDF §3.2 式10–11及阈值吸收段已核；OpenReview 被拦后由 NeurIPS 正式库取得

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。

- **真正增量：** 冻结身份下退化为已有 active 位与权重加法；没有新增幅值原语。 多模块组合没有额外可证明机制。

- **代价与反证：** eps=1 严格 > 会删 ±1；量化/截断与门控均需新 AEE。常量 θ 若≤eps 甚至整层静默；最小负数 abs 必须9位。

- **迁移方式：** 冻结版本删除 p，只作为普通二值协议/门控基线；实值版本需重训并另建身份。

- **结论：** 否决为主机制；禁止套 ep34


#### 08.firstHW.shortlist.1 · STH-Gate

定位：L257。N/F/H/T：**2 / 1 / 6 / 2**。

- **原机制：** 在线判定 head 更重空间/时间，选择块掩码和数据布局。

- **最近先验与访问状态：** [SparseVideoGen / SparseVideoGen2](https://arxiv.org/abs/2502.01776)；原文件来源；本轮未独立取得全文

- **冻结适配：** 视频 DiT 的 attention 稀疏结构不是 H67 T=2 Motion-XOR 保真条件。

- **真正增量：** head 分类和稀疏 mask 已有；换事件统计尚无新服务原语。

- **代价与反证：** profile/sort 开销和误删分数；动态 BN 及残差仍依赖所有 token。

- **迁移方式：** 可只改变无损执行顺序；真正减少分数需新 AEE 或严谨界。

- **结论：** 原版不作为冻结主机制


#### 08.firstHW.shortlist.2 · MW-ΔBuf

定位：L258。N/F/H/T：**3 / 1 / 5 / 2**。

- **原机制：** 运动对齐旧 feature，球面缓冲和补边卷积处理新区域。

- **最近先验与访问状态：** [MotionDeltaCNN, Parger et al., ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Parger_MotionDeltaCNN_Sparse_CNN_Inference_of_Frame_Differences_in_Moving_Camera_ICCV_2023_paper.html)；CVF 一手摘要、PDF 方法段已核

- **冻结适配：** H67 没有免费的运动对齐信号；动态 BN 仍是障碍。

- **真正增量：** 环形地址与补边控制已有；光流作为输入/输出不构成电路区别。

- **代价与反证：** 当前输出不能充当前置先验；warp 差错、边界和历史状态均收费。

- **迁移方式：** 只可用因果外部运动先验做新 AEE 路线；精确迁移局限线性叶。

- **结论：** 原版不作为冻结主机制


#### 08.firstHW.shortlist.3 · TokMerge-OF

定位：L259。N/F/H/T：**2 / 0 / 5 / 2**。

- **原机制：** 合并相似时空 token，再展开或传播输出。

- **最近先验与访问状态：** [Token Merging, ICLR 2023; TESTA; TempMe](https://arxiv.org/abs/2210.09461)；原文件来源；本轮未独立取得全文

- **冻结适配：** 改变冻结 token 几何、Swin 窗口和 BN 样本统计。

- **真正增量：** 通用 token merge；OF 标签不是新增原语。

- **代价与反证：** 边缘光流与遮挡 AEE、unmerge、位置编码及动态统计必须重评。

- **迁移方式：** 只作新模型 Pareto；不能与无损 C2 同表。

- **结论：** 原版不作为冻结主机制


#### 08.firstHW.shortlist.4 · SinkSparse

定位：L260。N/F/H/T：**2 / 0 / 5 / 1**。

- **原机制：** 代表块/首帧 sink 选择稀疏 attention 支持集合。

- **最近先验与访问状态：** [RainFusion2.0; SPADE; VSA（原文件）](https://arxiv.org/abs/2505.13389)；原文件来源；本轮未独立取得全文

- **冻结适配：** 冻结 Tw=2 没有固定第一帧 sink 保真合同。

- **真正增量：** 已有稀疏注意力启发式重命名。

- **代价与反证：** 选块/采样开销、误删分数和 AEE；近期来源未全文核。

- **迁移方式：** 需要新评价；当前不主打。

- **结论：** 原版不作为冻结主机制


#### 08.firstHW.shortlist.5 · CW-Reuse

定位：L261。N/F/H/T：**3 / 2 / 5 / 2**。

- **原机制：** 利用通道相关/重复 chunk 捕获并复用注意力中间结果。

- **最近先验与访问状态：** [Kaleido（原文件 arXiv2607.13770）](https://arxiv.org/abs/2607.13770)；原文件来源；本轮未独立取得全文

- **冻结适配：** 冻结图未见 RoPE；部分通道相似不证明整分数相同。

- **真正增量：** 精确 chunk 匹配只能是 CSE/部分和的特例。

- **代价与反证：** 比较、字典、累加再装配与索引版本成本；来源未全文核验。

- **迁移方式：** 仅在精确子表达式和合法上下文上保留；与 SumMerge/RSR++ 比。

- **结论：** 原版不作为冻结主机制


### research/09_ROUND2_SYNTHESIS_C1_C2_UPGRADE.md

精读完成；SHA256 `2ec4aa7e27d3871611fad81c414da7e050ac77c68f6ac783619ca98f48c65401`。


#### R2-OP · OP-STW / DPP-Skip

定位：21；76；103。N/F/H/T：**3 / 1 / 5 / 2**。

- **原机制：** 方向或流残差决定 tile 是否执行。

- **最近先验与访问状态：** [Liu et al., An Ultra-High Performance and Scalable Optical Flow Hardware Accelerator Based on FPGA for Autonomous Driving, TCAS-I 2025](https://ieeexplore.ieee.org/document/11030859/)；本轮 IEEE 原始摘要已核：方向预测、可配置 PE、金字塔管线；未取得全文

- **冻结适配：** 低；ep34 无前置当前 flow 预测器，跳过非零 tile 改变函数。

- **真正增量：** 增加任务相关门控条件；仅阈值比较未构成执行原语。

- **代价与反证：** 当前 flow 用于跳过产生它的推理存在因果环；漏唤醒、边界依赖、旧结果读写未收费。

- **迁移方式：** 改成来自前一帧或便宜独立前级的因果预测；有损另开 AEE，精确版需要输出不变证明。

- **结论：** 原案不能入冻结主线；可作为算法重构入口


#### R2-PRRC · PRRC / pyramid residual capture

定位：22。N/F/H/T：**2 / 1 / 4 / 2**。

- **原机制：** 粗层写流残差，细层只算 warp 对齐 ROI。

- **最近先验与访问状态：** [Ling et al., FlowAcc, DATE 2022](https://doi.org/10.23919/DATE54114.2022.9774506)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** 分层 U-Net 不等于粗流驱动的迭代残差算法。

- **真正增量：** 新增粗流反馈与 ROI 算法，而非改现有存储政策。

- **代价与反证：** 原前向无所需控制边；ROI halo、warp、残差存储和恢复成本。

- **迁移方式：** 若坚持粗细残差需明确新 forward 和训练；不能复用 ep34 无损数字。

- **结论：** 原冻结适配失败


#### R2-OGEC · OGEC / occlusion-gated exact capture

定位：23；80。N/F/H/T：**3 / 1 / 4 / 3**。

- **原机制：** 遮挡/未匹配 token 转廉价传播通路，匹配走精确。

- **最近先验与访问状态：** [Xu et al., GMFlow: Learning Optical Flow via Global Matching, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_GMFlow_Learning_Optical_Flow_via_Global_Matching_CVPR_2022_paper.pdf)；本轮 CVF 原始摘要／引言已核；全局匹配和传播属于其算法

- **冻结适配：** ep34 没有该传播分支；遮挡不是可删的无效像素。

- **真正增量：** 选路+新增算法分支；GMFlow 已有传播。

- **代价与反证：** 双向一致性可能需额外推理；遮挡正是光流难例，邻居填充没有等价证明。

- **迁移方式：** 独立 AEE 分支；若只预取保留全部执行则创新很薄。

- **结论：** 需算法重构，不能打 exact 标签


#### R2-ECP · ECP-QKV / FACT before projection

定位：24；70；103/105。N/F/H/T：**4 / 3 / 5 / 4**。

- **原机制：** 投影前以便宜特征预测相关性并删 Q/K/V 投影。

- **最近先验与访问状态：** [Qin et al., FACT: FFN-Attention Co-optimized Transformer Architecture with Eager Correlation Prediction, ISCA 2023](https://doi.org/10.1145/3579371.3589057)；本轮 Crossref 出版者题名／作者／日期已核；ACM 403、Crossref 无摘要，具体算法未全文核

- **冻结适配：** K兼V，没有独立V；投影前无 sn_q/sn_k，也无实值ATLIF直方图。

- **真正增量：** “提前于昂贵投影”是有价值执行位置；原案只是换预测特征。

- **代价与反证：** 模型预测代价/误判；K被attention与value共同消费，裁K可能联动两路。

- **迁移方式：** 重构为二值阈值投影结果的精确确定证书，需从当前输入与静态W推导，不用未来flow。

- **结论：** 保留执行位置；FACT全文未核，不能高分定案


#### R2-MW · MW-DeltaBuf

定位：25；71；103。N/F/H/T：**3 / 4 / 5 / 4**。

- **原机制：** 翘曲旧特征/膜后只算变化并保持参考。

- **最近先验与访问状态：** [Parger et al., MotionDeltaCNN, ICCV 2023](https://arxiv.org/html/2210.09887)；本轮原文 §3.1–3.2／讨论已核：非线性状态、更新扩散、截断累计与对齐是必要成本

- **冻结适配：** 任意warp与卷积、位置偏置、PSN不交换；旧膜不能自由跨帧。

- **真正增量：** MotionDeltaCNN 已处理对齐后的稀疏增量；事件/神经元边界尚需新合同。

- **代价与反证：** 非线性全状态、截断累计、halo扩散、重置、warp成本；原文并非简单零成本warp。

- **迁移方式：** 只保留exact二值差分及依赖闭包；先定义哪些层可线性增量、何处必须完整重算。

- **结论：** 可迁移重构，原拼接案不成立


#### R2-DELTA · Delta-MaskPipe / DF-OS

定位：26。N/F/H/T：**3 / 4 / 5 / 4**。

- **原机制：** 差分mask逐层传播，OS部分和更新。

- **最近先验与访问状态：** [Parger et al., DeltaCNN, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Parger_DeltaCNN_End-to-End_CNN_Inference_of_Sparse_Frame_Differences_in_Videos_CVPR_2022_paper.html)；机制通过 MotionDeltaCNN 原文背景核；DeltaCNN 本文未全文核

- **冻结适配：** 线性conv可增量；阈值/PSN/注意力要状态与边界证明。

- **真正增量：** 旧增量卷积架构迁移，尚未给神经元原生新处理。

- **代价与反证：** 更新会经3x3迅速扩散；OS标签不消除此代价。

- **迁移方式：** 候选为带精确发放证书的差分执行，而非换缓存布局。

- **结论：** 保留执行内容重构


#### R2-EV · EV-Wake

定位：27；80。N/F/H/T：**2 / 3 / 5 / 2**。

- **原机制：** 事件/体素能量超过阈值才走exact路径。

- **最近先验与访问状态：** [Stumpp et al., hARMS: A Hardware Acceleration Architecture for Real-Time Event-Based Optical Flow, IEEE Access 2022](https://arxiv.org/html/2112.06772)；原文可访问；本轮概要核读，未逐条核实现数字

- **冻结适配：** 输入无事件不代表网络内部零输出或状态无需服务。

- **真正增量：** 普通输入活动门控。

- **代价与反证：** 卷积halo、bias、PSN时混、门控分母都可能影响输出。

- **迁移方式：** 只保留有依赖闭包的精确无变化检测。

- **结论：** 原gate不足，作基线


#### R2-HEAT · HeatFlow-Tok

定位：28；81。N/F/H/T：**3 / 2 / 5 / 3**。

- **原机制：** 逐层根据事件/残差剪并合并token。

- **最近先验与访问状态：** [HeatViT: Hardware-Efficient Adaptive Token Pruning for Vision Transformers, HPCA 2023](https://arxiv.org/abs/2211.08110)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** 密集flowhead和skip需要完整空间网格。

- **真正增量：** 既有selector换OF特征。

- **代价与反证：** 复原被删位置与位置信息、选择器训练和成本。

- **迁移方式：** 若有损另开AEE；精确版仅安全零区域可删。

- **结论：** 算法候选，非现有硬件创新


#### R2-HBG · HBG-RP / gate＋real payload

定位：41；72；104。N/F/H/T：**1 / 0 / 6 / 0**。

- **原机制：** 独立二值门控制时钟/访存，幅度载荷做 MAC。

- **最近先验与访问状态：** AT-LIF, NeurIPS 2025（原 idea 的二值阈值发放对照）；官方题名链接本轮未核；二值 {0,theta}、静态阈值可吸收由用户冻结合同独立确定

- **冻结适配：** 与冻结 {0,theta} 直接冲突；theta静态，天然非零不是int8载荷。

- **真正增量：** 无法吸收的载荷是虚构的前提，门控本身已有。

- **代价与反证：** 增加幅度存储和乘法；负号协议不代表实值 ATLIF。

- **迁移方式：** 真实多比特 gate×weight 可另定名字和接口；不能偷换 ATLIF 出口。

- **结论：** 冻结主线淘汰


#### R2-ADP · ADP-MAC / 双边 bit particle

定位：42；79。N/F/H/T：**2 / 0 / 5 / 1**。

- **原机制：** 权重和 ATLIF 幅度双边 bit-skip、slot donation。

- **最近先验与访问状态：** [BBS: Bi-directional Bit-level Sparsity for Deep Learning Acceleration, MICRO 2024](https://arxiv.org/abs/2409.05227)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** 不适配；ATLIF 出口为二值，静态 theta 可折权重。

- **真正增量：** 多比特幅度数据通路是假定新增，不是已有算法独特性。

- **代价与反证：** 增加 bit-serial 延迟和粒度调度；自然源侧没有第二组幅度位可跳。

- **迁移方式：** 仅迁到 gate×weight 的真实多比特门控路径且重定合同；不能用 ATLIF 名义。

- **结论：** 淘汰 ATLIF 主线版本


#### R2-ARM · ARM-Acc / motion hypotheses

定位：43；77。N/F/H/T：**2 / 0 / 4 / 1**。

- **原机制：** Acc 槽存多方向假设并由证据选赢家。

- **最近先验与访问状态：** [Stumpp et al., hARMS: A Hardware Acceleration Architecture for Real-Time Event-Based Optical Flow, IEEE Access 2022](https://arxiv.org/html/2112.06772)；原文可访问；本轮概要核读，未逐条核实现数字

- **冻结适配：** ep34 通道/head 无显式运动假设和赢家提交语义。

- **真正增量：** 仅给 Acc 槽改标签不改执行；真正迁移需新算子。

- **代价与反证：** K 假设会增加计算/状态；hARMS 多尺度估计不等同 Transformer head。

- **迁移方式：** 只有明确定义并训练 hypothesis tensor 后再考虑共享求证算子。

- **结论：** 原案为语义换名，淘汰


#### R2-MFBD · MFBD / motion-feature bundles

定位：44；78。N/F/H/T：**2 / 3 / 5 / 2**。

- **原机制：** 时间相邻运动状态束内共享权重并广播。

- **最近先验与访问状态：** [Shi et al., VideoFlow: Exploiting Temporal Cues for Multi-frame Optical Flow Estimation, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Shi_VideoFlow_Exploiting_Temporal_Cues_for_Multi-frame_Optical_Flow_Estimation_ICCV_2023_paper.html)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** ep34 无 TROF/MOP 迭代状态；普通 token×time 可打包。

- **真正增量：** 同权重广播本质未改，运动标签不增加代数机会。

- **代价与反证：** 不能借 TMA 少迭代速度；新帧上下文缓冲/身份检查。

- **迁移方式：** 抽取多目的共同表达式才改执行量；普通广播保留强基线。

- **结论：** 降为基线，不作主机制


#### R2-SP · SP-Gate / attention mass

定位：45。N/F/H/T：**3 / 3 / 5 / 3**。

- **原机制：** 小注意力质量不发后级请求。

- **最近先验与访问状态：** [SpAtten: Efficient Sparse Attention Architecture with Cascade Token and Head Pruning, HPCA 2021](https://arxiv.org/abs/2012.09852)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** H67 分数含共静默/时间项，低 Q 发放不等于低 gate；K=V。

- **真正增量：** 非零重要性裁剪是现有稀疏注意力迁移。

- **代价与反证：** 省流量前要算分数；有损阈值改变归一化分母和输出。

- **迁移方式：** 寻找量化门控精确为零的充分条件，并收费界计算；部署合同另列。

- **结论：** 保留精确界重构


#### R2-SMAM · SMAM-RP / dual-spike mask-add

定位：46；73；104。N/F/H/T：**2 / 1 / 6 / 1**。

- **原机制：** gate做Mask-Add，实值载荷做MAC。

- **最近先验与访问状态：** [Li et al., An Efficient Sparse Hardware Accelerator for Spike-Driven Transformer, 2025](https://arxiv.org/html/2501.07825)；本轮原文 III-C/D 已核：二值 Hadamard／token 累加／阈值 mask；不是 H67 Motion-XOR

- **冻结适配：** 原SMAM的token累加阈值mask不是Motion-XOR；不可吸收载荷不存在。

- **真正增量：** 两条已知路径相连；关键差异假设错误。

- **代价与反证：** 有真实multi-bit gate但不能称ATLIF幅度；多项score及归一化不能省。

- **迁移方式：** 借比较/位图交集电路，重建三计数＋K复用叶；新机制另命题。

- **结论：** 原RP版本淘汰


#### R2-TTB · Motion-TTB / motion bundle keys

定位：47；74；105。N/F/H/T：**1 / 4 / 6 / 2**。

- **原机制：** 把token-time改为(tile,dt,hyp)并共享权重。

- **最近先验与访问状态：** [Xu et al., Bishop: Sparsified Bundling Spiking Transformers on Heterogeneous Cores with Error-Constrained Pruning, ISCA 2025](https://arxiv.org/html/2505.12281)；本轮原文 §3/§5.1 已核：TTB、二值 QK 点积界、BSA／ECP-aware 训练均为已有内容

- **冻结适配：** token×T可用，但没有hyp；T2/T10要分开。

- **真正增量：** TTB与束内/束间权重复用已做；键改名无执行变化。

- **代价与反证：** 打包和尾部利用率；不能把Bishop倍数搬到本岛。

- **迁移方式：** 只保留TTB强基线；新的多目的公共部分和另研究。

- **结论：** 降为强基线


#### R2-OFECP · OF-ECP / error-bounded bundle prune

定位：47；74。N/F/H/T：**4 / 3 / 5 / 4**。

- **原机制：** 以bundle活跃计数限制注意力删边误差。

- **最近先验与访问状态：** [Xu et al., Bishop: Sparsified Bundling Spiking Transformers on Heterogeneous Cores with Error-Constrained Pruning, ISCA 2025](https://arxiv.org/html/2505.12281)；本轮原文 §3/§5.1 已核：TTB、二值 QK 点积界、BSA／ECP-aware 训练均为已有内容

- **冻结适配：** Bishop界是二值QK；H67共静默/时间项、门控分母和K=V均改变界。

- **真正增量：** 若推导新算术的可执行精确界，可能有实质增量；原文直接写AEE界不成立。

- **代价与反证：** 局部分数误差不能直接变全网AEE界；裁Q与裁K不对称。

- **迁移方式：** 优先推导门控舍入后严格零/严格相同输出证书，非把classification界改名。

- **结论：** 保留严谨重构，不接受现成AEE界


#### R2-STH · STH-Gate

定位：48；75；105。N/F/H/T：**3 / 2 / 4 / 3**。

- **原机制：** 按空间/时间head类型选不同稀疏pattern。

- **最近先验与访问状态：** Sparse VideoGen: Accelerating Video Diffusion Transformers with Spatial-Temporal Sparsity, ICML 2025；原分配文件未给可靠链接；本轮未全文核

- **冻结适配：** H67共12块统一Motion-XOR路径；无已定义head类型。

- **真正增量：** 将视频扩散的头分类搬到OF；未有自然分型证据。

- **代价与反证：** 分类器、稀疏模式/误差；Tw2不足以直接套长视频时间head。

- **迁移方式：** 先以真实贡献分布检验可分型性；必要时独立稀疏训练。

- **结论：** 未证假说，不按旧9分推进


#### R2-PARO · Trajectory-Reorder Attn

定位：49；77。N/F/H/T：**2 / 2 / 5 / 2**。

- **原机制：** 以轨迹/几何重排得到块结构，再稀疏量化。

- **最近先验与访问状态：** [Zhao et al., PAROAttention: Pattern-Aware ReOrdering for Efficient Sparse and Quantized Attention in Visual Generation Models, 2025](https://arxiv.org/abs/2506.16054)；本轮 arXiv 原始摘要已核；原文件将其笼统称专用 PE 加速器，全文未核，暂不承接 ASIC 实现断言

- **冻结适配：** 当前无轨迹或对极约束；位置偏置和窗口mask必须同步映射。

- **真正增量：** 全排列等价则只有调度变化；删边/量化则新算法。

- **代价与反证：** 排序/逆置换、mask一致性、轨迹来源；仅换序不能降低算术量。

- **迁移方式：** 只当结构稀疏对照；需证明真正移除哪些执行。

- **结论：** 原主创新不足


#### R2-TOME · TokMerge-OF

定位：50；81。N/F/H/T：**3 / 2 / 5 / 3**。

- **原机制：** 合并相近token以减少后续执行。

- **最近先验与访问状态：** [Bolya et al., Token Merging: Your ViT But Faster](https://arxiv.org/abs/2210.09461)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** 稠密光流和skip需要位置逐像素恢复；二值同token也可能位置偏置不同。

- **真正增量：** token merging已有；OF应用本身非新原语。

- **代价与反证：** 匹配/聚类、位置恢复、AEE边界；近似合并改变函数。

- **迁移方式：** 研究严格等价消费者集合的聚合，而非近似平均。

- **结论：** 可转精确共享归约，原案有损


#### R2-TSM · TS-Shift

定位：50。N/F/H/T：**2 / 2 / 5 / 1**。

- **原机制：** 部分通道时间移位，用数据搬运替代时间算子。

- **最近先验与访问状态：** [Lin et al., TSM: Temporal Shift Module for Efficient Video Understanding, ICCV 2019](https://arxiv.org/abs/1811.08383)；未全文核；仅原 idea 文件列引，不采用其性能或首创断言

- **冻结适配：** PSN learned temporal mix不能任意替换成shift。

- **真正增量：** 既有TSM迁移。

- **代价与反证：** 零MAC仍有数据路由/缓存；需要训练和时间边界处理。

- **迁移方式：** 不替换ep34，作为新训练模型对照。

- **结论：** 本轮不迁


#### R2-SINK · SinkSparse

定位：50。N/F/H/T：**1 / 2 / 4 / 1**。

- **原机制：** 保留sink及局部块的稀疏注意力。

- **最近先验与访问状态：** RainFusion / sink sparse attention；原文件简称不足；未核

- **冻结适配：** 没有冻结sink定义；短窗不等于自回归KV场景。

- **真正增量：** 源文献未核，现案只是模式名称。

- **代价与反证：** 可能删掉有效Motion-XOR交互；必须新mask/AEE。

- **迁移方式：** 若存在精确共同项可消元，另推代数；勿凭sink标签。

- **结论：** 来源及适配未闭


#### R2-CW · CW-Reuse / latent channel reuse

定位：50。N/F/H/T：**3 / 4 / 5 / 3**。

- **原机制：** 跨时空复用通道部分结果。

- **最近先验与访问状态：** [Miao et al., Kaleido: Algorithm-Hardware Co-Design for Video Diffusion Transformers by Exploiting Latent Space Correlations, 2026](https://arxiv.org/abs/2607.13770)；本轮 arXiv 原始摘要已核；通道部分结果复用＋硬件已存在，具体误差与实现未全文核

- **冻结适配：** 二值相同可精确复用；“相似”膜/激活不能直接代替。

- **真正增量：** 通道复用与专用硬件先验已存在。

- **代价与反证：** 比较、参考状态、warp及失配修正；全通道真实相等率未给。

- **迁移方式：** 改为精确输出不变证书或多源虚拟基；避免缓存容量调优故事。

- **结论：** 需更强执行原语


#### R2-PACK1 · C1* residual/eager/wake联合包

定位：17–33；61。N/F/H/T：**2 / 1 / 3 / 1**。

- **原机制：** 将方向、残差、遮挡与预测器并列接到C1。

- **最近先验与访问状态：** [Liu et al., An Ultra-High Performance and Scalable Optical Flow Hardware Accelerator Based on FPGA for Autonomous Driving, TCAS-I 2025](https://ieeexplore.ieee.org/document/11030859/)；本轮 IEEE 原始摘要已核：方向预测、可配置 PE、金字塔管线；未取得全文

- **冻结适配：** 多个原算法不存在的反馈边；不是ep34原位实现。

- **真正增量：** 组件堆叠不能自动增加单机制新意。

- **代价与反证：** 成本和误差耦合，TCAS-II 4.5页无法支撑每个新算法。

- **迁移方式：** 拆出一个确实少执行的原语，其余最多底座。

- **结论：** 否决整包贡献叙事


#### R2-PACK2 · C2* dual-rail/motion/sparse联合包

定位：37–64。N/F/H/T：**1 / 0 / 3 / 0**。

- **原机制：** 实值双轨、运动context、分型稀疏及Mask-Add堆叠。

- **最近先验与访问状态：** [Xu et al., Bishop: Sparsified Bundling Spiking Transformers on Heterogeneous Cores with Error-Constrained Pruning, ISCA 2025](https://arxiv.org/html/2505.12281)；本轮原文 §3/§5.1 已核：TTB、二值 QK 点积界、BSA／ECP-aware 训练均为已有内容

- **冻结适配：** 实值载荷和hyp身份冲突；SDSA与H67混用。

- **真正增量：** 多数是已知路由或标签换名。

- **代价与反证：** 多后端面积/训练/误差未闭，不能相乘收益。

- **迁移方式：** 去掉虚构载荷与hyp，围绕一个真实消费者不变量重写。

- **结论：** 否决整包


### research/10_cim_spike_opticalflow_accelerators.md

精读完成；SHA256 `39262b1aa4f0274999c8f302347abddffa0fc45ead29e363fb7b4592501c8f92`。


#### R3C-M1 · Fused-WVMEM-CIM / ATLIF-FuseCIM

定位：61–70;44;224;257。N/F/H/T：**2 / 1 / 1 / 1**。

- **原机制：** 在定制 SRAM 内共享 W/膜状态位线完成更新与阈值比较

- **最近先验与访问状态：** [IMPULSE, Agrawal et al., 2021](https://arxiv.org/html/2105.08217v1)；PRIMARY_FULLTEXT_READ; §II–III, 10T SRAM、多行读和 IF/LIF/RMP 已核

- **冻结适配：** 原方案不适配：ep34 是满秩 PSN，无 IF/LIF reset；出口无 int8 载荷

- **真正增量：** 搬到 OF 和修改标签，没有给出新存储服务原语

- **代价与反证：** 10T 与多行同时读非 TS1N28 1RW；膜语义错误

- **迁移方式：** 只借局部状态驻留；须重新定义 A*x+b 数据流，不能保留 fused-CIM 主张

- **结论：** REJECT_AS_WRITTEN


#### R3C-M2 · SpiDR-Reconf-CIM / PRRC-CIM-Modes

定位：72–81;45;228。N/F/H/T：**2 / 2 / 2 / 1**。

- **原机制：** 按精度/规模重配置，零跳过并异步衔接 CIM

- **最近先验与访问状态：** [SpiDR, 2024 preprint](https://arxiv.org/html/2411.02854v1)；PRIMARY_FULLTEXT_SECTIONS_READ; §II-A–F, 10T macro、dual-port IFspad、IF/LIF/reset 已核

- **冻结适配：** 不同精度、head 分流和 IF/LIF 不属于冻结模型

- **真正增量：** 按金字塔选择模式属于应用映射

- **代价与反证：** 定制 10T、dual-port IFspad、精度改变；无真实 AEE/宏路径

- **迁移方式：** 只借任务粒度配置与有限 FIFO；不得搬 silicon PPA

- **结论：** REJECT_AS_HEADLINE


#### R3C-M3a · MSB-Skip-CIM

定位：83–92;227;260。N/F/H/T：**2 / 5 / 6 / 2**。

- **原机制：** 抑制权重符号扩展/高位活动

- **最近先验与访问状态：** [Neuro-CIM, VLSI 2022/JSSC extension](https://doi.org/10.1109/VLSITechnologyandCir46769.2022.9830276)；NOT_ACCESSED; 仅本地调研引文，具体数字和电路未经本审阅复核

- **冻结适配：** 无损冗余位隔离可作为整数部署优化；冻结 FP 未量化

- **真正增量：** OF 权重统计驱动普通位隔离

- **代价与反证：** 位级稀疏先验密集；模拟位线方案禁用；实际 INT8 分布未核

- **迁移方式：** 转为数字位切片活动隔离配套，不作主机制

- **结论：** SUPPORT_ONLY


#### R3C-M3b · EarlyStop-Neuron-CIM / NeuroGate-CIM

定位：83–92;227;260。N/F/H/T：**2 / 1 / 4 / 2**。

- **原机制：** 按神经元或流残差阈值停止后续计算

- **最近先验与访问状态：** [Neuro-CIM, VLSI 2022/JSSC extension](https://doi.org/10.1109/VLSITechnologyandCir46769.2022.9830276)；NOT_ACCESSED; 仅本地调研引文，具体数字和电路未经本审阅复核

- **冻结适配：** 残差预算早停改变模型；普通 SNN 阈值不覆盖 full-rank PSN

- **真正增量：** 应用残差替换终止条件

- **代价与反证：** 未证明输出不变；没有未来输入/动态 BN 的界

- **迁移方式：** 只能转为保守消费者证书，参考附加候选；不可继续原有 lossy 条件

- **结论：** REJECT_AS_WRITTEN


#### R3C-M4 · CD-IF-SpikeCIM

定位：94–103;223;256。N/F/H/T：**1 / 0 / 0 / 0**。

- **原机制：** charge-domain IF 负责二值门，另一轨算幅值

- **最近先验与访问状态：** [Spike-CIM, A-SSCC 2022](https://doi.org/10.1109/A-SSCC56115.2022.9980797)；NOT_ACCESSED; 本地引文

- **冻结适配：** 冻结 ATLIF 没有该 int8 幅值轨；Motion 也非原 SDSA

- **真正增量：** 两类现有 CIM 并置

- **代价与反证：** 模拟 CIM 禁令；charge IF 不等于 PSN；错误载荷身份

- **迁移方式：** 不迁移物理架构，仅保留比较结果是二值的事实

- **结论：** REJECT


#### R3C-M5 · TwinCol-TTFS-OF

定位：105–114;229;261。N/F/H/T：**1 / 0 / 1 / 0**。

- **原机制：** 正负双列和首次发放时间编码流假设

- **最近先验与访问状态：** [TFSRAM, TCAS-AI 2024](https://doi.org/10.1109/TCASAI.2024.3452649)；NOT_ACCESSED; 本地引文

- **冻结适配：** TTFS 重新编码、粗流假设均不在 ep34

- **真正增量：** 将既有正负权重命名成流方向

- **代价与反证：** 新编码/新模型，模拟电路，符号不等同 u/v 方向

- **迁移方式：** 保留文献；不能作为 frozen RTL

- **结论：** REJECT


#### R3C-M6 · TD-CIM-Gate

定位：116–125;231;247。N/F/H/T：**1 / 1 / 1 / 1**。

- **原机制：** 异步时间域 synapse/neuron 用于事件唤醒

- **最近先验与访问状态：** [Park et al., time-domain SNN CIM, TCAS-I 2025](https://doi.org/10.1109/TCSI.2024.3480350)；NOT_ACCESSED; 本地引文

- **冻结适配：** 原始事件可触发 IO，但不证明深层计算可免

- **真正增量：** 增加一个前置唤醒域

- **代价与反证：** 模拟/时间编码离开数字28nm；false-wake 与 missed-update 无证书

- **迁移方式：** 普通 always-on 唤醒仅系统配套

- **结论：** REJECT_AS_HEADLINE


#### R3C-M7 · ISNA-nvCIM-FFN

定位：127–136;230;248。N/F/H/T：**2 / 2 / 0 / 1**。

- **原机制：** 静态 FFN 权重驻 RRAM，内存内激活

- **最近先验与访问状态：** [Yan et al., RRAM in-situ nonlinear activation, VLSI 2019](https://doi.org/10.23919/VLSIT.2019.8776485)；NOT_ACCESSED; 本地引文

- **冻结适配：** 冻结图的静态权重存在，NVM/IF/readout 不是现有合同

- **真正增量：** 静动态分离是现成映射

- **代价与反证：** RRAM 不在 foundry SRAM 合同；量化/噪声与激活语义未闭

- **迁移方式：** 仅借静态 W 与动态激活的寿命区分

- **结论：** OUT_OF_SCOPE


#### R3C-M8 · Het-FrameEvent-CIM / Het-OF-CIM

定位：138–147;221;245。N/F/H/T：**2 / 0 / 0 / 0**。

- **原机制：** frame CNN 放 RRAM、event SNN 放 SRAM，分别供电

- **最近先验与访问状态：** [Lele et al., fused frame/event RRAM/SRAM SoC, JSSC 2024](https://doi.org/10.1109/JSSC.2023.3297411)；NOT_ACCESSED; 本地引文

- **冻结适配：** ep34 事件体素单输入；没有 APS/frame CNN 分支

- **真正增量：** 将 tracking 改成 OF 的系统组合

- **代价与反证：** 新增传感/帧路径、RRAM、双轨载荷均改身份；五页容纳不了多子系统

- **迁移方式：** 不能迁移整机；只借低开销休眠/唤醒接口

- **结论：** REJECT


#### R3C-M9 · XForm-Split-SDSA

定位：149–158;225;258。N/F/H/T：**2 / 3 / 2 / 2**。

- **原机制：** 静态投影/FFN 与动态 attention 存储介质分流

- **最近先验与访问状态：** [X-Former, TVLSI 2023](https://arxiv.org/abs/2303.07470)；NOT_ACCESSED_THIS_AUDIT; 本地引文

- **冻结适配：** 寿命差别存在；H67 为 Motion-XOR 非 SDSA，K作V

- **真正增量：** 应用到新算子的组织映射

- **代价与反证：** NVM宏未有；普通静动态分离不足标题贡献

- **迁移方式：** 数字 SRAM 中区分权重与配对 K 寿命可作实现基础

- **结论：** SUPPORT_ONLY_AFTER_REWRITE


#### R3C-M10a · ACAM-NonVMM

定位：160–172。N/F/H/T：**1 / 0 / 0 / 0**。

- **原机制：** 用 ACAM 支持 softmax/非线性等非 VMM

- **最近先验与访问状态：** [RACE-IT](https://arxiv.org/abs/2312.06532)；NOT_ACCESSED; 本地引文，ICCD 年份未复核

- **冻结适配：** 冻结有特定归一化路径；文内“spike所以无softmax”不适用

- **真正增量：** 通用 analog Transformer 技术直接挂入

- **代价与反证：** 模拟禁令；非 VMM 数值与 ep34/部署 Shiftmax 不同

- **迁移方式：** 不迁移

- **结论：** REJECT


#### R3C-M10b · NoiseAware-ATLIF-Q

定位：160–172。N/F/H/T：**1 / 0 / 0 / 0**。

- **原机制：** 噪声感知训练和 attention 量化匹配 PCM

- **最近先验与访问状态：** [Spoon et al., Toward Software-Equivalent Accuracy on Transformer-Based DNNs With Analog Memory Devices, 2021](https://doi.org/10.3389/fncom.2021.675741)；NOT_ACCESSED; 本地引文

- **冻结适配：** 重训练/量化与冻结关闭 quant 冲突

- **真正增量：** 训练流程应用映射

- **代价与反证：** 没有模拟后端；真实 ATLIF 出口二值，无声称实值载荷

- **迁移方式：** 仅另开有损研究才可谈，此次不采用

- **结论：** REJECT


#### R3C-M11 · PCM-FFN + SSA-OF / SSA-MotionMask

定位：174–183;226;259。N/F/H/T：**2 / 1 / 1 / 1**。

- **原机制：** PCM 线性层加 stochastic AND-count attention

- **最近先验与访问状态：** [Xpikeformer](https://arxiv.org/abs/2408.08794)；NOT_ACCESSED_THIS_AUDIT; 不把本地 HW 标签当硅证据

- **冻结适配：** 随机注意力不等于冻结 Motion-XOR；没有实值 ATLIF 第二轨

- **真正增量：** motion mask 叠加现成混合架构

- **代价与反证：** 噪声/随机采样/重训；原稿将 simulation 混为 true HW 的风险

- **迁移方式：** 可借流式 attention 服务组织，不借随机算术/模拟宏

- **结论：** REJECT_AS_WRITTEN


#### R3C-M12 · DigiCIM-PayloadLane

定位：185–194;232;256。N/F/H/T：**1 / 1 / 2 / 1**。

- **原机制：** 多比特全精度数字 CIM 处理 ATLIF 幅值

- **最近先验与访问状态：** [Chih et al., all-digital SRAM CIM, ISSCC 2021](https://doi.org/10.1109/ISSCC42613.2021.9365766)；NOT_ACCESSED; 本地引文

- **冻结适配：** 出口二值，幅值轨动机错误；PSN内部实值另论

- **真正增量：** 现成数字宏作一条载荷通路

- **代价与反证：** 定制 CIM 并非 foundry 1RW，不能用强迫1bit的AEE崩溃论证

- **迁移方式：** 数字多操作数归约可配套，不保留 payload claim

- **结论：** REJECT_AS_HEADLINE


#### R3C-M13 · MW-CIM-TileGate

定位：196–204;222;244。N/F/H/T：**3 / 1 / 1 / 2**。

- **原机制：** warped residual 超阈值才改写/激活 CIM tile

- **最近先验与访问状态：** [MotionDeltaCNN, Parger et al., 2022–2023](https://arxiv.org/html/2210.09887v5)；PRIMARY_FULLTEXT_SECTIONS_READ; §3.1–3.7，运动对齐、模拟状态、全缓冲刷新已核

- **冻结适配：** 冻结没有 warped-frame residual 路；阈值跳过未证

- **真正增量：** 把 DeltaCNN 条件接到宏唤醒

- **代价与反证：** 动态 BN 与全局归一化破坏局部免算；CIM禁令；预测误差

- **迁移方式：** 仅可抽出带完整证书的跨输入执行抑制，见新增候选

- **结论：** REJECT_AS_WRITTEN


#### R3C-M14 · DualRail-CIM

定位：206–213;223;256;264。N/F/H/T：**1 / 0 / 0 / 0**。

- **原机制：** 二值门 charge-CIM＋实值 ATLIF DigiCIM

- **最近先验与访问状态：** [Spike-CIM, A-SSCC 2022](https://doi.org/10.1109/A-SSCC56115.2022.9980797)；NOT_ACCESSED; 本地引文

- **冻结适配：** 与冻结二值 ATLIF 直接冲突

- **真正增量：** 物理介质并置而非新计算机制

- **代价与反证：** 不存在待保留int8幅值；模拟禁令；两宏面积/域转换

- **迁移方式：** 不作为新岛

- **结论：** REJECT


#### R3C-LANDSCAPE-SPIKEWL · Spike-WL-Gate-CIM / ADC-less

定位：52–53;28–36。N/F/H/T：**1 / 3 / 3 / 1**。

- **原机制：** 按输入 spike 关字线并以比较器替代 ADC

- **最近先验与访问状态：** [Neuro-CIM, VLSI 2022/JSSC extension](https://doi.org/10.1109/VLSITechnologyandCir46769.2022.9830276)；NOT_ACCESSED; 仅本地调研引文，具体数字和电路未经本审阅复核

- **冻结适配：** 输入二值这一事实成立；不推出标准1RW可多行计算

- **真正增量：** 没有区别化增量

- **代价与反证：** 普通零跳过和已有宏技术；不可凭迁移任务给高分

- **迁移方式：** 零操作数隔离作为强基线

- **结论：** BASELINE_ONLY


### research/11_event_camera_stack_accelerators.md

精读完成；SHA256 `a1f2a4e6f229e98ca9dc810a9143778e6c21b533197f02a2ed87886cd8d2a524`。


#### R3E-01 · AdaptSlice-Tw

定位：45–56;295;322;400。N/F/H/T：**2 / 1 / 5 / 2**。

- **原机制：** 事件计数/流反馈改变积分时间片和SPE唤醒

- **最近先验与访问状态：** [EDFLOW, Liu and Delbruck, TCSVT 2022](https://doi.org/10.1109/TCSVT.2022.3156653)；NOT_ACCESSED; 本地引文

- **冻结适配：** 改变输入体素时间划分、T=10或T_w=2，非冻结前处理

- **真正增量：** 将曝光反馈迁到网络前端

- **代价与反证：** 没有同一输入张量等价；低事件率也不意味着深层结果不变

- **迁移方式：** 只可保持原体素值的事件打包；自适应窗需新AEE路线

- **结论：** REJECT_AS_WRITTEN


#### R3E-02 · SpatLoc-Wake

定位：60–71;306;326;403。N/F/H/T：**1 / 3 / 6 / 1**。

- **原机制：** 空间活动簇限定被执行的邻域/tile

- **最近先验与访问状态：** [ASNA-Flow, Wang et al., TVLSI 2025](https://ieeexplore.ieee.org/document/11142472/)；ACCESS_BLOCKED; 出版记录已识别，未取得可核全文；不引用本地芯片数值

- **冻结适配：** 仅跳过已证明无贡献项合法，不能限制固定450token窗口

- **真正增量：** ASNA空间局部性改任务标签

- **代价与反证：** ASNA已占近邻；动态BN/共静默分母使空位置仍有消费者

- **迁移方式：** 实际位图可作执行元数据；不是论文原语

- **结论：** BASELINE_ONLY


#### R3E-03 · BitHyp-Prior

定位：75–86;296;320;340;401。N/F/H/T：**3 / 2 / 5 / 3**。

- **原机制：** 占据位图移位并比较速度假设，预测昂贵路径是否执行

- **最近先验与访问状态：** [EventShiftFlow](https://arxiv.org/abs/2605.28312)；NOT_ACCESSED; 本地 2026 引文及 FPGA 指标未独立核

- **冻结适配：** 可旁路预测调度顺序；按置信度省结果会改ep34

- **真正增量：** 便宜先验接昂贵网络的层级组合

- **代价与反证：** 错预测不能漏算；新增hyp非冻结；FPGA指标未复核

- **迁移方式：** 若只提前取数/排序且不丢任务可保图，但收益更薄

- **结论：** RESEARCH_ONLY_REDEFINE


#### R3E-04 · ISC-TokBudget

定位：90–101;297;325;399。N/F/H/T：**2 / 0 / 2 / 1**。

- **原机制：** 传感器autoencoder压缩latent控制token配额

- **最近先验与访问状态：** [3-D In-Sensor Computing for Real-Time DVS Data Compression, LSSC 2024](https://doi.org/10.1109/LSSC.2024.3375110)；NOT_ACCESSED; 原稿 silicon 标签未复核

- **冻结适配：** 原输入不是该latent，删除quiet区域不等价

- **真正增量：** 压缩metadata驱动配额

- **代价与反证：** 新传感器/网络/有损latent；silicon标签未核

- **迁移方式：** 无损IO压缩可配套，不能沿用AE丢token

- **结论：** REJECT


#### R3E-05 · SkipEdge-ROI

定位：105–116;298;325;399。N/F/H/T：**2 / 1 / 3 / 1**。

- **原机制：** 边缘/ROI传感器读出控制exact/propagate路径

- **最近先验与访问状态：** [Guo et al., stacked CIS+EVS, JSSC 2023](https://doi.org/10.1109/JSSC.2023.3303154)；NOT_ACCESSED; 本地引文

- **冻结适配：** 内部位置、APS补纹理不是ep34消费者图

- **真正增量：** 传感元数据接调度器

- **代价与反证：** 传感器能力不在本地输入合同；删事件/新增帧分支改模型

- **迁移方式：** 只可用ROI做无损服务优先级，不能跳合法窗口

- **结论：** REJECT_AS_WRITTEN


#### R3E-06 · ShiftTS-SPE

定位：120–131;297;327;400。N/F/H/T：**2 / 2 / 6 / 2**。

- **原机制：** 移位近似时间面或直方图编码，ping-pong缓冲

- **最近先验与访问状态：** [HOMI](https://arxiv.org/abs/2508.12637)；NOT_ACCESSED; 本地引文

- **冻结适配：** SETS/SLTS不是冻结体素；近似指数编码改变输入

- **真正增量：** 分类encoder换OF任务

- **代价与反证：** 前处理不同需新AEE；shift ALU/双缓冲先验普通

- **迁移方式：** 可借实现事件体素精确累加的缓冲结构，必须保原bin与极性

- **结论：** SUPPORT_ONLY_AFTER_REWRITE


#### R3E-07 · TDE3-Prior

定位：135–146;296;318;339;401。N/F/H/T：**3 / 1 / 4 / 2**。

- **原机制：** 方向敏感时间差/抑制脉冲生成廉价流先验

- **最近先验与访问状态：** [TDE-3](https://arxiv.org/abs/2402.11662)；NOT_ACCESSED; 本地引文

- **冻结适配：** TDE不是满秩PSN；据其置信度跳SDSA改模型

- **真正增量：** TDE先验＋残差网络组合

- **代价与反证：** 额外状态与调参；先验一致不等于网络输出一致

- **迁移方式：** 只作不漏算的请求排序/预取线索；否则另开训练路线

- **结论：** REJECT_AS_WRITTEN


#### R3E-08 · TMA-Agg

定位：150–161;305;319;337;402。N/F/H/T：**2 / 0 / 2 / 1**。

- **原机制：** 切事件片、相关体线性lookup、跨时motion aggregation并减少refinement

- **最近先验与访问状态：** [TMA, Liu et al., ICCV 2023](https://arxiv.org/html/2303.11629v2)；PRIMARY_FULLTEXT_SECTIONS_READ; §3.2/式4–6，RAFT 与 correlation volumes 已核

- **冻结适配：** 与ep34直接不匹配；原文§3.2基于RAFT和H×W×H×W correlation

- **真正增量：** 把已有TMA算子新增到spikeformer

- **代价与反证：** 冻结无相关体、无RAFT迭代；并非可换名的硬件调度

- **迁移方式：** 仅可借时序信息重用问题意识，不能搬TMA算子

- **结论：** REJECT


#### R3E-09 · EvQ-Win

定位：165–176;306;321;341;403。N/F/H/T：**3 / 3 / 6 / 3**。

- **原机制：** 像素事件队列先空间后时间筛邻居

- **最近先验与访问状态：** [EvGNN, Yang, Kneip and Frenkel](https://arxiv.org/html/2404.19489v2)；PRIMARY_FULLTEXT_SECTIONS_READ; §III/IV-B/IV-C，prism、16邻居截断和因果图已核

- **冻结适配：** 固定15×15×2窗口可借地址分离；directed graph和16邻居截断不合法

- **真正增量：** 专用于固定窗的无损gather需要重新设计

- **代价与反证：** 原文IV-B2有Dmax16提前停止；变半球/prism会改邻接；未来peer不能因果略过

- **迁移方式：** 只借空间索引/时间tag分离，完整供给450token及peer；先比较直接地址发生器

- **结论：** SUPPORT_ONLY_AFTER_REWRITE


#### R3E-10 · SubMan-Pipe

定位：180–191;306;324;342;403。N/F/H/T：**2 / 1 / 5 / 2**。

- **原机制：** 保持输入/输出位置同支撑集的稀疏流水

- **最近先验与访问状态：** [ESDA, Gao et al., FPGA 2024](https://arxiv.org/html/2401.05626v1)；PRIMARY_FULLTEXT_SECTIONS_READ; §3.2–3.3，submanifold 改变输出支撑集、稀疏行缓冲已核

- **冻结适配：** 不适配；原文§3.2明确强制零位置仍零，而普通卷积会扩张

- **真正增量：** 把新的稀疏算子替换冻结普通卷积

- **代价与反证：** 改变输出位置集合；BN可激活原零位置；不能说只是数据流

- **迁移方式：** 保留ESDA行缓冲ready/valid模板，必须重新生成真实卷积输出支撑

- **结论：** REJECT_SEMANTICS_KEEP_PROTOCOL


#### R3E-11 · EV-Wake+

定位：195–206;298;399。N/F/H/T：**1 / 2 / 6 / 1**。

- **原机制：** always-on事件检测器触发dense NPU

- **最近先验与访问状态：** [Cha et al., event-based NPU triggering, ISCAS 2025](https://doi.org/10.1109/ISCAS56072.2025.11043213)；NOT_ACCESSED; 本地引文

- **冻结适配：** 原始event低活动不足以免算整个冻结网络

- **真正增量：** 已有唤醒机制改为OF残差控制

- **代价与反证：** 当前输入无事件也可有跨时/归一化输出；错误跳过需证明

- **迁移方式：** 电源域唤醒与吞吐调度可配套，不作新算法

- **结论：** BASELINE_OR_SYSTEM_SUPPORT


#### R3E-12 · SNE-DualPath

定位：210–221;329。N/F/H/T：**1 / 1 / 3 / 1**。

- **原机制：** SNE处理event，ANN处理帧，分别power-gate

- **最近先验与访问状态：** [Kraken/SNE](https://arxiv.org/abs/2209.01065)；NOT_ACCESSED; 本地引文

- **冻结适配：** 新增APS/ANN路径，二值ATLIF与{gate,payload}两轨不同

- **真正增量：** 现成异构SoC应用到OF

- **代价与反证：** 超出一机制短文范围；无可接入frame合同

- **迁移方式：** 只借执行岛接口和空闲管理

- **结论：** REJECT_AS_HEADLINE


#### R3E-13 · PlaneHist-Prior

定位：225–236;329;401。N/F/H/T：**2 / 1 / 5 / 2**。

- **原机制：** 近期事件历史平面拟合产生廉价coarse flow

- **最近先验与访问状态：** [Aung et al., plane-fitting optical-flow FPGA, ISCAS 2018](https://doi.org/10.1109/ISCAS.2018.8351588)；NOT_ACCESSED; 本地引文

- **冻结适配：** 新增粗流推断/置信度跳过不等价

- **真正增量：** 传统前端＋网络回退组合

- **代价与反证：** aperture/纹理场景误差、历史缓存；与ARM-Acc模型冲突

- **迁移方式：** 保图时最多排序/预取，需证明净带宽收益

- **结论：** REJECT_AS_WRITTEN


#### R3E-14 · DualEng-SDSA

定位：240–251;307;323;404。N/F/H/T：**2 / 2 / 6 / 2**。

- **原机制：** 二值attention引擎与ATLIF实值payload引擎分离

- **最近先验与访问状态：** [FireFly-T](https://arxiv.org/abs/2505.12771)；PRIMARY_FULLTEXT_RELEVANT_SECTIONS_READ_IN_PRECEDING_RESEARCH; FPGA attention/transpose，未复核期刊年份

- **冻结适配：** Motion-XOR非AND-only；ATLIF实值载荷轨不存在

- **真正增量：** 分类attention硬件换OF并增加已知项

- **代价与反证：** 错误二轨动机；FPGA LUT6与标准单元不可直接比；量化身份混淆

- **迁移方式：** 只借分离score/value/linear服务，完整同功能MX baseline

- **结论：** REWRITE_TO_IMPLEMENTATION


#### R3E-15 · PredExit-OF

定位：255–266;328;405。N/F/H/T：**2 / 0 / 4 / 1**。

- **原机制：** 预测后续RAFT迭代无用而早退

- **最近先验与访问状态：** [ERAFT, frame RAFT FPGA, ISCAS 2025](https://doi.org/10.1109/ISCAS56072.2025.11043529)；NOT_ACCESSED; 不能混为 E-RAFT

- **冻结适配：** ep34没有迭代update block；T=10也不是可随时结束的RAFT迭代

- **真正增量：** 将迭代exit嫁接PSN

- **代价与反证：** fullrankPSN未读输入仍影响各步；freeze无exit训练

- **迁移方式：** 若保留必须改成严格二值消费者证明，不是残差阈值

- **结论：** REJECT


### research/12_ROUND3_SYNTHESIS_CIM_EVENT.md

精读完成；SHA256 `5639c9a4e7ed494c654622362360b00b0e25722230526e961315a81ee64ec453`。


#### R3S-01 · MW-CIM-TileGate

定位：13;20;40。N/F/H/T：**3 / 1 / 1 / 2**。

- **原机制：** warped residual 超阈值才改写/激活 CIM tile

- **最近先验与访问状态：** [MotionDeltaCNN, Parger et al., 2022–2023](https://arxiv.org/html/2210.09887v5)；PRIMARY_FULLTEXT_SECTIONS_READ; §3.1–3.7，运动对齐、模拟状态、全缓冲刷新已核

- **冻结适配：** 冻结没有 warped-frame residual 路；阈值跳过未证

- **真正增量：** 把 DeltaCNN 条件接到宏唤醒

- **代价与反证：** 动态 BN 与全局归一化破坏局部免算；CIM禁令；预测误差

- **迁移方式：** 仅可抽出带完整证书的跨输入执行抑制，见新增候选

- **结论：** REJECT_AS_WRITTEN


#### R3S-02 · Het-FrameEvent-CIM / Het-OF-CIM

定位：14;40。N/F/H/T：**2 / 0 / 0 / 0**。

- **原机制：** frame CNN 放 RRAM、event SNN 放 SRAM，分别供电

- **最近先验与访问状态：** [Lele et al., fused frame/event RRAM/SRAM SoC, JSSC 2024](https://doi.org/10.1109/JSSC.2023.3297411)；NOT_ACCESSED; 本地引文

- **冻结适配：** ep34 事件体素单输入；没有 APS/frame CNN 分支

- **真正增量：** 将 tracking 改成 OF 的系统组合

- **代价与反证：** 新增传感/帧路径、RRAM、双轨载荷均改身份；五页容纳不了多子系统

- **迁移方式：** 不能迁移整机；只借低开销休眠/唤醒接口

- **结论：** REJECT


#### R3S-03 · SpiDR-Reconf-CIM / PRRC-CIM-Modes

定位：15。N/F/H/T：**2 / 2 / 2 / 1**。

- **原机制：** 按精度/规模重配置，零跳过并异步衔接 CIM

- **最近先验与访问状态：** [SpiDR, 2024 preprint](https://arxiv.org/html/2411.02854v1)；PRIMARY_FULLTEXT_SECTIONS_READ; §II-A–F, 10T macro、dual-port IFspad、IF/LIF/reset 已核

- **冻结适配：** 不同精度、head 分流和 IF/LIF 不属于冻结模型

- **真正增量：** 按金字塔选择模式属于应用映射

- **代价与反证：** 定制 10T、dual-port IFspad、精度改变；无真实 AEE/宏路径

- **迁移方式：** 只借任务粒度配置与有限 FIFO；不得搬 silicon PPA

- **结论：** REJECT_AS_HEADLINE


#### R3S-04 · TDE3-Prior

定位：16;39。N/F/H/T：**3 / 1 / 4 / 2**。

- **原机制：** 方向敏感时间差/抑制脉冲生成廉价流先验

- **最近先验与访问状态：** [TDE-3](https://arxiv.org/abs/2402.11662)；NOT_ACCESSED; 本地引文

- **冻结适配：** TDE不是满秩PSN；据其置信度跳SDSA改模型

- **真正增量：** TDE先验＋残差网络组合

- **代价与反证：** 额外状态与调参；先验一致不等于网络输出一致

- **迁移方式：** 只作不漏算的请求排序/预取线索；否则另开训练路线

- **结论：** REJECT_AS_WRITTEN


#### R3S-05 · BitHyp-Prior

定位：16;39。N/F/H/T：**3 / 2 / 5 / 3**。

- **原机制：** 占据位图移位并比较速度假设，预测昂贵路径是否执行

- **最近先验与访问状态：** [EventShiftFlow](https://arxiv.org/abs/2605.28312)；NOT_ACCESSED; 本地 2026 引文及 FPGA 指标未独立核

- **冻结适配：** 可旁路预测调度顺序；按置信度省结果会改ep34

- **真正增量：** 便宜先验接昂贵网络的层级组合

- **代价与反证：** 错预测不能漏算；新增hyp非冻结；FPGA指标未复核

- **迁移方式：** 若只提前取数/排序且不丢任务可保图，但收益更薄

- **结论：** RESEARCH_ONLY_REDEFINE


#### R3S-06 · AdaptSlice-Tw

定位：17;39。N/F/H/T：**2 / 1 / 5 / 2**。

- **原机制：** 事件计数/流反馈改变积分时间片和SPE唤醒

- **最近先验与访问状态：** [EDFLOW, Liu and Delbruck, TCSVT 2022](https://doi.org/10.1109/TCSVT.2022.3156653)；NOT_ACCESSED; 本地引文

- **冻结适配：** 改变输入体素时间划分、T=10或T_w=2，非冻结前处理

- **真正增量：** 将曝光反馈迁到网络前端

- **代价与反证：** 没有同一输入张量等价；低事件率也不意味着深层结果不变

- **迁移方式：** 只可保持原体素值的事件打包；自适应窗需新AEE路线

- **结论：** REJECT_AS_WRITTEN


#### R3S-07 · SkipEdge-ROI

定位：17。N/F/H/T：**2 / 1 / 3 / 1**。

- **原机制：** 边缘/ROI传感器读出控制exact/propagate路径

- **最近先验与访问状态：** [Guo et al., stacked CIS+EVS, JSSC 2023](https://doi.org/10.1109/JSSC.2023.3303154)；NOT_ACCESSED; 本地引文

- **冻结适配：** 内部位置、APS补纹理不是ep34消费者图

- **真正增量：** 传感元数据接调度器

- **代价与反证：** 传感器能力不在本地输入合同；删事件/新增帧分支改模型

- **迁移方式：** 只可用ROI做无损服务优先级，不能跳合法窗口

- **结论：** REJECT_AS_WRITTEN


#### R3S-08 · EV-Wake+

定位：17;39。N/F/H/T：**1 / 2 / 6 / 1**。

- **原机制：** always-on事件检测器触发dense NPU

- **最近先验与访问状态：** [Cha et al., event-based NPU triggering, ISCAS 2025](https://doi.org/10.1109/ISCAS56072.2025.11043213)；NOT_ACCESSED; 本地引文

- **冻结适配：** 原始event低活动不足以免算整个冻结网络

- **真正增量：** 已有唤醒机制改为OF残差控制

- **代价与反证：** 当前输入无事件也可有跨时/归一化输出；错误跳过需证明

- **迁移方式：** 电源域唤醒与吞吐调度可配套，不作新算法

- **结论：** BASELINE_OR_SYSTEM_SUPPORT


#### R3S-09 · ShiftTS-SPE

定位：18。N/F/H/T：**2 / 2 / 6 / 2**。

- **原机制：** 移位近似时间面或直方图编码，ping-pong缓冲

- **最近先验与访问状态：** [HOMI](https://arxiv.org/abs/2508.12637)；NOT_ACCESSED; 本地引文

- **冻结适配：** SETS/SLTS不是冻结体素；近似指数编码改变输入

- **真正增量：** 分类encoder换OF任务

- **代价与反证：** 前处理不同需新AEE；shift ALU/双缓冲先验普通

- **迁移方式：** 可借实现事件体素精确累加的缓冲结构，必须保原bin与极性

- **结论：** SUPPORT_ONLY_AFTER_REWRITE


#### R3S-10 · DualRail-CIM

定位：28;33;42。N/F/H/T：**1 / 0 / 0 / 0**。

- **原机制：** 二值门 charge-CIM＋实值 ATLIF DigiCIM

- **最近先验与访问状态：** [Spike-CIM, A-SSCC 2022](https://doi.org/10.1109/A-SSCC56115.2022.9980797)；NOT_ACCESSED; 本地引文

- **冻结适配：** 与冻结二值 ATLIF 直接冲突

- **真正增量：** 物理介质并置而非新计算机制

- **代价与反证：** 不存在待保留int8幅值；模拟禁令；两宏面积/域转换

- **迁移方式：** 不作为新岛

- **结论：** REJECT


#### R3S-11 · Fused-WVMEM-CIM / ATLIF-FuseCIM

定位：28。N/F/H/T：**2 / 1 / 1 / 1**。

- **原机制：** 在定制 SRAM 内共享 W/膜状态位线完成更新与阈值比较

- **最近先验与访问状态：** [IMPULSE, Agrawal et al., 2021](https://arxiv.org/html/2105.08217v1)；PRIMARY_FULLTEXT_READ; §II–III, 10T SRAM、多行读和 IF/LIF/RMP 已核

- **冻结适配：** 原方案不适配：ep34 是满秩 PSN，无 IF/LIF reset；出口无 int8 载荷

- **真正增量：** 搬到 OF 和修改标签，没有给出新存储服务原语

- **代价与反证：** 10T 与多行同时读非 TS1N28 1RW；膜语义错误

- **迁移方式：** 只借局部状态驻留；须重新定义 A*x+b 数据流，不能保留 fused-CIM 主张

- **结论：** REJECT_AS_WRITTEN


#### R3S-12 · XForm-Split-SDSA

定位：29。N/F/H/T：**2 / 3 / 2 / 2**。

- **原机制：** 静态投影/FFN 与动态 attention 存储介质分流

- **最近先验与访问状态：** [X-Former, TVLSI 2023](https://arxiv.org/abs/2303.07470)；NOT_ACCESSED_THIS_AUDIT; 本地引文

- **冻结适配：** 寿命差别存在；H67 为 Motion-XOR 非 SDSA，K作V

- **真正增量：** 应用到新算子的组织映射

- **代价与反证：** NVM宏未有；普通静动态分离不足标题贡献

- **迁移方式：** 数字 SRAM 中区分权重与配对 K 寿命可作实现基础

- **结论：** SUPPORT_ONLY_AFTER_REWRITE


#### R3S-13 · MSB-Skip-CIM

定位：29。N/F/H/T：**2 / 5 / 6 / 2**。

- **原机制：** 抑制权重符号扩展/高位活动

- **最近先验与访问状态：** [Neuro-CIM, VLSI 2022/JSSC extension](https://doi.org/10.1109/VLSITechnologyandCir46769.2022.9830276)；NOT_ACCESSED; 仅本地调研引文，具体数字和电路未经本审阅复核

- **冻结适配：** 无损冗余位隔离可作为整数部署优化；冻结 FP 未量化

- **真正增量：** OF 权重统计驱动普通位隔离

- **代价与反证：** 位级稀疏先验密集；模拟位线方案禁用；实际 INT8 分布未核

- **迁移方式：** 转为数字位切片活动隔离配套，不作主机制

- **结论：** SUPPORT_ONLY


#### R3S-14 · EarlyStop-Neuron-CIM / NeuroGate-CIM

定位：29。N/F/H/T：**2 / 1 / 4 / 2**。

- **原机制：** 按神经元或流残差阈值停止后续计算

- **最近先验与访问状态：** [Neuro-CIM, VLSI 2022/JSSC extension](https://doi.org/10.1109/VLSITechnologyandCir46769.2022.9830276)；NOT_ACCESSED; 仅本地调研引文，具体数字和电路未经本审阅复核

- **冻结适配：** 残差预算早停改变模型；普通 SNN 阈值不覆盖 full-rank PSN

- **真正增量：** 应用残差替换终止条件

- **代价与反证：** 未证明输出不变；没有未来输入/动态 BN 的界

- **迁移方式：** 只能转为保守消费者证书，参考附加候选；不可继续原有 lossy 条件

- **结论：** REJECT_AS_WRITTEN


#### R3S-15 · TMA-Agg

定位：30;33;41;44。N/F/H/T：**2 / 0 / 2 / 1**。

- **原机制：** 切事件片、相关体线性lookup、跨时motion aggregation并减少refinement

- **最近先验与访问状态：** [TMA, Liu et al., ICCV 2023](https://arxiv.org/html/2303.11629v2)；PRIMARY_FULLTEXT_SECTIONS_READ; §3.2/式4–6，RAFT 与 correlation volumes 已核

- **冻结适配：** 与ep34直接不匹配；原文§3.2基于RAFT和H×W×H×W correlation

- **真正增量：** 把已有TMA算子新增到spikeformer

- **代价与反证：** 冻结无相关体、无RAFT迭代；并非可换名的硬件调度

- **迁移方式：** 仅可借时序信息重用问题意识，不能搬TMA算子

- **结论：** REJECT


#### R3S-16 · EvQ-Win

定位：31;33;41;44。N/F/H/T：**3 / 3 / 6 / 3**。

- **原机制：** 像素事件队列先空间后时间筛邻居

- **最近先验与访问状态：** [EvGNN, Yang, Kneip and Frenkel](https://arxiv.org/html/2404.19489v2)；PRIMARY_FULLTEXT_SECTIONS_READ; §III/IV-B/IV-C，prism、16邻居截断和因果图已核

- **冻结适配：** 固定15×15×2窗口可借地址分离；directed graph和16邻居截断不合法

- **真正增量：** 专用于固定窗的无损gather需要重新设计

- **代价与反证：** 原文IV-B2有Dmax16提前停止；变半球/prism会改邻接；未来peer不能因果略过

- **迁移方式：** 只借空间索引/时间tag分离，完整供给450token及peer；先比较直接地址发生器

- **结论：** SUPPORT_ONLY_AFTER_REWRITE


#### R3S-17 · DualEng-SDSA

定位：31。N/F/H/T：**2 / 2 / 6 / 2**。

- **原机制：** 二值attention引擎与ATLIF实值payload引擎分离

- **最近先验与访问状态：** [FireFly-T](https://arxiv.org/abs/2505.12771)；PRIMARY_FULLTEXT_RELEVANT_SECTIONS_READ_IN_PRECEDING_RESEARCH; FPGA attention/transpose，未复核期刊年份

- **冻结适配：** Motion-XOR非AND-only；ATLIF实值载荷轨不存在

- **真正增量：** 分类attention硬件换OF并增加已知项

- **代价与反证：** 错误二轨动机；FPGA LUT6与标准单元不可直接比；量化身份混淆

- **迁移方式：** 只借分离score/value/linear服务，完整同功能MX baseline

- **结论：** REWRITE_TO_IMPLEMENTATION


#### R3S-18 · SubMan-Pipe

定位：31。N/F/H/T：**2 / 1 / 5 / 2**。

- **原机制：** 保持输入/输出位置同支撑集的稀疏流水

- **最近先验与访问状态：** [ESDA, Gao et al., FPGA 2024](https://arxiv.org/html/2401.05626v1)；PRIMARY_FULLTEXT_SECTIONS_READ; §3.2–3.3，submanifold 改变输出支撑集、稀疏行缓冲已核

- **冻结适配：** 不适配；原文§3.2明确强制零位置仍零，而普通卷积会扩张

- **真正增量：** 把新的稀疏算子替换冻结普通卷积

- **代价与反证：** 改变输出位置集合；BN可激活原零位置；不能说只是数据流

- **迁移方式：** 保留ESDA行缓冲ready/valid模板，必须重新生成真实卷积输出支撑

- **结论：** REJECT_SEMANTICS_KEEP_PROTOCOL


### research/13_isscc_vlsi_hotchips_edge_npus.md

精读完成；SHA256 `d8221c69f451f6abe3d09ba71c201a3b8265620aee800812fcf82927d1c4f2ca`。


#### 13.M1 · FMSkip-OPSTW / DynPort-MFBD

定位：L56–65。N/F/H/T：**1 / 6 / 6 / 1**。

- **原机制：** 按 feature-map 零值跳过，动态重配访存端口。

- **最近先验与访问状态：** Samsung mobile DNN NPU, ISSCC / ISCA 2021（原文件）；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 普通零跳过可用；TS1N28 宏仍是1RW，K8带宽不变。

- **真正增量：** 零跳过已有；动态端口需说明物理多路复用的新增服务。

- **代价与反证：** 不能把逻辑端口分配说成新增 SRAM 端口；交叉网、排队与 bank 冲突收费。

- **迁移方式：** 仅作为 bank 调度强基线；无新执行内容不立项。

- **结论：** 不作为冻结主机制


#### 13.M2 · Entropy-EE-OF / MP-Pred-PRRC / TileVFS-Motion

定位：L67–76。N/F/H/T：**2 / 0 / 4 / 1**。

- **原机制：** 按置信度提前退出、预测精度并调 V/F。

- **最近先验与访问状态：** [Tambe et al., ISSCC 2023 Transformer accelerator; EdgeBERT MICRO 2021](https://sld.cs.columbia.edu/pubs/tambe_isscc23.pdf)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 冻结无多退出头，不能按分类熵直接判断每像素 flow 正确。

- **真正增量：** 置信度策略改应用已有；无法替代严格消费者判定。

- **代价与反证：** 新增训练/AEE、精度切换和物理电压域表征；早停 PSN 时间步错误。

- **迁移方式：** 从原包剔除；若另模型授权才做完整精度-能量 Pareto。

- **结论：** 不作为冻结主机制


#### 13.M3 · Asymp-OoO-ECP

定位：L78–87。N/F/H/T：**3 / 2 / 5 / 3**。

- **原机制：** 近似预筛分数，异步/乱序调度重算。

- **最近先验与访问状态：** [Wang et al., ISSCC 2022; Ayaka JSSC 2024](https://doi.org/10.1109/ISSCC42614.2022.9731686)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 冻结 Motion-XOR 分数公式可调度，但近似筛选没有保真条件。

- **真正增量：** OoO 是常见机制；若有精确剩余分数界则可重构。

- **代价与反证：** 分数复核/队列/结果恢复/银行带宽；不能把 approximate asymptotic 当无损。

- **迁移方式：** 研究 exact bound 前置裁决，需与简单全部三popcount比较，暂不实现。

- **结论：** 不作为冻结主机制


#### 13.M4 · BLT-CIM-SDSA

定位：L89–98。N/F/H/T：**2 / 0 / 1 / 0**。

- **原机制：** 定制 CIM 位线转置以降低动态 attention 转置成本。

- **最近先验与访问状态：** [Tu et al., ISSCC 2022](https://doi.org/10.1109/ISSCC42614.2022.9731645)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 普通1RW宏无位线计算/转置操作。

- **真正增量：** 地址转置可借但不能继承定制 CIM 能量与密度。

- **代价与反证：** 缺宏/器件/物理验证；用户禁 CIM。

- **迁移方式：** 剔除；普通 SRAM 转置仅工程。

- **结论：** 不作为冻结主机制


#### 13.M5 · BigLittle-DualRail

定位：L100–109。N/F/H/T：**2 / 0 / 4 / 1**。

- **原机制：** 轻重双引擎按输入难度选择 dense/implicit-weight 路径。

- **最近先验与访问状态：** [C-Transformer, ISSCC 2024](https://doi.org/10.1109/ISSCC49657.2024.10454330)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 冻结没有实值 ATLIF 双轨或遮挡专用 dense 模型。

- **真正增量：** 通用 big/little 配置；没有被冻结的可切换等价算子。

- **代价与反证：** 第二条模型/精度、选路与两套存储，需新训练评价。

- **迁移方式：** 当前剔除；不要为讲架构添加不存在的数据流。

- **结论：** 不作为冻结主机制


#### 13.M6 · HetSchedule-OF

定位：L111–120。N/F/H/T：**1 / 0 / 2 / 0**。

- **原机制：** 数字/模拟核按精度/密度异构调度。

- **最近先验与访问状态：** [DIANA, ISSCC 2022 / JSSC](https://doi.org/10.1109/ISSCC42614.2022.9731716)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 数字28nm普通宏与模拟核不相符。

- **真正增量：** 去掉模拟后是常规多引擎调度。

- **代价与反证：** 桥接、ADC与精度校准；仅调度没有主机制新意。

- **迁移方式：** 只借服务能量核算；不建异构CIM。

- **结论：** 不作为冻结主机制


#### 13.M7 · Retain-Prior-eMRAM

定位：L122–131。N/F/H/T：**1 / 1 / 3 / 1**。

- **原机制：** 睡眠保持模型/先验，低功耗唤醒恢复状态。

- **最近先验与访问状态：** [TinyVers, VLSI 2022 / JSSC 2023](https://doi.org/10.1109/JSSC.2023.3236566)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 没有可用 eMRAM 宏，也无冻结跨帧先验状态。

- **真正增量：** 常规 retention/always-on；任务名称不新增机制。

- **代价与反证：** 漏电、电源域、掉电一致性和状态有效期；替成SRAM不能继承eMRAM指标。

- **迁移方式：** 仅系统电源方案另任务，当前排除。

- **结论：** 不作为冻结主机制


#### 13.M8 · EWC-Cluster-MAC / EC-Psum-Gate

定位：L133–142。N/F/H/T：**2 / 3 / 6 / 2**。

- **原机制：** 合并相同有效权重，或用误差补偿预测部分和是否应继续。

- **最近先验与访问状态：** [QNAP, ISSCC 2021 / JSSC 2021; UCNN ISCA 2018](https://doi.org/10.1109/JSSC.2021.3113569)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 权重重复可精确；预测 ReLU 输出不直接等于 full-T PSN + 动态BN。

- **真正增量：** 普通相同码分组已由 UCNN/SumMerge覆盖；预测需新证书/误差合同。

- **代价与反证：** UCNN索引/组和位宽；动态BN统计仍需实际贡献；QNAP原文未取得，不细化未核ECP公式。

- **迁移方式：** 精确共享完整投影+阈值区间是另新模型候选；单码重复不足。

- **结论：** 不作为冻结主机制


#### 13.M9 · PVScale-ATLIF

定位：L144–153。N/F/H/T：**1 / 2 / 7 / 1**。

- **原机制：** 向量/组共享尺度的低位激活权重量化。

- **最近先验与访问状态：** [Keller et al., VLSI 2022 / JSSC 2023](https://doi.org/10.1109/JSSC.2023.3234893)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** ATLIF只有静态θ；部署权重量化另身份。

- **真正增量：** per-vector scale已存在；per-motion组名不产生原语。

- **代价与反证：** scale读写/乘法、饱和、AEE；Q1.7门控与int4载荷不可混同。

- **迁移方式：** 作为量化对照，当前不独立主打。

- **结论：** 不作为冻结主机制


#### 13.M10 · SalCascade-EVWake

定位：L155–164。N/F/H/T：**2 / 1 / 3 / 1**。

- **原机制：** 显著度→新颖性→DNN逐级唤醒，尽早关闭下游。

- **最近先验与访问状态：** [CogniVision, Gupta/Vohra/Alioto, VLSI 2024](https://doi.org/10.1109/VLSITechnologyandCir46783.2024.10631426)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 缺传感器/SoC与定义好的漏检任务；稠密OF不等于存在检测。

- **真正增量：** 级联wake是原论文已有；换成event energy不够。

- **代价与反证：** 唤醒延迟、电源域、漏检与输出质量；不能从岛推整机mW。

- **迁移方式：** 仅系统级未来方向，不塞入本短文。

- **结论：** 不作为冻结主机制


#### 13.M11 · AoV-Cascade-CardH

定位：L166–175。N/F/H/T：**1 / 1 / 3 / 1**。

- **原机制：** 驻片视觉流水与细粒度漏电管理。

- **最近先验与访问状态：** [Alpha-Vision, ISSCC 2026](https://ieeexplore.ieee.org/document/11409322)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 脸检测延迟/功率不能迁移到全分辨率OF；本项目无芯片顶层。

- **真正增量：** 商业/研究AoV系统组织，不是C2执行原语。

- **代价与反证：** 完整电源和片外存储成本；原文未取得，不引用其数字。

- **迁移方式：** 只作为证据完整度参照。

- **结论：** 不作为冻结主机制


#### 13.M12 · MEDU-OPSTW

定位：L177–186。N/F/H/T：**1 / 0 / 4 / 1**。

- **原机制：** 运动事件检测关闭成像与DNN，自适应帧分辨率。

- **最近先验与访问状态：** [A 0.82 µW CIS-Based Action Recognition SoC With Self-Adjustable Frame Resolution for Always-on IoT Devices, TCAS-II 2021](https://doi.org/10.1109/TCSII.2021.3067151)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 冻结输入事件体素且输出分辨率固定。

- **真正增量：** 已有低功耗成像策略；应用标签不构成新增。

- **代价与反证：** 删帧/降分辨率需AEE和系统任务合同；65nm仿真不是我们的实测。

- **迁移方式：** 当前排除，仅AoV系统研究参考。

- **结论：** 不作为冻结主机制


#### 13.M13 · SRAM-Resident-MFBD

定位：L188–197。N/F/H/T：**1 / 6 / 7 / 1**。

- **原机制：** 权重、程序和工作集驻留片上SRAM，减少DRAM。

- **最近先验与访问状态：** [Tesla FSD, Bannon/Venkataramanan, Hot Chips 31 2019 official slides](https://old.hotchips.org/hc31/HC31_2.3_Tesla_Hotchips_ppt_Final_0817.pdf)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 驻留原则适配；容量、宏和工作集必须真实。

- **真正增量：** 驻留字典/唤醒图不是新机制。

- **代价与反证：** 不能免费足量SRAM或忽略载入；无单芯片顶层可谈。

- **迁移方式：** 作为同容量数据移动基线。

- **结论：** 不作为冻结主机制


#### 13.M14 · CubeVec-HBG Map

定位：L199–208。N/F/H/T：**1 / 3 / 6 / 1**。

- **原机制：** 矩阵/向量/标量核异构分工。

- **最近先验与访问状态：** [DaVinci, Heng Liao et al., Hot Chips 31 2019 official slides](https://old.hotchips.org/hc31/HC31_1.11_Huawei.Davinci.HengLiao_v4.0.pdf)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 二值源不需该实值payload cube支路；其他真实算子可映射。

- **真正增量：** 成熟处理器分工，硬件划分不等于新原语。

- **代价与反证：** 桥接/存储/调度成本；不可写成现有SoC。

- **迁移方式：** 常规架构对照。

- **结论：** 不作为冻结主机制


#### 13.M15 · CartProd-TTB

定位：L210–219。N/F/H/T：**1 / 5 / 6 / 1**。

- **原机制：** 压缩激活/权重做Cartesian product并分散累加。

- **最近先验与访问状态：** [SCNN, Parashar et al., ISCA 2017](https://arxiv.org/abs/1708.04485)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 二值源可映射，但冻结权重不是已剪枝；当前K8已有读共享。

- **真正增量：** 稀疏乘积分发已有；Motion-TTB命名不新增。

- **代价与反证：** scatter网络/Acc端口、稀疏metadata、权重剪枝需新AEE。

- **迁移方式：** 强稀疏基线；不可重复记TSBG取权收益。

- **结论：** 不作为冻结主机制


#### 13.M16 · PRRC-BitFuse / Stripes / BitFusion / UNPU / LNPU

定位：L221–234。N/F/H/T：**2 / 2 / 6 / 2**。

- **原机制：** 激活与权重逐位跳零，空闲 slot donation。

- **最近先验与访问状态：** [BitFusion MICRO 2018; FireFly-v2](https://arxiv.org/abs/1712.01507)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。 激活只有1位，双侧幅值位稀疏前提消失。

- **真正增量：** bit-serial/slot donation 已有；不能重新命名为光流原语。

- **代价与反证：** 零位译码、移位、交换网、符号扩展与最坏精度时间都收费。

- **迁移方式：** 只保留权重位串行作为同资源对照；无新意暂不重做。

- **结论：** 不作主机制


#### 13.M17 · RS-MW-Reuse / DVAFS-PRRC / Thinker-Split-SDSA / DNPU

定位：L236–249。N/F/H/T：**1 / 8 / 8 / 1**。

- **原机制：** 保持权重或累加器驻留并铺排时空维。

- **最近先验与访问状态：** [Eyeriss, JSSC 2017; Lee & Li ICCD 2020](https://doi.org/10.1109/JSSC.2016.2616357)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** 冻结 PSN 为完整 T×T 线性映射（T=2/10），不是按漏电、复位递推的 LIF。 完整时间维铺排可合法实现。

- **真正增量：** 映射选择，不是新计算语义。

- **代价与反证：** 至少保留全 T 输入/中间向量；不能免费无状态。

- **迁移方式：** 纳入硬件基线，不作为标题贡献。

- **结论：** 仅工程


#### 13.M18 · StripTile-OFPipe

定位：L251–260。N/F/H/T：**2 / 3 / 6 / 2**。

- **原机制：** ISP/卷积按strip tile直连，减少中间DRAM。

- **最近先验与访问状态：** [An AI-ISP strip-tile accelerator, TCSVT 2024](https://doi.org/10.1109/TCSVT.2024.3510939)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 无APS分支；动态BN必须全域统计栅栏，不能全图直接strip流完。

- **真正增量：** 局部层融合已有；跨BN突破必须改变统计生成而非普通寄存器。

- **代价与反证：** halo、中间存储、跨strip统计/replay与接口成本。

- **迁移方式：** 仅局部线性层融合基线；V2统计先行可解决不同问题。

- **结论：** 不作为冻结主机制


#### 13.M19 · BlkSkip-CIM-Wake

定位：L262–271。N/F/H/T：**1 / 0 / 1 / 0**。

- **原机制：** 组相联块零跳过，ping-pong写权重隐藏写入。

- **最近先验与访问状态：** [Yue et al., ISSCC 2021](https://doi.org/10.1109/ISSCC42613.2021.9365958)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 定制CIM不可用；静态W无所述动态假设字典写入负载。

- **真正增量：** 去掉CIM只剩普通块零跳过。

- **代价与反证：** 宏接口、电荷域、双缓冲容量不得免费。

- **迁移方式：** 不迁移。

- **结论：** 不作为冻结主机制


#### 13.M20 · Marsellus / Card-H host orchestration

定位：L273–277。N/F/H/T：**0 / 5 / 6 / 0**。

- **原机制：** 主机顺序器协调DMA、多核、唤醒与实际能量。

- **最近先验与访问状态：** [Marsellus, PULP project primary paper（原文件）](https://pulp-platform.org/)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 作为报告功耗/系统边界纪律可用，非新执行原语。

- **真正增量：** 必须量化控制能量；无本岛创新增量。

- **代价与反证：** 没有测量顶层；主机面积/电源域不免费。

- **迁移方式：** 引用方法纪律，不作为标题或新岛。

- **结论：** 不作为冻结主机制


#### 13.card.1 · Entropy-EE-OF

定位：L285。N/F/H/T：**2 / 0 / 4 / 1**。

- **原机制：** 按置信度提前退出、预测精度并调 V/F。

- **最近先验与访问状态：** [Tambe et al., ISSCC 2023 Transformer accelerator; EdgeBERT MICRO 2021](https://sld.cs.columbia.edu/pubs/tambe_isscc23.pdf)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 冻结无多退出头，不能按分类熵直接判断每像素 flow 正确。

- **真正增量：** 置信度策略改应用已有；无法替代严格消费者判定。

- **代价与反证：** 新增训练/AEE、精度切换和物理电压域表征；早停 PSN 时间步错误。

- **迁移方式：** 从原包剔除；若另模型授权才做完整精度-能量 Pareto。

- **结论：** 不作为冻结主机制


#### 13.card.2 · MP-Pred-PRRC

定位：L286。N/F/H/T：**2 / 0 / 4 / 1**。

- **原机制：** 按置信度提前退出、预测精度并调 V/F。

- **最近先验与访问状态：** [Tambe et al., ISSCC 2023 Transformer accelerator; EdgeBERT MICRO 2021](https://sld.cs.columbia.edu/pubs/tambe_isscc23.pdf)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 冻结无多退出头，不能按分类熵直接判断每像素 flow 正确。

- **真正增量：** 置信度策略改应用已有；无法替代严格消费者判定。

- **代价与反证：** 新增训练/AEE、精度切换和物理电压域表征；早停 PSN 时间步错误。

- **迁移方式：** 从原包剔除；若另模型授权才做完整精度-能量 Pareto。

- **结论：** 不作为冻结主机制


#### 13.card.3 · TileVFS-Motion

定位：L287。N/F/H/T：**2 / 0 / 4 / 1**。

- **原机制：** 按置信度提前退出、预测精度并调 V/F。

- **最近先验与访问状态：** [Tambe et al., ISSCC 2023 Transformer accelerator; EdgeBERT MICRO 2021](https://sld.cs.columbia.edu/pubs/tambe_isscc23.pdf)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 冻结无多退出头，不能按分类熵直接判断每像素 flow 正确。

- **真正增量：** 置信度策略改应用已有；无法替代严格消费者判定。

- **代价与反证：** 新增训练/AEE、精度切换和物理电压域表征；早停 PSN 时间步错误。

- **迁移方式：** 从原包剔除；若另模型授权才做完整精度-能量 Pareto。

- **结论：** 不作为冻结主机制


#### 13.card.4 · Asymp-OoO-ECP

定位：L288。N/F/H/T：**3 / 2 / 5 / 3**。

- **原机制：** 近似预筛分数，异步/乱序调度重算。

- **最近先验与访问状态：** [Wang et al., ISSCC 2022; Ayaka JSSC 2024](https://doi.org/10.1109/ISSCC42614.2022.9731686)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 冻结 Motion-XOR 分数公式可调度，但近似筛选没有保真条件。

- **真正增量：** OoO 是常见机制；若有精确剩余分数界则可重构。

- **代价与反证：** 分数复核/队列/结果恢复/银行带宽；不能把 approximate asymptotic 当无损。

- **迁移方式：** 研究 exact bound 前置裁决，需与简单全部三popcount比较，暂不实现。

- **结论：** 不作为冻结主机制


#### 13.card.5 · FMSkip-OPSTW

定位：L289。N/F/H/T：**1 / 6 / 6 / 1**。

- **原机制：** 按 feature-map 零值跳过，动态重配访存端口。

- **最近先验与访问状态：** Samsung mobile DNN NPU, ISSCC / ISCA 2021（原文件）；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 普通零跳过可用；TS1N28 宏仍是1RW，K8带宽不变。

- **真正增量：** 零跳过已有；动态端口需说明物理多路复用的新增服务。

- **代价与反证：** 不能把逻辑端口分配说成新增 SRAM 端口；交叉网、排队与 bank 冲突收费。

- **迁移方式：** 仅作为 bank 调度强基线；无新执行内容不立项。

- **结论：** 不作为冻结主机制


#### 13.card.6 · DynPort-MFBD

定位：L290。N/F/H/T：**1 / 6 / 6 / 1**。

- **原机制：** 按 feature-map 零值跳过，动态重配访存端口。

- **最近先验与访问状态：** Samsung mobile DNN NPU, ISSCC / ISCA 2021（原文件）；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 普通零跳过可用；TS1N28 宏仍是1RW，K8带宽不变。

- **真正增量：** 零跳过已有；动态端口需说明物理多路复用的新增服务。

- **代价与反证：** 不能把逻辑端口分配说成新增 SRAM 端口；交叉网、排队与 bank 冲突收费。

- **迁移方式：** 仅作为 bank 调度强基线；无新执行内容不立项。

- **结论：** 不作为冻结主机制


#### 13.card.7 · HetSchedule-OF

定位：L291。N/F/H/T：**1 / 0 / 2 / 0**。

- **原机制：** 数字/模拟核按精度/密度异构调度。

- **最近先验与访问状态：** [DIANA, ISSCC 2022 / JSSC](https://doi.org/10.1109/ISSCC42614.2022.9731716)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 数字28nm普通宏与模拟核不相符。

- **真正增量：** 去掉模拟后是常规多引擎调度。

- **代价与反证：** 桥接、ADC与精度校准；仅调度没有主机制新意。

- **迁移方式：** 只借服务能量核算；不建异构CIM。

- **结论：** 不作为冻结主机制


#### 13.card.8 · SalCascade-EVWake

定位：L292。N/F/H/T：**2 / 1 / 3 / 1**。

- **原机制：** 显著度→新颖性→DNN逐级唤醒，尽早关闭下游。

- **最近先验与访问状态：** [CogniVision, Gupta/Vohra/Alioto, VLSI 2024](https://doi.org/10.1109/VLSITechnologyandCir46783.2024.10631426)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 缺传感器/SoC与定义好的漏检任务；稠密OF不等于存在检测。

- **真正增量：** 级联wake是原论文已有；换成event energy不够。

- **代价与反证：** 唤醒延迟、电源域、漏检与输出质量；不能从岛推整机mW。

- **迁移方式：** 仅系统级未来方向，不塞入本短文。

- **结论：** 不作为冻结主机制


#### 13.card.9 · AoV-Cascade-CardH

定位：L293。N/F/H/T：**1 / 1 / 3 / 1**。

- **原机制：** 驻片视觉流水与细粒度漏电管理。

- **最近先验与访问状态：** [Alpha-Vision, ISSCC 2026](https://ieeexplore.ieee.org/document/11409322)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 脸检测延迟/功率不能迁移到全分辨率OF；本项目无芯片顶层。

- **真正增量：** 商业/研究AoV系统组织，不是C2执行原语。

- **代价与反证：** 完整电源和片外存储成本；原文未取得，不引用其数字。

- **迁移方式：** 只作为证据完整度参照。

- **结论：** 不作为冻结主机制


#### 13.card.10 · BigLittle-DualRail

定位：L294。N/F/H/T：**2 / 0 / 4 / 1**。

- **原机制：** 轻重双引擎按输入难度选择 dense/implicit-weight 路径。

- **最近先验与访问状态：** [C-Transformer, ISSCC 2024](https://doi.org/10.1109/ISSCC49657.2024.10454330)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 冻结没有实值 ATLIF 双轨或遮挡专用 dense 模型。

- **真正增量：** 通用 big/little 配置；没有被冻结的可切换等价算子。

- **代价与反证：** 第二条模型/精度、选路与两套存储，需新训练评价。

- **迁移方式：** 当前剔除；不要为讲架构添加不存在的数据流。

- **结论：** 不作为冻结主机制


#### 13.card.11 · BLT-CIM-SDSA

定位：L295。N/F/H/T：**2 / 0 / 1 / 0**。

- **原机制：** 定制 CIM 位线转置以降低动态 attention 转置成本。

- **最近先验与访问状态：** [Tu et al., ISSCC 2022](https://doi.org/10.1109/ISSCC42614.2022.9731645)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 普通1RW宏无位线计算/转置操作。

- **真正增量：** 地址转置可借但不能继承定制 CIM 能量与密度。

- **代价与反证：** 缺宏/器件/物理验证；用户禁 CIM。

- **迁移方式：** 剔除；普通 SRAM 转置仅工程。

- **结论：** 不作为冻结主机制


#### 13.card.12 · PVScale-ATLIF

定位：L296。N/F/H/T：**1 / 2 / 7 / 1**。

- **原机制：** 向量/组共享尺度的低位激活权重量化。

- **最近先验与访问状态：** [Keller et al., VLSI 2022 / JSSC 2023](https://doi.org/10.1109/JSSC.2023.3234893)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** ATLIF只有静态θ；部署权重量化另身份。

- **真正增量：** per-vector scale已存在；per-motion组名不产生原语。

- **代价与反证：** scale读写/乘法、饱和、AEE；Q1.7门控与int4载荷不可混同。

- **迁移方式：** 作为量化对照，当前不独立主打。

- **结论：** 不作为冻结主机制


#### 13.card.13 · CartProd-TTB

定位：L297。N/F/H/T：**1 / 5 / 6 / 1**。

- **原机制：** 压缩激活/权重做Cartesian product并分散累加。

- **最近先验与访问状态：** [SCNN, Parashar et al., ISCA 2017](https://arxiv.org/abs/1708.04485)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 二值源可映射，但冻结权重不是已剪枝；当前K8已有读共享。

- **真正增量：** 稀疏乘积分发已有；Motion-TTB命名不新增。

- **代价与反证：** scatter网络/Acc端口、稀疏metadata、权重剪枝需新AEE。

- **迁移方式：** 强稀疏基线；不可重复记TSBG取权收益。

- **结论：** 不作为冻结主机制


#### 13.card.14 · PRRC-BitFuse

定位：L298。N/F/H/T：**2 / 2 / 6 / 2**。

- **原机制：** 激活与权重逐位跳零，空闲 slot donation。

- **最近先验与访问状态：** [BitFusion MICRO 2018; FireFly-v2](https://arxiv.org/abs/1712.01507)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。 激活只有1位，双侧幅值位稀疏前提消失。

- **真正增量：** bit-serial/slot donation 已有；不能重新命名为光流原语。

- **代价与反证：** 零位译码、移位、交换网、符号扩展与最坏精度时间都收费。

- **迁移方式：** 只保留权重位串行作为同资源对照；无新意暂不重做。

- **结论：** 不作主机制


#### 13.card.15 · StripTile-OFPipe

定位：L299。N/F/H/T：**2 / 3 / 6 / 2**。

- **原机制：** ISP/卷积按strip tile直连，减少中间DRAM。

- **最近先验与访问状态：** [An AI-ISP strip-tile accelerator, TCSVT 2024](https://doi.org/10.1109/TCSVT.2024.3510939)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 无APS分支；动态BN必须全域统计栅栏，不能全图直接strip流完。

- **真正增量：** 局部层融合已有；跨BN突破必须改变统计生成而非普通寄存器。

- **代价与反证：** halo、中间存储、跨strip统计/replay与接口成本。

- **迁移方式：** 仅局部线性层融合基线；V2统计先行可解决不同问题。

- **结论：** 不作为冻结主机制


#### 13.card.16 · SRAM-Resident-MFBD

定位：L300。N/F/H/T：**1 / 6 / 7 / 1**。

- **原机制：** 权重、程序和工作集驻留片上SRAM，减少DRAM。

- **最近先验与访问状态：** [Tesla FSD, Bannon/Venkataramanan, Hot Chips 31 2019 official slides](https://old.hotchips.org/hc31/HC31_2.3_Tesla_Hotchips_ppt_Final_0817.pdf)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 驻留原则适配；容量、宏和工作集必须真实。

- **真正增量：** 驻留字典/唤醒图不是新机制。

- **代价与反证：** 不能免费足量SRAM或忽略载入；无单芯片顶层可谈。

- **迁移方式：** 作为同容量数据移动基线。

- **结论：** 不作为冻结主机制


#### 13.card.17 · Retain-Prior-eMRAM

定位：L301。N/F/H/T：**1 / 1 / 3 / 1**。

- **原机制：** 睡眠保持模型/先验，低功耗唤醒恢复状态。

- **最近先验与访问状态：** [TinyVers, VLSI 2022 / JSSC 2023](https://doi.org/10.1109/JSSC.2023.3236566)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 没有可用 eMRAM 宏，也无冻结跨帧先验状态。

- **真正增量：** 常规 retention/always-on；任务名称不新增机制。

- **代价与反证：** 漏电、电源域、掉电一致性和状态有效期；替成SRAM不能继承eMRAM指标。

- **迁移方式：** 仅系统电源方案另任务，当前排除。

- **结论：** 不作为冻结主机制


#### 13.card.18 · BlkSkip-CIM-Wake

定位：L302。N/F/H/T：**1 / 0 / 1 / 0**。

- **原机制：** 组相联块零跳过，ping-pong写权重隐藏写入。

- **最近先验与访问状态：** [Yue et al., ISSCC 2021](https://doi.org/10.1109/ISSCC42613.2021.9365958)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 定制CIM不可用；静态W无所述动态假设字典写入负载。

- **真正增量：** 去掉CIM只剩普通块零跳过。

- **代价与反证：** 宏接口、电荷域、双缓冲容量不得免费。

- **迁移方式：** 不迁移。

- **结论：** 不作为冻结主机制


#### 13.card.19 · EC-Psum-Gate

定位：L303。N/F/H/T：**2 / 3 / 6 / 2**。

- **原机制：** 合并相同有效权重，或用误差补偿预测部分和是否应继续。

- **最近先验与访问状态：** [QNAP, ISSCC 2021 / JSSC 2021; UCNN ISCA 2018](https://doi.org/10.1109/JSSC.2021.3113569)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 权重重复可精确；预测 ReLU 输出不直接等于 full-T PSN + 动态BN。

- **真正增量：** 普通相同码分组已由 UCNN/SumMerge覆盖；预测需新证书/误差合同。

- **代价与反证：** UCNN索引/组和位宽；动态BN统计仍需实际贡献；QNAP原文未取得，不细化未核ECP公式。

- **迁移方式：** 精确共享完整投影+阈值区间是另新模型候选；单码重复不足。

- **结论：** 不作为冻结主机制


#### 13.card.20 · EWC-Cluster-MAC

定位：L304。N/F/H/T：**2 / 3 / 6 / 2**。

- **原机制：** 合并相同有效权重，或用误差补偿预测部分和是否应继续。

- **最近先验与访问状态：** [QNAP, ISSCC 2021 / JSSC 2021; UCNN ISCA 2018](https://doi.org/10.1109/JSSC.2021.3113569)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 权重重复可精确；预测 ReLU 输出不直接等于 full-T PSN + 动态BN。

- **真正增量：** 普通相同码分组已由 UCNN/SumMerge覆盖；预测需新证书/误差合同。

- **代价与反证：** UCNN索引/组和位宽；动态BN统计仍需实际贡献；QNAP原文未取得，不细化未核ECP公式。

- **迁移方式：** 精确共享完整投影+阈值区间是另新模型候选；单码重复不足。

- **结论：** 不作为冻结主机制


#### 13.card.21 · RS-MW-Reuse

定位：L305。N/F/H/T：**1 / 8 / 8 / 1**。

- **原机制：** 保持权重或累加器驻留并铺排时空维。

- **最近先验与访问状态：** [Eyeriss, JSSC 2017; Lee & Li ICCD 2020](https://doi.org/10.1109/JSSC.2016.2616357)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** 冻结 PSN 为完整 T×T 线性映射（T=2/10），不是按漏电、复位递推的 LIF。 完整时间维铺排可合法实现。

- **真正增量：** 映射选择，不是新计算语义。

- **代价与反证：** 至少保留全 T 输入/中间向量；不能免费无状态。

- **迁移方式：** 纳入硬件基线，不作为标题贡献。

- **结论：** 仅工程


#### 13.card.22 · MEDU-OPSTW

定位：L306。N/F/H/T：**1 / 0 / 4 / 1**。

- **原机制：** 运动事件检测关闭成像与DNN，自适应帧分辨率。

- **最近先验与访问状态：** [A 0.82 µW CIS-Based Action Recognition SoC With Self-Adjustable Frame Resolution for Always-on IoT Devices, TCAS-II 2021](https://doi.org/10.1109/TCSII.2021.3067151)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 冻结输入事件体素且输出分辨率固定。

- **真正增量：** 已有低功耗成像策略；应用标签不构成新增。

- **代价与反证：** 删帧/降分辨率需AEE和系统任务合同；65nm仿真不是我们的实测。

- **迁移方式：** 当前排除，仅AoV系统研究参考。

- **结论：** 不作为冻结主机制


#### 13.card.23 · IrrNoC-MotionTTB

定位：L307。N/F/H/T：**1 / 6 / 6 / 1**。

- **原机制：** 按不规则稀疏包自适应路由/均衡。

- **最近先验与访问状态：** [SCNN ISCA 2017; ELSA ISCA 2026; general sparse NoC](https://arxiv.org/abs/1708.04485)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 可维持合法typed consumer，但无新token语义。

- **真正增量：** 普通稀疏路由；运动名词未改变端口服务。

- **代价与反证：** 增crossbar/FIFO/仲裁必收费；固定8bank总入口不增加。

- **迁移方式：** 同资源调度基线，非主机制。

- **结论：** 不作为冻结主机制


#### 13.card.24 · CubeVec-HBG Map

定位：L308。N/F/H/T：**1 / 3 / 6 / 1**。

- **原机制：** 矩阵/向量/标量核异构分工。

- **最近先验与访问状态：** [DaVinci, Heng Liao et al., Hot Chips 31 2019 official slides](https://old.hotchips.org/hc31/HC31_1.11_Huawei.Davinci.HengLiao_v4.0.pdf)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 二值源不需该实值payload cube支路；其他真实算子可映射。

- **真正增量：** 成熟处理器分工，硬件划分不等于新原语。

- **代价与反证：** 桥接/存储/调度成本；不可写成现有SoC。

- **迁移方式：** 常规架构对照。

- **结论：** 不作为冻结主机制


#### 13.rank.1 · EE + MP + VFS

定位：L316。N/F/H/T：**2 / 0 / 4 / 1**。

- **原机制：** 按置信度提前退出、预测精度并调 V/F。

- **最近先验与访问状态：** [Tambe et al., ISSCC 2023 Transformer accelerator; EdgeBERT MICRO 2021](https://sld.cs.columbia.edu/pubs/tambe_isscc23.pdf)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 冻结无多退出头，不能按分类熵直接判断每像素 flow 正确。

- **真正增量：** 置信度策略改应用已有；无法替代严格消费者判定。

- **代价与反证：** 新增训练/AEE、精度切换和物理电压域表征；早停 PSN 时间步错误。

- **迁移方式：** 从原包剔除；若另模型授权才做完整精度-能量 Pareto。

- **结论：** 不作为冻结主机制


#### 13.rank.2 · SalCascade / AoV

定位：L317。N/F/H/T：**2 / 1 / 3 / 1**。

- **原机制：** 显著度→新颖性→DNN逐级唤醒，尽早关闭下游。

- **最近先验与访问状态：** [CogniVision, Gupta/Vohra/Alioto, VLSI 2024](https://doi.org/10.1109/VLSITechnologyandCir46783.2024.10631426)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 缺传感器/SoC与定义好的漏检任务；稠密OF不等于存在检测。

- **真正增量：** 级联wake是原论文已有；换成event energy不够。

- **代价与反证：** 唤醒延迟、电源域、漏检与输出质量；不能从岛推整机mW。

- **迁移方式：** 仅系统级未来方向，不塞入本短文。

- **结论：** 不作为冻结主机制


#### 13.rank.3 · Asymp

定位：L318。N/F/H/T：**3 / 2 / 5 / 3**。

- **原机制：** 近似预筛分数，异步/乱序调度重算。

- **最近先验与访问状态：** [Wang et al., ISSCC 2022; Ayaka JSSC 2024](https://doi.org/10.1109/ISSCC42614.2022.9731686)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 冻结 Motion-XOR 分数公式可调度，但近似筛选没有保真条件。

- **真正增量：** OoO 是常见机制；若有精确剩余分数界则可重构。

- **代价与反证：** 分数复核/队列/结果恢复/银行带宽；不能把 approximate asymptotic 当无损。

- **迁移方式：** 研究 exact bound 前置裁决，需与简单全部三popcount比较，暂不实现。

- **结论：** 不作为冻结主机制


#### 13.rank.4 · HetSchedule

定位：L319。N/F/H/T：**1 / 0 / 2 / 0**。

- **原机制：** 数字/模拟核按精度/密度异构调度。

- **最近先验与访问状态：** [DIANA, ISSCC 2022 / JSSC](https://doi.org/10.1109/ISSCC42614.2022.9731716)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 数字28nm普通宏与模拟核不相符。

- **真正增量：** 去掉模拟后是常规多引擎调度。

- **代价与反证：** 桥接、ADC与精度校准；仅调度没有主机制新意。

- **迁移方式：** 只借服务能量核算；不建异构CIM。

- **结论：** 不作为冻结主机制


#### 13.rank.5 · DynPort + FMSkip

定位：L320。N/F/H/T：**1 / 6 / 6 / 1**。

- **原机制：** 按 feature-map 零值跳过，动态重配访存端口。

- **最近先验与访问状态：** Samsung mobile DNN NPU, ISSCC / ISCA 2021（原文件）；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 普通零跳过可用；TS1N28 宏仍是1RW，K8带宽不变。

- **真正增量：** 零跳过已有；动态端口需说明物理多路复用的新增服务。

- **代价与反证：** 不能把逻辑端口分配说成新增 SRAM 端口；交叉网、排队与 bank 冲突收费。

- **迁移方式：** 仅作为 bank 调度强基线；无新执行内容不立项。

- **结论：** 不作为冻结主机制


#### 13.rank.6 · BigLittle

定位：L321。N/F/H/T：**2 / 0 / 4 / 1**。

- **原机制：** 轻重双引擎按输入难度选择 dense/implicit-weight 路径。

- **最近先验与访问状态：** [C-Transformer, ISSCC 2024](https://doi.org/10.1109/ISSCC49657.2024.10454330)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 冻结没有实值 ATLIF 双轨或遮挡专用 dense 模型。

- **真正增量：** 通用 big/little 配置；没有被冻结的可切换等价算子。

- **代价与反证：** 第二条模型/精度、选路与两套存储，需新训练评价。

- **迁移方式：** 当前剔除；不要为讲架构添加不存在的数据流。

- **结论：** 不作为冻结主机制


#### 13.rank.7 · BLT-CIM

定位：L322。N/F/H/T：**2 / 0 / 1 / 0**。

- **原机制：** 定制 CIM 位线转置以降低动态 attention 转置成本。

- **最近先验与访问状态：** [Tu et al., ISSCC 2022](https://doi.org/10.1109/ISSCC42614.2022.9731645)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 普通1RW宏无位线计算/转置操作。

- **真正增量：** 地址转置可借但不能继承定制 CIM 能量与密度。

- **代价与反证：** 缺宏/器件/物理验证；用户禁 CIM。

- **迁移方式：** 剔除；普通 SRAM 转置仅工程。

- **结论：** 不作为冻结主机制


#### 13.rank.8 · StripTile + SRAM-resident

定位：L323。N/F/H/T：**2 / 3 / 6 / 2**。

- **原机制：** ISP/卷积按strip tile直连，减少中间DRAM。

- **最近先验与访问状态：** [An AI-ISP strip-tile accelerator, TCSVT 2024](https://doi.org/10.1109/TCSVT.2024.3510939)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 无APS分支；动态BN必须全域统计栅栏，不能全图直接strip流完。

- **真正增量：** 局部层融合已有；跨BN突破必须改变统计生成而非普通寄存器。

- **代价与反证：** halo、中间存储、跨strip统计/replay与接口成本。

- **迁移方式：** 仅局部线性层融合基线；V2统计先行可解决不同问题。

- **结论：** 不作为冻结主机制


#### 13.rank.9 · CartProd + PVScale + BitFuse

定位：L324。N/F/H/T：**1 / 5 / 6 / 1**。

- **原机制：** 压缩激活/权重做Cartesian product并分散累加。

- **最近先验与访问状态：** [SCNN, Parashar et al., ISCA 2017](https://arxiv.org/abs/1708.04485)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 二值源可映射，但冻结权重不是已剪枝；当前K8已有读共享。

- **真正增量：** 稀疏乘积分发已有；Motion-TTB命名不新增。

- **代价与反证：** scatter网络/Acc端口、稀疏metadata、权重剪枝需新AEE。

- **迁移方式：** 强稀疏基线；不可重复记TSBG取权收益。

- **结论：** 不作为冻结主机制


#### 13.rank.10 · EC-Psum / EWC

定位：L325。N/F/H/T：**2 / 3 / 6 / 2**。

- **原机制：** 合并相同有效权重，或用误差补偿预测部分和是否应继续。

- **最近先验与访问状态：** [QNAP, ISSCC 2021 / JSSC 2021; UCNN ISCA 2018](https://doi.org/10.1109/JSSC.2021.3113569)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 权重重复可精确；预测 ReLU 输出不直接等于 full-T PSN + 动态BN。

- **真正增量：** 普通相同码分组已由 UCNN/SumMerge覆盖；预测需新证书/误差合同。

- **代价与反证：** UCNN索引/组和位宽；动态BN统计仍需实际贡献；QNAP原文未取得，不细化未核ECP公式。

- **迁移方式：** 精确共享完整投影+阈值区间是另新模型候选；单码重复不足。

- **结论：** 不作为冻结主机制


#### 13.upgrade.1 · OP-STW upgrades

定位：L333。N/F/H/T：**3 / 1 / 5 / 2**。

- **原机制：** 用上一时刻 coarse flow/差分和事件计数预测 tile wake，睡眠 tile 复用输出。

- **最近先验与访问状态：** [MotionDeltaCNN ICCV 2023; motion-gated vision SoCs](https://openaccess.thecvf.com/content/ICCV2023/html/Parger_MotionDeltaCNN_Sparse_CNN_Inference_of_Frame_Differences_in_Moving_Camera_ICCV_2023_paper.html)；CVF 一手摘要和 PDF 方法段已读

- **冻结适配：** H67 无现成 coarse-flow 迭代支路；no_running 动态 BN 的全域均值/方差使局部不变不能直接推出消费者不变。

- **真正增量：** 加 motion 名称不足；若以严格消费者证书取代启发式 wake 才形成新执行问题。

- **代价与反证：** 当前帧 flow 不能提前免费获得；睡眠输出未必精确，必须计预测器、warp、缓存与 AEE。

- **迁移方式：** 原版只可有损新评价；精确重构必须全域 BN 依赖闭合。

- **结论：** 原版不作为冻结主线


#### 13.upgrade.2 · ECP-QKV upgrades

定位：L334。N/F/H/T：**3 / 2 / 5 / 3**。

- **原机制：** 近似预筛分数，异步/乱序调度重算。

- **最近先验与访问状态：** [Wang et al., ISSCC 2022; Ayaka JSSC 2024](https://doi.org/10.1109/ISSCC42614.2022.9731686)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 冻结 Motion-XOR 分数公式可调度，但近似筛选没有保真条件。

- **真正增量：** OoO 是常见机制；若有精确剩余分数界则可重构。

- **代价与反证：** 分数复核/队列/结果恢复/银行带宽；不能把 approximate asymptotic 当无损。

- **迁移方式：** 研究 exact bound 前置裁决，需与简单全部三popcount比较，暂不实现。

- **结论：** 不作为冻结主机制


#### 13.upgrade.3 · PRRC / HeatFlow upgrades

定位：L335。N/F/H/T：**2 / 0 / 4 / 1**。

- **原机制：** 按置信度提前退出、预测精度并调 V/F。

- **最近先验与访问状态：** [Tambe et al., ISSCC 2023 Transformer accelerator; EdgeBERT MICRO 2021](https://sld.cs.columbia.edu/pubs/tambe_isscc23.pdf)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 冻结无多退出头，不能按分类熵直接判断每像素 flow 正确。

- **真正增量：** 置信度策略改应用已有；无法替代严格消费者判定。

- **代价与反证：** 新增训练/AEE、精度切换和物理电压域表征；早停 PSN 时间步错误。

- **迁移方式：** 从原包剔除；若另模型授权才做完整精度-能量 Pareto。

- **结论：** 不作为冻结主机制


#### 13.upgrade.4 · MW-ΔBuf upgrades

定位：L336。N/F/H/T：**2 / 3 / 6 / 2**。

- **原机制：** ISP/卷积按strip tile直连，减少中间DRAM。

- **最近先验与访问状态：** [An AI-ISP strip-tile accelerator, TCSVT 2024](https://doi.org/10.1109/TCSVT.2024.3510939)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 无APS分支；动态BN必须全域统计栅栏，不能全图直接strip流完。

- **真正增量：** 局部层融合已有；跨BN突破必须改变统计生成而非普通寄存器。

- **代价与反证：** halo、中间存储、跨strip统计/replay与接口成本。

- **迁移方式：** 仅局部线性层融合基线；V2统计先行可解决不同问题。

- **结论：** 不作为冻结主机制


#### 13.upgrade.5 · EV-Wake / TDE3 upgrades

定位：L337。N/F/H/T：**2 / 1 / 3 / 1**。

- **原机制：** 显著度→新颖性→DNN逐级唤醒，尽早关闭下游。

- **最近先验与访问状态：** [CogniVision, Gupta/Vohra/Alioto, VLSI 2024](https://doi.org/10.1109/VLSITechnologyandCir46783.2024.10631426)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 缺传感器/SoC与定义好的漏检任务；稠密OF不等于存在检测。

- **真正增量：** 级联wake是原论文已有；换成event energy不够。

- **代价与反证：** 唤醒延迟、电源域、漏检与输出质量；不能从岛推整机mW。

- **迁移方式：** 仅系统级未来方向，不塞入本短文。

- **结论：** 不作为冻结主机制


#### 13.upgrade.6 · HBG/SMAM upgrades

定位：L338。N/F/H/T：**2 / 0 / 4 / 1**。

- **原机制：** 轻重双引擎按输入难度选择 dense/implicit-weight 路径。

- **最近先验与访问状态：** [C-Transformer, ISSCC 2024](https://doi.org/10.1109/ISSCC49657.2024.10454330)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 冻结没有实值 ATLIF 双轨或遮挡专用 dense 模型。

- **真正增量：** 通用 big/little 配置；没有被冻结的可切换等价算子。

- **代价与反证：** 第二条模型/精度、选路与两套存储，需新训练评价。

- **迁移方式：** 当前剔除；不要为讲架构添加不存在的数据流。

- **结论：** 不作为冻结主机制


#### 13.upgrade.7 · ADP/ARM upgrades

定位：L339。N/F/H/T：**2 / 3 / 6 / 2**。

- **原机制：** 合并相同有效权重，或用误差补偿预测部分和是否应继续。

- **最近先验与访问状态：** [QNAP, ISSCC 2021 / JSSC 2021; UCNN ISCA 2018](https://doi.org/10.1109/JSSC.2021.3113569)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 权重重复可精确；预测 ReLU 输出不直接等于 full-T PSN + 动态BN。

- **真正增量：** 普通相同码分组已由 UCNN/SumMerge覆盖；预测需新证书/误差合同。

- **代价与反证：** UCNN索引/组和位宽；动态BN统计仍需实际贡献；QNAP原文未取得，不细化未核ECP公式。

- **迁移方式：** 精确共享完整投影+阈值区间是另新模型候选；单码重复不足。

- **结论：** 不作为冻结主机制


#### 13.upgrade.8 · MFBD/Motion-TTB upgrades

定位：L340。N/F/H/T：**1 / 6 / 6 / 1**。

- **原机制：** 按 feature-map 零值跳过，动态重配访存端口。

- **最近先验与访问状态：** Samsung mobile DNN NPU, ISSCC / ISCA 2021（原文件）；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 普通零跳过可用；TS1N28 宏仍是1RW，K8带宽不变。

- **真正增量：** 零跳过已有；动态端口需说明物理多路复用的新增服务。

- **代价与反证：** 不能把逻辑端口分配说成新增 SRAM 端口；交叉网、排队与 bank 冲突收费。

- **迁移方式：** 仅作为 bank 调度强基线；无新执行内容不立项。

- **结论：** 不作为冻结主机制


#### 13.upgrade.9 · HetOF/DualRailCIM upgrades

定位：L341。N/F/H/T：**1 / 0 / 2 / 0**。

- **原机制：** 数字/模拟核按精度/密度异构调度。

- **最近先验与访问状态：** [DIANA, ISSCC 2022 / JSSC](https://doi.org/10.1109/ISSCC42614.2022.9731716)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 数字28nm普通宏与模拟核不相符。

- **真正增量：** 去掉模拟后是常规多引擎调度。

- **代价与反证：** 桥接、ADC与精度校准；仅调度没有主机制新意。

- **迁移方式：** 只借服务能量核算；不建异构CIM。

- **结论：** 不作为冻结主机制


#### 13.upgrade.10 · SP-Gate/STH upgrades

定位：L342。N/F/H/T：**2 / 1 / 6 / 2**。

- **原机制：** 按 attention mass 选择 hot token 并抑制 cold token 的后续 FC 计算。

- **最近先验与访问状态：** [SparseVideoGen; token pruning / early-exit](https://arxiv.org/abs/2502.01776)；原文件给出来源；本轮未独立取得全文

- **冻结适配：** 冻结 attention 的 gate 不是 token 整体无贡献证明；FC/BN/残差仍有消费者。

- **真正增量：** 普通分数剪枝；没有精确停止证书。

- **代价与反证：** 剪 token 改动态 BN 统计及稠密光流输出；分数读取/排序成本。

- **迁移方式：** 仅新 AEE Pareto；精确版必须完整依赖界。

- **结论：** 冻结无损路线否决


#### 13.CardH.1 · SalCascade FSM

定位：L353。N/F/H/T：**2 / 1 / 3 / 1**。

- **原机制：** 显著度→新颖性→DNN逐级唤醒，尽早关闭下游。

- **最近先验与访问状态：** [CogniVision, Gupta/Vohra/Alioto, VLSI 2024](https://doi.org/10.1109/VLSITechnologyandCir46783.2024.10631426)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 缺传感器/SoC与定义好的漏检任务；稠密OF不等于存在检测。

- **真正增量：** 级联wake是原论文已有；换成event energy不够。

- **代价与反证：** 唤醒延迟、电源域、漏检与输出质量；不能从岛推整机mW。

- **迁移方式：** 仅系统级未来方向，不塞入本短文。

- **结论：** 不作为冻结主机制


#### 13.CardH.2 · EE / VFS policy

定位：L354。N/F/H/T：**2 / 0 / 4 / 1**。

- **原机制：** 按置信度提前退出、预测精度并调 V/F。

- **最近先验与访问状态：** [Tambe et al., ISSCC 2023 Transformer accelerator; EdgeBERT MICRO 2021](https://sld.cs.columbia.edu/pubs/tambe_isscc23.pdf)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 冻结无多退出头，不能按分类熵直接判断每像素 flow 正确。

- **真正增量：** 置信度策略改应用已有；无法替代严格消费者判定。

- **代价与反证：** 新增训练/AEE、精度切换和物理电压域表征；早停 PSN 时间步错误。

- **迁移方式：** 从原包剔除；若另模型授权才做完整精度-能量 Pareto。

- **结论：** 不作为冻结主机制


#### 13.CardH.3 · heterogeneous dispatcher

定位：L355。N/F/H/T：**1 / 0 / 2 / 0**。

- **原机制：** 数字/模拟核按精度/密度异构调度。

- **最近先验与访问状态：** [DIANA, ISSCC 2022 / JSSC](https://doi.org/10.1109/ISSCC42614.2022.9731716)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 数字28nm普通宏与模拟核不相符。

- **真正增量：** 去掉模拟后是常规多引擎调度。

- **代价与反证：** 桥接、ADC与精度校准；仅调度没有主机制新意。

- **迁移方式：** 只借服务能量核算；不建异构CIM。

- **结论：** 不作为冻结主机制


#### 13.CardH.4 · prior retention

定位：L356。N/F/H/T：**1 / 1 / 3 / 1**。

- **原机制：** 睡眠保持模型/先验，低功耗唤醒恢复状态。

- **最近先验与访问状态：** [TinyVers, VLSI 2022 / JSSC 2023](https://doi.org/10.1109/JSSC.2023.3236566)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 没有可用 eMRAM 宏，也无冻结跨帧先验状态。

- **真正增量：** 常规 retention/always-on；任务名称不新增机制。

- **代价与反证：** 漏电、电源域、掉电一致性和状态有效期；替成SRAM不能继承eMRAM指标。

- **迁移方式：** 仅系统电源方案另任务，当前排除。

- **结论：** 不作为冻结主机制


#### 13.CardH.5 · ISP/event DMA strip interface

定位：L357。N/F/H/T：**2 / 3 / 6 / 2**。

- **原机制：** ISP/卷积按strip tile直连，减少中间DRAM。

- **最近先验与访问状态：** [An AI-ISP strip-tile accelerator, TCSVT 2024](https://doi.org/10.1109/TCSVT.2024.3510939)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 无APS分支；动态BN必须全域统计栅栏，不能全图直接strip流完。

- **真正增量：** 局部层融合已有；跨BN突破必须改变统计生成而非普通寄存器。

- **代价与反证：** halo、中间存储、跨strip统计/replay与接口成本。

- **迁移方式：** 仅局部线性层融合基线；V2统计先行可解决不同问题。

- **结论：** 不作为冻结主机制


#### 13.CardH.6 · host sequencer

定位：L358。N/F/H/T：**0 / 5 / 6 / 0**。

- **原机制：** 主机顺序器协调DMA、多核、唤醒与实际能量。

- **最近先验与访问状态：** [Marsellus, PULP project primary paper（原文件）](https://pulp-platform.org/)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 作为报告功耗/系统边界纪律可用，非新执行原语。

- **真正增量：** 必须量化控制能量；无本岛创新增量。

- **代价与反证：** 没有测量顶层；主机面积/电源域不免费。

- **迁移方式：** 引用方法纪律，不作为标题或新岛。

- **结论：** 不作为冻结主机制


### research/14_ROUND4_SYNTHESIS_EDGE_NPU.md

精读完成；SHA256 `9ab54c982849682e73eeaa6d151382bb05289b5815604cb80da145ae27e3bb44`。


#### 14.transfer.1 · Entropy-EE / TileVFS

定位：L10。N/F/H/T：**2 / 0 / 4 / 1**。

- **原机制：** 按置信度提前退出、预测精度并调 V/F。

- **最近先验与访问状态：** [Tambe et al., ISSCC 2023 Transformer accelerator; EdgeBERT MICRO 2021](https://sld.cs.columbia.edu/pubs/tambe_isscc23.pdf)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 冻结无多退出头，不能按分类熵直接判断每像素 flow 正确。

- **真正增量：** 置信度策略改应用已有；无法替代严格消费者判定。

- **代价与反证：** 新增训练/AEE、精度切换和物理电压域表征；早停 PSN 时间步错误。

- **迁移方式：** 从原包剔除；若另模型授权才做完整精度-能量 Pareto。

- **结论：** 不作为冻结主机制


#### 14.transfer.2 · MP-Pred-PRRC

定位：L11。N/F/H/T：**2 / 0 / 4 / 1**。

- **原机制：** 按置信度提前退出、预测精度并调 V/F。

- **最近先验与访问状态：** [Tambe et al., ISSCC 2023 Transformer accelerator; EdgeBERT MICRO 2021](https://sld.cs.columbia.edu/pubs/tambe_isscc23.pdf)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 冻结无多退出头，不能按分类熵直接判断每像素 flow 正确。

- **真正增量：** 置信度策略改应用已有；无法替代严格消费者判定。

- **代价与反证：** 新增训练/AEE、精度切换和物理电压域表征；早停 PSN 时间步错误。

- **迁移方式：** 从原包剔除；若另模型授权才做完整精度-能量 Pareto。

- **结论：** 不作为冻结主机制


#### 14.transfer.3 · SalCascade / AoV

定位：L12。N/F/H/T：**2 / 1 / 3 / 1**。

- **原机制：** 显著度→新颖性→DNN逐级唤醒，尽早关闭下游。

- **最近先验与访问状态：** [CogniVision, Gupta/Vohra/Alioto, VLSI 2024](https://doi.org/10.1109/VLSITechnologyandCir46783.2024.10631426)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 缺传感器/SoC与定义好的漏检任务；稠密OF不等于存在检测。

- **真正增量：** 级联wake是原论文已有；换成event energy不够。

- **代价与反证：** 唤醒延迟、电源域、漏检与输出质量；不能从岛推整机mW。

- **迁移方式：** 仅系统级未来方向，不塞入本短文。

- **结论：** 不作为冻结主机制


#### 14.transfer.4 · Asymp-OoO / FMSkip

定位：L13。N/F/H/T：**3 / 2 / 5 / 3**。

- **原机制：** 近似预筛分数，异步/乱序调度重算。

- **最近先验与访问状态：** [Wang et al., ISSCC 2022; Ayaka JSSC 2024](https://doi.org/10.1109/ISSCC42614.2022.9731686)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 冻结 Motion-XOR 分数公式可调度，但近似筛选没有保真条件。

- **真正增量：** OoO 是常见机制；若有精确剩余分数界则可重构。

- **代价与反证：** 分数复核/队列/结果恢复/银行带宽；不能把 approximate asymptotic 当无损。

- **迁移方式：** 研究 exact bound 前置裁决，需与简单全部三popcount比较，暂不实现。

- **结论：** 不作为冻结主机制


#### 14.transfer.5 · HetSchedule / DynPort

定位：L14。N/F/H/T：**1 / 6 / 6 / 1**。

- **原机制：** 按 feature-map 零值跳过，动态重配访存端口。

- **最近先验与访问状态：** Samsung mobile DNN NPU, ISSCC / ISCA 2021（原文件）；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 普通零跳过可用；TS1N28 宏仍是1RW，K8带宽不变。

- **真正增量：** 零跳过已有；动态端口需说明物理多路复用的新增服务。

- **代价与反证：** 不能把逻辑端口分配说成新增 SRAM 端口；交叉网、排队与 bank 冲突收费。

- **迁移方式：** 仅作为 bank 调度强基线；无新执行内容不立项。

- **结论：** 不作为冻结主机制


#### 14.transfer.6 · BigLittle / BLT-CIM

定位：L15。N/F/H/T：**2 / 0 / 4 / 1**。

- **原机制：** 轻重双引擎按输入难度选择 dense/implicit-weight 路径。

- **最近先验与访问状态：** [C-Transformer, ISSCC 2024](https://doi.org/10.1109/ISSCC49657.2024.10454330)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 冻结没有实值 ATLIF 双轨或遮挡专用 dense 模型。

- **真正增量：** 通用 big/little 配置；没有被冻结的可切换等价算子。

- **代价与反证：** 第二条模型/精度、选路与两套存储，需新训练评价。

- **迁移方式：** 当前剔除；不要为讲架构添加不存在的数据流。

- **结论：** 不作为冻结主机制


#### 14.CardH · Card H always-on orchestrator

定位：L17–18。N/F/H/T：**0 / 5 / 6 / 0**。

- **原机制：** 主机顺序器协调DMA、多核、唤醒与实际能量。

- **最近先验与访问状态：** [Marsellus, PULP project primary paper（原文件）](https://pulp-platform.org/)；原文件的一手论文/官方演示链接；本轮未取得全文，原性能数字不准入

- **冻结适配：** 作为报告功耗/系统边界纪律可用，非新执行原语。

- **真正增量：** 必须量化控制能量；无本岛创新增量。

- **代价与反证：** 没有测量顶层；主机面积/电源域不免费。

- **迁移方式：** 引用方法纪律，不作为标题或新岛。

- **结论：** 不作为冻结主机制


#### 14.order · 先 Card A + Card B

定位：L23–24。N/F/H/T：**1 / 0 / 9 / 1**。

- **原机制：** 把幅值拆为非零门控 g 与 int8 实值 p，g 控制时钟/请求。

- **最近先验与访问状态：** [Activity Pruning for Efficient Spiking Neural Networks, Bu/Shi/Yu, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/fa12d67b5939c37ea8a4659c31a08d2c-Paper-Conference.pdf)；一手 PDF §3.2 式10–11及阈值吸收段已核；OpenReview 被拦后由 NeurIPS 正式库取得

- **冻结适配：** ep34 自然 ATLIF 为 {0,θ}，θ 静态可并入 W；没有每事件实值 payload。

- **真正增量：** 冻结身份下退化为已有 active 位与权重加法；没有新增幅值原语。

- **代价与反证：** 排序把失配的 HBG 置于第一阶段；历史‘用户锁/立即开干’不能覆盖当前冻结合同与审查任务。

- **迁移方式：** 冻结版本删除 p，只作为普通二值协议/门控基线；实值版本需重训并另建身份。

- **结论：** 否决为主机制；禁止套 ep34


### research/17_OPENROAD_PNR_SCOREBOARD_GROKBOT.md

精读完成；SHA256 `16cd7816f5aaae97527e4f15a6d39f5e78d5af8a8e52678ddb292c4941f19ec0`。

全文57行，逐项覆盖18行；未把表中文字当独立验证原始DEF/DRC/STA通过。


#### ROOT-PNR-01 · HBG-RP

定位：表第1行（正文第24行附近）。N/F/H/T：**— / — / — / —**。

- **原机制：** 隔离sky130hd模块/包装器综合布局布线演示

- **最近先验与访问状态：** 对应原始机制记录；本条为入口/工程证据；本地全文

- **冻结适配：** 以二值 ep34/no_running BN/full-rank PSN 为准

- **真正增量：** 存在可布线骨架不增加算法或电路机制新意

- **代价与反证：** 该文件声明无PDN/RCX/SPEF/多角签核；不属于28nm宏与同负载闭环，也没有功能差分覆盖证明

- **迁移方式：** 仅隔离可综合性/后端探索记录；机制分见相应卡及microarch；不得当本项目ASIC PPA或录用证据

- **结论：** 工程演示，创新评分不适用；PPA_ADMISSION=0


#### ROOT-PNR-02 · OGEC

定位：表第2行（正文第25行附近）。N/F/H/T：**— / — / — / —**。

- **原机制：** 隔离sky130hd模块/包装器综合布局布线演示

- **最近先验与访问状态：** 对应原始机制记录；本条为入口/工程证据；本地全文

- **冻结适配：** 以二值 ep34/no_running BN/full-rank PSN 为准

- **真正增量：** 存在可布线骨架不增加算法或电路机制新意

- **代价与反证：** 该文件声明无PDN/RCX/SPEF/多角签核；不属于28nm宏与同负载闭环，也没有功能差分覆盖证明

- **迁移方式：** 仅隔离可综合性/后端探索记录；机制分见相应卡及microarch；不得当本项目ASIC PPA或录用证据

- **结论：** 工程演示，创新评分不适用；PPA_ADMISSION=0


#### ROOT-PNR-03 · SMAM-RP

定位：表第3行（正文第26行附近）。N/F/H/T：**— / — / — / —**。

- **原机制：** 隔离sky130hd模块/包装器综合布局布线演示

- **最近先验与访问状态：** 对应原始机制记录；本条为入口/工程证据；本地全文

- **冻结适配：** 以二值 ep34/no_running BN/full-rank PSN 为准

- **真正增量：** 存在可布线骨架不增加算法或电路机制新意

- **代价与反证：** 该文件声明无PDN/RCX/SPEF/多角签核；不属于28nm宏与同负载闭环，也没有功能差分覆盖证明

- **迁移方式：** 仅隔离可综合性/后端探索记录；机制分见相应卡及microarch；不得当本项目ASIC PPA或录用证据

- **结论：** 工程演示，创新评分不适用；PPA_ADMISSION=0


#### ROOT-PNR-04 · OP-STW

定位：表第4行（正文第27行附近）。N/F/H/T：**— / — / — / —**。

- **原机制：** 隔离sky130hd模块/包装器综合布局布线演示

- **最近先验与访问状态：** 对应原始机制记录；本条为入口/工程证据；本地全文

- **冻结适配：** 以二值 ep34/no_running BN/full-rank PSN 为准

- **真正增量：** 存在可布线骨架不增加算法或电路机制新意

- **代价与反证：** 该文件声明无PDN/RCX/SPEF/多角签核；不属于28nm宏与同负载闭环，也没有功能差分覆盖证明

- **迁移方式：** 仅隔离可综合性/后端探索记录；机制分见相应卡及microarch；不得当本项目ASIC PPA或录用证据

- **结论：** 工程演示，创新评分不适用；PPA_ADMISSION=0


#### ROOT-PNR-05 · ADP-MAC

定位：表第5行（正文第28行附近）。N/F/H/T：**— / — / — / —**。

- **原机制：** 隔离sky130hd模块/包装器综合布局布线演示

- **最近先验与访问状态：** 对应原始机制记录；本条为入口/工程证据；本地全文

- **冻结适配：** 以二值 ep34/no_running BN/full-rank PSN 为准

- **真正增量：** 存在可布线骨架不增加算法或电路机制新意

- **代价与反证：** 该文件声明无PDN/RCX/SPEF/多角签核；不属于28nm宏与同负载闭环，也没有功能差分覆盖证明

- **迁移方式：** 仅隔离可综合性/后端探索记录；机制分见相应卡及microarch；不得当本项目ASIC PPA或录用证据

- **结论：** 工程演示，创新评分不适用；PPA_ADMISSION=0


#### ROOT-PNR-06 · ECP-QKV

定位：表第6行（正文第29行附近）。N/F/H/T：**— / — / — / —**。

- **原机制：** 隔离sky130hd模块/包装器综合布局布线演示

- **最近先验与访问状态：** 对应原始机制记录；本条为入口/工程证据；本地全文

- **冻结适配：** 以二值 ep34/no_running BN/full-rank PSN 为准

- **真正增量：** 存在可布线骨架不增加算法或电路机制新意

- **代价与反证：** 该文件声明无PDN/RCX/SPEF/多角签核；不属于28nm宏与同负载闭环，也没有功能差分覆盖证明

- **迁移方式：** 仅隔离可综合性/后端探索记录；机制分见相应卡及microarch；不得当本项目ASIC PPA或录用证据

- **结论：** 工程演示，创新评分不适用；PPA_ADMISSION=0


#### ROOT-PNR-07 · STH-Gate

定位：表第7行（正文第30行附近）。N/F/H/T：**— / — / — / —**。

- **原机制：** 隔离sky130hd模块/包装器综合布局布线演示

- **最近先验与访问状态：** 对应原始机制记录；本条为入口/工程证据；本地全文

- **冻结适配：** 以二值 ep34/no_running BN/full-rank PSN 为准

- **真正增量：** 存在可布线骨架不增加算法或电路机制新意

- **代价与反证：** 该文件声明无PDN/RCX/SPEF/多角签核；不属于28nm宏与同负载闭环，也没有功能差分覆盖证明

- **迁移方式：** 仅隔离可综合性/后端探索记录；机制分见相应卡及microarch；不得当本项目ASIC PPA或录用证据

- **结论：** 工程演示，创新评分不适用；PPA_ADMISSION=0


#### ROOT-PNR-08 · PRRC ledger

定位：表第8行（正文第31行附近）。N/F/H/T：**— / — / — / —**。

- **原机制：** 隔离sky130hd模块/包装器综合布局布线演示

- **最近先验与访问状态：** 对应原始机制记录；本条为入口/工程证据；本地全文

- **冻结适配：** 以二值 ep34/no_running BN/full-rank PSN 为准

- **真正增量：** 存在可布线骨架不增加算法或电路机制新意

- **代价与反证：** 该文件声明无PDN/RCX/SPEF/多角签核；不属于28nm宏与同负载闭环，也没有功能差分覆盖证明

- **迁移方式：** 仅隔离可综合性/后端探索记录；机制分见相应卡及microarch；不得当本项目ASIC PPA或录用证据

- **结论：** 工程演示，创新评分不适用；PPA_ADMISSION=0


#### ROOT-PNR-09 · MW-ΔBuf

定位：表第9行（正文第32行附近）。N/F/H/T：**— / — / — / —**。

- **原机制：** 隔离sky130hd模块/包装器综合布局布线演示

- **最近先验与访问状态：** 对应原始机制记录；本条为入口/工程证据；本地全文

- **冻结适配：** 以二值 ep34/no_running BN/full-rank PSN 为准

- **真正增量：** 存在可布线骨架不增加算法或电路机制新意

- **代价与反证：** 该文件声明无PDN/RCX/SPEF/多角签核；不属于28nm宏与同负载闭环，也没有功能差分覆盖证明

- **迁移方式：** 仅隔离可综合性/后端探索记录；机制分见相应卡及microarch；不得当本项目ASIC PPA或录用证据

- **结论：** 工程演示，创新评分不适用；PPA_ADMISSION=0


#### ROOT-PNR-10 · Motion-TTB

定位：表第10行（正文第33行附近）。N/F/H/T：**— / — / — / —**。

- **原机制：** 隔离sky130hd模块/包装器综合布局布线演示

- **最近先验与访问状态：** 对应原始机制记录；本条为入口/工程证据；本地全文

- **冻结适配：** 以二值 ep34/no_running BN/full-rank PSN 为准

- **真正增量：** 存在可布线骨架不增加算法或电路机制新意

- **代价与反证：** 该文件声明无PDN/RCX/SPEF/多角签核；不属于28nm宏与同负载闭环，也没有功能差分覆盖证明

- **迁移方式：** 仅隔离可综合性/后端探索记录；机制分见相应卡及microarch；不得当本项目ASIC PPA或录用证据

- **结论：** 工程演示，创新评分不适用；PPA_ADMISSION=0


#### ROOT-PNR-11 · ARM-Acc

定位：表第11行（正文第34行附近）。N/F/H/T：**— / — / — / —**。

- **原机制：** 隔离sky130hd模块/包装器综合布局布线演示

- **最近先验与访问状态：** 对应原始机制记录；本条为入口/工程证据；本地全文

- **冻结适配：** 以二值 ep34/no_running BN/full-rank PSN 为准

- **真正增量：** 存在可布线骨架不增加算法或电路机制新意

- **代价与反证：** 该文件声明无PDN/RCX/SPEF/多角签核；不属于28nm宏与同负载闭环，也没有功能差分覆盖证明

- **迁移方式：** 仅隔离可综合性/后端探索记录；机制分见相应卡及microarch；不得当本项目ASIC PPA或录用证据

- **结论：** 工程演示，创新评分不适用；PPA_ADMISSION=0


#### ROOT-PNR-12 · SP-Gate

定位：表第12行（正文第35行附近）。N/F/H/T：**— / — / — / —**。

- **原机制：** 隔离sky130hd模块/包装器综合布局布线演示

- **最近先验与访问状态：** 对应原始机制记录；本条为入口/工程证据；本地全文

- **冻结适配：** 以二值 ep34/no_running BN/full-rank PSN 为准

- **真正增量：** 存在可布线骨架不增加算法或电路机制新意

- **代价与反证：** 该文件声明无PDN/RCX/SPEF/多角签核；不属于28nm宏与同负载闭环，也没有功能差分覆盖证明

- **迁移方式：** 仅隔离可综合性/后端探索记录；机制分见相应卡及microarch；不得当本项目ASIC PPA或录用证据

- **结论：** 工程演示，创新评分不适用；PPA_ADMISSION=0


#### ROOT-PNR-13 · MFBD

定位：表第13行（正文第36行附近）。N/F/H/T：**— / — / — / —**。

- **原机制：** 隔离sky130hd模块/包装器综合布局布线演示

- **最近先验与访问状态：** 对应原始机制记录；本条为入口/工程证据；本地全文

- **冻结适配：** 以二值 ep34/no_running BN/full-rank PSN 为准

- **真正增量：** 存在可布线骨架不增加算法或电路机制新意

- **代价与反证：** 该文件声明无PDN/RCX/SPEF/多角签核；不属于28nm宏与同负载闭环，也没有功能差分覆盖证明

- **迁移方式：** 仅隔离可综合性/后端探索记录；机制分见相应卡及microarch；不得当本项目ASIC PPA或录用证据

- **结论：** 工程演示，创新评分不适用；PPA_ADMISSION=0


#### ROOT-PNR-14 · ExactCapt

定位：表第14行（正文第37行附近）。N/F/H/T：**— / — / — / —**。

- **原机制：** 隔离sky130hd模块/包装器综合布局布线演示

- **最近先验与访问状态：** 对应原始机制记录；本条为入口/工程证据；本地全文

- **冻结适配：** 以二值 ep34/no_running BN/full-rank PSN 为准

- **真正增量：** 存在可布线骨架不增加算法或电路机制新意

- **代价与反证：** 该文件声明无PDN/RCX/SPEF/多角签核；不属于28nm宏与同负载闭环，也没有功能差分覆盖证明

- **迁移方式：** 仅隔离可综合性/后端探索记录；机制分见相应卡及microarch；不得当本项目ASIC PPA或录用证据

- **结论：** 工程演示，创新评分不适用；PPA_ADMISSION=0


#### ROOT-PNR-15 · C1*-stats

定位：表第15行（正文第38行附近）。N/F/H/T：**— / — / — / —**。

- **原机制：** 隔离sky130hd模块/包装器综合布局布线演示

- **最近先验与访问状态：** 对应原始机制记录；本条为入口/工程证据；本地全文

- **冻结适配：** 以二值 ep34/no_running BN/full-rank PSN 为准

- **真正增量：** 存在可布线骨架不增加算法或电路机制新意

- **代价与反证：** 该文件声明无PDN/RCX/SPEF/多角签核；不属于28nm宏与同负载闭环，也没有功能差分覆盖证明

- **迁移方式：** 仅隔离可综合性/后端探索记录；机制分见相应卡及microarch；不得当本项目ASIC PPA或录用证据

- **结论：** 工程演示，创新评分不适用；PPA_ADMISSION=0


#### ROOT-PNR-16 · C2*-stats

定位：表第16行（正文第39行附近）。N/F/H/T：**— / — / — / —**。

- **原机制：** 隔离sky130hd模块/包装器综合布局布线演示

- **最近先验与访问状态：** 对应原始机制记录；本条为入口/工程证据；本地全文

- **冻结适配：** 以二值 ep34/no_running BN/full-rank PSN 为准

- **真正增量：** 存在可布线骨架不增加算法或电路机制新意

- **代价与反证：** 该文件声明无PDN/RCX/SPEF/多角签核；不属于28nm宏与同负载闭环，也没有功能差分覆盖证明

- **迁移方式：** 仅隔离可综合性/后端探索记录；机制分见相应卡及microarch；不得当本项目ASIC PPA或录用证据

- **结论：** 工程演示，创新评分不适用；PPA_ADMISSION=0


#### ROOT-PNR-17 · front_pipe

定位：表第17行（正文第40行附近）。N/F/H/T：**— / — / — / —**。

- **原机制：** 隔离sky130hd模块/包装器综合布局布线演示

- **最近先验与访问状态：** 对应原始机制记录；本条为入口/工程证据；本地全文

- **冻结适配：** 以二值 ep34/no_running BN/full-rank PSN 为准

- **真正增量：** 存在可布线骨架不增加算法或电路机制新意

- **代价与反证：** 该文件声明无PDN/RCX/SPEF/多角签核；不属于28nm宏与同负载闭环，也没有功能差分覆盖证明

- **迁移方式：** 仅隔离可综合性/后端探索记录；机制分见相应卡及microarch；不得当本项目ASIC PPA或录用证据

- **结论：** 工程演示，创新评分不适用；PPA_ADMISSION=0


#### ROOT-PNR-18 · back_pipe

定位：表第18行（正文第41行附近）。N/F/H/T：**— / — / — / —**。

- **原机制：** 隔离sky130hd模块/包装器综合布局布线演示

- **最近先验与访问状态：** 对应原始机制记录；本条为入口/工程证据；本地全文

- **冻结适配：** 以二值 ep34/no_running BN/full-rank PSN 为准

- **真正增量：** 存在可布线骨架不增加算法或电路机制新意

- **代价与反证：** 该文件声明无PDN/RCX/SPEF/多角签核；不属于28nm宏与同负载闭环，也没有功能差分覆盖证明

- **迁移方式：** 仅隔离可综合性/后端探索记录；机制分见相应卡及microarch；不得当本项目ASIC PPA或录用证据

- **结论：** 工程演示，创新评分不适用；PPA_ADMISSION=0


### research/codex_deep_rebuild_20260906/README.md

精读完成；SHA256 `622eee69d4f830e34d326c9230c05a5a9cb42ccbe6c1fcabff16e9a115d486ae`。

按用户返工要求复审上一轮4个优先方向。原脚本、CPU结果及PDF保持原样；不将它们计入42个原始文件的分母。


#### ROOT-PREV-01 · 穿线DFS+有限父槽+精确重算

定位：开发顺序1。N/F/H/T：**2 / 8 / 7 / 3**。

- **原机制：** 少父槽、按需重算原单父森林

- **最近先验与访问状态：** 对应原始机制记录；本条为入口/工程证据；本地全文

- **冻结适配：** 以二值 ep34/no_running BN/full-rank PSN 为准

- **真正增量：** 减少live状态的实现探索；未改变共享代数，Checkmate/DTR等已有重算调度

- **代价与反证：** 已有CPU源项数好看也不能证明创新和可移除九宏

- **迁移方式：** 降为新C1机制的存储对照，不作主创新

- **结论：** 实现或背景，不作主创新


#### ROOT-PREV-02 · 双原始权重槽+延迟投递

定位：开发顺序2。N/F/H/T：**2 / 8 / 7 / 3**。

- **原机制：** 缓存两个权重向量、拼接可用输入槽

- **最近先验与访问状态：** 对应原始机制记录；本条为入口/工程证据；本地全文

- **冻结适配：** 以二值 ep34/no_running BN/full-rank PSN 为准

- **真正增量：** 暂存和选择的局部服务优化，广播/聚合基础已知

- **代价与反证：** 输入选择、双捕获、读取仍付费，未计电路面积能量；用户已否定其主创新价值

- **迁移方式：** 保留stronger C2 baseline，不再优先开展

- **结论：** 实现或背景，不作主创新


#### ROOT-PREV-03 · PSN保守输出判定

定位：开发顺序3。N/F/H/T：**3 / 5 / 5 / 3**。

- **原机制：** 由阈值界跳过部分状态算术

- **最近先验与访问状态：** 对应原始机制记录；本条为入口/工程证据；本地全文

- **冻结适配：** 以二值 ep34/no_running BN/full-rank PSN 为准

- **真正增量：** SnaPEA/经典区间判定已有；单纯比较器不足

- **代价与反证：** 不能忽略fullrankT和动态BN；未测PSN余量

- **迁移方式：** 与动态统计重构分别审；普通版本不作第三加速岛

- **结论：** 实现或背景，不作主创新


#### ROOT-PREV-04 · Motion-XOR K配对驻留

定位：开发顺序3。N/F/H/T：**2 / 8 / 7 / 3**。

- **原机制：** 时间对端K驻留或复用

- **最近先验与访问状态：** 对应原始机制记录；本条为入口/工程证据；本地全文

- **冻结适配：** 以二值 ep34/no_running BN/full-rank PSN 为准

- **真正增量：** 普通局部复用/保持，Motion-XOR叶存在

- **代价与反证：** 旧注意力份额不可直接证明新系统收益

- **迁移方式：** 保留CardC实现参考，先核T0–T5

- **结论：** 实现或背景，不作主创新


### research/grok46_20260905/00_READ_THIS_FIRST.md

精读完成；SHA256 `cfd7f274f61c288b2dd11d42b8c5f462463a894c6f9d7882f7e9901f117b8aa3`。


#### G00-MX · MX3P 三 popcount Motion-XOR

定位：7;32;59。N/F/H/T：**3 / 7 / 8 / 4**。

- **原机制：** AND overlap、共静默、K与时间peer的XOR合成score

- **最近先验与访问状态：** [FireFly-T](https://arxiv.org/abs/2505.12771)；PRIMARY_FULLTEXT_RELEVANT_SECTIONS_READ_IN_PRECEDING_RESEARCH; FPGA attention/transpose，未复核期刊年份

- **冻结适配：** 算子结构适配；原文Q7 /64、/4不是冻结FP的0.02、0.125

- **真正增量：** 增加既有位运算项与peer供给；尚无超过直接实现的机制

- **代价与反证：** 三popcount不够顶级主张；同功能对照需完整score+归一化；attention份额未测

- **迁移方式：** 可作H67数字映射基线；重点另找peer/value服务机制

- **结论：** BASELINE_FOR_NEW_ISLAND


#### G00-DIRTY · Temporal-peer dirty / score memo

定位：7;32;59。N/F/H/T：**3 / 6 / 6 / 4**。

- **原机制：** 比较Q/K/peer依赖，仅复用确实不变的score

- **最近先验与访问状态：** [DeltaCNN, Parger et al., CVPR 2022](https://arxiv.org/html/2203.03996v2)；PRIMARY_FULLTEXT_SECTIONS_READ; §3.1、§4、§8.4，静态融合 BN 与累计 FP 误差已核

- **冻结适配：** 必须按真实score依赖和整row分母核证；≈0不是精确0

- **真正增量：** 将视频delta复用应用到score叶

- **代价与反证：** 跨步一次Q/K差异不覆盖所有query/key/peer身份；score同不必row归一化同

- **迁移方式：** 精确依赖dirty图和配对peer缓存可配套；不以97%旧P0冒充周期收益

- **结论：** RESEARCH_ONLY


#### G00-PRIVATE · Mixed-horizon binary ATLIF / membrane firewall

定位：7;32;60。N/F/H/T：**2 / 6 / 6 / 3**。

- **原机制：** 神经元内部保存T状态，仅向外发送二值fire

- **最近先验与访问状态：** [Chen/Chang Spike-IAND-Former](https://arxiv.org/abs/2503.19643)；NOT_ACCESSED_THIS_AUDIT; T≤4 unroll 来自本地摘录

- **冻结适配：** 出口二值正确；原文leak/reset错误，实际fullrank A*x+b

- **真正增量：** 类型/速率转换；尚未给新神经元执行机制

- **代价与反证：** T10不能从T≤4 LIF muxunroll推导廉价；内部非经典递推膜

- **迁移方式：** 重写PSN完整时间矩阵服务，二值边界为事实非创新

- **结论：** SUPPORT_ONLY_CORRECT_SEMANTICS


#### G00-FIBER · LoAS FTP mixed-T fiber join

定位：32。N/F/H/T：**2 / 6 / 5 / 3**。

- **原机制：** 将T位向量压成非静默fiber并inner-join

- **最近先验与访问状态：** [LoAS, MICRO 2024](https://arxiv.org/abs/2407.14073)；PRIMARY_FULLTEXT_RELEVANT_SECTIONS_READ_IN_PRECEDING_RESEARCH; LIF reset 与 temporal fibers；不等同 PSN

- **冻结适配：** 二值T2/T10可分开封包；不能照搬LoAS LIF/reset或稀疏W

- **真正增量：** 不同fiber宽度和Motion项服务

- **代价与反证：** 共静默不能从删掉的坐标丢失；索引/merge开销与1RW端口未核

- **迁移方式：** 只作配套索引组织；强对照直接bitset/popcount

- **结论：** SUPPORT_ONLY


#### G00-ECP · Bishop bound-without-S

定位：32。N/F/H/T：**4 / 4 / 5 / 4**。

- **原机制：** 以低成本活动上界判断attention计算是否必需

- **最近先验与访问状态：** [Bishop, ISCA 2025](https://arxiv.org/abs/2505.12281)；PRIMARY_FULLTEXT_RELEVANT_SECTIONS_READ_IN_PRECEDING_RESEARCH; TTB、AAC、ECP；不把其训练边界移入 ep34

- **冻结适配：** 二值Q/K允许界；不能只界overlap就删含静默/motion项的score

- **真正增量：** 针对完整Motion和分母的输出不变判据尚缺

- **代价与反证：** 低score仍进共享分母；BSA/ECP训练改模型；Q7舍入边界

- **迁移方式：** 仅在证明最终门控/输出不变时跳；否则界只是排序/预取

- **结论：** RESEARCH_ONLY_NEEDS_PROOF


#### G00-DLSS · CICC DLSS temporal similarity

定位：32。N/F/H/T：**2 / 1 / 4 / 2**。

- **原机制：** 相邻时刻特征相似时跳深U-Net

- **最近先验与访问状态：** Zhang et al., optical-flow processor, CICC 2026；LOCAL_RESEARCH_NOTE_ONLY; docs/228 指向原文；本审阅不把其数值当新验证

- **冻结适配：** 近似跳层改变ep34

- **真正增量：** 可抽出时间相似作exact dirty候选

- **代价与反证：** 原AEE退化非无损；完整T/BN使局部相似不够

- **迁移方式：** 只取精确score依赖统计，不复制近似decoder跳

- **结论：** REJECT_AS_WRITTEN


#### G00-ITA · ITA shift-round leaf

定位：32。N/F/H/T：**1 / 5 / 8 / 1**。

- **原机制：** 把score或gate当纯2的幂以省乘法

- **最近先验与访问状态：** [ITA](https://arxiv.org/abs/2307.03493)；NOT_ACCESSED; 本地引文

- **冻结适配：** 不适配；Q7分数、Q1.7多码且gate×W存在

- **真正增量：** 无

- **代价与反证：** 算法解释错误；低exp删项为有损；冻结FP与部署需隔离

- **迁移方式：** 仅舍入电路实现，不采纳错误的整数幂gate解释

- **结论：** SUPPORT_ONLY


#### G00-MIXED · T2/T10 rate conversion

定位：32;50。N/F/H/T：**2 / 8 / 7 / 2**。

- **原机制：** 不同时间长度分流并在二值出口匹配速率

- **最近先验与访问状态：** [Chen/Chang Spike-IAND-Former](https://arxiv.org/abs/2503.19643)；NOT_ACCESSED_THIS_AUDIT; T≤4 unroll 来自本地摘录

- **冻结适配：** T2是window、T10是PSN服务；正确对象不同

- **真正增量：** 仅混合长度适配

- **代价与反证：** 不是首个mixed-T；mux数量、跨T等待、BN屏障计费

- **迁移方式：** 配套组织

- **结论：** SUPPORT_ONLY


### research/grok46_20260905/01_kill_list.md

精读完成；SHA256 `4bb36afd0cbaa3eac6c0b05b569cd3eec1f77c0108d525dcfab1efd243eae80c`。


#### G01-K01 · C1 exact-subset product capture

定位：7。N/F/H/T：**1 / 8 / 8 / 2**。

- **原机制：** 完整行parent+源残差复用

- **最近先验与访问状态：** [Prosperity, HPCA 2025](https://arxiv.org/html/2503.03379v1)；NOT_ACCESSED_THIS_SUBAUDIT; 由本地记录及主代理负责全文核；不得假定无限容量

- **冻结适配：** 现有部署底座合法，非新方向

- **真正增量：** 只有1RW/有限服务实现差异待证明

- **代价与反证：** 不能称Prosperity无限缓存；当前倍率包含继承机制

- **迁移方式：** 作为强对照与底座

- **结论：** BASELINE_ONLY


#### G01-K02 · C2 TSBG / Gustavson broadcast

定位：8。N/F/H/T：**1 / 9 / 8 / 2**。

- **原机制：** group-major共享weight row并更新独立目的

- **最近先验与访问状态：** [Eyeriss v2, JETCAS 2019](https://people.csail.mit.edu/emer/media/papers/2019.04.jetcas.eyeriss_v2.pdf)；NOT_ACCESSED_THIS_AUDIT; 本地作者全文链接

- **冻结适配：** 适配自然+1二值流，协议sign定向覆盖

- **真正增量：** 实现前请求抑制/共享控制，未形成新计算图

- **代价与反证：** 广播/loop交换先验强；低复用能量反向；不能用单K1

- **迁移方式：** 作为公平基线；新贡献要减少归约或证明免算

- **结论：** BASELINE_ONLY


#### G01-K03 · Empty tile / inactive source skip

定位：9。N/F/H/T：**0 / 7 / 9 / 0**。

- **原机制：** live-mask抑制空操作

- **最近先验与访问状态：** [Bishop, ISCA 2025](https://arxiv.org/abs/2505.12281)；PRIMARY_FULLTEXT_RELEVANT_SECTIONS_READ_IN_PRECEDING_RESEARCH; TTB、AAC、ECP；不把其训练边界移入 ep34

- **冻结适配：** 源乘权的真零可以跳；attention分母不可按空QK删除

- **真正增量：** 无

- **代价与反证：** 成熟baseline；firing并非合法免算比例

- **迁移方式：** 强baseline必须包括

- **结论：** BASELINE_ONLY


#### G01-K04 · Empty tile / inactive source skip

定位：10。N/F/H/T：**0 / 7 / 9 / 0**。

- **原机制：** live-mask抑制空操作

- **最近先验与访问状态：** [Bishop, ISCA 2025](https://arxiv.org/abs/2505.12281)；PRIMARY_FULLTEXT_RELEVANT_SECTIONS_READ_IN_PRECEDING_RESEARCH; TTB、AAC、ECP；不把其训练边界移入 ep34

- **冻结适配：** 源乘权的真零可以跳；attention分母不可按空QK删除

- **真正增量：** 无

- **代价与反证：** 成熟baseline；firing并非合法免算比例

- **迁移方式：** 强baseline必须包括

- **结论：** BASELINE_ONLY


#### G01-K05 · Shiftmax integer-power gating

定位：11。N/F/H/T：**0 / 0 / 4 / 0**。

- **原机制：** 把score或gate当纯2的幂以省乘法

- **最近先验与访问状态：** [ITA](https://arxiv.org/abs/2307.03493)；NOT_ACCESSED; 本地引文

- **冻结适配：** 不适配；Q7分数、Q1.7多码且gate×W存在

- **真正增量：** 无

- **代价与反证：** 算法解释错误；低exp删项为有损；冻结FP与部署需隔离

- **迁移方式：** 仅固定舍入/移位叶工程

- **结论：** REJECT_FALSE_PREMISE


#### G01-K06 · SpatLoc-Wake

定位：12。N/F/H/T：**1 / 3 / 6 / 1**。

- **原机制：** 空间活动簇限定被执行的邻域/tile

- **最近先验与访问状态：** [ASNA-Flow, Wang et al., TVLSI 2025](https://ieeexplore.ieee.org/document/11142472/)；ACCESS_BLOCKED; 出版记录已识别，未取得可核全文；不引用本地芯片数值

- **冻结适配：** 仅跳过已证明无贡献项合法，不能限制固定450token窗口

- **真正增量：** ASNA空间局部性改任务标签

- **代价与反证：** ASNA已占近邻；动态BN/共静默分母使空位置仍有消费者

- **迁移方式：** 实际位图可作执行元数据；不是论文原语

- **结论：** BASELINE_ONLY


#### G01-K07 · MX3P 三 popcount Motion-XOR

定位：13。N/F/H/T：**3 / 7 / 8 / 4**。

- **原机制：** AND overlap、共静默、K与时间peer的XOR合成score

- **最近先验与访问状态：** [FireFly-T](https://arxiv.org/abs/2505.12771)；PRIMARY_FULLTEXT_RELEVANT_SECTIONS_READ_IN_PRECEDING_RESEARCH; FPGA attention/transpose，未复核期刊年份

- **冻结适配：** 算子结构适配；原文Q7 /64、/4不是冻结FP的0.02、0.125

- **真正增量：** 增加既有位运算项与peer供给；尚无超过直接实现的机制

- **代价与反证：** 三popcount不够顶级主张；同功能对照需完整score+归一化；attention份额未测

- **迁移方式：** 可作H67数字映射基线；重点另找peer/value服务机制

- **结论：** BASELINE_FOR_NEW_ISLAND


#### G01-K08 · binary add-forest / post-ATLIF add-sub

定位：14。N/F/H/T：**1 / 7 / 8 / 1**。

- **原机制：** 二值激活令乘积退化为W或0，构建父和残差

- **最近先验与访问状态：** [Prosperity, HPCA 2025](https://arxiv.org/html/2503.03379v1)；NOT_ACCESSED_THIS_SUBAUDIT; 由本地记录及主代理负责全文核；不得假定无限容量

- **冻结适配：** 整数部署与阈值folding需固定尺度；FP重排并非逐位等价

- **真正增量：** product-sparsity本来就用于二值SNN；换成add-forest无新增

- **代价与反证：** 95个theta恰1也不等于所有theta可无误差折叠；TSBG已共享单项

- **迁移方式：** 作为执行合同，不保留重命名主张

- **结论：** BASELINE_ONLY


#### G01-K09 · PCM-FFN + SSA-OF / SSA-MotionMask

定位：15。N/F/H/T：**2 / 1 / 1 / 1**。

- **原机制：** PCM 线性层加 stochastic AND-count attention

- **最近先验与访问状态：** [Xpikeformer](https://arxiv.org/abs/2408.08794)；NOT_ACCESSED_THIS_AUDIT; 不把本地 HW 标签当硅证据

- **冻结适配：** 随机注意力不等于冻结 Motion-XOR；没有实值 ATLIF 第二轨

- **真正增量：** motion mask 叠加现成混合架构

- **代价与反证：** 噪声/随机采样/重训；原稿将 simulation 混为 true HW 的风险

- **迁移方式：** 可借流式 attention 服务组织，不借随机算术/模拟宏

- **结论：** REJECT_AS_WRITTEN


#### G01-K10 · SpiDR-Reconf-CIM / PRRC-CIM-Modes

定位：16。N/F/H/T：**2 / 2 / 2 / 1**。

- **原机制：** 按精度/规模重配置，零跳过并异步衔接 CIM

- **最近先验与访问状态：** [SpiDR, 2024 preprint](https://arxiv.org/html/2411.02854v1)；PRIMARY_FULLTEXT_SECTIONS_READ; §II-A–F, 10T macro、dual-port IFspad、IF/LIF/reset 已核

- **冻结适配：** 不同精度、head 分流和 IF/LIF 不属于冻结模型

- **真正增量：** 按金字塔选择模式属于应用映射

- **代价与反证：** 定制 10T、dual-port IFspad、精度改变；无真实 AEE/宏路径

- **迁移方式：** 只借任务粒度配置与有限 FIFO；不得搬 silicon PPA

- **结论：** REJECT_AS_HEADLINE


#### G01-K11 · C1 exact-subset product capture

定位：17。N/F/H/T：**1 / 8 / 8 / 2**。

- **原机制：** 完整行parent+源残差复用

- **最近先验与访问状态：** [Prosperity, HPCA 2025](https://arxiv.org/html/2503.03379v1)；NOT_ACCESSED_THIS_SUBAUDIT; 由本地记录及主代理负责全文核；不得假定无限容量

- **冻结适配：** 现有部署底座合法，非新方向

- **真正增量：** 只有1RW/有限服务实现差异待证明

- **代价与反证：** 不能称Prosperity无限缓存；当前倍率包含继承机制

- **迁移方式：** 作为强对照与底座

- **结论：** BASELINE_ONLY


#### G01-K12 · 4bit / dual-side W sparsity

定位：18。N/F/H/T：**2 / 0 / 5 / 1**。

- **原机制：** 剪权/降精度换更稀疏PE服务

- **最近先验与访问状态：** [BBS/BitVert](https://arxiv.org/abs/2409.05227)；NOT_ACCESSED_THIS_AUDIT; 本地引文

- **冻结适配：** 不适配冻结quant=false

- **真正增量：** 新训练/部署Pareto，不是无损硬件

- **代价与反证：** 现有dense W不能套97%稀疏；无重训练不等于无损

- **迁移方式：** 另行明确模型身份才可重训，此轮不迁移

- **结论：** NEW_MODEL_ONLY


#### G01-K13 · LoAS FTP mixed-T fiber join

定位：19。N/F/H/T：**2 / 6 / 5 / 3**。

- **原机制：** 将T位向量压成非静默fiber并inner-join

- **最近先验与访问状态：** [LoAS, MICRO 2024](https://arxiv.org/abs/2407.14073)；PRIMARY_FULLTEXT_RELEVANT_SECTIONS_READ_IN_PRECEDING_RESEARCH; LIF reset 与 temporal fibers；不等同 PSN

- **冻结适配：** 二值T2/T10可分开封包；不能照搬LoAS LIF/reset或稀疏W

- **真正增量：** 不同fiber宽度和Motion项服务

- **代价与反证：** 共静默不能从删掉的坐标丢失；索引/merge开销与1RW端口未核

- **迁移方式：** 只作配套索引组织；强对照直接bitset/popcount

- **结论：** SUPPORT_ONLY


#### G01-K14 · ELSA elastic first-response

定位：20。N/F/H/T：**2 / 2 / 5 / 2**。

- **原机制：** token逐层弹性先到先算

- **最近先验与访问状态：** [ELSA, ISCA 2026](https://arxiv.org/html/2605.20802v1)；NOT_ACCESSED_THIS_SUBAUDIT; 动态 BN 屏障由本地源码独立确认

- **冻结适配：** 24 dynamic BN需整域统计；fullrankPSN需完整T输入

- **真正增量：** 仅局部队列打包可保留

- **代价与反证：** 分类first-correct不能变成denseflow完整结果；无全网闭环

- **迁移方式：** 只保局部ready/valid与BAER，不写弹性网络加速

- **结论：** REJECT_SYSTEM_CLAIM


#### G01-K15 · Bishop bound-without-S

定位：21。N/F/H/T：**4 / 4 / 5 / 4**。

- **原机制：** 以低成本活动上界判断attention计算是否必需

- **最近先验与访问状态：** [Bishop, ISCA 2025](https://arxiv.org/abs/2505.12281)；PRIMARY_FULLTEXT_RELEVANT_SECTIONS_READ_IN_PRECEDING_RESEARCH; TTB、AAC、ECP；不把其训练边界移入 ep34

- **冻结适配：** 二值Q/K允许界；不能只界overlap就删含静默/motion项的score

- **真正增量：** 针对完整Motion和分母的输出不变判据尚缺

- **代价与反证：** 低score仍进共享分母；BSA/ECP训练改模型；Q7舍入边界

- **迁移方式：** 仅在证明最终门控/输出不变时跳；否则界只是排序/预取

- **结论：** RESEARCH_ONLY_NEEDS_PROOF


#### G01-K16 · CICC MaxPool/ReLU redundancy speculation

定位：22。N/F/H/T：**2 / 0 / 3 / 1**。

- **原机制：** 用nonlinearity特性推测冗余输出

- **最近先验与访问状态：** Zhang et al., optical-flow processor, CICC 2026；LOCAL_RESEARCH_NOTE_ONLY; docs/228 指向原文；本审阅不把其数值当新验证

- **冻结适配：** Shiftmax共分母不满足同类跳过条件

- **真正增量：** 转用需要重新证明

- **代价与反证：** 静默项仍改变分母；local note原文访问欠缺

- **迁移方式：** 不移植原speculation；可启发BNREFINE但需全新证明

- **结论：** REJECT


#### G01-K17 · CICC BWAC bitmap weight compression

定位：23。N/F/H/T：**1 / 6 / 7 / 1**。

- **原机制：** 小组bitmap描述非零权重

- **最近先验与访问状态：** Zhang et al., optical-flow processor, CICC 2026；LOCAL_RESEARCH_NOTE_ONLY; docs/228 指向原文；本审阅不把其数值当新验证

- **冻结适配：** 可无损编码部署INT8，不能称冻结模型PPA

- **真正增量：** 普通编码映射

- **代价与反证：** 本地记录稀疏率低且bitmap扩张；数字须主代理封存核

- **迁移方式：** 作为被更强压缩方案对比的基线

- **结论：** LIKELY_NO_GAIN_SUPPORT


#### G01-K18 · CICC DLSS temporal similarity

定位：24。N/F/H/T：**2 / 1 / 4 / 2**。

- **原机制：** 相邻时刻特征相似时跳深U-Net

- **最近先验与访问状态：** Zhang et al., optical-flow processor, CICC 2026；LOCAL_RESEARCH_NOTE_ONLY; docs/228 指向原文；本审阅不把其数值当新验证

- **冻结适配：** 近似跳层改变ep34

- **真正增量：** 可抽出时间相似作exact dirty候选

- **代价与反证：** 原AEE退化非无损；完整T/BN使局部相似不够

- **迁移方式：** 只取精确score依赖统计，不复制近似decoder跳

- **结论：** REJECT_AS_WRITTEN


#### G01-K19 · correlation / RAFT pyramid residual island

定位：25。N/F/H/T：**1 / 0 / 2 / 0**。

- **原机制：** 增加cost volume、lookup、迭代流更新

- **最近先验与访问状态：** [TMA, Liu et al., ICCV 2023](https://arxiv.org/html/2303.11629v2)；PRIMARY_FULLTEXT_SECTIONS_READ; §3.2/式4–6，RAFT 与 correlation volumes 已核

- **冻结适配：** 冻结为U-Net flow head，无该图

- **真正增量：** 新模型不属于C1/C2重构

- **代价与反证：** 新projection/lookup/状态/训练；H81禁令不可绕

- **迁移方式：** 只作外部先验文献

- **结论：** REJECT


#### G01-K20 · GANAX ConvTranspose inserted-zero skip

定位：26。N/F/H/T：**1 / 8 / 7 / 1**。

- **原机制：** 不计算插入的零，调整并行映射

- **最近先验与访问状态：** [Sparseloop, MICRO 2022](https://sparseloop.mit.edu/documents/2022-micro-sparseloop.pdf)；NOT_ACCESSED_THIS_AUDIT; 模型不是 RTL 证据

- **冻结适配：** 数学可无损，但冻结96lane当前已满

- **真正增量：** 现有工作映射未见新增机会

- **代价与反证：** 本地封存上限≤1需主代理核；不能另编第三加速

- **迁移方式：** 实现基线支持

- **结论：** NO_NEW_ISLAND


#### G01-K21 · DualRail-CIM

定位：27。N/F/H/T：**1 / 0 / 0 / 0**。

- **原机制：** 二值门 charge-CIM＋实值 ATLIF DigiCIM

- **最近先验与访问状态：** [Spike-CIM, A-SSCC 2022](https://doi.org/10.1109/A-SSCC56115.2022.9980797)；NOT_ACCESSED; 本地引文

- **冻结适配：** 与冻结二值 ATLIF 直接冲突

- **真正增量：** 物理介质并置而非新计算机制

- **代价与反证：** 不存在待保留int8幅值；模拟禁令；两宏面积/域转换

- **迁移方式：** 不作为新岛

- **结论：** REJECT


#### G01-K22 · Fixed-T10 coverage service

定位：29。N/F/H/T：**0 / 7 / 8 / 1**。

- **原机制：** 精确服务T10状态与commit

- **最近先验与访问状态：** [LoAS, MICRO 2024](https://arxiv.org/abs/2407.14073)；PRIMARY_FULLTEXT_RELEVANT_SECTIONS_READ_IN_PRECEDING_RESEARCH; LIF reset 与 temporal fibers；不等同 PSN

- **冻结适配：** 覆盖需以真实PSN为准；旧C3结果不证明fullFP整网

- **真正增量：** 覆盖基础没有公平速度对照

- **代价与反证：** 不能当第三倍率；经典膜叙事要更正

- **迁移方式：** 精确输出/接口验证基础

- **结论：** COVERAGE_ONLY


### research/grok46_20260905/02_ranked_mechanisms.md

精读完成；SHA256 `68b75362151720a388ef5e1c1d1860a00bf91011125b8af31e4651e23f44a3ee`。


#### G02-R1 · MX3P 三 popcount Motion-XOR

定位：7–22。N/F/H/T：**3 / 7 / 8 / 4**。

- **原机制：** AND overlap、共静默、K与时间peer的XOR合成score

- **最近先验与访问状态：** [FireFly-T](https://arxiv.org/abs/2505.12771)；PRIMARY_FULLTEXT_RELEVANT_SECTIONS_READ_IN_PRECEDING_RESEARCH; FPGA attention/transpose，未复核期刊年份

- **冻结适配：** 算子结构适配；原文Q7 /64、/4不是冻结FP的0.02、0.125

- **真正增量：** 增加既有位运算项与peer供给；尚无超过直接实现的机制

- **代价与反证：** 三popcount不够顶级主张；同功能对照需完整score+归一化；attention份额未测

- **迁移方式：** 可作H67数字映射基线；重点另找peer/value服务机制

- **结论：** BASELINE_FOR_NEW_ISLAND


#### G02-R4 · Temporal-peer dirty / score memo

定位：24–36。N/F/H/T：**3 / 6 / 6 / 4**。

- **原机制：** 比较Q/K/peer依赖，仅复用确实不变的score

- **最近先验与访问状态：** [DeltaCNN, Parger et al., CVPR 2022](https://arxiv.org/html/2203.03996v2)；PRIMARY_FULLTEXT_SECTIONS_READ; §3.1、§4、§8.4，静态融合 BN 与累计 FP 误差已核

- **冻结适配：** 必须按真实score依赖和整row分母核证；≈0不是精确0

- **真正增量：** 将视频delta复用应用到score叶

- **代价与反证：** 跨步一次Q/K差异不覆盖所有query/key/peer身份；score同不必row归一化同

- **迁移方式：** 精确依赖dirty图和配对peer缓存可配套；不以97%旧P0冒充周期收益

- **结论：** RESEARCH_ONLY


#### G02-R2 · LoAS FTP mixed-T fiber join

定位：38–46。N/F/H/T：**2 / 6 / 5 / 3**。

- **原机制：** 将T位向量压成非静默fiber并inner-join

- **最近先验与访问状态：** [LoAS, MICRO 2024](https://arxiv.org/abs/2407.14073)；PRIMARY_FULLTEXT_RELEVANT_SECTIONS_READ_IN_PRECEDING_RESEARCH; LIF reset 与 temporal fibers；不等同 PSN

- **冻结适配：** 二值T2/T10可分开封包；不能照搬LoAS LIF/reset或稀疏W

- **真正增量：** 不同fiber宽度和Motion项服务

- **代价与反证：** 共静默不能从删掉的坐标丢失；索引/merge开销与1RW端口未核

- **迁移方式：** 只作配套索引组织；强对照直接bitset/popcount

- **结论：** SUPPORT_ONLY


#### G02-R3 · Mixed-horizon binary ATLIF / membrane firewall

定位：48–61。N/F/H/T：**2 / 6 / 6 / 3**。

- **原机制：** 神经元内部保存T状态，仅向外发送二值fire

- **最近先验与访问状态：** [Chen/Chang Spike-IAND-Former](https://arxiv.org/abs/2503.19643)；NOT_ACCESSED_THIS_AUDIT; T≤4 unroll 来自本地摘录

- **冻结适配：** 出口二值正确；原文leak/reset错误，实际fullrank A*x+b

- **真正增量：** 类型/速率转换；尚未给新神经元执行机制

- **代价与反证：** T10不能从T≤4 LIF muxunroll推导廉价；内部非经典递推膜

- **迁移方式：** 重写PSN完整时间矩阵服务，二值边界为事实非创新

- **结论：** SUPPORT_ONLY_CORRECT_SEMANTICS


#### G02-ADD · binary add-forest / post-ATLIF add-sub

定位：62;105。N/F/H/T：**1 / 7 / 8 / 1**。

- **原机制：** 二值激活令乘积退化为W或0，构建父和残差

- **最近先验与访问状态：** [Prosperity, HPCA 2025](https://arxiv.org/html/2503.03379v1)；NOT_ACCESSED_THIS_SUBAUDIT; 由本地记录及主代理负责全文核；不得假定无限容量

- **冻结适配：** 整数部署与阈值folding需固定尺度；FP重排并非逐位等价

- **真正增量：** product-sparsity本来就用于二值SNN；换成add-forest无新增

- **代价与反证：** 95个theta恰1也不等于所有theta可无误差折叠；TSBG已共享单项

- **迁移方式：** 作为执行合同，不保留重命名主张

- **结论：** BASELINE_ONLY


#### G02-R5 · Bishop bound-without-S

定位：64–70。N/F/H/T：**4 / 4 / 5 / 4**。

- **原机制：** 以低成本活动上界判断attention计算是否必需

- **最近先验与访问状态：** [Bishop, ISCA 2025](https://arxiv.org/abs/2505.12281)；PRIMARY_FULLTEXT_RELEVANT_SECTIONS_READ_IN_PRECEDING_RESEARCH; TTB、AAC、ECP；不把其训练边界移入 ep34

- **冻结适配：** 二值Q/K允许界；不能只界overlap就删含静默/motion项的score

- **真正增量：** 针对完整Motion和分母的输出不变判据尚缺

- **代价与反证：** 低score仍进共享分母；BSA/ECP训练改模型；Q7舍入边界

- **迁移方式：** 仅在证明最终门控/输出不变时跳；否则界只是排序/预取

- **结论：** RESEARCH_ONLY_NEEDS_PROOF


#### G02-R6 · K-zero / K-as-V gating

定位：72–76。N/F/H/T：**1 / 8 / 8 / 1**。

- **原机制：** K=0时抑制value读取/乘门

- **最近先验与访问状态：** [FireFly-T](https://arxiv.org/abs/2505.12771)；PRIMARY_FULLTEXT_RELEVANT_SECTIONS_READ_IN_PRECEDING_RESEARCH; FPGA attention/transpose，未复核期刊年份

- **冻结适配：** 值路径严格合法；原文“同时跳score”错误

- **真正增量：** 已知零值跳过应用到K作V

- **代价与反证：** score仍由共静默和peer决定，并贡献分母；不得删归一化项

- **迁移方式：** 保留value-gather/gate-multiply隔离；score按完整原算术

- **结论：** BASELINE_ONLY_SCORE_SKIP_REJECTED


#### G02-R7 · Spatial dirty-run scheduler

定位：78–84。N/F/H/T：**2 / 7 / 7 / 2**。

- **原机制：** 按连续dirty位置打包请求

- **最近先验与访问状态：** [ExSpike](https://arxiv.org/abs/2606.20414)；NOT_ACCESSED; 本地引文

- **冻结适配：** 可改变独立请求顺序，不删消费者

- **真正增量：** dirty标签替代普通活动标签

- **代价与反证：** RLE/队列先验普通；短run和尾部会抵消；无ep34分布

- **迁移方式：** 作为配对score服务配套，不单列主机制

- **结论：** SUPPORT_ONLY


#### G02-R8 · Inter-frame dirty-tile encoder skip

定位：86–88;105。N/F/H/T：**3 / 2 / 4 / 3**。

- **原机制：** 无新事件的tile复用上次深层输出

- **最近先验与访问状态：** [DeltaCNN, Parger et al., CVPR 2022](https://arxiv.org/html/2203.03996v2)；PRIMARY_FULLTEXT_SECTIONS_READ; §3.1、§4、§8.4，静态融合 BN 与累计 FP 误差已核

- **冻结适配：** 原始输入不变不保证含BN/窗口/PSN依赖的输出不变

- **真正增量：** 普通CBinfer/delta推到网络

- **代价与反证：** no_running BN造成全域耦合；卷积halo/跨窗/跨时依赖；FP长期误差

- **迁移方式：** 必须新全局消费者证书；参见 BNREFINE，不直接免算

- **结论：** REJECT_SIMPLE_VERSION


#### G02-W1 · ELSA bundled AER packing

定位：94。N/F/H/T：**1 / 7 / 7 / 1**。

- **原机制：** 成组编码二值事件，局部细粒度流水

- **最近先验与访问状态：** [ELSA, ISCA 2026](https://arxiv.org/html/2605.20802v1)；NOT_ACCESSED_THIS_SUBAUDIT; 动态 BN 屏障由本地源码独立确认

- **冻结适配：** 局部协议可迁移，全网络elastic不可

- **真正增量：** 仅局部队列打包可保留

- **代价与反证：** 分类first-correct不能变成denseflow完整结果；无全网闭环

- **迁移方式：** 只保局部ready/valid与BAER，不写弹性网络加速

- **结论：** SUPPORT_ONLY


#### G02-W2 · SpikeX activity tags

定位：95。N/F/H/T：**1 / 7 / 7 / 1**。

- **原机制：** 按连续dirty位置打包请求

- **最近先验与访问状态：** [SpikeX](https://arxiv.org/html/2505.12292v1)；NOT_ACCESSED_THIS_AUDIT

- **冻结适配：** 可改变独立请求顺序，不删消费者

- **真正增量：** dirty标签替代普通活动标签

- **代价与反证：** RLE/队列先验普通；短run和尾部会抵消；无ep34分布

- **迁移方式：** 作为配对score服务配套，不单列主机制

- **结论：** SUPPORT_ONLY


#### G02-W3 · Bishop asymmetric TTB

定位：96。N/F/H/T：**2 / 7 / 6 / 2**。

- **原机制：** 不同时间长度分流并在二值出口匹配速率

- **最近先验与访问状态：** [Chen/Chang Spike-IAND-Former](https://arxiv.org/abs/2503.19643)；NOT_ACCESSED_THIS_AUDIT; T≤4 unroll 来自本地摘录

- **冻结适配：** T2是window、T10是PSN服务；正确对象不同

- **真正增量：** 仅混合长度适配

- **代价与反证：** 不是首个mixed-T；mux数量、跨T等待、BN屏障计费

- **迁移方式：** 配套组织

- **结论：** SUPPORT_ONLY


#### G02-W4 · 4bit / dual-side W sparsity

定位：97。N/F/H/T：**2 / 0 / 5 / 1**。

- **原机制：** 剪权/降精度换更稀疏PE服务

- **最近先验与访问状态：** [BBS/BitVert](https://arxiv.org/abs/2409.05227)；NOT_ACCESSED_THIS_AUDIT; 本地引文

- **冻结适配：** 不适配冻结quant=false

- **真正增量：** 新训练/部署Pareto，不是无损硬件

- **代价与反证：** 现有dense W不能套97%稀疏；无重训练不等于无损

- **迁移方式：** 另行明确模型身份才可重训，此轮不迁移

- **结论：** NEW_MODEL_ONLY


#### G02-W5 · Scene-adaptive theta

定位：98。N/F/H/T：**1 / 0 / 6 / 0**。

- **原机制：** 运行时根据场景改theta

- **最近先验与访问状态：** [Chen/Chang Spike-IAND-Former](https://arxiv.org/abs/2503.19643)；NOT_ACCESSED_THIS_AUDIT; T≤4 unroll 来自本地摘录

- **冻结适配：** 冻结推理theta静态，不适配

- **真正增量：** 新增自适应控制

- **代价与反证：** 改变发放及模型函数

- **迁移方式：** 只保checkpoint参数加载

- **结论：** REJECT


#### G02-W6 · Polar ON/OFF dual-rail

定位：99。N/F/H/T：**1 / 0 / 6 / 0**。

- **原机制：** 把正负脉冲两轨当ATLIF输出

- **最近先验与访问状态：** [Spike-CIM, A-SSCC 2022](https://doi.org/10.1109/A-SSCC56115.2022.9980797)；NOT_ACCESSED; 本地引文

- **冻结适配：** 事件输入有极性不等于ATLIF仍为ternary；出口binary

- **真正增量：** 物理介质并置而非新计算机制

- **代价与反证：** 错误层级身份

- **迁移方式：** 不作为新岛

- **结论：** REJECT


### research/grok46_20260905/03_algorithm_native_motionxor_atlif.md

精读完成；SHA256 `7a800f0bc8cab0d046c63d6a3b5571ae9195ccee98b966ef769ae6f58acca2fe`。


#### G03-A · MX3P 三 popcount Motion-XOR

定位：5–32。N/F/H/T：**3 / 7 / 8 / 4**。

- **原机制：** AND overlap、共静默、K与时间peer的XOR合成score

- **最近先验与访问状态：** [FireFly-T](https://arxiv.org/abs/2505.12771)；PRIMARY_FULLTEXT_RELEVANT_SECTIONS_READ_IN_PRECEDING_RESEARCH; FPGA attention/transpose，未复核期刊年份

- **冻结适配：** 算子结构适配；原文Q7 /64、/4不是冻结FP的0.02、0.125

- **真正增量：** 增加既有位运算项与peer供给；尚无超过直接实现的机制

- **代价与反证：** 三popcount不够顶级主张；同功能对照需完整score+归一化；attention份额未测

- **迁移方式：** 可作H67数字映射基线；重点另找peer/value服务机制

- **结论：** BASELINE_FOR_NEW_ISLAND


#### G03-B · Mixed-horizon binary ATLIF / membrane firewall

定位：34–51。N/F/H/T：**2 / 6 / 6 / 3**。

- **原机制：** 神经元内部保存T状态，仅向外发送二值fire

- **最近先验与访问状态：** [Chen/Chang Spike-IAND-Former](https://arxiv.org/abs/2503.19643)；NOT_ACCESSED_THIS_AUDIT; T≤4 unroll 来自本地摘录

- **冻结适配：** 出口二值正确；原文leak/reset错误，实际fullrank A*x+b

- **真正增量：** 类型/速率转换；尚未给新神经元执行机制

- **代价与反证：** T10不能从T≤4 LIF muxunroll推导廉价；内部非经典递推膜

- **迁移方式：** 重写PSN完整时间矩阵服务，二值边界为事实非创新

- **结论：** SUPPORT_ONLY_CORRECT_SEMANTICS


#### G03-C · T2/T10 rate conversion

定位：53–61。N/F/H/T：**2 / 8 / 7 / 2**。

- **原机制：** 不同时间长度分流并在二值出口匹配速率

- **最近先验与访问状态：** [Chen/Chang Spike-IAND-Former](https://arxiv.org/abs/2503.19643)；NOT_ACCESSED_THIS_AUDIT; T≤4 unroll 来自本地摘录

- **冻结适配：** T2是window、T10是PSN服务；正确对象不同

- **真正增量：** 仅混合长度适配

- **代价与反证：** 不是首个mixed-T；mux数量、跨T等待、BN屏障计费

- **迁移方式：** 配套组织

- **结论：** SUPPORT_ONLY


#### G03-D1 · Temporal-peer dirty / score memo

定位：69–71。N/F/H/T：**3 / 6 / 6 / 4**。

- **原机制：** 比较Q/K/peer依赖，仅复用确实不变的score

- **最近先验与访问状态：** [DeltaCNN, Parger et al., CVPR 2022](https://arxiv.org/html/2203.03996v2)；PRIMARY_FULLTEXT_SECTIONS_READ; §3.1、§4、§8.4，静态融合 BN 与累计 FP 误差已核

- **冻结适配：** 必须按真实score依赖和整row分母核证；≈0不是精确0

- **真正增量：** 将视频delta复用应用到score叶

- **代价与反证：** 跨步一次Q/K差异不覆盖所有query/key/peer身份；score同不必row归一化同

- **迁移方式：** 精确依赖dirty图和配对peer缓存可配套；不以97%旧P0冒充周期收益

- **结论：** RESEARCH_ONLY


#### G03-D2 · Spatial dirty-run scheduler

定位：72。N/F/H/T：**2 / 7 / 7 / 2**。

- **原机制：** 按连续dirty位置打包请求

- **最近先验与访问状态：** [ExSpike](https://arxiv.org/abs/2606.20414)；NOT_ACCESSED; 本地引文

- **冻结适配：** 可改变独立请求顺序，不删消费者

- **真正增量：** dirty标签替代普通活动标签

- **代价与反证：** RLE/队列先验普通；短run和尾部会抵消；无ep34分布

- **迁移方式：** 作为配对score服务配套，不单列主机制

- **结论：** SUPPORT_ONLY


#### G03-D3 · K-zero / K-as-V gating

定位：73。N/F/H/T：**1 / 8 / 8 / 1**。

- **原机制：** K=0时抑制value读取/乘门

- **最近先验与访问状态：** [FireFly-T](https://arxiv.org/abs/2505.12771)；PRIMARY_FULLTEXT_RELEVANT_SECTIONS_READ_IN_PRECEDING_RESEARCH; FPGA attention/transpose，未复核期刊年份

- **冻结适配：** 值路径严格合法；原文“同时跳score”错误

- **真正增量：** 已知零值跳过应用到K作V

- **代价与反证：** score仍由共静默和peer决定，并贡献分母；不得删归一化项

- **迁移方式：** 保留value-gather/gate-multiply隔离；score按完整原算术

- **结论：** BASELINE_ONLY_SCORE_SKIP_REJECTED


### research/grok46_20260905/04_paper_survey.md

精读完成；SHA256 `6b2ce7d0be35dc77fb1270fd52789921a2d85f09870a5fa2a848fadd5f19cac9`。


#### G04-01 · Prosperity product forest

定位：9。N/F/H/T：**1 / 8 / 8 / 2**。

- **原机制：** 完整行parent+源残差复用

- **最近先验与访问状态：** [Prosperity, HPCA 2025](https://arxiv.org/html/2503.03379v1)；NOT_ACCESSED_THIS_SUBAUDIT; 由本地记录及主代理负责全文核；不得假定无限容量

- **冻结适配：** 现有部署底座合法，非新方向

- **真正增量：** 只有1RW/有限服务实现差异待证明

- **代价与反证：** 不能称Prosperity无限缓存；当前倍率包含继承机制

- **迁移方式：** 作为强对照与底座

- **结论：** BASELINE_ONLY


#### G04-02 · LoAS FTP / CSF join

定位：10。N/F/H/T：**2 / 6 / 5 / 3**。

- **原机制：** 将T位向量压成非静默fiber并inner-join

- **最近先验与访问状态：** [LoAS, MICRO 2024](https://arxiv.org/abs/2407.14073)；PRIMARY_FULLTEXT_RELEVANT_SECTIONS_READ_IN_PRECEDING_RESEARCH; LIF reset 与 temporal fibers；不等同 PSN

- **冻结适配：** 二值T2/T10可分开封包；不能照搬LoAS LIF/reset或稀疏W

- **真正增量：** 不同fiber宽度和Motion项服务

- **代价与反证：** 共静默不能从删掉的坐标丢失；索引/merge开销与1RW端口未核

- **迁移方式：** 只作配套索引组织；强对照直接bitset/popcount

- **结论：** SUPPORT_ONLY


#### G04-03 · APEX PASC-IF

定位：11。N/F/H/T：**1 / 0 / 4 / 1**。

- **原机制：** 神经元内部保存T状态，仅向外发送二值fire

- **最近先验与访问状态：** [APEX](https://arxiv.org/abs/2608.19046)；NOT_ACCESSED; 本地预印本引文，非已核顶会

- **冻结适配：** 换PASC-IF改变PSN/ATLIF

- **真正增量：** 借existing dataflow+neuron组织不构成新执行

- **代价与反证：** T10不能从T≤4 LIF muxunroll推导廉价；内部非经典递推膜

- **迁移方式：** 重写PSN完整时间矩阵服务，二值边界为事实非创新

- **结论：** NEW_MODEL_ONLY


#### G04-04 · FireFly-T dual engine / LUT6 popcount / byte-write

定位：12。N/F/H/T：**2 / 5 / 6 / 3**。

- **原机制：** AND overlap、共静默、K与时间peer的XOR合成score

- **最近先验与访问状态：** [FireFly-T](https://arxiv.org/abs/2505.12771)；PRIMARY_FULLTEXT_RELEVANT_SECTIONS_READ_IN_PRECEDING_RESEARCH; FPGA attention/transpose，未复核期刊年份

- **冻结适配：** 可借二值score组织，三项函数不同且无byte-write宏

- **真正增量：** 标准CMOS+peer供给需新设计

- **代价与反证：** 三popcount不够顶级主张；同功能对照需完整score+归一化；attention份额未测

- **迁移方式：** 可作H67数字映射基线；重点另找peer/value服务机制

- **结论：** BASELINE_FOR_NEW_ISLAND


#### G04-05 · FireFly-S dual sparsity

定位：13。N/F/H/T：**2 / 0 / 5 / 1**。

- **原机制：** 剪权/降精度换更稀疏PE服务

- **最近先验与访问状态：** FireFly-S；NOT_ACCESSED; 本地表格引文，无补造链接

- **冻结适配：** 不适配冻结quant=false

- **真正增量：** 新训练/部署Pareto，不是无损硬件

- **代价与反证：** 现有dense W不能套97%稀疏；无重训练不等于无损

- **迁移方式：** 另行明确模型身份才可重训，此轮不迁移

- **结论：** NEW_MODEL_ONLY


#### G04-06 · Bishop ECP / TTB / AAC

定位：14。N/F/H/T：**4 / 4 / 5 / 4**。

- **原机制：** 以低成本活动上界判断attention计算是否必需

- **最近先验与访问状态：** [Bishop, ISCA 2025](https://arxiv.org/abs/2505.12281)；PRIMARY_FULLTEXT_RELEVANT_SECTIONS_READ_IN_PRECEDING_RESEARCH; TTB、AAC、ECP；不把其训练边界移入 ep34

- **冻结适配：** 二值Q/K允许界；不能只界overlap就删含静默/motion项的score

- **真正增量：** 针对完整Motion和分母的输出不变判据尚缺

- **代价与反证：** 低score仍进共享分母；BSA/ECP训练改模型；Q7舍入边界

- **迁移方式：** 仅在证明最终门控/输出不变时跳；否则界只是排序/预取

- **结论：** RESEARCH_ONLY_NEEDS_PROOF


#### G04-07 · SpikeX weight reuse / HAS tags

定位：15。N/F/H/T：**1 / 6 / 6 / 1**。

- **原机制：** 按连续dirty位置打包请求

- **最近先验与访问状态：** [SpikeX](https://arxiv.org/html/2505.12292v1)；NOT_ACCESSED_THIS_AUDIT; 本地引文

- **冻结适配：** 可改变独立请求顺序，不删消费者

- **真正增量：** dirty标签替代普通活动标签

- **代价与反证：** RLE/队列先验普通；短run和尾部会抵消；无ep34分布

- **迁移方式：** 作为配对score服务配套，不单列主机制

- **结论：** SUPPORT_ONLY


#### G04-08 · ELSA BAER / elastic / Gustavson

定位：16。N/F/H/T：**2 / 2 / 5 / 2**。

- **原机制：** token逐层弹性先到先算

- **最近先验与访问状态：** [ELSA, ISCA 2026](https://arxiv.org/html/2605.20802v1)；NOT_ACCESSED_THIS_SUBAUDIT; 动态 BN 屏障由本地源码独立确认

- **冻结适配：** 24 dynamic BN需整域统计；fullrankPSN需完整T输入

- **真正增量：** 仅局部队列打包可保留

- **代价与反证：** 分类first-correct不能变成denseflow完整结果；无全网闭环

- **迁移方式：** 只保局部ready/valid与BAER，不写弹性网络加速

- **结论：** REJECT_SYSTEM_CLAIM


#### G04-09 · GustavSNN tick-batch Gustavson

定位：17。N/F/H/T：**1 / 9 / 8 / 2**。

- **原机制：** group-major共享weight row并更新独立目的

- **最近先验与访问状态：** GustavSNN, HPCA 2026；NOT_ACCESSED; 本地表格引文

- **冻结适配：** 适配自然+1二值流，协议sign定向覆盖

- **真正增量：** 实现前请求抑制/共享控制，未形成新计算图

- **代价与反证：** 广播/loop交换先验强；低复用能量反向；不能用单K1

- **迁移方式：** 作为公平基线；新贡献要减少归约或证明免算

- **结论：** BASELINE_ONLY


#### G04-10 · ASTER membrane persistence

定位：18。N/F/H/T：**1 / 2 / 2 / 1**。

- **原机制：** 神经元内部保存T状态，仅向外发送二值fire

- **最近先验与访问状态：** [ASTER](https://arxiv.org/abs/2511.06770)；NOT_ACCESSED; 本地引文

- **冻结适配：** 只有局部驻留原则可借；analog PIM不在现有路线

- **真正增量：** 类型/速率转换；尚未给新神经元执行机制

- **代价与反证：** T10不能从T≤4 LIF muxunroll推导廉价；内部非经典递推膜

- **迁移方式：** 重写PSN完整时间矩阵服务，二值边界为事实非创新

- **结论：** SUPPORT_ONLY_CORRECT_SEMANTICS


#### G04-11 · Chen/Chang mux-unroll

定位：19。N/F/H/T：**2 / 8 / 7 / 2**。

- **原机制：** 不同时间长度分流并在二值出口匹配速率

- **最近先验与访问状态：** [Chen/Chang Spike-IAND-Former](https://arxiv.org/abs/2503.19643)；NOT_ACCESSED_THIS_AUDIT; T≤4 unroll 来自本地摘录

- **冻结适配：** T2是window、T10是PSN服务；正确对象不同

- **真正增量：** 仅混合长度适配

- **代价与反证：** 不是首个mixed-T；mux数量、跨T等待、BN屏障计费

- **迁移方式：** 配套组织

- **结论：** SUPPORT_ONLY


#### G04-12 · ITA shift normalize

定位：20。N/F/H/T：**1 / 5 / 8 / 1**。

- **原机制：** 把score或gate当纯2的幂以省乘法

- **最近先验与访问状态：** [ITA](https://arxiv.org/abs/2307.03493)；NOT_ACCESSED; 本地引文

- **冻结适配：** 固定整数部署中移位/舍入有用；不能说冻结无归一化

- **真正增量：** 通用算术实现

- **代价与反证：** Q7部署与FP算术不同；移位不消除所有门乘法

- **迁移方式：** 仅固定舍入/移位叶工程

- **结论：** SUPPORT_ONLY


#### G04-13 · SpinalFlow temporal-code membrane removal

定位：21。N/F/H/T：**1 / 0 / 4 / 1**。

- **原机制：** 神经元内部保存T状态，仅向外发送二值fire

- **最近先验与访问状态：** SpinalFlow, ISCA 2020；NOT_ACCESSED_THIS_AUDIT; 本地表格引文

- **冻结适配：** 时间戳编码不等于满秩PSN输出

- **真正增量：** 类型/速率转换；尚未给新神经元执行机制

- **代价与反证：** 改变编码/神经元，不能移除A*x所需T输入

- **迁移方式：** 重写PSN完整时间矩阵服务，二值边界为事实非创新

- **结论：** REJECT


#### G04-14 · ExSpike adjacent-position compression

定位：22。N/F/H/T：**2 / 7 / 7 / 2**。

- **原机制：** 按连续dirty位置打包请求

- **最近先验与访问状态：** [ExSpike](https://arxiv.org/abs/2606.20414)；NOT_ACCESSED; 本地引文

- **冻结适配：** 可改变独立请求顺序，不删消费者

- **真正增量：** dirty标签替代普通活动标签

- **代价与反证：** RLE/队列先验普通；短run和尾部会抵消；无ep34分布

- **迁移方式：** 作为配对score服务配套，不单列主机制

- **结论：** SUPPORT_ONLY


#### G04-15 · Comperity AND-base / XOR-diff

定位：23。N/F/H/T：**1 / 6 / 5 / 2**。

- **原机制：** 完整行parent+源残差复用

- **最近先验与访问状态：** [Comperity](https://api.crossref.org/works/10.1145/3828526)；PUBLISHER_METADATA_ONLY_IN_LOCAL_RECORD; ACM 全文访问受阻，不能宣称完成实现级排除

- **冻结适配：** 现有部署底座合法，非新方向

- **真正增量：** 与Motion-XOR不同；与C1/C2共享组合是近邻

- **代价与反证：** ACM全文未取得，禁止凭题名排除归约近邻

- **迁移方式：** 作为强对照与底座

- **结论：** PRIOR_COMPARISON_GAP


#### G04-16 · Sparse HW for Spike-driven Transformer

定位：24。N/F/H/T：**1 / 7 / 8 / 1**。

- **原机制：** 二值激活令乘积退化为W或0，构建父和残差

- **最近先验与访问状态：** [Sparse Hardware Accelerator for Spike-driven Transformer](https://arxiv.org/abs/2501.07825)；NOT_ACCESSED_THIS_AUDIT; 本地引文

- **冻结适配：** 整数部署与阈值folding需固定尺度；FP重排并非逐位等价

- **真正增量：** product-sparsity本来就用于二值SNN；换成add-forest无新增

- **代价与反证：** 95个theta恰1也不等于所有theta可无误差折叠；TSBG已共享单项

- **迁移方式：** 作为执行合同，不保留重命名主张

- **结论：** BASELINE_ONLY


#### G04-17 · Phi pattern-wise products

定位：25。N/F/H/T：**2 / 7 / 6 / 3**。

- **原机制：** 完整行parent+源残差复用

- **最近先验与访问状态：** [Phi, ISCA 2025](https://arxiv.org/html/2505.10909v1)；PRIMARY_FULLTEXT_SECTIONS_READ; §2.4/3.1，分区模式表、双向校正、无模式回退已核

- **冻结适配：** 现有部署底座合法，非新方向

- **真正增量：** 无损模式结果表不同于输入行parent，不应笼统因有微调整项杀掉

- **代价与反证：** 原文§3.1双向残差及fallback已完整覆盖基本代数；查表成本仍大

- **迁移方式：** 作为强对照与底座

- **结论：** BASELINE_OR_RESEARCH_NEIGHBOR


#### G04-18 · CBinfer dirty pixels

定位：31。N/F/H/T：**3 / 2 / 4 / 3**。

- **原机制：** 无新事件的tile复用上次深层输出

- **最近先验与访问状态：** CBinfer, Cavigelli et al.；NOT_ACCESSED_THIS_AUDIT; 用DeltaCNN一手相关工作交叉定位，不称全文已读

- **冻结适配：** 原始输入不变不保证含BN/窗口/PSN依赖的输出不变

- **真正增量：** 普通CBinfer/delta推到网络

- **代价与反证：** no_running BN造成全域耦合；卷积halo/跨窗/跨时依赖；FP长期误差

- **迁移方式：** 必须新全局消费者证书；参见 BNREFINE，不直接免算

- **结论：** REJECT_SIMPLE_VERSION


#### G04-19 · Delta networks / DeltaCNN

定位：32。N/F/H/T：**3 / 2 / 4 / 3**。

- **原机制：** 无新事件的tile复用上次深层输出

- **最近先验与访问状态：** [DeltaCNN, Parger et al., CVPR 2022](https://arxiv.org/html/2203.03996v2)；PRIMARY_FULLTEXT_SECTIONS_READ; §3.1、§4、§8.4，静态融合 BN 与累计 FP 误差已核

- **冻结适配：** 数学epsilon=0也不保证原FP位等价，BN静态前提与ep34不同

- **真正增量：** 普通CBinfer/delta推到网络

- **代价与反证：** no_running BN造成全域耦合；卷积halo/跨窗/跨时依赖；FP长期误差

- **迁移方式：** 必须新全局消费者证书；参见 BNREFINE，不直接免算

- **结论：** REJECT_SIMPLE_VERSION


#### G04-20 · SparTen bitmask join

定位：33。N/F/H/T：**1 / 6 / 6 / 1**。

- **原机制：** 将T位向量压成非静默fiber并inner-join

- **最近先验与访问状态：** SparTen, MICRO 2019；NOT_ACCESSED; 本地表格引文

- **冻结适配：** 二值T2/T10可分开封包；不能照搬LoAS LIF/reset或稀疏W

- **真正增量：** 仅可借prefix/bitmask，LoAS已SNN化

- **代价与反证：** 共静默不能从删掉的坐标丢失；索引/merge开销与1RW端口未核

- **迁移方式：** 只作配套索引组织；强对照直接bitset/popcount

- **结论：** SUPPORT_ONLY


#### G04-21 · GoSPA on-the-fly intersection

定位：34。N/F/H/T：**1 / 6 / 5 / 1**。

- **原机制：** 将T位向量压成非静默fiber并inner-join

- **最近先验与访问状态：** GoSPA, ISCA 2021；NOT_ACCESSED; 本地表格引文

- **冻结适配：** 二值T2/T10可分开封包；不能照搬LoAS LIF/reset或稀疏W

- **真正增量：** 通常交集实现替换；mixed-T本身薄

- **代价与反证：** 共静默不能从删掉的坐标丢失；索引/merge开销与1RW端口未核

- **迁移方式：** 只作配套索引组织；强对照直接bitset/popcount

- **结论：** SUPPORT_ONLY


#### G04-22 · SCNN sparse Cartesian product

定位：35。N/F/H/T：**1 / 3 / 4 / 1**。

- **原机制：** 将T位向量压成非静默fiber并inner-join

- **最近先验与访问状态：** SCNN；NOT_ACCESSED; 本地表格引文

- **冻结适配：** spike稀疏可用，frozen W稀疏不足

- **真正增量：** 不同fiber宽度和Motion项服务

- **代价与反证：** 双稀疏路由开销；无全新稀疏来源

- **迁移方式：** 只作配套索引组织；强对照直接bitset/popcount

- **结论：** NO_HEADLINE


#### G04-23 · Eyeriss row stationary

定位：36。N/F/H/T：**1 / 9 / 8 / 2**。

- **原机制：** group-major共享weight row并更新独立目的

- **最近先验与访问状态：** [Eyeriss v2, JETCAS 2019](https://people.csail.mit.edu/emer/media/papers/2019.04.jetcas.eyeriss_v2.pdf)；NOT_ACCESSED_THIS_AUDIT; 本地作者全文链接

- **冻结适配：** 适配自然+1二值流，协议sign定向覆盖

- **真正增量：** 实现前请求抑制/共享控制，未形成新计算图

- **代价与反证：** 广播/loop交换先验强；低复用能量反向；不能用单K1

- **迁移方式：** 作为公平基线；新贡献要减少归约或证明免算

- **结论：** BASELINE_ONLY


#### G04-24 · GANAX inserted-zero skip

定位：37。N/F/H/T：**1 / 8 / 7 / 1**。

- **原机制：** 不计算插入的零，调整并行映射

- **最近先验与访问状态：** GANAX / Chang ConvTranspose；NOT_ACCESSED_THIS_AUDIT; 收益否决依据本地映射合同，非新的先验性能核证

- **冻结适配：** 数学可无损，但冻结96lane当前已满

- **真正增量：** 现有工作映射未见新增机会

- **代价与反证：** 本地封存上限≤1需主代理核；不能另编第三加速

- **迁移方式：** 实现基线支持

- **结论：** NO_NEW_ISLAND


#### G04-25 · FEATHER reorder-in-reduction

定位：38。N/F/H/T：**2 / 6 / 4 / 2**。

- **原机制：** 归约过程中重排数据，避免单独搬运阶段

- **最近先验与访问状态：** [FEATHER, ISCA 2024](https://arxiv.org/abs/2405.13170)；NOT_ACCESSED_THIS_AUDIT; 本地引文

- **冻结适配：** 适配自然+1二值流，协议sign定向覆盖

- **真正增量：** C2本地reduce→destination布局若有真实transpose瓶颈可用

- **代价与反证：** 整套可重构阵列过大，当前TSBG未证有该瓶颈

- **迁移方式：** 作为公平基线；新贡献要减少归约或证明免算

- **结论：** SUPPORT_ONLY


#### G04-26 · OpenEye sparse stream / variable FIFO

定位：39。N/F/H/T：**1 / 7 / 6 / 1**。

- **原机制：** 按连续dirty位置打包请求

- **最近先验与访问状态：** OpenEye；NOT_ACCESSED; 本地表格引文

- **冻结适配：** 精确流接口可作decoder支持

- **真正增量：** 一般队列组织

- **代价与反证：** 变长队列/回压/尾包常规实现，未给H67特有机制

- **迁移方式：** 作为配对score服务配套，不单列主机制

- **结论：** SUPPORT_ONLY


#### G04-27 · SpAtten attention-mass cascade

定位：40。N/F/H/T：**1 / 1 / 4 / 1**。

- **原机制：** 以低成本活动上界判断attention计算是否必需

- **最近先验与访问状态：** SpAtten, HPCA 2021；NOT_ACCESSED; 本地表格引文

- **冻结适配：** 剪token/head改冻结函数

- **真正增量：** attention-mass阈值非新

- **代价与反证：** 分母与AEE；不能和无损候选混用

- **迁移方式：** 仅在证明最终门控/输出不变时跳；否则界只是排序/预取

- **结论：** NEW_MODEL_ONLY


#### G04-28 · ASNA-Flow spatial locality

定位：46。N/F/H/T：**1 / 3 / 6 / 1**。

- **原机制：** 空间活动簇限定被执行的邻域/tile

- **最近先验与访问状态：** [ASNA-Flow, Wang et al., TVLSI 2025](https://ieeexplore.ieee.org/document/11142472/)；ACCESS_BLOCKED; 出版记录已识别，未取得可核全文；不引用本地芯片数值

- **冻结适配：** 仅跳过已证明无贡献项合法，不能限制固定450token窗口

- **真正增量：** ASNA空间局部性改任务标签

- **代价与反证：** ASNA已占近邻；动态BN/共静默分母使空位置仍有消费者

- **迁移方式：** 实际位图可作执行元数据；不是论文原语

- **结论：** BASELINE_ONLY


#### G04-29 · Zhang CICC DLSS

定位：47。N/F/H/T：**2 / 1 / 4 / 2**。

- **原机制：** 相邻时刻特征相似时跳深U-Net

- **最近先验与访问状态：** Zhang et al., optical-flow processor, CICC 2026；LOCAL_RESEARCH_NOTE_ONLY; docs/228 指向原文；本审阅不把其数值当新验证

- **冻结适配：** 近似跳层改变ep34

- **真正增量：** 可抽出时间相似作exact dirty候选

- **代价与反证：** 原AEE退化非无损；完整T/BN使局部相似不够

- **迁移方式：** 只取精确score依赖统计，不复制近似decoder跳

- **结论：** REJECT_AS_WRITTEN


#### G04-30 · Zhang CICC BWAC

定位：47。N/F/H/T：**1 / 6 / 7 / 1**。

- **原机制：** 小组bitmap描述非零权重

- **最近先验与访问状态：** Zhang et al., optical-flow processor, CICC 2026；LOCAL_RESEARCH_NOTE_ONLY; docs/228 指向原文；本审阅不把其数值当新验证

- **冻结适配：** 可无损编码部署INT8，不能称冻结模型PPA

- **真正增量：** 普通编码映射

- **代价与反证：** 本地记录稀疏率低且bitmap扩张；数字须主代理封存核

- **迁移方式：** 作为被更强压缩方案对比的基线

- **结论：** LIKELY_NO_GAIN_SUPPORT


#### G04-31 · Zhang CICC MaxPool/ReLU speculation

定位：47。N/F/H/T：**2 / 0 / 3 / 1**。

- **原机制：** 用nonlinearity特性推测冗余输出

- **最近先验与访问状态：** Zhang et al., optical-flow processor, CICC 2026；LOCAL_RESEARCH_NOTE_ONLY; docs/228 指向原文；本审阅不把其数值当新验证

- **冻结适配：** Shiftmax共分母不满足同类跳过条件

- **真正增量：** 转用需要重新证明

- **代价与反证：** 静默项仍改变分母；local note原文访问欠缺

- **迁移方式：** 不移植原speculation；可启发BNREFINE但需全新证明

- **结论：** REJECT


#### G04-32 · SpiDR zero skip/CIM

定位：48。N/F/H/T：**2 / 2 / 2 / 1**。

- **原机制：** 按精度/规模重配置，零跳过并异步衔接 CIM

- **最近先验与访问状态：** [SpiDR, 2024 preprint](https://arxiv.org/html/2411.02854v1)；PRIMARY_FULLTEXT_SECTIONS_READ; §II-A–F, 10T macro、dual-port IFspad、IF/LIF/reset 已核

- **冻结适配：** 不同精度、head 分流和 IF/LIF 不属于冻结模型

- **真正增量：** 按金字塔选择模式属于应用映射

- **代价与反证：** 定制 10T、dual-port IFspad、精度改变；无真实 AEE/宏路径

- **迁移方式：** 只借任务粒度配置与有限 FIFO；不得搬 silicon PPA

- **结论：** REJECT_AS_HEADLINE


#### G04-33 · Michigan dense optical-flow chip

定位：49。N/F/H/T：**0 / 1 / 0 / 0**。

- **原机制：** dense frame-OF silicon comparison template

- **最近先验与访问状态：** Michigan 28nm dense optical-flow chip；NOT_ACCESSED; 未给完整题名，原数值不准入

- **冻结适配：** 输入/模型不同，不可直接比FPS/能耗

- **真正增量：** 无候选机制，仅评价参照

- **代价与反证：** 原始芯片与本项目组件范围/工艺/工作负载不同

- **迁移方式：** 只借评价维度

- **结论：** REFERENCE_ONLY


#### G04-34 · Ultra-Flow direction prediction

定位：50。N/F/H/T：**3 / 2 / 5 / 3**。

- **原机制：** 占据位图移位并比较速度假设，预测昂贵路径是否执行

- **最近先验与访问状态：** Ultra-Flow / Liu TCAS-I 2025；NOT_ACCESSED; 本地表格引文

- **冻结适配：** 可旁路预测调度顺序；按置信度省结果会改ep34

- **真正增量：** 便宜先验接昂贵网络的层级组合

- **代价与反证：** 错预测不能漏算；新增hyp非冻结；FPGA指标未复核

- **迁移方式：** 若只提前取数/排序且不丢任务可保图，但收益更薄

- **结论：** RESEARCH_ONLY_REDEFINE


#### G04-35 · ERAFT prediction

定位：51。N/F/H/T：**2 / 0 / 4 / 1**。

- **原机制：** 预测后续RAFT迭代无用而早退

- **最近先验与访问状态：** [ERAFT, frame RAFT FPGA, ISCAS 2025](https://doi.org/10.1109/ISCAS56072.2025.11043529)；NOT_ACCESSED; 不能混为 E-RAFT

- **冻结适配：** ep34没有迭代update block；T=10也不是可随时结束的RAFT迭代

- **真正增量：** 将迭代exit嫁接PSN

- **代价与反证：** fullrankPSN未读输入仍影响各步；freeze无exit训练

- **迁移方式：** 若保留必须改成严格二值消费者证明，不是残差阈值

- **结论：** REJECT


#### G04-36 · SENECA spike sparsity

定位：52。N/F/H/T：**1 / 3 / 3 / 1**。

- **原机制：** live-mask抑制空操作

- **最近先验与访问状态：** [SENECA event optical flow](https://arxiv.org/abs/2407.20421)；NOT_ACCESSED; 本地引文

- **冻结适配：** 源乘权的真零可以跳；attention分母不可按空QK删除

- **真正增量：** 平台映射，非1RW电路创新

- **代价与反证：** 成熟baseline；firing并非合法免算比例

- **迁移方式：** 强baseline必须包括

- **结论：** BASELINE_ONLY


#### G04-37 · SDformerFlow → H67 first digital mapping

定位：58。N/F/H/T：**3 / 8 / 6 / 4**。

- **原机制：** AND overlap、共静默、K与时间peer的XOR合成score

- **最近先验与访问状态：** [SDformerFlow](https://arxiv.org/abs/2409.04082)；LOCAL_ALGORITHM_SOURCE_READ; 公开论文全文本次未完整重读，不把upstream同H67混同

- **冻结适配：** 算子结构适配；原文Q7 /64、/4不是冻结FP的0.02、0.125

- **真正增量：** 特定新算法的首份数字实现是机会，尚不自动有性能机制

- **代价与反证：** 三popcount不够顶级主张；同功能对照需完整score+归一化；attention份额未测

- **迁移方式：** 可作H67数字映射基线；重点另找peer/value服务机制

- **结论：** BASELINE_FOR_NEW_ISLAND


#### G04-38 · Spike-driven Transformer mask+add

定位：59。N/F/H/T：**1 / 7 / 8 / 1**。

- **原机制：** 二值激活令乘积退化为W或0，构建父和残差

- **最近先验与访问状态：** Spike-driven Transformer, NeurIPS 2023；NOT_ACCESSED_THIS_AUDIT; 多个FPGA prior已列，不宣称首次mapping

- **冻结适配：** 整数部署与阈值folding需固定尺度；FP重排并非逐位等价

- **真正增量：** product-sparsity本来就用于二值SNN；换成add-forest无新增

- **代价与反证：** 95个theta恰1也不等于所有theta可无误差折叠；TSBG已共享单项

- **迁移方式：** 作为执行合同，不保留重命名主张

- **结论：** BASELINE_ONLY


#### G04-39 · A²OS²A

定位：60。N/F/H/T：**1 / 0 / 3 / 1**。

- **原机制：** 二值attention引擎与ATLIF实值payload引擎分离

- **最近先验与访问状态：** [A²OS²A](https://arxiv.org/abs/2503.00226)；NOT_ACCESSED; 本地表格引文

- **冻结适配：** binary Q/ReLU K/ternary V与H67二值K作V不同

- **真正增量：** 换编码不在冻结范围

- **代价与反证：** 改变注意力/激活；无softmax类比不适用

- **迁移方式：** 只借分离score/value/linear服务，完整同功能MX baseline

- **结论：** REJECT


#### G04-40 · α-XNOR co-silence

定位：61。N/F/H/T：**1 / 7 / 8 / 1**。

- **原机制：** 给共同静默位置非零分数

- **最近先验与访问状态：** α-XNOR Spiking Self-Attention, Xiao et al., CVPR 2025；SOURCE_IDENTIFIED_BUT_FULLTEXT_NOT_ACCESSED_THIS_SUBAUDIT; URL 待主代理一手记录补齐

- **冻结适配：** 是H67已存在项，不是新硬件点

- **真正增量：** 加权项映射已知

- **代价与反证：** α符号多义；不得用论文0.3/0.5或Q7 1/64替换ep34 0.02

- **迁移方式：** 可作H67数字映射基线；重点另找peer/value服务机制

- **结论：** ALGORITHM_PRIOR


#### G04-41 · AT-LIF adaptive threshold

定位：62。N/F/H/T：**1 / 9 / 7 / 1**。

- **原机制：** 训练期阈值自适应、推理固定theta二值输出

- **最近先验与访问状态：** AT-LIF, NeurIPS 2025；LOCAL_FROZEN_CODE_AND_CONFIG_READ; 出版方全文URL未核，本子审阅用源码证明输出

- **冻结适配：** 固定输出码成立

- **真正增量：** 已训练算子身份，不是新电路

- **代价与反证：** 不能把训练homeostasis称runtime硬件

- **迁移方式：** 重写PSN完整时间矩阵服务，二值边界为事实非创新

- **结论：** IDENTITY_REFERENCE


#### G04-42 · TMA / E-RAFT / BAT / EVA-Flow / IDNet

定位：63。N/F/H/T：**1 / 0 / 2 / 0**。

- **原机制：** 增加cost volume、lookup、迭代流更新

- **最近先验与访问状态：** [TMA（同列其余算法全文未逐一访问）](https://arxiv.org/html/2303.11629v2)；PRIMARY_FULLTEXT_SECTIONS_READ; §3.2/式4–6，RAFT 与 correlation volumes 已核

- **冻结适配：** 冻结为U-Net flow head，无该图

- **真正增量：** 原表把不同算法列作参考；本审阅仅对引入correlation/anytime算子否决，不声称这些论文原理相同

- **代价与反证：** 新projection/lookup/状态/训练；H81禁令不可绕

- **迁移方式：** 只作外部先验文献

- **结论：** REJECT


#### G04-43 · PSN parallel time matrix

定位：64。N/F/H/T：**3 / 9 / 6 / 4**。

- **原机制：** 满秩时间矩阵 A*x+b 后逐值阈值化，无reset

- **最近先验与访问状态：** Parallel Spiking Neurons, NeurIPS 2023；LOCAL_IMPLEMENTATION_READ; fullrank addmm/no reset 源码已核，不挪GPU收益

- **冻结适配：** 正是冻结核心；不是LIF递推

- **真正增量：** 完整T数据复用/编译有研究空间，但一般矩阵乘已有

- **代价与反证：** 无低秩保证，FP关联律不成立；GPU加速非电路加速

- **迁移方式：** 重写PSN完整时间矩阵服务，二值边界为事实非创新

- **结论：** RESEARCH_OBJECT


### research/grok46_20260905/05_profile_evidence.md

精读完成；SHA256 `6368a441c8c6438ce887683f7f12ce5d131e3bedefc8eabcad1260567700a4cb`。


### research/grok46_20260905/06_two_workflow_conflict.md

精读完成；SHA256 `f8589e64581853786e0e28ccff99ae5213c44c0606cdcbed71665d41db136b2e`。


#### G06-A1 · MX3P 三 popcount Motion-XOR

定位：7;40。N/F/H/T：**3 / 7 / 8 / 4**。

- **原机制：** AND overlap、共静默、K与时间peer的XOR合成score

- **最近先验与访问状态：** [FireFly-T](https://arxiv.org/abs/2505.12771)；PRIMARY_FULLTEXT_RELEVANT_SECTIONS_READ_IN_PRECEDING_RESEARCH; FPGA attention/transpose，未复核期刊年份

- **冻结适配：** 算子结构适配；原文Q7 /64、/4不是冻结FP的0.02、0.125

- **真正增量：** 增加既有位运算项与peer供给；尚无超过直接实现的机制

- **代价与反证：** 三popcount不够顶级主张；同功能对照需完整score+归一化；attention份额未测

- **迁移方式：** 可作H67数字映射基线；重点另找peer/value服务机制

- **结论：** BASELINE_FOR_NEW_ISLAND


#### G06-A2 · Mixed-horizon binary ATLIF / membrane firewall

定位：7;40。N/F/H/T：**2 / 6 / 6 / 3**。

- **原机制：** 神经元内部保存T状态，仅向外发送二值fire

- **最近先验与访问状态：** [Chen/Chang Spike-IAND-Former](https://arxiv.org/abs/2503.19643)；NOT_ACCESSED_THIS_AUDIT; T≤4 unroll 来自本地摘录

- **冻结适配：** 出口二值正确；原文leak/reset错误，实际fullrank A*x+b

- **真正增量：** 类型/速率转换；尚未给新神经元执行机制

- **代价与反证：** T10不能从T≤4 LIF muxunroll推导廉价；内部非经典递推膜

- **迁移方式：** 重写PSN完整时间矩阵服务，二值边界为事实非创新

- **结论：** SUPPORT_ONLY_CORRECT_SEMANTICS


#### G06-A3 · LoAS FTP mixed-T fiber join

定位：9。N/F/H/T：**2 / 6 / 5 / 3**。

- **原机制：** 将T位向量压成非静默fiber并inner-join

- **最近先验与访问状态：** [LoAS, MICRO 2024](https://arxiv.org/abs/2407.14073)；PRIMARY_FULLTEXT_RELEVANT_SECTIONS_READ_IN_PRECEDING_RESEARCH; LIF reset 与 temporal fibers；不等同 PSN

- **冻结适配：** 二值T2/T10可分开封包；不能照搬LoAS LIF/reset或稀疏W

- **真正增量：** 不同fiber宽度和Motion项服务

- **代价与反证：** 共静默不能从删掉的坐标丢失；索引/merge开销与1RW端口未核

- **迁移方式：** 只作配套索引组织；强对照直接bitset/popcount

- **结论：** SUPPORT_ONLY


#### G06-A4 · T2/T10 rate conversion

定位：9;48。N/F/H/T：**2 / 8 / 7 / 2**。

- **原机制：** 不同时间长度分流并在二值出口匹配速率

- **最近先验与访问状态：** [Chen/Chang Spike-IAND-Former](https://arxiv.org/abs/2503.19643)；NOT_ACCESSED_THIS_AUDIT; T≤4 unroll 来自本地摘录

- **冻结适配：** T2是window、T10是PSN服务；正确对象不同

- **真正增量：** 仅混合长度适配

- **代价与反证：** 不是首个mixed-T；mux数量、跨T等待、BN屏障计费

- **迁移方式：** 配套组织

- **结论：** SUPPORT_ONLY


#### G06-A5 · Bishop bound-without-S

定位：9;24。N/F/H/T：**4 / 4 / 5 / 4**。

- **原机制：** 以低成本活动上界判断attention计算是否必需

- **最近先验与访问状态：** [Bishop, ISCA 2025](https://arxiv.org/abs/2505.12281)；PRIMARY_FULLTEXT_RELEVANT_SECTIONS_READ_IN_PRECEDING_RESEARCH; TTB、AAC、ECP；不把其训练边界移入 ep34

- **冻结适配：** 二值Q/K允许界；不能只界overlap就删含静默/motion项的score

- **真正增量：** 针对完整Motion和分母的输出不变判据尚缺

- **代价与反证：** 低score仍进共享分母；BSA/ECP训练改模型；Q7舍入边界

- **迁移方式：** 仅在证明最终门控/输出不变时跳；否则界只是排序/预取

- **结论：** RESEARCH_ONLY_NEEDS_PROOF


#### G06-A6 · CICC DLSS temporal similarity

定位：9;28。N/F/H/T：**2 / 1 / 4 / 2**。

- **原机制：** 相邻时刻特征相似时跳深U-Net

- **最近先验与访问状态：** Zhang et al., optical-flow processor, CICC 2026；LOCAL_RESEARCH_NOTE_ONLY; docs/228 指向原文；本审阅不把其数值当新验证

- **冻结适配：** 近似跳层改变ep34

- **真正增量：** 可抽出时间相似作exact dirty候选

- **代价与反证：** 原AEE退化非无损；完整T/BN使局部相似不够

- **迁移方式：** 只取精确score依赖统计，不复制近似decoder跳

- **结论：** REJECT_AS_WRITTEN


#### G06-B1 · ELSA elastic first-response

定位：27。N/F/H/T：**2 / 2 / 5 / 2**。

- **原机制：** token逐层弹性先到先算

- **最近先验与访问状态：** [ELSA, ISCA 2026](https://arxiv.org/html/2605.20802v1)；NOT_ACCESSED_THIS_SUBAUDIT; 动态 BN 屏障由本地源码独立确认

- **冻结适配：** 24 dynamic BN需整域统计；fullrankPSN需完整T输入

- **真正增量：** 仅局部队列打包可保留

- **代价与反证：** 分类first-correct不能变成denseflow完整结果；无全网闭环

- **迁移方式：** 只保局部ready/valid与BAER，不写弹性网络加速

- **结论：** REJECT_SYSTEM_CLAIM


#### G06-B2 · CICC MaxPool/ReLU redundancy speculation

定位：28。N/F/H/T：**2 / 0 / 3 / 1**。

- **原机制：** 用nonlinearity特性推测冗余输出

- **最近先验与访问状态：** Zhang et al., optical-flow processor, CICC 2026；LOCAL_RESEARCH_NOTE_ONLY; docs/228 指向原文；本审阅不把其数值当新验证

- **冻结适配：** Shiftmax共分母不满足同类跳过条件

- **真正增量：** 转用需要重新证明

- **代价与反证：** 静默项仍改变分母；local note原文访问欠缺

- **迁移方式：** 不移植原speculation；可启发BNREFINE但需全新证明

- **结论：** REJECT


#### G06-B3 · CICC BWAC bitmap weight compression

定位：28。N/F/H/T：**1 / 6 / 7 / 1**。

- **原机制：** 小组bitmap描述非零权重

- **最近先验与访问状态：** Zhang et al., optical-flow processor, CICC 2026；LOCAL_RESEARCH_NOTE_ONLY; docs/228 指向原文；本审阅不把其数值当新验证

- **冻结适配：** 可无损编码部署INT8，不能称冻结模型PPA

- **真正增量：** 普通编码映射

- **代价与反证：** 本地记录稀疏率低且bitmap扩张；数字须主代理封存核

- **迁移方式：** 作为被更强压缩方案对比的基线

- **结论：** LIKELY_NO_GAIN_SUPPORT


#### G06-B4 · correlation / RAFT pyramid residual island

定位：29。N/F/H/T：**1 / 0 / 2 / 0**。

- **原机制：** 增加cost volume、lookup、迭代流更新

- **最近先验与访问状态：** [TMA, Liu et al., ICCV 2023](https://arxiv.org/html/2303.11629v2)；PRIMARY_FULLTEXT_SECTIONS_READ; §3.2/式4–6，RAFT 与 correlation volumes 已核

- **冻结适配：** 冻结为U-Net flow head，无该图

- **真正增量：** 新模型不属于C1/C2重构

- **代价与反证：** 新projection/lookup/状态/训练；H81禁令不可绕

- **迁移方式：** 只作外部先验文献

- **结论：** REJECT


#### G06-B5 · GANAX ConvTranspose inserted-zero skip

定位：30。N/F/H/T：**1 / 8 / 7 / 1**。

- **原机制：** 不计算插入的零，调整并行映射

- **最近先验与访问状态：** [Sparseloop, MICRO 2022](https://sparseloop.mit.edu/documents/2022-micro-sparseloop.pdf)；NOT_ACCESSED_THIS_AUDIT; 模型不是 RTL 证据

- **冻结适配：** 数学可无损，但冻结96lane当前已满

- **真正增量：** 现有工作映射未见新增机会

- **代价与反证：** 本地封存上限≤1需主代理核；不能另编第三加速

- **迁移方式：** 实现基线支持

- **结论：** NO_NEW_ISLAND


### research/grok46_20260905/07_next_stats_rtl_gates.md

精读完成；SHA256 `ce4bdc1c118e87559683dae9eeb2cfd4688f04f7a283c7c28663e323ec652078`。


### research/grok46_20260905/INDEX.json

精读完成；SHA256 `6658f44ff430fc476ba50fe7b838860fc0b540390b43c31c65acf710749e7f31`。


### research/m2067_handshake_fix_status.md

精读完成；SHA256 `618bb04773e5e2cab5bb950b39040f8222fd6adb9c0ba2cc6a91d4261e303740`。

全文189行；当前接手禁令优先，未运行scratch、VCS或提议补丁。


#### ROOT-HS-01 · sticky accept采样修复

定位：第2–3节。N/F/H/T：**— / — / — / —**。

- **原机制：** 通过sticky捕获及稳定header避免TB晚采样丢握手

- **最近先验与访问状态：** 对应原始机制记录；本条为入口/工程证据；本地全文

- **冻结适配：** 以二值 ep34/no_running BN/full-rank PSN 为准

- **真正增量：** 验证协议修复，不是加速机制

- **代价与反证：** 笔记的scratch重现不等于真实quarantine根因完整闭环；不能由此重跑旧identity

- **迁移方式：** 保留诊断；禁止FAILED_DO_NOT_CITE_NO_RETRY续跑

- **结论：** 验证史料，不作创新与性能证据


#### ROOT-HS-02 · 新identity+独立hammer续跑计划

定位：第4–7节。N/F/H/T：**— / — / — / —**。

- **原机制：** 修改TB后重新封合同并独立核验

- **最近先验与访问状态：** 对应原始机制记录；本条为入口/工程证据；本地全文

- **冻结适配：** 以二值 ep34/no_running BN/full-rank PSN 为准

- **真正增量：** 验证流程，无新计算原语

- **代价与反证：** 本轮不是修旧仿真任务；笔记的下一步不构成本轮生产授权

- **迁移方式：** 不执行，不改wrapper legality，不复活失败路径

- **结论：** 历史计划，不执行

