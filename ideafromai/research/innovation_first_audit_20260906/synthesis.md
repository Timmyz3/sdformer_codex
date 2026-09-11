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
