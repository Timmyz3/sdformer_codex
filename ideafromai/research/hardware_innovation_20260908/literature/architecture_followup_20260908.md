# GustavSNN 后续架构借鉴：围绕实际供数与状态约束

2026-09-08。本轮收敛四篇，完整阅读 HYTE §5–6、SeaCache §4、HyMM §III–IV、Swift §IV，并核对其评价方法。**优先完整承接 HYTE 的分块与数据管理，再改 GustavSNN 的局部输出上下文；其余三篇按实际访存瓶颈选用。** 下列原文倍率属于各自平台与工作负载，不是本网络预测。

当前 C384 FC1 单输出权重行只有 384 B；8 KiB 容量利用不足首先来自只暴露一行输出上下文。类别表示仍保留真实 θg 的幅度合同；τ 是判决门限。普通时间行、onehot7、当前学生都应享有相同的分块、双缓冲、广播和端口优化。

**1. HYTE — ISCA 2025，最直接的完整调度底座。** 原问题是稀疏 tile 大小不规则，而固定形状、固定循环顺序和固定 A/B/C 存储分配产生重复取数。它先采样估计计算及中间输出量，联合选择形状、顺序和容量；硬件 accessor 管理 fiber 的边界位置、数据/元数据、驻留/流式模式与溢出旁路，再用四象限计数微调形状。相同 32 MAC、4 MB SRAM、68 GB/s 配置下，Gustavson 周期模拟**包含 CPU 调度时间**后，相对 DRT/HARP/Tailors 为 3.3×/4.5×/6.2×。定位 §5、§6、§7–8.1；[作者原文](https://people.iiis.tsinghua.edu.cn/~gaomy/pubs/hyte.isca25.pdf)、[作者代码](https://github.com/tsinghua-ideal/HYTE-sim)。

可完整借其搜索、metadata accessor、循环边界、旁路及分阶段计时。我们的新增能力是把输出行数 F 放进 PE 的交错上下文，按 **P×F×R** 类别状态与真实端口共同选择 P/F。原 §5.2 在固定 PE 数据流下会剪掉 Gustav 最外层输出分块；本改造改变 PE 内复用，不能照抄该剪枝。强控制是固定 GP 也充分搜索合法 P/F，并允许 R6/R7 各自选择最佳点；只让类别方案享受 F2 会制造弱分母。先测静态选择，动态调形只在跨 tile 差异带来净收益时启用。

**2. SeaCache — MICRO 2025，借有限元数据与访存组织。** 原问题是变长稀疏 fiber 无法填满固定 cache block，以及“知道未来重用”的替换策略自身占用过多存储。完整结构含短 fiber 合装、长 fiber 分段、ID 索引、gLFU 计数/virtual tag、预取元数据与实值数据共用容量及自适应分配。原相同 32 PE、2 MB cache 下，Gustavson 周期模拟相对 X-Cache 2.8×、Tailors 式 scratchpad 2.1×；这些并非实芯片测速。定位 §4–6.1；[作者原文](https://people.iiis.tsinghua.edu.cn/~gaomy/pubs/seacache.micro25.pdf)、[作者代码](https://github.com/tsinghua-ideal/SeaCache-sim)。

可借 NRV 变长包的装填/分段与明确的预读窗口，原作已有的 virtual tag、额外 tag 端口、计数更新及跨 PE 合并请求不能省略；其只读 A/B 缓存不能直接当可丢弃的未完成 S 状态。**384 B 稠密 W 行未占满 8 KiB 不是 cache-line 碎片**；先用多行连续 scratchpad。若类别码已紧凑连续存放，动态 cache 可能徒增费用；强控制是相同容量/带宽的直接地址双缓冲。仅当实际重放含冷包占位或变长碎片时再迁移完整缓存机制。

**3. HyMM — DATE 2025，借数据流切换时的有限部分和管理。** 它对度排序后的图分区：高复用区域分别用 CSC 外积或 CSR 行积，低度区域用行积减少归并；同一 PE buffer 切换保存输入/输出，统一 DMB 旁加累加器，LSQ 做转发与等待。七个 GCN 图的周期模拟中，Amazon-Photo 相对纯外积最高 4.78×、片外访问减少 91%；不是对最强行积的统一倍率，排序成本另列。定位 §III、§IV-A–E、§V；[作者原文](https://filedn.com/luEeJVCCazShDlU4ibloXvu/publication/gcn_accel_date25/gcn_accel_date25.pdf)、[作者页面](https://csarch.korea.ac.kr/publication/gcn_accel_date25/)。

可以完整借输入/输出状态切换、near-buffer 累加、读写队列与依赖转发；必须付出额外累加器和端口。我们的类别更新不是静态 power-law 图：每个输入通道最多向 P 个位置发送，类别每帧变化，不能免费度排序。若采用多上下文，LSQ 还需区分 `(p,h,class)` 的未完成更新。强控制是给固定 GP 同样累加/转发能力，再判断混合数据流本身的增量；先验存在不妨碍迁移，但“换两种循环”不是充分收益原因。

**4. Swift — HPCA 2026，借协调布局与尾部任务组织。** 原问题是 GPU SpMM 只优化稀疏侧后，稠密侧仍不连续。它同时重排稀疏列和稠密行，划分规则 warp 块/不规则尾部；尾部拆长列均衡，规则块按目的行分段归约减少 atomic。原 RTX 4080s、FP64、N128 的 Table I 核执行几何均值相对 ASpT 为 1.79×；预处理另评，不能称含排序的端到端收益。定位 §IV-A–D、§V-A/G；[作者原文](https://ranger.uta.edu/~jiang/publication/Conferences/2026/HPCA26_camera_ready.pdf)、[HPCA 官方条目](https://2026.hpca-conf.org/details/hpca-2026-main-conference/42/Swift-High-Performance-Sparse-Dense-Matrix-Multiplication-on-GPUs)。

可借“源包与 W 采用同一通道排列”、规则块/尾部独立调度及本地归并，完整迁移时保留目的索引和冲突处理。我们的双口 W 已可按通道顺序读；GPU 合并访存不自动变成 ASIC 收益，逐帧重排还需付排序与搬运。强控制是训练后固定通道顺序、连续 W burst、普通 bank 交错；只有实际端口事务或同步尾部减少才值得加运行时重排。

**对当前 P8/F1→P4/F2 的具体结论。** 同为 56 个 S15/PE 时，onehot7 两点都是 PFR=56，F2 可以让一次源包服务两输出行；代价包括两行权重读取、更细 P 的索引、上下文控制、τ/输出包和同步尾部。R6 也必须允许相同 P/F 搜索。8 KiB 足够放几行 W 不代表这些行可同时读取，容量与端口分别计量。

当前“类归约后一次消费”组织的释放分两步：完整 C 到齐后得到完整类别和；所有需要它的时间行消费完后，才回收其槽。线性 B 可以提前消费部分和，但须另付更新次数/状态，最终判决仍要完整贡献。LIF 的逐 tick 退休不适用；零源类也不能略去 τ 产生的常量输出。若所有源已完整缓存在前级，可计算更早的精确完成边界，但元数据生产不是免费步骤。

对 C1 昂贵卷积，优先迁移同一套输出通道分块、有限 W 驻留和原位完成规则，并完整走九个 tap；邻域供数继续与普通滑窗缓冲比较。空间重排若破坏滑窗连续性，必须收费 halo、索引与输出恢复。当前未做卷积时间码训练，不能把 S2 学生的类别表示直接当成已覆盖 C1；本轮不恢复旧 Prosperity∪APEC 布局。
