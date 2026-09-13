# r0 卷积的稀疏与打包机制研究

优先候选是**普通 N:M 支持选择与有界 lane 发射的联合设计**。先把普通 2:4 的完整 r0 Conv2 做成强执行底座，再检验同一精度预算下，按实际系数字请求和 lane 尾长选支持能否获得额外收益。第二候选是**原生空间脉冲字的消费者闭包与提前供数判定**，用于查清哪些剪枝真正消除了源字、哪些只省了下层重放。两者都尚未证明净性能优势；独立队列、分层剪枝、时间打包、零检测均是借入机制。

本报告只有机制研究与下轮工单，没有新训练、CPU 机制筛选、RTL 仿真或 EDA。上轮 CPU 原型不是逐个写 RTL，更不构成这些候选已经被 RTL 否决。[完整来源主表](source_master.csv)列出九个 primary，均读到方法或微架构正文；SpiDR 按未核实最终 venue 的作者稿使用，不冒充 JSSC/ISSCC。本文页码均为归档 PDF 的页序。

## 当前昂贵切口与不能借用的结论

当前采样的是 matched-dense stage320 的实际前向。r0 Conv1 和 Conv2 均为 `T10 × C96 × H240 × W320`，权重 `N96 × C96 × 3 × 3`，每层名义 63.701 G MAC，占已采矩阵算术分母的 10.678%；输入非零率分别约 4.3602% 和 3.6127%。前端 `patch.conv.conv.0` 为 31.8505 G MAC。**这些比例只定位计算形状，不是 ASIC 周期占比。** S2 FC1 已经替换，不能拿旧 dense FC1 当未改造大头。[当前瓶颈](../../open_fusion_execution/major_operator_fusions_20260913/root_owned/current_bottlenecks.md) · [原始 profile](../../open_fusion_execution/major_operator_fusions_20260913/root_owned/profile.json)

本轮两个候选首先落到 r0 Conv2。原生 AT-LIF 源为 `{0,θ}`，推理 θ 可静态折入权重，事件只需 1 bit；不能把连续幅值负载重新塞回脉冲接口。r0 的 MS_ResBlock 路径是 `sn2 → conv2 → norm2 → + identity`，因此 Conv2 后仍有连续输出。首个算子 RTL 检查点可以是完整 Conv2 张量，但升格为完整 block 必须核对真实 norm2 和残差结果。非因果 T10 PSN 必须保留全部十时刻依赖，连续 PED 必须独立保存；这些约束主要关系到随后 r1/投影链迁移，不能把它们硬造为 r0 sn2 同一脉冲字上的两个消费者。[MS_ResBlock 源码](../../../../../SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py)

旧固定 `fill1:4` 已有真实失败点：r0 请求优化版局部服务 C 为 284,177，普通 2:4 为 231,001；diverse10 AEE 分别约 1.176247 和 1.159183。r1 完整已覆盖后继窗口里，fill 也未胜普通 2:4。这不能重命名成“自适应非零填充”重新申领胜出。另一方面，r0 表还没有结清源收集、im2col、包生成、仲裁和写出；H8 外循环造成 41,472 次 16 B 源包读取，含零包，源被十二组 H8 重放。另一份公共 walker 模型发现 `max(events)` 曾被误当发射长度，已改为公共 `slot × time` 的并集。这两个未闭合接口才是本轮切口。[旧稀疏结论](../../open_fusion_execution/major_operator_fusions_20260913/sparse_owned/README.md) · [独立审计](../../open_fusion_execution/major_operator_fusions_20260913/root_owned/independent_new_fusions.md)

质量门使用同任务、同帧、同 GT/有效像素和聚合口径的 NB0 比较。diverse10 NB0 为 1.45460286107；825 帧历史 NB0 为 1.44535253468，后者不是本轮新验证。允许有损，不要求相对当前父只退化 0.005。普通 2:4 已是很强质量控制；无残差 SVD8 等也已过新门，应作为相同任务质量附近的竞争路线。局部 L2、参数量、发放率都不能直接决定晋级。

## 四项独立假说的筛选

初始假说已单独固定在[独立假说记录](hypotheses_initial.md)。H1 是按原生源字及其 halo 消费者剪枝，H2 是有界 lane 私有压紧，H3 是 producer 原生包直接供卷积，H4 是由真实 prefix 产生 burst 级近似续算决定。

阅读后，H2 保留为第一候选。H1 与 H3 合成第二候选，但其中 producer 压缩、line buffer、分层 mask 都先归强 A。H4 暂不作为第三个创新候选：CGNet 已把真实 prefix、阈值、任务队列、条件卷积取数做成 RTL 架构；本项目旧 preview 也已有邻近接口。r0 输出连续，又没有“门确定即全部无用”的合法捷径。下轮可以实现 CGNet 式完整对照，但没有理由仅换成整字阈值便宣称新机制。

## 最近先验读到了什么

下表只总结与两个挂点直接相撞的机制。完整 title、venue、year、primaryURL、codeURL、read_scope 在[主表](source_master.csv)。未确认代码写为空，绝不等同于断言没有代码。

| Primary / 精读位置 | 已有能力与本轮必须承认的碰撞 |
|---|---|
| [Bishop，ISCA 2025](https://arxiv.org/pdf/2505.12281v1)，pp.4–8，§3.2、4.1、5.1、5.3–5.4 | TTB 在 token×time 维打包，活动 tag 由组内是否有脉冲决定；BSA 训练组稀疏；stratifier 记录 feature index 并对齐 W/输入；稀疏核采用 SIGMA-like 分发/归约，稠密核与稀疏核合并 psum。**分组、异构派发、权重复用都不新。** ECP 的二值 Q/K 界不直接证明 r0 连续残差可删。 |
| [LoAS，MICRO 2024](https://arxiv.org/pdf/2407.14073v3)，pp.5–8，§III、IV.A–D | 时间维放 inner loop，非静默神经元存完整时间位图；位图 AND、优先编码和前缀和求交；一个 fast、一个 laggy prefix-sum 配合 pseudo/correction accumulators、FIFO 和 FiberCache。**T10 打包、晚到操作数、缓存和零跳过均已有。** 其 LIF 后处理不等于本网 PSN。 |
| [HighLight，MICRO 2023](https://arxiv.org/pdf/2305.12718)，pp.5–9，§4–6 | 超普通 2:4 的两级 HSS、逐级坐标 metadata、两级 skipping、VFMU 对齐整字读取与变长流；输入 B 的非结构稀疏用于压缩与 gating，并未自动减少 PE 周期。**粗细两级剪枝和只在需要时发 GLB 请求已有。** |
| [S2TA，HPCA 2022](https://arxiv.org/pdf/2107.07983v2)，pp.4–7，§3–6 | 权重/激活 DBB；DAP 在线保留 top-NNZ；time-unrolled DP1M4 每周期消费一个有效激活，层间密度可变；TPE 与专用 maxpool 阵列已支付硬件。**按激活密度缩短执行及软硬约束稀疏不是空白。** 对 `{0,θ}` 同幅脉冲照搬幅值 top-k 没有辨识力。 |
| [CGNet，MICRO 2019](https://www.csl.cornell.edu/~zhiruz/pdfs/cgnet-micro2019.pdf)，pp.3–5，§2、3.1–3.3 | 真正计算 prefix，然后比较可学习阈值；固定长度 task queue 控制 conditional path；改变 PE 映射、空间九 bank 和 crossbar 保证稀疏窗口取数。**近似门、任务压紧和条件路径搬运必须一起比较。** |
| [Eyeriss v2，JETCAS 2019](https://arxiv.org/pdf/1807.07928v2)，pp.9–12，§IV–V | CSC 分段维护滑窗；地址先于数据、激活先于权重；七级流水/五 SPad；SIMD 真正一次取两个系数，psum 增为双读双写，奇数尾填零。**独立稀疏 PE、系数字打包和增加端口的代价都有完整先例。** |
| [VENOM，SC 2023](https://arxiv.org/pdf/2310.02065v1)，pp.3–5，§3、4.1，Figs.3–8 | V:N:M 先向量删列再映射 2:4；column-loc 先读以选择 B 的行；实际 128-bit 事务、RF fragment 排布、输出共享内存 padding。**输出组共支持、避免加载无用输入、重排与 bank 规避已有。** CUDA GPU kernel 不是现成 ASIC RTL。 |
| [SpiDR，2024 作者稿](https://arxiv.org/pdf/2411.02854)，pp.3–5，§II.A–E | 原始脉冲输入、硬件 im2col loader、双口 IFspad；trailing-zero detector 形成地址，16 深度 even/odd FIFO 批处理降低外围切换。**原生输入到实际地址队列的零跳过已经可在真实芯片中实现。** 本文不借它的 CIM 宏面积到数字加法核。 |
| [FlexHiNM-GP，ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/file/9e33fdc35b68781132e836964a326bf3-Paper-Conference.pdf)，pp.4–7，§3–5 | 0:4/2:4/4:4 区域分配、二阶重要性、输出置换与 tile 内输入置换、动态 mask；稀疏/稠密 CUDA stream 和 atomicAdd 合并。**混合格式、两轴置换、二阶分数不可再当 X。** 正文已读；其证明和代码未独立复核。 |

本轮至少三个超普通 2:4 原法、两个真实物理打包例子、多个零跳过和一个近似跳过原法。没有为凑 venue 引用只有摘要的论文。Bit-Tactical 正式作者 PDF 地址返回 404，未用可读的摘要代替全文计入九篇；搜索中看到的二手说明只作为待补最近邻，不承担新颖性结论。

## 候选一：普通 N:M 支持选择 × 有界 lane epoch

### 结构、格式、调度

保留普通 2:4 六选二支持和重拟合能力；首版不引入 pair/fill。一个 epoch 处理一个 H8 输出组、一个 K4 输入组、两个输出位置及完整 T10，共 20 个空间时间位置。原生卷积 loader 生成 `G[k=0..3, u=0..19]` 的 80-bit 源位图。八个 lane 各有两个已解码支持 `s[l,0], s[l,1]`、两个系数锁存器和两个 20-bit 掩码 `M[l,j,u]=G[s[l,j],u]`。每 lane 独立扫描自己的 40 位活跃项，形成 `(coefficient_slot, psum_u)`，送一条加法通路；epoch 完成屏障要求八个 lane 全部提交后才能释放源/系数。

这是真正的 lane 内事件压紧：lane0 可以更新 t2，lane1 同拍更新 t7，彼此写不同 lane 本地 RF。不是把公共 walker 的长度改成 `max()`。跨 K4 不作投机重排；每个 psum 的 K 顺序确定；重复地址遇流水 RAW 时必须旁路或停顿。后继 PSN 只在某空间位置所有 K 和全部 T10 提交后收到完整向量。

首工单建议固定 `Wq16`、有符号 48-bit accumulator，θ 折入 W 后按输出通道静态尺度量化；原始整数和不溢出时先做到逐位一致，再定义一次 RNE/饱和输出转换。它不是 FP32 硬件。AEE 桥接必须让所有控制使用同一 Wq16 与舍入规则，单列量化质量影响。若最终采用 Wq32，物理布局应重新结账，不能沿用 Q16 字数。

普通 2:4 元数据每 lane 为 3 bit，每 H8-K4 为 24 bit；紧密打包，跨字实际读。Q16 下十六个系数恰好占一个 32 B 字；Q32 下可按两个 slot 各占一个 32 B 字。格式、位宽、slot 排布会改变可取消请求条件，是联合目标必须观察的物理事实。

### 究竟取消什么

Q16 的一个系数字只有在 `OR(l,j,u) M[l,j,u]=0` 时才能不发请求；个别 lane 空不能取消整字。Q32 若按 slot 分字，则 slot j 在 `OR(l,u) M[l,j,u]=0` 时可不读。源字取消另由候选二或普通 zero-header 路径决定；lane 压紧本身主要减少实际发射拍及无效 RF 访问，**不能把 lane 利用率提升写成源 SRAM 减半。**

只有在掩码/系数已就绪、每 lane 一拍处理一项、无 RAW/回压时，epoch 的理想发射长度才等于 `max_l Σ_j popcount(M[l,j])`。全层时间还包含 metadata、源形成、系数填入、屏障、回压、清零和完整写回。每 lane 独立优先编码、mask 寄存器、系数锁存、psum 地址与旁路要在相同预算下付费。

### 借入的完整 A 与候选 X

强 A 必须包括：普通 2:4/3:4 所有合法支持；幅值、Gram refit、一次请求代价优化；紧密支持编码及真实字节取数；dense-zero 执行相同有效 W；普通 T10 位图/静默神经元压缩；系数驻留；源/psum 驻留和相同 bank/缓存容量；原生窗口生成；有界队列、RAW、背压、边界填零、零包、清零及写回。**这些能力全部给普通方法，不能为突出 X 而收窄普通支持空间。**

最近碰撞是 Bishop 的打包后稀疏核分发/归约，以及 Eyeriss v2 独立稀疏 PE、地址依赖流水与真实 SIMD 端口；LoAS 的 temporal fiber 和求交也必须借入。故“八条独立队列”“lane 私有扫描”“多个时刻并行”均不承担新颖性。原论文完整异构核或 P-LIF 未迁入前，只能说与其机制相容或不同，不能称已胜 Bishop/LoAS。

剩余 X 限定为：**在相同质量预算、同一物理位宽与字布局、同一 bounded epoch 和端口预算下，联合选择普通 N:M 支持，使真实字请求与付费 lane 尾长共同降低。** 旧一次 request 优化只证明 CR 目标有局部作用；本轮新增的是以可执行 lane 地址/RAW/屏障为约束的目标，不能把旧参数重命名。

可观察的对照句：

> 给 A 和 X 相同 lane-private RTL、相同普通 2:4 支持搜索与 refit 预算后，只将支持目标从“重构误差或旧 CR”改为“实际 CR＋lane 提交尾长＋阻塞代价”。若完整 r0 的净周期/能耗没有新增改善，则 X 不成立；独立队列带来的收益全部归强 A。

必须做三组分离：固定 W 时公共/私有 walker；固定私有 RTL 时误差/CR/联合目标；联合 W 在 dense-zero/紧密索引两种执行。还要让普通 3:4、分层普通稀疏和已过 NB0 的低秩控制竞争质量—资源曲线。第一组只能证明硬件实现价值，第二组才隔离候选耦合。

### 首个完整 RTL 模块与停机条件

首模块 `r0_conv2_nm_native`：从原生 `T10,C96,H240,W320` bit 输入及真实常量存储器读入，完成窗口生成、K864 全归约、N96 全输出、所有空间位置和四边 padding、输出量化及写回；不能让 TB 直接提供 lane mask、预展开 im2col、最优发射表或 goldmask。检查点为完整 `[10,96,240,320]` 原始 Conv2 张量；外接原生 norm2/identity 的独立数值桥接得到第二检查点。只有把 norm2/residual 也纳入 RTL 后才能称完整 block RTL。

先验非常近，当前只适合作为“强 A 的完整实现＋可隔离 X”。若独立 walker 的 mask/地址/屏障成本已吃掉公共空槽，或同硬件上联合 mask 目标未胜旧 CR/Gram 控制，就停止该 X，不否定普通 N:M 底座。

## 候选二：原生脉冲字闭包 × 提前供数判定

### 结构、格式、调度

把 K 顺序显式转换为 `(ky,kx,C4)`，W 同步静态置换；源保留原生空间坐标。每个逻辑包为同一行的相邻两个像素、四个输入通道、T10，共 80 bit。首版用固定 16 B payload slot，并另存每通道跨两个像素及 T10 的 OR 摘要，共 4 bit。16 B header 字容纳 32 个 payload slot 的摘要。固定 slot 可避免动态 prefix/address 队列；代价是 80→128 bit 对齐膨胀，必须和紧密 80-bit 位流及普通 LoAS 压缩比较。

producer 在真实脉冲产生时累积摘要并写入 header；consumer 先查静态需求，再读 header，必要时才请求 payload。native line buffer 按原坐标向 3×3 halo 分发，有限 tag/valid 记录跨 H8 与相邻输出位置复用。首版不承诺同时供九个窗口字：单口或有限 bank 的读出冲突按拍收费。不能从离线已知活动 mask 直接生成“应有 header”再把 producer 费用删掉。

权重结构允许两级：上层以 `(H8,C4,全部9 taps)` 为粗向量组，组内再用普通 N:M 或 dense。粗组删掉可取消该 H8 对 C4 的窗口服务，但这是已有向量/分层剪枝的专门化。细支持与空间位置映射共同定义某原生字 w 的所有有效消费者集合 `E(w)`；边界、stride、halo 和输出组均在集合中。对每个字形成保守通道需求 `D(w)`，header 摘要为 `P(w)`。

### 物理跳过条件与闭包边界

若 `D(w)=0`，该范围消费者对该字完全无需求，可跳 header 与 payload。若 `D(w)≠0` 而 `(D(w)&P(w))=0`，可在 header 响应后取消 payload。否则必须读 payload，再按准确位置/时刻形成细掩码；header 的 OR 摘要是充分非必要判据，不会预测未读位。需要更强摘要可以增加 metadata，但不免费。

**省哪个层级必须分别报告。** 单 H8 没需求只可能省该 H8 的 line-buffer 出口读/重复分发；若另外 H8 仍需要，同一原生字首次源 SRAM/DMA 读不能删除。若所有十二组 H8 都无需求，才可以取消该层首次源读；在某些静态结构下这会退化成普通通道剪枝，不能称全新算法。若普通驻留缓存已经把源重放消掉，X 无权再领取十二倍收益。跨空间位置也同样：中心输出跳过而邻接 halo 仍需要时，源字不能释放或省略。

### 代价与完整强控制

必须支付 4-bit 摘要写入/读取、跨 header 字边界、tag 查询、摘要 latch、payload 空间、line-buffer 冲突、边界重读及 replay。按一个可实现的小条带算例，宽16输出加两侧halo时，三行原生脉冲环需要 `3×18×96×10 bits = 6,480 B`；条带间 halo 不能免费。若为让一份源服务 N96 而保留 `P2×T10×N96×48-bit` psum，另需 11,520 B；相反只存 H8 时只有960 B，却可能增加源重放。这是资源取舍，不是无条件广播收益。

完整 A 借入清单：普通 native line buffer/滑窗；Eyeriss v2 的分段压缩和地址先行；LoAS temporal fiber、零包抑制/缓存；HighLight HSS 坐标/VFMU 的真实整字服务思想；VENOM/HiNM 的粗向量、N:M、输入/输出置换与二阶评分；同一 producer header 和同缓存权限；dense-zero、普通全通道剪枝、普通 H8 向量剪枝、无联合目标的 HSS、相同字布局的普通 Gram/request N:M。现有代码只读参考，没有把原论文整套微架构迁入并宣称完成。

X 仅是：**让剪枝/支持选择的代价对象等于可实现的原生字需求闭包，并以 producer 摘要驱动 admission，区分首次源读、halo 留存和下层重复服务。** 大方向与既有 F1 重合；本轮新增内容是 80-bit native word→4-bit 摘要→静态 halo 需求→实际 payload 请求这条可写 RTL 的接口。若仅获得“换分组后更稀疏”，它仍是普通 HSS/BSA 映射，不足以申领新机制。

可观察的对照句：

> 给普通 HSS/V:N:M 和候选完全相同的原生字格式、producer、line buffer、置换与稀疏执行能力，只改变 mask 选择目标是否包含真实 halo/H8 消费者闭包；比较各层实际被取消的请求。若只减少逻辑 MAC 或被普通缓存完全吸收，则闭包 X 不成立。

### 首个完整 RTL 模块与风险

首模块 `r0_conv2_word_closure` 复用候选一整层卷积算子；增量包含真实 producer 摘要生成、native word/header SRAM 布局、需求 ROM/有限组合生成器、admission、line-buffer tag/valid、回压与重放。输出仍为完整 K864/N96/T10 的 Conv2 张量，逐位对同一有效 W 的 dense 参考；metadata 只能由真实源和部署常量产生。若先测单 stripe，也必须全 K/N/T 和完整 halo，且明确它不是全层周期结论。

主要风险是活跃源跨 T10/P2/C4 的并集太大，摘要绝大多数非零；过强共支持损害质量；header/对齐膨胀大于省掉的payload；普通驻留已解释全部收益。源稀疏率 3.6% 不能推算 T10 组静默率，必须用完整未参与支持校准的帧核查。允许有损并且 NB0 门较宽，值得保留结构探索；目前新颖性信心低于候选一，不应先做大规模 RTL。

## 下轮应交的证据

先交一个完整强 A 的 r0 Conv2 RTL，而不是为九篇论文各写一个裸调度器。工单细节见[RTL 工单](rtl_workorders.md)。完整执行的服务账要至少分出源 payload/header、weight/metadata、RF 读写、有效与空发射、RAW/队列/输出回压、源重放、清零和写回；只报总非零项或缓存命中率不足以判断。

精度证据至少分三层：整数 RTL 对独立整数模型逐位正确；有损量化/剪枝相对各自有效权重的模型语义正确；真实完整前向同任务质量优于 NB0。单层数值核、diverse10 探索门、valid825 和硬件全层延迟是不同证据，不能互相继承。候选只有在同样借足普通强控制后出现新 Pareto 点，才值得继续作为软硬协同创新。
