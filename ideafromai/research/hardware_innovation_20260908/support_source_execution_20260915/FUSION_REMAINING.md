# 两路径复核：实际完成层级与三个下一执行接口

2026-09-15。本文复核既有结果，不把条目数当作实验数。使用 [48 项总账](../FUSION_STATUS_20260915.md)、[18 项机制台账](../paper_mechanism_transfer_20260915/MECHANISM_RESEARCH.md)，再以其后落盘的 RTL/数值报告更新状态。仅新执行一个 CPU 退休粒度探针；本目录根代理正在做的 source 判码 RTL 由其最终报告验收，本文不提前代签完成。未运行 GPU、训练、RTL、EDA，也未改旧目录。

**尚未完成的是具体生产、表示和消费者接口，不是缺更多 mode。** 已有效的 R8 bitmap/RR、支持码查表和局部门证书应保留为强 A；已失败的 Phi 冷窄条带、私有尾、时间合码和动态 BN 下的局部细节，只能停止各自布局。下面只保留三个有区别的下一试验：当前支持码 PSN 的同资源门消费；patch 的完整 TC/TR 双因子消费；更早完整粗流与固定 decoder2 BN 后的细节选择。它们均未被本次报告宣告有效或有新颖性。

## 1. 先修正覆盖状态

| 18 项台账对应项 | 已经做过，不能再写成“只在纸面” | 仍缺的真实执行接口与两路径判断 |
|---|---|---|
| 1/3/4：固定森林、Phi、公共节点与结果生命周期 | [fixed_forest_phi](../fixed_forest_phi_20260915/RESULTS.md) 已做八臂、signed 修正、PWP 实取/实建、父值最后发布及按需查询适配；4128 任务零差。全 K864 的54条带叶周期原森林6340→Phi6644，仍慢4.795%。 | 原图 TCAM/外层流水、Phi 双路 packer/lookahead/PAFT、跨 K 硬件归约与最终消费者未接；本次只是窄 N8 冷 K16 求值布局的负结果，**不是完整 Phi 迁移失败**。最多两个公共项的正子集控制已经有，不再当新接口。 |
| 2/5/10/14：有限表、复用、切换、按需系数 | [support_lut](../support_lut_execution_20260915/README.md) 已从实际投影后位图到完整 PSN 做五模式 RTL；32 real 总服务256980→184184，普通LUT10省28.33%。目录/内容去重有实际负消融。 | 初始外存装入、前级最近码投影、跨 H 复用/重叠、FC2/BN2/shortcut 未齐。NeRN 按需生成器没有因此被迁入；GROW 等完整算法也未因普通预取而完成。**有效 A 应接真实生产，随后才谈新 X**。本轮 source 判码属于正在补的其中一段。 |
| 6/7：bundle 训练、二值/连续两侧剪枝 | NR4、组剪、shared/private、R8/空间/3M/box2 等都有算法或 RTL 证据；[共同执行](../shared_execution_20260915/COMPARISON.md) 确认空间候选连续 Q2 发射约3.1倍，不能靠少一乘抵销。 | 未完整迁入 Bishop 的可核 active-bundle 训练实现；也未闭合物理事务目标下的 patch TC/TR 两侧执行。一个固定投影/损失失败不等于训练表示无效。见试验②。 |
| 8/9/12：写时转置、跨 context W、位宽重组 | R8 原生4P、borrowRR、count21、小 bitmap7 和真实 I24 回收均已在共同双context重跑。bitmap ready较native省4.69–5.13%，是已有效的融合底座。 | 新 FFN 的 producer 完成后转置尚未随 R8 自动完成；旧 Q2 cache/共同广播已存在。跨 context W 的0.409%乐观局部机会不能升级为独立标题，更不能要求再做一套普通 RR。 |
| 11：连续权重求值图 | Q2 DA/lazy 构表和 CMVM/位面等实际做过部分。 | 完整 Transitive Array 的图和物理执行未做；DA 的负结果不能替它宣判。当前 PSN 96-MAC 下也未完成完整 BitL/CSE 同预算对照。 |
| 13：粗流→细节依赖 | 已有 P2 coarse 全网算法；[新闭包](../support_lut_execution_20260915/FLOW_NEXT_INTERFACE.md) 已确认 P1→D2→P2、动态全域 BN、真实 ConvTranspose 与插值 halo。 | P1 更早直接输出、仅固定 D2 BN 的全细节、真实决策 mask 与供数准入均未实跑。**动态 BN 是可改前提，不能用它杀细节家族**。见试验③。 |
| 15：OSS 概率时间近似 | C-Transformer ISSCC 全文/部分机制已读。 | 本图重新采样时间位的学生/恢复训练及 AEE 未做；非因果 T10 不容把发放率相同当逐位相同。属于未迁移，而非已测性能失败。 |
| 16：COMPASS 推测与恢复 | 身份/AE 的配置、hook、部分包装代码已核。 | 最终正文与核心恢复机制仍缺，不能用 BiLD 或 OSS 补写成作者算法；没有完整迁移实验可判输赢。 |
| 17：门早定、完整值继续 | [cert_transport](../shared_execution_20260915/cert_transport/RESULTS.md) 已有真实 Y/τ 供数和门退休 RTL，post-Y、known-τ 范围为正。旧 rank0/low-rank/行见证、CSD/CMVM 编译也已做过。 | 原始 Y 生产、阈值代次、完整值消费者和实际 H96 资源还没共同闭合。不能拿 T5 十路加和树复制96份后称同资源；本轮新探针量此差别。见试验①。 |
| 18：真实生产摘要→准入 | [producer 结果](../paper_mechanism_transfer_20260915/PRODUCER_RESULTS.md) 已做1728任务、552960门；seen-code与普通nonempty等效。 | 这个摘要 X 没有增量，应归入 A；完整前级量化器、作者整芯片与后继整网仍未随其完成。不能继续把“再补 NRV 交集/seen”当新优先任务。 |

因此早期总账的“支持码仅CPU、固定父Phi尚无RTL、Gustav真实生产未接”均已过期；反过来，“已有局部RTL”也没有把作者全方法或真实整层流水自动补齐。[representation 覆盖](../representation_transfer_20260914/transfer_coverage.md) 的空间共同执行与 Gustav 两个旧 NEXT 现已推进，应以较新的 shared/paper/support 报告为准。Tucker整数后端、自由U同函数直接两相位OS、完整LoAS、Maestro有序秩训练、Uz+Uδφ等继续留原队列；本文没有将未选中的接口宣判失败。

## 2. 本轮唯一新探针：证书何时能让实际消费者组停止

**预先问题：** 单通道的早定比例可能不能兑现为当前 H12 bank/H96 向量的停止。探针不重做按码时间收缩，也不重跑旧四份 T5 冷运输；它将既有精确符号前缀/尾部界应用于**当前 support 的同一32 real、同 A/τ/Y**，量不同已有消费者粒度的最后退休时间与字面加和树工作。三种粒度是标量/现bank/现向量三种观察单位，不是训练或参数扫描。

Y 从真实 S、W 独立计算；全 T10、所有96通道及六个源组均保留。每组实际有效位宽 e 在完整 Y 到齐后计算，符号头后逐平面判断 `lo≥τ` 或 `hi<τ`；它保持整数函数，含负权重和负 Y。本批实际 gain 全正、无 constant；不能用它额外宣称负gain新覆盖。源码/结果为 [probe_psn_retirement_granularity.py](probe_psn_retirement_granularity.py)、[JSON](probe_psn_retirement_granularity.json)。

| 共同退休粒度 | 组数，32命令 | full 符号头+平面工作 | cert 符号头+平面工作 | 本粒度减少 | 字面加和树映射96 ALU的乐观算术下界 |
|---|---:|---:|---:|---:|---:|
| 一个 h，T10共同退休 | 98,304 | 913,385 | 320,319 | 64.93% | 90,610 |
| 一个 H12 bank，120门共同退休 | 8,192 | 83,970 | 53,878 | 35.84% | 105,038 |
| 一个 H96 向量，960门共同退休 | 1,024 | 11,027 | 9,539 | 13.49% | 120,533 |

三行工作单位不同，**不能按列总量互相宣称周期快慢**。当前原生 PSN 为83,400次96路 MAC、118,344真实RTL拍，占完整服务64.25%。表中下界只数直接条件加和树/前缀更新，给首次赋值、移位和任意 lane 打包最有利处理；尚未计区间加法、2.44M/3.17M/3.92M次比较、供数、清空或控制。H96 这版连该下界都超过原完整 PSN。它只否定“把旧树直接铺大并免费保留其局部倍率”，**不构成优化CSE、子集LUT或完整BitL的复杂度下界**。

数值：**983040个唯一U**与源参考相等，三种观察粒度共 **2949120次门核验零差**。个体门早定很多，组尾部却持续到更深平面；该观察解释了为什么仅报单h门核百分比不够。与前一次 [时间合码失败](../support_lut_execution_20260915/TEMPORAL_CODE_CONTRACTION.md) 原因不同：那里丢失六组先合并的共享，这里是实际并行消费者的长尾与加和资源。

实际供数还存在明确缺口：当前 yhold 只保存一行H96的288B；同时形成T10位面需2880B，若直接添寄存器是额外2592B。复用已存在90KiB Y容量也必须通过原读口，不能同时许诺十个时间行免费读。当前 A/τ 已驻留，原 cert_transport 每组冷读τ的账不能重复搬来；本方新接口应标明 `(h,epoch)` 合法复用。

## 3. 只保留三个下一试验

### ① 当前支持码 PSN：先补同资源完整 A，再融合本地产生的门消费状态

**A。** 当前 LUT10→Y24→完整16×24乘法 PSN、既有静态 CMVM/CSE 编译、BF full/cert 是不同强对照。近期 ANN 的 [BitL，MICRO 2025](https://doi.org/10.1145/3725843.3756044) 原作 §4 是静态权重 bit 子tile 的行/列混合路径、预分析 metadata 与动态 pivot，**不是两个5bit子集和表就算完整BitL**。A在本图应先落实其可见的路径/符号/metadata/供数，再与同96-MAC完整点积比，不仅与朴素 bit-serial 比。

**B。** 本轮新证据表明，当前源已连续化、A全密集；字面加和树放大到H96后，早定被尾部和ALU需求吞掉。旧门证书又从外部冷读完整Y/τ，没利用支持码FC1已经持有的状态。两种接口都不等同当前64% PSN。

**X，待证。** 在FC1的**最后写入**边界形成局部有效位宽/符号信息，按真实尚未完成的 `(p,t,h)` 门义务选择位级或原生完整字路径；阈值只在相同 h/epoch 内复用。先以H12银行为可检查单元，让门退休释放相应计算义务，源Y只有其所有实际消费者均完成才回收。普通完整MAC、full bit路径获得同缓存、holding、bank门控和任务重排权利。新意若只来自普通缓存或BitL原生pivot，归给A。

**最小可运行试验。** 第一片直接复用当前32 real的S/W/A/τ，继续在真实FC1后产生Y；只换PSN消费模块，保留96路48bit ALU、96乘法器、原Y每bank读写能力和输出ready/valid。先把完整MAC与完整新求值路径逐U/gate核等，再开门证书；提前结束臂只保证应用gate，U若作为真实输出就必须继续算，不能由TB补答案。记录有效位宽形成、Y/τ读、MAC/加法、位面、bank拒绝、最后门、输出stall及全部配置。冷/暖、原两BP与跨命令epoch实际收费。禁止免费增加96套T5树、LUT端口或跨bank任意置换。

**辨别。** 若完整新A已经慢于native，先保留原失败臂，只允许按实测的加和/读口瓶颈改一次；不能把64.93%单h工作降幅当收益。若X只少比较/加法但最后门周期不降，结论为流量/能量待测或无时延增量，不以局部早定率晋级。该试验直接针对64%大头，**并非按码再交换两次求和**。

### ② Patch：把有效两因子完整接到物理 TC/TR 消费，再处理“少lane而字仍活着”

**A。** [r1.conv1 两因子学生](../algorithm/patch_probe/factor_completion_20260909/README.md) 已有实际训练、整数参考和完整网络恢复；普通R32删除控制的FP/QDQ valid825为1.201095/1.203043。它们不能借给后来Aq14整数版本。可见CFMP的共同TC→稀疏Z→TR恢复链仍缺完整端口RTL；不能将private56的微小请求收益等同完整CFMP失败。近期 ANN [LQER，ICML 2024 §3](https://arxiv.org/html/2402.02446v3) 对量化误差做激活诱导的低秩重构；[TASD，MLSys 2025 §3–4](https://proceedings.mlsys.org/paper_files/paper/2025/file/e2ec2530db26b54d0b3b060c1e4a1bda-Paper-Conference.pdf) 把张量分为结构稀疏项并共同消费输入/psum。这些是分解/稀疏的强先验，不是本方额外发明。

**B。** 已有[完整一帧物理请求表](../algorithm/patch_probe/factor_completion_20260909/latent_stage_train16/physical_u_requests.md)明确：private56少6.15%标量U更新，对普通shared56的256bit U请求只少0.03999%；同字其它列/T/P消费者使字仍然活着。V×Z、A、Conv2/BN2/shortcut还未进入同一次真实服务，不能只量U；父网络已采用便宜preview-only，不能将删除的尾重新放到分母后申领收益。

**X，待证。** 取消单位从单J2/J4列改为**同一物理权重字的全部消费者义务**；TC/TR布局和潜变量顺序共同表达这一单位，并让尚未完成门的需求精确映射回整字。先做一次确定性的消费者签名列排列/打包，U列与V行成对置换，保留数值；若静态重排仍无法形成可取消字，再允许一次同预算恢复训练约束整字消费者，而不是继续优化标量请求loss。共同引用计数/排序本身属于普通A，X必须通过完整服务差分证明。

**最小可运行试验。** 固定同一r1.conv1挂点 `P4×T10、K864、H96`。先跑已保存普通compact/R32和一种固定TC/TR学生的完整两因子：实际原门字→U8 AAC→Zi16→连续V→Yi32→Aq14/Ui48门，接真实Conv2/BN2/shortcut；各自独立整数gold，不能将QDQ输出用作新整数答案。只有完整A跑通后再加上述一次布局适配。W/source/Z/Y总容量和端口共同允许，所有clear、目录、TC/TR转换、V供数与mask/段间存活收费。普通无mask compact与相同字粒度的普通组剪枝享有相同重排/量化/恢复预算。

**辨别。** 若只减少U而连续V/PSN/consumer抵销，就记录该链的真实转移成本；如果普通R32在质量/总服务上占优，保留它。单Conv1因子化尚未完整迁移，不能在未计V/消费者之前写失败；反过来，旧恢复训练的AEE通过也不能写硬件收益已通过。这里是已有算法正点的完整执行债，与试验①的精确PSN求值不同。

### ③ 光流：普通更早粗头先做，动态BN失配后只改一个明确前提

**A。** [现存图](../support_lut_execution_20260915/FLOW_NEXT_INTERFACE.md) 中P1 `[10,1,2,60,80]` 先可用；D2拼接`[P1,D1,E1]`成386通道，经PSN→3×3 stride2 ConvTranspose→96通道120×160→动态BN→P2。P1/P2都是完整flow，当前P2已是coarse退出口；最后decoder3已删，不复活它作昂贵分母。普通更早基线是直接 `sumT(P1)` 双线性到480×640，AEE尚未测。

**B。** 当前D2动态BN使任何非空细节请求依赖全域统计；原图不能直接省deconv主体。即使假设固定BN，合成最终10%随机像素经插值反推也需要99.83%的P2位置。该失败前提已准确定位，并非“所有稀疏细节无效”。

**X，待证。** 仅冻结/受限重训D2 BN后，由完整P1的对应关系风险选择连续细节块；预算按真实ConvTranspose phase/halo依赖并集计，保留共享源直至所有消费者退休。最终输出逐位置选择`U4(sumT(P2))`或`U8(sumT(P1))`，不把P2加到P1当残差。映射碰撞/越界与事件可观测性只作有损决策线索，不当误差证书或免费后向flow。

**最小可运行试验。** 第一步先在同固定validation子集比较P1直接输出、原动态BN全D2、仅固定D2 BN全D2；冻结BN统计只用训练数据。只有固定BN函数可用再接一个普通连续块mask和一个对应风险mask，不扫描阈值。记录P1决策可用时点与D2首次真实source/W请求；闭包用已有CPU探针，再在一个真实边界/内部窗口做同口source→D2→head→输出的实执行。全部放大到825以前先核本函数gold、全空/全细节、halo和BP。本文没有启动该GPU/RTL工作。

**强控与辨别。** [WaveletVFI，TIP 2023 Algorithm 1](https://arxiv.org/abs/2309.03508) 已有粗层高频信息→mask→稀疏细节重建；[BiLD，NeurIPS 2023](https://papers.nips.cc/paper_files/paper/2023/file/7b97adeafa1c51cf65263459ca9d0d7c-Paper-Conference.pdf) 给出推迟调用和实际验证/回滚。故普通mask、依赖扩张和延迟提交不是X。若P1直接早退胜过复杂选择，采用这个普通强控制；若同固定BN全细节已过不了质量门，则先修这一函数，不能用mask隐藏BN退化，更不能用免费完整P2验证来报节省。

## 4. 共同停止条件与证据范围

三项的顺序都遵循两路径：**完整成熟A有效→保留并接真实边界→在新失配处试X；完整A负迁移→定位具体算术/供数/消费者原因→只改一个条件再对照。** 不以“未完整迁移”直接判失败，不因某个X被普通A解释而删除成熟底座，也不要求先复刻作者整颗芯片才允许有界组件实验。

以上是人工选择的三个有区别的下一接口，不是从十八项自动评分选出的赢家；没有新性能或论文创新承诺。质量按同环境NB0，既有+0.005门不恢复；不同学生、不同BN/量化函数的AEE与周期分开。下一真正结果必须同时写源/参数身份、真实有限端口/状态、最终消费者、强对照、净服务以及未覆盖项。本文CPU结果只能支持试验①的资源/粒度判断。

文献边界：本轮重新读BitL官方全文检索页的方法/硬件段、TASD官方全文§3–4、LQER作者全文§3，并复查既有WaveletVFI/BiLD primary；未将这些阅读计为本方完整复现。没有补齐COMPASS缺稿或Bishop surrogate，也没有声称搜索穷尽ANN硬件。论文原倍率均未转作本方预测。

方法记录：本轮用本地 [hypothesis-generation SKILL.md](/home/zhumd/.agents/skills/hypothesis-generation/SKILL.md) 区分观察、候选、反对解释和判别结果，材料始终本地处理。按其要求附方法工具引用：Timothy Kassis、Vinayak Agarwal、Yuhuan He、Darshil Patel、Aubrey M. Brueckner，*Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*，2026，[arXiv:2609.00065](https://arxiv.org/abs/2609.00065)。该工具不验证研究结论或自动选择假说。

复现本轮唯一CPU探针：

```bash
/opt/anaconda3/bin/python3.12 /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/support_source_execution_20260915/probe_psn_retirement_granularity.py
```
