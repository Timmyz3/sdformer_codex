# 候选覆盖与还需执行的接口

**全部适配接口尚未试完。** 本轮把统一目录的全部 **305 个候选视图逐行映射**，并另列 Grok 与本阶段补文献中的接口。305 包括 252 条逐论文融合提示、32 条旧筛选、13 条 brainstorm 记录和 8 条 Pro 候选段落；其中有 234 个不同的非空 catalog ID。别名、同论文多接口和同接口多论文互相重叠，不能把这些数称为独立 idea 数、已实现论文数或全文阅读数。

逐行交付在 [coverage_all_views.csv](coverage_all_views.csv)，每行保留原 A/B/X、原状态、来源、边界，新增具体接口家族、已有代表执行、证据、该视图自己的剩余内容和下一批次。覆盖映射在 [build_coverage.py](build_coverage.py) 中显式指定；脚本检查每个原行恰好分配一次。25 个家族只是导航分组，其中 F07 与 F23 来自本阶段补充接口，在原 305 行中为零。另见 [coverage_summary.json](coverage_summary.json) 和 [supplemental_interfaces.csv](supplemental_interfaces.csv)；补充表的 21 行同样不是 21 个独立新 idea。

CSV与家族摘要已按本轮实测更新；`view_specific_remaining` 和 `stale_source_status` 保留原材料的历史措辞，当前是否已试应看 `coverage_state`、`concrete_family_execution`、`family_evidence` 和 `remaining_executable_interface`，不能把原文的“未试”当本轮最新结论。训练/源执行/局部链/825使用动态结果目录；825未完成不填数。

“已实做局部”只表示有明确范围内的代表接口执行，**不表示该行所列原论文完整迁入**。例如原双指针与本轮共享索引不等于完整 Gustav、SCNN 或现有稀疏引擎；旧 ep35 Motion-XOR 切片也不等于当前 R24 学生全层。表内证据包含实际结果文件和已读的结果报告，未逐篇复现所有原作。仅有目录、文件名或摘要的条目不会据此升级为完成实验。

本轮新增执行是 [NRV∩W 同址合并与共享缓存](NEW_INTERFACE_RESULT.md)。采用真实 C384→全 T10 fixture、有限存储、实际源解码与 NRV 读写，给 dense 同样的同址合并和缓存。它关闭了“旧私人索引慢是否只是缺共享接口”的一部分缺口，结果不支持将当前索引布局升为新的性能候选。完整源生产、后继消费者与当前学生上的接口仍开放。

后续又完成 [固定F_live=2](flive2/README.md)：同56字S、总共双NR4、普通预取/合并/编译同权，352次有界数值执行；当前布局平均慢于最强本次F1。另已实做 [完整native→全域BN→PED join](../hardware/native_bn_join/README.md)，同Engine执行192000×96域并支付外部raw/PED缓冲读写；起点仍是外部实际gate/PED，**真实I24前段整层尚未闭合**。后者普通fused是已测强后缀，14.26%是公共融合收益；其GPU偏差不继承825，见 [独立审阅](native_chain_review.md)。这些都是具体范围的新增执行，不据此升级为原作完整实现。

固定[两块目录跨H32复用](../hardware/native_bn_join/directory_reuse/README.md)现已完成并补审：在真实增加W流量后仍比原fused少6.464856%服务，是更强的普通blocking分母，不能当新X或能效改善。表示、低位码/尺度和新三结构进度已进入下表，不再写作未试；[16+80一页设计](INTERLEAVE_NEXT_INTERFACE.md)仍只是设计。

| 家族 | 原视图数 | 已执行的具体范围 | 仍不同、适配而未闭的接口 | 批次 |
|---|---:|---|---|---|
| F01 NRV×W/共享索引 | 23 | 原双指针 RTL；本轮 paid merge/cache CPU | Gustav 8×8完整层、原生NRV/NR4到下游同机 | B1 |
| F02 FTP/在途上下文 | 15 | packed-class/time；本轮同56字S的单tile F2已测 | 不同多上下文布局、完整阵列与16+80交织 | B2 |
| F03 重复/包含/公共部分和 | 8 | M20/K16/H8 旧有限融合负结果 | 固定子图匹配、有限父值跨 K 生命周期 | B6 |
| F04 物理字删读/结构剪枝 | 25 | global/phase/row-phase、all_keep 控制 | 同预算窄稠密、2:4、物理字并集训练 | B3 |
| F05 时间结构/配对/编译 | 13 | 新三结构同父320步/十帧/源RTL/局部CPU；两项signed-PoT新dense/lifting的CPU门差与源RTL | 主825/GPU halo与两项量化新函数十帧待评；I24整层、不同配对/PIT | B3 |
| F06 证书/共同接受 | 18 | 原9/11 P2/分级余量、门尾；本轮独立复放 | 不同参考轴/可训裕度、连续消费者联合接受 | B5 |
| F07 PED 低秩/残差表示 | 0 | 旧R24父full825；普通P1留Z；两学生×五表示十帧，含affine/diag；有费CR/RF两种5+5解码 | 完整消费者内残差码/step/默认值与门就绪；不继承父825 | B4 |
| F08 低位 W/符号补偿 | 7 | sign+rank旧CPU；25条packed W4/W8 code+row-scale完整局部链 | 不同解码/低位datapath、MiLo/ReverB同预算恢复 | B4 |
| F09 比特面 | 5 | 旧 I24 位串行比普通 MAC 慢 | 同面积 bit-PE、低校正密度训练 | B7 |
| F10 动态 BN/晚参数 | 5 | 完整外部gate/PED→native→实算全域BN→join；两块目录复用更强普通分母 | 真实I24前段整层、新训练参数整链、替代norm恢复 | B1 |
| F11 生命周期/重算/交织 | 10 | 8088/驻RF/P1留Z；两项dense同函数13工作RF/229ROM控制过RTL；16+80仅设计 | 同RF/issue真交织，小RF普通同权；量化质量另评 | B2 |
| F12 真帧间差分 | 3 | 已有连续 4 帧、3 对机会探针 | 当前学生完整数值/动态 BN/历史端口 | B5 |
| F13 因果唤醒/细节教师 | 23 | 上一完成 flow 支持统计 | 事件/TDE wake、错过与恢复实际成本 | B5 |
| F14 Motion-XOR/精确注意力 skip | 13 | 旧学生 census/叶/脏行机会 | 当前学生行 memo、三 pop 项独立门控 | B7 |
| F15 注意力/残差替换 | 30 | 原叶分析，不能算替代网络已训练 | 一固定 block、等预算恢复 | B8 |
| F16 tokenizer/stem/decoder | 7 | 旧 decoder shards | 原 GT 下完整层与固定替换 | B8 |
| F17 平台/混合引擎/CIM | 34 | 少数可借数字叶 | 与器件无关的调度迁移；原宏未迁 | B9 |
| F18 工具/库/地图 | 11 | 本地 Verilator 等实际用途 | 具体 kernel 所需时再迁，非逐库凑数 | B9 |
| F19 backbone/数据/任务对照 | 29 | 本地 NB0 | 多数替代网络未训练，需同任务比较 | B8 |
| F20 时间/尺度训练先验 | 2 | 三结构同预算QAT；两项常量投影无训实试，非完整DeepShift/PIT/S-TLLR | 原PIT/尺度训练配方、量化新函数质量 | B3 |
| F21 无当前数字接口器件/应用 | 19 | 当前对象不适配 | 出现可迁数字算子后再开 | B9 |
| F22 字布局/广播 | 3 | 旧 TSBG B8 广播 | 同内容容量的 transpose 与转换成本 | B7 |
| F23 精确 codec | 0 | FP32/I24 往返与压缩率 | 融合后仍必须 spill 的有费 codec | B7 |
| F24 AT-LIF 身份 | 1 | 推理 {0,θ}，固定 θ 可折权 | 变神经元/检查点才是新函数 | B0 |
| F25 FireFly→SCNN 错绑 | 1 | 已定位错绑 | 题名/正文对应后再迁 | B0 |

家族计数是唯一原行分配的总和，不改变某视图同时涉及其他接口的事实；原 A/B/X 和 `view_specific_remaining` 用于保留这一交叉关系。外部正文缺口、旧学生身份等更细状态在逐行 CSV 单独覆盖家族状态。

本轮读了统一视图表/works 表、主执行队列、20260911 Pro/Grok 对照报告、Grok round3/round4 原报告及 handoff、两份 Pro 报告的全部 8 个入表候选段落、本阶段 FUSION_CANDIDATES 与补文献表，并检查与选择接口有关的实际代码/输出。本轮没有重新全文阅读 775 条 works，也不称为外部盲审。Pro 候选段落之外的原报告全文阅读历史未在这里重新认证。

有几项必须纠正旧标签：

- **真帧间不是零执行。** `motion/capture/frames.json` 有连续时间戳的 4 个 Zurich 帧，`motion/result.json` 已做 3 对相邻推理帧的支持/状态字节机会探针，使用上一帧已完成 flow。另一份 `open_fusion_execution/motion_residual` 是单次推理 T10 块内，不能代替帧间。尚缺的是当前学生完整数值、动态 BN、历史读写与恢复成本。
- **8088 等待归因已做。** 它来自给定压力波形，不能继续把此控制写成未执行，也不能当作 PED 最后读取证据。
- **质量门采用用户最新口径。** 同口径优于本地 NB0 originalSDformerFlow 即可；旧“再加 0.005”、只允许 G1、因公开占位就停整个家族的旧文案均非当前约束。
- **当前学生与小集分开。** 两个 R24+BN2 组合已有 full825 结果；Dg/W/source34 的小集探针不能继承 full825。当前阶段的 13 个 distinct configs、15 个 population rows、1780 行配对计量属于另外一张结果表，不是本表的覆盖数。
- **公开机制仍是强对照。** 固定 θ 折权、普通共享缓存/合并、BN onepass、普通 DFS/重算权限都应给分母。原作换一个名称或与 lifting 并列不会自动产生 X。
- **旧Grok身份否决不沿用。** 已另读 `grok_review_20260911/GROKBOT_NATIVE_VERDICT.md`；其“连续θg不要折W”、据此停grok46二值岛和AEE≤1.259均不是当前约束，F14等新布局保留。MAIN-R057在catalog本来就标“FireFly→正文SCNN”错绑，覆盖表是保留提醒，不是本轮新发现。详见 [补正](flive2/COVERAGE_ADDENDUM.md)。

明确缺资料的原 305 视图只有本轮读到的 7 条：CSA、COMPASS 两别名、ASNA-Flow 两别名、ERAFT FPGA、SPARTA。它们表示**本地尚无指定原作完整正文/工件**，不是证明互联网不可取得；通用数字接口仍可先试。补文献 L07/L08/L09 另有 3 个正文/工件缺口，在 supplemental 表列出。FireFly→SCNN 是身份冲突；其他很多工具/器件是当前对象不适配，这两类不能混叫缺外部资料。

具体后续批次在 [REMAINING_BATCHES.md](REMAINING_BATCHES.md)。本轮既有实际执行，也承认其余适配接口尚未全做；不以扩大目录计数代替实验。
