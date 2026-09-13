独立审阅，2026-09-13。审阅者未参与 `decomposition_owned`、`sparse_owned` 的实现；只读其源码和结果，并独立重算 AEE 与当前 SVD8 算术项。此文区分方法新意、本网适配与性能证据，分数不是录用概率。

**判断：优先把已经过新精度门的无残差 SVD8 做成完整强 A；保留 H8 共同 pair-ID 和请求代价驱动支持选择这两个 X 接口。** 现有空间/低秩加残差、逐输出 pair、固定 nonzero-fill 布局尚未证明净性能优势。负结果约束这些具体布局，不能据此排除分解、整数基或非零填充值家族。也不能因相对当前父有退化而否决已优于 NB0 的候选。

实际读过：`prototype.py`、`cost_and_check.py`、`adapter.py`、`count_basis.py`、`h8_shared.py`、`paid_schedule.py`；`probe.py`、`response_execution.py`、`fill_chain.py`、`r0_transfer/transfer.py` 及对应 JSON；另追到 `ParentNetwork`、`LiteralForward`、`evaluate_axis`、数据输入与 NB0 源身份。重点目标确为当前 r0 Conv2 的 96×96×3×3：63.701 G 是名义稠密 MAC extent，占当前已采矩阵算术的 10.678%，不是时间占比。r1 U16 完整后继窗口不等于这个 r0 大层。

**先验覆盖与新颖性。** 以下按原始论文的方法段判断，没有把论文提到某关键词当作完整工件已经迁入。

| Primary | 已经覆盖什么 | 仍需由本轮证明什么 |
|---|---|---|
| [SumMerge，ICS 2021，§3.2–3.4、§4](https://cwfletcher.github.io/content/research/2021.ics.summerge.paper.pdf) | 同权重输入先求和；跨输出共享子归约；离线集合交集、MaxScore 图构建、拓扑遍历；按输出组限制图规模以控制数据传递。 | 两输入求和、先剔除例外再应用共同系数、H8 共享本身不能主张新代数。需要与同样允许等权/反号合并、图分组和输入归约共享的强编译控制比较。本轮未迁入完整 SumMerge 编译器。 |
| [Finch，OOPSLA 2025，§3.2–3.3、表4](https://commit.csail.mit.edu/papers/2025/Finch-OOPSLA-2025.pdf) | 可配置 fill value、仅存非 fill 区域、重复值/区段、结构化遍历与代数简化已有。 | `c+exception` 不是新格式概念。但 Finch 的通用表示能力也不自动证明它已经解决本次 H8 CR32 请求并集与有界 gate collector；这部分需要实际映射比较。 |
| [LegoNet，ICML 2019，§2–3](https://proceedings.mlr.press/v97/yang19c/yang19c.pdf) | 共享低维 filter、二值选择、split-transform-merge、计算并复用中间特征已有。 | 本轮固定两位和/差的有限整数状态有具体 θg 适配价值，但不能称共享基/消费者选择首次出现；未迁入 LegoNet 的连续 filter 与 STE 训练。 |
| [TASD，MLSys 2025，§3–4、图11](https://proceedings.mlsys.org/paper_files/paper/2025/file/e2ec2530db26b54d0b3b060c1e4a1bda-Paper-Conference.pdf) | 多个结构稀疏项之和、精度/执行预算选择，以及分解项间保留共同输入 B 和 psum C 已有。 | “分解项共享输入/psum”不是 X。本轮尚未借全 TASDER 与 TTC；真正差异应来自 θg 的有限整数状态、消费者布局及完整取数后的收益。 |

分数只评价当前可辨认的增量：0–3 为已有机制的直接实现/专门化，4–6 为有明确接口差异但尚需强先验隔离的增量，7–10 需更强机制与证据；不是用性能输赢决定新意。

| 候选 | 新颖性 /10 | 本网适配 | 性能证据 |
|---|---:|---|---|
| 平铺/激活 SVD、空间分解、Tucker，加或不加 N:M 残差 | 2 | 高：真实大层，多轴展开与 θg→连续状态边界均被执行；可成为强 A。 | 因子链数值与 AEE 已有；没有无残差小 rank 的完整端口时间线。 |
| 逐输出 selected pair，含共享 signs | 3 | 中高：中间值限制在 −2…2，可用加/减/倍增；PSN/PED 未被误作位流。 | 消费者选择费用高；signed 比 unsigned 少的加权项仅约 1.118%，不是整条相对 dense 的收益。 |
| H8 共用 pair-ID＋T10 小计数选择 | 4 | 高：真实减少描述符与动态消费者选择次数；同 H8 普通3:4控制已补。 | 有实际 H8 参数和整网 AEE；修正后的公共发射模型证实较逐输出 pair 便宜，仍未胜普通控制。 |
| nonzero-fill exception 的真实请求目标选择＋读响应 collector | 4 | 高：源操作数决定请求，普通 N:M 获同样 Gram refit、支持选择和误差额度；已移到 r0 大层。 | Gram→request 有隔离增量，但固定格式仍被普通2:4的质量与该局部服务计数同时支配。 |

**性能证据与最强漏洞。**

`r0_transfer` 的最终五臂均采用实际 FP32 bytes、紧密支持码和单 32 B metadata latch；跨界请求被计入，系数请求不提前窥视权重零。校准为首帧偶序32位置，局部测试为奇序32位置，各 T10，不能称独立帧。五臂的 bytes 解码点积与各自有效 W 均为零差。

| r0 格式 | value CR32 | metadata CR32 | SIMD8 MAC | C | diverse10 AEE |
|---|---:|---:|---:|---:|---:|
| ordinary2:4 | 43,718 | 4,014 | 107,637 | 231,001 | 1.1591833472 |
| ordinary3:4 | 57,108 | 2,688 | 132,238 | 279,730 | 1.1860722596 |
| fill1:4 magnitude | 54,499 | 2,688 | 152,278 | 294,552 | 1.1777446144 |
| fill1:4 Gram | 54,249 | 2,688 | 151,381 | 293,155 | 1.1949790382 |
| fill1:4 request | 52,882 | 2,688 | 145,137 | 284,177 | 1.1762471512 |

C 明确定义为 `2×(value+metadata CR32)+SIMD8 MAC+27,900 support decode`。Gram→request：C −3.06%，value 请求 −2.52%，MAC −4.12%，局部相对平方误差 .0287020→.0290850。这个增量成立；把 magnitude→request 的全部改善归因请求目标则混入了支持拟合目标的变化。request 相对 ordinary2:4 的 C 仍多23.02%。元数据少33.03%并不能直接变成周期收益。

该 r0 表还缺 source gather/im2col、包生成、仲裁、延迟与目标写出；20 个 T/P accumulator vectors 的 H8 外循环使每个源包读12次，共41,472个16 B包，包含零包。其 C 是服务计数，不能与另一目录的时间线相除。r1 的 `fill_chain` 则确实将完整 K864 的 U 接到 F/raw merge、gate、U24/V96、连续 PED 和写出，并复用原 Machine 的端口/RAW规则：dense 612,457、ordinary2:4 604,847、shift1:4 606,702、允许先抵消的 fill 606,688 服务槽。fill 相对 dense 仅少0.94%，仍比普通2:4多0.30%；压力序列也未反转。上游 PSN/preview、原生 projection conv/global BN 在该窗口之外。r1 普通 N:M 仍用旧 2n-bit 码，进一步压紧只会增强已经胜出的普通控制。[执行结果](../sparse_owned/fill_chain_results.json)

H8 最终表已重新核查：8个48-bit加法lane，逐lane96×48-bit RF、2R1W，256-bit系数/描述符口、48-bit源口、256-bit输出口；单个10-bit公共walker，所有lane共用同一slot/time。系数填入、选择、mask/地址遍历和加法串行；每wave60次SIMD8累加器清零另付，共7,680槽/臂。审阅者核对全部十臂的64位置加总、busy项加共同源填入/输出费用，以及每臂27,648次wave内源组读取，恒等式全通过。[最终源码/表](../decomposition_owned/paid_schedule.py) · [JSON](../decomposition_owned/paid_schedule.json)

| 最终公共walker模型 | 64位置完整N96/K864/T10服务槽 |
|---|---:|
| dense reconstructed／逐输出普通3:4的dense-zero执行 | 713,544 |
| unsigned pair，独立生成 | 1,938,998 |
| unsigned pair，共享六个计数、逐输出pair-ID | 1,797,852 |
| signed pair，共享六个计数、逐输出pair-ID | 1,799,978 |
| H8共同pair-ID | 890,128 |
| H8共同普通3:4，紧密索引执行 | 729,862 |
| 同一个H8普通3:4有效W，dense-zero执行 | 617,255 |

H8 pair比逐输出共享count的unsigned版少50.49%服务，但有效W/精度也变化，不能把全部差额视作无损调度收益；它比dense多24.75%、比H8普通3:4索引版多21.96%、比同值dense-zero控制多44.21%。保留dense-zero控制很关键：源很稀时，压缩索引解码会使普通3:4本身比直接存零更贵。此表没有同排程的普通2:4，不可将隔壁r0的C硬接成其同资源周期。

这是**有限服务模型**，不是完整物理层排程或FP32硬件实现：权重/metadata仅按tight位流服务extent收费，未从静态byte地址实际取回并验证burst/跨界；源packed-word形成、im2col gather和全层外循环未入表；物理RF交叉连接/时钟未验证。FP32数值核与32-bit系数/48-bit累加器宽度是假设不同的两层证据，θW量化、溢出、48→32的RNE并未实做，32-bit输出只能叫服务extent。这些限制不影响把已支付部分作为明确局部负结果，但不足以证明部署速度或数值闭合。

两项实际发现影响排程可信度：第一，2R1W 已被系数读、psum读写占满时，不能再免费并发 coefficient fill/selector；第二，`max(events)` 不等于公共 SIMD 发射长度，除非另外实现并支付 lane 私有活跃项压紧、位扫描与系数/psum地址生成。第二项由本次独立审阅指出，作者已确认并改为公共 slot×time union；全零组每wave读取及累加器清零也已补付。341,997/236,880及363,014/473,805等旧版数全部弃作paid结论，未用旧数评新接口。

**可晋级的三个具体增量。**

1. **先完整执行无残差 SVD8，再研究分解项与 θg producer/消费者的驻留接口。** 四个纯分解均过新 NB0 门：flat8=1.3479650409、activation8=1.4030194175、spatial16=1.2691297866、Tucker8=1.4170152223。旧局部 relative L2 约0.4–0.54不能据此淘汰。审阅者另用当前640×864真实输入独立计算：flat/activation8 的第一阶段222.1125 AAC/位置，后段最多768连续 MAC，系数30,720 B；中间值非零率76.875%，绝不能套源脉冲率。ordinary2:4同输入1391.5172 AAC，dense2665.35；只算算术时，MAC/AAC单位服务比低于1.5227才可能胜普通2:4、低于3.1813才可能胜dense。这是可检验的敏感性阈值，不是时钟周期。下一对照应固定同乘加能力与总buffer，完整 K864→R8→N96，支付中间量编码/读写、输出存活、padding、所有权重与源重放，保留后继 BN/neuron；将 ordinary2:4/3:4 的强供数一起迁入。Tucker8参数仅2,112个，可作容量压力点；spatial16的4,608连续 MAC/位置应保留，不因AEE更好便宣称更快。
2. **把 H8 pair 作为共同选择接口继续，而非包装成新分解定理。** H8 pair AEE=1.1966744037，H8普通3:4=1.2144504382，均过NB0；pair metadata从逐输出7,776 B降到972 B，普通同H8控制648 B。先以真正公共发射、相同source/weight缓存和完整当前层证明接口费用，再将支持选择目标从纯权重平方误差转为同误差额度下的实际selector/CR请求。普通H8 2:4/3:4也应获得该目标与所有合法支持；不能拿普通方法被人为限制的共支持版本作唯一最强对照。现有 per-output unsigned AEE更好并不抵消其选择器费用；反过来，某次H8速度仍输也不证明所有共享基失败。
3. **请求优化优先移交给强普通 N:M，再考查 fill 何时提供额外能力。** Gram消融已证明目标能改变实际请求；可将现有一次遍历优化器作为共有 A，冻结独立帧源统计和字布局，比较普通2:4、普通3:4、fill及一个每物理H8-K4组选择格式的有界接口。选择须包含模式标签、pack/decode与collector成本；必须在未用于支持选择的完整帧算子上结账，不能由同帧奇偶32点直接放大全层。只有 fill 在普通格式已经同获请求优化后仍新增可用的质量/服务取舍，才值得升为主要 X。

**AEE 独立审计。** 原“三批”是 `aee=12`、`count_aee=3`、`r0_sparse_aee=6`，合计21臂/210次前向，22不是这三批文件的实际计数。审阅时另有 `control_aee=7` 与 `temporal_fusion_aee=4`，最终共32臂/320次；所有 summary.complete 为 true，每臂恰好十个相同且顺序一致的帧名。审阅者从每帧 `aee_sum/valid_pixels` 重算帧等权均值，与每臂summary误差均小于1e-10；每帧有效像素数均匹配 NB0 CSV。十帧NB0=1.4546028610700001，全部32臂低于它；原825行另重算为1.4453525346809697，并非本轮新做825次验证。

候选之间确实共享 `evaluate_axis`：逐帧reset，当前实际 `preds.2` 时间求和，双线性恢复480×640（align_corners=False），无flow值缩放，同canonical GT与mask，每帧有效像素均值再帧等权；没有新优化器更新。`evaluate_fusions.py` 每臂重新安装/恢复模块，组合臂均实际前向，不是把单项AEE相加。lifting的四臂具有自己的paired parent=1.1653433024；最新 `aee_all` 已分别对dense/lifting对应父求delta，保留了该身份区别。

对NB0应写“同帧、同GT/有效像素和聚合口径的任务质量比较”，不能升级成同一个网络或逐位同执行协议：NB0是本地upstream PSN/SDSA复现的完整 `flow[-1]`，候选是已改造的粗头；旧NB0的FP32归约/CSV与当前FP64累加FP32 EPE也不同。像素数相同不单独证明历史mask位图逐位相同，原mask未归档。[已核原身份与边界](../../accuracy_baseline/README.md) 第一帧参与部分源统计校准，所有十帧又是项目已见样本；这不否定用户允许的探索性NB0门，但不构成未见分布泛化或显著性证据。新臂均未继承旧学生的valid825与硬件结论。

上述建议只要求下一次可执行比较的具体增量，不承诺芯片已复现、性能胜出或论文录用。Prosperity 的独立 psum 实验另已完成：官方本来就有跨K输出驻留，8-bit下无所臆造的DDR spill可消除；宽值参考中的普通驻留集是强A，T10整组准入未胜它。[psum完整结果与边界](../prosperity_owned/psum_probe/README.md)
