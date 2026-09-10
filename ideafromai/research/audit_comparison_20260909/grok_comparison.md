# 两份文献盘点的逐项对照（2026-09-09）

已逐条读完 Grok 127 项，映射到现表 358 记录；结果在 [逐项 crosswalk](grok_crosswalk.json)。两份原目录均未修改。这是有上下文的交叉复核，不是独立盲法发散，也不以两份表的同意视作实验证据。

**总体判断：两份计划的主要取舍接近，但 Grok 表不是 127 篇已核原文。它补出真实遗漏，也把若干别名、论文族和本地实验混为同一种条目；本方358也有漏项、未核题名和阅读深度模板化问题。**

## 数量和来源边界

| 对照类别 | Grok条目数 | 含义 |
|---|---:|---|
| 同一论文或别名 | 98 | 已有记录，不等于已完整迁移 |
| 多论文聚合 | 7 | FA1/2/3、DeltaCNN/MotionDeltaCNN、BNFF/AVP等须拆分 |
| 本地身份、对照、实验或提案 | 14 | 不是新增外部论文；映射仅关联先验 |
| 新增且本轮作者入口核实 | 4 | QSD、SFA/V3、VD56G3论文、Sparse VideoGen |
| 新增背景线索 | 2 | Guo混合CIS/EVS、Finateu/Prophesee-Sony传感器 |
| 无法唯一确定身份 | 2 | Parallel Time Batching、UED SNN wake-up |

Grok JSON中，**0/127有独立source_url/DOI/arXiv字段，0/127在任意字段含HTTP原文URL；19/127有正文内arXiv/DOI线索**。127条都有本地evidence_paths；没有缺失路径，但LX006只指向一个目录，没有指定文件/段落（详见逐行JSON）。本地文件路径证明“何处提过”，不证明原文已读。其自报阅读层级为 html_arxiv=30, secondhand=64, name_only=12, full_text=15, author_pdf=6；html_arxiv和author_pdf只说明载体/自报，缺章节或访问凭据时不能替代全文审阅。

本方358也不能称358篇全文：其中28条没有source_url，56条题名为空或待核；还有内部别名与合并版本。若depth用“方法深度各异，Gustav有全文”等模板，不能把它扩成每一芯片都精读。两边应保留“引用入口已核／正文具体方法已读／本地迁移完成”三个独立字段。

## 最改变判断的纠错

| Grok条目 | 问题与处理 |
|---|---|
| SN005/006/008/010/013 | θg连续幅值不等于任意逐事件INT8载荷。合法下游里静态θ可编入effective W，不能据此断言SpinalFlow/SATO/FireFly都需新幅值MAC。真正需核的是时间依赖、折权条件和消费者。也不能把θ当1。 |
| LX006 | 把row34、continuous334与common3合并。2.03%逻辑取权差属于continuous334对优化row34；common3有不同依赖图、数值学生与许可实验，须三条分别对照。 |
| SP001及计划 | 2×4 RTL不是仅mount-only：已接W地址/返回、私人W0压紧与NR4、S→PSN→全部T10和背压，352任务是功能切片。仍无完整8×8/源DMA/FC2/BN2/shortcut，不能称完整Gustav或PPA。补基线不自动授权再做类别小优化。 |
| LX003/004 | 当前零响应X、停止U/CSE提法是概念/标题门停止，未做相应新训练，不应写attempted_variant_failed。 |
| SN018 | MFPSN不是完全未试：已有局部拟合/蒸馏失败；完整端到端原训练仍未完成。应保留这两个不同状态。 |
| SP003/004/CR003 | 只迁输出分组≠完整Gyro/HiNM；本地INT10 LUT载荷≠T-MAC已迁入。 |
| CR008 | CFMP被降为一般“硅片对照”，遗漏最直接两因子+tiled mask+TC→TR恢复强方法。应提高完整迁移优先级；现有非线性两Conv并非原CFMP。 |
| LX008 | rows3更快但AEE与integer bits3不同，不能称同精度支配；同函数time控制吃掉大部优势才是直接负证据。 |
| SP019 | Graham/SSCN子流形卷积与SBNet残差稀疏是不同工作，不能合为已实现的“普通零列底座”；本方SSCN单独书目也缺。 |

Avalanche（SP007）的DOI对应 **ISCA2025**，Grok写ASPLOS错。本轮DOI页面返回错误，采用此前取得的出版社Crossref元数据，不把访问失败当不存在。[出版社入口](https://doi.org/10.1145/3695053.3730990)

AT-LIF（SN001）两表都写 **NeurIPS2025**，本轮官方论文核对一致；因此**没有该年份冲突**，不为纠错而纠错。[官方论文](https://proceedings.neurips.cc/paper_files/paper/2025/file/fa12d67b5939c37ea8a4659c31a08d2c-Paper-Conference.pdf)

其他可直接用现有一手记录补齐的venue包括：Prosperity HPCA2025、VENOM SC2023、CRISP DATE2024、SparseInfer DATE2025、BitFair JETCAS2026接受版、LUT-NN MobiCom2023、LUT-DLA HPCA2025、T-MAC EuroSys2025、USEFUSE JSA2025、Flextron ICML2024。FA三篇应分别为NeurIPS2022/ICLR2024/NeurIPS2024，不能统一NeurIPS2022。这些身份修正不自动升级阅读或迁移状态。

## 真正补入的线索与目前用途

1. **QSD-Transformer（SN002）**：作者明确ICLR2025，值得补作允许量化/蒸馏后最强简单训练对照；不等同已有V2。尚未完整读/迁其方法。[作者入口](https://arxiv.org/abs/2501.13492)
2. **SFA/V3（SN003）**：正式题名是Scaling Spike-driven Transformer with Efficient Spike Firing Approximation Training，作者页给TPAMI DOI；本轮未核刊期。是整数训练/脉冲推理的重要遗漏，不能用静态θ差异直接排除。[作者入口](https://arxiv.org/abs/2411.16061)
3. **VD56G3对应论文（OF005）**：A Fast and Accurate Optical Flow Camera for Resource-Constrained Edge Applications，作者注明IWASI2023；保留系统背景，不抢当前数字执行主线。[作者入口](https://arxiv.org/abs/2305.13087)
4. **Sparse VideoGen（SP021）**：真实独立论文，不是FlightVGM/PARO别名。作者页可核时空attention稀疏与定制kernel；旧08称ICML2025，官方会议身份本轮未再核。需要先看Motion-XOR完整费用，不能因视频稀疏就恢复小attention主加速。[作者入口](https://arxiv.org/abs/2502.01776)

另两传感器只是背景遗漏，保留Guo JSSC DOI与Finateu ISSCC作者线索。Parallel Time Batching / UED两项身份不够，先消歧，不能计成已核新论文。Grok的“新增点”大多是既有方向的重新排队，尚未证明新的硬件机制。

## 五项最影响下一步计划的建议

1. **先修正模型身份与比较单位。** 拆row34/continuous334/common3与各S0/S2/patch学生；θg有效权合同明确后，撤掉“幅值必需新MAC”这条错误创新动机。它会影响原底座选择与费用，而非只是文字。
2. **贵patch优先完整CFMP＋SBNet/DynConv强控制。** 原两因子训练、真实mask生产、halo/索引和T10消费者一起迁；不把零梯度mask或当前许可负结果当整法失败。QSD/SFA作为下一量化训练前的简洁补读控制，不另开大文献池。
3. **PSN先补完整da4ml/CMVM，不再引用旧73.81%作未经编译的新瓶颈。** 同一当前学生、共同缩位/供数/跨p流水；编译后若无足够物理增量，停止X而保留优化底座。
4. **Gustav欠项按接口工单补齐。** 当前2×4功能价值保留；同址索引/W共享、64PE有限供数与真实FC2/BN2/shortcut是未闭处。F_live>1仅作有上界支持的强控制，不能因“完整A还缺”自动恢复薄类别标题。
5. **统一证据粒度但不覆盖两原表。** 127条逐行保留，聚合/别名/本地提案拆类型；新四篇加到独立待读队列，传感器与身份不明项分开。所有净收益门是项目筛选标准，不是TCAS-II稳接收保证；保留简单方法的精度—费用前沿。

本次未训练、未运行RTL/EDA、未改两份原清单，也未更改main.tex或生产树。逐项不同意见、来源解析及对应R编号全部保留在JSON中。
