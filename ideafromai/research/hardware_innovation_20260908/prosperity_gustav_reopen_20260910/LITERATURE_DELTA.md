# 本轮融合用的先验增补：2026-09-10

目标是重新推进 Prosperity/Gustav，区分“旧布局失败”与“完整研究家族被否证”。未覆盖 Grok 的平行目录，也没有把文献数当新颖性分数。沿 brainstorming-research-ideas 的组合/边界/条件变化框架，以及 scientific-brainstorming 的反对与修正讨论推进；三位代理都看过项目背景和已有方向，**不是背对背独立发散**，评分不是录用概率。

| 原始来源／会议期刊 | 本轮实际阅读与归属 | 迁入决定及边界 |
|---|---|---|
| [Prosperity，HPCA 2025](https://arxiv.org/html/2503.03379v2) | 代理复读§III–VI，另核§VII-F及官方LoAS模拟路径；根复核旧完整层模型 | 恢复为优先底座。检测、pruning、stable排序、父继承/残差、双缓冲都属A。原作已有剪枝模型，不能说首次融合稀疏W；其公开匹配定义由输入集合驱动，输出W支撑参与关系/计划选择是本轮待测差分。 |
| [GustavSNN，HPCA 2026](https://doi.org/10.1109/HPCA68181.2026.11408587) | 本轮复核本地原文对应的§V/VI/VII-B实现说明及真实2×4切片/四帧完整模型；不是再宣称完整原作工程已闭 | 恢复为优先执行线。CPTB、NRV、源∩W、共享权重/局部状态是A。私人字节索引导致的慢不是原作固有代价；普通共享缓存和供数能力要补齐。 |
| [HiT，ISCA 2026](https://www.comp.nus.edu.sg/~tcarlson/pdfs/xiang2026hausafhmm.pdf) | 根与代理深读§III/IV-A/B，包含PIDU、ring/背压、DMAccum；发表身份核[正式日程](https://www.iscaconf.org/isca2026/program/) | 借完整双侧交集、分层供数和归约组织。输出θg稀疏不意味着其前的连续Y稀疏；HS×HS压缩收益和128×128规模不能照搬。 |
| [SegFold，ISCA 2026](https://arxiv.org/abs/2606.26701) | 根下载全文，读方法/微结构III–IV、方法学V及部分评价；不是全篇评价已精读 | SELECTA有限k窗口、SEGMENTBC压缩目的空间、IPM及空间/时间fold归A。可参考共享图碎片调度；其多数极稀SpGEMM与本地经常稠密Y差别大，未直接移植。 |
| [Avalanche，ISCA 2025](https://doi.org/10.1145/3695053.3730990) | 两代理复读§4–5及§6.7的ACM正文 | ROD、完成释放、复用缓存、RAW均纳入强控制。已迁到Gustavson/RoW风格，普通移植不算X。完成矩阵归约还不等于prefix/完整T10/连续支路均消费结束。 |
| [Uni-STC，HPCA 2026](https://www.ssslab.cn/assets/papers/2026-lian-UniSTC.pdf) | 代理复读§IV-A–G、§V、Alg.1–2和[官方工件](https://github.com/SuperScientificSoftwareLaboratory/Uni-STC) | BBC/TMS/DPG、控制数据分离、预归约和冲突归A。静态BBC构造、预知C结构不能免费用于PSN动态输出；普通在线编码器必须先给完整能力。 |
| [SeaCache，MICRO 2025](https://people.iiis.tsinghua.edu.cn/~gaomy/pubs/seacache.micro25.pdf) | 代理复读§4.1–4.4、§5和[工件](https://github.com/tsinghua-ideal/SeaCache-sim) | 合装/分段/替换可借。只读固定fiber和允许饱和误差的gLFU不能充当活PSN精确释放计数；普通精确引用计数/静态分区是强控制。 |
| [ExSpike，2026稿](https://arxiv.org/html/2606.20414v2) | 根本轮精读APEC§III-A2、体系接口及IV-A，沿用本地此前全文调查 | 公共交集＋残差是A。原文默认G2，不能用旧G4代表其最优；本轮另给更强任意候选公共节点及零残差别名控制。原文也明确共享可能增加取权等待，不能只引用事件减量。 |
| [ReShare，ISCAS 2024](https://www.cs.nthu.edu.tw/~ychung/Conference/2024-ISCAS.pdf) | 代理从作者PDF的搜索索引正文核§II–III/Alg.1/IV；直接下载403，阅读方式有此限制 | 纳入A与反对。拆位权重pattern、按结果块欠缺pattern数排序、计算/分发异步及block-write均已有。它尤其对准“共享省算术但输出阻塞”。借方法，不借ReRAM PPA。 |
| [CRPIM，Journal of Systems Architecture 2024](https://www.cs.nthu.edu.tw/~ychung/Journal/2024-JSA.pdf) | 同样由作者PDF索引正文核§3.1–3.3、Alg.1–2、§5.2；直接下载403 | 输入exact-pattern缓存＋权重bit-pattern重构、有限缓存配置纳入强控制。主体复用不改函数，不能说它已做本轮proper-subset支撑学习，也不能把两级复用本身算X。 |
| PRAP-PIM，High-Confidence Computing 2023；PattPIM，DAC 2020 | 本轮仅核ReShare原文引用与方法关联，未精读完整训练 | 必须排进下一次训练对照补全；已有“weight-pattern-reuse-aware pruning”概念，不能把普通重复模式训练当新X。未复现，不因是CIM而拒读。 |
| [Celty，2026-08预印本](https://arxiv.org/abs/2608.01536) | 本轮只定位摘要与作者信息，未精读全文，不赋予会议身份 | RLC-CSC与SIMT双稀疏解码可作后续格式参考；当前未实施，不把GPU数字迁作本地性能。 |

**从这轮文献中保留的候选层次：** 输出相关产品图＋支撑/计划联合选择优先；先算残差再注入父值为同线第二执行布局；跨层支撑/供数为Gustav条件轴；动态BBC构造、共享缓存、一般双消费者生存期先补为完整A。不是把每篇论文起成一张新Card。

**独审后的关键改正：** 公共节点可消去“相反父方向”的代数新意；H8独立argmax导致的多父数不是最低成本；部分lane继承必须给普通控制；普通窄门缓存推翻了“后继宽状态必须常驻”的理由。修正的是具体新增句，不把Prosperity/Gustav/剪枝整体移出研究。
