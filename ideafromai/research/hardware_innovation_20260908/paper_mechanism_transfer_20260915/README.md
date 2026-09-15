# 2026-09-15：定向精读、迁移缺口与实际生产RTL

先读 [综合研究报告与下一步](MECHANISM_RESEARCH.md)。它串起 Prosperity、Phi、Bishop、GustavSNN、FireFly四篇、C-Transformer 与 COMPASS，以及关键非SNN来源。18项是假设台账，收敛四项；没有把18项说成完成的实验。

本轮实际新增：[生产接口预先说明](PRODUCER_EXPERIMENT.md) → [RTL结果](PRODUCER_RESULTS.md)。54案例×32配置，1728任务、552960门判决通过。seen-code与普通nonempty同效果，摘要不作为独立X。

| 文档 | 内容 |
|---|---|
| [Bishop/Phi/Prosperity](literature/bishop_phi_prosperity.md) | 三篇全文、关键原作、旧融合实验重新分清、固定原父的Phi接口 |
| [Gustav/GAMMA/MatRaptor/GROW](literature/gustav_and_spmm.md) | 宽状态与稀疏格式的连续问题链，TCAS-II剪枝/量化强控制 |
| [FireFly家族](literature/firefly_family.md) | 四篇全文与v1/v2源码、BitFusion/Trapezoid/LSQ、实际资源条件 |
| [COMPASS/C-Transformer](literature/compass_ctransformer.md) | COMPASS正文缺口；ISSCC全文；BiLD/NeRN机制与代码边界 |
| [独立复核](REVIEW.md) | 两个优先候选目前新颖性仍低，补强Phi-alone校准、压缩PWP与旧按需构表控制 |

实现与复现：运行 `/opt/anaconda3/bin/python3.12 run_producer.py`，复用上级已保存真实case与PSN PE；本目录生成自己的源RAM/提交/摘要集成RTL和6个边界案例。结果为 [TSV](producer_rtl/results.tsv)、[JSON](producer_rtl/summary.json)。生产器起点是实际packed-code字，未包含前级网络量化器，终点是完整T10门，未包含FC2/BN2/shortcut。没有EDA/PPA或生产RTL改动。

后续从“完整成熟A”继续：先固定原Prosperity父图，逐笔比较原生、已有正子集字典、Phi取PWP、Phi本地生成；同口汇合/结果完成属于待验证执行接口，不能先当创新。再做有限表的普通压缩/预取强控制；光流侧导出已有coarse之后的detail依赖和决策提前量。

材料说明：专题报告的作者原文链接是可重取入口；PDF/TXT、渲染图、作者工程镜像以及构建产物留在本机阅读缓存，不纳入本次Git提交。主报告、研究记录、自有RTL/脚本、必要来源身份表和结果表进入Git。第三方代码静读不等于编译/复现，缺稿没有被计为精读完成。

方法说明：本轮采用 `brainstorming-research-ideas` 的问题→失配→类比→组合→反对意见流程，并用 `deep-research` 组织primary来源报告；原始18项受已知工程合同约束，不是声称背对背、不受先验影响的独立发散。三个专题由并行子代理完成，主代理读报告、追Gustav引用、实现生产接口并整合；独立复核是同一团队的另一任务，不是外部同行评审。分数是内部判断，不是接收概率。
