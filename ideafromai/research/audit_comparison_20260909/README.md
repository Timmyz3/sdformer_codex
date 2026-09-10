# 两份文献盘点的交叉复核与电路会刊补充

2026-09-09。原目录 `literature_audit_20260909/` 与 `literature_audit_mushaolong_20260909/` 均保留，不相互覆盖。本目录为新增审阅与计划，没有运行训练、RTL 或 EDA。

建议阅读顺序：

1. [修订推进计划](REVISED_PLAN.md)：具体取舍、同负载边界、两条优先问题、完整借鉴范围、阶段与停止条件。
2. [两表逐项对照](grok_comparison.md)：Grok 127 项对旧358条的对应、遗漏与不同意见；机器可读版为 [grok_crosswalk.json](grok_crosswalk.json)。
3. [CICC 补充](CICC_SUPPLEMENT.md)：16条定向记录（15新题名＋1已有论文全文复核），逐项标一手来源、阅读深度、用途与未试部分。
4. [其他薄弱电路会刊补充](venue_supplement_other.md)：ESSCIRC/ESSERC、TCAS-II、A-SSCC、VLSI新增8篇；[结构化条目](venue_supplement_other.json)。
5. [Grok 五项候选的新颖性复核](novelty_review.md)：原文覆盖范围、概念评分及反对意见、两项一次性判别设计。

主要改变：patch 的少生产与共同完成合并；结构剪枝作为该挂点的使能和强对照；PSN先补完整常矩阵及状态基线；新增电路先验使能量问题获得独立、预先设门的研究资格。旧速度版失败不改指标续命。

文献条数不等于全文数，也不等于完整迁移数；新颖性评分不是录用概率。模型、功能仿真与目标工艺验证分开。生产 C1/C2 创新重构仍未完成。
