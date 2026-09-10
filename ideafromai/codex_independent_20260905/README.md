# Codex 独立评审与研究假说

整理日期：2026-09-05。

状态：`RESEARCH_HYPOTHESIS_ONLY_NOT_VALIDATED`。

授权范围：用户要求将本轮及前序讨论的 idea 写入 `ideafromai`。本目录是研究记录，不是实验启动卡，不授权修改论文、RTL、模型或启动 EDA。

## 阅读顺序

1. [01 — 当前判断与首要假说](01_review_and_selective_shared_reduction.md)：C1/C2 的创新性问题，以及“动态公共部分和的选择性构造与广播”的推导、近邻、风险与淘汰条件。
2. [02 — 文献与备选路线档案](02_literature_and_alternative_ideas.md)：此前提出的模式表、无损压缩、融合累加器、有限窗口重组等。保留其来源与限制，不把历史排序当成已确定路线。

## 当前结论

- C1/C2 可作为实现、基线和验证基础；不能因为已有投入，就要求它们继续担任两条主要创新。
- 优先研究的具体假说：按 source 的目的 token 掩码，选择性生成共享部分和，随后送往独立累加器。它改变算术图，不仅改变读取顺序。
- 该假说的代数原理与 Mailman algorithm、公共子表达式消除有明确联系，不能宣称新数学原理；尚未通过完整查重、真实工作负载机会筛查或 PPA 验证。
- 此前“无损压缩第一”“短期先做融合累加器”等排序是阶段性意见，不是当前已批准计划。
- 用户已经取消 9.20 时间限制。原先基于该日期的优先级不再适用；取消截止日期不等于取消范围、证据和执行授权约束。

## 来源与隔离

- 工作仓库：`/home/zhumd/work/sdformer_codex`。
- 分支要求：`autoresearch/neuron-ops-20260507`；本次未切换分支。
- 评审读取过该仓库 HEAD 下的 `SDformer/hw_autoresearch_nts07/paper/iscas2027/main.tex` 与 `paper/tcasii/main.tex`。研究期间另一个 session 持续更新仓库，因此本目录不是某个实验结果的冻结审计包。
- 写入时只读查询得到 HEAD：`7e3d303014c77c6ea4b2da64381aac53194f2dcd`。该值仅用于环境定位，不表示本文逐项审核了此提交的全部结果。
- 本目录独立于已有 Grok 文件。未修改其 README、执行卡、合同、计划，也未采纳其中任何默认执行授权。
- 未修改论文或 RTL；未运行 VCS/DC/PTPX/ICC2/Formality；未提交或推送 Git。

本文的代数示例、容量选择和验证建议均不是已完成实验。任何后续实现必须先获得用户明确授权，使用新 experiment namespace，并独立审核结果后才可引用。
