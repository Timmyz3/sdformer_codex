# APEX

- uid/来源: `MAIN-R110`｜audit_deep 核对复用
- 全文出处: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/literature/patch_conditional_fusion_followup_20260908.md; /home/zhumd/work/sdformer_codex/ideafromai/research/grok46_20260905/04_paper_survey.md; /home/zhumd/work/sdformer_codex/ideafromai/research/innovation_first_audit_20260906/report-source.md
- 精读深度: §II-B/III-B/E
- 题名: APEX: A Dual-Sparsity Accelerator for Precise and Efficient SNN Inference
- venue: arXiv预印本 2026 / arXiv 2026

## 可继承 A
近似/精度可调执行与早期退出式费用交换先验。

## 强对照 B
固定全精度全算。

## 可差分 X线索
近似执行≠证书跳过；需本地 finite 互斥与 AEE 门。

## 与 F1–F7 / Stage B 关系
F4/F2 旁路；第二队列。

## 不可搬用边界
§II-B/III-B/E；非完整系统迁入。

## 可复用 idea 点
- 精度-费用旋钮映射到门控生存期
- 与无证书全算对照

## 杀门建议
验证/近似费≥省下的计算 → 停该布局。

## 审计原文摘要（核对用）
- status: 伪和/负修正/完整时间输入后发放先验 / 仅调研/待迁移
- reason: 旧只写未读状态已过时；负修正spike或全时间后修正不是新电路。PASC-IF不等价本地原坐标非因果PSN。；条件变化：允许换训练模型后可考虑，不应继续以冻结神经元差异永久否定。 / 双稀疏机制不能因存在先验而淘汰，亦不能仅换名成为贡献。
- what_untried: 完整训练/编码/兴奋抑制修正/输出，真实θg及同模型精度；原40nm结果不迁本地。 / 原文完整精确推理数据流未在 ep34 上验证。
