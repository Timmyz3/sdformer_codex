# TASD

- uid/来源: `MAIN-R029`｜audit_deep 核对复用
- 全文出处: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/literature/sparsity_transfer_followup_20260908.md
- 精读深度: §3/4.2–4.4/图11
- 题名: Enabling Unstructured Sparse Acceleration on Structured Sparse Accelerators
- venue: MLSys 2025

## 可继承 A
时间/结构感知稀疏调度先验，可对照 lifting 半步边界上的组完成。

## 强对照 B
静态窄层；无时间感知调度。

## 可差分 X线索
调度先验借入；挂点必须换到本地半步/RNE 才可能差分。

## 与 F1–F7 / Stage B 关系
F2；第二队列。

## 不可搬用边界
未完整迁入；图11级方法不等于本任务精度门。

## 可复用 idea 点
- 时间感知调度映射到半步检查点
- 与独立预测+同组关闭对照

## 杀门建议
预测费吞收益或 AEE 破门 → 停。

## 审计原文摘要（核对用）
- status: 规则项+例外项保留；未迁入 / 原文方法已读；未实施完整分解
- reason: 多项共享输入/部分和可借，动态激活提取及中间读不是免费；例外散布可能使删掉的C16又全读回。 / 用于防止把一种结构稀疏硬件的限制当全部稀疏执行的限制。
- what_untried: 完整项选择/提取器、例外真实物理字触达和多项累加，只跑一次PSN的同精度执行。 / 完整张量分解、额外项恢复和合并费用；在 θg/完整 T10 下的实际源字与输出状态比较
