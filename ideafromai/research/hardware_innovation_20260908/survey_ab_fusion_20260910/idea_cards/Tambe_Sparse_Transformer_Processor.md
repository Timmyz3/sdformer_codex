# Tambe Sparse Transformer Processor

- uid/来源: `MAIN-R324`｜audit_deep 核对复用
- 全文出处: /home/zhumd/work/sdformer_codex/ideafromai/research/13_isscc_vlsi_hotchips_edge_npus.md
- 精读深度: 本轮重新读作者全文（审计标注）
- 题名: Sparse Transformer Processor with Entropy-Based Early Exit, Mixed-Precision Predication and Fine-Grained Power Management
- venue: ISSCC 2023

## 可继承 A
稀疏 Transformer 处理器数据流与 token/注意力稀疏硬件接口。

## 强对照 B
稠密注意力；FlashAttention 类软件核（非芯片对照）。

## 可差分 X线索
解码器旁路可能省费用；搬到 r1 主岛易诱拐课题。

## 与 F1–F7 / Stage B 关系
F7 旁路；明确不进主岛、不抢 Stage B。

## 不可搬用边界
任务与光流 SNN PSN 主岛不同；勿用 LLM serving 指标替代 AEE/净服务。

## 可复用 idea 点
- 仅评估解码器侧并集费用
- 与本地 K=0 注意力有限份额对照

## 杀门建议
主岛 AEE/净服务无增益 → 停进主贡献。

## 审计原文摘要（核对用）
- status: 保留借鉴；未迁移
- reason: 值得纳入完整先验。光流是连续回归，分类熵不能直接替代可靠误差指示；现有粗头试验没有复现整套芯片控制闭环。
- what_untried: 光流可用的廉价判据、校准、错误代价、FP4/8与V/F联合控制；不能借用原文实测收益。
