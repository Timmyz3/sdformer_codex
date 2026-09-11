# VENOM

- uid/来源: `MAIN-R027`｜audit_deep 核对复用
- 全文出处: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/literature/sparsity_transfer_followup_20260908.md; /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/algorithm/group_pruning_probe/README.md
- 精读深度: §3–4/§6 + 官方剪枝代码
- 题名: VENOM: A Vectorized N:M Format for Unleashing the Power of Sparse Tensor Cores
- venue: SC 2023

## 可继承 A
结构化/模式稀疏与官方剪枝代码级先验；可作 F1 开源强对照工具链一端。

## 强对照 B
非结构化幅值剪枝；无模式约束。

## 可差分 X线索
模式稀疏训练/内核≠自动 X；需同槽服务对比。

## 与 F1–F7 / Stage B 关系
F1 强对照；Stage B 后可挂。

## 不可搬用边界
官方代码读过≠本地完整迁入 GPU/ASIC。

## 可复用 idea 点
- 用 VENOM 模式作为 HiNM/2:4 之外的结构对照
- 官方剪枝流程对齐 r1 双消费者损失

## 杀门建议
同服务不胜窄稠密 → 停该模式布局。

## 审计原文摘要（核对用）
- status: 两级结构强基线；未完整GPU/ASIC迁入 / 原文方法及官方工件已定位；非完整复现
- reason: 共享掩码本身不是新意；原GPU极高稀疏度收益不能用于本地50% W，任意列仍可能读满源64bit字。；条件变化：训练授权使先前未试条件改变；但本地少步探针不能当原全流程已否定。 / 本地 H8×C16 共掩码仅是易实现控制；没有把 VENOM 的全部剪枝/补偿/格式一起迁入。
- what_untried: 完整column-loc/行内索引、pairwise OBS、渐进恢复和Spatha输入/输出bank处理。 / 完整向量格式、OBS 类保留权重补偿和渐进训练；与真实 P4 源字及 FC1/PSN 生命周期一起比较
