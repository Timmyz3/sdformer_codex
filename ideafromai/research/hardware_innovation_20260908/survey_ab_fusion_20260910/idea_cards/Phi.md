# Phi

- uid/来源: `MAIN-R005`｜audit_deep 核对复用
- 全文出处: /home/zhumd/work/sdformer_codex/ideafromai/codex_independent_20260905/02_literature_and_alternative_ideas.md; /home/zhumd/work/sdformer_codex/ideafromai/research/prosperity_fusion_20260906/README.md; /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/README.md; /home/zhumd/work/sdformer_codex/ideafromai/codex_independent_20260905/01_review_and_selective_shared_reduction.md
- 精读深度: 全文层次pattern/PWP/处理器
- 题名: Phi: Leveraging Pattern-based Hierarchical Sparsity for High-Efficiency Spiking Neural Networks
- venue: ISCA 2025

## 可继承 A
层次 pattern / PWP / 处理器式稀疏执行组织可作结构剪枝与执行解耦对照。

## 强对照 B
平坦稠密；无层次 pattern 的剪枝。

## 可差分 X线索
层次 pattern 本身≠本地标题；只有落到 r1 物理源字删字且过 AEE 才有差分。

## 与 F1–F7 / Stage B 关系
F1 对照轴；Stage B 后可挂。

## 不可搬用边界
未完整迁移原处理器；勿搬 PPA。

## 可复用 idea 点
- 层次 pattern 作为 F1 删字候选的结构约束
- PWP 式打包与并集读端口对照

## 杀门建议
不胜 HiNM/窄稠密或破 AEE → 停该布局。

## 审计原文摘要（核对用）
- status: 部分CPU迁移；onehot类共享成为强基线 / 强模式稀疏训练底座
- reason: 本地按需Phi基结果+纯残差森林在选定q8/cap16上比原Prosperity多加法/系数读；这只是变体。后续onehot7类聚合仍被采用为最强普通控制。；条件变化：原自然高签名数失败条件被训练固定K8改变；静态模式方法仍应保留。 / PAFT 是本文内部训练方法；本地 K8/类别实验不等于完整 Phi 复现。
- what_untried: 原八输入并行归约、PWP存储/预取、冲突处理及完整Phi→TA式融合；不能只与每次重算基的弱Phi比。 / 原 PAFT 在 ep34 上的完整训练、层间布局与执行仍未做。
