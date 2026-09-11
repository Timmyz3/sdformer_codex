# SCNN

- uid/来源: `MAIN-R031`｜audit_deep 核对复用
- 全文出处: /home/zhumd/work/sdformer_codex/ideafromai/research/13_isscc_vlsi_hotchips_edge_npus.md; /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/literature/patch_crosslayer_fusion_followup_20260908.md
- 精读深度: §III–IV/图4–6
- 题名: SCNN: An Accelerator for Compressed-sparse Convolutional Neural Networks
- venue: ISCA 2017

## 可继承 A
压缩稀疏卷积数据流经典底座：非零探测、笛卡尔积式稀疏执行、PE 间传递。

## 强对照 B
稠密卷积；忽略激活稀疏的加速器。

## 可差分 X线索
SCNN 是祖先底座≠X；本地 PSN/门控消费者接口不同。

## 与 F1–F7 / Stage B 关系
F1/F5 数据流对照；旁路主岛时作方法学参照。

## 不可搬用边界
CNN 假设；直接搬到 T10 PSN 非法。

## 可复用 idea 点
- 非零源字广播域与 r1 删字域对齐讨论
- 稀疏执行计费口径对照 Stage B same-port

## 杀门建议
计费口径无法对齐 same-port → 仅作文献对照不停主实验。

## 审计原文摘要（核对用）
- status: patch跨层融合强底座；未完整迁入
- reason: 普通零跳过/乘积笛卡尔展开不是新机制；非因果T10改变融合last-use，不能假设零中间存储而忽略Conv2 halo/输出累加。；条件变化：fixedBN/结构PSN已改变旧全域屏障；应重评昂贵patch融合，而非再开薄类别优化。
- what_untried: 完整输入/输出halo、bank冲突与压缩finish；两Conv+PSN+BN2/shortcut同域融合目前未闭。
