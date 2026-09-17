# arXiv:2606.03428 · PrimeSVT Memory-aware Prioritized Pruning

- uid/来源: `ARX-035`｜arxiv_2606.03428+本地excerpt（`p0_excerpt_batches_gap/gap_06.json`）
- 题名: PrimeSVT: An Automated Memory-aware Pruning Framework with Prioritized Compression Policy for Spiking Vision Transformers
- 精读深度: 方法级（仅方法摘录窗口+补§III PDF）+依据：目标层选择B分；优先从大层顺序非均匀剪；L2通道滤；约束驱动候选选择；自动化相对PSViT；细部鲁棒度量可能截断

## 可继承 A
内存感知优先压缩：候选层集B分选型 + 自大到小顺序非均匀剪率 + 约束验收选模——自动化SViT结构化剪与手工均匀剪对照（借入≠X）。

## 强对照 B
手工均匀剪率；非结构化；仅token稀疏；无内存约束的盲目剪。

## 可差分 X线索
PrimeSVT自动化剪≠lifting X；压缩旁路，差分不在26.68%内存%。

## 与 F1–F7 / Stage B 关系
F1相关（结构压缩策略）。压缩旁路。不抢 Stage B。f_candidates含F1。

## 不可搬用边界
ImageNet Top-1≠valid825；仅方法窗；优先策略≠RTL调度合同。

## 可复用 idea 点
- B=准确率×内存收益作目标层合同
- 自大到小优先序+非均匀率作压缩模板
- L2通道滤作结构剪单元
- 约束门控候选选模旁证
- 负结果只停该自动化剪挂接

## 杀门建议
自动化无优于PSViT手工或约束不可满足挤占 Stage B → 保持旁路。
