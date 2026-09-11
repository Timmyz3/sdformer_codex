# arXiv:2203.00158 · GROW

- uid/来源: `ARX-066`｜arxiv_2203.00158+本地excerpt（`p0_excerpt_batches_gap/gap_01.json`）
- 题名: GROW: A Row-Stationary Sparse-Dense GEMM Accelerator for Memory-Efficient Graph Convolutional Neural Networks
- 精读深度: 方法级（仅方法摘录窗口）+依据：行驻留稀疏–稠密 GEMM；HDN 缓存+图划分提时间局部性；runahead 多输出行隐藏缺失延迟；MSHR/LDN 表；摘录偏缓存/runahead§，完整阵列数据通路可能截断

## 可继承 A
行驻留 + 高复用行缓存 + runahead 重叠缺失——有限 RF/带宽下稀疏–稠密供数与延迟隐藏对照（借入≠X）。

## 强对照 B
无 HDN 缓存的朴素 SpMM；单输出行串行等缺失；无图划分的全局度缓存。

## 可差分 X线索
GCN/GROW≠lifting X；借驻留与 runahead，勿称图加速 X。

## 与 F1–F7 / Stage B 关系
F5相关（行驻留/生存期/缓存周转）。第二队列旁证。不抢 Stage B。f_candidates含F5。

## 不可搬用边界
GCN 图基准/40nm 面积≠AEE；图预处理摊销假设须明示；仅摘录窗。

## 可复用 idea 点
- HDN scratchpad 作「谁该驻留」对照 F5
- runahead 多行重叠作背压下继续服务模板
- 图划分改变邻接局部性→打包边界类比 F7
- 负结果只停 GROW 名替换数字链

## 杀门建议
无同端口重读改善或映射不上残差链 → 停类比，不杀有限 RF 家族。
