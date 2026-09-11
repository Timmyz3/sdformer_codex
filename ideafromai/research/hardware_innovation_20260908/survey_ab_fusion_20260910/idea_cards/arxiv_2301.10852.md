# arXiv:2301.10852 · Flexagon

- uid/来源: `ARX-065`｜arxiv_2301.10852+本地excerpt（`p0_excerpt_batches_gap/gap_02.json`）
- 题名: Flexagon: A Multi-Dataflow Sparse-Sparse Matrix Multiplication Accelerator for Efficient DNN Processing
- 精读深度: 方法级（仅方法摘录窗口）+依据：同硬件可切换 IP/OP/Gustavson 三数据流；stationary/streaming/merging 相；PSRAM 按行分块+valid/k 标签；MRN 加法/比较树合并纤维；摘录§3.2 walk-through，完整§3.4 存储细部可能截断

## 可继承 A
一芯多数据流 SpMSpM（IP/OP/Gust）+ PSRAM 多 k 并行部分和——稀疏–稀疏供数与归约合并对照（借入≠X）。

## 强对照 B
固定单一数据流加速器；无 merging 相的朴素写出；不知 nnz 时串行等纤维。

## 可差分 X线索
Flexagon≠lifting X；借多数据流切换与 PSRAM 分块，勿称 SpMSpM 芯片 X。

## 与 F1–F7 / Stage B 关系
F5相关（驻留/部分和生存期/多 k 并行）。第二队列旁证。不抢 Stage B。f_candidates含F5。

## 不可搬用边界
DNN SpMSpM 基准≠AEE；仅摘录窗 walk-through；禁止虚报完整面积/功耗表。

## 可复用 idea 点
- IP/OP/Gust 可切换作负载自适应合同
- PSRAM valid+k 块作多迭代驻留旁证
- MRN 合并树作多纤维归约模板
- 负结果只停 Flexagon 名替换数字链

## 杀门建议
无同端口重读/归约改善或映射不上 → 停类比，不杀多数据流家族。
