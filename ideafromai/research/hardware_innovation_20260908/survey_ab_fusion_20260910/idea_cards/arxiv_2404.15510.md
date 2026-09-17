# arXiv:2404.15510 · NeuraChip

- uid/来源: `ARX-063`｜arxiv_2404.15510+本地excerpt（`p0_excerpt_batches_gap/gap_03.json`）
- 题名: NeuraChip: Accelerating GNN Computations with a Hash-based Decoupled Spatial Accelerator
- 精读深度: 方法级（仅方法摘录窗口）+依据：§3.1改装Gustavson行驻留SpGEMM；哈希常数时间索引匹配；滚动驱逐抑部分和膨胀；NeuraCore与NeuraMem解耦+HACC；完整PPA可能截断

## 可继承 A
哈希解耦空间加速：Gustavson式聚合 + 哈希引擎常数时间累加部分积 + 滚动驱逐——GNN稀疏–稀疏供数与部分和生存期对照（借入≠X；联读Flexagon/GROW）。

## 强对照 B
朴素三嵌套SpGEMM；无哈希的排序合并匹配；无驱逐的部分和缓存膨胀；计算–存储紧耦合单核。

## 可差分 X线索
NeuraChip/GNN≠lifting X；借驻留与哈希归约，勿称图芯片X。

## 与 F1–F7 / Stage B 关系
F5相关（行驻留/部分和生存期/驱逐周转）。第二队列旁证。不抢 Stage B。f_candidates含F5。

## 不可搬用边界
GNN基准/加速比≠AEE；仅摘录窗walk-through；禁止虚报完整面积功耗表。

## 可复用 idea 点
- 哈希累加作索引匹配分母合同（对照排序合并）
- 滚动驱逐作F5谁该留下模板
- Core/Mem解耦+HACC路由作多消费者归约旁证
- 与Flexagon/GROW成簇 | 负结果只停NeuraChip名替换

## 杀门建议
无同端口重读/归约改善或映射不上 → 停类比，不杀Gustavson/哈希家族。
