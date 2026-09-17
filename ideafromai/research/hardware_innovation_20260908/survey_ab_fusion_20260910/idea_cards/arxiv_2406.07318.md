# arXiv:2406.07318 · EFGCN

- uid/来源: `MAIN-R239`｜arxiv_2406.07318+本地excerpt（`p0_excerpt_batches_gap/gap_04.json`）
- 题名: Embedded Graph Convolutional Networks for Real-Time Event Data Processing on SoC FPGAs
- 精读深度: 方法级（仅方法摘录窗口）+依据：§3 邻域矩阵构图/PointNetConv/时向有向边/3D MaxPool；§4 流水线 PL+PS、无片外存；完整资源表细部可能截断

## 可继承 A
事件→固定尺寸邻域矩阵构图 + 时向有向边免 k-hop 重算 + PointNetConv(max) + 3D 图池化缩放——边侧事件图实时供数与片上驻留对照（借入≠X；联读 EvGNN）。

## 强对照 B
全图欧氏搜邻 O(n)；双向/逆时边致属性级联更新；2D 空间 MaxPool 致结构漂移；SplineConv 高内存；片外 DRAM 图存储。

## 可差分 X线索
EFGCN/SoC FPGA 图加速≠lifting 数字残差链 X；借事件图流水，勿搬 MEPS/LUT 当净服务%。

## 与 F1–F7 / Stage B 关系
F1/F5弱相关（事件图表示、片上驻留）。旁路/前端。不抢 Stage B。f_candidates空。

## 不可搬用边界
分类准确率/200MHz/无片外≠valid825/same-port%；仅摘录窗；禁止虚报完整功耗面积表。

## 可复用 idea 点
- 邻域矩阵+半径 R 限搜作固定足迹构图合同
- 时向有向边只更新新顶点作异步硬件旁证
- PointNetConv 线性ϕ+max 省略γ作轻量消息传递
- 3D MaxPool+位置/g 缩放保持时向边 | 与EvGNN成簇 | 负结果只停EFGCN前端替换

## 杀门建议
前端替换无增益或挤占 Stage B → 保持旁路。
