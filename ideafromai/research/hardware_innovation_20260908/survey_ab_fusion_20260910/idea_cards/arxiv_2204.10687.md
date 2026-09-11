# arXiv:2204.10687 · SNE

- uid/来源: `MAIN-R242`｜arxiv_2204.10687+本地excerpt（`p0_excerpt_batches_gap/gap_02.json`）
- 题名: SNE: an Energy-Proportional Digital Accelerator for Sparse Event-Based Convolutions
- 精读深度: 方法级（仅方法摘录窗口）+依据：LIF（线性衰减近似）；事件四元组 OP∈{RST,UPDATE,FIRE}；slice+C-XBAR+DMA；filter buffer≤256 组权重按地址选；显式事件循环+时空复用广播；摘录§III，完整阵列/PPA细部可能截断

## 可继承 A
显式事件驱动卷积：slice 并行 + 突触交叉开关 + 能量比例于活动事件——稀疏事件供数与状态驻留对照（借入≠X）。

## 强对照 B
稠密帧卷积扫描全图；无 OP 分相的统一 MAC；无 filter buffer 的全局权重重载。

## 可差分 X线索
SNE/Kraken-SNE≠lifting X；借事件比例计费与驻留，勿搬 DVS-Gesture mW。

## 与 F1–F7 / Stage B 关系
F7相关（事件寻址/权重选择打包）兼 F2/能量比例旁证。第二队列。不抢 Stage B。f_candidates含F7。

## 不可搬用边界
IBM-DVSGesture 精度/能效≠valid825/same-port%；线性衰减 LIF≠本地 θg 声明；仅摘录窗。

## 可复用 idea 点
- RST/UPDATE/FIRE 分相作事件服务合同
- filter buffer 按地址选权重→多消费者权重驻留旁证
- 时空复用广播作同端口对照
- 负结果只停 SNE 名替换数字链

## 杀门建议
无同端口活动比例改善或映射不上残差链 → 停类比，不杀事件比例家族。
