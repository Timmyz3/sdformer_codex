# arXiv:2205.00778 · Sparse Compressed SNN Detector Accel

- uid/来源: `ARX-053`｜arxiv_2205.00778+本地excerpt（`p0_excerpt_batches_gap/gap_02.json`）
- 题名: Sparse Compressed Spiking Neural Network Accelerator for Object Detection
- 精读深度: 方法级（仅方法摘录窗口）+依据：gated one-to-all product；零激活门控；零权重跳过；稀疏压缩权重+Weight Map/NZ；LIF 累加寄存器；摘录偏结果/对比表§，完整微架构§可能截断

## 可继承 A
稀疏压缩权重 + 零激活门控 + gated one-to-all 乘积——活动稀疏下减 DRAM/PE 动态的 SNN 检测加速对照（借入≠X）。

## 强对照 B
稠密 SNN 全权重扫描；无门控的恒定 PE 翻转；未压缩层权反复外存读取。

## 可差分 X线索
该检测加速器≠lifting X；借稀疏门控合同，勿搬 35.88TOPS/W。

## 与 F1–F7 / Stage B 关系
F2相关（时间/活动稀疏打包）。第二队列旁证。不抢 Stage B。f_candidates含F2。

## 不可搬用边界
28nm 布局能效/mAP≠AEE；摘录偏对比表禁止虚报已读完整数据通路；1024×576@29fps≠服务%。

## 可复用 idea 点
- 零激活门控作活动比例计费旁证
- 稀疏压缩权作驻留体积合同
- gated one-to-all 与多消费者广播对照
- 负结果只停该检测加速布局

## 杀门建议
同资源无服务增益或挤占 Stage B → 停该布局。
