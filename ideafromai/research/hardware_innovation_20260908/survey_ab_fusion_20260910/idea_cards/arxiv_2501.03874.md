# arXiv:2501.03874 · Neuromorphic Optical Tracking (scattering)

- uid/来源: `ARX-179`｜arxiv_2501.03874+本地excerpt（`p0_excerpt_batches_gap/gap_04.json`；原gap摘录取到参考文献，PDF METHODS/SNN 窗补齐）
- 题名: Neuromorphic Optical Tracking and Imaging of Randomly Moving Targets through Strongly Scattering Media
- 精读深度: 方法级（PDF方法窗补齐）+依据：透射/反射散射光路+DVS；OTM编码下采样+FC 出 (x,y)；重建模块残差 SNN U-Net；LIF+ATan 代理；能量 AC/MAC 对照；完整光学参数表可能截断

## 可继承 A
散射介质下事件驱动跟踪+重建双模块 SNN（OTM 坐标 + 残差尖峰 U-Net）——极端光学前端与稀疏尖峰推理对照（借入≠X）。

## 强对照 B
帧/SPAD 阵列散射成像；纯相关散斑跟踪无重建；同结构 ANN 全 MAC；无代理梯度的转换 SNN。

## 可差分 X线索
散射光路/光子实验≠lifting 数字链 X；应用/传感旁路，勿搬跟踪误差或能耗比。

## 与 F1–F7 / Stage B 关系
传感/应用旁路；与执行岛弱相关。不抢 Stage B。f_candidates空。

## 不可搬用边界
MNIST/Kanji 投影≠自然光流；实验室光路≠芯片；仅 PDF 补齐窗；禁止虚报完整光学标定。

## 可复用 idea 点
- 跟踪/重建双分支共享事件输入作任务解耦合同
- 残差 SNN 块+上下采样作尖峰 U-Net 旁证
- AC vs MAC 分层记账作稀疏能耗分母模板
- 负结果只停该光学前端替换数字出口

## 杀门建议
前端不可映射到数字同端口或挤占 Stage B → 保持旁路。
