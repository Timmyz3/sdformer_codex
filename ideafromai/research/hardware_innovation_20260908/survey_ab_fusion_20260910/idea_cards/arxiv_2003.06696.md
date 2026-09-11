# arXiv:2003.06696 · Spike-FlowNet

- uid/来源: `MAIN-R229`｜arxiv_2003.06696+本地excerpt（`p0_excerpt_batches_gap/gap_01.json`）
- 题名: Spike-FlowNet: Event-based Optical Flow Estimation with Energy-Efficient Hybrid Neural Networks
- 精读深度: 方法级（仅方法摘录窗口）+依据：IF 膜积分；四通道事件帧；SNN 编码器+输出累加器→ANN residual/decoder 混合；消隐尖峰对策；AEE@dt=1/4；另有 unresolved MUSHA-OF011 勿双计为全文

## 可继承 A
浅层 SNN 编码器累加尖峰 + 深层 ANN 解码——混合事件光流架构与能量/稀疏对照（借入≠X）。

## 强对照 B
全 ANN EV-FlowNet；全深度 SNN（易 vanishing spike）；无累加器逐帧直通解码。

## 可差分 X线索
混合 SNN–ANN≠结构化 T10/lifting X；旁路不进主岛。

## 与 F1–F7 / Stage B 关系
F2弱相关；光流算法对照。不抢 Stage B。与 Adaptive-SpikeNet/Best of Both Worlds 成对。f_candidates含F2。

## 不可搬用边界
MVSEC AEE≠valid825；混合网≠nts07；仅摘录窗；勿与 MUSHA-OF011 unresolved 卡双计为两套标题。

## 可复用 idea 点
- 输出累加器作「时间打包后再交 ANN」检查点对照 F2
- vanishing spike→混合深度消融法
- dt=1 vs dt=4 合同表
- 负结果只停该混合深度布局

## 杀门建议
本地不胜结构化 PSN/ANN 基线或破 AEE → 停该混合布局，保留家族。
