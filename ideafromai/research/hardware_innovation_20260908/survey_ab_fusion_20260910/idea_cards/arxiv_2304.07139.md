# arXiv:2304.07139 · Neuromorphic OF (spiking enc-dec)

- uid/来源: `MAIN-R263`｜arxiv_2304.07139+本地excerpt（`p0_excerpt_batches_gap/gap_02.json`）
- 题名: Neuromorphic Optical Flow and Real-time Implementation with Event Cameras
- 精读深度: 方法级（仅方法摘录窗口）+依据：尖峰卷积编码器–解码器；层间复发可选；voxel grid（Nin=6）对尖峰、count 对非尖峰 sSNU；池化下采样+双线性上采样；摘录§3.2 网络，完整实时/FPGA 实现可能截断

## 可继承 A
尖峰 U-Net 式编解码 + voxel 输入估稀疏光流——神经形态事件光流前端对照（借入≠X）。

## 强对照 B
ANN 稠密光流；无复发的纯前馈尖峰；单一 count 编码硬套尖峰网。

## 可差分 X线索
该尖峰光流≠lifting X；旁路；MVSEC AEE/WAEE≠本地合同。

## 与 F1–F7 / Stage B 关系
F2弱相关（事件时间编码/尖峰动态）。旁路基线。不抢 Stage B。f_candidates含F2。

## 不可搬用边界
MVSEC AEE≠valid825；仅摘录窗；禁止虚报已读完整实时实现节。

## 可复用 idea 点
- voxel vs count 编码作输入合同
- 层间复发作时间依赖旁证
- 与 TrueNorth/Spike-Flow 成簇
- 负结果只停该尖峰头替换

## 杀门建议
前端替换无增益或挤占 Stage B → 保持旁路。
