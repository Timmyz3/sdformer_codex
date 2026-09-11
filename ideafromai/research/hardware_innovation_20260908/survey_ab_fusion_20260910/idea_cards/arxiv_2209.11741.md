# arXiv:2209.11741 · Adaptive-SpikeNet

- uid/来源: `MAIN-R208`｜arxiv_2209.11741+本地excerpt（`p0_excerpt_batches/batch_01.json`）
- 题名: Adaptive-SpikeNet: Event-based Optical Flow Estimation using Spiking Neural Networks with Learnable Neuronal Dynamics
- 精读深度: 方法级（仅方法摘录窗口）+依据：可学习 LIF 阈值/泄漏；体素 former/latter 4 通道事件表示；U-Net 与 Fire-FlowNet 全脉冲；自监督 photometric+smooth；宣称克服深层 spike vanishing

## 可继承 A
可学习神经动态（vth、λ）与事件体素时序表示——作光流 SNN 训练/动态底座对照；隐式复发保时序（借入≠X）。

## 强对照 B
固定阈值/泄漏 LIF；纯 ANN EV-FlowNet；混合 SNN-ANN 编码器；显式 ConvRNN 加重参。

## 可差分 X线索
可学习 LIF≠结构化 T10/lifting X；本地已有 θg/非因果 T10 适配纪律，不得把原作 EPE 增益当本地净服务。

## 与 F1–F7 / Stage B 关系
光流任务相关对照（算法轴）；不直接挂 F1–F7 硬件主线。不抢 Stage B。

## 不可搬用边界
MVSEC/DSEC 自监督 EPE≠valid825 AEE 合同；全脉冲 U-Net≠生产 nts07；仅方法摘录窗口。

## 可复用 idea 点
- 可学习泄漏/阈值作普通神经元对照，对比结构化 PSN
- former/latter bin 表示作事件输入合同对照
- 缩模型时 SNN vs ANN 差距保持——支持「动态捕捉」主张的验证法
- 负结果只停某神经元超参布局，不杀事件光流线

## 杀门建议
同训练预算下不胜固定动态/混合基线或破本地 AEE 门 → 停该自适应布局。
