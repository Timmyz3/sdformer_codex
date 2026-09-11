# arXiv:2501.13610 · Efficient synaptic-delay implementation

- uid/来源: `MAIN-R205`｜arxiv_2501.13610+本地excerpt（`p0_excerpt_batches/batch_02.json`）
- 题名: Efficient Synaptic Delay Implementation in Digital Event-Driven AI Accelerators
- 精读深度: 方法级（仅方法摘录窗口）+依据：SCDQ 上 WVU（weight-value-useful）剪枝过滤实现零跳转发；避免零权仍占 FIFO/查表；Seneca Delay IP vs RISC-V 软件；与 ring/shared queue 谱系衔接

## 可继承 A
共享延迟队列出口的「权有用」零跳转发（WVU）——把稀疏权信息前移到延迟供数路径，减少无效排队（借入≠X）。

## 强对照 B
延迟出队后再查零权；无 WVU 的满流量 SCDQ；per-neuron ring buffer；纯软件延迟。

## 可差分 X线索
WVU 零跳≠ lifting 源字删字 X；可作供数路径「早停」工程对照，不作标题。

## 与 F1–F7 / Stage B 关系
F1/供数路径对照；与 MAIN-R204 成对。不抢 Stage B。

## 不可搬用边界
Seneca 多核事件加速器；WVU 矩阵假设≠本地源∩W 编码；仅方法摘录窗口。

## 可复用 idea 点
- 供数路径早过滤零消费者——对照并集读前掩码
- WVU 元数据开销计入费用杀门
- Delay IP 与 NCC/RISC-V 软硬对照测法可复用
- 负结果只停该过滤布局

## 杀门建议
WVU 存储/查表吃掉稀疏收益 → 停该零跳转发布局。
