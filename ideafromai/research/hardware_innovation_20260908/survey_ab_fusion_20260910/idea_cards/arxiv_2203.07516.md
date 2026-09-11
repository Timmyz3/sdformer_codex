# arXiv:2203.07516 · Skydiver

- uid/来源: `MAIN-R067`｜arxiv_2203.07516+本地excerpt（`p0_excerpt_batches/batch_01.json`）
- 题名: Skydiver: A Spiking Neural Network Accelerator Exploiting Spatio-Temporal Workload Balance
- 精读深度: 方法级（仅方法摘录窗口）+依据：APRC 离线改结构预测通道相对负载（无精度损失声明）、CBWS 通道均衡调度使平衡比>90%、SPE cluster/权重 bank/脉冲调度器架构、时空稀疏动机与 spikerate 分布

## 可继承 A
离线可预测的通道负载先验 + 调度均衡——可作多消费者/多 bank 并集不均的调度对照（借入≠X）。

## 强对照 B
在线动态猜测负载无先验；静态通道到 PE 固定映射导致 <90% 平衡。

## 可差分 X线索
CBWS 本身≠X；差分须证明对 lifting 源活动不均仍降 same-port 周期。

## 与 F1–F7 / Stage B 关系
F2/F5 调度对照；第二队列。不抢 Stage B。

## 不可搬用边界
FPGA XC7Z045 分类/分割吞吐≠本地 VCS；APRC 结构改动细节摘录不全（仅方法摘录窗口）；勿搬 22.6KFPS。

## 可复用 idea 点
- 离线相对负载预测作 Stage B 前静态分区上界
- 通道均衡调度对照 Gustav 柱并行负载不均
- 时空稀疏「不可预测」问题陈述对齐本地 spike/门活动
- 负结果只停某 CBWS 映射，不杀 SNN 加速家族

## 杀门建议
预测负载与真实 r1 活动相关低或调度后周期不降 → 停该均衡布局。
