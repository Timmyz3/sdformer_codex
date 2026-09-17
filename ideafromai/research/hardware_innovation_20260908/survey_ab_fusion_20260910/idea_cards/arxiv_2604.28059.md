# arXiv:2604.28059 · NeuroRing

- uid/来源: `ARX-174`｜arxiv_2604.28059+本地excerpt（`p0_excerpt_batches_gap/gap_05.json`）
- 题名: NeuroRing: Scaling Spiking Neural Networks via Multi-FPGA Bidirectional Ring Topologies and Stream-Dataflow Architectures
- 精读深度: 方法级（仅方法摘录窗口+补架构段）+依据：§4流式核NPU+SynapseRouter；双向环+Aurora跨FPGA；突触表按目的近邻排序；局部/全局同步令牌；HLS U55C；摘录窗偏评测已补架构

## 可继承 A
多FPGA双向环流式SNN：NPU八车道 + 突触表近邻排序左右环路由 + 同步令牌防死锁——可扩展尖峰仿真与环拓扑对照（借入≠X）。

## 强对照 B
单卡脉动SNN；无环的星型/总线互连；NEST纯CPU强扩展；无近邻排序的乱序突触发射。

## 可差分 X线索
NeuroRing多FPGA≠lifting X；仿真加速旁路，差分不在RTF/nJ。

## 与 F1–F7 / Stage B 关系
F5/F7弱相关（突触表驻留、跨核打包通信）。平台旁路。不抢 Stage B。f_candidates空。

## 不可搬用边界
皮层微环路RTF≠valid825；仅方法窗；HBM突触表≠lifting源字RF合同。

## 可复用 idea 点
- 双向环+较短路径左右发射作跨核路由合同
- 突触表按目的近邻排序作通信局域模板
- 局部/全局同步令牌作反压/死锁旁证
- NPU∥SynapseRouter流重叠作吞吐语言 | 负结果只停该环拓扑替换

## 杀门建议
环拥塞或同步税吃掉扩展增益挤占 Stage B → 保持旁路。
