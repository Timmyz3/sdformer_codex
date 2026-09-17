# arXiv:2606.05362 · MOSAIC Heterogeneous NPU DSE

- uid/来源: `ARX-033`｜arxiv_2606.05362+本地excerpt（`p0_excerpt_batches_gap/gap_06.json`）
- 题名: MOSAIC: A Workload-Driven Simulation and Design-Space Exploration Framework for Heterogeneous NPUs
- 精读深度: 方法级（仅方法摘录窗口）+依据：§3四层(输入/编译/仿真/校准)+DSE；DAG映射+op分裂；屋顶线；异构tile(MAC/DSP/SFU含LIF)；NoC/DRAM共享；ASAP7校准；完整RTL门控可能截断

## 可继承 A
异构NPU工作负载DSE：精度/融合/DAG映射+分裂 + 七模块tile仿真(含LIF SFU) + 多种子扫+GA——异构核探索与同构脉动对照（借入≠X）。

## 强对照 B
单tile同构仿真；无op分裂的整算子映射；静态DRAM均分；无SNN专用SFU的通用DNN DSE。

## 可差分 X线索
MOSAIC仿真DSE≠lifting X；工具旁路，勿搬Pareto延迟/能当净服务%。

## 与 F1–F7 / Stage B 关系
F5/F7弱相关（映射调度、跨tile打包）。工具旁路。不抢 Stage B。f_candidates空。

## 不可搬用边界
仿真PPA≠硅测；仅摘录窗；LIF SFU公式≠完整神经形态合同。

## 可复用 idea 点
- DAG拓扑序+兼容tile最早完成作映射合同
- OC/B/IC分裂+显式reduce税作并行模板
- 动态DRAM带宽共享作异构旁证
- SFU LIF公式作尖峰算子接入边
- 负结果只停该DSE挂接

## 杀门建议
校准偏差大或搜索不可复现挤占 Stage B → 保持旁路。
