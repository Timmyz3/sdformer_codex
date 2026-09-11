# arXiv:2505.12292 · SpikeX

- uid/来源: `MAIN-R061`｜arxiv_2505.12292+本地excerpt（`p0_excerpt_batches/batch_03.json`）
- 题名: SpikeX: Exploring Accelerator Architecture and Network-Hardware Co-Optimization for Sparse Spiking Neural Networks
- 精读深度: 方法级（仅方法摘录窗口）+依据：引言/背景级 Agile SpatioTemporal Dispatch、Activation-induced Weight Tailoring、NTWU、稀疏参数化能耗时延模型做网-芯共设计；完整调度算法细节可能截断

## 可继承 A
面向时空不规则尖峰的敏捷派发 + 激活诱导权裁剪 + 稀疏桥接的网芯共优化接口——作「占用/活动驱动的执行与训练共设计」底座（借入≠X；与本地 F6/块占用叙事相关）。

## 强对照 B
ANN 式脉动阵列无视尖峰稀疏；训练与加速器配置割裂；固定映射无视时间窗。

## 可差分 X线索
SpikeX 派发/裁剪≠ lifting 结构化因子图 X；共优化接口可作对照，标题差分仍在 T10 执行对象。

## 与 F1–F7 / Stage B 关系
F1/F6/F7 相关（活动裁剪/占用费用）；第二队列。不抢 Stage B。

## 不可搬用边界
EDP 15×–150× 勿搬；摘录以动机/总览为主，缺完整 Algo 细步；仅方法摘录窗口。

## 可复用 idea 点
- NTWU（神经元×时间窗工作单元）作打包粒度对照
- Activation-induced Weight Tailoring↔活动感知供数面
- 稀疏参数化能耗模型作训练目标接口模板（对照 F5/F6）
- 负结果只停「照搬 SpikeX 映射到 PSN」

## 杀门建议
共优化后同精度净服务不胜静态窄层/普通稀疏跳过 → 停该共设计布局。
