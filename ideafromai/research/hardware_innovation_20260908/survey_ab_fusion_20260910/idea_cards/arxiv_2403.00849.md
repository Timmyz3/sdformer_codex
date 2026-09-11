# arXiv:2403.00849 · NeuraLUT

- uid/来源: `MAIN-R158`｜arxiv_2403.00849+本地excerpt（`p0_excerpt_batches/batch_02.json`）
- 题名: NeuraLUT: Hiding Neural Network Density in Boolean Synthesizable Functions
- 精读深度: 方法级（仅方法摘录窗口）+依据：§III 把稠密 MLP 子网藏进 L-LUT、LogicNets 式 a priori fan-in 稀疏、子网内 skip-connection 藏于 LUT、电路级层间高稀疏/量化 β

## 可继承 A
「电路级稀疏壳 + LUT 内藏稠密子网」共设计：高表达力落在布尔可综合函数里、层间仅暴露稀疏量化连边——可作 F1/F4「删暴露源字、保留隐藏计算」与浅电路延迟对照底座（借入≠X）。

## 强对照 B
LogicNets 线性神经元 L-LUT；PolyLUT 多项式 L-LUT；暴露 datapath 的 LUTNet/BNN；深电路级 DNN。

## 可差分 X线索
FPGA LUT 封装≠ lifting 结构化 T10/源字删字 X；差分须落到 r1 物理暴露端口与双 PED 损失，而非复述 L-LUT 子网。

## 与 F1–F7 / Stage B 关系
F1/F4 结构压缩与暴露面对照；旁路主岛。不抢 Stage B。

## 不可搬用边界
MNIST/jet tagging FPGA 延迟≠本地 same-port 净服务；勿搬 1.3–4.3×；仅方法摘录窗口（训练超参/综合细节后半可能截断）。

## 可复用 idea 点
- 暴露面 fan-in/β 作「可共同删物理字」上界模板
- 隐藏 skip 不改电路拓扑→对照 F1 删字时保留表达通道
- 电路级层数压缩作延迟杀门叙事对照（非搬 LUT）
- 负结果只停「LUT 子网挂 r1」布局

## 杀门建议
映射到源字后暴露端口/服务不降或破 AEE → 停该封装布局，保留 LUT 文献对照。
