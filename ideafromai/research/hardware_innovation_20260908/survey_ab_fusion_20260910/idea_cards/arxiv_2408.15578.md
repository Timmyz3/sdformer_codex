# arXiv:2408.15578 · FireFly-S

- uid/来源: `MAIN-R059`｜arxiv_2408.15578+本地excerpt（`p0_excerpt_batches/batch_02.json`）
- 题名: FireFly-S: Exploiting Dual-Side Sparsity for Spiking Neural Networks Acceleration with Reconfigurable Spatial Architecture
- 精读深度: 方法级（仅方法摘录窗口）+依据：联合剪枝+量化训练；Algo1 时序膜电位评估做静默通道剪枝；gradient rewiring；双端（W+spike）稀疏；片上 spatial 架构动机

## 可继承 A
双端稀疏（权+脉冲）联合压缩 + Algo1 时序膜电位识别静默输出通道——可作 F1 共同删字/通道级结构剪枝与 Gustav 供数不规则性对照（借入≠X）。

## 强对照 B
仅权稀疏忽略脉冲端；overlay 反复片外访存；剪枝与量化分步+再训练；8–16b 无联合优化。

## 可差分 X线索
FireFly-S 双端稀疏≠标题 X；须落到 r1 源∩W 物理字与双 PED；与批01 错绑 FireFly→SCNN 区分。

## 与 F1–F7 / Stage B 关系
F1/F7 强相关对照；Stage B 后可挂。不抢 Stage B。

## 不可搬用边界
FPGA SNN 分类吞吐≠本地光流 AEE 合同；Algo1 通道剪枝细节摘录有截断；勿搬原作加速比。

## 可复用 idea 点
- Algo1 膜电位时序静默通道→F1 通道级共同删候选模板
- 联合 prune+quant 顺序作压缩序列杀门
- 双端稀疏计费对齐源∩W 并集费用
- 负结果只停该联合压缩布局，不杀双端稀疏家族

## 杀门建议
通道剪后 AEE 破门或并集读仍满/服务不升 → 停该 FireFly-S 布局。
