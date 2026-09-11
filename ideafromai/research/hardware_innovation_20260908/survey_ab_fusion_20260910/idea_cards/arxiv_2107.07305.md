# arXiv:2107.07305 · Delta Activation Layer

- uid/来源: `ARX-055`｜arxiv_2107.07305+本地excerpt（`p0_excerpt_batches_gap/gap_01.json`）
- 题名: Training for temporal sparsity in deep neural networks, application in video processing
- 精读深度: 方法级（仅方法摘录窗口）+依据：Delta Activation Layer 记前激活、量化传播差分；训练期稀疏惩罚+可学量化级；无显式阈值/PID；可插训练/精炼/仅推理；指出全层状态使 ResNet-50 内存+≈40%；摘录§1–2，完整公式/UCF101表可能截断

## 可继承 A
训练诱导时间差分稀疏（Δ激活）并铸成空间稀疏——时间打包/半步检查点的可训稀疏底座（借入≠X）。

## 强对照 B
仅空间 ReLU/剪枝稀疏；推理期固定阈值 delta-net（易漂移需周期复位）；无状态的无记忆层。

## 可差分 X线索
Delta 层≠lifting 源字 X；差分须落到同端口服务，而非只报激活稀疏×3。

## 与 F1–F7 / Stage B 关系
F2/F4相关（时间检查点、有损差分）；第二队列对照。不抢 Stage B。f_candidates含F2,F4。

## 不可搬用边界
UCF101 准确率/稀疏比≠valid825；状态内存+40%须进分母；仅摘录窗。

## 可复用 idea 点
- Δ激活作 lifting 半步「接受/继续」的可训先验对照 F2
- 可学量化级↔稀疏–精度帕累托
- 部分层置状态缓解 RF 压力→F5 旁证
- 负结果只停全层 Delta 布局

## 杀门建议
稀疏升但同端口服务不降，或状态重读吃尽余量 → 停该全层布局，保留部分层轴。
