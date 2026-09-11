# arXiv:2102.04159 · SEW-ResNet

- uid/来源: `MAIN-R257`｜arxiv_2102.04159+本地excerpt（`p0_excerpt_batches_gap/gap_01.json`）
- 题名: Deep Residual Learning in Spiking Neural Networks
- 精读深度: 方法级（仅方法摘录窗口）+依据：统一 IF/LIF 离散方程；Spiking ResNet vs SEW 块；g∈{ADD,AND,IAND} 实现恒等映射；Fig.1；摘录偏§3 块设计，完整训练/深度扩展可能截断

## 可继承 A
尖峰元素级残差（SEW）用二元运算实现恒等映射——深层 SNN 残差通路与 shortcut 语义对照（借入≠X）。

## 强对照 B
ReLU ResNet 直接换 SN 的 Spiking ResNet（难恒等）；无 shortcut；仅膜域连续残差。

## 可差分 X线索
SEW/ADD–AND≠lifting 结构化 PSN X；可借残差可训性，勿称 SEW-X。

## 与 F1–F7 / Stage B 关系
算法/训练底座旁证；弱挂 F4（残差通路合并纪律）。不抢 Stage B。

## 不可搬用边界
ImageNet/CIFAR 准确率≠AEE 合同；仅摘录窗；禁止把 SEW 速度数字写成 same-port%。

## 可复用 idea 点
- ADD/AND/IAND 恒等条件作 shortcut 合同表
- 与本地残差链 sn1→Conv→sn2 语义对照
- 负结果只停某 g 函数布局，不杀残差家族

## 杀门建议
同预算不胜普通 Spiking ResNet/窄稠密或破 AEE → 停该 SEW 布局。
