# arXiv:2603.15184 · CATFormer

- uid/来源: `MAIN-R211`｜arxiv_2603.15184+本地excerpt（`p0_excerpt_batches/batch_03.json`）
- 题名: CATFormer: When Continual Learning Meets Spiking Transformers With Dynamic Thresholds
- 精读深度: 方法级（仅方法摘录窗口）+依据：动态阈值适应防遗忘；Gated Dynamic Head Selection（门控 MLP 路由任务头）；Algo1/2 训练与推理；摘录偏实验对比，机制公式可能截断

## 可继承 A
动态阈值 + 门控任务头路由实现无回放持续学习——算法侧「阈值/路由适应」对照，非执行稀疏主线（借入≠X）。

## 强对照 B
无适应固定阈值 SNN；EWC/MAS 正则；依赖记忆缓冲的 rehearsal。

## 可差分 X线索
持续学习动态阈值≠ lifting/源字 X；与本地光流主岛正交。

## 与 F1–F7 / Stage B 关系
F2 弱（路由/选择）；旁路。不抢 Stage B。

## 不可搬用边界
CIL 精度表≠AEE；无缓冲声明勿外推硬件 RF；仅方法摘录窗口。

## 可复用 idea 点
- 动态阈值作神经元级适应旋钮阅读
- 门控头选择作多专家路由对照
- 无 rehearsal 约束作存储边界叙事
- 负结果只停「CIL 头挂光流」

## 杀门建议
对主岛无净服务/精度贡献 → 保持旁路，不进标题。
