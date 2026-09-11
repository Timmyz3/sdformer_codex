# arXiv:2511.06770 · ASTER

- uid/来源: `MAIN-R203`｜arxiv_2511.06770+本地excerpt（`p0_excerpt_batches/batch_03.json`）
- 题名: ASTER: Attention-based Spiking Transformer Engine for Event-driven Reasoning
- 精读深度: 方法级（仅方法摘录窗口）+依据：TAFT 低活动注意力块 identity 跳过；CBET 置信度早退时步；贝叶斯选 (τ,β)；PIM/RRAM 层次；可编程 WL 按非零激活选通；SDSA 二值 mask-and-add 数据流

## 可继承 A
任务感知层跳过 + 置信度时步早退 + 按输入稀疏选通 WL 的 PIM 数据流——「动态减少注意力/时步工作量」完整先验，紧贴 F2/F4/F6（借入≠X）。

## 强对照 B
固定全层全时步推理；粗粒度 bank 门控；无稀疏感知的稠密矩阵向量。

## 可差分 X线索
TAFT/CBET/PIM≠ lifting 半步检查点 X；差分挂在 lifting RNE 边界与 r1 并集费用，而非 RRAM 模拟乘。

## 与 F1–F7 / Stage B 关系
F2/F4/F7 强相关；第二队列。不抢 Stage B。

## 不可搬用边界
PIM/RRAM 能耗数字勿搬到数字 same-port；层跳过一次静态决定≠在线 lifting 取消；仅方法摘录窗口。

## 可复用 idea 点
- TAFT 低尖峰块→identity 作「整块退休」模板对照 F2
- CBET 早退作时步维有损共同完成
- (τ,β) 帕累托选阈作杀门标定方法
- 细粒度 WL mask 按 Hamming 计费↔活动驱动供数

## 杀门建议
跳过/早退后净服务不胜静态窄层+同组关闭，或 AEE 破门 → 停该动态剪布局。
