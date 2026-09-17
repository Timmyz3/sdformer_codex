# arXiv:2602.12590 · Functional Binning FBP

- uid/来源: `ARX-002`｜arxiv_2602.12590+本地excerpt（`p0_excerpt_batches_gap/gap_05.json`）
- 题名: Unbiased Gradient Estimation for Event Binning via Functional Backpropagation
- 精读深度: 方法级（仅方法摘录窗口）+依据：§3.1–3.3 FBP链规则；弱导数κ=l∗k；Algo1前向不变+反向κ′；偏置分析vs rect/linear/gauss；多维扩展可能截断

## 可继承 A
事件分箱用函数反传合成弱导数κ=l∗k，前向核不变、仅改反向——无偏事件分箱梯度与启发式代理对照（借入≠X）。

## 强对照 B
不可微rect分箱；线性/高斯启发式代理；只反传到权不反传到坐标；有偏长程差分。

## 可差分 X线索
FBP分箱梯度≠lifting X；训练前端旁路，勿搬光流/IWE指标当净服务%。

## 与 F1–F7 / Stage B 关系
F2弱相关（时间分箱/连续状态梯度）。训练旁路。不抢 Stage B。f_candidates空。

## 不可搬用边界
IWE/光流基准≠valid825；仅摘录窗；弱导数实现≠硬件事件前端。

## 可复用 idea 点
- κ=l∗k替换k′作无偏弱导数合同
- Algo1前向不变+VJP用κ′作训练模板
- Dirac梳余切重构作连续运动流旁证
- 与linear/gauss代理成簇 | 负结果只停该分箱反传替换

## 杀门建议
无偏梯度无下游增益或算力税挤占 Stage B → 保持旁路。
