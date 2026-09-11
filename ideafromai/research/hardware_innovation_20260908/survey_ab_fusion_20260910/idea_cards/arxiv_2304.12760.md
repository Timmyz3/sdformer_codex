# arXiv:2304.12760 · PSN(Fang)

- uid/来源: `MAIN-R198`｜arxiv_2304.12760+本地excerpt（`p0_excerpt_batches/batch_01.json`）
- 题名: Parallel Spiking Neurons with High Efficiency and Ability to Learn Long-term Dependencies
- 精读深度: 方法级（仅方法摘录窗口）+依据：去复位后线性充电可并行前缀；PSN 为可学习 W∈R^{T×T} 满阶；k-order masked PSN（因果带状掩码+渐进掩码训练）；sliding PSN 共享时延卷积核

## 可继承 A
并行/带状/滑动 PSN 公式族——与本地结构化 T10/lifting PSN 直接相关的神经元底座与强对照谱系（借入原作并行化≠X）。

## 强对照 B
逐步迭代 LIF/IF 含复位；Vth=∞ 禁放；无掩码满阶非因果导致层延迟 T。

## 可差分 X线索
原作可学习满 W 或 sliding≠本地 lifting 结构化因子/半步图 X；本地主张在结构化执行对象与消费者接口。

## 与 F1–F7 / Stage B 关系
直接支撑 lifting/PSN 主岛理解；Stage B 用普通 dense-source/raw 对照时本族为算法先验。不抢 Stage B 排程。

## 不可搬用边界
原作分类/序列任务；masked/sliding 渐进训练≠本地 θg/非因果 T10 合同；不得把并行前缀加速写成 Stage B 净服务%；仅方法摘录窗口。

## 可复用 idea 点
- k-order 掩码作因果/有限依赖对照，对比本地非因果 T10
- 渐进掩码 λ:1→Mk 作训练稳定技巧（借入）
- sliding 共享核作「时间维低参」强对照 vs lifting 结构化
- 去复位并行前缀只作仿真加速先验，不进硬件标题

## 杀门建议
若仅替换为原作满 W/sliding 而无 lifting 结构，相对 ordinary 无净服务/精度门 → 停「当标题」，保留对照。
