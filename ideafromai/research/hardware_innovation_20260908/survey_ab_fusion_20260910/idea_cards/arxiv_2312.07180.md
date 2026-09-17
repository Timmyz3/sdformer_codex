# arXiv:2312.07180 · DFlow / Context-aware Iteration Policy

- uid/来源: `MAIN-R235`｜arxiv_2312.07180+本地excerpt（`p0_excerpt_batches_gap/gap_03.json`）
- 题名: Context-Aware Iteration Policy Network for Efficient Optical Flow Estimation
- 精读深度: 方法级（仅方法摘录窗口）+依据：统计瓶颈观察；Gumbel mask跳迭代；历史隐状态+迭代嵌入+增量损失；资源偏好r；完整损失/四骨干表可能截断

## 可继承 A
上下文感知迭代策略网：按样本动态跳过光流更新步（瓶颈/边际收益）——迭代预算自适应与可控算力合同对照（借入≠X）。

## 强对照 B
固定大迭代（RAFT 24/32）；无策略的早退阈值；仅当前特征无历史/未来上下文的跳步。

## 可差分 X线索
迭代策略≠lifting源字X；旁路效率轴，差分不在Sintel EPE%。

## 与 F1–F7 / Stage B 关系
F2/F4相关（时间/迭代检查点、有损跳过）。第二队列对照。不抢 Stage B。f_candidates含F2,F4。

## 不可搬用边界
FLOPs降约40%/20%≠same-port服务%；帧光流≠事件valid825；仅摘录窗。

## 可复用 idea 点
- Gumbel可微mask作接受/继续半步旁证对照F2/F4
- 历史隐元+正弦迭代嵌入作上下文合同
- 资源偏好r单网多档作可控预算
- 与RAFT族成簇 | 负结果只停该策略头

## 杀门建议
跳步伤EPE超预算或策略本身费用抵消收益 → 停该策略，保留固定迭代基线。
