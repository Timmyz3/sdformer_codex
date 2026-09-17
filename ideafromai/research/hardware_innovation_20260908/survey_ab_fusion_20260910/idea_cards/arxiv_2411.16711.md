# arXiv:2411.16711 · TSkips

- uid/来源: `ARX-011`｜arxiv_2411.16711+本地excerpt（`p0_excerpt_batches_gap/gap_04.json`）
- 题名: TSkips: Efficiency Through Explicit Temporal Delay Connections in Spiking Neural Networks
- 精读深度: 方法级（仅方法摘录窗口）+依据：§3.3 前/后向 TSkip+Δt 约束；Eq.3–4 α 混合；NASWOT-SAHD 搜 Δt/位置；BPTT+BNTT；DSEC 骨干挂 encoder–decoder skip；完整 NAS 表可能截断

## 可继承 A
显式时间延迟跨层捷径（前/后向 TSkip，0<Δt，禁未来）+ 可学习 α 混合当前/延迟——SNN 时间依赖传播与检查点旁证（借入≠X）。

## 强对照 B
标准 skip Δt=0；vRNN 固定 Δt=1 后向；无延迟稠密残差；仅加深不加时间捷径。

## 可差分 X线索
TSkips≠lifting 源字/半步图 X；表示/训练旁路，差分不在 DSEC AEE%。

## 与 F1–F7 / Stage B 关系
F1/F2/F4相关（表示、时间延迟、残差/检查点）。旁路基线。不抢 Stage B。f_candidates含F1,F2,F4。

## 不可搬用边界
DSEC/Gesture 精度≠valid825；仅摘录窗；NAS 搜到的 Δt 勿写 same-port%；禁止虚报完整搜索表。

## 可复用 idea 点
- 显式 Δt 前/后向捷径作时间检查点合同（对照F2/F4）
- α 混合当前与延迟通道作可控时间混合旁证
- NASWOT-SAHD 无训分作 Δt/位置搜索模板
- 挂 EV-FlowNet 式编解码 skip 作光流骨干试挂 | 负结果只停该 TSkip 布局

## 杀门建议
挂主网无 AEE/服务增益或延迟缓冲挤占 Stage B → 保持旁路。
