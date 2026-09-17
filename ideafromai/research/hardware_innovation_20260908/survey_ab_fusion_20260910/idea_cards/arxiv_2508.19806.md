# arXiv:2508.19806 · CSSL

- uid/来源: `ARX-004`｜arxiv_2508.19806+本地excerpt（`p0_excerpt_batches_gap/gap_05.json`）
- 题名: Context-aware Sparse Spatiotemporal Learning for Event-based Vision
- 精读深度: 方法级（仅方法摘录窗口）+依据：§II CSSL；Eq.1 上下文阈vth=σ(Wvx+bv)+Heaviside掩码；残差/Minimal Gated RNN扩展；代理梯度；完整表可能截断

## 可继承 A
上下文感知阈控卷积/循环：按输入分布学像素级vth，Heaviside掩码抑冗余激活——事件稀疏时空学习与激活密度合同对照（借入≠X）。

## 强对照 B
固定ReLU阈；稀疏损失手调；无上下文的事件卷积；稠密帧式卷积全激活。

## 可差分 X线索
CSSL阈控≠lifting X；事件检测/光流旁路，差分不在mAP/EPE%。

## 与 F1–F7 / Stage B 关系
F1弱相关（敏感激活/源活动剪枝语言）。旁路基线。不抢 Stage B。f_candidates空。

## 不可搬用边界
检测/光流指标≠valid825；仅摘录窗；代理梯度稀疏≠same-port净服务%。

## 可复用 idea 点
- 像素级vth=σ(Wv x)+H(ỹ−vth)作激活密度合同
- 残差块后阈控保持稀疏作层间模板
- 扩展到卷积RNN/Minimal Gated作时空稀疏旁证
- 与固定阈/稀疏损失成簇 | 负结果只停该阈控替换

## 杀门建议
阈控无增益或代理梯度不稳挤占 Stage B → 保持旁路。
