# arXiv:2605.09770 · Spiking Bandpass Wavelets

- uid/来源: `ARX-041`｜arxiv_2605.09770+本地excerpt（`p0_excerpt_batches_gap/gap_05.json`）
- 题名: Encoding and Decoding Temporal Signals with Spiking Bandpass Wavelets
- 精读深度: 方法级（仅方法摘录窗口）+依据：Algo1 DoE/DoT尺度空间差分→LIF极化尖峰；低通+带通通道；最小二乘解码器；框架界/误差界；Morlet/Szu对照；完整证明可能截断

## 可继承 A
尖峰带通小波：差分低通尺度空间得DoE/DoT带通 + LIF量化 + 可证框架界重建——因果稀疏时间编码与非因果Morlet对照（借入≠X）。

## 强对照 B
非因果Morlet；稠密小波系数；无框架界的启发式LIF编码；仅低通无带通通道。

## 可差分 X线索
尖峰小波编码≠lifting X；表示/前端旁路，勿搬重建误差当净服务%。

## 与 F1–F7 / Stage B 关系
F1弱相关（多尺度表示选择）。表示旁路。不抢 Stage B。f_candidates空。

## 不可搬用边界
重建MSE≠valid825；仅摘录窗；LS解码≠在线流式解码器。

## 可复用 idea 点
- ΔL_k=L_k−L_{k−1}+LIF作带通尖峰合同
- 框架界A,B作可证重建模板
- 低通通道分立携带粗结构作合成旁证
- 与Morlet/Szu成簇：因果稀疏 vs 非因果 | 负结果只停该小波前端替换

## 杀门建议
重建界松或在线解码差距大挤占 Stage B → 保持旁路。
