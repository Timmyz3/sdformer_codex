# arXiv:2411.14733 · FLARE / BitSift

- uid/来源: `MAIN-R149`｜arxiv_2411.14733+本地excerpt（`p0_excerpt_batches/batch_02.json`）
- 题名: FLARE: FP-Less PTQ and Low-ENOB ADC Based AMS-PiM for Error-Resilient, Fast, and Efficient Transformer Acceleration
- 精读深度: 方法级（仅方法摘录窗口）+依据：BitSift-GEMV（比特级稀疏远高于值级）；低 ENOB ADC + 误差韧性；FP-less PTQ；最长输入切片组合硬件友好技巧；AMS-PiM 阵列周期账

## 可继承 A
比特级跳零 GEMV + 模拟存内低精度 ADC 误差韧性——普通压缩/近似计算对照；「细粒度稀疏>>粗粒度」实证模板（借入≠X）。

## 强对照 B
高 ENOB ADC 脆弱传感裕量；值级稀疏 GEMV；含 FP 的 PTQ 流水；无比特解析的稠密模拟阵列。

## 可差分 X线索
AMS-PiM/BitSift≠ lifting X；模拟误差模型不可写成 Stage B 数字净服务。

## 与 F1–F7 / Stage B 关系
普通压缩/近似旁路（F7）；不抢 Stage B。

## 不可搬用边界
Transformer AMS-PiM 预印本；工艺/ADC 假设≠本地数字 RTL；仅方法摘录窗口。

## 可复用 idea 点
- 比特稀疏 vs 值稀疏差距图作 F1 粒度选择证据
- 低精度容错作 F4 有损完成证书对照
- 最长切片组合作硬件友好打包启发式
- 负结果只停模拟 PiM 布局

## 杀门建议
误差韧性不足破 AEE，或数字同分母无增益 → 停该 BitSift/ADC 布局。
