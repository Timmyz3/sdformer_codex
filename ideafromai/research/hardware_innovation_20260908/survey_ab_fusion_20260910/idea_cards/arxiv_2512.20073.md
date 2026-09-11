# arXiv:2512.20073 · 3DS-ISC

- uid/来源: `MAIN-R240`｜arxiv_2512.20073+本地excerpt（`p0_excerpt_batches/batch_06.json`）
- 题名: 3D Stack In-Sensor-Computing (3DS-ISC): Accelerating Time-Surface Construction for Neuromorphic Event Cameras
- 精读深度: 方法级（仅方法摘录窗口）+依据：摘要/贡献列 eDRAM 漏电作指数衰减 Time-Surface、MOMCAP 电荷存储与 LL switch 延时、3D 堆叠感存算一体相对 2D/16-bit SRAM-TPI；摘录窗因命中 abstract「method」偏前，§III 电路共设计/非理想细部不在窗内

## 可继承 A
感存算一体的 per-pixel Time-Surface 驻留（用器件物理衰减替代多比特时间戳 SRAM）——作「前端表示驻留替代片外时间戳搬运」的工程对照分母（借入≠X）。

## 强对照 B
片外算 TS；片上 16-bit SRAM/TPI 存时间戳；纯事件队列无空间索引；无衰减的 SAE（溢出/复位问题）。

## 可差分 X线索
模拟 eDRAM-TS / 3D-ISC ≠ lifting 源字物理删字 X；主岛在数字残差执行，不在传感器模拟阵列。

## 与 F1–F7 / Stage B 关系
F5 弱相关（每址状态驻留叙事）；前端旁路。不抢 Stage B。

## 不可搬用边界
勿搬 69×/2.2×/1.9× 或「三数量级功耗」原作 PPA；N-MNIST/Gesture 分类与 SSIM≠valid825 AEE；仅方法摘录窗口（§III 截断）。

## 可复用 idea 点
- 「用物理衰减代替数字时间戳字」作表示驻留费用对照
- 3D 近传感互联能量叙事 ↔ 同端口搬移税对照（非搬 TSV RTL）
- STCF 去噪+TS 输入 DNN 的前端合同边界
- 与 SSER/EBBI 等事件前端成对阅读
- 负结果只停「模拟 TS 前端替换本地」布局

## 杀门建议
前端替换后 AEE/服务无增益，或模拟非理想无法映射到数字合同 → 保留文献对照，不进 A+B 标题。
