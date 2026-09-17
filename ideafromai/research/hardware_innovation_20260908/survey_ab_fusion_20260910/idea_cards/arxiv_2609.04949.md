# arXiv:2609.04949 · Hopf-Bifurcation Spike Detector

- uid/来源: `ARX-021`｜arxiv_2609.04949+本地excerpt（p0_excerpt_batches_gap/gap_08.json）+补PDF方法（Hopf/NDR装置）
- 题名: Noise-Resilient Detection of Neuronal Spikes by a Hopf-Bifurcation Device
- 精读深度: 方法级（摘录较短；补PDF摘要+实验方法）+依据：NDR近Hopf分岔；相干阈穿→全有全无振荡尖峰；异步无时钟检测；MEA电生理对照传统带通+峰值检测；完整理论可能截断

## 可继承 A
分岔工程NDR弱信号检测：近Hopf工作点把持续相干输入转为全有全无电压尖峰、抑制快随机起伏——相对纯数字滤波/同步检测的模拟事件前端旁路对照（借入≠X；传感前端）。

## 强对照 B
已知波形匹配滤波/同步检波；连续高速数字化后软件峰值检测；线性放大后阈值。

## 可差分 X线索
Hopf尖峰检测器≠lifting X；电生理前端旁路，差分不在−54dB演示点。

## 与 F1–F7 / Stage B 关系
传感/前端旁路。不抢 Stage B。f_candidates空。

## 不可搬用边界
光伏/MEA尖峰一致率≠valid825；补PDF；模拟分岔≠lifting源字物理删字。

## 可复用 idea 点
- 近分岔相干积分作噪声抑制合同
- 全有全无尖峰作事件化ADC旁证
- 模拟域判别减连续数字化作植入带宽边
- 负结果：不挂主岛RTL

## 杀门建议
无执行/供数接口 → 不进入Stage B候选池。
