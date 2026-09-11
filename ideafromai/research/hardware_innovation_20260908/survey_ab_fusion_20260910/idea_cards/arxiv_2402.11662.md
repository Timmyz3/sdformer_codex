# arXiv:2402.11662 · TDE-3

- uid/来源: `MAIN-R218`｜arxiv_2402.11662+本地excerpt（`p0_excerpt_batches/batch_01.json`）
- 题名: TDE-3: An improved prior for optical flow computation in spiking neural networks
- 精读深度: 方法级（仅方法摘录窗口）+依据：相对 TDE-2 增加抑制输入消除残差增益→纹理下 DSI=1；BPTT+代理梯度；spike count vs ISI 推理/训练；速度线性映射、空间频率与噪声稳健性实验

## 可继承 A
生物启发相关运动检测先验（方向选择+可训速度编码）——光流/运动旁路对照；ISI 训更省脉冲的观察（借入≠X）。

## 强对照 B
TDE-2（无抑制，纹理下方向选择性差）；无监督的手工调参 TDE。

## 可差分 X线索
TDE 先验≠lifting/r1 主岛 X；运动旁路有限份额，不排主岛。

## 与 F1–F7 / Stage B 关系
旁路（motion）；明确不进 F1–F7 主线。不抢 Stage B。

## 不可搬用边界
合成纹理/单检测器；非端到端网络硬件；仅方法摘录窗口（结构§2 可能在窗口外）。

## 可复用 idea 点
- 抑制消除残差增益作方向选择性结构先验
- spike count vs ISI 训练目标对能耗/动态范围的消融模板
- 空间频率变化下的速度编码稳健性测试法
- 负结果只停 TDE 旁路布局，不影响 Stage B

## 杀门建议
接入主网络后无净 AEE/服务收益或挤占主岛档期 → 停旁路，不杀主实验。
