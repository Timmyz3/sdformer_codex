# arXiv:1802.03806 · ThUnderVolt

- uid/来源: `MAIN-R137`｜arxiv_1802.03806+本地excerpt（`p0_excerpt_batches/batch_01.json`）
- 题名: ThUnderVolt: Enabling Aggressive Voltage Underscaling and Timing Error Resilience for Energy Efficient Deep Learning Accelerators
- 精读深度: 方法级（仅方法摘录窗口）+依据：§3.1 TE-Drop（Razor 检错、不重算错误 MAC）、§3.2/Algo1 逐层电压欠标度与误差预算均分、Fig.5 在线误差跟踪、与 Zero-Skip 协同评测

## 可继承 A
有界误差预算下的「可丢弃/可近似」完成：TE-Drop 遇时序错跳过该 MAC 贡献；Algo1 按层排序均分 ptot；在线列求和误差跟踪——可作 F2/F4「有损共同完成/回退证书」的电路侧对照底座（借入≠X）。

## 强对照 B
标准 TED 重放纠错；全层统一欠压；无误差预算的盲目近似乘；纯 Zero-Skip。

## 可差分 X线索
电压/时序近似≠ lifting 半步检查点 X；只有把「错误/不确定则跳过或回退」挂到 lifting RNE/末5 证书且同分母才可能差分。

## 与 F1–F7 / Stage B 关系
F2/F4 备选对照；第二队列。不抢 Stage B。

## 不可搬用边界
TPU 式 256×256 MAC 系统仿真≠本地 finite_service；不得搬 36–56% 能效；仅方法摘录窗口（TE-Drop 微结构细节有截断）。

## 可复用 idea 点
- 用「误差预算均分层」类比 F2 组接受预算，不作电路移植
- TE-Drop「检错后不重算」作 F4 证书失败回退的负对照（重算贵）
- 与 Zero-Skip 协同说明稀疏跳零与近似完成可正交
- 负结果只停「电压近似挂 r1」布局

## 杀门建议
同分母下近似完成净服务不胜静态窄层/严格计算，或 AEE 破门 → 停该近似布局。
