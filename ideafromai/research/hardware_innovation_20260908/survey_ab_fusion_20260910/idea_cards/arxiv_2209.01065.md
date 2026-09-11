# arXiv:2209.01065 · Kraken

- uid/来源: `MAIN-R243`｜arxiv_2209.01065+本地excerpt（`p0_excerpt_batches_gap/gap_02.json`）
- 题名: Kraken: A Direct Event/Frame-Based Multi-sensor Fusion SoC for Ultra-Efficient Visual Processing in Nano-UAVs
- 精读深度: 方法级（仅方法摘录窗口）+依据：异构 SoC=SNE（COO 事件→稠密突发/4b 3×3+8b LIF）+CUTIE（三元全展开/片上权）+PULP 8核混合精度；FC+1MiB L2；摘录§II，完整RTL/时钟域可能截断（摘录含\0噪声已清洗）

## 可继承 A
事件 SNN / 三元 TNN / 量化 DNN 三引擎同 SoC 分工——多模态视觉前端异构加速对照（借入≠X）。

## 强对照 B
单一稠密 DNN 核扛全部传感；无 COO 突发的朴素事件扫描；权反复外存的非展开 TNN。

## 可差分 X线索
Kraken SoC≠lifting X；借异构分工与 COO 突发，勿搬 TOp/s/W。

## 与 F1–F7 / Stage B 关系
F1弱相关（异构引擎/表示选择）；旁路/平台。不抢 Stage B。f_candidates含F1。

## 不可搬用边界
Nano-UAV 能效/inf≠valid825；SNE 子引擎≠完整 Gustav；仅摘录窗；禁止虚报已读完整芯片图。

## 可复用 idea 点
- COO→稠密突发作稀疏事件服务合同
- 三引擎功率门控作任务分域旁证
- 与 SNE 卡成簇勿双计底座
- 负结果只停「整 SoC 替换数字链」

## 杀门建议
异构迁入无 AEE/服务增益或挤占 Stage B → 保持旁路。
