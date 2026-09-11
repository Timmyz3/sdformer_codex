# arXiv:2105.08217 · IMPULSE

- uid/来源: `ARX-056`｜arxiv_2105.08217+本地excerpt（`p0_excerpt_batches_gap/gap_01.json`）
- 题名: IMPULSE: A 65nm Digital Compute-in-Memory Macro with Fused Weights and Membrane Potential for Spike-based Sequential Learning Tasks
- 精读深度: 方法级（仅方法摘录窗口；原 gap 摘录取到参考文献，已用同稿 arXiv PDF 方法窗补齐）+依据：10T-SRAM 融合 WMEM/VMEM；交错映射+可重构列外设；AccW2V/SpikeCheck/ResetV 等片内指令；IF/LIF/残差膜；输入稀疏跳过；65nm 实测

## 可继承 A
权与膜同宏融合+片内 SNN 指令集——有限 RF 下「权/膜谁共位」与事件稀疏跳写对照（借入≠X；联读 FlexSpIM）。

## 强对照 B
权/膜分体存储反复搬运；仅 ANN CIM；无 SpikeCheck/Reset 片内闭环；固定非交错位宽映射。

## 可差分 X线索
CIM 宏≠lifting 数字残差链 X；借驻留/稀疏，勿称 CIM-X。

## 与 F1–F7 / Stage B 关系
F5旁证（膜生存期/共位）；不进主岛。不抢 Stage B。

## 不可搬用边界
65nm 0.99 TOPS/W / 85%稀疏 EDP≠本地 AEE/same-port%；仅方法窗；勿搬能量外推。

## 可复用 idea 点
- AccW2V 同周期更权膜作共位分母
- SpikeCheck→条件 Reset 跳写未尖峰膜
- 交错奇偶列映射作位宽不对齐模板
- 与 FlexSpIM 驻留选择成对
- 负结果只停 CIM 名替换数字链

## 杀门建议
无同端口重读/带宽改善或相位不可映射 → 停类比，不杀有限 RF 家族。
