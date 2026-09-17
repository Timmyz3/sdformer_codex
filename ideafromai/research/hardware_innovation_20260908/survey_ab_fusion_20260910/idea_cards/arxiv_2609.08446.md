# arXiv:2609.08446 · FlexSpIM Hybrid-Stationary CIM

- uid/来源: `ARX-158`｜arxiv_2609.08446+本地excerpt（p0_excerpt_batches_gap/gap_08.json）+补arxiv HTML方法（§III–IV）
- 题名: FlexSpIM: An Event-Based Digital Compute-In-Memory Accelerator with Flexible Operand Resolution and Layer-Wise Hybrid Stationarity
- 精读深度: 方法级（方法摘录窗口+补HTML §III宏/§IV-A混合驻留）+依据：数字列内CIM+可配外设；任意权/膜电位分辨率；WS/OS层间混合驻留；膜电位bitcell双向进位消交错；ZigZag映射栈；完整系统评测可能截断

## 可继承 A
事件数字CIM：可重构分辨率PC + 权/膜电位统一存储（WS∥OS）+ 层间混合驻留(HS)减重载 + 层优先调度保DVS低延迟——相对固定分辨率/仅WS的SNN-CIM对照（借入≠X）。

## 强对照 B
固定位宽/少数档位CIM；仅权驻留(WS)；层间反复装权；阵列内模拟CIM+ADC主导；帧优先丢事件延迟。

## 可差分 X线索
FlexSpIM≠lifting X；CIM加速旁路，勿搬TOPS/W或宏能效当净服务%。

## 与 F1–F7 / Stage B 关系
加速器旁路。不抢 Stage B。f_candidates空。

## 不可搬用边界
SCNN映射能效≠valid825；方法窗+HTML；混合驻留≠lifting多消费者并集合同。

## 可复用 idea 点
- 层间HS(WS/OS)作驻留选择合同
- 可配分辨率PC作精度-面积旋钮
- 膜电位bitcell双向进位作消交错模板
- 层优先∥事件稀疏作DVS延迟边
- 负结果只停该CIM挂接

## 杀门建议
映射几何与Stage B同分母冲突或无端口差分 → 保持旁路。
