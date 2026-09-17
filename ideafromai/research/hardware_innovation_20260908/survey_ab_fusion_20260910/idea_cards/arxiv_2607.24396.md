# arXiv:2607.24396 · SpiNNaker2 Many-Core Platform

- uid/来源: `ARX-163`｜arxiv_2607.24396+本地excerpt（`p0_excerpt_batches_gap/gap_07.json`）
- 题名: The SpiNNaker2 chip: a many-core platform for flexible and scalable brain-inspired computing
- 精读深度: 方法级（方法摘录窗口+补本地PDF §III架构）+依据：152 PE(ARM M4F+MLA 16×4+数值加速)；Quad/NoC；六角事件路由；PE级DVFS；OctopuScheduler；原摘录偏评测已补架构

## 可继承 A
可扩展脑启发众核：软件定义PE + 片上MLA/数值加速 + 六角组播事件路由 + PE级DVFS/Auto-PL + OctopuScheduler层调度——相对固定突触-神经元ASIC与无DVFS平台的灵活/自适应对照（借入≠X）。

## 强对照 B
固定突触-神经元专用核；无加速器的纯MCU；无DVFS的恒定高压；仅片外调度的DNN。

## 可差分 X线索
SpiNNaker2平台≠lifting X；众核平台旁路，勿搬TOPS/W或突触事件/s当净服务%。

## 与 F1–F7 / Stage B 关系
加速器/平台旁路。不抢 Stage B。f_candidates空。

## 不可搬用边界
芯片PPA/DVS准确率≠valid825；补PDF架构；组播路由≠lifting并集合同。

## 可复用 idea 点
- 软件PE+MLA作混合SNN/DNN合同
- 六角事件路由组播作稀疏通信模板
- PE级DVFS/Auto-PL作尖峰负载自适应旁证
- OctopuScheduler片上瓦片调度作DNN边
- 负结果只停该众核平台挂接

## 杀门建议
平台迁移成本或与Stage B同分母冲突 → 保持旁路。
