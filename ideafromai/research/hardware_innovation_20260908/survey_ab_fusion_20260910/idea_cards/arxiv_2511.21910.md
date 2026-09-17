# arXiv:2511.21910 · Platinum

- uid/来源: `ARX-185`｜arxiv_2511.21910+本地excerpt（`p0_excerpt_batches_gap/gap_05.json`）
- 题名: Platinum: Path-Adaptable LUT-Based Accelerator Tailored for Low-Bit Weight Matrix Multiplication
- 精读深度: 方法级（仅方法摘录窗口）+依据：§III离线build path+在线四段LUT构造；Algo1/2；bit-serial↔ternary路径切换；PPE查询/聚合；完整RTL面积表可能截断

## 可继承 A
离线生成可切换构造路径、在线按path做shortcut LUT；ternary专用LUT省并项——低比特LUT加速与Prosperity动态调度对照（借入≠X）。

## 强对照 B
Prosperity运行时shortcut调度；纯bit-serial三元编码；离线预存全LUT；无路径切换的固定构造。

## 可差分 X线索
Platinum LUT路径≠lifting X；同Prosperity族底座，差分不在BitNet×速%。

## 与 F1–F7 / Stage B 关系
F3相关（联合图/公共虚节点/路径调度语言；与Prosperity对照）。第二队列线索。不抢 Stage B。f_candidates含F3。

## 不可搬用边界
BitNet加速比≠valid825；仅摘录窗；LUT路径≠lifting源DAG产品mask。

## 可复用 idea 点
- 离线path+Finish token作去调度器构造合同
- bit-serial↔ternary路径切换作精度特化模板
- LUT[dst]=LUT[src]±a[j]四段流水作面积旁证
- 与Prosperity成簇：静态path vs动态调度 | 负结果只停该LUT路径替换

## 杀门建议
路径切换无增益或LUT构造仍瓶颈挤占 Stage B → 保持旁路/降为F3线索。
