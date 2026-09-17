# arXiv:2609.07544 · Anti-Gravity Thrust-Rate MPC

- uid/来源: `ARX-020`｜arxiv_2609.07544+本地excerpt（p0_excerpt_batches_gap/gap_08.json）+补arxiv HTML方法（§IV–V）
- 题名: Anti-Gravity Walking by a Flying Humanoid Robot via Thrust-Rate Input Whole-Body Model Predictive Control
- 精读深度: 方法级（方法摘录窗口+补HTML推力率WB-MPC）+依据：推力率作输入保接触切换连续；CWC软惩罚；双支撑法向力下界平滑转移；实时MPC；完整硬件验证可能截断

## 可继承 A
飞行人形反重力步行：推力率输入WB-MPC保接触切换推力连续 + 法向力下界负载转移 + CWC软约束——相对直接推力输入致尖峰的机器人控制旁路（借入≠X；离主岛）。

## 强对照 B
直接推力作控制输入；CWC硬约束致求解变慢；无双支撑负载转移；仅地面重力步行。

## 可差分 X线索
推力率MPC≠lifting X；机器人控制旁路，无SNN/稀疏加速接口。

## 与 F1–F7 / Stage B 关系
离主岛旁路。不抢 Stage B。f_candidates空。

## 不可搬用边界
仿真/样机步行指标≠valid825；仅方法窗；与融合RTL无共同分母。

## 可复用 idea 点
- 推力率输入作切换连续合同
- 法向力下界转移作反重力支撑模板
- CWC软惩罚换实时性作求解边
- 负结果：主题离岛，不挂主实验

## 杀门建议
与AB融合无接口 → 不进入Stage B候选池。
