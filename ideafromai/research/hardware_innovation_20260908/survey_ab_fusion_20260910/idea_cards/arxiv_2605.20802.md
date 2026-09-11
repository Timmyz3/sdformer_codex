# arXiv:2605.20802 · ELSA (SNN 2026)

- uid/来源: `MAIN-R013`｜arxiv_2605.20802+本地excerpt（`p0_excerpt_batches/batch_03.json`）
- 题名: ELSA: An ELastic SNN Inference Architecture for Efficient Neuromorphic Computing
- 精读深度: 方法级（仅方法摘录窗口）+依据：弹性 SNN（片上存权+膜）；spine/token 级细粒度流水；对比表含 Prosperity/Phi 等；Gustavson 依赖/mini-batch 线索；摘录偏面积拆解与对比表，微架构细部可能截断

## 可继承 A
弹性推理（权/膜驻留片上）+ spine/token 级流水提高弹性加速器面积效率——与 F5 生存期/驻留及 Gustav 供数叙事同族对照（借入≠X）。

## 强对照 B
非弹性（权流式、膜换出）高面积效率但欠灵活；无 spine/token 流水的弹性基线。

## 可差分 X线索
ELSA 弹性流水≠ lifting 因子对齐打包 X（F7）；Stage B 先用未对齐部件作分母。

## 与 F1–F7 / Stage B 关系
F5/F7 相关；Stage B 后可挂。不抢 Stage B。

## 不可搬用边界
对比表 TOPS/W 勿搬；弹性定义以摘录为准；全文 382k 字仅窗口；仅方法摘录窗口。

## 可复用 idea 点
- spine/token 流水作细粒度弹性调度模板
- 片上权+膜驻留↔有限 RF 周转（F5）
- 与 Prosperity/Phi 同表阅读定位
- mini-batch/Gustavson 依赖线索待补全文核

## 杀门建议
对齐/弹性改造后物理服务不优于同 Gustav+ordinary 序 → 停该弹性挂载布局。
