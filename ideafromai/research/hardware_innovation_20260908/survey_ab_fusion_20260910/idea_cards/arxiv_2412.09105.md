# arXiv:2412.09105 · ResFlow / ResHTR

- uid/来源: `MAIN-R222`｜arxiv_2412.09105+本地excerpt（`p0_excerpt_batches/batch_04_residual.json`）
- 题名: ResFlow: Fine-tuning Residual Optical Flow for Event-based High Temporal Resolution Motion Estimation
- 精读深度: 方法级（仅方法摘录窗口）+依据：两阶段残差范式——全局阶段聚合成准 LTR 流，残差阶段共享 residual refiner 预测 HTR 残差；LTR GT 监督靠速度变换+区域噪声模拟残差；Fig.1 共享 cost volume/运动编码；摘录止于 §III 事件体素/切分，§IV 训练策略公式不全

## 可继承 A
「低频稳健参考 + 高频残差精修」与共享相关体——作 HTR 运动估计的算法底座/复杂度分母对照（借入≠X）。

## 强对照 B
纯累积 HTR（误差滚雪球）；仅自监督对比最大化；从零预测中间流依赖全局全对相关；无噪声对齐的 LTR→HTR 硬监督。

## 可差分 X线索
残差光流头≠ lifting 源字 X；同任务旁路，主岛仍在 SNN/执行合同。

## 与 F1–F7 / Stage B 关系
F7 弱相关（运动特征/残差打包）；旁路基线。不抢 Stage B。

## 不可搬用边界
10Hz 监督→150Hz 推理与 EPE/warp loss 勿写成 same-port%；方法窗截断于 prelim；禁止假装已读完整噪声课程公式。

## 可复用 idea 点
- 共享 cost volume 降双阶段参数作「Stage B 后可叠头」纪律
- 区域噪声对齐 LTR/HTR 残差分布 → 合同域迁移模板
- 与 EDCFlow/BAT/TMA 成对作光流旁路簇
- 负结果只停「ResFlow 头替换本地出口」

## 杀门建议
挂主网后 AEE/服务无增益或挤占 Stage B → 保持旁路，不进标题。
