# arXiv:2503.03256 · BAT

- uid/来源: `MAIN-R224`｜arxiv_2503.03256+本地excerpt（`p0_excerpt_batches/batch_04_residual.json`）
- 题名: BAT: Learning Event-based Optical Flow with Bidirectional Adaptive Temporal Correlation
- 精读深度: 方法级（仅方法摘录窗口）+依据：BTC 双向时间相关（前向多目标帧 + 后向多参考帧）把时密线索变空密；自适应时间采样；SATMA 可变形稀疏注意聚合并抑制不一致运动；体素表示；摘录进入 Method/Event Representation，完整模块公式可能截断

## 可继承 A
双向时间相关 + 空间自适应时序聚合——作「时密→空密」运动线索打包与遮挡/未来帧预测对照（借入≠X）。

## 强对照 B
E-RAFT 单对相关；TMA 仅前向多帧相关；无自适应采样的线性索引邻帧；无后向线索的暖启动未来预测。

## 可差分 X线索
BTC/SATMA ≠ lifting 源字 X；同任务光流旁路，差分不在 DSEC EPE 排行。

## 与 F1–F7 / Stage B 关系
F2/F7 弱相关（双向时间打包、自适应聚合）；旁路基线。不抢 Stage B。

## 不可搬用边界
DSEC 1PE/+39% 等勿写成 same-port%；仅方法摘录窗口；禁止假装已读完整 SATMA 公式/训练超参。

## 可复用 idea 点
- 后向相关处理出画遮挡 + 仅用过去事件预测未来流 的合同能力表
- SATMA「聚一致、抑不一致」作 F7 运动特征融合模板
- 与 TMA/EDCFlow/ResFlow 成簇对照迭代次数–误差曲线
- 负结果只停「BAT 头替换本地」

## 杀门建议
替换后 AEE/服务无增益或迭代费用挤占 Stage B → 保持旁路。
