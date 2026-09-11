# arXiv:2506.03512 · EDCFlow

- uid/来源: `MAIN-R223`｜arxiv_2506.03512+本地excerpt（`p0_excerpt_batches/batch_05.json`）
- 题名: EDCFlow: Exploring Temporally Dense Difference Maps for Event-based Optical Flow Estimation
- 精读深度: 方法级（仅方法摘录窗口）+依据：引言/贡献中 O(TNC) 时密特征差 vs O(TN²C) 时密代价体；§3.1 总览：高分辨率(1/4)差动运动特征 + 低分辨率(1/8)相关 + 注意力多尺度差分层 + 自适应融合；可作 RAFT-like 即插精修；摘录在 Fig.2 总览处截断，§3.2+ 公式细部不全

## 可继承 A
时密相邻帧特征差（warp→多尺度差→注意力聚合）与低分辨率相关代价的互补运动编码，作「降二次搜索、保中间运动」的同任务（事件光流）算法底座——复杂度叙事 O(TNC) 对照（借入≠X）。

## 强对照 B
仅低分辨率全对相关（E-RAFT 式忽略中间运动）；时密多代价体 O(TN²C)（TMA/MultiFlow）；高分辨率全代价体硬扩；无相关仅差动（噪声敏感）或无差动仅相关。

## 可差分 X线索
差动+相关融合≠ lifting/源字物理删字 X；任务同域但主岛在 SNN Transformer+lifting 执行，不在 ANN 光流精修头。

## 与 F1–F7 / Stage B 关系
F7 弱相关（运动特征打包/互补融合）；同任务旁路基线。不抢 Stage B。

## 不可搬用边界
DSEC/MVSEC EPE ≠ 本地 AEE 合同；参数量/FLOPs 降勿写成 same-port%；方法窗截断于 Fig.2，禁止假装已读完整自适应融合公式/训练细节。

## 可复用 idea 点
- 「高分辨便宜差动 + 低分辨稳健相关」互补作多分辨率运动分母模板
- 即插精修模块边界对照「Stage B 后可叠层、不改主分母」纪律
- 差动对边界清晰、相关抗噪 → 对照 F1 删字时保留哪类运动通道
- 与 EventShiftFlow 等非学习/轻量光流旁路成对阅读
- 负结果只停「EDCFlow 头替换本地光流出口」

## 杀门建议
挂到主网后 AEE/服务无增益或挤占 Stage B 档期 → 保持旁路基线，不进 A+B 标题。
