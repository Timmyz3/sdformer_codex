# arXiv:2211.13726 · IDNet

- uid/来源: `MAIN-R225`｜arxiv_2211.13726+本地excerpt（`p0_excerpt_batches_gap/gap_02.json`）
- 题名: Lightweight Event-based Optical Flow Estimation via Iterative Deblurring
- 精读深度: 方法级（仅方法摘录窗口）+依据：无相关体；事件轨迹/模糊作搜索方向；backbone RNN；ID（同批迭代去模糊）与 TID（时间流式迭代）；运动补偿去模糊；摘录偏 Intro/Related+流水示意，完整网络式可能截断

## 可继承 A
用事件时空轨迹直接迭代去模糊估流、避开 4D 相关体——轻量事件光流前端与低存储迭代对照（借入≠X）。

## 强对照 B
E-RAFT/TMA 式显式相关体+RNN；金字塔粗到细无去模糊；离线大相关存储。

## 可差分 X线索
IDNet/TID≠lifting X；旁路/前端，勿搬 Jetson 延迟当净服务%。

## 与 F1–F7 / Stage B 关系
运动前端旁路；弱 F2（流式事件迭代）。不抢 Stage B。

## 不可搬用边界
DSEC SOTA/参数量≠valid825；仅摘录窗；禁止虚报已读完整网络超参表。

## 可复用 idea 点
- 无相关体去模糊迭代作存储分母合同
- TID 流式 8ms 级作在线服务旁证
- 与 E-RAFT/TMA 成簇对照相关体费用
- 负结果只停 IDNet 头替换

## 杀门建议
前端替换无 AEE/服务增益或挤占 Stage B → 保持旁路。
