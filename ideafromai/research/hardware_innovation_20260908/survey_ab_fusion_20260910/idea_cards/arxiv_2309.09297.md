# arXiv:2309.09297 · EOLO

- uid/来源: `ARX-015`｜arxiv_2309.09297+本地excerpt（`p0_excerpt_batches_gap/gap_03.json`；原摘录偏Intro，已用同稿arXiv PDF补齐§III方法窗）
- 题名: Chasing Day and Night: Towards Robust and Efficient All-Day Object Detection Guided by an Event Camera
- 精读深度: 方法级（方法摘录窗+PDF补齐）+依据：SNN事件骨干；ETA时维max/avg注意力；SREF对称融合；随机光流事件合成；E-VOC/E-MSCOCO

## 可继承 A
轻量SNN事件编码 + Event Temporal Attention保边+时序 + Symmetric RGB-Event Fusion（不偏置单模态）——全天多模态对称融合与事件时序注意对照（借入≠X）。

## 强对照 B
非对称RGB主/事件辅融合（RENet类）；纯RGB YOLO；事件转帧丢时序；固定光流合成事件。

## 可差分 X线索
EOLO/ETA/SREF≠lifting X；检测应用旁路，差分不在mAP。

## 与 F1–F7 / Stage B 关系
F1/F7弱相关（表示选择、融合打包）。旁路/感知。不抢 Stage B。f_candidates含F1,F7。

## 不可搬用边界
E-VOC/E-MSCOCO合成事件≠真实事件链；mAP/SNN mJ≠AEE/same-port%；合成偏差须明示。

## 可复用 idea 点
- ETA：时维max与avg门控作边缘保留合同
- SREF对称融合作模态重要性自适应旁证
- 随机光流合成作缺配对数据纪律
- 负结果只停EOLO头/融合替换

## 杀门建议
融合无全天鲁棒增益或挤占 Stage B → 保持旁路。
