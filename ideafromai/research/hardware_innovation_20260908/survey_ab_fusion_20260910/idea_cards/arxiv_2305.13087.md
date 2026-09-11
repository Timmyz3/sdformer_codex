# arXiv:2305.13087 · VD56G3_onsensor_OF

- uid/来源: `MUSHA-OF005`｜arxiv_2305.13087+本地excerpt（`p0_excerpt_batches_gap/gap_02.json`）
- 题名: On-sensor VD56G3 ASIC optical flow (arXiv:2305.13087)
- 精读深度: 方法级（仅方法摘录窗口）+依据：片上 FAST+BRIEF；每 16×16 限描述子数；最多 2048 向量；Hamming 最佳/次佳；MIPI 传图+流；ROI 三分；摘录偏传感器/流水+数据集，完整模拟前端可能截断

## 可继承 A
传感器内 FAST/BRIEF 稀疏特征匹配出运动向量——片上光流前端与稀疏向量服务对照（借入≠X）。

## 强对照 B
片外稠密光流；全像素相关；无描述子配额的失控特征爆炸。

## 可差分 X线索
VD56G3 ASIC≠lifting X；旁路传感，勿搬 fps/向量数。

## 与 F1–F7 / Stage B 关系
传感前端旁路。不抢 Stage B。

## 不可搬用边界
帧率/向量数≠valid825；帧相机 OF≠事件链；仅摘录窗。

## 可复用 idea 点
- 描述子配额作稀疏前端合同
- best/2nd Hamming 比作可信门控旁证
- 与事件光流卡分簇勿混传感器
- 负结果只停「片上 OF 替换数字出口」

## 杀门建议
替换无增益或挤占 Stage B → 保持旁路。
