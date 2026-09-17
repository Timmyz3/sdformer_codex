# arXiv:2510.24231 · Microsaccade Event Benchmark

- uid/来源: `ARX-003`｜arxiv_2510.24231+本地excerpt（`p0_excerpt_batches_gap/gap_05.json`）
- 题名: Benchmarking Microsaccade Recognition with Event Cameras: A Novel Dataset and Evaluation
- 精读深度: 方法级（仅方法摘录窗口）+依据：§3 Blender→v2e；角类+时长/计数重叠重采样；§4 Spiking-VGGFlow双头（分类+Farneback流监督）；推理卸流头；完整真值标签可能截断

## 可继承 A
合成微眼跳事件集 + 训练期光流辅助头逼模型学运动而非事件计数——微眼跳事件基准与运动正则对照（借入≠X）。

## 强对照 B
仅按事件计数分类；无重叠重采样的时长泄漏；帧相机微眼跳；推理期仍跑流头。

## 可差分 X线索
微眼跳基准≠lifting X；数据集/应用旁路，勿搬准确率当净服务%。

## 与 F1–F7 / Stage B 关系
F2弱相关（时间运动表示）。数据/应用旁路。不抢 Stage B。f_candidates空。

## 不可搬用边界
合成/EV-Eye检测率≠valid825；仅摘录窗；Farneback监督≠真实流GT。

## 可复用 idea 点
- 时长与事件计数重叠重采样作防泄漏合同
- 训练期Lclass+λLflow、推理卸头作零成本正则模板
- Blender角类+v2e作微运动事件供数旁证
- 与纯Spiking-VGG成簇 | 负结果只停该流正则挂接

## 杀门建议
流正则无增益或数据集偏差挤占 Stage B → 保持旁路。
