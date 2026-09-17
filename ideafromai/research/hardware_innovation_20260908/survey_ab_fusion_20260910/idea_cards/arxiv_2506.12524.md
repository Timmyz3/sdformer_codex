# arXiv:2506.12524 · Inference-Time Gaze Refinement

- uid/来源: `ARX-005`｜arxiv_2506.12524+本地excerpt（`p0_excerpt_batches_gap/gap_04.json`）
- 题名: Inference-Time Gaze Refinement for Micro-Expression Recognition: Enhancing Event-Based Eye Tracking with Motion-Aware Post-Processing
- 精读深度: 方法级（仅方法摘录窗口）+依据：Algo1 运动感知自适应中值滤；Algo2 局部光流对齐纠偏；模型无关推理后处理；Jitter 速度度量（KL+谱熵）；完整推导§5可能截断

## 可继承 A
事件瞳孔轨迹推理期：运动感知中值抑眨眼尖峰 + 局部光流邻域纠偏——模型无关后处理稳定合同对照（借入≠X）。

## 强对照 B
仅重训骨干抑抖动；固定核中值无视运动方差；无光流的位置回归；只报点精度无视时间连续。

## 可差分 X线索
Gaze 后处理≠lifting X；应用旁路，差分不在瞳孔像素误差%。

## 与 F1–F7 / Stage B 关系
F2/F7相关（时间连续、后处理打包）。旁路。不抢 Stage B。f_candidates含F2,F7。

## 不可搬用边界
瞳孔/微表情指标≠valid825；仅摘录窗；后处理延迟勿写 same-port 主路径%。

## 可复用 idea 点
- 自适应核中值按局部运动方差作连续轨迹合同
- 局部光流一致性检查作偏移纠偏旁证
- 模型无关挂接作不重训稳定模板
- Jitter（KL速度分布+谱熵）作时间连续度量 | 负结果只停该后处理挂接

## 杀门建议
后处理无增益或延迟挤占 Stage B → 保持旁路。
