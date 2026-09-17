# arXiv:2605.07653 · Aq-FireNet

- uid/来源: `ARX-001`｜arxiv_2605.07653+本地excerpt（`p0_excerpt_batches_gap/gap_05.json`）
- 题名: Aquatic Neuromorphic Optical Flow
- 精读深度: 方法级（仅方法摘录窗口）+依据：§II计数编码；CM损失；线性翘曲+各向同性缩放ϕ；可学习阈/漏LIF；Aq-FireNet；ϕ/λ0消融；完整水下集细节可能截断

## 可继承 A
水下事件光流：对比最大化 + 平移⊕各向同性微折射缩放ϕ + 尖峰FireNet变体——流体场景自监督光流与陆地线性翘曲对照（借入≠X）。

## 强对照 B
纯线性CM翘曲；监督ANN光流；无L0噪声项；帧相机水下光流。

## 可差分 X线索
水下神经形态光流≠lifting X；应用旁路，差分不在AEE/FWL%。

## 与 F1–F7 / Stage B 关系
F2弱相关（时间翘曲/事件表示）。应用旁路。不抢 Stage B。f_candidates空。

## 不可搬用边界
水下AEE≠valid825；仅摘录窗；各向同性ϕ假设≠真实湍流。

## 可复用 idea 点
- v⊕ϕ各向同性缩放作流体翘曲合同
- LCM含L0抑扩散事件作噪声模板
- 事件计数图+可学习LIF作高效编码器旁证
- ϕ=0/λ0=0消融成簇 | 负结果只停该水下光流挂接

## 杀门建议
ϕ无增益或水下数据不可复现挤占 Stage B → 保持旁路。
