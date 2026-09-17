# arXiv:2607.11986 · SpikeDS Dual Sparsity Spikformer

- uid/来源: `ARX-027`｜arxiv_2607.11986+本地excerpt（`p0_excerpt_batches_gap/gap_07.json`）
- 题名: SpikeDS: Dual Sparsity Spikformer for Perineural Invasion Prediction in 3D MRI
- 精读深度: 方法级（仅方法摘录窗口）+依据：SPS→层次SpikeDSBlock；DSSA=发放率窗剪枝+W-EMSA专家路由；非对称CW-SSA；ρ阶梯剪枝；能量分解AC/SOP；完整临床可能截断

## 可继承 A
双稀疏尖峰Transformer：发放率引导窗剪枝∥W-EMSA头专家门控 + 非对称CW-SSA(Q自活跃/KV全窗)——激活稀疏与空间稀疏复合对照（借入≠X）。

## 强对照 B
全窗Spikformer；对称QKV仅自活跃；无专家路由的W-SSA；全阶段同等剪枝。

## 可差分 X线索
SpikeDS医学PNI≠lifting X；应用/注意力旁路，勿搬AUC/mJ当净服务%。

## 与 F1–F7 / Stage B 关系
F1/F2/F7相关（结构剪枝、时间窗打包、事件注意）。应用旁路。不抢 Stage B。f_candidates含F1,F2,F7。

## 不可搬用边界
AUC/理论能量≠valid825；仅摘录窗；ρ窗剪≠lifting源字删字合同。

## 可复用 idea 点
- 发放率top-k窗剪作空间稀疏合同
- W-EMSA二值路由作头专家模板
- 非对称CW-SSA保剪枝窗KV作全局旁证
- Stage0禁剪+后级ρ作渐进边
- 负结果只停该双稀疏注意挂接

## 杀门建议
任务无关或窗剪伤早期特征挤占 Stage B → 保持旁路。
