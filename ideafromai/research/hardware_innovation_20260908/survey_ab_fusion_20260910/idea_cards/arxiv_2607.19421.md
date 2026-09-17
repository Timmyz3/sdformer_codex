# arXiv:2607.19421 · Opto-ViT-v2 Photonic Fine-Tuning

- uid/来源: `ARX-123`｜arxiv_2607.19421+本地excerpt（`p0_excerpt_batches_gap/gap_07.json`）
- 题名: Opto-ViT-v2: Noise-Resilient On-Chip Fine-Tuning for Photonic Near-Sensor Vision Transformer Accelerators
- 精读深度: 方法级（仅方法摘录窗口）+依据：张量化低秩因子前/反传；压缩激活缓存X·U；五光核前向-反向流水；梯度累积稀疏头；热串扰噪声模型；完整工艺可能截断

## 可继承 A
光子近传感器ViT片上微调：冻结光权重W0+电子低秩因子链 + 仅缓存X·U + 五核前/反交错调度 + 梯度累积稀疏分类头 + 热串扰结构化噪声——光电子共训与稠密反传对照（借入≠X）。

## 强对照 B
全激活反传缓存；稠密头全参微调；独立高斯器件噪声；无流水重用光核。

## 可差分 X线索
Opto-ViT-v2光子微调≠lifting X；光学/训练旁路，勿搬VTAB分当净服务%。

## 与 F1–F7 / Stage B 关系
F7弱相关（稀疏头/噪声鲁棒）。光学旁路。不抢 Stage B。f_candidates含F7。

## 不可搬用边界
VTAB/噪声模拟≠valid825；仅摘录窗；MRR阵列≠数字same-port合同。

## 可复用 idea 点
- 张量化因子链作低存储微调合同
- 缓存X·U+边界重算作激活压缩模板
- 五核B1/B2/B3交错作光核复用旁证
- 暖机+累积梯度top-k头掩码作稀疏头边
- 负结果只停该光子微调挂接

## 杀门建议
热串扰或光学工艺不可迁挤占 Stage B → 保持旁路。
