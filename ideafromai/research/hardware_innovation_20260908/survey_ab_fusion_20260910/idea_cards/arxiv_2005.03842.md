# arXiv:2005.03842 · GOBO

- uid/来源: `MAIN-R051`｜arxiv_2005.03842+本地excerpt（`p0_excerpt_batches/batch_01.json`）
- 题名: GOBO: Quantizing Attention-Based NLP Models for Low Latency and Energy Efficient Inference
- 精读深度: 方法级（仅方法摘录窗口）+依据：高斯拟合自动离群比例、少数字典质心（典型8）3b 索引、免微调、硬件侧不展开为 FP、与 Deep Compression/outlier-aware 对比

## 可继承 A
离群保真 + 主体字典量化（免微调）作为「普通压缩」强权限底座；可与 Gustav 供数/ lifting 常量图同权限并列（借入≠X）。

## 强对照 B
需微调的 Q-BERT/Q8BERT；固定 3–5% 离群线性量化；Huffman 展开后再 MAC。

## 可差分 X线索
GOBO 量化本身≠X；只有量化后仍过 AEE 且 Stage B 净服务相对稠密有余量才谈叠层。

## 与 F1–F7 / Stage B 关系
F7/普通压缩对照；Stage B 后可挂量化轴。不抢 Stage B。

## 不可搬用边界
BERT/注意力模型；摘录偏引言+相关工作，质心选择算法细节标注仅方法摘录窗口；勿搬 7×/10×。

## 可复用 idea 点
- 离群保真策略作 lifting 系数/中间量量化的强对照
- 「计算中保持索引、不展开」减少片上展开带宽的思路对照
- 免微调 PTQ 作本地 fixed-BN 后压缩权限边界
- 负结果只停某量化网格，不杀量化家族

## 杀门建议
同精度下不胜普通均匀量化/PoT 强对照，或破 AEE → 停该字典布局。
