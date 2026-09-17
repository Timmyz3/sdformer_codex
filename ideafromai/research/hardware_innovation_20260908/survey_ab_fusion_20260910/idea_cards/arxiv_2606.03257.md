# arXiv:2606.03257 · PSViT Structured Pruning SViT

- uid/来源: `ARX-036`｜arxiv_2606.03257+本地excerpt（`p0_excerpt_batches_gap/gap_06.json`）
- 题名: PSViT: A Methodology for Structurally Pruning Spiking Vision Transformers
- 精读深度: 方法级（仅方法摘录窗口+补§III PDF）+依据：层敏度/内存剖析→均匀通道剪→细粒度块敏度覆盖；单次结构化；SDTv2/ImageNet；无迭代重训依赖；细部比率表可能截断

## 可继承 A
SViT单次结构化剪枝：层敏度剖析 + 均匀通道剪大层 + 细粒度弱敏块加码——可映射脉动/常规阵列的压缩与非结构化彩票对照（借入≠X）。

## 强对照 B
非结构化Sparsespikformer；仅token/时空稀疏无减参；迭代剪枝+多次微调；忽略层敏度的一律剪。

## 可差分 X线索
PSViT结构化剪≠lifting X；压缩旁路，勿搬ImageNet Top-1当净服务%。

## 与 F1–F7 / Stage B 关系
F1相关（结构选择/通道剪）。压缩旁路。不抢 Stage B。f_candidates含F1。

## 不可搬用边界
ImageNet Top-1≠valid825；仅方法窗；通道剪≠lifting源字并集合同。

## 可复用 idea 点
- 层敏度×内存剖析作剪前合同
- 均匀通道剪大层作结构友好模板
- 弱敏块细粒度加码作精度-内存旁证
- 单次无重训作嵌入约束边
- 负结果只停该SViT剪枝挂接

## 杀门建议
精度掉出3%窗或不可迁移他骨干挤占 Stage B → 保持旁路。
