# arXiv:2607.19248 · SparHiXcel-v2 Column-Wise Sparse CNN

- uid/来源: `ARX-219`｜arxiv_2607.19248+本地excerpt（`p0_excerpt_batches_gap/gap_07.json`）
- 题名: A Flexible Sparsity-Aware FPGA Accelerator with Column-Wise Compression for Efficient CNN Inference
- 精读深度: 方法级（仅方法摘录窗口+补arxiv HTML §III/VI–VIII）+依据：列向核压缩；可变列PE分配；V-Line/V-Node归约；GA通道-滤波排序；SIPR四相剪枝/复活；原摘录偏结果已补方法

## 可继承 A
FPGA稀疏CNN：列向核压缩(行内左移消零列) + 可变列宽PE映射 + V-Line分布式归约 + GA排序∥SIPR硬件感知剪枝/复活——非结构灵活与结构效率之间的桥接对照（借入≠X）。

## 强对照 B
刚性通道/块结构稀疏；无压缩的非结构重控开销；无排序/复活的朴素剪枝；固定列分配。

## 可差分 X线索
SparHiXcel压缩≠lifting X；FPGA CNN旁路，勿搬TOPS/GOP/s/W当净服务%。

## 与 F1–F7 / Stage B 关系
F1相关（稀疏图案与映射共设计）。加速器旁路。不抢 Stage B。f_candidates含F1。

## 不可搬用边界
ImageNet GOP/s≠valid825；补HTML≠原gap窗全文；V-Line争用≠lifting并集同步。

## 可复用 idea 点
- 列向压缩+原列索引作消零列合同
- 可变列PE+MUX-T作灵活映射模板
- V-Node三模归约作通道累加旁证
- GA排序+SIPR复活作利用率边
- 负结果只停该列压缩挂接

## 杀门建议
1×1层V-Line争用或剪枝掉点挤占 Stage B → 保持旁路。
