# arXiv:1908.08976 · MASR

- uid/来源: `MAIN-R046`｜arxiv_1908.08976+本地excerpt（`p0_excerpt_batches/batch_01.json`）
- 题名: MASR: A Modular Accelerator for Sparse RNNs
- 精读深度: 方法级（仅方法摘录窗口）+依据：双稀疏（W+激活）RNN、bit-mask 编码相对 CSR/EIE、解耦流水线隔离不规则性、动态负载均衡、蒸馏+ReLU 造隐藏态稀疏、BN 改造保跨层输入稀疏

## 可继承 A
双稀疏 bitmask 供数与「前端解耦不规则、后端不断流」流水线；动态重分激活负载——可作 Gustav/lifting 并集供数不规则性与同k-ID 屏障的工程对照（借入≠X）。

## 强对照 B
仅权稀疏/仅激活稀疏；EIE CSR 指针随 PE 扩展；无动态负载均衡的静态划分。

## 可差分 X线索
ASR RNN bitmask≠标题 X；差分须落到 r1 源∩W 物理字与 T10 消费者并集，而非 RNN 隐藏态 bitmask。

## 与 F1–F7 / Stage B 关系
F5/F7 供数不规则性对照；旁路。不抢 Stage B。

## 不可搬用边界
双向 GRU/ASR 工作负载≠光流残差链；勿搬 14×/15×；仅方法摘录窗口（PE 微架构后半可能截断）。

## 可复用 idea 点
- bitmask 常数元数据开销作稀疏索引对照（相对指针）
- 解耦流水线：不规则发号与部分和计算分离，对照 Stage B 背压口
- 动态负载再平衡作多消费者并集不均的负/正对照
- 蒸馏造稀疏仅作训练先验，不自称硬件 X

## 杀门建议
位掩码迁到源字后元数据/译码吃掉并集收益 → 停该编码布局。
