# arXiv:2012.09852 · SpAtten

- uid/来源: `MAIN-R073`｜arxiv_2012.09852+本地excerpt（`p0_excerpt_batches/batch_01.json`）
- 题名: SpAtten: Efficient Sparse Attention Architecture with Cascade Token and Head Pruning
- 精读深度: 方法级（仅方法摘录窗口）+依据：Algo2 级联 token/head 剪枝（累计 attention prob / |attention out| top-k）、一旦剪掉后续层永不恢复、 progressive 量化（MSB→按分布平坦度取 LSB）、注意力协处理器分工

## 可继承 A
级联重要性累计 + 永久删除的粗粒度结构剪枝；渐进位宽——可作 F1「按消费者重要性共同删」与 F6 取消列的算法对照（借入≠X）。

## 强对照 B
每层独立重评剪枝；非级联随机 token drop；无累计分的静态稀疏注意力。

## 可差分 X线索
NLP token cascade≠光流源字剪枝 X；差分须用 lifting 改后源活动+双 PED 损失，而非抄 attention prob。

## 与 F1–F7 / Stage B 关系
F1/F2/F7 对照；Stage B 后可挂。不抢 Stage B。

## 不可搬用边界
注意力序列≠r1 残差链；cascade 永久删除对非因果 T10 可能过激；仅方法摘录窗口。

## 可复用 idea 点
- 累计重要性 + top-k 作 F1 共同删字候选排序模板（换损失）
- 「剪后永不回」对照 F6 可取消列的可逆性杀门
- 渐进 MSB/LSB 作端口位宽与误差证书对照
- 协处理器边界：只加速瓶颈段，不吞整网——对照 Stage B 分母范围

## 杀门建议
换成 r1 损失后不胜 HiNM/窄稠密，或级联删坏 AEE → 停该级联布局。
