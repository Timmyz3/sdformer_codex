# arXiv:2607.22790 · Sparsity Tax SIMD vs SIMT

- uid/来源: `ARX-164`｜arxiv_2607.22790+本地excerpt（`p0_excerpt_batches_gap/gap_07.json`）
- 题名: The Sparsity Tax: Weight Sparsity Trade-offs in Event-Driven SIMD and SIMT Neuromorphic Cores
- 精读深度: 方法级（仅方法摘录窗口）+依据：三变体(SIMD/Bitmap Sparse-SIMD/SIMT+4b RLC)；共享指令∥每PE AGU；R7时钟使能位图；NEORV32松耦合；稀疏税面积/功耗/吞吐；完整综合可能截断

## 可继承 A
事件驱动核稀疏税：锁步SIMD vs 位图门控Sparse-SIMD vs 每PE-AGU的SIMT+RLC——在中等稀疏下量化元数据/分歧成本与跳零收益对照（借入≠X）。

## 强对照 B
默认稀疏必赢；仅一种执行模型；无元数据开销的理想跳零；稠密权无门控。

## 可差分 X线索
Sparsity Tax比较≠lifting X；微架构旁路，差分不在面积/功耗点。

## 与 F1–F7 / Stage B 关系
F1/F2相关（权稀疏执行组织、事件更新）。微架构旁路。不抢 Stage B。f_candidates含F1,F2。

## 不可搬用边界
RTL门级功耗≠valid825；仅方法窗；SIMT分歧≠lifting并集同步合同。

## 可复用 idea 点
- 三变体公平对照作稀疏税合同
- R7位图门控作锁步跳读模板
- 每PE AGU+4b RLC作压缩遍历旁证
- 分歧再汇聚作控制开销边
- 负结果只停该稀疏执行挂接

## 杀门建议
中等稀疏下税大于收益或挤占 Stage B → 保持旁路/作负对照。
