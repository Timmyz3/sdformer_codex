# arXiv:2606.13354 · SupraSNN Synapse-Level Parallelism

- uid/来源: `ARX-170`｜arxiv_2606.13354+本地excerpt（`p0_excerpt_batches_gap/gap_06.json`）
- 题名: SupraSNN: Exploiting Synapse-Level Parallelism in Spiking Neural Network Accelerators through Co-Optimized Mapping and Scheduling
- 精读深度: 方法级（仅方法摘录窗口）+依据：§4 SPU并行+路由位串；MC Tree组播；ME Tree无缓冲归约；中心Neuron Unit LIF；超标量类比；映射/调度共优化；完整评测可能截断

## 可继承 A
突触级并行SNN加速：多SPU + 路由位串导引MC树组播 + ME树同步归约部分和 + 中心膜态单元——稀疏连接下平衡并行与归约对照（借入≠X）。

## 强对照 B
ODIN时分全交叉棒；Spiker+神经元全并行吃满片上权；仅互联优化的SpiNeMap；突触/神经元分核却部分和DRAM瓶颈。

## 可差分 X线索
SupraSNN映射调度≠lifting X；加速器旁路，差分不在吞吐/能效点。

## 与 F1–F7 / Stage B 关系
F5/F7相关（突触表并行驻留、事件组播打包）。加速器旁路。不抢 Stage B。f_candidates空。

## 不可搬用边界
仿真/加速指标≠valid825；仅方法窗；中心Neuron Unit≠分布式膜态合同。

## 可复用 idea 点
- 路由位串+MC树作按需组播合同
- ME树无缓冲同步归约作部分和模板
- SPU∥中心LIF作超标量类比旁证
- 映射/调度共优化作负载均衡边
- 负结果只停该突触并行挂接

## 杀门建议
归约树成为瓶颈或位串存储税过大挤占 Stage B → 保持旁路。
