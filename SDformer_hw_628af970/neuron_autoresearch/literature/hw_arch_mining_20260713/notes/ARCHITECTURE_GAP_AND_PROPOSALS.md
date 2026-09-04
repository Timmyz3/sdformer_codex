# 硬件架构缺口诊断与架构级创新提案（2026-07-13）

## 1. 诊断：当前缺的是 Architecture，不是更多定点算子

当前 H67 硬件工作主要是：
- 算法语义的定点化（Motion-XOR, Q7 score, Q1.7 gate）
- 行内核 datapath 优化（SCS-Shiftmax, 精确深度）
- 验证/DC 交接

这些属于 **operator co-design / RTL micro-optimization**。
DATE/ISCA/HPCA 语境下的 **architecture** 需要至少覆盖：核组织、存储层次、调度/NoC、稀疏执行模型、跨层数据流之一，并用 cycle/energy 模型证明。

仓库内已有更接近架构的文档（尚未完全 RTL 化）：
`hw_autoresearch_nts07/docs/47_TTB稀疏异构架构文献映射与实现候选.md`。

## 2. 文献机制（已全文/节选）

| 论文 | 架构机制 | 对 H67 合法迁移 | 禁止直接照搬 |
|---|---|---|---|
| Bishop ISCA25 | TTB + density stratifier + dense/sparse 双核 | TTB descriptor、双路径、负载均衡审计 | ECP 近似剪枝；其 PPA 数字 |
| SpAtten | cascade token/head pruning + progressive quant | cascade **exact hierarchy** issue | 在线 prune 改变语义 |
| FLAT | attention dataflow 减 off-chip | pair-K co-resident fused pipeline | 通用 QKV softmax 假设 |
| Energon | runtime sparse filter + sparse engine | metadata-first + active-K engine | learned weak pair 过滤若改结果 |
| HeatViT | hardware-aware token pruning | density metadata 调度 | 无重训硬删 token |
| NEURAL | hybrid data-event + elastic FIFO + on-the-fly attention | 弹性 FIFO/反压；稀疏与计算解耦 | 全套 QKFormer 嵌入假设 |
| SpikeX 2505.12292 | sparse SNN arch + network-HW co-opt | 稀疏表示与硬件协同 DSE | 网络改写需软件重训 |
| Softermax/I-ViT | base-2 softmax/Shiftmax | 你们已用类似路径 | 不能称发明 Shiftmax |
| LoAS MICRO24 | temporal-inner dataflow | T=2 pair 连续布局 | 权值稀疏 workload 不同 |
| Prosperity HPCA25 | product sparsity exact reuse | exact mask 复用（高门槛） | 未 profile 就上 TCAM |
| Sparseloop | gating/skipping 成本分类 | 论文分账方法 | 不是 RTL |

## 3. 推荐架构创新包（按优先级）

### Arch-A MD-HETE（主推架构贡献）
Motion-Density Heterogeneous Dual-Path：
metadata-first TTB -> stratifier -> Dense core / Sparse core -> shared SCS backend。
相对 Bishop：workload=H67 TTX/Motion-XOR；exact silent 注入；SCS 后端是你们独有。

### Arch-B FUSED Dataflow Memory Hierarchy
三级 bank：temporal-pair 64b / active-entry / class histogram；
流水：META-SCORE-CLASSIFY-SCS-EMIT；双缓冲 window。

### Arch-C Cascade Exact Issue Hierarchy（SpAtten 形、exact 神）
L0 empty 常量注入；L1 K-zero 仅 den；L2 motion-zero 关 XOR；L3 full active。

### Arch-D ETCR（条件）
u=0 temporal score 复用；需 bit-exact delta 参考。

### Arch-E Full-Encoder Skeleton
12-block descriptor 调度 + 93 ATLIF lane cluster + S0-S2 skip 端口。
否则永远是 row 子系统不是架构。

## 4. 证据门槛
- bit-exact vs software
- ordered-trace cycle / p99 / energy proxy 分账
- dense-only vs Arch-C vs Arch-A 同约束对照
- 有库后 DC/SAIF/Formality

## 5. 论文写法
可写：面向 all-binary 光流的 motion-density 异构 + SCS 共享后端协同数据流。
不可写：已完成整芯片；复制 Bishop/SpAtten 的加速比；定点格式=架构创新。
