# 硬件加速器设计参考论文速查 (2026-05)

快速链接与要点（与 SDformerFlow 四引擎设计强相关）。

## 最高优先级（立即精读 + 引用）

1. Bishop ISCA'25 (arXiv:2505.12281)
   - TTB (Token-Time Bundle) + 异构 core + ECP
   - 直接复制其 TTB scheduler 思路到我们的 Sparse MAC + window 调度

2. FireFly-T (IEEE TC 2026, arXiv ~2505.12771)
   - Dual Sparse + Binary engine, AND-PopCount for spiking attn
   - 我们的 Binary Engine (SC popcount + Shiftmax) 的最强对标，Figure 要并排画 microarch

3. Spiking Transformer 3D (ICCAD'24, arXiv:2411.07397)
   - 3D memory-on-logic 解决大中间激活 (decoder stripe 问题)
   - 权重复用策略

4. Reconfigurable Spiking Transformer Accelerator (arXiv:2503.19643)
   - Parallel timestep + 低功耗 spiking ViT 设计

5. ASNA-Flow (IEEE TVLSI 2025)
   - Event-based async optical flow 加速器，7.9mW / 104FPS
   - Event Scatter + voxel scatter-add 的直接模板

## 次优先（补充 sparsity / memory / FPGA 原型）

- Prosperity HPCA'25 (arXiv:2503.03379) : product sparsity
- Phi ISCA'25 (arXiv:2505.10909) : pattern-based hierarchical sparsity
- SENECA 系列 (2023) : 3-level mem, event-driven processing
- SpinalFlow ISCA'20 : SNN 经典数据流
- SeaSNN PeerJ 2025 : FPGA spiking attention 实现 (原型先走这条)
- FireFly 系列 (TVLSI'23 + v2 + S) : FPGA overlay 参考

完整清单 + 引用建议见：
docs/literature/hardware_accelerators.md

建议在 DATE/ISCA 投稿的 Hardware section 至少精读前 5 篇并做详细对比表。