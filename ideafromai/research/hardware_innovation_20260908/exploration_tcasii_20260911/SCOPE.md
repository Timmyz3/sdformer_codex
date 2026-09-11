# Exploration session — 2026-09-11

**Focal question:** 在事件相机二维光流 SNN-Transformer（Motion C12 / H67 / ep34；AT-LIF 为 \(\{0,\theta\}\)，推理时 \(\theta\) 吸进下一层 \(W\)）上，什么样的**先验完整迁移 + 真增量 X**，能同时满足 TCAS-II 电路短文的新颖性与可测性能，冲击强录用？

**Owner:** 用户（硕士课题决策者）。本会话只产提案，不产发现，不改生产 RTL / `main.tex`。

**Audience:** 迷茫于硬件从何入手的作者；需要可执行的 A/B/X 融合方案，而不是模块动物园。

**Horizon:** 数周内能用 CPU 服务模型 + 定点 AEE 杀门的方案；RTL/PPA 只在过门之后。

## In scope
- patch r1 残差链（source T10 PSN + 双消费者：门与连续 PED）
- 同资源服务、有限端口、完成/ready 依赖、完整常量编译
- 顶会顶刊/arxiv/开源机制的**完整抄全**再差分
- TCAS-II 五页：一个机制、一张因果图、组件级指标

## Out of scope
- 把 AT-LIF 写成不可吸收的连续/int8 payload（官方身份是 \(\{0,\theta\}\) 且推理吸进 \(W\)）
- MX3P 岛当本刊身份
- 模拟 CIM / 空泛存内计算当数字贡献
- Yosys/OpenROAD 当 ASIC PPA
- 组件倍率相乘、整网 FPS
- 把 GrokBot 阈值比较器当标题
- 临床/双用途/绕过伦理

## Constraints
- 真实：AT-LIF 为 \(\{0,\theta\}\)，推理时 \(\theta\) 吸进下一层 \(W\)；valid825 AEE 绝对 ≤1.259、相对强对照 ≤+0.005；完整目标链同资源净服务 ≥15%
- 假定：lifting 家族可保留，停的是布局
- 可谈：是否保留 lifting 当主标题 vs 换主岛（须有更强 A+X）
- 未知：完整链同资源净服务；原生投影 BN 全幅统计的真实费用

## Current observations (not interpretations)
见 `PROBLEM.md`。Codex 正在做 full_chain 数值，尚未闭合服务排程。
