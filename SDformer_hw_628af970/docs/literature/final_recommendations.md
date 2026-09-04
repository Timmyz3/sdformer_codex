# Final Recommendations for SDFormer-based SNN Optical Flow

## Direct answers

### 1. 哪些近年 Transformer 效率优化最适合迁移到 SNN Optical Flow？

最适合的不是“最激进的稀疏 attention”，而是以下三类：

- 结构化窗口裁剪
  - 代表：SparseViT
  - 原因：对 optical flow，空间局部连续性强；对 SNN，窗口级 mask 不会破坏 spike sparsity；对硬件，最容易做 block/tile 调度
- 减少 attention 计算频次并做时间复用
  - 代表：LaViT
  - 原因：事件流跨 timestep 高相关，attention/mask 可以沿 `T` 复用
- 任务相关 token 保留与背景 token 融合
  - 代表：Token Cropr, Zero-TPrune, MCTF, PPT
  - 原因：optical flow 的核心信息集中在运动边界、遮挡区、快速运动区；静态背景更适合 merge 而不是全保留

### 2. 哪些方法在 ANN/LLM/ViT 上有效，但不适合 SNN 或不适合硬件实现？

- Selective Attention, Twilight, Top-Theta
  - 它们对长上下文 LLM 很有效，但仍以 dense attention score 或 KV-cache 访问模型为中心
  - 对 SNN optical flow 的直接价值弱，且动态 top-p / threshold / context pruning 会引入不规则访存和复杂索引
- 原版 ToMe 的全局匹配
  - 算法有效，但全局 pair matching 不够 RTL-friendly
  - 若做局部窗口化 merge 才适合本项目
- 原版 MADTP
  - 多模态 alignment guidance 对单模态事件流并不直接成立
  - 其价值在“任务相关保留”思想，不在原始模块本身

### 3. 哪些方法分别适合做成网络创新点、软硬件协同创新点、加速器架构创新点？

- 网络创新点
  - Temporal attention reuse
  - Motion-aware token selection and fusion
  - Background merge + edge-preserving prune
- 软硬件协同创新点
  - Structured window scheduler with timestep hysteresis
  - Latency-aware pruning controller
  - Block-sparse local attention mask with fixed-radius neighborhood
- 加速器架构创新点
  - Window/block scheduler controller
  - Reuse-aware skip controller across timesteps
  - Binary/shift-friendly low-bit mixer and sparse FFN path

### 4. 最终推荐的 baseline + 3 个插件式创新模块是什么？

推荐 baseline：

- 当前 `SDFormerFlowAdapter` + spike encoder + upstream SDFormerFlow SNN 主干
- 保持输入输出 shape 和训练流程不变，优先在 pre-backbone plug-in 路径验证收益

推荐 3 个主插件：

1. `temporal_attention_reuse`
   - 主打时间连续性复用
2. `activity_window_scheduler`
   - 主打结构化窗口级跳算
3. `graph_token_pruner`
   - 主打事件感知 token 保留

备选第 4 个插件：

- `similarity_token_merger`
  - 适合做“精度恢复”和“背景压缩”增强件

### 5. 这些模块组合后，最有希望形成怎样的论文主线？

最强主线不是“把所有稀疏技巧都堆上去”，而是：

- 利用事件流的时空稀疏性与时间连续性
- 用结构化窗口调度先删大块无效背景
- 用事件/运动引导 token pruning 保留边缘和运动区域
- 对低变化 timestep 复用 attention 或 feature alignment
- 最后从控制器和 RTL 角度证明真实 latency、memory traffic 和规则性收益

一句话概括：

- From dense per-timestep attention to structured, reuse-aware, event-driven sparse scheduling.

## Six concrete SDFormer modification candidates

| 候选改造点 | 替换 SDFormer 的哪一层/哪一部分 | 输入输出 shape 是否变化 | 是否影响时间维 T | 是否影响 spike 编码 / 膜电位更新 | 对 FLOPs / latency / memory 的理论影响 | 对硬件映射的利弊 | 当前状态 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1. `activity_window_scheduler` | spike encoder 后、backbone 前；后续可深入 stage-1/2 窗口 attention 前 | 当前不变，仍是 `[B,T,C,H,W]`；深度集成后可减少 active windows | 通过 hysteresis 让相邻 timestep 共享活跃窗口分布 | 不改 spike 编码，不改神经元方程；只对后续计算做结构化屏蔽 | 若保留窗口比例为 `r_w`，attention/FFN 的有效工作集近似按 `r_w` 缩小，SRAM 读写同步下降 | 优点：规则 window/block，易 tile 化；缺点：极小运动目标可能被整窗误删 | 已实现 |
| 2. `graph_token_pruner` | 现阶段放在 pre-backbone；后续可插到 patch embedding 后或 stage 之间 | 当前不变；深度集成时可把 token length 从 `N` 减到 `r_t N` | 只用局部 temporal context 评分，不改变 `T` 长度 | 不改膜电位，只保留高活动/高变化 token | 若 token 真正进入 attention 前被缩短，则 attention 复杂度可由 `O(N^2)` 变成 `O((r_tN)^2)` | 优点：可转为结构化 top-k；缺点：若做全局 top-k，访存会变不规则 | 已实现 |
| 3. `temporal_attention_reuse` | stage 间或 timestep 间 attention/feature 对齐模块；当前先作为前处理特征复用 | 不变 | 直接作用于 `T`，对低运动 timestep 做 reuse | 对 spike 编码无侵入；若将来深入主干，可扩展为“跳过膜更新/部分更新”策略 | 若复用比例为 `r_u`，attention 重算频次约降到 `1-r_u`；memory access 同步下降 | 优点：控制器简单、硬件友好；缺点：快速运动场景需要可靠的 reuse 失效检测 | 已实现 |
| 4. `similarity_token_merger` | 建议插入低分辨率 stage 前，或窗口调度之后 | 当前实现保持 shape 不变；未来可通过 merge-map 真正压缩 token 序列 | 可把 merge-map 在邻近 timestep 复用，提升时序稳定性 | 不影响 spike 编码；若过度 merge 可能模糊运动边界 | 真正深度集成后，token 数降到 `r_mN`，attention / memory 近似同比下降 | 优点：局部均值/质心 merge 容易硬件化；缺点：全局 pair matching 不适合 RTL | 已实现原型 |
| 5. `block_sparse_attention_masker` | 未来替换 stage 内 attention mask 生成逻辑 | 当前输入不变，额外输出 block mask metadata | 可在多个 timestep 共用同一局部半径 mask | 不直接影响脉冲神经元，仅影响 attention 连接图 | 若每个 window 只看固定邻域，attention 元素数从 `N^2` 降到 `N*k` | 优点：固定半径 mask 极易映射到 RTL；缺点：若引入动态全局连接，复杂度会迅速上升 | 已实现 metadata 原型 |
| 6. `structured_latency_controller` | 作为全局 budget 控制器，放在 preprocessing 或 future stage controller | 不变，输出预算与 mask metadata | 可直接决定 timestep keep ratio 和 per-stage budget | 不改 spike 编码；是调度层而非神经元层 | 不直接省 FLOPs，但能把 token/window/timestep 三种预算与真实 latency 对齐 | 优点：最适合与 RTL controller 协同；缺点：需要 profiling 数据才能校准预算映射 | 已实现 metadata 原型 |

## Recommended implementation order

### Priority-1: 最适合 SNN + Optical Flow + Hardware

- `temporal_attention_reuse`
- `activity_window_scheduler`
- `graph_token_pruner`

原因：

- 都不要求改变当前 backbone 的 tensor contract
- 都能直接利用事件稀疏性或时间连续性
- 都能映射到结构化 mask / skip / reuse 控制

### Priority-2: 精度/效率折中最好

- `similarity_token_merger`

原因：

- 纯 pruning 容易伤边界和小目标
- merge 更适合静态背景压缩与精度恢复

### Priority-3: 最容易形成论文创新点

- `block_sparse_attention_masker`
- `structured_latency_controller`

原因：

- 它们直接连接到软硬件协同与可解释控制器
- 更容易写出 “算法-架构联合” 的论文故事线

## Final recommended research package

### Baseline

- Upstream SDFormerFlow SNN backbone
- Existing spike encoder and modular pre-processing hook

### Plugin package

- `temporal_attention_reuse`
- `activity_window_scheduler`
- `graph_token_pruner`

### Paper-friendly extension

- Add `similarity_token_merger` as accuracy-recovery branch
- Add `structured_latency_controller` to align algorithmic sparsity with real latency

## Why this package is the best fit

它满足三个同时成立的条件：

- 对 event optical flow 有任务先验
  - 运动连续、背景冗余、边缘重要
- 对 SNN 有脉冲先验
  - spike sparse, timestep correlated, local activity meaningful
- 对硬件有实现先验
  - 结构化 window、固定半径 mask、controller 可调度、少全局排序

换句话说，最值得做的不是“把 LLM sparse attention 直接套到事件视觉”，而是把最近高效 Transformer 的思想压缩成：

- structured scheduling
- temporal reuse
- event-aware token retention

这三点正好与 SDFormer-based SNN optical flow 的任务属性、模型属性和硬件属性对齐。
