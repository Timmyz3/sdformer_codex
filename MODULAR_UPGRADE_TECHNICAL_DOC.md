# SDformerFlow 模块化改进技术文档

## 1. 文档目的

本文档整理当前已经在仓库中落地的模块化改进工作，目标是给后续实验、论文撰写、软硬件协同设计和深层 attention 替换提供统一技术说明。

本文档聚焦三个问题：

1. 当前到底新增了哪些可拔插模块。
2. 这些模块接在软件栈的什么位置，输入输出契约是什么。
3. 这些模块如何继续演进到真正的 block-level attention 优化和硬件 accelerator 映射。

---

## 2. 工作概览

本轮工作没有直接重写 upstream `SDformerFlow` 的内部 attention，而是先完成了一套可插拔、可消融、可向硬件映射扩展的小模块框架。

当前已经完成的内容包括：

- 新增一套 `plug_in_modules` 串联式预处理管线。
- 新增 4 个可拔插小模块：
  - `temporal_shift`
  - `timestep_budget`
  - `window_topk`
  - `head_group`
- 将模块输出的稀疏元数据统一为：
  - `timestep_mask`
  - `token_mask`
  - `window_mask`
  - `head_mask`
- 将 profiler 从原先只看 `token_keep_ratio` 和 `active_timestep_ratio`，扩展到同时统计：
  - `active_window_ratio`
  - `head_keep_ratio`
- 新增模块示例配置 `variant_modular.yaml`。
- 新增模块库说明文档 `MODULE_ZOO.md`。

这些工作已经形成了后续做论文涨点的基础骨架，但还没有深入替换 upstream encoder block 内部的真实 attention 计算图。

---

## 3. 当前基线与问题定位

### 3.1 当前真实基线

当前仓库默认模型仍然是 upstream `MS_SpikingformerFlowNet_en4`，由本地 adapter 包装后调用。

关键代码位置：

- 模型 adapter: `src/models/sdformer/backbone.py`
- upstream attention 实现: `third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py`

### 3.2 当前仓库改进前的主要问题

在本轮工作前，仓库已经具备：

- spike encoder
- timestep 早退
- token pruning
- RMSNorm 替换

但这些能力主要停留在 wrapper 层，存在两个明显问题：

1. 稀疏优化更多是在输入侧做“置零”，而不是在 block 内部做真正的跳过执行。
2. attention registry 还是描述层，没有真正替换 upstream attention kernel。

因此，单纯再加 variant 配置项意义不大，必须先把“模块能力”和“稀疏元数据契约”抽象出来。

---

## 4. 本轮新增的软件结构

### 4.1 入口位置

本轮新增的模块执行入口位于：

- `src/models/sdformer/backbone.py`

关键变化：

- `SDFormerFlowAdapter` 在初始化时读取 `model.plug_in_modules`
- 每个模块按顺序串联执行
- 模块既可以改写 `tensor`，也可以输出 mask 元数据

### 4.2 执行顺序

当前数据流如下：

```text
event_voxel
  -> polarity split
  -> spike_encoder
  -> plug_in_modules[0]
  -> plug_in_modules[1]
  -> ...
  -> normalize_nonzero
  -> upstream SDformerFlow backbone
```

### 4.3 当前 hook 的张量契约

模块 hook 当前统一处理：

- 输入张量：`[B, T, C, H, W]`
- 输出张量：`[B, T, C, H, W]`

模块可选输出以下附加字段：

- `tensor`
- `timestep_mask`
- `token_mask`
- `window_mask`
- `head_mask`

其中：

- `tensor` 表示模块处理后的新特征
- 其余字段用于 profiler、后续硬件导出、后续 block-level skip 控制

---

## 5. 新增模块说明

### 5.1 `temporal_shift`

文件：

- `src/models/modules/token_mixer/temporal_shift.py`

设计动机：

- 参考 TokShift 这类时序 shift 思路
- 用零参数方式引入轻量 temporal mixing

核心思想：

- 沿时间维把部分通道左移或右移
- 不引入额外权重矩阵
- 保留事件数据的时序结构

输入输出：

- 输入：`[B, T, C, H, W]`
- 输出：`[B, T, C, H, W]`

适用价值：

- 适合作为最便宜的时序增强模块
- 对硬件友好，因为更接近地址重排，不是重 MAC

当前边界：

- 这是外层 feature mixer，不是 block 内 attention temporal mixer

---

### 5.2 `timestep_budget`

文件：

- `src/models/modules/sparse_ops/timestep_budget.py`

设计动机：

- 将原先写死在 adapter 里的 `adaptive_t` 提升为显式可插拔模块
- 为后续 block-level timestep skip 做统一接口

核心思想：

- 根据每个 timestep 的活动强度打分
- 支持 threshold 或 top-k budget
- 至少保留 `min_keep` 个 timestep，避免整段被裁空

输入输出：

- 输入：`[B, T, C, H, W]`
- 输出：
  - `tensor`
  - `timestep_mask`

适用价值：

- 适合作为时间维稀疏的统一入口
- 后续可直接映射到硬件控制器中的 timestep-level skip

当前边界：

- 当前仍然是通过 mask 后乘法置零实现
- 还没有真正下沉到 upstream block 的执行控制

---

### 5.3 `window_topk`

文件：

- `src/models/modules/sparse_ops/window_pruning.py`

设计动机：

- 参考 HeatViT 这类结构化窗口裁剪方法
- 比 unstructured token pruning 更适合硬件调度

核心思想：

- 先按局部窗口聚合活动强度
- 每个 timestep 选择 top-k active windows
- 扩展为 `window_mask` 与对应的 `token_mask`

输入输出：

- 输入：`[B, T, C, H, W]`
- 输出：
  - `tensor`
  - `window_mask`
  - `token_mask`

适用价值：

- 比直接 token top-k 更适合作为片上 window scheduler 的上游信号
- 适合配合 local attention / local SRAM tile 调度

当前边界：

- 当前 window skip 还没有进入 upstream attention 的 window partition 流程
- 还只是对输入特征置零

---

### 5.4 `head_group`

文件：

- `src/models/modules/sparse_ops/head_pruning.py`

设计动机：

- 参考 SpAtten 这类 head/token 联合稀疏思路
- 当前先用 channel group 近似 attention head group

核心思想：

- 将通道切分成多个 group
- 对每个 group 做活跃度打分
- 保留 top-k group，并输出 `head_mask`

输入输出：

- 输入：`[B, T, C, H, W]`
- 输出：
  - `tensor`
  - `head_mask`

适用价值：

- 为后续真正的 head-level skip 提供统一元数据格式
- 非常适合作为未来 controller metadata 的一部分

当前边界：

- 由于当前 hook 还在 backbone 外部，这里剪的是 channel group，不是真正的 multi-head attention head
- 真正价值要在未来深入到 upstream block 后才能完全发挥

---

### 5.5 `structured_token`

文件：

- `src/models/modules/sparse_ops/token_pruning.py`

本轮补充点：

- 原有模块已存在
- 现在统一补充输出 `token_mask`

意义：

- 统一了 profiler、导出、硬件接口中的 mask 命名

---

## 6. 配置方式

### 6.1 新增配置字段

当前推荐用 `model.plug_in_modules` 显式定义模块链：

```yaml
model:
  plug_in_modules:
    - kind: token_mixer
      name: temporal_shift
      shift_div: 2
    - kind: sparse_ops
      name: timestep_budget
      threshold: 0.02
    - kind: sparse_ops
      name: window_topk
      keep_ratio: 0.75
      window_size: [8, 8]
    - kind: sparse_ops
      name: structured_token
      keep_ratio: 0.75
```

### 6.2 示例 variant

已新增示例配置：

- `configs/model_variants/variant_modular.yaml`

作用：

- 给后续 ablation 提供统一入口
- 便于逐模块启停和复现实验

---

## 7. Profiler 扩展

文件：

- `src/utils/profiler.py`

本轮改动后，profiler 除了原有信息外，还会统计：

- `active_timestep_ratio`
- `active_window_ratio`
- `token_keep_ratio`
- `head_keep_ratio`

意义：

- 为后续硬件建模提供更接近真实执行图的稀疏统计
- 特别是 `window_mask` 和 `head_mask`，是从“只看 FLOPs proxy”走向“能对接 controller 调度”的关键一步

当前边界：

- 仍然不是 cycle-accurate model
- 还没有统计真实 SRAM 访问和 stall

---

## 8. 与硬件协同设计的对应关系

当前新增模块并不是单纯的软件技巧，它们直接对应未来 accelerator 的控制元数据。

建议的映射关系如下：

| 软件模块 | 软件输出 | 硬件对应模块 | 硬件作用 |
| --- | --- | --- | --- |
| `timestep_budget` | `timestep_mask` | controller | timestep-level skip |
| `window_topk` | `window_mask` | window scheduler | active window dispatch |
| `structured_token` | `token_mask` | sparse datapath / token scheduler | token-level gating |
| `head_group` | `head_mask` | head selector | head-group skip |
| `temporal_shift` | transformed tensor | address remap / light data mover | 低成本时序混合 |

这也是为什么本轮优先做的是“小模块 + 统一 mask 契约”，而不是直接把 attention 改成一大坨难以拆解的变体。

---

## 9. 当前工作边界

本轮工作已经完成的是“模块化基础设施”，不是完整论文实现。

已经完成：

- 可插拔模块框架
- 4 个可拔插模块
- profiler 扩展
- 示例 variant
- 模块与硬件接口语义统一

尚未完成：

- upstream `Spiking_SwinTransformerBlock3D` 内部的真实 block-level hook
- 真正的 attention head 级别剪枝
- window partition 之前的显式 window skip
- cycle-accurate 或近 cycle-accurate 硬件性能模型
- 基于这些 mask 的 RTL 控制流实现

因此，当前版本更适合作为：

- 消融实验底座
- 模块库原型
- 软硬件协同接口规范

而不是最终论文结果版本。

---

## 10. 推荐的下一步实施顺序

建议后续按以下顺序推进：

1. 用 `variant_modular` 做外层模块消融，确认哪些模块组合有收益。
2. 将 `window_topk` 和 `head_group` 下沉到 upstream `Spiking_SwinTransformerBlock3D`。
3. 将 `timestep_mask` 从“张量置零”升级为“block-level skip control”。
4. 在 profiler 中增加 `metadata_bytes`、`estimated_qk_ops`、`active_head_groups` 等统计。
5. 更新 `hw/docs/interfaces.md` 与 `hw/docs/perf_model.md` 的实现细节。
6. 再进入真正的 `attention_unit.v` 实现。

---

## 11. 对论文写作的价值

本轮工作为后续论文至少提供了三类可以清晰展开的内容：

1. 模块化事件时空稀疏增强框架
2. 统一的多级稀疏元数据契约
3. 面向 accelerator 的软件先行抽象

如果后续完成 block-level attention hook 和硬件映射，这套工作可以自然扩展成：

- 模型贡献
- 系统贡献
- 硬件贡献

三者统一的一条技术线。

---

## 12. 相关文档索引

当前建议结合以下文档一起阅读：

- `PAPER_CO_DESIGN_PROPOSAL.md`
- `MODULE_ZOO.md`
- `hw/docs/arch.md`
- `hw/docs/interfaces.md`
- `hw/docs/perf_model.md`

它们分别负责：

- 论文导向总体方案
- 小模块库说明
- 硬件架构说明
- 接口契约说明
- 性能建模说明

---

## 13. 结论

本轮工作完成的核心价值，不是直接把 SDformerFlow 改成了最终版论文模型，而是把“模块化研究、稀疏元数据、硬件映射语义”三件事统一到了同一套工程接口里。

这意味着后续你继续加涨点时，不需要每次都重新改整条链路，而是可以围绕以下统一框架推进：

- 加模块
- 做消融
- 观察 mask 统计
- 映射到硬件调度
- 再决定是否深入到 attention block 内核

这会明显降低后续研究迭代成本。
