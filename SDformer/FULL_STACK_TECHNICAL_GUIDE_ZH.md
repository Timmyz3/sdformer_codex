# SDformerFlow 全栈技术文档（中文）

## 1. 文档定位

这是一份面向当前整个工程目录的中文总文档，目标是把我已经完成的工作、当前仓库的运行逻辑、注意力与 token 稀疏改进方案、如何跑通原版与插件版模型、以及后续研究推进方式全部放到同一份文档中。

如果你只读一份文档，优先读这份。

---

## 2. 当前工程到底是什么

当前仓库并不是“完全重写的 SDformerFlow”，而是：

1. 保留 upstream 原版 `SDformerFlow`
2. 在外面包一层本地 adapter / trainer / dataset wrapper / profiler / hardware docs
3. 在 backbone 前增加可插拔模块链
4. 为后续 block-level attention 改造和硬件 accelerator 映射预留统一接口

因此，整个工程可以理解成：

```text
upstream 原版模型
  + 本地统一训练评估入口
  + 本地数据集封装
  + 本地插件模块系统
  + 本地 profiler / quant / RTL 文档
```

---

## 3. 整套运行逻辑

## 3.1 两条运行链路

### 链路 A：upstream 原版链路

作用：

- 验证原版论文代码是否可复现

入口：

- `third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py`
- `third_party/SDformerFlow/eval_DSEC_flow_SNN.py`
- `third_party/SDformerFlow/eval_MV_flow_SNN.py`

特点：

- 直接用 upstream 的配置、数据加载、loss、metrics、MLflow
- 最接近作者原始工程

### 链路 B：本地重构链路

作用：

- 验证本地 baseline
- 跑本地插件模块
- 形成统一消融和硬件映射入口

入口：

- `python -m src.trainers.train`
- `python -m src.trainers.eval`
- `python -m src.utils.profiler`

核心文件：

- `src/models/sdformer/backbone.py`
- `src/models/registry.py`
- `src/models/sdformer/layers.py`

---

## 3.2 本地链路的真实数据流

本地链路的逻辑是：

```text
YAML 配置
  -> build_dataset()
  -> DSEC/MVSEC wrapper
  -> build_model()
  -> SDFormerFlowAdapter
  -> 输入预处理与插件模块
  -> upstream SDformerFlow backbone
  -> flow_pred / flow_list
  -> loss / metrics
  -> checkpoint / summary / profiler / hw export
```

这意味着：

- 模型主干仍然主要来自 upstream
- 本地价值主要在于统一接口和插件实验能力

---

## 4. 代码目录功能说明

### 根目录

- `README.md`
- `TECHNICAL_DOCUMENTATION.md`
- `PAPER_CO_DESIGN_PROPOSAL.md`
- `MODULE_ZOO.md`
- `MODULAR_UPGRADE_TECHNICAL_DOC.md`
- `RUNBOOK_AND_RESEARCH_PLAN_2026.md`
- `FULL_STACK_TECHNICAL_GUIDE_ZH.md`

### 配置

- `configs/sdformer_baseline.yaml`
- `configs/model_variants/variant_a.yaml`
- `configs/model_variants/variant_b.yaml`
- `configs/model_variants/variant_c.yaml`
- `configs/model_variants/variant_modular.yaml`
- `configs/experiment_schemes/*.yaml`

### 软件主体

- `src/datasets/`
- `src/models/`
- `src/trainers/`
- `src/utils/`

### upstream 原版

- `third_party/SDformerFlow/`

### 硬件侧

- `hw/docs/`
- `hw/rtl/`
- `tools/`

---

## 5. 当前本地插件系统怎么工作的

当前插件系统不是插在 upstream attention block 内部，而是插在：

- `spike_encoder` 之后
- `upstream backbone` 之前

即当前 hook 位置是：

```text
event_voxel
  -> polarity split
  -> spike_encoder
  -> plug_in_modules[0]
  -> plug_in_modules[1]
  -> ...
  -> normalize_nonzero
  -> upstream backbone
```

统一张量契约：

- 输入：`[B, T, C, H, W]`
- 输出：`[B, T, C, H, W]`

统一元数据契约：

- `timestep_mask`
- `token_mask`
- `window_mask`
- `head_mask`

当前价值：

- 方便做可拔插消融
- 方便向硬件控制器语义映射

当前限制：

- 还不是真正的 block-level skip
- 还没有直接替换 upstream 内部 attention kernel

---

## 6. 当前已经落地的插件模块

### `temporal_shift`

位置：

- `src/models/modules/token_mixer/temporal_shift.py`

作用：

- 零参数时间混合
- 用通道沿时间维左移/右移的方式增强时序信息

### `timestep_budget`

位置：

- `src/models/modules/sparse_ops/timestep_budget.py`

作用：

- 用阈值或 top-k 选择有效 timestep
- 输出 `timestep_mask`

### `window_topk`

位置：

- `src/models/modules/sparse_ops/window_pruning.py`

作用：

- 先在窗口级判断是否活跃
- 输出 `window_mask`
- 同时扩展为 `token_mask`

### `head_group`

位置：

- `src/models/modules/sparse_ops/head_pruning.py`

作用：

- 以 channel group 近似 attention head group
- 输出 `head_mask`

### `structured_token`

位置：

- `src/models/modules/sparse_ops/token_pruning.py`

作用：

- 标准 top-k spatial token 剪枝
- 现在统一输出 `token_mask`

---

## 7. 注意力机制还可以怎么改

下面分成两类：

1. 当前立刻可测的“注意力代理方案”
2. 下一步要进入 upstream block 内部的“深层 attention 改造方案”

## 7.1 当前立刻可测的注意力代理方案

这些方案都已经给了对应配置文件，可以直接做消融。

### ATTN-S1：Temporal Shift Attention Proxy

配置：

- `configs/experiment_schemes/attention_temporal_shift.yaml`

模块链：

- `temporal_shift`

思想：

- 不改 upstream attention 内核
- 只在进入 backbone 前加入时序混合
- 看看纯时间混合是否能提升运动特征表达

建议观察：

- AEE 是否下降
- 是否几乎不影响 latency proxy

### ATTN-S2：Temporal + Timestep Budget

配置：

- `configs/experiment_schemes/attention_temporal_timestep.yaml`

模块链：

- `temporal_shift`
- `timestep_budget`

思想：

- 把无效时间步在进入 backbone 前裁掉
- 测试“时序增强 + 时间稀疏”是否能形成更强的 attention 先验

建议观察：

- `active_timestep_ratio`
- AEE / AAE 与 baseline 比较

### ATTN-S3：Temporal + Window Budget

配置：

- `configs/experiment_schemes/attention_temporal_window.yaml`

模块链：

- `temporal_shift`
- `timestep_budget`
- `window_topk`

思想：

- 用时序增强和窗口裁剪联合形成“局部时空注意力代理”
- 这是当前最接近硬件友好的注意力增强方案

建议观察：

- `active_window_ratio`
- `active_timestep_ratio`
- 精度损失是否在可接受范围内

### ATTN-S4：Temporal + Window + Head Group

配置：

- `configs/experiment_schemes/attention_temporal_window_head.yaml`

模块链：

- `temporal_shift`
- `timestep_budget`
- `window_topk`
- `head_group`

思想：

- 在当前 hook 位置用 channel group 去近似 head 稀疏
- 作为未来真正 head-level sparse attention 的预实验

建议观察：

- `head_keep_ratio`
- 多级稀疏叠加后精度是否明显下降

---

## 7.2 下一步应该深入做的 attention 方案

这些方案当前还没有在代码里真正写进 upstream block，但我认为最值得做。

### ATTN-D1：STAtten 风格时空联合 Spike-QK

参考：

- [STAtten](https://arxiv.org/abs/2409.19764)
- [STAtten official code](https://github.com/Intelligent-Computing-Lab-Yale/STAtten)

改法：

- 在 upstream `Spiking_SwinTransformerBlock3D` 内部加入 temporal chunk attention
- 不只看 spatial window，还显式利用时间依赖

适合做成：

- `temporal_spike_qk`

### ATTN-D2：QKFormer 风格 Channel/Token Q-K 选择

参考：

- [QKFormer](https://arxiv.org/abs/2403.16552)
- [QKFormer official code](https://github.com/zhouchenlin2096/QKFormer)

改法：

- 沿用 spike-friendly Q-K 核心
- 增加 token/channel importance 分离建模

适合做成：

- `qk_channel_gate`
- `qk_token_gate`

### ATTN-D3：FlashAttention 风格 IO-aware Window Kernel

参考：

- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [FlashAttention official code](https://github.com/Dao-AILab/flash-attention)

改法：

- 不改变宏观网络
- 只改变局部 window attention 的执行方式
- 强调 SRAM 驻留和 tile reuse

适合做成：

- `flash_window_qk`

但要注意：

- 这条线偏系统/硬件，不是简单插件

---

## 8. token 稀疏剪枝还能怎么做

同样分成“当前可测”与“下一步深做”。

## 8.1 当前立刻可测的 token 稀疏方案

### TOK-S1：Structured Token Top-k

配置：

- `configs/experiment_schemes/token_structured_topk.yaml`

模块链：

- `structured_token`

作用：

- 最简单的 token 剪枝基线

### TOK-S2：Window First, Token Later

配置：

- `configs/experiment_schemes/token_window_then_token.yaml`

模块链：

- `window_topk`
- `structured_token`

作用：

- 先粗粒度裁窗口，再细粒度裁 token
- 更符合硬件调度层次

### TOK-S3：Spatiotemporal Budget

配置：

- `configs/experiment_schemes/token_spatiotemporal_budget.yaml`

模块链：

- `timestep_budget`
- `window_topk`
- `structured_token`

作用：

- 时间、窗口、token 三层稀疏同时启用
- 是当前最值得跑的稀疏主方案之一

### TOK-S4：Window + Head + Token

配置：

- `configs/experiment_schemes/token_window_head_token.yaml`

模块链：

- `window_topk`
- `head_group`
- `structured_token`

作用：

- 把 token 稀疏和 head-group 稀疏一起考虑
- 适合作为 future block-level sparse attention 的前置验证

---

## 8.2 下一步应考虑的 token 方案

### TOK-D1：DynamicViT 风格动态 token 预测器

参考：

- [DynamicViT](https://arxiv.org/abs/2106.02034)
- [DynamicViT official code](https://github.com/raoyongming/DynamicViT)

想法：

- 不只用简单活动强度
- 用一个轻量 selector 预测 token 重要性

适合做成：

- `dynamic_token_gate`

### TOK-D2：HeatViT 风格硬件友好 token selector

参考：

- [HeatViT](https://arxiv.org/abs/2211.08110)

想法：

- 直接把 selector 设计成硬件友好结构
- 兼顾延迟与准确率

适合做成：

- `latency_aware_window_selector`
- `hardware_friendly_token_selector`

### TOK-D3：ToMe 风格 token merge

参考：

- [ToMe](https://arxiv.org/abs/2210.09461)

想法：

- 不是删 token，而是合并相似 token
- 更适合作为“精度损失较小”的压缩路径

适合做成：

- `token_merge`
- `window_token_merge`

### TOK-D4：A-ViT 风格自适应 halting

参考：

- [A-ViT](https://arxiv.org/abs/2112.07658)

想法：

- 将 halting 机制推广到 token 维度
- 在训练时学习不同样本的不同预算

适合做成：

- `adaptive_token_budget`

---

## 9. 我建议你怎么测这些方案

不要一上来就全量乱组合。

推荐顺序：

### 第一组：注意力代理

1. `attention_temporal_shift`
2. `attention_temporal_timestep`
3. `attention_temporal_window`
4. `attention_temporal_window_head`

目的：

- 看注意力增强与多级稀疏是否有正向叠加

### 第二组：token 稀疏

1. `token_structured_topk`
2. `token_window_then_token`
3. `token_spatiotemporal_budget`
4. `token_window_head_token`

目的：

- 找到精度/稀疏率/硬件映射三者平衡点

### 第三组：交叉对比

把最好的一组 attention 方案和最好的一组 token 方案交叉组合。

例如：

- `attention_temporal_window` vs `token_spatiotemporal_budget`
- `attention_temporal_window_head` vs `token_window_head_token`

---

## 10. 如何真正跑通模型

这一部分给出逻辑顺序，而不是只给命令。

## 10.1 为什么要先跑 upstream 原版

因为你必须先确认：

- 数据集路径正确
- upstream 模型本身能跑
- loss/metrics 没问题

否则你无法判断后面的错误究竟来自：

- 环境
- 数据
- upstream
- 本地 wrapper
- 还是插件模块

## 10.2 为什么再跑本地 baseline

因为本地 baseline 是：

- upstream 主干
- 本地 adapter
- 本地 dataset wrapper
- 本地 trainer/eval

如果 baseline 本地版能跑，说明：

- 本地重构没有把原版链路破坏掉

## 10.3 为什么最后才跑插件版

因为插件版是在 baseline 之上加的：

- `plug_in_modules`
- mask 统计
- profiler 扩展

所以它应该建立在 baseline 已经稳定的前提下。

---

## 11. PowerShell + Anaconda 下的推荐运行方式

你已经补充说明：

- Python 在 `D:` 盘 Anaconda 里

所以建议统一使用：

- PowerShell
- 显式 conda hook
- 绝对路径
- `python -m ...`

不要依赖：

- `.sh`
- 模糊相对路径
- 多套 shell 混用

如果你的 Anaconda 根目录是 `D:\Anaconda`，典型起手式是：

```powershell
& D:\Anaconda\shell\condabin\conda-hook.ps1
conda activate base
```

如果是 `D:\Anaconda3`，就换成对应目录。

---

## 12. 跑通顺序

推荐严格按以下顺序：

1. upstream eval
2. local baseline eval
3. local baseline smoke train
4. `variant_modular` smoke train
5. `variant_modular` profiler
6. `configs/experiment_schemes/*` 逐个 ablation

---

## 13. 当前整个工程中我已经做了什么

### 软件与插件

- 增加可串联 `plug_in_modules` 管线
- 增加 `temporal_shift`
- 增加 `timestep_budget`
- 增加 `window_topk`
- 增加 `head_group`
- 统一 `token_mask` / `window_mask` / `head_mask` / `timestep_mask`

### 配置与实验

- 增加 `variant_modular.yaml`
- 增加 `configs/experiment_schemes/*.yaml`

### 文档

- `PAPER_CO_DESIGN_PROPOSAL.md`
- `MODULE_ZOO.md`
- `MODULAR_UPGRADE_TECHNICAL_DOC.md`
- `RUNBOOK_AND_RESEARCH_PLAN_2026.md`
- `FULL_STACK_TECHNICAL_GUIDE_ZH.md`

### 硬件协同

- 补充 `hw/docs/arch.md`
- 补充 `hw/docs/interfaces.md`
- 补充 `hw/docs/perf_model.md`

---

## 14. 当前工作的边界

虽然工程基础已经搭起来，但目前仍然有明确边界：

- attention 真正的内部替换还没完成
- 现在的大部分稀疏仍然是“张量置零”，不是“真实执行跳过”
- `head_group` 目前还是 channel-group proxy
- `attention_unit.v` 还不是完整 accelerator

所以你当前的最合理目标不是“直接做最终论文版”，而是：

1. 先稳定复现
2. 再用插件做消融
3. 再把最好的两条线下沉到 block-level hook

---

## 15. 推荐的研究推进顺序

### 第一步

把原版和 baseline 跑通。

### 第二步

把 `configs/experiment_schemes/` 跑一轮。

### 第三步

从里面选 2 条收益最大、最稳定、最适合硬件映射的方案。

### 第四步

将这 2 条方案真正下沉到 upstream `Spiking_SwinTransformerBlock3D`。

### 第五步

配合 profiler、perf model、RTL 接口形成完整故事。

---

## 16. 相关文献与开源代码

建议重点看：

- [QKFormer](https://arxiv.org/abs/2403.16552)
- [QKFormer official code](https://github.com/zhouchenlin2096/QKFormer)
- [STAtten](https://arxiv.org/abs/2409.19764)
- [STAtten official code](https://github.com/Intelligent-Computing-Lab-Yale/STAtten)
- [DynamicViT](https://arxiv.org/abs/2106.02034)
- [DynamicViT official code](https://github.com/raoyongming/DynamicViT)
- [ToMe](https://arxiv.org/abs/2210.09461)
- [TSM](https://arxiv.org/abs/1811.08383)
- [TSM official code](https://github.com/mit-han-lab/temporal-shift-module)
- [HeatViT](https://arxiv.org/abs/2211.08110)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [FlowFormer](https://arxiv.org/abs/2203.16194)
- [FlowFormer official code](https://github.com/drinkingcoder/FlowFormer-Official)

---

## 17. 结论

当前这套工程最重要的价值，不是已经把最终模型做完，而是已经把：

- 原版复现入口
- 本地插件系统
- 稀疏 mask 契约
- profiler
- 硬件接口语义

这些关键层次串到了同一个工程里。

所以你现在最正确的动作不是继续无上限加模块，而是：

1. 跑通
2. 做可控消融
3. 选最有希望的 2 条线
4. 深挖到 attention block 内部

这才是后面能发文章、能做 accelerator、能把工作讲清楚的路线。
