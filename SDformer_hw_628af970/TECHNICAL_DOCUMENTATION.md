# SDformerFlow 光流 SNN Transformer 软硬件协同工程技术文档

## 1. 文档目的

本文档用于完整整理当前仓库已经完成的工程重构、软件适配、模块化扩展、量化导出、性能分析和 RTL 骨架实现工作。

文档目标有三个：

1. 说明当前仓库已经实现了什么，以及这些实现之间如何连接。
2. 说明当前工程与上游 `SDformerFlow` baseline 的关系，以及本地新增适配层的职责边界。
3. 为后续继续做 `variant_a / variant_b / variant_c`、实验复现、软硬件协同评测和 RTL 深化实现提供统一技术参考。

---

## 2. 项目背景与当前定位

### 2.1 目标问题

本工程的总体目标是基于事件相机输入，在 SNN Transformer 架构上完成 optical flow 预测估计，并进一步向以下方向扩展：

- 可复现的 `SDformerFlow` baseline
- 可插拔的 SNN Transformer 结构创新模块
- 面向时间步稀疏、token 稀疏和低比特计算的软硬件协同优化
- 最终可对齐 Python 与 RTL 的量化导出和 golden 验证链路

### 2.2 本次已完成的核心工作

本轮已经完成的工作不是“最终论文级完整实现”，而是完成了一个正确方向上的可扩展工程底座，具体包括：

- 将错误的时间序列 `SDformer` 仓库替换为面向 optical flow 的工作空间
- 引入正确 baseline `SDformerFlow` 作为 git submodule
- 建立统一配置、脚本、数据集包装、模型适配层和训练评估入口
- 建立可插拔模块注册机制和三组变体配置
- 建立 profiling、量化规格、quant 参数导出和 golden 仿真工具
- 建立参数化 RTL 骨架、testbench 和综合脚本模板

### 2.3 当前实现阶段

当前工程已经完成：

- `Phase 0`: 仓库重建
- `Phase A`: baseline 接入框架
- `Phase B`: 模块库骨架和变体配置入口
- `Phase D/E` 的基础接口与骨架

当前工程尚未完成：

- 真实 DSEC / MVSEC 指标复现
- 深度 attention 替换到 upstream backbone 内部
- 真正面向 `variant_c` 的完整硬件映射 RTL
- 自动绘图脚本和论文级实验表

---

## 3. 上游基线与本地工程关系

### 3.1 上游基线

- 上游仓库：`https://github.com/yitian97/SDformerFlow.git`
- 本地路径：`third_party/SDformerFlow/`
- 当前锁定提交：`13088516440ab3faba4142c986d162cf5dd7c299`

### 3.2 为什么采用 submodule

采用 submodule 而不是直接复制代码，有两个目的：

1. 保留上游实现的完整可追溯性。
2. 将“baseline 原始实现”和“本地研究增强层”明确隔离，方便后续对照实验和 patch 管理。

### 3.3 本地工程做了什么

本地工程没有直接修改上游训练入口，而是在 `src/` 下新增了一层适配框架，完成以下工作：

- 统一配置入口
- 构建上游模型实例
- 统一 DSEC/MVSEC 数据接口
- 统一训练、评估、checkpoint、summary 输出
- 在不破坏上游代码结构的前提下插入 spike 编码、时间步裁剪、结构化稀疏等实验位点

### 3.4 本地工程没有做什么

以下内容目前仍然保持为“上游能力 + 本地包装”，尚未完全重写：

- Upstream Swin-Spiking attention 内部计算图
- Upstream decoder / head 结构
- Upstream loss 的具体定义
- Upstream DSEC/MVSEC 原始数据读取逻辑

---

## 4. 仓库结构说明

当前仓库结构如下：

```text
project_root/
  README.md
  REPORT.md
  TECHNICAL_DOCUMENTATION.md
  environment.yml
  requirements.txt
  configs/
    sdformer_baseline.yaml
    hw/
      quant_spec.yaml
    model_variants/
      variant_a.yaml
      variant_b.yaml
      variant_c.yaml
  scripts/
    setup_env.sh
    download_data.sh
    run_train.sh
    run_eval.sh
    run_ablation.sh
    profile_latency.sh
    export_onnx.sh
  src/
    datasets/
    models/
    trainers/
    utils/
  experiments/
    logs/
    results/
  tools/
    export_quant_params.py
    golden_hw_sim.py
  hw/
    rtl/
    tb/
    scripts/
    docs/
  third_party/
    SDformerFlow/
```

### 4.1 根目录文档

- `README.md`: 快速入口和仓库概览
- `REPORT.md`: 阶段性报告与当前状态
- `TECHNICAL_DOCUMENTATION.md`: 完整技术说明，作为工程主文档

### 4.2 配置目录

- `configs/sdformer_baseline.yaml`: 统一 baseline 配置
- `configs/model_variants/*.yaml`: 三个实验变体配置
- `configs/hw/quant_spec.yaml`: Python 与 RTL 共用定点规格

### 4.3 软件目录

- `src/datasets/`: DSEC / MVSEC 数据包装
- `src/models/`: 上游模型适配层和模块注册系统
- `src/trainers/`: 本地训练评估入口
- `src/utils/`: config、logging、checkpoint、profiler 等公用工具

### 4.4 硬件目录

- `hw/rtl/`: RTL 模块
- `hw/tb/`: testbench 和向量目录
- `hw/scripts/`: 仿真与综合脚本
- `hw/docs/`: 架构、接口和性能模型文档

---

## 5. 软件总体架构

### 5.1 软件数据流

当前软件栈的数据流如下：

```text
YAML Config
  -> Dataset Wrapper
  -> SDFormerFlowAdapter
  -> Input Preprocess
  -> Upstream SDformerFlow Model
  -> Loss / Metrics
  -> Checkpoint / Summary / Profile / Export
```

### 5.2 入口脚本

当前本地脚本入口如下：

- `scripts/run_train.sh`
  - 调用 `python -m src.trainers.train`
- `scripts/run_eval.sh`
  - 调用 `python -m src.trainers.eval`
- `scripts/run_ablation.sh`
  - 顺序评估 baseline 和变体，并写入汇总表
- `scripts/profile_latency.sh`
  - 调用 `python -m src.utils.profiler`

注意：当前训练和评估路径走的是本地适配层，不是直接转发到 upstream 的 `train_flow_parallel_supervised_SNN.py` 或 `eval_DSEC_flow_SNN.py`。

### 5.3 配置机制

配置采用 YAML + 本地轻量继承机制：

- 加载器：`src/utils/config.py`
- 支持 `inherit_from`
- 采用递归合并

这保证了：

- baseline 和 variants 共用同一份主配置
- 变体只覆盖差异字段
- 配置项不会散落到 Python 源码

---

## 6. Baseline 适配层设计

### 6.1 核心文件

- `src/models/sdformer/backbone.py`
- `src/models/sdformer/layers.py`
- `src/models/sdformer/spiking_neurons.py`

### 6.2 适配层职责

`SDFormerFlowAdapter` 是当前软件侧最关键的桥接组件，其职责包括：

1. 将本地统一配置转换为上游可识别的配置结构。
2. 根据配置实例化上游 `SpikingformerFlowNet` 变体。
3. 对输入事件体进行统一预处理。
4. 在前向中显式 reset SNN 状态，避免跨 batch 的膜电位泄漏。
5. 输出统一格式：
   - `flow_pred`
   - `aux.flow_list`
   - `aux.attn`
   - `aux.token_mask`
   - `aux.timestep_mask`

### 6.3 输入输出约定

适配层当前支持两种输入：

- `[B, T, H, W]`
- `[B, T, 2, H, W]`

如果输入是 `[B, T, H, W]`，适配层会自动拆分为正负极性：

```text
pos = relu(x)
neg = relu(-x)
chunk = stack(pos, neg, dim=2)
```

输出为：

- `flow_pred`: `[B, 2, H, W]`
- `aux.flow_list`: 多尺度 flow 列表
- `aux.token_mask`: 稀疏 token mask
- `aux.timestep_mask`: 时间步有效 mask

### 6.4 上游配置映射

`src/models/sdformer/layers.py` 中的 `build_upstream_config()` 会将本地统一配置映射到上游字段，例如：

- `dataset.root -> upstream.data.path`
- `model.num_bins -> upstream.data.num_frames`
- `runtime.num_workers -> upstream.loader.n_workers`
- `model.attention.window_size -> upstream.swin_transformer.window_size`
- `model.neuron.* -> upstream.spiking_neuron.*`

这一步的意义是把“上游原始结构”纳入本地统一配置管理。

---

## 7. 数据集包装层

### 7.1 DSEC 包装

文件：`src/datasets/optical_flow_dsec.py`

当前实现通过包装 upstream `DSECDatasetLite`，统一输出以下字段：

- `event_voxel`
- `gt_flow`
- `valid_mask`
- `dataset_name`

其中：

- `event_voxel`: `[T, H, W]`
- `gt_flow`: `[2, H, W]`
- `valid_mask`: `[1, H, W]`

### 7.2 MVSEC 包装

文件：`src/datasets/optical_flow_mvsec.py`

当前实现通过包装 upstream `MvsecEventFlow`，统一输出与 DSEC 相同 schema。

当 `num_chunks == 2` 时，会将 `event_volume_old` 与 `event_volume_new` 在时间维拼接。

### 7.3 数据接口统一的意义

这个统一 schema 解决了两个问题：

1. 训练器和评估器不再依赖具体数据集内部命名。
2. 后续新增数据集时，只需要实现同样的 batch contract。

---

## 8. 模型注册与可插拔模块库

### 8.1 注册系统

文件：`src/models/registry.py`

提供两个入口：

- `build_model(cfg)`
- `build_module(kind, name, **kwargs)`

当前支持的模块类别：

- `attention`
- `spike_encoding`
- `normalization`
- `sparse_ops`
- `token_mixer`
- `spiking_neurons`

### 8.2 当前已实现的模块

#### 8.2.1 Spike Encoding

文件：`src/models/modules/spike_encoding/encoders.py`

当前包含：

- `VoxelSpikeEncoder`
- `TemporalContrastEncoder`
- `LatencySpikeEncoder`

其中：

- `VoxelSpikeEncoder` 为直通
- `TemporalContrastEncoder` 对相邻时间步差分并归一化
- `LatencySpikeEncoder` 基于幅值顺序构造时延式编码

#### 8.2.2 Normalization

文件：`src/models/modules/normalization/rmsnorm.py`

当前实现了 `RMSNorm`，用于替换上游 `LayerNorm`。

当前替换方式是：

- 遍历上游模型
- 找到 `nn.LayerNorm`
- 替换为本地 `RMSNorm`

#### 8.2.3 Sparse Ops

文件：`src/models/modules/sparse_ops/token_pruning.py`

当前实现：

- `StructuredTokenPruner`
- `ActivityStats`

`StructuredTokenPruner` 的当前逻辑是：

- 对 `[B, T, C, H, W]` 在 channel 维做平均绝对值
- 展平成 `[B, T, H*W]`
- 每个时间步按 top-k 保留 token
- 输出同 shape 的 pruned tensor 和 `[B, T, H, W]` mask

#### 8.2.4 Attention / Token Mixer / Spiking Neuron 描述符

这些模块当前主要用于配置描述和未来替换位点保留：

- `attention/window_attention.py`
- `token_mixer/identity.py`
- `spiking_neurons/wrappers.py`

这意味着目前它们已经接入“配置层”和“注册层”，但尚未完成对 upstream 内部真实计算图的深度替换。

### 8.3 当前模块接入程度

当前真正参与前向路径的模块包括：

- `spike_encoding`
- `sparse_ops`
- `RMSNorm` 替换
- 时间步 early-exit mask 逻辑

当前尚未真正深入接入 upstream attention block 的模块包括：

- `WindowSpikeAttentionSpec`
- 更硬件友好的 token mixer
- neuron 内部硬件映射替换

---

## 9. Variants 设计与当前落地状态

### 9.1 Baseline

配置：`configs/sdformer_baseline.yaml`

特点：

- 使用 upstream `MS_SpikingformerFlowNet_en4`
- voxel 编码
- 固定 `num_bins = 10`
- 默认 `psn` 神经元
- 不启用时间步自适应
- 不启用 token 稀疏
- 不启用量化

### 9.2 Variant A

配置：`configs/model_variants/variant_a.yaml`

目标：

- 使用窗口化 spike attention 设计意图
- 将归一化替换为 `RMSNorm`
- 将神经元类型切换到 `plif`

当前实际落地：

- `RMSNorm` 已接入
- `plif` 配置已接入
- `window_spike` 目前仍主要是配置描述层，尚未深入改写 upstream attention 内部

### 9.3 Variant B

配置：`configs/model_variants/variant_b.yaml`

目标：

- 引入 `TemporalContrastEncoder`
- 打开 `adaptive_t`
- 依据 `early_exit_threshold` 屏蔽低活动时间步

当前实际落地：

- 时间差分编码已接入
- `timestep_mask` 已接入
- 实际是“将低活动时间步置零”，不是对 upstream block 做严格控制流跳过

### 9.4 Variant C

配置：`configs/model_variants/variant_c.yaml`

目标：

- 综合 variant A + B
- 引入结构化 token pruning
- 打开量化配置
- 作为硬件映射目标版本

当前实际落地：

- `TemporalContrastEncoder`
- `adaptive_t`
- `StructuredTokenPruner`
- `RMSNorm`
- `plif`
- 量化规格配置

当前尚未完成：

- 真实低比特训练或 PTQ/QAT 执行
- 基于 token mask 的 upstream block 内部 skip
- 与 RTL 精确一一对应的运算图裁剪

---

## 10. 训练、评估与结果输出链路

### 10.1 训练入口

文件：`src/trainers/train.py`

训练器职责：

- 加载统一配置
- 构建 train / eval 数据集
- 构建模型
- 加载 checkpoint
- 训练一个或多个 epoch
- 每个 epoch 后调用本地评估
- 保存 checkpoint、JSON、CSV history

当前优化器为：

- `AdamW`

当前 scheduler 为：

- `MultiStepLR`

### 10.2 评估入口

文件：`src/trainers/eval.py`

评估器职责：

- 加载配置和 checkpoint
- 执行前向推理
- 调用 upstream loss / metrics 计算
- 将结果写入 `experiments/results/tables/`

当前会输出：

- 单配置 JSON summary
- `ablation_summary.csv/.md`
- `dsec_main.csv/.md`
- `mvsec_generalization.csv/.md`

### 10.3 Loss 与 Metrics

当前没有重写光流监督损失，而是复用 upstream 定义：

- `src/trainers/losses.py`
- `src/trainers/metrics.py`

复用内容：

- `flow_loss_supervised`
- `AEE`
- `AAE`

这样做的优点是 baseline 对齐更容易；缺点是仍然依赖 upstream loss 代码结构。

---

## 11. Profiling 与软硬件协同分析基础

### 11.1 核心文件

- `src/utils/profiler.py`

### 11.2 当前 profiling 的输入

当前 profiler：

- 加载一个 sample
- 走模型前处理逻辑
- 读取 upstream `record_flops()` 输出
- 结合当前 `token_keep_ratio`、`active_timestep_ratio` 和稀疏度计算 proxy 指标

### 11.3 当前输出指标

每层输出以下字段：

- `mac_proxy`
- `weight_bytes_proxy`
- `activation_bytes_proxy`
- `token_keep_ratio`
- `active_timestep_ratio`
- `spike_density`
- `latency_cycle_proxy`

### 11.4 当前 profiling 的限制

当前 profiler 仍然属于“proxy model”，而不是严格的 cycle-accurate 硬件分析，原因包括：

- 使用 upstream FLOPs 记录近似代表 MAC
- 没有真实 SRAM/DRAM 访问模型
- 没有对具体 window schedule 做周期展开
- 没有对 RTL pipeline stall 做建模

因此它现在更适合作为 Phase D 的前置分析工具，而不是最终论文中的硬件定量结论来源。

---

## 12. 量化规格与硬件导出链路

### 12.1 量化规格

文件：`configs/hw/quant_spec.yaml`

定义内容包括：

- weight / activation / membrane / threshold / accumulator bitwidth
- rounding 模式
- clamp 模式
- tile 参数
- bus 宽度
- 误差预算

### 12.2 参数导出工具

文件：`tools/export_quant_params.py`

当前功能：

- 读取 checkpoint
- 根据量化配置计算每层 `weight_scale`
- 导出 layer shape、scale、tile 配置
- 输出 JSON

当前限制：

- 只导出参数级 scale
- 尚未导出完整 activation calibration 结果
- 尚未导出结构化稀疏 mask 的真实运行时统计文件

### 12.3 Golden 仿真工具

文件：`tools/golden_hw_sim.py`

当前功能：

- 使用固定点规则生成示例 stimulus
- 模拟简化 spike 累加、阈值比较和 reset
- 输出：
  - `manifest.json`
  - `stimulus.hex`
  - `expected.hex`

当前定位：

- 这是面向当前 RTL 骨架的最小 golden 流
- 还不是完整 optical-flow block 的 golden reference

---

## 13. RTL 架构说明

### 13.1 目标定位

当前 RTL 不是最终论文级 attention accelerator，而是一个为后续扩展预留参数和接口的最小可综合骨架。

目标版本指定为：

- `variant_c`

### 13.2 RTL 文件说明

- `hw/rtl/top.v`
  - 顶层流式接口
  - 连接 controller、attention、token mixer、spike 单元
- `hw/rtl/controller.v`
  - 有效握手与 fire 控制
- `hw/rtl/attention_unit.v`
  - 当前为直通占位
- `hw/rtl/token_mixer.v`
  - 当前为直通占位
- `hw/rtl/spike_unit.v`
  - 实现膜电位累加、阈值比较和 reset
- `hw/rtl/pe_array.v`
  - 参数化 signed MAC 阵列
- `hw/rtl/sram_if.v`
  - 简化 buffer shell

### 13.3 当前硬件数据流

```text
in_data
  -> controller
  -> attention_unit
  -> token_mixer
  -> spike_unit
  -> out_data
```

### 13.4 当前实现的关键参数

- `LANES = 8`
- `DATA_W = 8`
- `MEM_W = 12`
- `ACC_W = 24`
- `THRESHOLD = 4`

### 13.5 当前 RTL 的边界

当前 RTL 仅完成了以下级别的映射：

- lane-level 数据通路骨架
- spike accumulate / compare / reset
- 基础流式接口与时序

尚未完成：

- window attention 的真实计算
- token pruning 控制流
- timestep-level skip 调度
- 完整片上缓存调度
- 与具体 SDformerFlow encoder block 一一对应的 layer mapping

---

## 14. Testbench 与仿真流程

### 14.1 Testbench

文件：`hw/tb/tb_top.sv`

当前 testbench 会：

- 例化 `top`
- 提供 3 组固定输入
- 检查 `out_valid`
- 检查 `out_data` 是否匹配期望 spike bit
- PASS 后退出

### 14.2 仿真脚本

文件：`hw/scripts/run_sim.sh`

当前仿真流程：

- 用 `iverilog` 编译
- 用 `vvp` 执行

### 14.3 综合脚本

文件：`hw/scripts/run_synth.sh`

当前综合流程：

- 用 `yosys` 读取 RTL
- 指定 `top` 综合
- 输出 `stat`

### 14.4 当前仿真验证范围

当前只验证了简化 spike path，而不是完整 flow-estimation 运算图。

---

## 15. 接口契约

### 15.1 Python 侧 batch contract

当前本地训练器和评估器假设 batch 至少包含：

- `event_voxel`
- `gt_flow`
- `valid_mask`

### 15.2 模型输出 contract

`SDFormerFlowAdapter.forward()` 返回：

- `flow_pred`
- `aux.flow_list`
- `aux.attn`
- `aux.token_mask`
- `aux.timestep_mask`

### 15.3 RTL stream contract

详见 `hw/docs/interfaces.md`，当前定义：

- `in_valid / in_ready / in_last`
- `in_data[LANES*DATA_W-1:0]`
- `out_valid / out_ready / out_last`
- `out_data[LANES-1:0]`

### 15.4 误差契约

当前硬件接口文档中定义：

- RTL 与 golden 元素误差 `<= 1 LSB`
- 端到端 dequantized 输出偏差目标 `<= 0.02 EPE`

---

## 16. 环境与脚本说明

### 16.1 环境文件

- `environment.yml`
- `requirements.txt`

### 16.2 环境搭建脚本

文件：`scripts/setup_env.sh`

当前行为：

- 优先使用 `micromamba`
- 否则使用 `conda`
- 尝试创建环境
- 检查 `python`、`iverilog`、`yosys`

### 16.3 数据下载脚本

文件：`scripts/download_data.sh`

当前状态：

- 不自动下载数据
- 输出 DSEC / MVSEC 的手动下载和目录摆放说明

这是当前合理的实现，因为这两个数据集都涉及体积较大和手动获取流程。

---

## 17. 当前工程状态总结

### 17.1 已经完成的

- 正确 baseline 已接入
- 仓库结构已按 optical-flow 工程重建
- baseline / variants 的统一配置入口已建立
- DSEC / MVSEC wrapper 已建立
- 本地训练与评估入口已建立
- profiler 已建立
- quant spec 和导出工具已建立
- RTL 与 testbench 骨架已建立
- 架构、接口、性能模型文档已建立

### 17.2 尚未完成的关键项

- 本地环境未装好，无法实际运行训练和仿真
- baseline 数值复现尚未验证
- `variant_a/b/c` 尚未完成深入的 backbone 级替换
- 自动绘图脚本尚未实现
- RTL 仍是骨架，不是完整 accelerator
- Python golden 与完整 optical-flow block 对齐尚未完成

### 17.3 当前代码最重要的价值

当前代码最重要的价值不是“结果已经完成”，而是：

- 方向已经纠正到正确 baseline
- 软件与硬件接口已经统一
- 变体实验和硬件映射已经有稳定插入点
- 后续所有工作都可以在这个工程底座上继续推进，而不需要重新搭架构

---

## 18. 当前已知风险与技术债

### 18.1 环境风险

当前机器上缺少：

- 可用 `python`
- 可用 `pip`
- 可用 `iverilog`
- 可用 `yosys`

因此当前所有实现都属于“静态落地但未执行验证”的状态。

### 18.2 模型替换深度不足

当前 `variant_a` 中的 attention 替换仍然更多停留在配置层，而不是真正改写 upstream Swin attention 计算。

### 18.3 稀疏优化仍是前处理级

当前 token 稀疏和时间步 early-exit 的主要动作是：

- 置零
- 输出 mask

尚未在上游 block 级实现真正的控制流跳过。

### 18.4 硬件映射仍是最小原型

当前 RTL 骨架适合作为接口与验证链路的开始，但不能直接作为论文中的最终 accelerator 结果。

---

## 19. 推荐的下一步实施顺序

推荐后续按以下顺序推进：

1. 安装 Python 环境与 `iverilog` / `yosys`，完成 baseline smoke test。
2. 跑通 `scripts/run_train.sh` 和 `scripts/run_eval.sh` 的 baseline 全链路。
3. 验证 DSEC baseline 指标是否接近 upstream。
4. 将 `variant_b` 的 adaptive timestep 从“置零”升级为更严格的 block-level skip。
5. 将 `variant_a` 的 attention 替换真正深入到 upstream encoder block。
6. 完成 `variant_c` 的真实 PTQ/QAT 路径。
7. 将 profiler 从 FLOPs proxy 升级为更可信的 memory/cycle model。
8. 将 RTL 从当前 spike-path 原型升级到窗口 attention + token mixer 的真实映射。
9. 打通 `golden_hw_sim.py -> tb_vectors -> run_sim.sh` 的完整数据对齐。
10. 最后再补实验表、绘图脚本和论文材料。

---

## 20. 文档对应代码清单

为了便于快速定位，以下文件是当前工程最关键的实现入口：

### 软件入口

- `src/models/sdformer/backbone.py`
- `src/models/registry.py`
- `src/trainers/train.py`
- `src/trainers/eval.py`
- `src/utils/profiler.py`

### 配置入口

- `configs/sdformer_baseline.yaml`
- `configs/model_variants/variant_a.yaml`
- `configs/model_variants/variant_b.yaml`
- `configs/model_variants/variant_c.yaml`
- `configs/hw/quant_spec.yaml`

### 数据入口

- `src/datasets/optical_flow_dsec.py`
- `src/datasets/optical_flow_mvsec.py`

### 硬件入口

- `tools/export_quant_params.py`
- `tools/golden_hw_sim.py`
- `hw/rtl/top.v`
- `hw/tb/tb_top.sv`
- `hw/scripts/run_sim.sh`
- `hw/scripts/run_synth.sh`

---

## 21. 结论

当前工程已经完成了从错误 baseline 工作区向正确 optical-flow SNN Transformer 工程的迁移，并建立了一个统一的软件与硬件协同研发底座。

这个底座的主要价值在于：

- baseline 来源正确
- 目录结构清晰
- 配置统一
- 数据接口统一
- 变体实验位点清晰
- profiling / quant / RTL 的接口已经先对齐

当前最需要做的不是继续新增文档，而是基于这份文档和当前代码继续推进“环境验证、baseline 复现、变体深入替换和 RTL 深化实现”。

