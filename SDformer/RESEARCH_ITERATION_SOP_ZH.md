# SDformer 研究迭代 SOP

## 1. 目标

这份 SOP 用于回答两个问题：

1. 改模块、改注意力、做剪枝/稀疏之后，如何不用每次都跑完整 DSEC 训练，就快速判断是否值得继续投入。
2. 如何尽早判断这些改动在硬件实现上是否可行，而不是等到模型定型后才发现算子、量化或数据流走不通。

这份 SOP 适用于以下类型的改动：

- 新增模块
- 替换注意力
- 轻量化 block
- 通道剪枝 / head 剪枝 / block 剪枝
- 稀疏化
- 定点量化前的结构预调

## 2. 核心原则

每个改动都按 5 个阶段推进，不允许一上来就跑 full DSEC。

1. `Smoke 级`
   只验证链路和数值稳定性。
2. `方向级`
   用小数据子集看趋势，不追求最终精度。
3. `恢复级`
   看短训后是否能追平或超过基线趋势。
4. `硬件级`
   看算子、量化、数据流、稀疏模式是否可实现。
5. `论文级`
   只有前 4 级都通过，才进入 full DSEC 正式训练。

## 3. 固定基线

在开始任何改动前，先冻结一份可复用基线。后续所有改动都必须和这份基线比较。

基线至少包含：

- 代码提交哈希
- 配置文件路径
- DSEC 预处理版本
- 单序列 smoke 指标
- 小开发子集短训指标
- full DSEC 正式训练指标
- 显存占用
- 每 step 时间
- 参数量 / FLOPs

建议固定三条基线：

1. `Smoke baseline`
   单序列 `zurich_city_09_a`
2. `Dev-subset baseline`
   2 到 4 条代表性训练序列
3. `Full baseline`
   完整 DSEC 训练

## 4. 阶段 0：改动前检查

### 4.1 输入

- 你的改动目标
- 改动文件列表
- 预期收益类型

### 4.2 你必须先写清楚的内容

- 这次改动是为了解决什么问题
- 预期提升的是精度、速度、显存，还是硬件友好性
- 改动属于哪一类：
  - 精度导向
  - 轻量化导向
  - 硬件友好导向
  - 剪枝/稀疏导向

### 4.3 输出

形成一条实验记录，至少包括：

- `experiment_id`
- `baseline`
- `change_summary`
- `expected_gain`
- `risk`

## 5. 阶段 1：Smoke 级验证

### 5.1 目标

确认改动没有破坏训练链路、数据链路和评估链路。

### 5.2 使用的数据

- 单序列：`zurich_city_09_a`

### 5.3 使用的流程

1. 单序列预处理
2. 单序列 1 epoch 训练
3. 单序列评估

### 5.4 你要观察的指标

- 能否正常 forward / backward
- train loss 是否下降
- valid loss 是否可计算
- eval 是否能恢复权重
- 是否出现：
  - shape 错误
  - NaN
  - OOM
  - 显著速度退化

### 5.5 退出条件

满足以下条件才允许进入下一阶段：

- 训练和评估都能完整跑通
- loss 曲线没有明显异常
- 显存没有不可接受的增长
- step 时间没有灾难性恶化

如果不满足，直接回滚或修结构，不进入下一阶段。

## 6. 阶段 2：方向级开发子集验证

### 6.1 目标

在不跑 full DSEC 的前提下，快速判断改动方向有没有价值。

### 6.2 数据建议

不要只用一条序列。建议至少选 2 到 4 条代表性序列，包含：

- 1 条短序列
- 1 条中序列
- 1 条长序列
- 如果改动跟场景泛化有关，再加 1 条不同场景序列

建议优先从这些已带 GT 的训练序列中选：

- `zurich_city_09_a`
- `zurich_city_07_a`
- `zurich_city_02_c`
- `zurich_city_11_b`
- `thun_00_a`
- `zurich_city_10_a`

### 6.3 做法

1. 为开发子集生成单独 split
2. 固定训练预算
3. 所有候选改动都在同一预算下比较

固定预算建议：

- 固定 `epoch` 数，或
- 固定 `step` 数

推荐优先固定 `step` 数，因为不同改动的 dataloader / 模型速度可能不同。

### 6.4 必须记录的指标

- validation loss
- AEE
- AAE
- step time
- GPU memory
- 参数量
- FLOPs 或 MACs

### 6.5 判定规则

如果某个改动同时表现为：

- validation loss 下降更慢
- AEE / AAE 没改善
- 显存更高
- step time 更慢

则直接淘汰，不进入下一阶段。

如果某个改动在固定预算下表现为：

- loss 下降更快，或
- AEE / AAE 更好，或
- 精度基本持平但速度/显存更优

则进入下一阶段。

## 7. 阶段 3：恢复级短训验证

### 7.1 目标

验证改动是否具备继续训练的潜力，而不是只在极短训练里偶然占优。

### 7.2 做法

如果结构兼容，优先从基线 checkpoint warm start。

推荐两条路线二选一：

1. `warm start`
   适合轻改模块、注意力替换、小规模结构变化
2. `from scratch short train`
   适合结构变化较大、参数命名不兼容的改动

### 7.3 要看的信号

- 短训后是否仍优于 baseline 趋势
- 是否更快收敛
- 是否出现后期不稳定
- 是否出现过拟合变快

### 7.4 淘汰规则

短训后如果：

- 已经追不上 baseline
- 收敛更慢
- 波动更大
- 带来更高成本但收益不明显

则不值得进入 full train。

## 8. 阶段 4：硬件可行性评估

这一阶段和精度验证并行做，不能等模型完全定型后再做。

### 8.1 你必须回答的 4 个问题

1. `算子是否可落地`
2. `数据流是否可落地`
3. `量化是否可落地`
4. `稀疏是否真的可利用`

### 8.2 算子可落地检查

优先支持的算子类型：

- Conv
- Linear
- Add
- Compare
- Threshold
- Pooling
- 简单 reshape / concat

高风险算子类型：

- Softmax attention
- 大规模全局矩阵乘
- 动态 gather / scatter
- 大量不规则 index 操作
- 复杂 token 重排

结论标准：

- 如果新结构强依赖高风险算子，硬件落地风险高
- 如果能改写成固定窗口、局部连接、规则张量流，则优先

### 8.3 数据流可落地检查

需要记录：

- 各层输入输出 shape
- 中间特征图大小
- 是否需要频繁回写外存
- tile 切分是否规则
- head/channel/token 重排是否复杂

如果一个结构 FLOPs 更低，但中间访存更大，通常硬件收益未必成立。

### 8.4 量化可落地检查

建议每个候选结构都导出量化元数据。

当前仓库已有工具：

- [export_quant_params.py](/root/private_data/work/SDformer/tools/export_quant_params.py)
- [golden_hw_sim.py](/root/private_data/work/SDformer/tools/golden_hw_sim.py)

你要检查：

- 权重量化 scale 是否稳定
- 激活范围是否异常扩大
- SNN 膜电位 / 阈值是否容易固定点化
- 不同层是否需要过于不统一的位宽

如果改动后某些层 scale 非常极端，或者动态范围难以约束，硬件风险会明显升高。

### 8.5 稀疏可利用性检查

不要把“稀疏率高”直接等同于“硬件更快”。

优先考虑：

- channel pruning
- head pruning
- block pruning
- 规则 token reduction

谨慎对待：

- 非结构化权重稀疏
- 运行时动态不规则稀疏

你要明确记录：

- 稀疏模式是否规则
- 稀疏元数据开销
- tile 内有效利用率
- 是否破坏并行度

### 8.6 硬件阶段退出条件

满足以下条件才允许继续：

- 算子集合可被现有硬件设计表达
- 数据流没有明显不可承受的访存问题
- 量化导出结果合理
- 稀疏模式是结构化或可调度的

## 9. 阶段 5：进入 full DSEC 的门槛

只有满足以下全部条件，才允许进入 full DSEC 正式训练：

1. smoke 流程稳定
2. 开发子集短训优于 baseline，或等精度更低成本
3. 短训恢复能力通过
4. 硬件可行性没有明显红灯

如果缺少其中任意一项，不要上 full train。

## 10. 你每次实验必须留的记录

建议每次实验都写一条标准记录，字段如下：

- `experiment_id`
- `date`
- `git_commit`
- `config`
- `change_type`
- `change_summary`
- `baseline_name`
- `dataset_scope`
- `train_budget`
- `train_loss`
- `valid_loss`
- `AEE`
- `AAE`
- `step_time`
- `gpu_mem`
- `params`
- `flops`
- `spike_rate`
- `sparsity`
- `quant_status`
- `hw_risk`
- `decision`

其中 `decision` 只能是：

- `drop`
- `keep_for_short_train`
- `keep_for_full_train`

## 11. 推荐的研究节奏

### 11.1 注意力改进

建议顺序：

1. 先替换一个 block
2. 再替换一个 stage
3. 最后再替换全模型

不要一开始全模型替换，否则很难判断收益来自哪里。

### 11.2 新模块插入

建议顺序：

1. 先加在单一位置
2. 做位置消融
3. 再决定是否扩展到多层

### 11.3 剪枝

建议顺序：

1. 做层敏感性分析
2. 做结构化剪枝
3. 短训恢复
4. 再考虑更大范围压缩

### 11.4 稀疏

建议顺序：

1. 先看 spike activation 自然稀疏率
2. 再设计规则稀疏
3. 最后再看硬件调度是否能真正利用

## 12. 不建议做的事

- 每改一次结构就直接跑 full DSEC
- 没有固定预算就比较不同模型
- 只看 train loss，不看 AEE / AAE
- 只看参数/FLOPs，不看 latency 和访存
- 做非结构化稀疏后直接宣称硬件更快
- 等模型全部做完才考虑量化和硬件接口

## 13. 当前仓库可直接利用的工具

- [run_ablation.sh](/root/private_data/work/SDformer/scripts/run_ablation.sh)
  - 适合统一评估多个 variant checkpoint
- [profile_latency.sh](/root/private_data/work/SDformer/scripts/profile_latency.sh)
  - 适合做延迟 profiling
- [export_quant_params.py](/root/private_data/work/SDformer/tools/export_quant_params.py)
  - 导出量化参数与层信息
- [golden_hw_sim.py](/root/private_data/work/SDformer/tools/golden_hw_sim.py)
  - 生成固定点 golden vectors

注意：

- [export_onnx.sh](/root/private_data/work/SDformer/scripts/export_onnx.sh) 当前未接通，不能作为主导出路径

## 14. 一句话决策规则

一个改动只有在同时满足以下三点时，才值得进入 full DSEC：

1. 小预算下已经显示出精度或效率收益
2. 收益是实际可测的，不只是理论参数/FLOPs 变化
3. 算子、量化、数据流和稀疏模式在硬件上可实现

如果三条里缺任何一条，就停在当前阶段，不继续加算力。
