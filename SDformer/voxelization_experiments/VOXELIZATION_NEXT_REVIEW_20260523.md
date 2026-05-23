# 体素化优化可加实验 Review

日期：2026-05-23

## 当前 baseline 体素路径

SDformerFlow 当前训练主要读取已经预处理好的 DSEC voxel：

- baseline 预处理源码：`third_party/SDformerFlow/DSEC_dataloader/event_representations.py`
- baseline dataset 读取：`third_party/SDformerFlow/DSEC_dataloader/DSEC_dataset_lite.py`
- 训练中进入模型前的张量语义：signed voxel 先按正负拆分，变成 `[B, T, 2, H, W]`
- 当前外围实验入口：`autoresearch_sparsity.entrypoints.train`
- 当前外围 profile 入口：`autoresearch_sparsity.entrypoints.profile_upstream_sparse`
- 当前 adapter 位置：`autoresearch_sparsity/overlay/sparse_preprocess.py`

为了不破坏 baseline，体素化优化应该继续走外围 adapter 或单独预处理缓存两条路线。

## 已有实现状态

| 方案 | 文件/配置 | 状态 | 结论 |
|---|---|---|---|
| EDCFlow temporal diff | `TemporalDifferenceVoxel` / `v41a_edcflow_temporal_diff.yml` | 已实现，smoke 20 step 能训练；fixed2 没产出 summary | 可以继续加，优先做纯体素短测 |
| EventPillars lite | `EventPillarsLite` / `v41b_eventpillars_lite.yml` | 已实现，但还没看到有效 profile 结果 | 可以继续加，优先做 alpha sweep |
| SparseSpikFormer token mask | `s41a_sparsespikformer_token85.yml` | 已短测 | AAE 很差，且 SOPs 不降，不应作为体素主线 |
| QP-SNN SVS pruning | `s41b_qpsnn_svs90.yml` | 已短测 | AAE 很差，属于剪枝 proxy，不是体素主线 |
| SSF + QP + EDC | `s41c_ssf_qpsnn_edc.yml` | 配置已有 | 暂缓，变量太多 |

已有短测中 `s41a/s41b` 的 AAE 到 30+，说明输入级硬 mask 会破坏光流方向估计；体素化主线应先避免硬置零。

## 最适合马上加的体素化方案

### V42a：EDCFlow 轻量时间差分

动机：光流对相邻时间片变化敏感，EDCFlow 的 temporally dense difference maps 很贴合任务。

实现方式：保留原始 voxel 作为 carrier，只加小残差：

```text
out = voxel + alpha * (voxel[t] - voxel[t-1])
```

建议 sweep：

| 实验 | alpha | mode | 预期 |
|---|---:|---|---|
| V42a1 | 0.05 | residual | 最稳，先看是否不伤精度 |
| V42a2 | 0.10 | residual | 主候选 |
| V42a3 | 0.20 | residual | 已有 V41a 接近该设置，可能偏强 |
| V42a4 | 0.10 | residual_abs | 更强调运动边界，但可能增加 firing |

是否能加：能。无需改模型结构，无需重新生成数据，直接 adapter。

风险：如果 alpha 过大，输入分布变了，SNN firing 可能升高。必须 `rescale + clamp`。

### V42b：EventPillars Lite 密度/时间范围 cue

动机：EventPillars 的核心不是简单 voxel，而是把 density、temporal range、polarity activity 注入表示。

当前已有 `EventPillarsLite`，建议别一下子加太强：

| 实验 | density_alpha | range_alpha | 预期 |
|---|---:|---:|---|
| V42b1 | 0.04 | 0.02 | 最稳 |
| V42b2 | 0.08 | 0.04 | 主候选 |
| V42b3 | 0.12 | 0.08 | 已有 V41b 设置，可能偏强 |

是否能加：能。无需改模型结构。

风险：如果 density/range 在空体素位置注入非零，会让 firing 上升。当前实现已经有 `preserve_zero: true`，这是必须保留的。

### V42c：EDCFlow + EventPillars 轻融合

动机：一个补时间变化，一个补局部密度/时间范围，理论上互补。

建议只试一个很轻的组合：

```yaml
edc alpha: 0.05
pillars density_alpha: 0.04
pillars range_alpha: 0.02
```

是否能加：能，但需要 pipeline 支持两个 voxel adapter 顺序执行。当前 `build_sparsity_pipeline` 只支持一个 `voxel_adapter`，要小改成 `voxel_adapters: [...]`。

风险：融合容易让变量变多，应该放在 V42a/V42b 单独短测之后。

### V43：自适应密度归一化 ADM-lite

动机：事件光流对事件密度敏感，MDR/ADM 方向适合做体素预处理故事。

实现方式：不硬剪枝，只按空间局部密度做温和归一化或门控：

```text
density = avg_pool(abs(voxel))
gain = target_density / (density + eps)
out = voxel * clipped_gain
```

是否能加：能，代码量小，仍保持 `[B,T,2,H,W]`。

建议作为第二优先级。它比 token mask 更适合讲“体素化自适应”，也比直接置零更不伤光流方向。

风险：可能改变幅值统计，要做 minmax clamp 和 nonzero mean rescale。

## 暂时不建议马上加的方案

| 方案 | 原因 |
|---|---|
| Learnable / unbiased binning | 当前训练读取 `.npy` 预处理 voxel，没有 raw event list；要改 dataset 或重预处理，工程量大 |
| OmniEvent 完整版 | 已经接近模型 stem/attention 改动，和神经元实验变量混淆 |
| EventFlash 自适应时间窗口完整版 | 改变时间对齐，光流标签敏感；先做固定输出形状的轻量版本 |
| 原生 sparse voxel | 要重写模型计算图，短期无法和 baseline 公平对比 |
| hard token/window pruning | 之前 AAE 爆了；除非模型内部真实跳过计算，否则 SOPs 故事也不稳 |

## 推荐执行顺序

1. 先跑纯体素 V42 smoke：EDC alpha sweep + EventPillars alpha sweep，不叠加剪枝。
2. 只保留 AAE 不爆、SOPs/firing 不升的方案做 120-step 短训。
3. 如果 V42a/V42b 单独有效，再做 V42c 轻融合。
4. 如果纯体素对精度有帮助但 SOPs 不降，把它定位为“前端表示增强”，后续再和 H 系列神经元稀疏组合。
5. 如果目标是稀疏故事，优先试 V43 ADM-lite，而不是硬 token pruning。

## 需要补的工程点

- `rapid_screen_sparse.py` 需要把 V41/V42 纯体素方案单独队列化，避免和 sparse pruning 混跑。
- `build_sparsity_pipeline` 可扩展 `voxel_adapters: [...]`，支持 EDC + Pillars 顺序组合。
- profile 需要额外记录输入统计：nonzero ratio、mean abs、max、per-bin activity。否则只看模型 firing，很难判断体素化是否真的变稀疏。
- fixed2 中 V41a 有 train log 但没有 summary，后续队列要在 train 后检查 checkpoint 是否生成，并把失败原因写入 summary。

## 我建议马上排的配置

| 编号 | 配置内容 | 是否全量 |
|---|---|---|
| V42a1 | EDC residual alpha=0.05 | smoke -> 120 step |
| V42a2 | EDC residual alpha=0.10 | smoke -> 120 step |
| V42b1 | EventPillars density=0.04 range=0.02 | smoke -> 120 step |
| V42b2 | EventPillars density=0.08 range=0.04 | smoke -> 120 step |
| V43a | ADM-lite target density 温和归一化 | 先实现 smoke |

升级门槛建议：

- valid20 AAE 不超过 baseline valid20 太多，至少不能出现 30+；
- SOPs 不高于 baseline，或 firing 不升；
- 输入 nonzero ratio 不显著膨胀；
- 训练 120 step loss 不能明显异常。
