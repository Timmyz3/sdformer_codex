# 稀疏剪枝与体素化短测记录

## 当前目标

本目录只做不破坏 `third_party/SDformerFlow` 的外围实验。训练入口仍复用 baseline 入口逻辑，但通过 `autoresearch_sparsity.entrypoints.train` 在 batch 进入模型前插入 sparse/voxel adapter；推理和 SOPs 统计通过 `profile_upstream_sparse.py` 在同一位置插入 adapter，保证训练和评估一致。

## 已实现候选

| 配置 | 模块 | 参考动机 | 当前定位 |
|---|---|---|---|
| `s41a_sparsespikformer_token85.yml` | `SparseSpikFormerTokenPruning` | SparseSpikFormer 的 foreground token/activity 选择 | 输入 token mask 代理实验，适合判断预处理质量；如果模型内部计算图不跳过 masked token，SOPs 不一定下降 |
| `s41b_qpsnn_svs90.yml` | `QPSNNSVSPruning` | QP-SNN 的结构化重要性/奇异值思想 | 时序-极性整片剪枝，硬件友好，但目前是 SVS proxy，不等于完整 QP-SNN |
| `v41a_edcflow_temporal_diff.yml` | `TemporalDifferenceVoxel` | EDCFlow 类时间差分体素信息 | 体素增强/时间差分候选，已改名为 EDCFlow，输出 clamp 到 minmax 体素范围 |
| `v41b_eventpillars_lite.yml` | `EventPillarsLite` | EventPillars 的密度和时间范围 cue | 体素增强候选，保持空体素仍为空，避免凭空增加 firing |
| `s41c_ssf_qpsnn_edc.yml` | SSF + QP + EDCFlow | 组合短测 | 用于看剪枝和时间差分是否互补 |

## 本轮修复

- `train.py` 同时 patch 训练和验证 forward，避免只训练启用 adapter、验证没启用。
- `profile_upstream_sparse.py` 在 `prepare_batch` 后、`model(chunk)` 前插入同一个 pipeline，避免训练和 SOPs/指标统计不一致。
- `rapid_screen_sparse.py` 修复 profile 非零退出码仍可能被误判为 pass 的问题。
- `rapid_screen_sparse.py` 默认让 profile batch size 等于训练 batch size，并跳过训练脚本自带 validation，只保留统一 profile，避免口径不一致和重复前向。
- `QPSNNSVSPruning` 修正 `preserve_dc` 语义：现在 `remove_dc=False` 才表示保留 DC 分量；旧配置中的 `preserve_dc=True` 仍兼容。
- `QPSNNSVSPruning` 对粗粒度时序/极性单元使用 `ceil`，避免 0.95 keep ratio 在 10 个单元上被截断成 0.90。
- `TemporalDifferenceVoxel` 增加 clamp，避免 EDCFlow residual/diff 增强后越出 baseline minmax 输入范围。
- `EventPillarsLite` 增加 `preserve_zero` 和 clamp，避免空体素被密度/range 分支填成非零。

## 短测判断

旧版 `s41a` 120-step 已完成，但属于修复前结果，只能作为参考：

| 实验 | AEE | AAE | SOPs(G) | firing | 结论 |
|---|---:|---:|---:|---:|---|
| `s41a_sparsespikformer_token85_steps120` | 1.0742 | 8.2576 | 4.2183 | 0.09895 | 精度尚可，但 SOPs 未下降到 3.35G 以下，说明单纯输入 mask 不足以形成真实计算节省 |

修复后已启动 `s41_sparse_voxel_short120_fixed` 队列。后续只把同时满足精度和 SOPs 门槛的方案升级为更长训练；如果只有体素增强精度好但 SOPs 不降，应归类到体素化优化，不作为神经元/注意力稀疏主线证据。
