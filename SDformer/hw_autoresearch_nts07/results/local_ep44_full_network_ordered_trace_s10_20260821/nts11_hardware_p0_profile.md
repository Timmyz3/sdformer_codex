# NTS11 硬件 P0 Profiling 报告

- 实验：`dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50_hardware_order_q7q17_deploy`
- checkpoint：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50_20260812/checkpoint_epoch44.pth`
- samples：10
- 评估协议：`{'resolution': [480, 640], 'crop': None, 'window_size': [2, 15, 15], 'pretrained_window_size': [2, 9, 9], 'tokens_per_window': 450, 'remap': 'v1', 'bn_policy': 'no_running', 'bn_modules_changed': 78, 'eval_batch_size': 1, 'num_workers': 0}`
- 模块数量：`{'ATLIFTernaryPSN': 105, 'ShiftmaxAttention': 12}`
- 权重加载：`{'checkpoint': '/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50_20260812/checkpoint_epoch44.pth', 'checkpoint_overlay_keys': 210, 'model_overlay_keys': 210, 'missing_count': 0, 'unexpected_count': 0, 'overlay_missing_count': 0, 'overlay_unexpected_count': 0, 'missing_sample': [], 'unexpected_sample': [], 'remap': 'v1'}`
- ATLIF 阈值训练/部署语义：`{'threshold_modes': ['official_atlif'], 'homeostatic_freeze_after_step': 1224, 'homeostatic_update_frozen_after_boundary': True, 'optimizer_gradient_freeze_enabled': False, 'optimizer_threshold_lr': 5e-06, 'configured_min_threshold': 0.001, 'configured_max_threshold': 2.0, 'official_atlif_runtime_clamp_applied': False, 'inference_threshold_source': 'checkpoint_static_parameter', 'statement': 'threshold_freeze_after_step stops only the separate homeostatic threshold_update path; optimizer threshold gradients remain active unless freeze_threshold_grad_after_step is true. official_atlif does not apply the configured min/max runtime clamp. Inference uses the threshold parameter stored in the checkpoint.'}`
- H60 调用记录：0
- ATLIF 记录模块：93

## H60 分 stage 统计

| stage | calls | gate_entropy | effective_tokens | q_active | k_active | K-zero token | active entries/row | fold classes/row | TTB2 empty |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

## Exact Delta-TTX temporal toggle

| metric | element-weighted result |
|---|---:|
| temporal lanes | 0 |
| Q toggle density | 0.000000% |
| K toggle density | 0.000000% |
| Q-or-K update density | 0.000000% |
| t1 ideal lane skip | 0.000000% |
| full T=2 ideal TX compare reduction | 0.000000% |
| zero-update token/head | 0.000000% |
| mean changed-token run length | 0.0000 |
| empty 4-token update bundle | 0.000000% |
| empty 8-token update bundle | 0.000000% |

### Update lanes per token/head

| updated lanes | token/head count |
|---|---:|
| 0 | 0 |
| 1 | 0 |
| 2 | 0 |
| 3--4 | 0 |
| 5--8 | 0 |
| 9--16 | 0 |
| 17+ | 0 |

## True Token-Time Bundle density (T=2)

| spatial tokens/bundle | bundles | Q-or-K density | empty | K-zero | no K-motion | active 1--8 | active 1--12 | active 1--16 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| 2 | 0 | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| 4 | 0 | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% |
| 8 | 0 | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% | 0.000000% |

## TTX/H67 二值时间对充分统计

| metric | result |
|---|---:|
| temporal pairs | 0 |
| all-four-vector empty | 0.000000% |
| K motion zero | 0.000000% |
| Q/K temporal update zero | 0.000000% |
| TTX paired scores equal | 0.000000% |
| H67 paired scores equal | 0.000000% |
| both K slices zero | 0.000000% |
| exactly one K slice zero | 0.000000% |
| both K slices active | 0.000000% |
| both K zero and same TTX class | 0.000000% |
| both K zero and same H67 class | 0.000000% |
| per-token K zero | 0.000000% |
| TTX all score classes/row | 0.0000 |
| H67 all score classes/row | 0.0000 |
| TTX K-zero fold classes/row | 0.0000 |
| H67 K-zero fold classes/row | 0.0000 |

完整 Q/K cardinality、intersection、same-zero、motion、temporal-update、四向量事件数/并集、
TTX/H67 Q7 分数和行占用类直方图保存在 JSON；`--ordered-trace` 额外保存 Q/K/intersection/
四向量并集的 stage/block 有序压缩 trace。

## 光流样本特征与硬件 workload 相关性

| Pearson pair | r |
|---|---:|

相关性只用于判断是否值得做 stage/sample-aware 调度，不表示因果关系；profile100 仍需报告散点、
置信区间和异常样本，不能只挑选绝对值最大的相关系数。

## Activation / Skip 存储口径

| kind | calls | elements | density | FP16 bytes | ternary packed bytes |
|---|---:|---:|---:|---:|---:|
| decoder | 40 | 1059840000 | 1.000000 | 2119680000 | 264960000 |
| downsample | 30 | 161280000 | 1.000000 | 322560000 | 40320000 |
| patch | 10 | 184320000 | 1.000000 | 368640000 | 46080000 |
| prediction | 40 | 20400000 | 1.000000 | 40800000 | 5100000 |
| resblock | 20 | 46080000 | 1.000000 | 92160000 | 11520000 |
| stage_skip_final | 10 | 23040000 | 1.000000 | 46080000 | 5760000 |
| stage_skip_predownsample | 30 | 322560000 | 1.000000 | 645120000 | 80640000 |
| stage_x_out | 40 | 184320000 | 1.000000 | 368640000 | 46080000 |
| swin_block | 120 | 875520000 | 1.000000 | 1751040000 | 218880000 |

## ATLIF 活性快照

| group | modules | activity | pos_rate | neg_rate |
|---|---:|---:|---:|---:|
| ternary | 0 | 0.000000 | 0.000000 | 0.000000 |
| binary | 93 | 0.065271 | 0.065271 | 0.000000 |

## 读法

- `stage_skip_predownsample` 只对应 S0/S1/S2 的 downsample 前 skip。
- `stage_skip_final` 对应 S3 final-stage output，硬件上要跨 bottleneck 保留给 decoder i=0。
- 旧 `TTB2 empty` 按整个 window/head 的 Q 活性聚合，只保留作历史代理，不能证明完整 attention 可跳过。
- `True Token-Time Bundle` 按 T=2 × contiguous spatial tokens × 32 lanes 统计 Q-or-K、K-zero 与 K-motion。
- Q/K empty 仍会产生 silent/silent score并参与 Shiftmax；只有 Delta score reuse、K-zero value gating等具备单独等价证明的路径可无损跳过。

## 同序列相邻样本的stage边界变化

| 边界 | 可比较样本对 | 采样值 | 精确相等 | active翻转 | 符号类变化 | 归一化绝对变化 |
|---|---:|---:|---:|---:|---:|---:|
| S0.skip | 9 | 9216000 | 0.000000 | 0.000000 | 0.360384 | 1.094183 |
| S0.x_out | 9 | 9216000 | 0.000000 | 0.000000 | 0.366374 | 1.104509 |
| S1.skip | 9 | 9216000 | 0.000000 | 0.000000 | 0.347090 | 1.057441 |
| S1.x_out | 9 | 8294400 | 0.000000 | 0.000000 | 0.364230 | 1.111009 |
| S2.skip | 9 | 8294400 | 0.000000 | 0.000000 | 0.295964 | 0.916852 |
| S2.x_out | 9 | 6912000 | 0.000000 | 0.000000 | 0.290549 | 0.892138 |
| S3.skip | 9 | 6912000 | 0.000000 | 0.000000 | 0.240118 | 0.752599 |
| S3.x_out | 9 | 6912000 | 0.000000 | 0.000000 | 0.242624 | 0.758387 |

该表只比较验证列表中同一sequence的相邻条目，并对每个张量最多确定性采样2^20个值。
它用于筛选persistent-HTT或增量更新候选，不等价于证明整帧可复用。

## Linear与卷积运行时操作分账

| 范围 | 模块 | 调用 | dense标量MAC | 输入活动率 | 活动率加权MAC代理 |
|---|---:|---:|---:|---:|---:|
| bottleneck | 4 | 40 | 637009920000 | 0.122941 | 78314764032.000 |
| encoder | 71 | 710 | 5621391360000 | 0.119227 | 438522242400.000 |
| prediction | 4 | 40 | 2119680000 | 0.117529 | 249124842.000 |

dense标量MAC按运行时输出元素与weight fan-in计算。活动率加权MAC对Linear为连通度代理，
对带padding/stride的卷积不是精确SOP；它仍优于用全网单一firing rate缩放所有层。
