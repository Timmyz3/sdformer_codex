# NTS11 硬件 P0 Profiling 报告

- 实验：`dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_hardware_order_q7q17_deploy`
- checkpoint：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_20260805/checkpoint_epoch35.pth`
- samples：1
- 评估协议：`{'resolution': [480, 640], 'crop': None, 'window_size': [2, 15, 15], 'pretrained_window_size': [2, 15, 15], 'tokens_per_window': 450, 'remap': 'v1', 'bn_policy': 'no_running', 'bn_modules_changed': 78, 'eval_batch_size': 1, 'num_workers': 0}`
- 模块数量：`{'ATLIFTernaryPSN': 105, 'ShiftmaxAttention': 12}`
- 权重加载：`{'checkpoint': '/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_20260805/checkpoint_epoch35.pth', 'checkpoint_overlay_keys': 210, 'model_overlay_keys': 210, 'missing_count': 0, 'unexpected_count': 0, 'overlay_missing_count': 0, 'overlay_unexpected_count': 0, 'missing_sample': [], 'unexpected_sample': [], 'remap': 'v1'}`
- ATLIF 阈值训练/部署语义：`{'threshold_modes': ['official_atlif'], 'homeostatic_freeze_after_step': 1224, 'homeostatic_update_frozen_after_boundary': True, 'optimizer_gradient_freeze_enabled': False, 'optimizer_threshold_lr': 5e-06, 'configured_min_threshold': 0.001, 'configured_max_threshold': 2.0, 'official_atlif_runtime_clamp_applied': False, 'inference_threshold_source': 'checkpoint_static_parameter', 'statement': 'threshold_freeze_after_step stops only the separate homeostatic threshold_update path; optimizer threshold gradients remain active unless freeze_threshold_grad_after_step is true. official_atlif does not apply the configured min/max runtime clamp. Inference uses the threshold parameter stored in the checkpoint.'}`
- H60 调用记录：12
- ATLIF 记录模块：93

## H60 分 stage 统计

| stage | calls | gate_entropy | effective_tokens | q_active | k_active | K-zero token | active entries/row | fold classes/row | TTB2 empty |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2 | 8.8137 | 449.98 | 0.00281 | 0.01059 | 0.8585 | 63.69 | 2.79 | 0.3761 |
| 1 | 2 | 8.8137 | 449.98 | 0.00136 | 0.00346 | 0.9595 | 18.23 | 2.71 | 0.4271 |
| 2 | 6 | 8.8137 | 449.92 | 0.00750 | 0.02396 | 0.7855 | 96.55 | 3.97 | 0.3093 |
| 3 | 2 | 8.8135 | 449.86 | 0.00792 | 0.05219 | 0.6243 | 169.07 | 3.62 | 0.1938 |

## Exact Delta-TTX temporal toggle

| metric | element-weighted result |
|---|---:|
| temporal lanes | 48384000 |
| Q toggle density | 0.813120% |
| K toggle density | 2.912012% |
| Q-or-K update density | 3.698706% |
| t1 ideal lane skip | 96.301294% |
| full T=2 ideal TX compare reduction | 48.150647% |
| zero-update token/head | 67.185317% |
| mean changed-token run length | 4.1353 |
| empty 4-token update bundle | 51.907112% |
| empty 8-token update bundle | 44.164101% |

### Update lanes per token/head

| updated lanes | token/head count |
|---|---:|
| 0 | 1015842 |
| 1 | 165936 |
| 2 | 90886 |
| 3--4 | 101387 |
| 5--8 | 91859 |
| 9--16 | 43862 |
| 17+ | 2228 |

## True Token-Time Bundle density (T=2)

| spatial tokens/bundle | bundles | Q-or-K density | empty | K-zero | no K-motion | active 1--8 | active 1--12 | active 1--16 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1512000 | 2.062182% | 66.973347% | 75.976455% | 76.069114% | 28.946825% | 31.517262% | 32.552579% |
| 2 | 759360 | 2.062182% | 59.284661% | 71.047988% | 71.104088% | 30.351612% | 34.219474% | 36.649547% |
| 4 | 383040 | 2.062182% | 51.804250% | 66.062030% | 66.097797% | 30.179355% | 34.794538% | 37.816155% |
| 8 | 194880 | 2.062182% | 44.098420% | 60.654249% | 60.676827% | 29.357553% | 34.428879% | 37.959257% |

## TTX/H67 二值时间对充分统计

| metric | result |
|---|---:|
| temporal pairs | 1512000 |
| all-four-vector empty | 66.973347% |
| K motion zero | 76.069114% |
| Q/K temporal update zero | 67.185317% |
| TTX paired scores equal | 97.787368% |
| H67 paired scores equal | 97.538161% |
| both K slices zero | 75.976455% |
| exactly one K slice zero | 16.030886% |
| both K slices active | 7.992659% |
| both K zero and same TTX class | 75.805688% |
| both K zero and same H67 class | 75.805688% |
| per-token K zero | 83.991898% |
| TTX all score classes/row | 1.4449 |
| H67 all score classes/row | 5.1885 |
| TTX K-zero fold classes/row | 1.0537 |
| H67 K-zero fold classes/row | 4.3513 |

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
| decoder | 4 | 105984000 | 1.000000 | 211968000 | 26496000 |
| downsample | 3 | 16128000 | 1.000000 | 32256000 | 4032000 |
| patch | 1 | 18432000 | 1.000000 | 36864000 | 4608000 |
| prediction | 4 | 2040000 | 1.000000 | 4080000 | 510000 |
| resblock | 2 | 4608000 | 1.000000 | 9216000 | 1152000 |
| stage_skip_final | 1 | 2304000 | 1.000000 | 4608000 | 576000 |
| stage_skip_predownsample | 3 | 32256000 | 1.000000 | 64512000 | 8064000 |
| stage_x_out | 4 | 18432000 | 1.000000 | 36864000 | 4608000 |
| swin_block | 12 | 87552000 | 1.000000 | 175104000 | 21888000 |

## ATLIF 活性快照

| group | modules | activity | pos_rate | neg_rate |
|---|---:|---:|---:|---:|
| ternary | 0 | 0.000000 | 0.000000 | 0.000000 |
| binary | 93 | 0.063273 | 0.063273 | 0.000000 |

## 读法

- `stage_skip_predownsample` 只对应 S0/S1/S2 的 downsample 前 skip。
- `stage_skip_final` 对应 S3 final-stage output，硬件上要跨 bottleneck 保留给 decoder i=0。
- 旧 `TTB2 empty` 按整个 window/head 的 Q 活性聚合，只保留作历史代理，不能证明完整 attention 可跳过。
- `True Token-Time Bundle` 按 T=2 × contiguous spatial tokens × 32 lanes 统计 Q-or-K、K-zero 与 K-motion。
- Q/K empty 仍会产生 silent/silent score并参与 Shiftmax；只有 Delta score reuse、K-zero value gating等具备单独等价证明的路径可无损跳过。

## Linear与卷积运行时操作分账

| 范围 | 模块 | 调用 | dense标量MAC | 输入活动率 | 活动率加权MAC代理 |
|---|---:|---:|---:|---:|---:|
| bottleneck | 4 | 4 | 63700992000 | 0.120421 | 7670965248.000 |
| encoder | 71 | 71 | 562139136000 | 0.114358 | 41018040720.000 |
| prediction | 4 | 4 | 211968000 | 0.114804 | 24334804.000 |

dense标量MAC按运行时输出元素与weight fan-in计算。活动率加权MAC对Linear为连通度代理，
对带padding/stride的卷积不是精确SOP；它仍优于用全网单一firing rate缩放所有层。
