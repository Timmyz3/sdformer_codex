# NTS11 硬件 P0 Profiling 报告

- 实验：`dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_hardware_order_q7q17_deploy`
- checkpoint：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_20260805/checkpoint_epoch35.pth`
- samples：10
- 评估协议：`{'resolution': [480, 640], 'crop': None, 'window_size': [2, 15, 15], 'pretrained_window_size': [2, 15, 15], 'tokens_per_window': 450, 'remap': 'v1', 'bn_policy': 'no_running', 'bn_modules_changed': 78, 'eval_batch_size': 1, 'num_workers': 0}`
- 模块数量：`{'ATLIFTernaryPSN': 105, 'ShiftmaxAttention': 12}`
- 权重加载：`{'checkpoint': '/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_20260805/checkpoint_epoch35.pth', 'checkpoint_overlay_keys': 210, 'model_overlay_keys': 210, 'missing_count': 0, 'unexpected_count': 0, 'overlay_missing_count': 0, 'overlay_unexpected_count': 0, 'missing_sample': [], 'unexpected_sample': [], 'remap': 'v1'}`
- ATLIF 阈值训练/部署语义：`{'threshold_modes': ['official_atlif'], 'homeostatic_freeze_after_step': 1224, 'homeostatic_update_frozen_after_boundary': True, 'optimizer_gradient_freeze_enabled': False, 'optimizer_threshold_lr': 5e-06, 'configured_min_threshold': 0.001, 'configured_max_threshold': 2.0, 'official_atlif_runtime_clamp_applied': False, 'inference_threshold_source': 'checkpoint_static_parameter', 'statement': 'threshold_freeze_after_step stops only the separate homeostatic threshold_update path; optimizer threshold gradients remain active unless freeze_threshold_grad_after_step is true. official_atlif does not apply the configured min/max runtime clamp. Inference uses the threshold parameter stored in the checkpoint.'}`
- H60 调用记录：120
- ATLIF 记录模块：93

## H60 分 stage 统计

| stage | calls | gate_entropy | effective_tokens | q_active | k_active | K-zero token | active entries/row | fold classes/row | TTB2 empty |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 20 | 8.8137 | 449.98 | 0.00290 | 0.01092 | 0.8547 | 65.36 | 2.78 | 0.3784 |
| 1 | 20 | 8.8138 | 449.98 | 0.00137 | 0.00355 | 0.9581 | 18.84 | 2.60 | 0.4523 |
| 2 | 60 | 8.8137 | 449.92 | 0.00765 | 0.02381 | 0.7848 | 96.83 | 4.00 | 0.3187 |
| 3 | 20 | 8.8135 | 449.85 | 0.00839 | 0.05238 | 0.6358 | 163.91 | 3.69 | 0.1956 |

## Exact Delta-TTX temporal toggle

| metric | element-weighted result |
|---|---:|
| temporal lanes | 483840000 |
| Q toggle density | 0.837690% |
| K toggle density | 2.936416% |
| Q-or-K update density | 3.748345% |
| t1 ideal lane skip | 96.251655% |
| full T=2 ideal TX compare reduction | 48.125828% |
| zero-update token/head | 66.869868% |
| mean changed-token run length | 4.2273 |
| empty 4-token update bundle | 51.857769% |
| empty 8-token update bundle | 44.457102% |

### Update lanes per token/head

| updated lanes | token/head count |
|---|---:|
| 0 | 10110724 |
| 1 | 1653657 |
| 2 | 920999 |
| 3--4 | 1039571 |
| 5--8 | 926501 |
| 9--16 | 445510 |
| 17+ | 23038 |

## True Token-Time Bundle density (T=2)

| spatial tokens/bundle | bundles | Q-or-K density | empty | K-zero | no K-motion | active 1--8 | active 1--12 | active 1--16 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15120000 | 2.086293% | 66.662083% | 75.807262% | 75.897566% | 29.216858% | 31.818029% | 32.864676% |
| 2 | 7593600 | 2.086293% | 59.075985% | 70.907962% | 70.959914% | 30.438738% | 34.375909% | 36.832267% |
| 4 | 3830400 | 2.086293% | 51.761539% | 66.018484% | 66.050282% | 29.906146% | 34.645546% | 37.741907% |
| 8 | 1948800 | 2.086293% | 44.389830% | 60.747640% | 60.771860% | 28.607707% | 33.743688% | 37.344314% |

## TTX/H67 二值时间对充分统计

| metric | result |
|---|---:|
| temporal pairs | 15120000 |
| all-four-vector empty | 66.662083% |
| K motion zero | 75.897566% |
| Q/K temporal update zero | 66.869868% |
| TTX paired scores equal | 97.741726% |
| H67 paired scores equal | 97.499153% |
| both K slices zero | 75.807262% |
| exactly one K slice zero | 16.142070% |
| both K slices active | 8.050668% |
| both K zero and same TTX class | 75.609504% |
| both K zero and same H67 class | 75.609504% |
| per-token K zero | 83.878297% |
| TTX all score classes/row | 1.4310 |
| H67 all score classes/row | 5.1348 |
| TTX K-zero fold classes/row | 1.0523 |
| H67 K-zero fold classes/row | 4.3043 |

完整 Q/K cardinality、intersection、same-zero、motion、temporal-update、四向量事件数/并集、
TTX/H67 Q7 分数和行占用类直方图保存在 JSON；`--ordered-trace` 额外保存 Q/K/intersection/
四向量并集的 stage/block 有序压缩 trace。

## 光流样本特征与硬件 workload 相关性

| Pearson pair | r |
|---|---:|
| label_flow_mag_p90__vs__s1_pair_empty_ratio | -0.81569 |
| sample_aee__vs__s0_pair_empty_ratio | 0.76072 |
| sample_aee__vs__pair_empty_ratio | 0.69176 |
| label_flow_mag_mean__vs__s1_pair_empty_ratio | -0.68654 |
| label_flow_gradient_mean__vs__s1_pair_empty_ratio | -0.67567 |
| sample_aee__vs__mean_union_lanes_per_pair | -0.66056 |
| sample_aee__vs__mean_events_per_pair | -0.64569 |
| sample_aee__vs__s3_pair_empty_ratio | -0.63748 |
| input_event_density__vs__s1_pair_empty_ratio | -0.52699 |
| sample_aee__vs__token_kzero_ratio | 0.50306 |
| label_flow_gradient_mean__vs__s2_pair_empty_ratio | 0.47995 |
| input_active_pixel_ratio__vs__s3_pair_empty_ratio | -0.47014 |
| input_active_pixel_ratio__vs__s1_pair_empty_ratio | -0.42418 |
| sample_aee__vs__s2_pair_empty_ratio | 0.39437 |
| label_flow_mag_mean__vs__mean_union_lanes_per_pair | 0.38094 |
| input_event_density__vs__s3_pair_empty_ratio | -0.36859 |

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
| binary | 93 | 0.062964 | 0.062964 | 0.000000 |

## 读法

- `stage_skip_predownsample` 只对应 S0/S1/S2 的 downsample 前 skip。
- `stage_skip_final` 对应 S3 final-stage output，硬件上要跨 bottleneck 保留给 decoder i=0。
- 旧 `TTB2 empty` 按整个 window/head 的 Q 活性聚合，只保留作历史代理，不能证明完整 attention 可跳过。
- `True Token-Time Bundle` 按 T=2 × contiguous spatial tokens × 32 lanes 统计 Q-or-K、K-zero 与 K-motion。
- Q/K empty 仍会产生 silent/silent score并参与 Shiftmax；只有 Delta score reuse、K-zero value gating等具备单独等价证明的路径可无损跳过。

## 同序列相邻样本的stage边界变化

| 边界 | 可比较样本对 | 采样值 | 精确相等 | active翻转 | 符号类变化 | 归一化绝对变化 |
|---|---:|---:|---:|---:|---:|---:|
| S0.skip | 9 | 9216000 | 0.000000 | 0.000000 | 0.370524 | 1.123540 |
| S0.x_out | 9 | 9216000 | 0.000000 | 0.000000 | 0.371612 | 1.116808 |
| S1.skip | 9 | 9216000 | 0.000000 | 0.000000 | 0.366775 | 1.122736 |
| S1.x_out | 9 | 8294400 | 0.000000 | 0.000000 | 0.368461 | 1.118504 |
| S2.skip | 9 | 8294400 | 0.000000 | 0.000000 | 0.336571 | 1.026155 |
| S2.x_out | 9 | 6912000 | 0.000000 | 0.000000 | 0.300654 | 0.917516 |
| S3.skip | 9 | 6912000 | 0.000000 | 0.000000 | 0.276859 | 0.852142 |
| S3.x_out | 9 | 6912000 | 0.000000 | 0.000000 | 0.278150 | 0.856657 |

该表只比较验证列表中同一sequence的相邻条目，并对每个张量最多确定性采样2^20个值。
它用于筛选persistent-HTT或增量更新候选，不等价于证明整帧可复用。

## Linear与卷积运行时操作分账

| 范围 | 模块 | 调用 | dense标量MAC | 输入活动率 | 活动率加权MAC代理 |
|---|---:|---:|---:|---:|---:|
| bottleneck | 4 | 40 | 637009920000 | 0.119448 | 76089632256.000 |
| encoder | 71 | 710 | 5621391360000 | 0.113035 | 405311393472.000 |
| prediction | 4 | 40 | 2119680000 | 0.115720 | 245288974.000 |

dense标量MAC按运行时输出元素与weight fan-in计算。活动率加权MAC对Linear为连通度代理，
对带padding/stride的卷积不是精确SOP；它仍优于用全网单一firing rate缩放所有层。
