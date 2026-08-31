# Local5 硬件特征 Profile

- 配置：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50_hardware_order_q7q17_deploy.yml`
- checkpoint：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50_20260812/checkpoint_epoch44.pth`
- samples：`100`
- evidence level：`post_g0`
- 评估协议：`{'resolution': [480, 640], 'crop': None, 'window_size': [2, 15, 15], 'pretrained_window_size': [2, 9, 9], 'tokens_per_window': 450, 'remap': 'v1', 'bn_policy': 'no_running', 'bn_modules_changed': 78, 'eval_batch_size': 1, 'num_workers': 0}`
- 模块数量：`{'ATLIFTernaryPSN': 105, 'ShiftmaxAttention': 12}`
- 权重加载：`{'checkpoint': '/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50_20260812/checkpoint_epoch44.pth', 'checkpoint_overlay_keys': 210, 'model_overlay_keys': 210, 'missing_count': 0, 'unexpected_count': 0, 'overlay_missing_count': 0, 'overlay_unexpected_count': 0, 'missing_sample': [], 'unexpected_sample': [], 'remap': 'v1'}`
- ATLIF 阈值训练/部署语义：`{'threshold_modes': ['official_atlif'], 'homeostatic_freeze_after_step': 1224, 'homeostatic_update_frozen_after_boundary': True, 'optimizer_gradient_freeze_enabled': False, 'optimizer_threshold_lr': 5e-06, 'configured_min_threshold': 0.001, 'configured_max_threshold': 2.0, 'official_atlif_runtime_clamp_applied': False, 'inference_threshold_source': 'checkpoint_static_parameter', 'statement': 'threshold_freeze_after_step stops only the separate homeostatic threshold_update path; optimizer threshold gradients remain active unless freeze_threshold_grad_after_step is true. official_atlif does not apply the configured min/max runtime clamp. Inference uses the threshold parameter stored in the checkpoint.'}`
- 数值边界：这是绑定最终 config/checkpoint 身份的 post-G0 profile；
  ordered trace 仍属于 workload/transaction 证据，不自动等价于 RTL、PPA
  或 full-encoder speedup。

## 总结

| 指标 | 数值 | 证据用途 |
|---|---:|---|
| 四方向 K XOR lane density | 2.825310% | RCSD |
| 四方向 exact-K edge | 79.833584% | Prosperity exact reuse |
| delta count p50/p95/p99 | 0/7/11 | direct/delta 模式 |
| QFSA-W2 score cycle reduction | 67.586440% | joint direction residual 模型 |
| QFSA-W4 score cycle reduction | 69.521924% | joint direction residual 模型 |
| QFSA-W4 vs 4xW1 cycle reduction | 7.973113% | 同总residual宽度强基线 |
| XBF-QFSA vs 4xW1 cycle reduction | -1.441645% | XOR-bank蝶形分配强候选 |
| XBF-QFSA-T8 vs 4xW1 cycle reduction | -34.817204% | 可综合threshold router |
| QFSA-W8 score cycle reduction | 71.132240% | joint direction residual 模型 |
| source-resident 理论 K-bit read 减少 | 78.873239% | line buffer |
| source-resident 活动 K-lane read 减少 | 78.887815% | source multicast |
| 有效 gate=0 | 0.000000% | 预修复 gate 指标 |
| gate cardinality mean/p95 | 1.0318/1 | Shiftmax5/term |
| multiplicity mean/p95 | 1.2479/2 | MFEP |
| offset term / active edge product | 10.846035% | 低风险 DCTF 基线 |
| MFEP term / active edge product | 4.475969% | 多重集折叠 |
| DiSEP source-gate-lane term / active edge product | 32.594034% | source-major projection |
| active source 比例 | 16.156283% | source 调度占用 |
| active source gate cardinality mean/p95 | 1.5141/2 | DiSEP product reuse |
| all-source gate cardinality mean | 0.2446 | 含空 source，不能与上一行混用 |
| DQFS row value product reduction | 46.587141% | `(lane,gate,weight_epoch)`跨source精确复用 |
| DQFS row value keys mean/p95 | 6.9291/44 | lane-local目录+term SRAM |
| DQFS row terms p95/max | 88/702 | 双context容量 |
| DQFS lane gate cardinality p95/max | 4/8 | 目录way数 |
| DQFS value chain length p95/max | 5/15 | product驻留长度 |
| DQFS 6-way overflow groups/terms | 135/2300 | exact RAW fallback压力 |
| unsafe set term / active edge product | 2.542873% | 仅显示错误去重上界 |
| gate 二进制非零位均值 | 1.0409 | shift-add/CSD 前筛 |

## 分 Stage

| Stage | XOR density | exact-K | QFSA-W4 vs serial-direct | QFSA-W4 vs 4xW1 | XBF-oracle vs 4xW1 | XBF-T8 vs 4xW1 | active K read reduction | MFEP term ratio | multiplicity p95 | MFEP term/window-head p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | 1.910262% | 83.448609% | 71.443090% | 6.329417% | -2.134679% | -32.725550% | 78.911766% | 4.226886% | 2 | 168 |
| S1 | 0.528090% | 94.766733% | 76.483673% | 1.950165% | -1.401391% | -12.706036% | 78.858185% | 7.227353% | 2 | 88 |
| S2 | 4.528164% | 70.770125% | 65.079465% | 10.525081% | -0.987596% | -41.421795% | 78.886889% | 4.224028% | 2 | 293 |
| S3 | 7.086888% | 55.937068% | 58.061326% | 13.360412% | -0.748231% | -50.966472% | 78.863856% | 4.934661% | 3 | 406 |

## 解释边界

- `topology_k_read_reduction` 比较 query-major 五邻域重复取 K 与每个 source K
  在行缓冲中读取一次；尚未加入 SRAM 端口、halo 和控制能量。
- `offset term` 按 self/N/S/E/W 分开，目的 bitmap 无重复，最容易复用现有 DCTF。
- `MFEP term` 使用 `(gate,lane,multiplicity)`，保持 Local5 多重集语义。
- `source_gate_lane_terms` 使用 `(source token, final gate, lane)`，
  delivery 必须与 active edge-lane product 守恒；用于 DiSEP 强基线。
- `source_gate_cardinality` 默认只在 active source 上统计；报告另列
  all-source 均值与 active-source 比例，禁止通过排除空 source 夸大收益。
- `DQFS row value` 使用 `(lane,gate)`值键，默认同一profile记录内
  `weight_epoch`不变；它不包含source，source只决定destination链。
- DQFS reduction只统计product生成机会，不等于周期或能耗降低；
  目录、term SRAM、重排反压和fallback必须由RTL/PPA计入。
- `QFSA-W* score cycle reduction` 枚举四方向 direct/residual 选择，
  口径为一个共享32-lane anchor/direct popcount加W-lane带方向残差后端；
  不包含 compactor、SRAM、Shiftmax、projection 和控制周期。
- `unsafe set term` 丢弃 multiplicity，只是错误 OR 去重能达到的乐观下界，
  不允许作为可实现结果。
- 本报告是 workload profile，不是 cycle、PPA 或端到端加速结果。
- JSON 已记录 ordered dataset sample-key manifest 的 SHA256：`32138614734d4ca9e14253ba9863f554ddb4d57552531e0ed153652d1acda125`；跨模型比较时必须核对该 hash。
