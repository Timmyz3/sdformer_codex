# Neuron Smoke Results - 2026-04-24

运行环境：`conda run -n sdformerflow`，`SDFORMER_USE_MLFLOW=0`。

配置：所有实验使用各自的 `configs/smoke.yml`。该配置只读取 `neuron_experiments/_templates/` 下 1 条 train split 和 1 条 valid split，用于验证训练入口、overlay 调用、forward/backward、checkpoint 保存。

| 实验 | 状态 | train loss | valid loss | max GPU mem GiB | checkpoint | log |
|---|---|---:|---:|---:|---|---|
| E0_psn_baseline | pass | 7.4170 | 6.5922 | 4.608 | `neuron_experiments/E0_psn_baseline/results/checkpoint_epoch0.pth` | `neuron_experiments/E0_psn_baseline/results/smoke_20260424_223825.log` |
| E1_exp_sn | pass | 6.7747 | 6.3207 | 4.878 | `neuron_experiments/E1_exp_sn/results/checkpoint_epoch0.pth` | `neuron_experiments/E1_exp_sn/results/smoke_20260424_223846.log` |
| E2_exp_atlif | pass | 9.4264 | 6.3016 | 6.015 | `neuron_experiments/E2_exp_atlif/results/checkpoint_epoch0.pth` | `neuron_experiments/E2_exp_atlif/results/smoke_20260424_223906.log` |
| E3_exp_lmh | pass | 9.8544 | 6.9244 | 6.203 | `neuron_experiments/E3_exp_lmh/results/checkpoint_epoch0.pth` | `neuron_experiments/E3_exp_lmh/results/smoke_20260424_223924.log` |
| E4_exp_tslif | pass | 17.5771 | 17.6853 | 10.749 | `neuron_experiments/E4_exp_tslif/results/checkpoint_epoch0.pth` | `neuron_experiments/E4_exp_tslif/results/smoke_20260424_223942.log` |
| E5_exp_tsn | pass | 10.8713 | 8.5419 | 4.870 | `neuron_experiments/E5_exp_tsn/results/checkpoint_epoch0.pth` | `neuron_experiments/E5_exp_tsn/results/smoke_20260424_224001.log` |
| F1_fused_adaptive_psn | pass | 8.1597 | 6.2164 | 5.599 | `neuron_experiments/F1_fused_adaptive_psn/results/checkpoint_epoch0.pth` | `neuron_experiments/F1_fused_adaptive_psn/results/smoke_20260424_224018.log` |
| F2_fused_lmh_atlif | pass | 6.9589 | 6.4206 | 7.343 | `neuron_experiments/F2_fused_lmh_atlif/results/checkpoint_epoch0.pth` | `neuron_experiments/F2_fused_lmh_atlif/results/smoke_20260424_224036.log` |
| F3_fused_adaptive_tslif | pass | 6.0360 | 6.3523 | 10.759 | `neuron_experiments/F3_fused_adaptive_tslif/results/checkpoint_epoch0.pth` | `neuron_experiments/F3_fused_adaptive_tslif/results/smoke_20260424_224053.log` |
| F4_fused_lmh_tslif | pass | 14.8995 | 15.3773 | 9.484 | `neuron_experiments/F4_fused_lmh_tslif/results/checkpoint_epoch0.pth` | `neuron_experiments/F4_fused_lmh_tslif/results/smoke_20260424_224113.log` |
| F5_fused_signed_hybrid | pass | 9.0083 | 6.8135 | 9.162 | `neuron_experiments/F5_fused_signed_hybrid/results/checkpoint_epoch0.pth` | `neuron_experiments/F5_fused_signed_hybrid/results/smoke_20260424_224132.log` |

备注：TSLIF 相关实验显存峰值明显偏高，后续跑 subset/full 时建议优先监控 E4、F3、F4、F5。
