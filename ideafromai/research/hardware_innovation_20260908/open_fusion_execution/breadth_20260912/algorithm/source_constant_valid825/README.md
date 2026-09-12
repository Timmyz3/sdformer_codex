# 两项 signed-PoT 源函数：fresh valid825

两套预先固定的新函数各重新载入、实际推理一次完整 valid825；无新增训练、扫参或验证择优。每臂 **825 帧 / 48,152,523 有效像素**与本地 upstream 复现 NB0 逐帧同文件、同有效像素数。NB0 帧均 AEE 为 **1.445352534681**，两臂均严格更低，按当前质量门通过。

| 新源函数 | 825 帧均 AEE | 像素加权 AEE | 相对 NB0 帧均 | 相对自身未量化 320 步父 |
|---|---:|---:|---:|---:|
| dense 两项 | 1.209053834 | 1.147157187 | −0.236298700 | +0.000724059 |
| lifting40 两项 | 1.235243345 | 1.175794005 | −0.210109190 | +0.009748630 |

完整集上 dense 质量更好；lifting40 在先前十帧上的改善没有扩展到完整 825 帧。两臂均使用同父、同 320 步恢复端点，随后仅投影 dense 的 `As_q16` 或 lifting40 的 `lifting_q12`；源指数、cutoff、RNE/sat24、配对/排列及消费者参数保持。参数为 [dense](../../source_constant_probe/dense/deployed_constants.npz) 与 [lifting40](../../source_constant_probe/lifting40/deployed_constants.npz)。AT-LIF `{0,theta}` 的固定幅度可静态折入权重。

每臂均观测到 825 次 helper 与 onepass BN 调用，实际 BN 域均为 192,000。以下源/消费者各有 60,825,600,000 个门位：

| 新源函数 | 源非零门位 | 消费者非零门位 | 实际 signed24 饱和项 |
|---|---:|---:|---:|
| dense 两项 | 2,988,989,350 | 684,738,059 | 0 |
| lifting40 两项 | 2,478,344,225 | 670,991,453 | 12 |

lifting40 的 12 项饱和全部发生在 `zurich_city_05_a_0191.npy` 的 `lift2b`，下界 7 项、上界 5 项；这些按既定 sat24 语义执行，已计入实际 AEE。其余已记录写回位置饱和为零。逐帧实际范围与饱和明细保留。

本评估沿用实际粗头 `preds.2` 的时间求和与 bilinear 480×640、`align_corners=False`；NB0 保留其原最终头。没有继承父或旧十帧质量，也不把运行耗时作为硬件速度。既有两 halo 整数核对不推广为整网 CPU/RTL 等价。

完整结果：[summary](summary.json)、[run](run.json)、[运行日志](run.log)。逐帧 AEE：[dense](dense/dense_frames.json)、[lifting40](lifting40/lifting40_frames.json)；逐帧 NB0 配对：[dense](dense/paired_NB0.json)、[lifting40](lifting40/paired_NB0.json)；活动：[dense](dense/activity_summary.json)、[lifting40](lifting40/activity_summary.json)；完整范围：[dense](dense/activity_ranges.json)、[lifting40](lifting40/activity_ranges.json)。

独立入口为 [evaluate_source_constants_valid825.py](../evaluate_source_constants_valid825.py)，调用方式为 `ROOT/env312/bin/python3.12 -u <runner> --root ROOT`。本次两臂已完成，入口会拒绝覆盖已有结果。
