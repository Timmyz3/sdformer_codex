# M156 H67 ep35 非 Conv 成组稀疏独立打铁审阅 r1

## 裁决

M156 的中心负结论可靠：冻结 H67 ep35 checkpoint 在审计定义下，确实没有现成的 16/32-channel weight-structured skip；若要获得这类结构化跳过，**训练是必要条件，但不是充分条件**。

允许安全使用的表述是：

> 在冻结 H67 ep35 的 12 个 FFN pair 和 12 个 h60 attention 模块中，`fc1 row + fc2 matching column`、以及 `Q/K output row + projection input column` 的 16/32-channel 组，在 float 和指定 canonical per-output INT8 下均为 0 个 exact-zero group。现有 checkpoint 不能直接启用这两类 weight-structured skip。

不允许扩展成“没有任何结构化稀疏”“不存在 activation skip”“训练一定成功”或硬件/系统加速声明。

综合评分 **92/100，P0=0、P1=2、P2=4**；结论等级为 `PASS_CENTRAL_NEGATIVE_RELIABLE_ATTENTION_RECIPE_NEEDS_CLOSURE`。

## 独立复现

我进行了两条互相独立的复核：

1. 将 production 脚本 fresh 运行到本审阅目录，输出 JSON 与 ledger 分别和原结果逐字节相同：SHA256 `97d5ca14...`、`41e5d5e...`。
2. 新写独立重算器，不导入 M156 analyzer，只复用冻结 checkpoint loader；重新发现模块、量化、分组、排序并计算预算/Amdahl。

独立重算结果：

| 检查 | 结果 |
|---|---:|
| 模型中 FFN / pair ledger | 12 / 12，集合完全相同 |
| attention 模块 | 12 |
| stage population | 2 / 2 / 6 / 2 |
| FFN / attention group rows | 1,656 / 414 |
| composite key missing/extra | 0 |
| NumPy vs Torch canonical INT8 mismatch | 0 |
| float / INT8 exact-zero group | 0 / 0 |
| zero-flag mismatch | 0 |
| group-energy 最大相对误差 | 9.737×10⁻¹⁶ |

原脚本对已存在输出目录以 rc=1 fail closed，未覆盖 fresh 结果。

## 模块与 paired 语义

### FFN

`fc1.weight` 为 `[expanded, C]`，`fc2.weight` 为 `[C, expanded]`。M156 对同一 expanded-channel interval 取 `fc1[start:end, :]` 和 `fc2[:, start:end]`，因此 `fc1 output row + fc2 matching input column` 的 paired 语义正确。当前模型的 12 个 FFN 名称也与 cycle pair ledger 完全相同；stage 2 六对为 89,745,788 cycles，占 FFN 56.17%。

### Attention

当前 checkpoint 的 12 个模块全部是：

- mode=`h60`；
- `head_dim=32`；
- 没有独立 `linear_v`；
- h60 使用 K 作为 value carrier，再进入 `proj`。

因此对当前 H67 而言，Q/K output row 与 projection input column 的方向是拓扑一致的；32-channel 恰好是一整个 head，16-channel 是半个 head。

但合同中的训练建议还不完整。h60 的 score/K carrier 同时依赖 `bn_q/sn_q`、`bn_k/sn_k` 和 K positional encoding。训练 shared mask 时必须同步约束或删除这些状态，并明确 16-channel 半 head 是跨 head 共享 mask、固定 lane mask还是允许不等宽 head。只 mask `linear_q/linear_k/proj` 三张 weight 不能构成完整的可执行剪枝合同。该问题不推翻“当前没有 exact-zero weight group”的负结论，但会阻塞后续 attention pruning admission。

## Canonical INT8

M156 使用 output-channel maxabs/127、float32 scale、RNE ties-to-even、clip `[-127,127]`；它和 M41 的量化规则一致，只是扩展到二维 Linear weight。独立的 NumPy 和 Torch 实现对全部矩阵得到 0 mismatch，且未产生 -128。

逐 scalar 的 canonical INT8 zero fraction 为：

| 矩阵 | zero fraction |
|---|---:|
| FFN fc1 | 1.0448% |
| FFN fc2 | 1.1448% |
| attention Q | 1.0189% |
| attention K | 1.0420% |
| attention proj | 1.0339% |

这些零值分散，未组成任何完整 paired group。需要保留边界：这是 census quantizer，不是已导出、valid825-qualified 的非 Conv deploy payload；其他量化策略可能改变 INT8 零值，但不会改变本 checkpoint 的 float exact-zero 结果。

## 能量排序、预算与 Amdahl

分组 `weight_energy` 是 paired weight 的 L2²，不是芯片能量，也不是 loss/Fisher saliency。其排序和求和数学正确。FFN 每个 pair 内最小 group energy/组均值为 0.9160–0.9825，coefficient of variation 仅 0.00831–0.03283，支持“没有明显的低 weight-energy 长尾”这一限定结论。

| group | requested budget | actual group fraction | removed weight L2² | FFN cycle sensitivity | envelope sensitivity |
|---:|---:|---:|---:|---:|---:|
| 16 | 5% | 4.3478% | 4.2395% | 4.2523% | 1.01107× |
| 16 | 10% | 9.4203% | 9.2232% | 9.1753% | 1.02421× |
| 16 | 25% | 25.0000% | 24.5514% | 25.0000% | 1.06883× |
| 16 | 50% | 50.0000% | 49.4329% | 50.0000% | 1.14784× |
| 32 | 5% | 4.3478% | 4.2273% | 4.8825% | 1.01274× |
| 32 | 10% | 8.6957% | 8.5812% | 8.5046% | 1.02240× |
| 32 | 25% | 25.0000% | 24.6886% | 25.0000% | 1.06883× |
| 32 | 50% | 50.0000% | 49.6023% | 50.0000% | 1.14784× |

5%/10% 使用 `max(1, floor(groups×budget))`，所以 requested 与 actual 不完全相同；结果已显式报告 actual。25%/50% 在所有 pair 上整除，因而没有取整偏差。

Amdahl 复算正确，但它假设每个 pair 的 cycle 与删除通道比例连续线性缩放。它没有 group-specific activation/work、96-lane tile ceil、sn2/ATLIF 变化、地址流量或重新训练后的 event-rate。因此 1.06883×/1.14784× 只能保留为 uniform-channel compute sensitivity，不能作为选中低能量组的硬件性能预测。

另外，16 与 32 两种 partition 覆盖的是同一批权重；`exact_summary.total_weight_energy` 将两套 partition 相加而双计。当前合同没有把该总量用于结论，但消费者不得将其视作唯一权重能量。

## 问题分级

### P1

1. **Attention 训练合同未闭合。** shared Q/K/proj weight mask 尚未包含 h60 的 BN/SN、K positional encoding 与 head-aware mask 约束。
2. **选中 group 到硬件周期未映射。** Amdahl 使用 uniform fractional cycle；缺 group-specific activation、lane/tile、ATLIF、memory 和训练后 event-rate。

### P2

1. production 脚本从 FFN ledger 迭代，未另行断言模型中不存在 ledger 外 FFN；独立检查确认当前 checkpoint 恰好完整覆盖 12/12。
2. group16/group32 的 `total_weight_energy` 是重叠 population，合计双计。
3. 5%/10% 的 per-pair floor/max-one 令 actual budget 与 requested 不同；字段已披露，25%/50% 不受影响。
4. canonical INT8 尚不是非 Conv deploy payload 身份；只能按合同称 canonical census。

## 最终建议

- FFN：M156 足以启动 stage-2 paired-mask 训练 pilot；训练前冻结 valid825 AEE/event-rate 与 group-specific work trace。
- Attention：继续冻结 skip RTL；先补 h60 完整结构 mask 合同，再决定是否训练。
- 论文：可使用“0 exact group，因此必须显式训练”的负结果；不得使用 sensitivity 作为硬件或系统加速。

本审阅只写入 `results/m156_independent_hammer_review_r1_20260824/`，未修改 production、contracts 或 `docs/359`。审阅快照中 `docs/359` SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## 复核

```bash
/opt/anaconda3/envs/pytorch310/bin/python \
  results/m156_independent_hammer_review_r1_20260824/independent_recompute.py
python3 results/m156_independent_hammer_review_r1_20260824/validate_review.py
sha256sum -c results/m156_independent_hammer_review_r1_20260824/source_manifest.sha256
sha256sum -c results/m156_independent_hammer_review_r1_20260824/manifest.sha256
```

`independent_recompute.py` 默认拒绝覆盖已有结果；复跑前应复制本审阅目录或指定新的工作副本。
