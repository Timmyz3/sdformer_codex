# 当前实验与后续计划 Review

日期：2026-05-20

## 一、当前正在跑的实验

当前 GPU 上跑的是自动 promotion 选出的全量实验：

```text
h23b_h18c_lr1e5_target035_auto_full_bs8_20260520_125502_setsid
```

入口：

```text
neuron_experiments/H9_bipolar_self_attention/entrypoints/promote_best_rapid_screen.py
```

配置：

```text
neuron_experiments/H9_bipolar_self_attention/configs/h23b_h18c_lr1e5_target035_auto_full_20260520_125502.yml
```

当前观察：

- 已保存 checkpoint 到 `checkpoint_epoch10.pth`。
- epoch8 valid loss 为 `1.1265`，比之前 I17e 那种 valid loss 9-10 的失败路线正常很多。
- 神经元统计显示三值负发放没有塌缩：`ternary_pos_mean` 和 `ternary_neg_mean` 基本接近。
- 但 `binary_activity_mean` 长期在 `0.13-0.14`，说明 FFN/downsample 二值模块仍然很密，是 SOPs 下不去的主要嫌疑。

## 二、主要结论

### 结论 1：当前 H23b 全量可以继续，但它大概率不是最终稀疏最优解

H23b 的 short valid40：

| 实验 | AEE | AAE | SOPs | firing |
|---|---:|---:|---:|---:|
| H23b short valid40 | 1.5535 | 7.6340 | 3.5324G | 0.08286 |
| H9a 参考 | 1.5044 | 7.6365 | 3.0847G | 0.07236 |
| baseline | 1.5848 | 7.5012 | 3.6219G | 0.08496 |

判断：

- H23b 相比 baseline 有一点 SOPs 下降，AEE 也略好。
- 但相比 H9a，SOPs 高了约 `0.45G`，稀疏故事不够强。
- 因此当前 full 更适合回答“direct alpha-XNOR/BSA 类注意力能否稳定训练”，不适合作为最终节能主线。

### 结论 2：H21-H27 的直接矩阵注意力目前都没打过 H9a 的 SOPs

我重新汇总了所有 H21-H27 valid40，最低 SOPs 结果如下：

| 实验 | AEE | AAE | SOPs | 判断 |
|---|---:|---:|---:|---|
| H22d target030 | 1.6326 | 8.0226 | 3.5037G | SOPs 最低，但精度偏弱 |
| H26h Hamming ternary sparse035 | 1.5970 | 7.7006 | 3.5144G | 稀疏尚可，但 AEE 弱 |
| H22c target035 | 1.5938 | 7.7547 | 3.5149G | 稀疏尚可，但没超过 H9a |
| H23b | 1.5535 | 7.6340 | 3.5324G | 当前全量，综合可观察 |
| H23e | 1.5034 | 7.3708 | 3.5863G | 精度最好，但 SOPs 偏高 |

这说明：直接矩阵注意力、strict BSA、Hamming 这些路线可以作为“注意力机制融合”实验，但暂时不是最强稀疏路线。

## 三、代码 Review 发现

### P1：promotion 评分把 SOPs 惩罚放得偏轻

位置：

```text
neuron_experiments/H9_bipolar_self_attention/entrypoints/promote_best_rapid_screen.py
```

当前评分大致是：

```python
score = AEE + 0.025 * AAE + 0.28 * max(0, SOPs - H9A_SOPs)
```

问题：

- H9a SOPs 是 `3.0847G`。
- H23b/H23e/H24/H26/H27 大多在 `3.5-3.7G`。
- 这个评分会优先选精度较好的候选，而不是稀疏最强的候选。

影响：

- 对“稀疏节能”主线来说，当前 promotion 会偏向“精度安全验证”，不够偏向“SOPs 突破”。

已做改进：

- 我已修改 promotion 脚本，让 full 结束后自动 profile 多个保存点，而不是只看最后 checkpoint。
- 现在会 profile `runtime.force_save_epochs` 里的 `9/19/29`，再加最新 checkpoint，生成 `best_profile_valid40.md`。

### P2：当前 full 只靠最后 checkpoint 判断会浪费训练

历史证据：

- H13n 在 epoch7 左右较好，epoch29 明显退化。
- 直接矩阵注意力训练过程中可能先好后坏。

问题：

- 原脚本 full 结束后只 profile latest checkpoint。
- 如果第 9 轮最好，会被最后一轮掩盖。

已做改进：

```text
promote_best_rapid_screen.py
```

新增：

- `checkpoints_for_profile`
- `--profile-epoch`
- 自动生成 `best_profile_valid40.md`

这能直接减少后续“全量跑完但评估错 checkpoint”的浪费。

### P2：FFN/downsample 二值模块没有 target_rate 控制

位置：

```text
neuron_experiments/H9_bipolar_self_attention/configs/h23b_h18c_lr1e5_target035_auto_full_20260520_125502.yml
```

当前多个 target group 都是：

```yaml
output_mode: binary
target_rate: null
activity_eta: 0.02 或 0.006
```

问题：

- Q/K 三值模块有 `target_rate=0.035`。
- 但 FFN/downsample 二值模块没有 target_rate，只靠 activity penalty。
- 当前日志里 `binary_activity_mean` 在 `0.13-0.14`，明显高于三值模块 `~0.03-0.04`。

影响：

- 这很可能是 H23/H24/H25/H26/H27 SOPs 迟迟压不到 H9a 水平的原因。

建议：

下一组不要继续盲扫注意力，而是新增二值模块 target_rate 控制：

```yaml
target_rate: 0.06
target_rate_eta: 0.02
max_threshold: 0.13
```

先对 FFN 二值模块做小范围 guard，观察 binary activity 能否从 `0.14` 压到 `0.07-0.09`，同时 AAE 不爆。

### P2：bsa_attention.py 的 mode 路由已经过长

位置：

```text
neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/bsa_attention.py
```

问题：

- `_qk_shiftmax_gate_forward` 里包含十几种 mode。
- `preserve_mean`、`row_sum`、`gate` 的后处理在多处重复。
- 继续加模式会降低可维护性，也容易引入“某个分支忘记乘/不该乘 preserve_mean”的错误。

建议：

暂时不要继续新增 attention mode。先把候选收敛到三类：

1. H9a carrier-preserving；
2. H18c direct alpha-XNOR；
3. H13v signed consensus。

等这三类的神经元稀疏控制稳定后，再重构 dispatch。

### P3：`_ternary_sign_ste` 太粗

位置：

```text
bsa_attention.py
```

当前：

```python
hard = x.sign()
return (hard - x).detach() + x
```

问题：

- forward 是 {-1,0,+1}，backward 是 identity。
- 当阈值幅值变大时，梯度尺度没有跟随阈值或活跃率调整。

建议：

后续可以试 clipped STE：

```python
mask = (x.abs() <= ste_clip).float()
return hard.detach() - (x * mask).detach() + x * mask
```

但这不是当前最高优先级，先解决 FFN 二值 activity 过高。

## 四、当前 full 是否建议继续

建议继续到 epoch9/19/29 三个保存点，至少拿到 epoch9 的 valid40 profile。

理由：

- 当前 valid loss 正常，不是 I17e 那种明显失败。
- epoch9/10 已经保存，后面可以评估。
- 但如果 epoch9 valid40 仍然是 `SOPs > 3.45G` 且 `AEE > 1.55`，后续 full 的论文价值有限。

建议停止条件：

| 条件 | 处理 |
|---|---|
| epoch9 valid40 AEE > 1.60 且 SOPs > 3.45G | 不再把 H23b 当主线 |
| epoch19 比 epoch9 AAE 上升超过 0.5 且 SOPs 不降 | 提前停止 |
| binary_activity_mean 一直 > 0.13 | 后续优先修 FFN target_rate |

## 五、更好的后续方案

### 方案 A：H28，FFN/downsample 二值 target-rate 控制

目的：

解决当前 SOPs 下不去的核心问题：binary FFN/downsample activity 高。

基础：

- 从 H23b 或 H9a-safe scope 继承。
- 保留当前表现较稳的 attention。
- 只给 binary target groups 加 target_rate。

建议配置：

```yaml
target_groups:
  - name: stage0_all_ffn_binary
    output_mode: binary
    target_rate: 0.08
    target_rate_eta: 0.02
    max_threshold: 0.13
  - name: stage1_half_even_ffn_binary
    target_rate: 0.07
    target_rate_eta: 0.02
    max_threshold: 0.13
  - name: stage2_half_even_ffn_binary
    target_rate: 0.06
    target_rate_eta: 0.02
    max_threshold: 0.13
```

预期：

- binary_activity 从 `0.13-0.14` 降到 `0.07-0.09`。
- SOPs 目标从 `3.5G` 压向 `3.1-3.3G`。

风险：

- FFN 二值发放被压太狠可能让 AAE 上升。
- 所以先 guard120 + valid40，不直接 full。

### 方案 B：回到 H9a carrier-preserving，叠加 H28 的 FFN 控制

目的：

直接冲 H9a 的 SOPs 基准，而不是继续在 direct attention 上消耗。

基础：

- attention 用 H9a `compat_qk_product`。
- Q/K 使用更平衡的 symmetric target-rate。
- FFN/downsample 加 target_rate。

这是我目前认为最有可能讲“稀疏节能”的路线。

### 方案 C：H13v 作为精度上限，不作为稀疏主线

H23e/H13v 的 short valid40：

| AEE | AAE | SOPs |
|---:|---:|---:|
| 1.5034 | 7.3708 | 3.5863G |

这个结果很适合说明：

- signed consensus 注意力对精度/AAE 有帮助；
- 但硬件稀疏收益不足。

因此它更适合作为 ablation，而不是最终主模型。

## 六、推荐下一步执行顺序

1. 等当前 H23b 至少出 epoch9 profile。
2. 如果 epoch9 profile 不明显优于 H9a，停止把 direct attention 当主线。
3. 新开 H28 guard：只改 FFN/downsample binary target_rate。
4. H28 先跑 H9a carrier-preserving 版本，再跑 H23b direct attention 版本。
5. 如果 H28 能把 SOPs 压到 `<=3.25G` 且 AAE `<=8.0`，再 full。

## 七、本轮已完成的非 GPU 工作

- Review 了当前 H23-H27 实验结果。
- Review 了 promotion 逻辑、H23b 配置、attention 代码和 ATLIF target-rate 逻辑。
- 修改 `promote_best_rapid_screen.py`，使 full 训练结束后自动 profile 多个 checkpoint，并生成最佳 checkpoint 排名。
- 运行语法检查通过。
- 用 `unittest` 跑 H9 相关测试，20 个测试通过。
