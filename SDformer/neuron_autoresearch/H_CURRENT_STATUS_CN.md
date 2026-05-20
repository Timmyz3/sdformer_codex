# 当前神经元/注意力实验状态

更新时间：2026-05-20 01:03 UTC

## 当前正在跑什么

当前正在跑 H23/H24/H25 主队列；后面已经接上 H26 和自动全量 promotion：

```text
neuron_experiments/H9_bipolar_self_attention/results/h23_h24_h25_main_queue_20260520_005352.stdout
neuron_experiments/H9_bipolar_self_attention/results/h26_attention_revisit_then_promote_20260520_010221.stdout
```

当前子实验：

```text
h23b_h18c_lr1e5_target035_guard120_steps120
```

它从 baseline checkpoint 续训 120 train steps，然后立刻做 valid10 推理和
SOPs/firing 统计；如果满足 promotion 条件，会自动补 valid40。

## 已完成的 H22 结论

H22 是 `alpha-XNOR + Shiftmax` 直接注意力方案，固定 H18c 的模块设计，
只扫关键超参。它证明了三值发放是正常的：pos/neg 基本平衡，例如 H22c 的
ternary activity `0.0393`，pos `0.0204`，neg `0.0189`。

但 H22 没有追上 H9a 的 `3.0847G` SOPs。当前 valid40 最低 SOPs 约为：

| 实验 | AEE | AAE | SOPs | firing | 判断 |
|---|---:|---:|---:|---:|---|
| H22d target030 | 1.6326 | 8.0226 | 3.5037G | 0.08219 | SOPs 较低但精度偏差 |
| H22c target035 | 1.5938 | 7.7547 | 3.5149G | 0.08245 | 稀疏较好但仍不如 H9a |
| H22j sign value | 1.5724 | 7.5779 | 3.5919G | 0.08426 | 硬件友好性更好，精度尚可 |

因此 H22 不直接全量，转入 H23/H24/H25。

## 接下来准备跑什么

当前队列顺序：

```text
H23：低学习率 + 强稀疏反馈组合
H24：回到 H9a 的低 SOPs 替换范围，注意力换成 alpha-XNOR + Shiftmax，扫角度 loss/正则/LR/稀疏
H25：Q/K 固定三值，FFN 升维/降维、二值/三值、downsample 进行排列组合
```

H23 候选包括：

| 实验 | 基础方案 | 目标 |
|---|---|---|
| H23a | H18c | LR `1e-5` + target rate `0.040` |
| H23b | H18c | LR `1e-5` + target rate `0.035` |
| H23c | H18c | LR `1e-5` + target rate `0.040` + score scale `0.75` |
| H23d | H13v | LR `1e-5` + target rate `0.040` |
| H23e | H13v | LR `1e-5` + target rate `0.035` |

目标不是盲目全量，而是先看 valid10/valid40 的 AEE、AAE、SOPs、firing。只有精度接近
H9a/H13v 且 SOPs 明显下降的方案才考虑全量。

H24/H25 的目标是把 H9a 的 `3.0847G` 作为稀疏目标，而不是只和 baseline
`3.62G` 比。

## H26 和自动全量策略

用户提醒“降级的注意力也可以试短测，稀疏可以靠超参/正则/三值方案补”，因此
H26 已经加入队列。H26 不是新开随机分支，而是回收之前被降级的注意力：

| 实验 | 注意力 | 额外变化 | 目的 |
|---|---|---|---|
| H26a | `alpha_xnor_matrix_l1` | H9a 替换范围 + sparse040 | 回收 H18d，判断 L1 是否只是当时超参不合适 |
| H26b | `a2os2a_direct` | H9a 替换范围 + sparse040 | 回收 H18e，测试直接矩阵注意力 |
| H26c | `hamming_ternary_active_direct` | sparse040 | 回收 H21b 三值 Hamming |
| H26d | `hamming_binary_direct` | sparse040 | 硬件友好的二值 Hamming 对照 |
| H26e | `alpha_xnor_matrix_shiftmax` | value 改为 `sign` | 降低阈值实数乘法带来的硬件风险 |
| H26f | `alpha_xnor_matrix_l1` | FFN 改三值 | 测试高 SOPs FFN 三值是否能额外降 SOPs |
| H26g | `a2os2a_direct` | FFN 升维三值/降维二值 | 测试 FFN 内部细分替换 |
| H26h | `hamming_ternary_active_direct` | sparse035 | 更强稀疏反馈下复测 Hamming |
| H26i | `alpha_xnor_matrix_l1` | flow reg 降到 `0.0003` | 检查 AAE 是否受 flow 正则牵制 |

全量不再手动等待。新增脚本：

```text
neuron_experiments/H9_bipolar_self_attention/entrypoints/promote_best_rapid_screen.py
```

它会等 H23/H24/H25/H26 valid40 结果齐全后，按 AEE、AAE、SOPs 综合分选出
一个候选，自动生成 full 配置，从 baseline checkpoint 续训 30 epoch，并在结束后
跑 valid40 推理和 SOPs/firing 统计。也就是说，这一轮一定会选一个进入全量，
不会只停在短测。

## 当前保留的方案

| 方案 | 状态 | 原因 |
|---|---|---|
| H13v 低学习率修复 | 保留，重点观察 | valid40 AEE `1.4864`、AAE `7.2360`，精度很好，但 SOPs `3.6648G` 偏高 |
| H13w 强稀疏反馈 | 保留，作为稀疏支线 | valid40 AEE `1.5350`、AAE `7.5568`，SOPs `3.5815G`，比 baseline 低但还不够低 |
| H18c direct alpha-XNOR + Shiftmax | 保留，作为注意力主线 | valid40 AEE `1.5600`、AAE `7.7102`，SOPs `3.6372G`，可继续调超参 |
| H22 系列 | 正在跑 | 固定 H18c，只扫关键超参，判断是否能压 SOPs |
| H23 系列 | 已排队 | 组合低 LR 和强稀疏反馈，验证“超参组合”是否比单项更好 |

## 暂时摈弃或降级的方案

| 方案 | 处理 | 原因 |
|---|---|---|
| H13n 全量 epoch29 | 摈弃 | epoch29 AEE/AAE 明显恶化，且 SOPs/firing 不优 |
| H18e A2OS2A direct L1 | 降级 | valid10 精度很好，但 SOPs `4.3253G` 太高，不适合当前稀疏故事 |
| H18d alpha-XNOR L1 | 降级 | valid10 可用，但 SOPs `4.2328G` 太高 |
| H21 Hamming attention | 降级 | H21b valid40 AEE `1.6768`、AAE `8.4236`，精度不如 H13v/H13w |
| 40-step direct attention 结果 | 不作为否定依据 | 多个 direct attention 在 40-step 崩，但 120-step 恢复，40-step 会误判 |

## 判断标准

当前目标是：

1. 精度不要明显差于 H9a/baseline，优先看 AEE 和 AAE。
2. SOPs/firing 要有可讲的下降，理想方向是接近或低于 `3G SOPs`。
3. 如果 valid10 好但 valid40 不稳，不进入全量。
4. 如果精度好但 SOPs 高，只作为“精度机制可行”保留，不作为稀疏主线。
5. 如果 SOPs 低但 AAE 爆炸，只作为失败对照，不全量。

## 查看方式

当前进程：

```bash
ps -eo pid,ppid,stat,pcpu,etime,cmd | rg 'rapid_screen.py|train.py|profile_sops.py'
```

GPU：

```bash
nvidia-smi
```

H22 当前汇总：

```bash
cat neuron_experiments/H9_bipolar_self_attention/results/rapid_screen_h22_h18c_hparam_20260520_001015/summary.md
```

队列日志：

```bash
tail -f neuron_experiments/H9_bipolar_self_attention/results/h22_h23_main_queue_20260520_001015.stdout
```

## 记录规范

从 2026-05-20 起，新增 md 实验记录默认使用中文。已有英文历史记录不强制一次性全部翻译，但新增加的结论、状态、方案说明、保留/摈弃原因都用中文写清楚。
