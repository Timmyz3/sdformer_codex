# All-Binary NTS/H60 P0 与量化验证结果

本文档记录 2026-06-19 完成的三项工作：

1. 对 all-binary NTS/H60 ft ep2 补跑硬件 P0 profiling；
2. 根据 P0 结果迭代硬件架构方案；
3. 对 all-binary NTS/H60 ft ep2 跑 H60 score/gate INT8 部署量化 valid825。

## 1. 本次实验对象

主 checkpoint：

```text
/root/private_data/work/sdformer_codex/SDformer/
neuron_experiments/H9_bipolar_self_attention/results/
date11full_all_binary_atlif_nts_stdlr_ft_ep29_ft5_bs8_20260618_141011_setsid/
checkpoint_epoch2.pth
```

主配置：

```text
/root/private_data/work/sdformer_codex/SDformer/
neuron_experiments/H9_bipolar_self_attention/configs/generated/
date11full_all_binary_atlif_nts_stdlr_ft_ep29_ft5.yml
```

结构审计：

| 项 | 结果 |
|---|---:|
| ATLIF wrapper | 105 |
| binary ATLIF | 105 |
| ternary ATLIF | 0 |
| H60/NTS attention | 12 |
| full encoder H60 | yes |

## 2. P0 Profiling 结果

输出目录：

```text
/root/private_data/work/sdformer_codex/SDformer/
neuron_experiments/H9_bipolar_self_attention/results/
nts11_hardware_p0_profiles/allbinary_nts_h60_ft_ep2_valid40
```

输出文件：

```text
activation_records.csv
atlif_activity.csv
h60_by_block.csv
h60_by_stage.csv
nts11_hardware_p0_profile.json
nts11_hardware_p0_profile.md
```

### 2.1 H60 / Shiftmax

40 个样本共记录 `480` 次 H60 调用：

```text
480 = 40 samples × 12 H60 blocks
```

这证明 all-binary 线仍然保持 full-encoder all12 H60/NTS attention，没有退回 mixed attention path。

| stage | calls | gate_entropy | top1_mass | top4_mass | effective_tokens | q_active | k_active | TTB1 empty | TTB2 empty | TTB4 empty |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 80 | 7.3398 | 0.0062 | 0.0247 | 162.00 | 0.00539 | 0.02602 | 0.5893 | 0.2790 | 0.0000 |
| 1 | 80 | 7.3398 | 0.0062 | 0.0247 | 162.00 | 0.00029 | 0.00183 | 0.8539 | 0.7378 | 0.0000 |
| 2 | 240 | 7.3398 | 0.0062 | 0.0247 | 162.00 | 0.00439 | 0.00478 | 0.7383 | 0.6301 | 0.0000 |
| 3 | 80 | 7.3398 | 0.0062 | 0.0247 | 162.00 | 0.00480 | 0.01019 | 0.7209 | 0.6449 | 0.0000 |

读法：

- Shiftmax gate 仍然接近均匀，不能讲成 top-k pruning。
- all-binary 的 Q/K activity 比 mixed NTS11 低很多，节能点更明确来自 binary event sparsity。
- TTB1/TTB2 的 empty ratio 明显升高，bundle skip 更有价值。
- TTB4 仍然基本没有 empty bundle，粒度过粗，不建议作为主调度粒度。

### 2.2 ATLIF 活性

| group | modules | activity | pos_rate | neg_rate |
|---|---:|---:|---:|---:|
| ternary | 0 | 0.000000 | 0.000000 | 0.000000 |
| binary | 93 | 0.044532 | 0.044532 | 0.000000 |

解释：

- 实际 forward 记录到 93 个 ATLIF 活动模块，安装阶段识别到 105 个 ATLIF wrapper。
- 所有被记录模块都是 binary mode，负事件为 0。
- binary activity 约 `4.45%`，低于 mixed NTS11bd ep19 的 binary activity `5.64%`，也避免了 ternary activity `15.14%` 带来的 sign rail 动态功耗。

### 2.3 Skip / Activation 存储

P0 表中原本给了 FP16 bytes 和 ternary packed bytes。对 all-binary 主线，更应该使用 1-bit packed bytes。

| kind | calls | elements | FP16 bytes | 2-bit packed bytes | 1-bit packed bytes |
|---|---:|---:|---:|---:|---:|
| stage_skip_predownsample | 120 | 464,486,400 | 928,972,800 | 116,121,600 | 58,060,800 |
| stage_skip_final | 40 | 33,177,600 | 66,355,200 | 8,294,400 | 4,147,200 |
| decoder | 160 | 1,526,169,600 | 3,052,339,200 | 381,542,400 | 190,771,200 |
| swin_block | 480 | 1,260,748,800 | 2,521,497,600 | 315,187,200 | 157,593,600 |

按每样本计：

| buffer | FP16 / sample | 2-bit / sample | 1-bit / sample |
|---|---:|---:|---:|
| S0/S1/S2 pre-downsample skip | 23,224,320 B | 2,903,040 B | 1,451,520 B |
| S3 final-stage retained output | 1,658,880 B | 207,360 B | 103,680 B |

硬件意义：

- all-binary 后 skip buffer 从 mixed 方案的 2-bit ternary packed 进一步降到 1-bit packed。
- S0/S1/S2 pre-downsample skip 仍是主要 skip 存储压力，但每样本只需约 `1.45 MB` 的 1-bit packed 容量。
- S3 retained output 很小，每样本约 `0.10 MB`。

## 3. 量化验证结果

生成配置：

```text
configs/generated/date11allbin_deploy_float_ref.yml
configs/generated/date11allbin_deploy_score_int8_mu_pow2_gate_int8.yml
```

量化配置：

| 项 | 设置 |
|---|---|
| hardware_quant_enabled | true |
| mu | `1/16 = 0.0625` |
| score step | `1/128` |
| score clamp | `[-2, 2]` |
| gate step | `1/128` |
| gate clamp | `[0, 2]` |

量化 valid825 输出目录：

```text
/root/private_data/work/sdformer_codex/SDformer/
neuron_experiments/H9_bipolar_self_attention/results/
date11allbin_deployment_quant_full825_20260619_ep2
```

valid825 对比：

| 方案 | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | energy_uj |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all-binary float ep2 | 1.4891 | 9.7785 | 0.5151 | 0.1924 | 0.0898 | 23.8206G | 5.1479% | 21045.91 |
| all-binary INT8 deploy ep2 | 1.4916 | 9.7762 | 0.5139 | 0.1921 | 0.0899 | 23.8241G | 5.1486% | 21048.98 |

差异：

| 指标 | 变化 |
|---|---:|
| AEE | `+0.0025` |
| AAE | `-0.0023` |
| total_spikes | `+0.0035G` |
| energy | `+3.07 uJ` |

结论：

- all-binary H60/NTS 对 score/gate INT8 和 `mu=1/16` 非常稳。
- INT8 部署近似没有实质精度损失。
- 这条线可以直接作为 DATE 硬件方案的 fixed-point deploy 主线，不需要保留 float score/gate 作为部署假设。

## 4. 硬件方案迭代

### 4.1 主线切换

旧主线：

```text
mixed NTS11
= 27 ternary ATLIF + 78 binary ATLIF + all12 H60/NTS
```

新主线：

```text
AllBinary-NTS/H60
= 105 binary ATLIF + all12 H60/NTS + INT8 deployable score/gate
```

建议论文命名：

| 层级 | 名字 |
|---|---|
| 实验名 | DATE11-Bin |
| 架构名 | UniBin-H60 |
| attention engine | Binary H60 Consensus Attention |
| 存储系统 | 1-bit Packed Event SRAM |

### 4.2 新数据流

建议画成以下流水线：

```text
Input voxel / feature
  ↓
Shared Binary ATLIF Encoder
  ↓
1-bit Event Tile Buffer
  ↓
Binary Q/K Projection Events
  ↓
Popcount Consensus Score Engine
  ↓
INT8 Score Quant + Shiftmax Token Gate
  ↓
INT8 Gate Quant
  ↓
Gated-K Binary Event Output
  ↓
1-bit Packed Skip / Activation SRAM
  ↓
Decoder Replay + Prediction
```

### 4.3 TX/SC engine 改法

mixed ternary 版本需要：

```text
positive rail
negative rail
same-polarity count
opposite-polarity count
single-active penalty
ternary decode/encode
```

all-binary 后改为：

```text
Q,K ∈ {0,1}
overlap = popcount(Q & K)
q_active = popcount(Q)
k_active = popcount(K)
mismatch = q_active + k_active - 2 * overlap
score = TX(overlap, q_active, k_active) + μ * SC(overlap, mismatch)
```

硬件单元：

- bitwise AND；
- popcount tree；
- small integer add/sub；
- score clamp/quant；
- Shiftmax；
- gate quant；
- gated-K multiplier 或 mask-scale unit。

### 4.4 删除或弱化的硬件模块

all-binary 主线下，不再需要把以下模块作为主路径：

- ternary sign rail；
- pos/neg 双 rail SRAM；
- ternary event packer/unpacker；
- opposite-polarity compare datapath；
- per-layer binary/ternary mode switch；
- mixed-format NoC packet。

这些可以放入 ablation/fallback，而不是主图。

## 5. 对 DATE 论文的影响

新的核心叙事：

> We convert the SDformer encoder into a unified binary-event H60 attention dataflow. All 105 ATLIF sites emit 1-bit events, all 12 encoder attention blocks share one H60 score-gate-output template, and the score/gate path is validated under INT8 deployment with negligible accuracy loss.

建议贡献点：

1. **Unified Binary Eventization**：105 个 ATLIF site 全部统一为 binary event。
2. **All-Encoder Binary H60 Attention**：12 个 encoder block 统一 H60/NTS 数据流。
3. **Popcount Consensus Score Engine**：binary Q/K 用 overlap/active/mismatch count 实现 TX/SC-compatible score。
4. **INT8 Deployable Shiftmax Gate**：score/gate INT8 + `mu=1/16` valid825 几乎不掉点。
5. **1-bit Packed Event SRAM and Skip Replay**：skip/activation buffer 统一 1-bit packed，控制和存储都比 mixed ternary 更简单。

## 6. 下一步

短期：

1. 把架构图正式改成 all-binary 主线：删除 ternary rail，突出 1-bit event SRAM 和 binary popcount consensus engine。
2. 更新总对比表：NB0、NTS07/09、mixed NTS11、all-binary float、all-binary INT8 deploy。
3. 导出 all-binary 的逐层 spike/hotspot 表，确认 downsample 是否仍是最高热点。

中期：

4. 跑 INT10/INT12 不再是必须项，因为 INT8 已经稳；可以作为 appendix sweep。
5. 继续做 ATLIF membrane/threshold 定点化，确认内部状态位宽。
6. 建立面积/功耗模型时，把主动态功耗口径改为 binary popcount、1-bit SRAM/NoC、ATLIF binary firing 和 Shiftmax INT8。

当前判断：

```text
all-binary NTS/H60 + INT8 score/gate
已经足够支撑 DATE 硬件主线。
```
