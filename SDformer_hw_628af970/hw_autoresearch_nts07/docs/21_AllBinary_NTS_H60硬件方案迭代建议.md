# All-Binary NTS/H60 硬件方案迭代建议

本文档基于 2026-06-18 的 all-binary NTS/H60 valid825 结果，重新评估硬件主线。结论是：**all-binary ATLIF + all12 NTS/H60 + short fine-tune 应从“备选消融”升级为当前最硬件友好的主线候选**；原 mixed NTS11 仍保留为机制参考、消融对照和 fallback。

更新：2026-06-19 已完成 all-binary ft ep2 的 P0 profiling 和 H60 score/gate INT8 部署量化 valid825，结果见 `docs/22_AllBinary_NTS_H60_P0与量化验证结果.md`。结论进一步支持 all-binary NTS/H60 作为 DATE 硬件主线。

## 1. 关键结果

### 1.1 all-binary fine-tune 后的最佳结果

运行目录：

```text
/root/private_data/work/sdformer_codex/SDformer/
neuron_experiments/H9_bipolar_self_attention/results/
date11full_all_binary_atlif_nts_stdlr_ft_ep29_ft5_bs8_20260618_141011_setsid
```

配置：

```text
date11full_all_binary_atlif_nts_stdlr_ft_ep29_ft5.yml
```

结构审计：

| 项 | 结果 |
|---|---:|
| ATLIF wrapper | 105 |
| binary ATLIF | 105 |
| ternary ATLIF | 0 |
| H60/NTS attention | 12 |
| Shiftmax attention | 12 |
| checkpoint overlay keys | 210/0/0 |

valid825 ranking：

| rank | epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | energy_uj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 1.4891 | 9.7785 | 0.5151 | 0.1924 | 0.0898 | 23.8206G | 5.1479% | 21045.91 |
| 2 | 4 | 1.5131 | 9.8042 | 0.5149 | 0.1970 | 0.0934 | 23.9705G | 5.1803% | 21193.41 |
| 3 | 0 | 1.5385 | 10.0256 | 0.5210 | 0.2008 | 0.0960 | 24.1349G | 5.2158% | 21327.85 |
| 4 | 3 | 1.5480 | 10.0104 | 0.5253 | 0.2028 | 0.0958 | 25.4267G | 5.4949% | 22457.95 |
| 5 | 1 | 1.5767 | 10.1547 | 0.5312 | 0.2083 | 0.1013 | 25.0779G | 5.4196% | 22146.16 |

对 NB0 baseline：

| 方案 | AEE | AAE | total_spikes | energy_uj | firing |
|---|---:|---:|---:|---:|---:|
| NB0 ep59 | 1.4872 | 9.9300 | 44.0488G | 37638.01 | n/a |
| mixed NTS11bj ep2 | 1.5159 | 9.9611 | 29.0414G | 23032.66 | 6.2761% |
| all-binary NTS/H60 ft ep2 | 1.4891 | 9.7785 | 23.8206G | 21045.91 | 5.1479% |

相对 NB0：

- AEE 基本持平：`1.4872 -> 1.4891`，约 `+0.13%`。
- AAE 略好：`9.9300 -> 9.7785`。
- total_spikes 下降约 `45.9%`。
- energy 下降约 `44.1%`。

相对 mixed NTS11bj ep2：

- AEE 更好：`1.5159 -> 1.4891`。
- AAE 更好：`9.9611 -> 9.7785`。
- spikes 更低：`29.0414G -> 23.8206G`，约再降 `18.0%`。
- energy 更低：`23032.66 -> 21045.91 uJ`，约再降 `8.6%`。

## 2. 对硬件主线的判断

### 2.1 推荐主线切换

之前主线是：

```text
mixed NTS11 = 27 ternary ATLIF + 78 binary ATLIF + all12 H60/NTS
```

现在建议改为：

```text
all-binary DATE11 = 105 binary ATLIF + all12 H60/NTS
```

命名建议：

| 用途 | 名字 |
|---|---|
| 实验线 | DATE11-Bin 或 NTS11-Bin |
| 架构名 | UniBin-H60 |
| 论文贡献名 | Unified Binary Event H60 Attention Accelerator |
| 消融对照 | Mixed NTS11 |

### 2.2 为什么 all-binary 更适合 DATE 硬件

all-binary 相比 mixed NTS11 同时满足三个条件：

1. 精度不差：AEE 几乎和 NB0 持平，优于 mixed NTS11bj ep2。
2. 能耗更低：spikes 和 energy 都低于 mixed NTS11。
3. 硬件更简单：全网只有 `{0,+1}` 事件，不需要 `-1/0/+1` 的 sign rail、pos/neg 双 rail、ternary pack/unpack、符号一致/冲突的全复杂路径。

这对 DATE 论文很关键：审稿人更容易接受“统一 binary event datapath”而不是“虽然模块统一，但 Q/K ternary、非 QK binary、attention score 又有 TX/SC 的混合路径”。

## 3. 硬件架构应如何改

### 3.1 原 mixed NTS11 架构

原方案核心模块：

```text
binary/ternary ATLIF wrapper
→ ternary Q/K event coding
→ TX/SC consensus score engine
→ single Shiftmax token gate
→ gated-K event output
→ binary/ternary event propagation
```

这个方案可行，但要解释两套 event format：

```text
binary event: 1-bit
ternary event: 2-bit 或 sign+magnitude
```

### 3.2 新 all-binary 架构

建议改为：

```text
binary ATLIF event encoder
→ binary Q/K event coding
→ binary TX/SC-compatible score engine
→ single Shiftmax token gate
→ gated-K binary event output
→ binary event propagation / binary skip buffer
```

硬件数据格式统一为：

| 数据 | 格式 | 说明 |
|---|---|---|
| activation event | 1-bit | 0/1 |
| Q event | 1-bit | 0/1 |
| K event | 1-bit | 0/1 |
| skip event buffer | 1-bit packed | 替代 2-bit ternary packed |
| ATLIF state | fixed-point membrane | 内部状态仍可定点 |
| H60 score | integer / fixed-point | 由 popcount/count 派生 |
| Shiftmax gate | INT8/INT10/INT12 候选 | 单独量化 |

### 3.3 TX/SC engine 的简化

在 ternary 方案中，TX/SC 要处理：

```text
Q,K ∈ {-1,0,+1}
same polarity
opposite polarity
single active mismatch
silence
```

all-binary 后，Q/K 只有：

```text
Q,K ∈ {0,1}
```

因此 score engine 可以简化成：

```text
active_q = popcount(Q)
active_k = popcount(K)
overlap  = popcount(Q & K)
union    = active_q + active_k - overlap
score    = TX(overlap/active) + μ * SC(overlap, mismatch)
```

硬件上主要是：

- AND
- popcount tree
- small integer add/sub
- optional LUT 或 shift-scale

不再需要：

- 正负事件拆 rail；
- sign compare；
- opposite-polarity penalty；
- ternary decode；
- pos/neg balance 统计。

### 3.4 Shiftmax 的角色不变

P0 profiling 已说明 Shiftmax gate 接近均匀，因此不要把 all-binary 方案讲成 token pruning。它仍然是：

```text
统一 token gate / normalization unit
```

all-binary 之后，Shiftmax 更适合做定点量化：

- gate 输入 score 是 integer/popcount 派生；
- score 动态范围更容易界定；
- gate 输出可用 INT8/INT10/INT12 sweep 确认。

### 3.5 skip / SRAM 的改进

之前 mixed NTS11 中，S0/S1/S2 pre-downsample skip 如果按 ternary packed 是 2-bit。all-binary 后可以统一 1-bit packed。

直接影响：

| 项 | mixed 方案 | all-binary 方案 |
|---|---|---|
| event SRAM | 1-bit + 2-bit 混合 | 全 1-bit |
| skip buffer | binary/ternary 分格式 | 全 binary packed |
| NoC packet | 带 mode tag | 可省 mode tag 或只保留 descriptor |
| decoder replay | 需解码格式 | 统一 1-bit replay |

因此硬件图应把 “binary event SRAM” 放成核心贡献之一。

## 4. 面积、功耗、吞吐评估口径更新

### 4.1 面积

原 mixed 方案面积项：

```text
ATLIF binary lane
ATLIF ternary lane
ternary encoder/decoder
TX/SC ternary compare-popcount
Shiftmax
gated-K
mixed-format SRAM/NoC
```

all-binary 后面积项应改为：

```text
shared binary ATLIF lane
binary Q/K popcount score engine
single Shiftmax gate
gated-K binary modulator
1-bit packed event SRAM
TTB scheduler
layer descriptor controller
```

对论文表述：面积节省来自消除 ternary sign rail、2-bit event SRAM、ternary compare logic 和 mixed-format control。

### 4.2 功耗

功耗估算主口径改为：

```text
dynamic_energy =
  E_popcount_binary(Q,K activity)
  + E_shiftmax(num_windows, num_heads, tokens)
  + E_gatedK(K activity)
  + E_ATLIF(binary firing)
  + E_SRAM_1bit(read/write bytes)
  + E_NoC_1bit(packet traffic)
```

all-binary ft ep2 的 valid825 实测软件 energy 是 `21045.91 uJ`，应作为新的主线软件 energy 对照。

### 4.3 吞吐

吞吐不应只看 MAC 数，应按统一流水线分解：

```text
Patch/encoder event production
→ 12 个 H60 block schedule
→ decoder replay
→ prediction head
```

all-binary 的吞吐优势：

- event decode 更简单；
- SRAM/NoC payload 更小；
- popcount lane 可直接用 bit-parallel；
- 控制器不需要 binary/ternary mode 切换。

### 4.4 存储

存储评估口径更新为：

| buffer | mixed NTS11 | all-binary |
|---|---|---|
| activation event | 1/2-bit mixed | 1-bit |
| Q/K tile | ternary 2-bit | binary 1-bit |
| skip buffer | 1/2-bit mixed | 1-bit |
| gate buffer | fixed-point | fixed-point |
| descriptor | mode 字段复杂 | mode 字段可简化 |

### 4.5 控制复杂度

all-binary 最大收益之一是控制复杂度下降：

- 不需要按层切换 ternary/binary datapath；
- 不需要 pos/neg rail 对齐；
- 不需要 ternary overflow / sign handling；
- descriptor 只需要记录 shape、stage、block、threshold/gate 参数，而不是 event format 分支。

## 5. 论文故事应如何改

### 5.1 旧故事

```text
NTS11 把 mixed encoder attention 收敛为统一 H60；
Q/K 用 ternary event coding；
TX/SC consensus score；
single Shiftmax gate；
gated-K 输出；
ATLIF wrapper 统一 binary/ternary propagation。
```

### 5.2 新故事

```text
DATE11-Bin / UniBin-H60 把 SDformer encoder 收敛为统一 binary event H60 attention：
全网 105 个 ATLIF site 均输出 binary events；
12 个 encoder attention block 共享 H60/NTS score-gate-output 数据流；
binary Q/K 通过 popcount consensus score engine 生成 token score；
single Shiftmax gate 做统一 token modulation；
gated-K binary event output 继续在 decoder/skip buffer 中以 1-bit packed 格式传播。
```

建议贡献点：

1. **Unified Binary Eventization**：把原 SDformer 中复杂的 PSN/ATLIF/mixed neuron path 转成 105-site binary ATLIF event graph。
2. **All-Encoder H60 Binary Attention**：12 个 encoder block 全部使用同一 H60/NTS score-gate-output 数据流。
3. **Popcount Consensus Score Engine**：用 binary Q/K 的 overlap、active count、mismatch count 实现 TX/SC-compatible score。
4. **Single Shiftmax Gate with Integer Scores**：保留精度友好的 Shiftmax token modulation，并证明可做定点部署。
5. **1-bit Packed Event Memory and Skip Replay**：全网 event SRAM/skip buffer/NoC 统一为 1-bit packed，降低控制和存储复杂度。

一句话主张：

> We show that a full SDformer-style spiking optical-flow encoder can be converted into a unified binary-event H60 attention dataflow, matching baseline accuracy while reducing spikes/energy by about 46%/44% and eliminating ternary/mixed event hardware.

## 6. 是否还需要量化

需要，但优先级在 all-binary 主线确认之后。

推荐顺序：

1. **先确认 all-binary ep2 是最终主 checkpoint**：当前已经很强，但建议做一次 profile/审计归档。
2. **H60 score/gate 定点化**：复用之前 NTS11bj 的量化开关，优先跑 all-binary ep2 的 valid825。
3. **Shiftmax gate 位宽 sweep**：INT8 / INT10 / INT12。
4. **ATLIF membrane/threshold 定点化**：binary output 已经固定，内部 membrane 才是剩余模拟量。
5. **Conv/Linear 权重量化**：最后做；这会影响训练/部署更多，不应抢在 event datapath 量化前。

判断：

- 量化是实现优化，不是现在主线是否成立的前提。
- all-binary 已经解决最大的硬件结构复杂度；量化用于把 DATE 方案从“event-friendly”推进到“deployable fixed-point”。

## 7. 下一步最小任务

P0：

1. 对 `date11full_all_binary_atlif_nts_stdlr_ft_ep29_ft5` 的 ep2 跑硬件 P0 profiling：H60 gate、Q/K activity、TTB1/TTB2、ATLIF binary activity、skip buffer bytes。
2. 更新硬件主线表：把 all-binary ft ep2 加入 NB0 / mixed NTS11 / NTS07 / NTS09 对比。
3. 画新版架构图：删除 ternary rail，改成全 1-bit event SRAM + binary popcount consensus engine。

P1：

4. 跑 all-binary ep2 的 H60 定点部署 valid825。
5. 跑 Shiftmax INT8/INT10/INT12 sweep。
6. 导出逐层 binary firing / layer category spikes，确认 downsample 是否仍是热点。

P2：

7. 若 all-binary quant 也稳，正式把 mixed NTS11 降为 ablation/fallback。
8. 更新 related work 叙事，把 FireFly-T binary attention engine 的迁移价值提高，因为现在我们确实是 binary attention 主线。

## 8. 当前结论

硬件方案应改进，而且方向很明确：

```text
主线从 mixed binary/ternary NTS11
切换为 all-binary ATLIF + all12 NTS/H60 + 1-bit event datapath。
```

这条线同时更准、更省、更简单。mixed NTS11 仍然有价值，但现在更适合作为“为什么我们最终选择 all-binary”的对照：它证明统一 H60 attention 有效，而 all-binary fine-tune 进一步证明 ternary Q/K 不是必要硬件负担。
