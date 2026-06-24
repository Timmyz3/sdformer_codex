# All-Binary NTS/H60 P0 Profiling 实测结果

**日期**：2026-06-19  
**对象**：`date11full_all_binary_atlif_nts_stdlr_ft_ep29_ft5` epoch 2  
**样本**：valid40  
**目的**：为 all-binary 硬件主线补齐 H60、TTB、ATLIF、skip buffer 的实测口径。

---

## 1. 运行信息

配置：

```text
/root/private_data/work/sdformer_codex/SDformer/
neuron_experiments/H9_bipolar_self_attention/configs/generated/
date11full_all_binary_atlif_nts_stdlr_ft_ep29_ft5.yml
```

checkpoint：

```text
/root/private_data/work/sdformer_codex/SDformer/
neuron_experiments/H9_bipolar_self_attention/results/
date11full_all_binary_atlif_nts_stdlr_ft_ep29_ft5_bs8_20260618_141011_setsid/
checkpoint_epoch2.pth
```

输出目录：

```text
/root/private_data/work/sdformer_codex/SDformer/
neuron_experiments/H9_bipolar_self_attention/results/
date11_hardware_p0_profiles/allbinary_nts_ft_ep2_valid40
```

运行状态：

| 项 | 结果 |
|---|---:|
| GPU | NVIDIA A800 80GB PCIe |
| 启动前 GPU 占用 | 0 MiB, 0% util |
| 结束后 GPU 占用 | 0 MiB, 0% util |
| samples | 40 |
| H60 调用记录 | 480 |
| ATLIF 安装模块 | 105 |
| ATLIF forward 记录模块 | 93 |
| H60/Shiftmax 模块 | 12 |

---

## 2. H60 / Shiftmax / QK 活性

| stage | calls | gate_entropy | top1_mass | top4_mass | effective_tokens | q_active | k_active | TTB1 empty | TTB2 empty | TTB4 empty |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 80 | 7.3398 | 0.0062 | 0.0247 | 162.00 | 0.00539 | 0.02602 | 0.5893 | 0.2790 | 0.0000 |
| 1 | 80 | 7.3398 | 0.0062 | 0.0247 | 162.00 | 0.00029 | 0.00183 | 0.8539 | 0.7378 | 0.0000 |
| 2 | 240 | 7.3398 | 0.0062 | 0.0247 | 162.00 | 0.00439 | 0.00478 | 0.7383 | 0.6301 | 0.0000 |
| 3 | 80 | 7.3398 | 0.0062 | 0.0247 | 162.00 | 0.00480 | 0.01019 | 0.7209 | 0.6449 | 0.0000 |

直接结论：

1. H60 调用数 `480 = 40 × 12`，证明 all-binary 仍然是全 encoder 统一 H60/NTS path。
2. Shiftmax gate 仍然接近均匀，不能作为强 token pruning 叙事。
3. Q/K 活性显著低于 mixed NTS11，尤其 S1/S2/S3 的 Q 活性极低。
4. TTB1/TTB2 的空 bundle 比例显著提高，all-binary 比 mixed NTS11 更适合 bundle skip。

和 mixed NTS11bd ep19 valid40 对比：

| stage | mixed q_active | all-binary q_active | mixed k_active | all-binary k_active |
|---:|---:|---:|---:|---:|
| 0 | 0.08670 | 0.00539 | 0.17082 | 0.02602 |
| 1 | 0.00869 | 0.00029 | 0.01268 | 0.00183 |
| 2 | 0.05240 | 0.00439 | 0.05461 | 0.00478 |
| 3 | 0.05547 | 0.00480 | 0.04074 | 0.01019 |

硬件含义：

- Binary H60 engine 的真实动态活动非常低，popcount tree 可以强依赖 event gating。
- TTB2 在 S1/S2/S3 都有很高跳过潜力，已经不是边缘优化。
- score engine 不需要为 ternary sign/mismatch 设计复杂前端，binary overlap/count 足够成为主线。

---

## 3. ATLIF 活性

| group | modules | activity | pos_rate | neg_rate |
|---|---:|---:|---:|---:|
| ternary | 0 | 0.000000 | 0.000000 | 0.000000 |
| binary | 93 | 0.044532 | 0.044532 | 0.000000 |

解读：

- 当前 forward 实测 93 个 ATLIF 记录模块，均为 binary。
- binary ATLIF 平均活性约 `4.45%`。
- neg_rate 为 0，硬件无需负事件 rail。
- 相比 mixed NTS11bd ep19 的 ternary activity `15.14%`、binary activity `5.64%`，all-binary 主线更稀疏且格式统一。

注意：安装阶段识别到 105 个 ATLIF wrapper，forward hook 记录 93 个模块。这个 105 vs 93 差异在 mixed NTS11 P0 里也存在，后续需要导出未调用模块列表，但不影响当前“全 binary event format”的判断。

---

## 4. Activation / Skip 存储

profiling 脚本原表仍叫 `ternary packed bytes`，其含义是 2-bit packed。all-binary 主线实际应使用 1-bit packed，因此这里额外给出 binary packed 估算。

| kind | calls | elements | FP16 bytes | 2-bit packed bytes | 1-bit packed bytes |
|---|---:|---:|---:|---:|---:|
| decoder | 160 | 1,526,169,600 | 3,052,339,200 | 381,542,400 | 190,771,200 |
| downsample | 120 | 232,243,200 | 464,486,400 | 58,060,800 | 29,030,400 |
| patch | 40 | 265,420,800 | 530,841,600 | 66,355,200 | 33,177,600 |
| prediction | 160 | 29,376,000 | 58,752,000 | 7,344,000 | 3,672,000 |
| resblock | 80 | 66,355,200 | 132,710,400 | 16,588,800 | 8,294,400 |
| stage_skip_final | 40 | 33,177,600 | 66,355,200 | 8,294,400 | 4,147,200 |
| stage_skip_predownsample | 120 | 464,486,400 | 928,972,800 | 116,121,600 | 58,060,800 |
| stage_x_out | 160 | 265,420,800 | 530,841,600 | 66,355,200 | 33,177,600 |
| swin_block | 480 | 1,260,748,800 | 2,521,497,600 | 315,187,200 | 157,593,600 |

每样本 skip buffer：

| skip | FP16 bytes/sample | 2-bit packed/sample | 1-bit packed/sample |
|---|---:|---:|---:|
| S0/S1/S2 pre-downsample skip | 23,224,320 | 2,903,040 | 1,451,520 |
| S3 final retained output | 1,658,880 | 207,360 | 103,680 |

硬件含义：

- all-binary 后 skip buffer 比 mixed ternary packed 再减半。
- S0/S1/S2 pre-downsample skip 仍是主要 skip 存储压力，但 1-bit packed 后每样本约 `1.45 MB`，已经比 FP16 小 16 倍。
- S3 retained buffer 每样本仅约 `0.10 MB`，可以作为独立小 buffer 处理。

---

## 5. 对硬件规划的更新

all-binary P0 profiling 支持以下设计决策：

1. **主线固定为全 1-bit event datapath**  
   不再为主线设计 ternary event SRAM、sign rail、pos/neg popcount。mixed NTS11 只保留为 appendix/fallback。

2. **Binary H60 engine 以 popcount overlap 为核心**  
   Q/K 活性低，尤其 S1/S2/S3 极低，应设计 event-gated popcount tree，而不是 dense token compute。

3. **TTB2 升级为默认调度策略**  
   all-binary 中 TTB2 empty ratio 在 S1/S2/S3 分别约 `73.8%/63.0%/64.5%`，比 mixed 更强。第一版硬件可以直接采用 TTB2，TTB1 作为上限分析。

4. **Shiftmax 只做统一 gate，不讲稀疏化**  
   gate 仍近似均匀，节能来自 event 稀疏和 bundle skip，而不是 Shiftmax top-k。

5. **1-bit packed skip SRAM 是核心贡献点**  
   all-binary 让 skip/activation/attention tile 全部统一成 1-bit event format，控制复杂度明显低于 mixed NTS11。

---

## 6. 下一步

必须补：

1. all-binary ep2 full valid825 P0 profiling，用于最终论文统计。
2. all-binary layer-category spikes，确认 downsample 是否仍是热点。
3. 105 vs 93 ATLIF coverage 列表，解释未 forward 记录模块。
4. all-binary H60 score/gate 定点 valid825。

可以并行写：

1. `05_module_interface_spec.md` 的 all-binary 新版。
2. UniBin-H60 四张架构图草图。
3. 面积/功耗/吞吐模型表。
