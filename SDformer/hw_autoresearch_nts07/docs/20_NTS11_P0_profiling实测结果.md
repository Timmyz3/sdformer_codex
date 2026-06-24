# NTS11 P0 Profiling 实测结果

本文档记录 2026-06-17 对 NTS11 硬件主线补跑的 P0 profiling。目标不是重新评估精度，而是补齐硬件设计前必须知道的数据流活动口径：H60 attention、Shiftmax gate、Q/K 事件密度、TTB 可跳过性、ATLIF firing、skip/activation 存储规模。

## 结论先行

1. NTS11 的 full-encoder H60 attention 路径已经实测覆盖：12 个 H60/Shiftmax block 均被 hook，40 个样本得到 480 次 H60 调用，stage 调用比例为 2:2:6:2，和 `swin_depths=[2,2,6,2]` 一致。
2. H60 gate 当前接近均匀分布：每个 window 的 token 数为 162，effective tokens 约 161.97-162.00，top1 mass 约 0.0062-0.0065，top4 mass 约 0.0247-0.0256。这说明现阶段不能把硬件故事讲成“Shiftmax 强 top-k 稀疏选择”，更适合讲成“低成本统一 token gate + event carrier gating”。
3. Q/K event density 有明显 stage 差异。NTS11bd ep19 中，S0 Q/K activity 约 8.67%/17.08%，S1 约 0.87%/1.27%，S2 约 5.24%/5.46%，S3 约 5.55%/4.07%。这支持硬件按 stage 调整 event-lane 利用率和空 bundle 跳过。
4. TTB 可跳过性真实存在，但粒度要选对。按 1-token bundle，空 bundle 比例约 38.9%-51.6%；按 2-token bundle，空 bundle 比例约 12.3%-36.8%；按 4-token bundle，空 bundle 基本为 0。因此 TTB 跳过更适合以 1 或 2 token 为调度粒度，而不是 4 token 粗粒度。
5. ATLIF firing 口径已经补齐。NTS11bd ep19 中，ternary ATLIF 平均 activity 15.14%，正/负事件约 8.11%/7.02%；binary ATLIF 平均 activity 5.64%。NTS11bj ep2 的 ternary activity 更低，约 9.12%，说明 checkpoint 成熟度会影响事件率，最终论文表格应优先引用成熟 checkpoint 或 valid825 全量统计。
6. skip 连接口径已修正：`stage_skip_predownsample` 只对应 S0/S1/S2 的 downsample 前分叉；S3 是 `stage_skip_final`，它是最终 stage 输出跨 bottleneck 保留给 decoder i=0，不是 downsample 前 skip。

## 本次运行

代码插桩：

- profiling 脚本：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/entrypoints/profile_nts11_hardware_p0.py`
- H60 collector 插桩：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/bsa_attention.py`

运行环境：

- GPU：NVIDIA A800 80GB PCIe
- GPU 启动前状态：0 MiB used，0% util
- Python：`/opt/conda/envs/sdformerflow/bin/python`
- 工作目录：`/root/private_data/work/sdformer_codex/SDformer/third_party/SDformerFlow`
- SNN backend：`cupy`

输出目录：

- NTS11bj ep2 valid40：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/nts11_hardware_p0_profiles/nts11bj_ep2_valid40`
- NTS11bd ep19 valid40：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/nts11_hardware_p0_profiles/nts11bd_ep19_valid40`

## H60 / Shiftmax 统计

### NTS11bj ep2 valid40

| stage | calls | gate_entropy | top1_mass | top4_mass | effective_tokens | q_active | k_active | TTB1 empty | TTB2 empty | TTB4 empty |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 80 | 7.3397 | 0.0065 | 0.0256 | 161.98 | 0.08522 | 0.16648 | 0.3929 | 0.1246 | 0.0000 |
| 1 | 80 | 7.3398 | 0.0062 | 0.0247 | 162.00 | 0.01112 | 0.01508 | 0.5135 | 0.3843 | 0.0000 |
| 2 | 240 | 7.3398 | 0.0063 | 0.0249 | 162.00 | 0.03767 | 0.03777 | 0.4541 | 0.3250 | 0.0000 |
| 3 | 80 | 7.3398 | 0.0062 | 0.0249 | 161.99 | 0.04499 | 0.03710 | 0.4707 | 0.2881 | 0.0000 |

### NTS11bd ep19 valid40

| stage | calls | gate_entropy | top1_mass | top4_mass | effective_tokens | q_active | k_active | TTB1 empty | TTB2 empty | TTB4 empty |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 80 | 7.3397 | 0.0065 | 0.0256 | 161.97 | 0.08670 | 0.17082 | 0.3892 | 0.1234 | 0.0000 |
| 1 | 80 | 7.3398 | 0.0062 | 0.0247 | 162.00 | 0.00869 | 0.01268 | 0.5164 | 0.3678 | 0.0000 |
| 2 | 240 | 7.3398 | 0.0063 | 0.0250 | 161.99 | 0.05240 | 0.05461 | 0.3995 | 0.2482 | 0.0000 |
| 3 | 80 | 7.3398 | 0.0063 | 0.0249 | 161.99 | 0.05547 | 0.04074 | 0.3942 | 0.1853 | 0.0000 |

### 硬件读法

- `calls=480` 证明 full encoder 全部经过统一 H60 path，没有 mixed attention path。
- `gate_entropy≈log2(162)=7.33985`，说明 Shiftmax gate 当前不是强稀疏选择器，而是一个低成本归一化 token gate。
- `q_active/k_active` 是 TX/SC engine 的真实事件输入密度，应该作为 popcount/XNOR/adder tree 的动态功耗估计主口径。
- `TTB1/TTB2 empty` 可用于 clock gating 或 bundle skip；`TTB4 empty=0` 说明 4-token bundle 过粗，不适合作为主要跳过粒度。

## ATLIF 事件活性

| run | ternary modules | ternary activity | pos_rate | neg_rate | binary modules | binary activity |
|---|---:|---:|---:|---:|---:|---:|
| NTS11bj ep2 valid40 | 27 | 0.091151 | 0.048741 | 0.042410 | 66 | 0.052804 |
| NTS11bd ep19 valid40 | 39 | 0.151381 | 0.081134 | 0.070247 | 54 | 0.056408 |

当前 hook 记录到 93 个实际 forward 的 ATLIF 模块；模型安装阶段识别到 105 个 ATLIF wrapper。这个差异需要在 full valid825 或更细模块覆盖表里继续确认，可能来自某些 wrapper 在当前配置/路径下没有被调用，或 hook 分组只覆盖了实际事件输出模块。

硬件含义：

- ternary event propagation 不是极低 firing，成熟 checkpoint 可按 15% 左右活动率估算动态功耗。
- 正负事件比例接近，ternary 编码硬件不能只优化单极性正事件，需要完整支持 `-1/0/+1`。
- binary ATLIF 活性约 5%-6%，适合用单 bit event lane 和简单 event counter 做低成本传播。

## Activation / Skip 存储规模

以下表格来自 NTS11bd ep19 valid40。这里的 density 是张量非零/搬运口径，不等同于 ATLIF firing；用于 SRAM/NoC/skip buffer 容量估算时更可靠，用于动态 event 功耗时应使用上一节 ATLIF/H60 activity。

| kind | calls | FP16 bytes | ternary packed bytes | 每样本 FP16 | 每样本 ternary packed |
|---|---:|---:|---:|---:|---:|
| stage_skip_predownsample | 120 | 928,972,800 | 116,121,600 | 23,224,320 | 2,903,040 |
| stage_skip_final | 40 | 66,355,200 | 8,294,400 | 1,658,880 | 207,360 |
| decoder | 160 | 3,052,339,200 | 381,542,400 | 76,308,480 | 9,538,560 |
| swin_block | 480 | 2,521,497,600 | 315,187,200 | 63,037,440 | 7,879,680 |
| patch | 40 | 530,841,600 | 66,355,200 | 13,271,040 | 1,658,880 |
| downsample | 120 | 464,486,400 | 58,060,800 | 11,612,160 | 1,451,520 |
| stage_x_out | 160 | 530,841,600 | 66,355,200 | 13,271,040 | 1,658,880 |
| resblock | 80 | 132,710,400 | 16,588,800 | 3,317,760 | 414,720 |
| prediction | 160 | 58,752,000 | 7,344,000 | 1,468,800 | 183,600 |

skip buffer 重点：

- S0/S1/S2 downsample 前 skip 合计每样本 FP16 约 23.22 MB；若按 ternary 2-bit packed，约 2.90 MB。
- S3 final-stage output 每样本 FP16 约 1.66 MB；若按 ternary 2-bit packed，约 0.21 MB。
- 因此硬件图里不能把 skip 只画成一根抽象线；至少要区分 S0/S1/S2 pre-downsample skip buffer 和 S3 bottleneck-retained buffer。

## 已完成和未完成

已完成：

- H60 score/gate profiling：完成 NTS11bj ep2 valid40 与 NTS11bd ep19 valid40。
- Token-Time Bundle density profiling：完成 TTB1/TTB2/TTB4 empty/low-density 代理统计。
- ATLIF activity snapshot：完成 ternary/binary activity、正负事件率。
- module coverage：完成安装数量 105、实际 forward 记录 93、H60/Shiftmax 12/12 覆盖。
- mixed datapath 检查：H60 调用数与 2/2/6/2 encoder block 完全匹配，当前 NTS11 attention 主干没有混合原 QKFormer attention path。
- skip 连接口径：已修正为 S0/S1/S2 pre-downsample skip + S3 final-stage retained output。

仍建议补：

1. full valid825 profiling：用 NTS11bd ep19 或最终 NTS11bj 更成熟 checkpoint 跑一次完整 valid825，作为论文最终统计。
2. downsample hotspot ablation：目前只知道 downsample 张量规模，尚未分离 downsample 内部 ATLIF/Conv 的动态能耗热点。
3. ATLIF 105 vs 93 覆盖解释：导出逐模块调用表，标注未调用 wrapper 的模块路径和原因。
4. H60 gate 分布可视化：补 per-block gate entropy/top1/top4 直方图，验证是否所有 block 都近似均匀。
5. 能耗模型校准：把 H60 TX/SC event density、ATLIF firing、skip buffer bytes 接到统一能耗估计表，区分 compute、SRAM、NoC、control 四类。

## 对硬件方案的直接影响

本次 profiling 强化了 NTS11 作为 DATE 硬件主线的理由，但叙事要精确：

- 主要贡献不应写成“attention gate 高稀疏 top-k”，因为实测 Shiftmax gate 近似均匀。
- 更稳的故事是“统一 full-encoder H60 event attention dataflow”：所有 encoder block 共享 TX/SC score engine、single Shiftmax token gate、gated-K event output 和 ATLIF event propagation。
- 动态节能主要来自 Q/K ternary event coding、ATLIF firing 稀疏、TTB1/TTB2 bundle skip、ternary packed skip buffer，而不是 Shiftmax gate 本身产生强 token 剪枝。
- PyTorch 里的 105 个 ATLIF wrapper 不应映射成 105 套硬件实例；硬件上应解释为一组可复用 ATLIF event lane，在 layer schedule 中按模块参数/阈值上下文复用。
