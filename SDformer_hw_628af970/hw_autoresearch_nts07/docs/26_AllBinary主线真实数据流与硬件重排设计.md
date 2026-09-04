# AllBinary 主线真实数据流与硬件重排设计

## 1. 本文结论

这次设计必须以当前 all-binary NTS/H60 软件主线为准，而不是从一个抽象 SNN/Transformer accelerator 反推。严格读代码后，主线语义应写成：

```text
DSEC voxel 输入
-> SDFormerFlowAdapter 预处理/归一化
-> MS Spiking patch embedding
-> 4-stage Spiking Swin3D encoder
   每个 stage 内若干 Swin block
   每个 block: H60 token-wise Q/K selector attention + MLP + 两处 ADD 残差
   stage0/1/2 后接 patch merging downsample
-> 最深层 2 个 MS residual blocks
-> 4 级 decoder
   每级 concat 对称 encoder skip
   第 2-4 级还 concat 上一级 prediction
-> 每级 prediction head
-> 对时间维求和
-> 上采样到输入分辨率 flow list
```

因此，硬件主线不能只画一个 H60 attention core。正确的 DATE 图应该把 H60 core 放在 encoder block 内，同时显式保留：

- patch embedding 的事件重排和前端卷积；
- stage0/1/2 的 patch merging；
- block 内 attention residual 和 MLP residual；
- 最深层 residual blocks；
- decoder 的 encoder skip concat；
- decoder 的 prediction skip concat；
- 多尺度 prediction 和最终时间求和。

## 2. 主线配置约束

当前 all-binary 主线配置：

```text
configs/generated/date11full_all_binary_atlif_nts_stdlr_ft_ep29_ft5.yml
```

关键项：

- 模型：`MS_SpikingformerFlowNet_en4`
- 输入：`num_bins=10`，训练 crop `288x384`
- encoder depths：`[2, 2, 6, 2]`，共 12 个 Swin block
- heads：`[3, 6, 12, 24]`
- base channel：`96`
- window：`[2, 9, 9]`
- neuron：原始为 `psn`
- ATLIF wrapper：`enabled=true`，`output_mode=binary`，`threshold_mode=official_atlif`
- H60 attention：`enabled=true`，`mode=h60`
- H60 target blocks：stage0 block0/1，stage1 block0/1，stage2 block0-5，stage3 block0/1，共 12 个
- `value_mode=threshold`
- `center_scores=true`
- `preserve_mean=true`
- `bipolar_mu=0.05`
- `consensus_score_norm=head_dim`
- `hardware_mu_pow2_shift=4` 的 deploy eval 配置已验证基本无损；它不是当前训练主线 yml 的原生字段，论文中应写成“部署量化验证结果”，不能写成训练配置自带特性。

注意：`value_mode=threshold` 不是 H60 forward 中直接读取的控制开关。当前 H60 的 K 值之所以是 threshold-valued spike，是因为 `sn_k` 所在 ATLIF wrapper 输出 `{0, threshold}`，随后 H60 固定执行 `attn = k_orig * gate`。硬件若想只存 1-bit event，必须做 scale folding 或 descriptor lookup，并补等价性验证。

## 3. 输入与 patch embedding 数据流

输入在软件中是：

```text
x: [B, num_bins, polarity_or_channel, H, W]
```

在 `MS_PED_Spiking_PatchEmbed_Conv_sfn.forward()` 中：

1. 若输入 bin 数超过 `num_bins`，先截断到 `num_bins`。
2. `x.permute(0, 2, 3, 4, 1)` 把 bin 维挪到最后。
3. 重新组织为：

```text
new_event_reprs: [B, num_ch, H, W, num_steps]
x: [T, B, num_ch, H, W]
```

4. 实际顺序是 event reorder -> spiking head conv(stride=1) -> patch conv(stride=2) -> residual encoding -> SpikingPEDLayer projection，其中 projection 的空间 stride 来自 `patch_size[2:]=2`。

对当前配置：

```text
num_bins = 10
num_steps = 10
num_ch = num_bins * 2 // num_steps = 2
输入 crop = 288 x 384
patch_size[2:] = 2 x 2
MS patch conv 还有一次 stride=2
patch_embed 输出空间 = 72 x 96
输出: [T=10, B, C=96, H=72, W=96]
```

硬件含义：

- 前端不是普通图像 patchify，而是把 10-bin event voxel 重新映射为 10 个时间步、每步 2 个极性/事件通道。
- patch embedding 需要一个小型时序卷积前端；它仍是 dense/mixed precision conv，不属于 H60 binary attention core。
- 该模块输出后才进入 Swin3D encoder。

## 4. Encoder stage/block 真实数据流

`MS_Spiking_SwinTransformer3D_v2.forward()`：

```text
patch_embed: [T,B,C,H,W]
rearrange:   [B,C,T,H,W]
for each stage:
    x, out_x = layer(x)
    if stage in out_indices:
        outs.append(out_x before downsample)
return outs
```

每个 `Spiking_Swin_BasicLayer.forward()`：

```text
input x: [B,C,D,H,W]
rearrange -> [B,D,H,W,C]
for each Swin block:
    x = blk(x, attn_mask)
out_x = x before downsample
if not last stage:
    x_out = patch_merging(out_x)
return x_out, out_x
```

窗口分块还有一个容易漏掉的细节：Swin block 会按 window size 做 padding、window partition、window reverse，最后 crop 回原始 `D/H/W`。当前 crop 下 `W=96/48/24/12` 对 `window_w=9` 并不总是整除，因此 padding token 会进入 window 内的 H60 selector。由于 H60 patched forward 忽略 `attn_mask`，这些 padding token 不会被 mask 掉；硬件必须复现同样的 pad/partition/reverse/crop 语义，不能默认只处理无 padding 的有效 token。

stage 尺寸按当前 crop 推导：

| stage | blocks | C | HxW before downsample | heads | head_dim | window tokens |
|---|---:|---:|---:|---:|---:|---:|
| S0 | 2 | 96 | 72x96 | 3 | 32 | 2x9x9=162 |
| S1 | 2 | 192 | 36x48 | 6 | 32 | 162 |
| S2 | 6 | 384 | 18x24 | 12 | 32 | 162 |
| S3 | 2 | 768 | 9x12 | 24 | 32 | 162 |

每个 block 的真实结构：

```text
shortcut = x
x_attn = SSA/H60(x)
x = drop_path(x_attn) + shortcut
x_mlp = MLP(x)
x = x_mlp + x
```

代码中 `cnf="ADD"`，所以这里是 ADD residual，不是 AND/IAND。

## 5. H60 attention 真实语义

当前 H60 patch 发生在：

```text
bsa_attention.py::_qk_shiftmax_gate_forward()
```

H60 分支明确写成：

```text
tx_scores, sc_scores = _tx_sc_fusion_score_pair(q_orig, k_orig, cfg)
scores = tx_scores + mu * sc_scores
scores = scores - mean(scores over token dim)    # center_scores=true
scores = optional INT8 quant
gate = Shiftmax(scores over token dim)
gate = gate * n_tokens                           # preserve_mean=true
gate = optional INT8 quant
attn = k_orig * gate                             # NO carrier, K as value stream
```

这不是标准 QK^T V，也不是原始 QKFormer 的：

```text
att_token = sn2_q(sum_channel(Q))
attn = K * att_token
```

H60 当前是 **同 token Q/K selector**：

```text
q_orig: [T=2, Bwin, heads, spatial_tokens=81, head_dim=32]
_qkformer_token_q(q_orig): [Bwin, heads, T*81=162, head_dim=32]
k_orig: [Bwin, heads, T*81=162, head_dim=32]
score_i = TX(q_i, k_i) + mu * SC(q_i, k_i)
gate_i = Shiftmax(score_i over 162 tokens)
out_i = k_i * gate_i
```

硬件设计必须保持这一点：H60 不需要构造 `162x162` attention matrix。它只需要为每个窗口、每个 head、每个 token 计算一个 score，然后做 token 维 Shiftmax 和逐 token K gate。

## 6. TX/SC score 的硬件等价形式

`_tx_sc_fusion_score_pair()` 返回：

```text
tx_scores = _ternary_alpha_xnor_token_scores(q_orig, k_orig)
sc_scores = _signed_consensus_token_scores(q_orig, k_orig)
```

当前 all-binary 配置下，ATLIF 输出是 binary official ATLIF，软件仍通过 `_ternary_sign_ste()` 把非零值映射为 `+1`，没有负事件。因此硬件主线可以按 binary event 简化：

```text
q_event, k_event in {0, 1}

SC:
  sc = popcount(q_event & k_event) / head_dim

TX:
  same_nonzero = popcount(q_event & k_event)
  same_zero    = popcount(~q_event & ~k_event)
  opposite     = 0
  one_sided    = popcount(q_event xor k_event)   # 当前 single_active_penalty=0，可不进 score
  tx = (same_nonzero + alpha0 * same_zero) / head_dim

H60:
  score = tx + mu * sc
```

因为 `alpha0=0.02`，`mu` 在训练中可由 `_scheduled_bipolar_mu` 从 0 warmup 到 0.05；推理或没有 `_h9_global_step` 时使用最终 `mu=0.05`。deploy 可再通过 `_apply_hardware_mu_quant` 近似到 `1/16`。因此硬件主线采用定点时，应把训练调度和部署常数分开写：

```text
mu = 1/16    # 已验证
score INT8  # 已验证
gate INT8   # 已验证
```

如果未来回到 ternary 版本，则必须支持：

```text
q_pos/q_neg/k_pos/k_neg
same polarity
opposite polarity
silence/silence
single-active
```

但 all-binary DATE 主线不应把 ternary datapath 放进主图，否则会削弱“统一二值事件流”的故事。

## 7. ATLIF wrapper 的准确硬件解释

当前 H9 ATLIF wrapper 的 forward 是：

```text
h_seq = bias + W_psn @ x_seq
spike = OfficialATLIFSurrogate(h_seq, threshold)
out = spike.view(x_seq.shape)
```

它不是逐时刻膜电位递推：

```text
mem[t] = leak * mem[t-1] + input[t]
```

因此，硬件主线应该把 ATLIF 写成：

```text
PSN temporal mixer + calibrated threshold event emitter
```

而不是写成在线 adaptive LIF。硬件可复用一个 ATLIF wrapper engine：

```text
输入: T=10 或 T=2 的时间向量/feature stream
参数: 小型 T x T PSN temporal matrix、bias、threshold
输出: threshold-valued binary spike 或 1-bit event + scale descriptor
```

之前新增的 `binary_atlif_state_unit.v` 只能作为备选的 ATLIF-lite 神经元验证模块，不能作为当前 all-binary 主线的精确映射。

## 8. Decoder 和输出数据流

`MS_Spikingformer_MultiResUNet.forward()`：

```text
blocks = encoders(x)
x = blocks[-1]

for resblock in residual_blocks:
    x = resblock(x)

for i in 0..3:
    x = concat(x, blocks[3-i], dim=channel)
    if i > 0:
        x = concat(predictions[-1], x, dim=channel)
    x = decoder_i(x)
    pred_i = pred_i(x)
    predictions.append(pred_i)
```

`MS_SpikingformerFlowNet.forward()` 再做：

```text
for flow in multires_flow:
    flow = sum(flow, dim=0)       # 时间维求和
    flow = interpolate(flow, input H,W)
return flow_list
```

硬件含义：

- encoder skip 不是只有 stage0/1/2，而是 decoder 四级都会 concat 对称 encoder output，包括最深 S3。
- 另外第 2-4 级 decoder 还 concat 上一级 prediction。
- decoder/prediction 仍是卷积型数据流；如果 DATE 篇幅有限，可把它作为 “shared event-conv backend”，但不能从系统图删除。

## 9. 硬件数据流重排：保持语义的可行方案

### 9.1 统一窗口-Token-Time Bundle 调度

借鉴 Bishop 的 TTB 思路，但不要照搬其完整 SSA matrix attention。我们的 bundle 单元应定义为：

```text
UniBin-H60 Bundle =
  一个 Swin window 内
  T_window=2
  H_window=9
  W_window=9
  tokens=162
  head_dim=32
  对某一 stage/head 的 Q/K/K-value event packet
```

每个 bundle 执行：

```text
load Q/K events
-> popcount same_nonzero / same_zero / active
-> score = TX + mu*SC
-> center score
-> Shiftmax token gate
-> K * gate
-> reshape/merge window
```

这样保持软件 H60 的 token-wise 语义，同时获得 TTB 的存储局部性和跳过粒度。

### 9.2 分层复用，不按 PyTorch module 实例化

硬件不实例化 105 个 ATLIF 或 12 个 H60 attention。采用分时复用：

```text
1 个或少量 H60 token-score engine
1 个 Shiftmax token gate engine
1 个 gated-K vector engine
1 个 PSN/ATLIF threshold emitter engine
共享 event SRAM / scale descriptor SRAM / window line buffer
```

调度粒度：

```text
for stage in S0..S3:
  for block in stage.blocks:
    for window bundle:
      for head group:
        run H60 score/gate/K
    run projection + block residual
    run MLP + block residual
  if stage < 3:
    run patch merging
```

### 9.3 稀疏跳过只允许跳过 bundle，不改变数学

可用 profiling 的 TTB empty rate 做：

```text
if Q bundle empty and K bundle empty:
    gate/output 可直接走零或保守 bypass
else:
    正常计算
```

但由于 H60 有 `same_zero` 项，严格来说 “Q/K 都空” 仍可能产生 TX 的 silence/silence score。因此如果要跳过，必须满足以下二选一：

1. 软件消融确认去掉 same_zero 或空 bundle bypass 不掉精度；
2. 硬件不跳过 score，只跳过 Q/K event memory 读取，用常数 same_zero 快速路径生成 score。

推荐 DATE 主线采用第 2 种：**empty-bundle constant score fast path**，语义更稳。

### 9.4 Shiftmax 的硬件化

软件 Shiftmax：

```text
shifted = scores - max(scores)
numerator = 2^shifted
denominator = 2^ceil(log2(sum(numerator)))
gate = numerator / denominator
if preserve_mean: gate *= n_tokens
```

硬件实现：

- score 已是 INT8；
- `2^shifted` 用小 LUT 或移位近似；
- denominator 用 leading-one / ceil-log2；
- `preserve_mean=162` 可拆成 `*128 + *32 + *2` 或合并进后续 scale；
- gate INT8 已验证基本无损。

### 9.5 threshold-valued K 的简化

软件 `value_mode=threshold` 让 K 保留 ATLIF threshold-valued spike。硬件有两条路线：

1. **精确路线**：K SRAM 存定点 threshold-valued spike，H60 gated-K 直接乘 gate。
2. **压缩路线**：K SRAM 存 1-bit event，另存 per-layer/per-channel threshold descriptor，在 gated-K 阶段恢复或折叠 scale。

当前软件 ATLIF threshold 是 wrapper 级标量，不是已经验证过的 per-channel scale。DATE 主线推荐写成：

```text
event SRAM stores 1-bit K activity;
threshold descriptor is folded into gated-K scale.
```

这里的 descriptor 粒度应先按“wrapper/layer 级标量”落地；per-channel descriptor 只能作为未来优化策略。无论哪种压缩，都需要补 golden-vector 或 full825 部署验证。没有验证前，不能宣称完全等价。

### 9.6 gated-K 后是否立即二值化

代码里 H60 后段是：

```text
attn = k_orig * gate
x = reshape(attn)
attn = self.attn_sn(x)
x = self.proj(x)
```

也就是说，`attn_sn(x)` 的输出被赋给局部变量 `attn`，但后续 `proj` 仍然使用原来的 `x`。因此当前软件语义不是：

```text
K * gate -> binary spike -> projection
```

而是：

```text
K * gate -> projection
```

硬件设计里可以保留一个统计/可选发放器，但主线精确映射不能把 gated-K 输出强制二值化后再送 projection。

### 9.7 shifted-window mask

原始 Swin block 会向 attention 传入 `attn_mask`，但 H60 patched forward 开头直接 `del mask`。因此当前 H60 软件语义不使用 shifted-window attention mask。

硬件如果实现 mask，会和当前实验结果不一致。DATE 设计应写成：

```text
padding / cyclic shift / window partition / window reverse / crop follows Swin block scheduling;
H60 selector itself ignores the passed mask, matching current software.
```

## 10. 可借鉴论文范式

### Bishop, ISCA 2025

来源：https://arxiv.org/html/2505.12281v1

可迁移点：

- Token-Time Bundle 把空间 token 和时间步打包，提升权重/事件复用。
- stratifier 根据 bundle 稀疏度分流 dense/sparse core。
- error-constrained pruning 给 attention 中 Q/K/V/score/value 的裁剪设误差边界。
- AAC array 用 AND + accumulate 加速 spiking attention。

迁移到本项目：

- 使用 TTB 作为 H60 的调度与存储粒度。
- 使用 empty/low-density bundle fast path。
- 使用 dense/sparse 双路径处理 patch/MLP 与 H60 event score。
- 不采用其完整 SSA matrix attention，因为我们的 H60 是同 token selector。

### BSA / Bipolar Self-Attention

来源：https://openreview.net/forum?id=nG45z7lJ7D

可迁移点：

- ternary polarity 事件提高 spiking attention 表达能力。
- Shiftmax 用 shift-friendly 方式替代 softmax。
- attention score 可由 event agreement/conflict 构造。

迁移到本项目：

- H60 已经继承 Shiftmax 和 TX/SC 事件共识思想。
- all-binary 主线可把 ternary polarity datapath 降级为二值 same/zero popcount。
- DATE 叙事重点从“新 attention 算法”转到“把 H60 的 score-level fusion 映射成统一 event dataflow”。

### Spike-driven Transformer, NeurIPS 2023

来源：https://proceedings.neurips.cc/paper_files/paper/2023/file/ca0f5358dbadda74b3049711887e9ead-Paper-Conference.pdf

可迁移点：

- spike-driven paradigm 强调 zero input 不触发计算。
- binary spike communication 把乘法转为 mask/add。
- self-attention 可避免传统 softmax/QK 矩阵重计算。

迁移到本项目：

- all-binary 事件 SRAM + popcount score engine 符合这个范式。
- 但 SDformerFlow 有 U-Net decoder 和 optical-flow 多尺度输出，不能只套分类 ViT 的全局池化结构。

### STATA, ICML 2024

来源：https://proceedings.mlr.press/v235/zhuge24b.html

可迁移点：

- timestep-wise anchor token 用统一标准识别重要 token。
- token sparsification 同时考虑时间内和时间间 alignment。

迁移到本项目：

- 对 H60 的 162-token window，可做 “anchor-token aided bundle scheduling”。
- 不建议第一版 RTL 加动态 token pruning；可以作为 profiling/消融项，验证是否可跳过低 gate token。

### BESTformer, IJCAI 2025

来源：https://www.ijcai.org/proceedings/2025/0458.pdf

可迁移点：

- binary event-driven spiking transformer 支持 binary attention 和二值化权重/激活。
- 强调 event-driven 与 binarization 的硬件友好性。

迁移到本项目：

- 支撑 all-binary 主线作为 DATE 硬件候选。
- 但 BESTformer 是模型方法，不直接解决 SDformerFlow 的 windowed Swin/U-Net 数据流。

### Xpikeformer, 2024

来源：https://arxiv.org/html/2408.08794v1

可迁移点：

- 混合模拟/数字加速 spiking transformer。
- 利用 SNN 的时间动态和 transformer 结构做专用映射。

迁移到本项目：

- 可作为 related work，对比我们选择纯数字 popcount/shift/INT8 gate 的可综合路线。

## 11. 建议 DATE 硬件架构

建议命名：

```text
UniBin-H60: A Unified Binary Event Dataflow for Windowed Spiking Flow Transformers
```

核心模块：

1. **Voxel-to-TTB Frontend**
   - 完成 event voxel 到 `[T,B,2,H,W]` 的时间重排。
   - 执行 patch embedding conv。

2. **Shared ATLIF-PSN Event Emitter**
   - 执行 PSN temporal mixer + threshold emitter。
   - 输出 1-bit event 和可选 threshold scale descriptor。

3. **H60 Token Score Engine**
   - 输入同 token Q/K event。
   - 计算 same_nonzero、same_zero、active、SC、TX。
   - 输出 INT8 score。

4. **Shiftmax Token Gate Engine**
   - 对 162 token 做 max、LUT/shift exponent、pow2 normalization。
   - 输出 INT8 gate。

5. **Gated-K Stream Engine**
   - 执行 `K * gate`。
   - 支持 threshold descriptor folding。

6. **Event Conv/MLP Backend**
   - 处理 projection、MLP、patch merging、resblock、decoder conv。
   - 与 H60 core 共享 event SRAM 和 scale descriptor。

7. **Skip/Prediction Buffer**
   - 保存 S0/S1/S2/S3 encoder skip。
   - 保存上一尺度 prediction，用于 decoder prediction skip concat。

## 12. 图应该怎么画

### 图 1：真实网络数据流

```text
Voxel input
 -> Adapter preprocess
 -> MS_PED Patch Embed
 -> Stage0: [Block0 H60+MLP, Block1 H60+MLP] -> skip S0 -> Downsample
 -> Stage1: [Block0 H60+MLP, Block1 H60+MLP] -> skip S1 -> Downsample
 -> Stage2: [6 x H60+MLP] -> skip S2 -> Downsample
 -> Stage3: [Block0 H60+MLP, Block1 H60+MLP] -> skip S3
 -> 2 x ResBlock
 -> Decoder0 concat S3 -> pred0
 -> Decoder1 concat S2 + pred0 -> pred1
 -> Decoder2 concat S1 + pred1 -> pred2
 -> Decoder3 concat S0 + pred2 -> pred3
 -> time-sum + upsample -> flow list
```

### 图 2：H60 window dataflow

```text
Q/K PSN-ATLIF event emitter
 -> Q/K bit-pack per head_dim=32
 -> same/zero/active popcount
 -> TX score
 -> SC score
 -> score fusion: TX + 1/16 SC
 -> mean centering
 -> INT8 score
 -> Shiftmax over 162 tokens
 -> INT8 gate
 -> K threshold-valued event/value * gate
 -> projection path keeps gated-K value, not attn_sn output
 -> window reverse
```

### 图 3：硬件复用架构

```text
Global event SRAM
Scale descriptor SRAM
Window/TTB scheduler
  -> H60 score engine
  -> Shiftmax engine
  -> gated-K engine
  -> event conv backend
Skip/prediction buffer
DMA/output scaler
```

### 图 4：软件 module 数量 vs 硬件 engine 复用

```text
105 ATLIF wrapper modules -> 1 shared ATLIF-PSN event emitter engine
12 H60 attention modules  -> 1-2 shared H60 token-score engines
4 decoder levels          -> shared event-conv backend
4 encoder skips + 3 prediction skips -> skip/pred buffer
```

其中 `105` 来自当前 profiling/module coverage 口径；正式论文表格必须附模块覆盖列表，标明每个 ATLIF wrapper 属于 attention、MLP、patch/downsample、decoder 还是 prediction path。硬件复用是“一个或少量 engine 分时调度这些 wrapper”，不是证明网络中真的只有一个 ATLIF 算子实例。

## 13. 需要立刻修正的旧硬件文档/RTL口径

1. `binary_atlif_state_unit.v` 不应作为当前 all-binary 主线精确 ATLIF。
   - 它可以保留为 ATLIF-lite 备选单元。
   - 主线应改为 PSN temporal mixer + threshold emitter。

2. `binary_atlif_unit.v` 只能叫 threshold emitter。
   - 它可用于 PSN 后发放。
   - 不能独立代表 ATLIF wrapper。

3. TTB skip 不能简单认为空 bundle 直接输出零。
   - H60 TX 有 same_zero 项。
   - 空 bundle 最多走 constant score fast path。

4. `gated_k_unit` 必须支持 threshold-valued K 或 scale descriptor folding。
   - 当前 1-bit K event 乘 gate 只是压缩近似。

5. 系统图必须纳入 decoder skip 和 prediction skip。
   - 不能只画 encoder H60。

6. H60 主线不能画 shifted-window mask。
   - 当前 patched forward 忽略 `mask`。

7. H60 主线不能画 gated-K 后强制二值化再 projection。
   - 当前 `attn_sn` 结果没有进入 `proj`。

## 14. 最小补验证清单

1. **H60 golden vector**
   - 从 PyTorch dump `q_orig/k_orig/tx_scores/sc_scores/scores/gate/attn`。
   - RTL/Python fixed-point model 对齐每一步。
   - 必须覆盖 padding token、cyclic shift、window reverse、crop 后的位置映射。

2. **ATLIF wrapper golden vector**
   - dump `x_seq/W_psn/bias/threshold/spike`。
   - 验证 PSN temporal mixer + threshold emitter。

3. **threshold-valued K folding**
   - 对比 `K_threshold * gate` 和 `K_event * folded_scale * gate`。
   - 至少跑 valid40，再决定是否跑 valid825。

4. **empty bundle constant-score fast path**
   - 统计 Q/K empty bundle 时 same_zero score 分布。
   - 验证 constant fast path 对输出的误差。

5. **decoder skip buffer sizing**
   - 分别统计 S0/S1/S2/S3 skip 和 pred0/pred1/pred2 buffer。
   - 旧文档只统计 encoder skip 不够。

6. **module coverage audit**
   - 列出 105 个 ATLIF wrapper 的路径、所属 stage、是否在 attention/MLP/patch/downsample/decoder/pred。
   - 设计复用调度表必须基于这个列表。

7. **完整 block golden**
   - 覆盖 H60 输出后的 `proj/proj_bn/proj_sn`、attention ADD residual、MLP、MLP ADD residual。
   - 不能只验证 H60 score/gate，否则会漏掉 block 级残差和投影路径。

8. **部署量化差异审计**
   - 分别比较 float ref、score INT8、gate INT8、`mu=1/16`、K scale folding 的增量误差。
   - 每个开关都要有独立 valid40/valid825 结果，避免多个近似叠加后无法定位来源。

## 15. 当前硬件方向

推荐从现在开始把硬件主线改成：

```text
UniBin-H60 = PSN-ATLIF event emitter + token-wise H60 score/gate + event-conv U-Net backend
```

不要再把主线写成：

```text
stateful LIF neuron array + abstract binary attention core
```

前者贴合当前软件和实验结果；后者更像另一个模型，需要重新训练/验证。
