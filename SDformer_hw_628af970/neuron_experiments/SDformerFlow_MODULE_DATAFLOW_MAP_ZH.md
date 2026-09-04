# SDFormerFlow 模块数据流与可改动地图

这份文档把当前 `MS_SpikingformerFlowNet_en4` baseline 拆成“数据怎么走、模块在哪里、可以改什么、风险多大”。目标是后续做神经元、FFN、注意力、体素化、剪枝/量化时，不用每次重新在代码里摸路线。

baseline 只读目录：

- `third_party/SDformerFlow/`

实验改动目录：

- `neuron_experiments/`
- 每个实验应独立建文件夹、独立入口、独立 overlay/config/results，不直接改 baseline 骨架。

## 1. 总数据流

```text
DSEC event/flow dataset
  -> dataloader 读 chunk, mask, label
  -> voxel polarity split: pos/neg
  -> input normalization / optional spike_th
  -> MS_SpikingformerFlowNet_en4
      -> MS_Spikingformer_MultiResUNet
          -> spiking patch embed / PED frontend
          -> 4-stage 3D spiking Swin encoder
              -> stage i
                  -> Swin block j
                      -> QK attention branch
                      -> FFN / MLP branch
                  -> optional patch merging / downsample
          -> residual blocks at bottleneck
          -> 4 decoder blocks with encoder skip + previous flow skip
          -> 4 multi-resolution prediction heads
      -> sum over time dimension
      -> interpolate all flows back to original H,W
  -> supervised multi-scale flow loss
  -> AEE / AAE / SOPs profile
```

核心文件：

| 部分 | 文件 | 关键类/函数 |
|---|---|---|
| 训练入口 | `third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py` | data transform, forward, loss, optimizer |
| 顶层模型 | `third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_STSwinNet.py` | `MS_SpikingformerFlowNet_en4` |
| UNet 组织 | `third_party/SDformerFlow/models/STSwinNet_SNN/SNN_models.py` | `SpikingMultiResUNet.forward` |
| Swin encoder | `third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py` | `Spiking_SwinTransformer3D_v2`, `MS_Spiking_Swin_BasicLayer` |
| 基础脉冲模块 | `third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py` | `Spiking_neuron`, patch embed, decoder, pred |
| loss/metric | `third_party/SDformerFlow/loss/flow_supervised.py` | `Supervised_Flow_Loss`, `AEE`, `AAE` |

## 2. 训练前处理数据流

训练 loop 中每个 batch 是：

```text
chunk, mask, label
```

关键变换：

1. `chunk` 转 CUDA float。
2. 如果 `encoding=voxel` 且 `polarity=true`：

```python
neg = relu(-chunk)
pos = relu(chunk)
chunk = cat(pos, neg, dim=2)
```

3. 如果 `norm_input=minmax`，只对非零元素做 min-max 归一化。
4. 如果 `spike_th` 不为 `None`，输入被硬阈值二值化。
5. `pred_list = model(chunk)`。
6. `loss_function(pred, label, mask)`。

可改点：

| 可改方向 | 入口 | 风险 | 备注 |
|---|---|---:|---|
| 体素化/预处理稀疏 | dataloader 或 train loop 的 `chunk` 处理 | 中 | 之前 eval-only `spike_th` 对精度破坏很大，必须训练适配 |
| polarity 组织方式 | `pos/neg` split | 中 | 改动会影响 patch embed 输入统计 |
| 输入归一化 | `norm_input` | 高 | `std` 曾导致 SOPs 和 AEE 变差 |
| 输入动态时间步 | dataloader + model num_steps | 高 | PSN 权重依赖 T，不能随便改 `num_bins/num_steps` |

优先级判断：

- 不建议先靠 `spike_th` 硬二值化讲稀疏故事。
- 更适合后续做“事件密度感知动态时间步”或“learned voxel gate”，但要和训练范式一起设计。

## 3. 顶层模型输出流

顶层模型：

```text
MS_SpikingformerFlowNet_en4
```

核心 forward：

```text
input chunk
  -> sttmultires_unet.forward(x)
  -> multires_flow: list[T,B,2,h,w]
  -> sum(flow, dim=0)
  -> interpolate to original H,W
  -> {"flow": flow_list, "attn": attns}
```

可改点：

| 可改方向 | 位置 | 风险 | 说明 |
|---|---|---:|---|
| 时间聚合方式 | 顶层 `sum(flow, dim=0)` | 中 | 可试 learnable temporal aggregation / last-step aggregation |
| 多尺度输出权重 | loss 或 pred list 聚合 | 中 | 当前 loss 平均多个尺度 |
| 输出头量化 | prediction head 或最终 flow | 高 | 直接影响 AEE/AAE |

现阶段建议：

- 不优先改输出头。baseline 敏感性显示 pred/decoder 对精度非常敏感。

## 4. UNet 主干数据流

UNet 主体：

```text
MS_Spikingformer_MultiResUNet
```

由四段组成：

```text
encoders -> resblocks -> decoders -> preds
```

在当前 `MS` baseline 中，`encoders` 不是普通 CNN，而是：

```text
MS_spiking_former_encoder
  -> MS_Spiking_SwinTransformer3D_v2
```

decoder 流：

```text
x bottleneck
  -> concat encoder skip
  -> concat previous prediction if i > 0
  -> decoder upsample
  -> pred head
```

可改点：

| 可改方向 | 位置 | 风险 | 当前证据 |
|---|---|---:|---|
| encoder Swin block 神经元 | `encoders.swin3d.layers.*` | 中 | 最适合当前神经元实验 |
| decoder 神经元 | `sttmultires_unet.decoders.*.sn` | 高 | SOPs 高，但零化消融精度崩 |
| pred head 神经元 | `sttmultires_unet.preds.*.sn` | 高 | SOPs 高，但非常敏感 |
| bottleneck resblocks | `resblocks.*.sn*` | 中高 | 尚未重点验证 |

建议：

- 神经元稀疏优先动 encoder 内的 Swin FFN/attention。
- decoder/pred 可以做量化或轻量蒸馏，但不建议先做强剪枝。

## 5. Patch Embed / 前端

当前前端：

```text
MS_PED_Spiking_PatchEmbed_Conv_sfn
```

内部数据流：

```text
B, num_bins, P=2, H, W
  -> 重排为 T,B,C,H,W
  -> head: SpikingConvEncoderLayer
  -> conv: MS_SpikingConvEncoderLayer, stride=2
  -> residual_encoding: 2 个 MS_ResBlock
  -> proj: SpikingPEDLayer
  -> T,B,C,H/?,W/?
```

SOPs 证据：

- patch embed 的 `head.sn`、`proj.sn`、`residual_encoding.resblocks.*.sn*` 都在 top SOPs 层里。
- 但 patch embed 是输入前端，之前硬消融 `patch_embed_hot` 精度损失很大。

可改点：

| 可改方向 | 风险 | 建议 |
|---|---:|---|
| 轻量卷积/深度可分离卷积 | 中高 | 可以做，但需要从 checkpoint 适配 |
| 前端神经元自适应阈值 | 高 | 容易破坏输入表征 |
| 输入事件 gate | 中 | 比直接剪前端神经元更自然 |
| patch embed 量化 | 中 | 可作为后期硬件优化 |

结论：

- patch embed SOPs 很诱人，但不是第一阶段最稳靶点。
- 如果要改，建议做“温和 gate / 低比特量化”，不要直接三值/强剪枝。

## 6. 4-stage Swin Encoder 数据流

encoder stage 配置：

```text
swin_depths:    [2, 2, 6, 2]
swin_num_heads: [3, 6, 12, 24]
base channels:  [96, 192, 384, 768]
``` 

```text
sttmultires_unet.encoders.swin3d.layers.{stage}.swin_blocks.{block}.attn.*
sttmultires_unet.encoders.swin3d.layers.{stage}.swin_blocks.{block}.mlp.*
sttmultires_unet.encoders.swin3d.layers.{stage}.downsample.sn
```

可改点优先级：

| 位置 | 优先级 | 原因 |
|---|---:|---|
| stage0 FFN/MLP | 很高 | SOPs 占比高，消融耐受好 |
| stage0 attention proj | 高 | 消融耐受好，但 H6b 三值会变密 |
| Q/K 神经元 | 高 | H6 epoch11 证明可降 SOPs 并保持 AEE |
| stage1 FFN/MLP | 中高 | SOPs 仍可观，值得做 H7b/H7c |
| stage2 深层 FFN | 中 | 单层收益下降，部分层发放率高 |
| stage3 FFN | 中低 | 空间分辨率低，收益有限 |
| downsample | 中 | 可降 SOPs，但影响跨 stage 信息压缩 |

## 7. Attention 模块数据流

当前 attention 不是标准 `QK^T softmax V`。

模块：

```text
Spiking_QK_WindowAttention3D
```

内部流：

```text
x
  -> proj_sn
  -> linear_q -> BN -> sn_q
  -> linear_k -> BN -> add positional_encoding -> sn_k
  -> reshape heads
  -> att_token = sum(q)
  -> sn2_q(att_token)
  -> attn = k * att_token
  -> attn_sn
  -> proj -> proj_bn
  -> output
```

attention 内的关键神经元：

| 神经元 | 作用 | 当前建议 |
|---|---|---|
| `proj_sn` | attention 前置投影脉冲 | 不宜直接三值，H6b 中发放率暴涨 |
| `sn_q` | Q 分支脉冲 | 适合三值/自适应阈值 |
| `sn_k` | K 分支脉冲 | 适合三值/自适应阈值 |
| `sn2_q` | Q token 聚合后的脉冲 | 可作为下一步 attention gate |
| `attn_sn` | attention 输出脉冲 | 可试正向二值 gate，不建议先三值 |

已有实验结论：

- H6a：只 Q/K 三值 + FFN/downsample 二值，valid40 SOPs 降 `8.86%`，AEE 降 `1.97%`，AAE 升 `9.32%`。
- H6b：把 `proj_sn` 也三值后，`proj_sn` 发放率明显变密，SOPs 上升，不划算。

建议：

- Q/K 可以继续做三值自适应阈值。
- `proj_sn` 若要改，优先做二值自适应或 positive-only gate，不要普通 signed ternary。

## 8. FFN / MLP 数据流

每个 Swin block 里的 FFN/MLP：

```text
MS_Spiking_Mlp
  -> sn1
  -> fc1, 通道升维 C -> mlp_ratio*C
  -> BN
  -> sn2
  -> fc2, 通道降维 mlp_ratio*C -> C
  -> BN
```

注意：当前 MS 版本是“先神经元，后 Linear”，所以 `sn1/sn2` 直接决定进入大矩阵乘的活动率。

这就是 FFN 值得改的原因：

- `fc1/fc2` 操作数很大。
- 如果前面的 spike 更稀疏，硬件上可以直接少算很多 synaptic ops。

baseline SOPs 证据：

| 层 | SOPs 排名 | 说明 |
|---|---:|---|
| `layers.0.swin_blocks.0.mlp.sn2` | 6 | stage0 FFN 大头 |
| `layers.0.swin_blocks.1.mlp.sn2` | 7 | stage0 FFN 大头 |
| `layers.0.swin_blocks.1.mlp.sn1` | 10 | stage0 FFN 大头 |
| `layers.0.swin_blocks.0.mlp.sn1` | 11 | stage0 FFN 大头 |
| `layers.1.swin_blocks.*.mlp.sn2` | 17,18 | stage1 仍值得试 |
| `layers.2.swin_blocks.*.mlp.*` | 21,25,26,30... | stage2 多但单层收益分散 |

已有 H6a 改动：

```text
layers.0.swin_blocks.0.mlp.sn1
layers.0.swin_blocks.0.mlp.sn2
layers.0.swin_blocks.1.mlp.sn1
layers.0.swin_blocks.1.mlp.sn2
```

也就是说 H6a 已经替换了一部分 FFN，但只替换了 stage0 两个 block。

下一步建议：

| 实验 | 替换范围 | 训练策略 | 目的 |
|---|---|---|---|
| H7a | H6a 原范围 | `trainable: all` | 看全网适配能否修 AAE |
| H7b | stage0 + stage1 FFN | `atlif_only` | 扩大 FFN 稀疏，保持归因干净 |
| H7c | stage0 + stage1 FFN | `all` | 最有希望的性能/稀疏折中 |
| H7d | stage0 + selected stage2 hot FFN | `all` | 更强 SOPs 下降，风险更高 |

不要一上来替换所有 FFN。stage2/3 很多层单层收益小，且可能累计扰动精度。

## 9. Patch Merging / Downsample

stage 间下采样：

```text
MS_SpikingPatchMerging
  -> sn
  -> reduction Linear(4C -> 2C)
  -> BN
```

位置：

```text
layers.0.downsample.sn
layers.1.downsample.sn
layers.2.downsample.sn
```

特点：

- `sn` 位于 `reduction` 前，因此稀疏它可以直接降低后续 reduction 的有效 SOPs。
- 但它控制跨 stage 信息压缩，过强剪枝会影响深层特征。

已有 H6a 改动：

```text
layers.0.downsample.sn
layers.2.downsample.sn
```

为什么没放 `layers.1.downsample.sn`：

- 初版先选了 stage0/stage2 两个点做混合验证。
- 从 SOPs 排名看 `layers.1.downsample.sn` 也值得纳入 H7b/H7c。

建议：

- downsample 用二值自适应阈值，不要三值。
- 每个 stage 设置不同稀疏强度：stage0 稍强，stage1/2 稍弱。

## 10. Decoder 和 Prediction Head

decoder：

```text
skip concat encoder feature
skip concat previous prediction
MS_SpikingTransposeDecoderLayer
  -> sn
  -> deconv
  -> BN
```

prediction head：

```text
MS_SpikingPredLayer
  -> sn
  -> 1x1 conv to flow
```

SOPs 上它们很大：

- `decoders.3.sn` 是 SOPs rank 1。
- `preds.3.sn` 是 SOPs rank 3。

但消融显示：

- decoder/pred 一旦硬置零，AEE/AAE 直接崩。

建议：

| 方向 | 是否优先 | 原因 |
|---|---|---|
| 强稀疏剪枝 | 不优先 | 精度高度敏感 |
| 低比特量化 | 可以后期做 | 输出路径可硬件优化 |
| 蒸馏辅助轻量化 | 可以后期做 | 需要 teacher/student 训练 |
| 自适应阈值神经元 | 谨慎 | 稀疏强度必须很弱 |

## 11. Loss / Metric / Profile 数据流

训练 loss：

```text
pred_list["flow"] -> Supervised_Flow_Loss
```

当前 loss：

- 对每个尺度 prediction 计算 endpoint/mod loss。
- 多尺度平均。
- `lambda_ang=0`，训练时没有直接优化 AAE。

这解释了一个现象：

- H6a epoch11 AEE 比 baseline 好，但 AAE 变差。
- 因为训练目标主要是 AEE/mod loss，不直接约束角度误差。

可改点：

| 可改方向 | 风险 | 作用 |
|---|---:|---|
| 打开小权重 angular loss | 中 | 可能修 AAE |
| 对稀疏率加 schedule | 中 | 防止后期 Q/K 被压到近零 |
| valid loss 早停 | 低 | H6a epoch11 明显优于 epoch29 |
| profile 分层指标 | 低 | 每次实验都应记录 |

建议：

- 后续 H7 如果全网训练，建议加入 early stop 或 sparse schedule。
- 如果 AAE 是论文指标之一，可以试 `lambda_ang > 0` 的弱约束。

## 12. 当前 SOPs 热点总结

baseline valid40：

| 指标 | 数值 |
|---|---:|
| AEE | 1.5848 |
| AAE | 7.5012 |
| firing | 0.08496 |
| SOPs | 3.6219G |

SOPs 集中度：

| target set | SOPs share |
|---|---:|
| top 10 layers | 47.71% |
| top 20 layers | 70.85% |
| top 40 layers | 89.36% |

stage 汇总：

| group | SOPs share | 建议 |
|---|---:|---|
| encoder.swin3d | 76.43% | 主战场 |
| decoder | 13.06% | 敏感，后期弱改 |
| pred/other | 9.49% | 敏感，后期弱改 |
| transformer_block top-level | 1.02% | 非主战场 |

最值得优先改的层：

| 优先级 | 层组 | 原因 |
|---:|---|---|
| 1 | stage0 FFN/MLP | SOPs 高，消融耐受好 |
| 2 | stage0 attention proj | 消融耐受好，但不能直接三值 |
| 3 | Q/K neuron | H6 已证明稀疏有效 |
| 4 | stage1 FFN/MLP | SOPs 仍高，下一步扩展 |
| 5 | downsample | 可省 SOPs，但要弱稀疏 |

暂不优先：

| 层组 | 原因 |
|---|---|
| decoder hot nodes | 零化后精度崩 |
| pred heads | 直接决定输出 flow |
| patch embed hot nodes | 输入前端敏感 |

## 13. 实验模块化接入建议

为了不破坏 baseline，每个新实验应有：

```text
neuron_experiments/<EXP_NAME>/
  configs/
  entrypoints/
  overlay/
  tests/
  results/
  README.md
```

推荐接入方式：

1. baseline 入口仍可复用，但实验入口先把 overlay 路径插入 `sys.path`。
2. 通过 installer 在模型构建后按 module path 替换局部模块。
3. 配置中只写 target path、输出模式、稀疏强度、训练策略。
4. profile 后保存：
   - `sops_summary.json`
   - `layer_firing_rates.csv`
   - 训练日志
   - checkpoint 路径

这种方式的好处：

- baseline 完整保留。
- 每个实验能独立复现。
- 后续写论文消融表时，能清楚说明“只替换了哪些层”。

## 14. 后续最建议的改动路线

### 路线 A：修 H6a 的 AAE

问题：

- H6a epoch11 SOPs 和 AEE 好，但 AAE 差。
- 继续训练会过剪，Q/K 近乎静默。

建议：

1. H7a：H6a 范围不变，`trainable: all`，低 LR。
2. 加 early stop，重点保留 epoch8-14。
3. 试弱 `lambda_ang`。
4. Q/K 增加 target rate guard，防止发放率跌到接近 0。

### 路线 B：扩展 FFN 稀疏

问题：

- FFN 是 SOPs 大头，H6a 只替换 stage0。

建议：

1. H7b：stage0 + stage1 FFN，二值 ATLIF。
2. H7c：H7b + `trainable: all`。
3. stage2 只选 hot FFN，不全替换。

### 路线 C：注意力机制改进

问题：

- 当前 attention 不是标准 softmax，不能直接套常规注意力论文。

建议：

1. 保留 Q/K 三值。
2. `proj_sn` 用二值 adaptive gate 或 positive-only ternary。
3. `sn2_q` 可作为 token-level attention gate。
4. softmax 改进类论文要先确认是否适配当前 QK-token 机制。

### 路线 D：硬件友好剪枝/量化

建议顺序：

1. 先固定 H7 的稀疏神经元方案。
2. 再做结构化剪枝评分。
3. 最后做量化。

原因：

- 剪枝和量化应该基于最终激活分布，不应在神经元机制还不稳定时先做。

## 15. 一句话地图

如果只记一条：

```text
输入/patch embed 很敏感，decoder/pred 很敏感；
encoder.swin3d 是主战场；
其中 stage0 FFN、stage0 attention proj、Q/K、stage1 FFN 是最值得优先改的地方。
```

