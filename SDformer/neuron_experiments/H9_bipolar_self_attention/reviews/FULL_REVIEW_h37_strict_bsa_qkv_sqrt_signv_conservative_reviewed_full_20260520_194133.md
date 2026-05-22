# H37 strict BSA-QKV conservative 全量前 review

## 外部 review 判断

其他 agent 提到的三个问题基本成立：

1. 旧 strictBSA 如果没有独立 V 分支，并且没有按 `sqrt(d)` 缩放，不能称为严格 BSA 范式。
2. alpha-XNOR 原论文主线是二元逻辑；我们此前的“三值 + 负极性 + 冲突惩罚”是扩展变体，不能直接命名为同一算法。
3. A2OS2A 如果没有完整 Q/K/V 三路，只能算借鉴其思想，不能作为严格复现。

## 已修正内容

- `strict_bsa_qkv_shiftmax` 已改为三路注意力：
  - `sign(Q) @ sign(K)^T / sqrt(head_dim)`
  - Shiftmax 归一
  - 乘独立 `V` 分支输出
- 独立 `V` 分支在 attention patch 前注册，初始化方式为 `copy_k`，因此 optimizer 能看到 `linear_v/bn_v/sn_v` 参数。
- alpha-XNOR 和 A2OS2A 相关模式只保留为备选短测，不作为本次全量主线。

## 停掉 neuronfast 的原因

`h37_strict_bsa_qkv_sqrt_signv_neuronfast_reviewed_auto_full_20260521_023504` 前 3 个 checkpoint 表明稀疏增长太快，精度明显坍塌：

| checkpoint | Valid loss | SOPs(valid40) | Firing | AEE | AAE |
|---|---:|---:|---:|---:|---:|
| epoch0 | 1.2704 | 3.0882G | 0.072442 | 1.6799 | 8.5602 |
| epoch1 | 1.6052 | 2.8763G | 0.067472 | 2.0119 | 9.4730 |
| epoch2 | 1.8753 | 2.5644G | 0.060156 | 4.4291 | 27.7011 |

这说明该配置不是单纯训练未收敛，而是新神经元/阈值学习率过激，导致表达能力过早丢失。

## 本次全量选择

启动配置：

`neuron_experiments/H9_bipolar_self_attention/configs/h37_strict_bsa_qkv_sqrt_signv_conservative_reviewed_full_20260520_194133.yml`

关键设置：

- 从 baseline `checkpoint_epoch59.pth` 续训。
- 训练 30 epoch，batch size 8，workers 8，AMP 开启。
- 注意力：修正后的 `strict_bsa_qkv_shiftmax`。
- Q/K：三值 PSN + ATLIF。
- 高 SOP FFN/downsample：二值 official ATLIF。
- 学习率：
  - backbone/norm: `5e-7`
  - neuron: `1.5e-5`
  - threshold: `5e-6`

预期：相比 neuronfast，稀疏下降更慢，优先保住 AEE/AAE；若 full 前期验证仍连续恶化，再切换到 H36 stage02_highsop conservative 作为稳态 fallback。
