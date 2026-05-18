# 三条主线后续方案：AAE 控制、注意力融合、负发放控制

日期：2026-05-18

本文整理当前 H 系列实验、autoresearch 调查和最新 I17 运行结果，并围绕目前暴露出来的三个问题给出下一阶段方案：

1. AAE 控制；
2. 注意力机制融合；
3. 负发放被压制，以及如何把负发放真正用起来。

所有方案都遵守目前的工程约束：不修改 `third_party/SDformerFlow` baseline 文件夹。新增代码放在 `neuron_experiments/...` 或 `neuron_autoresearch/...` 的实验目录里，通过 overlay 入口或配置调用。

## 一、当前证据汇总

### 1. baseline 参考

| 实验 | 验证集 | firing | SOPs | AEE | AAE |
|---|---:|---:|---:|---:|---:|
| E0 PSN epoch59 | valid40 | 0.084961 | 3.6219G | 1.584776 | 7.501204 |

### 2. 最近比较关键的结果

| 实验 | 机制 | 验证集 | firing | SOPs | AEE | AAE | 判断 |
|---|---|---:|---:|---:|---:|---:|---|
| H9a | H6/H8 稀疏栈 + 兼容式 QK Shiftmax | valid40 | 0.072360 | 3.0847G | 1.504376 | 7.636465 | 当前最好的综合折中 |
| H9e | H9a + 一半 FFN 偶数块，不动 downsample | valid40 | 0.077034 | 3.2840G | 1.497696 | 7.680031 | AAE 安全，但稀疏收益偏小 |
| H9c all6 | stage2 FFN，不动 downsample | valid40 | 0.072303 | 3.0823G | 1.424656 | 31.166865 | AEE 好，但 AAE 爆炸 |
| H10 token Shiftmax | 直接做 QKFormer-token Shiftmax 变体 | valid40 | 0.080291 | 3.4228G | 3.873903 | 71.638042 | 不可行 |
| H10c qk_bsa | 真实 Q/K 三值矩阵 + K carrier | valid40 | 0.081751 | 3.4850G | 1.732079 | 8.054272 | 方向相对稳，但 AEE/SOPs 都弱 |
| H9h-Ang | stage0+2 FFN/downsample + angular loss | valid40 | 0.081536 | 3.4759G | 1.537797 | 7.973107 | angular loss 能控 AAE，但稀疏收益被吃掉 |
| H9g | 全 FFN + 全 downsample | valid40 | 0.073004 | 3.1121G | 1.746386 | 9.043341 | 替换太激进 |
| I17d guard | 负发放率反馈控制，target 0.005，eta10 | valid10 | 0.083248 | 3.5488G | 1.085365 | 6.295033 | 当前负发放控制短测最好 |

### 3. 当前正在跑的 I17d medium

运行目录：

`neuron_autoresearch/experiments/i17_negative_rate_feedback/results/i17d_t005_eta10_scale30_medium3_20260518_150736`

当前已经完成本地 epoch0，进入 epoch1。epoch0 的关键神经元统计：

| 指标 | 数值 |
|---|---:|
| activity_mean | 0.0583616 |
| pos_mean | 0.0549618 |
| neg_mean | 0.0033998 |
| negative_scale_mean | 21.8209 |
| negative_scale_min | 10.1361 |
| negative_scale_max | 30.0 |

这个结果比较重要：之前很多方案的负发放几乎被压没，三值神经元实际退化成近似二值；而 I17d medium 的 `neg_mean=0.0034`，说明负分支还活着。

## 二、问题诊断

### 问题 1：AAE 爆炸不是简单的精度问题，而是方向结构被破坏

H9c 是最典型证据：

- AEE 降到 `1.424656`，看起来很好；
- SOPs 也保持在 `3.0823G`，稀疏也不错；
- 但 AAE 直接爆到 `31.166865`。

这说明模型可以在端点误差或幅值误差上表现不错，但预测向量方向已经明显偏了。也就是说，只靠原来的监督损失和稀疏正则不够，必须显式保护角度结构。

H9h-Ang 说明 angular loss 是有效的：

- AAE 被压回 `7.973107`；
- 但 SOPs 到 `3.4759G`，稀疏收益明显变弱。

结论：angular loss 需要用，但不能太重，也不能和强稀疏从第一轮同时猛压。更合理的是小权重 angular loss + 渐进式稀疏。

### 问题 2：注意力融合不能直接替换主 carrier

目前最稳的注意力方案仍然是 H9a 的 `compat_qk_product`：

- 保留原来的 QKFormer 主路径 carrier：`K * sn2_q(sum(Q))`；
- 只额外加一个来自三值 Q/K 的 Shiftmax 兼容门控；
- valid40 上得到 AEE `1.5044`、AAE `7.6365`、SOPs `3.0847G`。

H10 给出了很强的反例：

- 直接替换 token attention/主 carrier 后，AAE 到 `71.638042`；
- AEE 也变成 `3.873903`；
- 说明主注意力路径不能贸然替掉。

结论：下一步注意力融合应该保留 baseline carrier，只改辅助 gate。这样既能引入三值/正负事件信息，又不破坏原模型最敏感的方向表达。

### 问题 3：负发放不能被压没，也不能放任变密

ATLIF/三值神经元的故事里，负发放应该提供额外表达能力。但之前为了防止负分支过密，使用了很大的 `negative_threshold_scale=30`，结果很多实验里负发放几乎没有，三值退化成近似二值。

I17 的负发放率反馈控制是目前最有希望的修正：

| guard | target | eta | neg_mean | scale_mean | AEE | AAE | SOPs |
|---|---:|---:|---:|---:|---:|---:|---:|
| i17a | 0.003 | 5 | 0.001152 | 29.538 | 1.238055 | 6.720868 | 3.5482G |
| i17b | 0.005 | 5 | 0.001578 | 28.426 | 1.164265 | 6.941315 | 3.5187G |
| i17c | 0.003 | 10 | 0.001123 | 28.848 | 1.242334 | 6.523877 | 3.5408G |
| i17d | 0.005 | 10 | 0.002061 | 27.167 | 1.085365 | 6.295033 | 3.5488G |
| i17e | 0.005 | 5 + ang0.1 | 0.001874 | 28.663 | 1.134102 | 7.009279 | 3.5426G |

我的判断是，Q/K 里的负发放合理区间大概是：

`neg_mean ~= 0.002 - 0.006`

低于这个区间，负分支没有实际表达价值；高于这个区间，可能回到 H5 那种负分支过密、SOPs 上升的问题。

## 三、下一阶段实验方案

## P0：先完成并评估当前 I17d medium

实验目的：确认负发放率反馈控制是否能从短测扩展到更长训练。

当前运行：

- 结果目录：`neuron_autoresearch/experiments/i17_negative_rate_feedback/results/i17d_t005_eta10_scale30_medium3_20260518_150736`
- 配置：`neuron_autoresearch/experiments/i17_negative_rate_feedback/generated_configs/i17d_t005_eta10_scale30_medium3.yml`

需要记录：

- 每个 checkpoint 的 AEE、AAE、SOPs；
- `pos_mean`、`neg_mean`、`activity_mean`；
- `negative_scale_mean/min/max`；
- 和 baseline、H9a、H9e 对比。

晋级条件：

| 指标 | 条件 |
|---|---:|
| neg_mean | >= 0.002 |
| AAE | <= 8.5 |
| SOPs | <= 3.4G |

如果满足，I17 的负发放控制可以作为后面 H12/H13/H14 的基础模块。如果 AAE 开始漂移，就只把 I17 当成负发放诊断实验，不直接作为主线。

## P1：AAE 安全的渐进稀疏训练

实验名建议：`H12_aaps`

全称可以叫：Angular-Aware Progressive Sparsity。

核心思想：

不要从训练第一轮就强行压稀疏。先保护向量方向，再逐步增加稀疏压力。

基础选择：

- attention：沿用 H9a 的 `compat_qk_product`；
- 替换范围：先用 H9a，再试 H9e；
- loss：使用实验本地的 `h9_losses.py`；
- angular loss：打开，但权重先用小一点。

建议配置：

```yaml
loss:
  use_angular_loss: true
  lambda_ang: 0.25
```

渐进稀疏 schedule：

| 训练阶段 | 稀疏正则强度 |
|---|---:|
| epoch 0-5 | FFN/downsample `activity_eta = 0` |
| epoch 6-15 | 目标 eta 的 25% |
| epoch 16-25 | 目标 eta 的 60% |
| epoch 26-30 | 目标 eta 的 100% |

为什么不用 `lambda_ang=1.0` 起步：

H9h-Ang 已经证明大权重 angular loss 能控 AAE，但会削弱稀疏收益。我们现在要的是“精度不明显掉 + SOPs 明显下降”，所以 angular loss 应该作为方向约束，而不是压倒主任务 loss。

建议新增目录：

`neuron_experiments/H12_aaps`

建议配置：

| 配置 | 基础范围 | angular | schedule | 预期 |
|---|---|---:|---|---|
| `h12a_h9a_ang025_sched_full.yml` | H9a | 0.25 | 开 | 在 H9a 基础上保护 AAE，争取进一步降 SOPs |
| `h12b_h9e_ang025_sched_full.yml` | H9e | 0.25 | 开 | 最安全的 AAE 路径 |
| `h12c_h9h_ang025_sched_full.yml` | stage0+2 | 0.25 | 开 | 看 schedule 能不能修复 H9h 稀疏不足 |

晋级条件：

| 指标 | 条件 |
|---|---:|
| valid40 AAE | <= 7.8 |
| valid40 AEE | <= 1.55 |
| SOPs | <= 3.05G |

## P2：保留主 carrier 的三值注意力融合

实验名建议：`H13_tng`

全称可以叫：Ternary-Native Gate。

核心思想：

注意力里可以引入三值/正负事件信息，但不直接替换原始主 carrier。也就是说：

- 保留主路径：`attn = K * sn2_q(sum(Q))`；
- 只替换辅助 gate；
- gate 用三值 Q/K 的正负事件一致性来计算。

不建议继续 H10 那种直接替换主路径的路线，因为 H10 已经明显失败。

候选 gate 设计：

| gate | 机制 | 说明 |
|---|---|---|
| `sign_count` | 正正匹配 + 负负匹配 - 符号冲突 | H11 已经有雏形，最容易先跑 |
| `theta_min_conf` | 同号事件用 `min(abs(Q), abs(K))` 表示置信度 | 把 ATLIF 阈值/幅值信息用于注意力 |
| `dual_path_l1` | 正事件和负事件分别归一化后再合并 | 避免负事件被正事件完全淹没 |

建议新增目录：

`neuron_experiments/H13_tng`

建议从 H9/H11 overlay 复制结构，扩展 `bsa_attention.py` 的 mode：

- `ternary_event_compat`
- `theta_min_compat`
- `dual_path_compat`

所有 mode 都必须满足：

```python
attn = K * sn2_q(sum(Q))   # 主 carrier 保留
attn = attn * gate         # 只改辅助 gate
```

建议配置：

| 配置 | mode | 范围 | 关键参数 | 预期 |
|---|---|---|---|---|
| `h13a_event_h9a_full.yml` | sign-count | H9a | alpha 0.25, beta 1.0 | H9a 级别 AAE，更好讲三值注意力 |
| `h13b_event_i17_full.yml` | sign-count | H9a + I17 | alpha 0.25, beta 1.0 | 检验负发放是否真的参与注意力 |
| `h13c_theta_min_h9a_guard.yml` | theta-min | H9a | 阈值置信度 | 先短测 |
| `h13d_dual_path_h9a_guard.yml` | dual-path | H9a | 正负分路归一化 | 先短测 |

晋级条件：

| 阶段 | AAE | AEE | SOPs |
|---|---:|---:|---:|
| guard/valid10 | <= 6.6 | <= 1.12 | <= 3.60G |
| full/valid40 | <= 7.8 | <= 1.55 | <= 3.10G |

## P3：把负发放控制做成可复用模块

实验名建议：`H14_negctrl`

核心思想：

I17 现在还在 autoresearch 目录里，更像搜索实验。后面需要把负发放率反馈控制抽成一个干净的实验模块，方便和 H12/H13 组合。

建议新增目录：

`neuron_experiments/H14_negctrl`

默认参数建议：

```yaml
negative_threshold_scale: 30.0
negative_target_rate: 0.005
negative_target_eta: 10.0
negative_scale_min: 8.0
negative_scale_max: 60.0
negative_dense_guard_rate: 0.010
negative_dense_guard_eta: 0.1
```

组合实验建议：

| 配置 | 组合 | 目的 |
|---|---|---|
| `h14a_h9a_negctrl_full.yml` | H9a + 负发放控制 | 验证负发放控制本身 |
| `h14b_h9a_negctrl_ang025_full.yml` | H9a + 负发放控制 + 小 angular | 同时控 AAE 和负发放 |
| `h14c_h13_event_negctrl_ang025_full.yml` | H13 注意力 gate + 负发放控制 + 小 angular | 最完整的融合故事 |

## 四、推荐执行顺序

### 第一步：当前 I17d medium 跑完后立即评估

优先级最高。它决定负发放控制是不是值得进入主线。

需要输出：

- valid40 AEE/AAE；
- SOPs；
- firing；
- `neg_mean/pos_mean/activity_mean`；
- 和 baseline、H9a、H9e 放在同一张表里。

### 第二步：跑 H12a 和 H12b 的短训/guard

目的：验证小 angular loss + 渐进稀疏是否能同时保住 AAE 和 SOPs。

优先级：

1. `h12a_h9a_ang025_sched_full.yml`
2. `h12b_h9e_ang025_sched_full.yml`

如果 H12a AAE 稳，优先推进 H12a；如果 H12a 不稳但 H12b 稳，走 H12b。

### 第三步：跑 H13a

目的：先验证最简单的三值事件 gate。

优先做：

`h13a_event_h9a_full.yml`

不要一上来就做复杂 dual-path，因为现在最重要的是证明“注意力融合三值事件信息”不会破坏 baseline carrier。

### 第四步：做 H14 组合

如果 I17d 有效，就做：

`h14b_h9a_negctrl_ang025_full.yml`

如果 H13a 也有效，再做：

`h14c_h13_event_negctrl_ang025_full.yml`

## 五、暂时不要继续的方向

### 1. 不要全局替换 FFN/downsample 为强三值负发放

H5/H9g 已经说明这条路容易变密或者损伤精度。

### 2. 不要直接替换注意力主 carrier

H10 已经明显失败。后面只做辅助 gate 级别融合。

### 3. 不要只看 valid10 判断 AAE

之前有些短测看起来很好，但 valid40 会暴露 AAE 问题。AAE 相关结论必须至少看 valid40。

### 4. 不要默认用很大的 angular loss

`lambda_ang=1.0` 可以当安全检查，但不是默认训练策略。默认建议 `0.25` 起步。

## 六、最适合讲论文故事的路线

当前最合理的故事不是“把所有神经元都换成三值”，而是：

> 在 SDFormerFlow 的稳定 PSN/QKFormer 骨架上，引入 ATLIF 式自适应阈值来获得原生稀疏；通过负发放率反馈控制防止三值表达退化为二值；在注意力中保留原主 carrier，只用三值正负事件构造辅助门控；最后用轻量角度约束保护运动向量方向。

对应技术点：

| 问题 | 方法 | 实验 |
|---|---|---|
| 稀疏但 AAE 容易炸 | 小 angular loss + 渐进稀疏 | H12 |
| 注意力融合容易破坏主路径 | 保留 carrier，只改三值辅助 gate | H13 |
| 负发放被压没 | 负发放率反馈控制 | H14 |

如果这条路线跑通，预期目标是：

- AEE 不高于 baseline，最好小幅下降；
- AAE 与 baseline 接近，控制在 `7.5-8.0` 左右；
- SOPs 从 baseline `3.6219G` 降到 `3.0G` 左右；
- 负发放不塌缩，`neg_mean` 保持在 `0.002-0.006`；
- 论文叙事上能同时覆盖神经元稀疏、注意力融合、硬件友好三值事件表达。
