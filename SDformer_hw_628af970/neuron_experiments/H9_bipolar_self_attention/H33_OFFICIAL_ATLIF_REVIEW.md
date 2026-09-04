# H33 官方 ATLIF 范式复核

## 结论先行

你指出的问题是成立的：H9/H13 后续融合方案里的“ATLIF”已经不是严格 Activity-Pruning-SNN 官方范式，而是我为了兼容三值、BSA、target-rate 搜索加出来的一套变体。它能做工程搜索，但不能直接当“官方 ATLIF 原生稀疏”来讲。

当前已修复：
- 在 `ATLIFTernaryPSN` 中新增 `threshold_mode: official_atlif`。
- `official_atlif` 强制 binary 输出 `{0, threshold}`，不允许 ternary。
- 阈值更新使用官方 `Surrogate` 逻辑。
- 对 PSN 的时间维输出补上 `/T` 归一，避免把官方逐时间步更新误放大。
- `threshold_update` 在 `official_atlif` 下关闭 target-rate 反馈和 clamp，避免阈值被非官方机制拉低或截断。
- 单测已通过：`Ran 15 tests ... OK`。

## 官方范式是什么

官方 Activity-Pruning-SNN 的关键闭环在：

`optimization_sources/neuron_optimization/ATLIF_Activity-Pruning-SNN/models/submodules/layers.py`

核心逻辑：
- `Surrogate.forward`: `out = input >= threshold`
- 输出不是 0/1，而是 `out * threshold`
- 每次 forward 记录 `thre_updates`
- ATLIF 按时间步循环，每步累积 `thre_updates / T`
- loss 里可加 `regularize_spike(model) * eta2`
- optimizer step 后调用 `threshold_update(model, lr)`，执行 `threshold += update_value * lr`

也就是说，官方范式不是 target-rate 控制，也不是负阈值控制，也不是三值输出。

## 之前失败可能和它有关吗

| 系列 | 是否严格官方 ATLIF | 失败是否可能受影响 | 判断 |
| --- | --- | --- | --- |
| E2 | 接近官方 ATLIF，但替换的是全局普通 ATLIF，不是 PSN-ATLIF 融合 | 部分相关 | 源码范式较对，但和 SDFormerFlow baseline PSN 的时间混合差异太大，性能差不只是不官方 |
| H3 | 官方 surrogate + PSN 融合，但旧实现没有 `/T` 归一 | 相关 | 它证明 Q/K 局部 firing 能降，但全局 SOPs 没明显降；需要按修正版重测 |
| H6/H8/H9 | ATLIF + 三值/BSA/target-rate 混合 | 强相关 | target-rate 会把阈值往下拉，和“阈值持续增大带来稀疏”的故事冲突 |
| H13 | 在 H9 上继续叠 attention/三值修复 | 强相关 | 如果基础 ATLIF 范式已偏，后续 attention 结果也会混入这个偏差 |
| H32 | 扩大三值替换配置 | 强相关 | 暂时不应优先全量，应该先用 H33 修正 ATLIF 范式后再扩展 |

## 是否要把后续方案都换成官方范式

要分层处理，不能一句话全换：

1. 如果实验声称是 ATLIF 稀疏，就必须使用 `official_atlif`。
2. 如果实验声称是三值/BSA/TSN 融合，就不能叫“官方 ATLIF”，只能叫“ATLIF 启发的三值自适应阈值”。
3. FFN/downsample 这些高 SOP binary 稀疏层，最适合先换成官方 ATLIF，因为它们不需要负脉冲。
4. Q/K 如果继续三值，就无法完全官方 ATLIF；要么改成 binary official ATLIF 做对照，要么明确写成 BSA/TSN 分支。

## 新 H33 配置

生成脚本：

`entrypoints/make_h33_official_atlif_configs.py`

| 方案 | 配置 | 内容 | 用途 |
| --- | --- | --- | --- |
| H33a | `h33a_official_qk_binary_scale150k_act2_rapid.yml` | 所有 Q/K 换成 binary official ATLIF-PSN，关闭 Shiftmax/BSA | 纯官方 ATLIF-PSN 对照 |
| H33b | `h33b_official_qk_highsop_binary_scale150k_act2_rapid.yml` | Q/K + H28b 已选 FFN/downsample 全部 official binary ATLIF，关闭 Shiftmax/BSA | 看官方 ATLIF 扩到高 SOP 层是否能真降 SOPs |
| H33c | `h33c_h9_qkternary_highsop_official_scale150k_act2_rapid.yml` | 保留 H9 Q/K 三值 + Shiftmax；FFN/downsample 改 official binary ATLIF | 保留 H9 精度红利，同时修正高 SOP 层 ATLIF 范式 |
| H33d | `h33d_h9_qkternary_highsop_official_scale300k_act4_rapid.yml` | H33c 的更强稀疏版本 | 如果 H33c 稀疏不足，验证更强 activity/threshold 是否可用 |

每个方案同时生成了 full 配置，但必须先 rapid + valid40 筛选。

## 重跑优先级

优先跑：
1. H33c rapid：最贴近当前 H9/H28 路线，同时修复 high-SOP 层的官方 ATLIF 范式。
2. H33b rapid：纯官方 ATLIF 能不能在 Q/K + FFN/downsample 上讲通稀疏故事。
3. H33d rapid：如果 H33c SOPs 没降够，再试更强稀疏。
4. H33a rapid：作为纯 Q/K official ATLIF 对照。

暂缓：
- H32 扩大三值替换。它应该等 H33c/H33d 明确可行后，再基于“官方 ATLIF high-SOP + 三值 attention”继续扩展。

## 对已有全量的处理建议

当前 H28b 全量不建议再作为最终主结果，因为它的 target-rate 反馈正在把阈值均值往下拉，这与 ATLIF 原文故事冲突。它可以保留为“非官方 target-rate 变体”的参考结果，但不应该继续投入大量 full 训练时间。
