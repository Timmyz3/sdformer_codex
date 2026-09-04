# H37 外部 Review 处理记录

## 结论

其他 agent 提出的三个问题大体成立。旧实验里有些名字写得过满，实际是把论文思想适配到 SDFormerFlow 的 QKFormer 注意力，而不是严格复现原论文模块。H37 开始把“严格复现/较严格范式”和“启发式适配”分开。

## 三个问题逐项处理

| 问题 | 判定 | 修正 |
|---|---|---|
| `strict_bsa` 缺 `sqrt(d)`、缺独立 V，不是 BSA | 部分成立。代码支持 `sqrt_head_dim`，但 H35 主跑的 `signv_head` 用的是 `head_dim`；更关键的是原 baseline 没有 V，旧 `strict_bsa_shiftmax` 复用了 K 作 V。 | 新增 `strict_bsa_qkv_shiftmax`：`sign(Q) @ sign(K)^T / sqrt(d) -> Shiftmax -> V`，并在 overlay 中为每个注意力块动态挂 `linear_v/bn_v/sn_v`，由 K 分支 copy 初始化但后续独立训练。 |
| alpha-XNOR 原论文是二元，我们写成三元+负极性+冲突惩罚，改动过大 | 成立。旧 `alpha_xnor_matrix_*` 应叫 ternary alpha-XNOR-inspired signed similarity，不应称为原版 alpha-XNOR。 | 新增 `binary_alpha_xnor_matrix_shiftmax/l1`：只把正脉冲视为 1，其余为 0；静默匹配给 `alpha0` 权重；默认 `mismatch_penalty=0`，不加入三值负极性冲突项。 |
| A2OS2A Q/K/V 三路不完整，只借鉴想法 | 成立。旧 `a2os2a_direct` 用 binary Q、非负 K，但 V 仍复用 K。 | 新增 `a2os2a_qkv_l1`：binary Q、非负 K、独立 V，V 可取 `sign` 或 `threshold`。仍是 SDFormerFlow 适配版，但三路结构补齐。 |

## 代码改动

| 文件 | 改动 |
|---|---|
| `overlay/models/STSwinNet_SNN/bsa_attention.py` | 新增独立 V 分支安装、`strict_bsa_qkv_shiftmax`、`binary_alpha_xnor_matrix_*`、`a2os2a_qkv_l1`。 |
| `tests/test_bsa_attention.py` | 增加 QKV-BSA、binary alpha-XNOR、QKV-A2OS2A 前向测试；确认独立 V 分支会在优化器构建前创建。 |
| `entrypoints/make_h37_reviewed_attention_configs.py` | 生成 H37 修正版短测配置。 |

## 已验证

```bash
/opt/conda/envs/sdformerflow/bin/python -m unittest neuron_experiments/H9_bipolar_self_attention/tests/test_bsa_attention.py
```

结果：`Ran 7 tests ... OK`。

## H37 短测原则

- 神经元主线不变：Q/K 仍是三值 PSN+ATLIF；高 SOP 层是二值 official ATLIF。
- 只测试被 review 指出的注意力范式修正版。
- 学习率只用 H36 已经显示稳的两组：`conservative` 和 `neuronfast`。
- 如果 H37 修正版没有超过 H36 的 `stage02_highsop_conservative`，全量训练回退到 H36 最佳候选，避免为了“形式严格”牺牲指标。

