# H32 扩大三值替换配置记录

生成脚本：`entrypoints/make_h32_ternary_expansion_configs.py`

基础配置：`configs/h28b_diff_lr_newfast_steps360_auto_full_20260520_201852.yml`

共同前提：
- 所有 H32 方案都延续 H28b 的续训范式：从 baseline checkpoint 续训，全量参数训练，不冻结 backbone。
- Q/K 仍使用 PSN + ATLIF + 三值输出，这是 H9/H28 的主线。
- 注意力仍使用当前 H28b 的 `alpha_xnor_matrix_shiftmax`，先只扩大三值神经元替换范围，不同时改注意力公式。
- FFN/downsample 是否三值由各方案单独控制。
- 每个方案都有 `rapid` 和 `full` 两份配置：`rapid` 用 `max_train_steps=360` 做短测，`full` 用 30 epoch 全量续训。

| 方案 | 配置文件 | 替换范围 | 目的 | 风险 |
| --- | --- | --- | --- | --- |
| H32a | `h32a_expand_attn_aux_ternary_rapid.yml` / `h32a_expand_attn_aux_ternary_full.yml` | Q/K 三值；所有 block 的 `sn2_q`、`attn_sn`、`proj_sn` 也三值；FFN/downsample 保持 H28b 的 binary ATLIF | 检查 attention 内部二值门是否限制三值表达 | attention 内部全三值可能放大 AAE |
| H32b | `h32b_expand_ffn_sn1_selected_ternary_rapid.yml` / `h32b_expand_ffn_sn1_selected_ternary_full.yml` | Q/K 三值；H28b 已选 FFN 中只把升维侧 `mlp.sn1` 改三值；`mlp.sn2` 和 downsample 仍 binary | 较温和扩大 FFN 表达，避免 FFN 两侧同时扰动 | SOPs 下降可能有限 |
| H32c | `h32c_expand_stage23_ffn_selected_ternary_rapid.yml` / `h32c_expand_stage23_ffn_selected_ternary_full.yml` | Q/K 三值；stage2/stage3 已选 FFN 的 `sn1/sn2` 三值；stage0/stage1 FFN 和 downsample 仍 binary | 验证后部高语义层替换是否比前部更稳 | stage2 是高 SOP 区，三值可能增 firing |
| H32d | `h32d_expand_attn_aux_ffn_sn1_ternary_rapid.yml` / `h32d_expand_attn_aux_ffn_sn1_ternary_full.yml` | H32a + H32b，同时扩大 attention 内部和 FFN 升维侧 | 更激进地验证三值表达上限 | 可能同时影响 AAE 和 SOPs |
| H32e | `h32e_expand_all_selected_ffn_down_ternary_rapid.yml` / `h32e_expand_all_selected_ffn_down_ternary_full.yml` | Q/K 三值；H28b 已选 FFN/downsample 全部三值 | 上限测试：看大范围三值是否仍可训练 | 最大风险，若三值 firing 增多可能不稀疏 |

配置校验结果：
- H32a：显式 group path 52 个，其中 ternary 36 个、binary 16 个，无重复路径。
- H32b：显式 group path 16 个，其中 ternary 7 个、binary 9 个，无重复路径。
- H32c：显式 group path 16 个，其中 ternary 8 个、binary 8 个，无重复路径。
- H32d：显式 group path 52 个，其中 ternary 43 个、binary 9 个，无重复路径。
- H32e：显式 group path 16 个，其中 ternary 16 个、binary 0 个，无重复路径。

建议短测顺序：
1. H32b：最稳，先看 FFN 升维侧三值是否改善 AAE，同时 SOPs 不明显上升。
2. H32c：验证后部 stage 的 FFN 三值是否比全局替换更稳。
3. H32a：单独看 attention 内部三值兼容性。
4. H32d/H32e：只有前三个短测出现优势时再跑，避免盲目扩大替换浪费训练时间。
