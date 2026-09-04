# H34/H35 官方范式重做短测计划

## 边界约定

这轮重做遵守一个固定边界：所有被替换进去的神经元模块都是 `PSN + ATLIF`。替换范围只决定两个问题：

- 哪些 baseline PSN 节点被替换为 ATLIF-PSN 包装模块；
- 被替换位置输出是二值 `binary PSN+ATLIF`，还是三值 `ternary PSN+ATLIF`。

其中 Activity-Pruning-SNN 官方 ATLIF 范式本身是二值输出：`out = spike * threshold`，阈值更新来自 spike-surrogate 项并按时间步平均，训练后通过 `threshold_update` 让阈值随训练增大。因此 H34/H35 中的 FFN/downsample 高 SOP 层优先使用 `threshold_mode: official_atlif` 的二值 PSN+ATLIF。Q/K 三值分支为了服务 BSA/alpha-XNOR/三值注意力，仍保留三值 PSN+ATLIF。

## H34：神经元替换范围与超参重做

H34 用来重做之前的神经元范围短测，变量是替换范围和官方 ATLIF 稀疏强度。

| 类别 | 配置前缀 | Q/K | 高 SOP/附加层 | 注意力 |
|---|---|---|---|---|
| 纯官方 ATLIF | `h34_pure_official_*` | 二值 official PSN+ATLIF | 二值 official PSN+ATLIF 或不额外替换 | 关闭 BSA/Shiftmax |
| H9 混合 | `h34_hybrid_h9_*` | 三值 PSN+ATLIF | 二值 official PSN+ATLIF | 保留 H9/H28 的三值注意力 |

短测范围包括：

- `qkonly`：只替换 Q/K；
- `highsop`：替换既有高 SOP FFN/downsample 集合；
- `ffn_sn1`：只替换 FFN 升维侧；
- `stage23_ffn`：只替换 stage2/3 FFN；
- `stage02_highsop`：替换 stage0/2 的高 SOP 层；
- `attn_aux`：替换注意力内部 `sn2_q/attn_sn/proj_sn`；
- `attn_aux_highsop`：注意力内部节点 + 高 SOP 层。

稀疏强度扫描：

- `threshold_lr_scale=150000, activity_eta=1.0/2.0`
- `threshold_lr_scale=300000, activity_eta=2.0/4.0`

## H35：注意力方案短测重做

H35 固定神经元边界：Q/K 是三值 PSN+ATLIF，高 SOP FFN/downsample 是二值 official PSN+ATLIF。变量只放在注意力公式和注意力超参上。

| 配置前缀 | 注意力方案 | 目的 |
|---|---|---|
| `compat_qk_shiftmax` | 历史 H9a 兼容门控 | 保留 QKFormer carrier 的对照 |
| `alpha_xnor_shiftmax_*` | alpha-XNOR 矩阵 + Shiftmax | 测三值相似性矩阵与 Shiftmax 是否仍最优 |
| `alpha_xnor_l1_*` | alpha-XNOR 矩阵 + L1 | 去掉 Shiftmax，看 AAE 问题是否来自归一化 |
| `strict_bsa_*` | `sign(Q) @ sign(K)^T -> Shiftmax -> V` | 标准 BSA 范式重测 |
| `signed_consensus_*` | signed popcount token gate | 更硬件友好的类 Shiftmax/无 Shiftmax 方案 |
| `a2os2a_direct_l1` | binary Q、非负 K、L1 | A2OS2A 范式直接替换 |
| `hamming_ternary_active` | 三值 active Hamming attention | SpikeVideoFormer 线性注意力的三值改写 |

每个注意力方案先扫两档稀疏强度：

- `s150k_act2`
- `s300k_act4`

## 执行顺序

1. 等当前 H33c 短测结束，先跑一次 valid40 profile，确认官方 ATLIF 高 SOP 层是否让 SOPs/firing 进入合理区间。
2. 启动 H34 rapid screen：先跑代表性范围和稀疏强度，筛掉明显崩的替换范围。
3. 启动 H35 rapid screen：在 H34 的合理神经元边界上，重做注意力方案短测。
4. 对通过 valid10 的候选自动升到 valid40 profile。
5. 只把同时满足 AEE/AAE 不明显崩、SOPs/firing 有下降的方案推进到全量训练。

## 记录要求

每个 rapid screen 会写：

- `summary.md`：中文解释之外的机器表格排名；
- `summary.csv`：便于后处理；
- 每个候选的 `train.log`；
- 每个候选的 `sops_summary.json`。

后续我会把最终筛选结果再汇总进中文实验记录，避免 H 系列继续混乱。
