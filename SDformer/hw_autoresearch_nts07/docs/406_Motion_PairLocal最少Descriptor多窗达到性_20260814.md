# Motion pair-local 最少 descriptor 多窗达到性

## 1. 裁决

`PASS_PAIR_LOCAL_BOUND_ATTAINED_PROFILE_SEMANTIC`

该证据用于加固 TESC+RQTB 一条主贡献，不新增贡献名，不改 `docs/359`，也不把
profile 语义重放写成多样本 RTL。

证据：

- `scripts/audit_h67_pair_local_descriptor_bound.py`
- `results/h67_pair_local_descriptor_bound_profile100_20260814/report.json`

## 2. 合同与定义性达到

限定合同：

1. descriptor 只属于一个 spatial temporal-pair；
2. 每条 descriptor 只携带一个 Q7 score class；
3. temporal membership 必须恢复 pair 内的 t0/t1；
4. 不删除任何 token，也不改变 class multiplicity。

设一行有 `P` 个 pair，其中 `E` 个 pair 的两个 Q7 score 相等。相等 pair 至少需要
一条 descriptor，不等 pair 至少需要两条，因此：

`D_min = E + 2(P-E) = 2P-E`。

RQTB 的发射规则对相等 pair 发一条 `{class,mask=11}`，对不等 pair 发两条
`{class0,mask=01}` 和 `{class1,mask=10}`，因此逐 pair 达到该下界，且 temporal
membership popcount 总和仍为 `2P`。

这里的“达到”由 RQTB 发射规则直接给出，不是从 672,000 行独立 RTL descriptor
账本观察得到。多窗 profile 的实证内容是：从 ordered count 独立重算 Q7 score 后，
stored equal-pair 统计逐记录一致，并给出 `E/P` 在两个 checkpoint、四个 stage 和
100 个 sample 上的分布。该命题只是合同内的计数下界，不是信息论首创；global
cross-pair bitmap、CAM/histogram 等不同 metadata 合同不在命题范围内。

## 3. 两个 checkpoint 的完整多窗结果

| 项 | ep30 | ep35 |
|---|---:|---:|
| sample / record | 100 / 1,200 | 100 / 1,200 |
| head-row | 672,000 | 672,000 |
| temporal pair | 151,200,000 | 151,200,000 |
| equal pair | 147,530,440 | 147,396,481 |
| Fixed descriptor | 302,400,000 | 302,400,000 |
| RQTB descriptor | 154,869,560 | 155,003,519 |
| slot reduction | 48.7865% | 48.7422% |
| equal 统计重算一致行 | 672,000 / 672,000 | 672,000 / 672,000 |

逐 sample slot reduction：

- ep30：min 48.6377%，mean 48.7865%，p95 48.9479%，max 49.1153%；
- ep35：min 48.5954%，mean 48.7422%，p95 48.9154%，max 49.0937%。

逐行 RQTB descriptor：

- ep30：p50 225，p95 263，p99 323，max 434；
- ep35：p50 225，p95 265，p99 324，max 435。

## 4. Stage 稳定性

| Stage | ep30 slot reduction | ep35 slot reduction |
|---|---:|---:|
| 0 | 49.7620% | 49.7652% |
| 1 | 49.8574% | 49.8465% |
| 2 | 47.9501% | 47.8231% |
| 3 | 43.9726% | 43.9388% |

最深 stage 的等价率下降，但两个 checkpoint 的高等价率仍覆盖全部 stage，说明
RQTB 的收益来源不是 stage0 的偶然特例。定义性达到不随 stage 改变，不把它当成
额外实验证据。

## 5. 证据分层

- `[rtl]` 主锚点：ep35 公平 138 行已有逐行实际 descriptor 账本，138/138 行
  `rslots=450-equal`，且 equal 与独立整数 score 重算一致；汇总仍为
  `112589/94891=1.1865x`、slot `62100/34099`、equal `28001`；
- `[prof]`：本轮两 checkpoint 全窗口逐 pair 重算 score，并验证 stored equal-pair
  统计；descriptor 数按冻结 RQTB 发射规则推导；
- `[rtl]` 数值支撑：sample0/window0 已有 12 block、两个 checkpoint INT8
  输出通道的 Fixed2S/RQTB2S 联合 Acc32 miter，1,104 个标量比较零失配；
- `[待验证]`：ep35 多样本真实 Icarus/Verilator 周期、多样本/全输出通道联合
  真实权重投影、DC/SAIF。

禁止：

1. 把本轮 48.7% 写成全 encoder 加速；
2. 把 672,000 行/profile 写成 672,000 行真实 RTL；
3. 把 `D_min` 写成覆盖 global bitmap/CAM 的通用下界；
4. 混合 ep30 准确率与 ep35 硬件数字。

逐行 RTL 证据：

- `results/h67_fair_row_descriptor_bound_20260814/report.json`
- SHA256 `5c61b0e16a0fef89704d48e96809292c79d3dc598795c2d4d40d419e67fff4f2`

## 6. DATE 分线复评

Motion 架构创新由 `3.1` 提到 **`3.2/5` 的可守叙事**：不是因为公式复杂，而是
冻结 pair-local 执行对象的等价率在两个 checkpoint、全部 stage 和全部窗口上均有
分布证据，且冻结 RQTB 发射规则达到合同下界。
完整度仍约 `3.0/5`，因为多样本真实 RTL、全输出通道联合 Acc32 与目标库功耗尚缺。
