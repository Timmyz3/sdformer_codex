# M161 H67 dynamic BN + rank-3 fusion DSE 独立打铁审阅 r1

## 裁决

M161 的实数代数、几何和整数计数基本可复现，但关键硬件 DSE 把两条不同数值路径混在了一起，并使用不对称 baseline 得到 4.203× movement reduction。综合评分 **58/100，P0=2、P1=4、P2=2**，裁决为 `REVISE_HARDWARE_DSE_BEFORE_ADMISSION`。

建议保留“dynamic BN 可在实数 rank state 上修正”和几何计数，立即把硬件合同拆成：

1. **Q8 early-requant + PAFT 路径**：只有这条路径可条件性继承 M31 的 96 个 INT8 product slots 和 10.944M/21.888M ideal issues。
2. **Q24/raw exact-real 路径**：需 14–16×8 widened multiplier 或可证明的 limb decomposition，必须重算 issue/area/power，不能继承现有 M31 数字。

## 独立复算了什么

审阅器不导入、不执行 M161 analyzer，直接从 frozen config/profile/ordered trace、M30 合同、M31 RTL 和 checkpoint-derived M160 sumabs census 重算。

| 项目 | 独立结果 |
|---|---:|
| FFN modules / s10 fc1 calls | 12 / 120 |
| BN1 elements/frame | 350,208,000 |
| spatial-hidden columns | 35,020,800 |
| raw accumulator widths | 14b×2 / 15b×4 / 16b×6 modules |
| moment state | 1,252,608 bits |
| 32-lane moment ideal issues | 10,944,000 |
| Q8-only rank3 right/full projection issues | 10,944,000 / 21,888,000 |
| dense sn2 / rank3 projection-only ratio | 1.6667× |

stage 分解：

| stage | blocks | BN1 elements | columns | reduction population/channel | Q8-only right issues |
|---:|---:|---:|---:|---:|---:|
| 0 | 2 | 147,456,000 | 14,745,600 | 192,000 | 4,608,000 |
| 1 | 2 | 73,728,000 | 7,372,800 | 48,000 | 2,304,000 |
| 2 | 6 | 110,592,000 | 11,059,200 | 12,000 | 3,456,000 |
| 3 | 2 | 18,432,000 | 1,843,200 | 3,000 | 576,000 |

raw signed width、signed sum width 和 unsigned sumsq width 全部与 M161 CSV 逐模块一致，原 12×14 个整数字段 mismatch=0。这些 width 仍只是 canonical per-row INT8 weights + unit binary input 下的条件上界，不是已发布 FFN INT8 bridge。

## no-running BN 与 rank-space correction

独立 PyTorch 2.7.1 CPU float64 miter 直接创建 `BatchNorm2d(track_running_stats=False).eval()`，并和 `F.batch_norm(training=True)`、population variance 手算式交叉验证。100 组随机形状结果：

- module vs functional no-running：0 mismatch；
- functional vs `sum/sumsq` population-moment 公式：最大误差 `5.33e-15`；
- `R·BN(x)` vs `alpha·(R·x)+offset·rowsum(R)`：最大误差 `2.13e-14`。

因此实数代数 admission 可保留。它没有证明 rank-3 训练精度、INT8/Q24 定点顺序、溢出、舍入或 threshold 等价。

## P0-1：Q24/raw 路径不能继承 M31

M31 硬约束是：

- `IN_W=8`；
- `x_bank_q`、`multiplier_a`、`multiplier_b` 全是 signed INT8；
- 96 个 product slots 是 8×8；
- right projection 的 Q24 accumulator 在 stage1 结束时立即通过 `rne_sat_q24_to_q8` 写入 8-bit `t10_intermediate_q`。

M161 候选却让 right projection 直接消费 14–16b fc1 raw accumulators，并在 barrier 之后保留/修正 Q24 rank state。这不是现有 M31 datapath。

所以：

- `10.944M right issues`、`21.888M full issues`和 `1.667×` 只能标成 **Q8 early-requant + trained rank3 的 projection-only 条件数字**；
- Q24/raw 路径需新的 widened multiply 或 limb schedule；
- 32 square lanes 的 issue-count 相等也只能在该 Q8 条件路径下讨论。

## Q24 不能称 numeric exact

M161 只指定了 24-bit 宽度，没有冻结 rank factor、factor sumabs、binary point、dynamic alpha/offset 上界和 saturation 次序。仅用不受限的 signed-INT8 right factor 做保守上界，12 个模块在 correction 之前已需 **24–27 signed bits**；stage 1–3 不能由 Q24 保证不溢出。dynamic alpha 还可能放大值域。

`exact_q24_rank3_bits` 最多表示“在任意选定 24b 宽度下的精确 bit-count”，不能表示 exact numeric state。

## P0-2：4.203× baseline 不对称

M161 的 baseline 向 dense dynamic-BN 路径收取了五次移动：

1. raw write；
2. 独立 stats read；
3. normalize read；
4. normalized Q8 write；
5. ATLIF read。

候选路径却允许 moments 在 fc1 stream 上同时累积，并在 barrier 后直接消费 corrected rank state。对称 dense 基线也可以在 fc1 stream 上累积 moments，仅 raw write 一次、barrier 后 raw read 一次，归一化后直接输入 dense ATLIF。

| 口径 | bits/frame | 相对 Q24 rank |
|---|---:|---:|
| M161 五移动 baseline | 21,196,800,000 | 4.2032× |
| 对称 streaming dense BN1 | 10,395,648,000 | **2.0614×** |
| Q24 rank candidate | 5,042,995,200 | 1.0000× |

这些仍是抽象 local bit movement，不是 SRAM 事务或 energy。

## 32-lane overlap、ordering 和 barrier

32 lanes 的 `350,208,000/32=10,944,000` 计数正确，但不等于物理 overlap：

- ordered trace 只证明模块调用顺序和 tensor shape，没有 fc1 的 time/spatial/channel 发射顺序；
- rank right projection 每 issue 需同一 16-column tile 的两个 time values；
- M31 先用 5 个 input beats 填充 bank，再执行 stage1，并非“每拍投影直接消费 32 个 fc1 raw values”；
- 32 个 square units 处理 14–16b raw values，不是 96 个 INT8 product slots；
- broadcast、quantizer、moment SRAM 端口、`input_ready`、bank conflict 和 replay recurrence 未建模；
- no-running BN 需完成整个 module 的 `T×B×H×W` moments，left projection 不能在 global per-module barrier 前开始。

因此 32 lanes 仅 admission **count balance**，不 admission overlap/cycle balance。

## 21.888M 遗漏的工作

21.888M 只是 right+left temporal projection issues，没有包括：

- 每个 rank/spatial/hidden state 的 `alpha×v`；
- `offset×rowsum(R)` 和 add；
- sum/sumsq reduction；
- reciprocal-sqrt 与 alpha/offset 生成；
- barrier wait、SRAM write/read、bank conflict 与 replay。

因此 `dense 36.48M / rank3 21.888M = 1.667×` 可保留为 Q8 路径的 **projection-only arithmetic sensitivity**，不能当 dynamic-BN+rank3 融合模块加速。

## BN2 对完整 FFN 的影响

M161 确实披露了 BN2 仍是 87,552,000 elements 的 dynamic barrier，但它被排除在 4.203× 分子/分母之外。在同一 conditional INT8 width 下，BN2 raw write+read 公共代理为 2,801,664,000 bits/frame。把这个候选与 baseline 都必须支付的开销加回后：

`(10.395648G + 2.801664G) / (5.0429952G + 2.801664G) = 1.6823×`

这仍未包括 residual commit、moment/coefficient traffic 或实际 SRAM 事务，但比 4.203× 更接近完整 FFN 的公平中间移动比较。

## 问题分级

### P0

1. Q24/raw 路径错误继承 M31 INT8 的 issue/resource 模型。
2. 4.203× movement 使用了不对称 baseline；公平 BN1 比较为 2.0614×，加公共 BN2 后为 1.6823×。

### P1

1. Q24 未证明 factor/scale/alpha/overflow，不能称 numeric exact。
2. 32 square-lane 只有 count equality，没有 ordering/port/barrier schedule。
3. 21.888M 遗漏 rank correction、moment 和系数生成。
4. checkpoint-derived sumabs 仍是 canonical census，不是 released FFN INT8 bridge 或 trained rank3 payload。

### P2

1. rank barrier storage 的“total/exact”口径未纳入 moment/coefficient state。
2. 12 模块 barrier storage 求和不是同时物理 footprint；应使用 max block + banking/allocator。

## 修正建议

1. 立即发布 fail-closed overlay，拆分 Q8 与 Q24 两条路径。
2. Q8：冻结 early-requant scale/RNE/saturation、rank3 factors 和 threshold bridge，跑 PAFT + valid825；只在通过后继承 M31 issues。
3. Q24：选择 widened multiplier 或 limb decomposition，重做 VCS/DC/STA/SAIF/PTPX 和 issue 计数。
4. 给出 factor sumabs、dynamic alpha/offset、binary point、溢出与舍入证明。
5. 用 address-timed schedule 闭合 fc1 emission/reorder、global barrier、rank replay、BN2 和 residual commit。
6. 用公平 streaming dense baseline 重算 movement，并将 BN2 公共成本加回完整 FFN 比较。

本审阅只写入 `results/m161_independent_hammer_review_r1_20260824/`，未修改 production、contracts 或 `docs/359`。`docs/359` SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## 复核

```bash
/opt/anaconda3/envs/pytorch310/bin/python \
  results/m161_independent_hammer_review_r1_20260824/torch_no_running_bn_rank_miter.py \
  --output /tmp/m161_torch_miter_fresh.json
python3 results/m161_independent_hammer_review_r1_20260824/validate_review.py
sha256sum -c results/m161_independent_hammer_review_r1_20260824/source_manifest.sha256
sha256sum -c results/m161_independent_hammer_review_r1_20260824/manifest.sha256
```
