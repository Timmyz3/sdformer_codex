# M70/M71/M72 Phi-PAFT 独立打铁评审（R1，2026-08-23）

## 结论先行

**结论：M70/M71 的计数与容量算术可复现，但禁止直接启动 M71 五轮正式 PAFT；M72 证明 Phi 风格 Hamming Lloyd 比 top-frequency 明显更好，却仍只是 valid825 内部方向筛选，不是训练 catalog，也不是 DATE 性能结果。**

独立验证器没有导入 M43/M70/M71/M72/PAFT 生产实现；它从冻结 M40 little-endian positive planes 重建 Conv3x3 `I_KY_KX` 特征，复算 calibration 0--4、heldout 5--9 的全部 15 个 M70 配置，并逐 partition 核对 M71 catalog。结果为零 mismatch。M72 的逐中心、Lloyd 迭代数和 heldout 计数亦按独立实现复核。

五轮 PAFT 的正式启动判定为 **NO-GO**，原因不是算术错误，而是两项 P0：

1. M40 十样本来自 `valid_split_seq.csv` 前十条，M71 catalog 的 0--4 因而直接使用 valid825；M71 却声明 `test_or_validation_data_used=false`。后续若用 valid825 做准确率门，会发生数据污染。
2. 当前 PAFT proxy 对真实幅值 `activation` 直接计算 L1/Hamming 和 `activation.sum()`；冻结 M40 正值不是二值 1，而是 ATLIF threshold 幅值（例如 `0.9999928`），且阈值可训练。该 proxy 可通过缩幅值降低“硬件代价”，即使 support/vector-op 不变。

M71 top-frequency catalog 还不是 Phi 的 catalog 方法。Phi 路线是过滤 zero/one-hot 后逐 partition 做 Hamming k-means；因此 M71 只能保留为 top-frequency 负基线和 PAFT plumbing，不能成为正式训练 catalog。

## 冻结身份与独立范围

| 对象 | SHA256 / 身份 |
|---|---|
| M40 packed manifest | `e743364bb599214dc13ad2591bf96dbf6091d95f8cc5a585ddc86370ccc514d3` |
| M70 result | `d8a41e71b04752751d9ebf54b1022f01f104d90ccce3be80e72cabf2ccaa8922` |
| M70 analyzer | `1e9da853d97e7009ecb9c861b87b3fa3d4a52f60b188afbba1dc5b20f9d2dd28` |
| M71 catalog | `142e32f0d988721ce9edf25d4dcf3883d82f2604f2aee9c755cde87b2ef70cdd` |
| M71 builder | `69e28dcc62f126cb32136e1ad6107f70ac192303b5b68c7e10503cb245001b2a` |
| M71 revocation contract | `4a96226b35234366854c656db6f7443699f6d91131b8281c56a36039bf3a0238` |
| M72 valid825 internal screen | `e3f40697e1b1442d3b190c3aa2cc540ee5892a5db37366808d97d7c635250133` |
| M72 analyzer | `eb31555b6be64a8a9376647b16a1cb039dc3b49b19f176abb759b522dc93dfa2` |
| PAFT implementation（审阅时） | `90cfc096c30e5584ddaa42693786b7440602e0d080d74c518677ccdb167ba436` |
| 5-epoch config（审阅时） | `585290974e2a6dd67e2ddd28b1af834c31c1119146efd1b13c894769676eb631` |
| valid825 list | `7f3dc2800653e12caca10379c51ee8e8988aaf6bb80c391224a454a5879325d0`，825 条；M40 sample key 等于前十条 |

本评审未修改任何生产文件。隔离 probe 与独立验证器只写入本 review 目录。

根线程随后封存的 `m71_valid825_catalog_revocation_r1_20260823.json` 与本审查一致：撤销 M71 的 train-only/status/leakage claims，禁止 PAFT training、checkpoint selection、valid825 accuracy/independent speedup/DATE claim；仅保留 hook plumbing、整数位宽/容量机制测试和 valid825 内部 screen。该撤销合同评审通过，正式路线必须服从它。

## M70 全部独立复算

冻结 baseline 均为 `46,432,637` 个“每输出块等价”96-lane vector-op。下面的 candidate 是 `PWP + correction`，每项都满足守恒且与 M70 JSON 完全一致。

| k | q | candidate | speedup |
|---:|---:|---:|---:|
| 16 | 8 | 41,643,967 | 1.114991x |
| 16 | 16 | 39,060,535 | 1.188735x |
| 16 | 32 | 32,031,627 | 1.449587x |
| 16 | 64 | 27,158,040 | 1.709720x |
| 16 | 128 | 23,838,279 | **1.947818x** |
| 32 | 8 | 43,690,192 | 1.062770x |
| 32 | 16 | 42,458,732 | 1.093595x |
| 32 | 32 | 39,766,187 | 1.167641x |
| 32 | 64 | 34,189,726 | 1.358087x |
| 32 | 128 | 30,138,251 | 1.540655x |
| 64 | 8 | 45,148,473 | 1.028443x |
| 64 | 16 | 44,503,972 | 1.043337x |
| 64 | 32 | 43,182,623 | 1.075262x |
| 64 | 64 | 40,554,654 | 1.144940x |
| 64 | 128 | 37,150,248 | 1.249861x |

M70 best `k16/q128` 的细目也完全复现：

- `PWP=13,707,921`、`correction=10,130,358`，和为 `23,838,279`；
- exact-match fallback 为 `36,252,932`，即 `1.280797x`；
- 221,184 个 entries 中 heldout 使用 220,683 个，使用率 `99.7735%`，不能靠“只预取 used entries”显著缩表；
- M70 冻结 signed19 PWP 容量 `403,439,616 B = 384.75 MiB`，超过 256 MiB；
- `1.947818x < 3x`，计算门失败。因此 M70 自己的 `rtl_allowed=false` 正确。

一个可用于后续 DSE、但不能改变当前 admission 的观察：由于这里仍是 `k16` 且权重为 INT8，M71 的 12-bit 严格 PWP bound 若也用于 q128，则 PWP payload 为 `243.0 MiB`（含 pattern table 为 `243.421875 MiB`），而不是 signed19 的 384.75 MiB。它勉强落在 256 MiB 内，但仍缺量化权重/PWP payload、matcher、metadata、buffer 和真实端口，所以不能据此准入。

## M70 的操作数标签问题

M70 docstring 将结果称为“96-lane vector additions”，但代码没有乘 `OUTPUT_BLOCKS=8`。其数值实际是 **per-output-block equivalents**。比例不受影响，绝对物理 vector addition 数应为：

- baseline：`371,461,096`；
- k16/q16：`312,484,280`；
- k16/q128：`190,706,232`。

论文、图表和 simulator 接口必须改名或显式乘 8；否则 reviewer 会认为少算 8 倍工作。

## M71 catalog 与容量

M71 的纯 catalog 算术正确：

- 4 operators × 432 partitions × 16 patterns = `27,648` entries；
- calibration partition vectors `25,920,000`；baseline `46,207,835`；exact fallback `43,629,207`；
- INT8 `[-128,127]` 的 16 项和范围为 `[-2048,2032]`，signed 12-bit 足够；
- 每个 PWP vector：96 lanes × 12 bits = `1,152 bits = 144 B`；
- 全 PWP：27,648 × 8 × 144 = `31,850,496 B = 30.375 MiB`；
- one-partition/one-output-block：16 × 144 = `2,304 B = 2.25 KiB`。因此“2.304 KiB”单位写法不严谨：它是 2.304 kB，或 2.25 KiB。

但 `2,304 B` 不是可执行系统工作集。若 matcher assignment 只做一次并复用于 8 个输出块，同时驻留应至少为 `18,432 B = 18 KiB` PWP，再加 32 B pattern row、correction weights、accumulator、tag 与 FIFO。若只驻留 2,304 B，就必须 rematch 8 次或物化 assignment；单 operator/sample 的 3,000 rows × 432 partitions × 5-bit ID 已约 `810,000 B`。必须给出 partition-major/liveness schedule 和端口守恒。

此外，M70 k16/q16 的 heldout used entries 为 27,648/27,648；30.375 MiB 不能通过静态 used-entry pruning 继续缩小。当前 M39 约 365,760 B hard available 也容不下 full PWP，必须明确外部 L2/DRAM/prefetch 层次。

## M72：Phi 风格方法对齐后的实际收益

M72 将 M71 top-frequency 替换为：过滤 all-zero/one-hot、逐 operator/partition、count-weighted deterministic Hamming Lloyd、majority center update。隔离生产 probe 得到：

- baseline `46,432,637`；
- candidate `30,889,399 = 7,371,217 PWP + 23,518,182 correction`；
- nominal vector-op speedup **1.503190x**；
- 相比 M71/M70 top-frequency q16 的 candidate-op 减少 **1.264529x**；
- PWP 仍为 30.375 MiB；
- 四算子 speedup 为 `1.4210x / 1.5154x / 1.4836x / 1.7826x`。

这证明方向是对的，也证明 M71 top-frequency 不应进入正式训练。但它仍没有通过 3x：达到 3x 需 candidate 不超过约 `15.478M`；M72 当前 correction 为 `23.518M`。即便把每个 nonzero partition vector 都理想化为单 PWP，ceiling 也只有 `3.3873x`，3x 门只剩很窄余量。PAFT 若想过门，必须在不损准确率/不靠降活动率作弊的条件下消掉约九成 correction，难度很高。

M72 已把自己的口径修正为 `valid825 internal screen`、`train_catalog_eligible=false`、`paft_allowed=false`，这是正确处理。下一步必须从训练集捕获 disjoint calibration cohort 后重建相同 Lloyd catalog。

### 额外抓到的 source-SHA TOCTOU

本 review 的隔离 M72 probe 在运行期间恰逢生产源码从旧 heldout/train-eligible 口径修成 valid825-internal 口径。Python 进程已加载旧代码，但 payload 在输出时才执行 `sha256(Path(__file__))`，因而旧语义 probe 错误记录了修订后 analyzer SHA `eb3155...`。正式 M72 JSON 与 probe 语义不同，却可声称同一 analyzer SHA。

这是可复现的 receipt 身份漏洞，不影响本次独立数值计数，但证明“结果内自报的 analyzer SHA”不能单独建立代码身份。所有长任务必须在启动时读取/固定 source SHA，结束时再次计算并要求 start=end；最好由外部 launch manifest pin git blob/source SHA，输出 receipt 同时引用 launch identity。

## baseline 公平性与 PWP 流量

当前一项 baseline INT8 96-lane vector 为 96 B，一项 tight PWP 为 144 B；把两者都计作一个 op 对 bandwidth-limited 设计不公平。

| 配置 | nominal vector-op | 加一次可跨 8 blocks 共享的 1-cycle matcher | bit-tight byte-only | 32 B port，含一次共享 matcher |
|---|---:|---:|---:|---:|
| M71 top-frequency k16/q16 | 1.188735x | 1.097684x | 1.018082x | **0.950117x** |
| M72 Hamming Lloyd k16/q16 | 1.503190x | 1.360488x | 1.342954x | **1.258898x** |

这是非常乐观的 matcher：假设 16 candidates 一周期完成且 assignment 可跨 8 个输出块复用。若 DC 无法在目标频率做到，收益还会下降。candidate 与 baseline 必须使用同一 INT8 precision、96 lanes、SRAM/DRAM、port、frequency、buffer 和 accumulator 规则，并计入 pattern read、PWP read、correction weight read、assignment/rematch、packing、控制和写回。

M70 q128 在 signed19、32 B port 下，即使暂不计 matcher，流量等效也约 `0.9946x`；12-bit 假设可提升到约 `1.4080x`，但 128-way matcher 和近 243 MiB PWP 使它不是低风险主线。

## PAFT 代码审查

### P0：proxy 不是真实 binary/Hamming cost

`pattern_paft.py::_cost_proxy` 对 raw activation 做：

```text
hamming = abs(activation - pattern)
zero_fallback = activation.sum()
```

这只有 activation 严格属于 `{0,1}` 时才等于硬件计数。M40 明确 `all_values_integer=false`、`all_values_ternary=false`；ATLIF binary forward 返回 `{0, threshold}`，而 5-epoch config 允许 threshold/全模型训练。必须至少以 hard binary support 作为 forward cost，并配置正确的 STE/阈值策略，同时记录每 epoch 的真实 packed support replay。还要阻止负值进入 binary proxy。

### P0：catalog 数据来源声明错误

M40 tracer 源码明确 `DSECDatasetLite(..., file_list="valid")`，且 M40 keys 等于 local valid825 的前十条。M71 的 `test_or_validation_data_used=false` 是事实错误。0--4 与 5--9 的 record/sample key 确实互斥，但它们都来自同一 `zurich_city_09_a` valid sequence，既不是 cross-sequence，也不是可用于 valid825 accuracy gate 的 clean split。

### P1：M71 catalog loader 只信布尔字段

loader 只检查 schema、一个 leakage 布尔值和 shape；没有 pin catalog SHA、builder SHA、M40 manifest SHA、split list SHA、sample keys 和 operator identities。伪造或误标的 catalog 可直接通过。所有身份必须由运行 config/receipt fail-closed 固定，不能信 catalog 自报。

### P1：训练目标混淆 pattern gain 与通用稀疏化

目标最小化绝对 `min(popcount,1+hamming)`，天然奖励全部归零。必须增加同起点、同 seed、同 epoch 的 no-PAFT 基线，并报告 equal-activity/equal-rate 下的 pattern-specific gain；准确率、event density、threshold、support popcount 必须作为 guardrail。否则所谓“pattern 学习”可能只是稀疏塌缩。

### P1：算术 identity 未绑定真实幅值与 INT8 payload

实际输入是 `x = theta*b` 而非字面 0/1。`W*x=PWP[p]+W*(x-p)` 的 support 版本需要 `theta` carry/late-scale 的冻结数值桥。M71 只有 weight content hash，没有量化 weight 或真实 PWP payload，12-bit 是正确的数学容量上界，不是已验证 PWP 内容。

### P1：catalog 优化目标与正式路线

M71 选 calibration 中 top exact-frequency patterns，并未最小化 nearest-Hamming objective，也不符合 Phi 的 Hamming k-means。M72 已证明对齐后 candidate-op 改善 1.2645x。正式 PAFT 必须使用 training-only Lloyd catalog；M71 仅作负基线。

### P1：长任务 source SHA 存在 TOCTOU

M70/M71/M72 builder 都在输出时对当前脚本文件计算 SHA，而不是证明进程实际加载的代码。隔离 probe 已实证源码热修改会让旧代码产物记录新 SHA。应使用 start/end 双 SHA fail-closed 或外部 launch manifest。

### P2：实现/效率问题

- pattern tensor 留在 CPU，并在每个 partition chunk、每次 forward 调 `.to(device,dtype)`，会产生反复 host-to-device copy；应在安装时注册 device-aware buffer/cache。
- 每 module/step 固定取 64 个 deterministic positions，可能长期采到相同栅格；需 epoch/step 可复现轮换，并以全量 frozen replay 判定。
- config 中 `hardware_fanout_output_blocks`、`partition_bits`、`patterns_per_partition`、`runtime_cost` 未被 installer 完整交叉检查/使用，存在配置漂移静默通过。
- hard nearest/min 的梯度只经过 winner，正式训练前需做 gradient smoke、support 0/1 assertion、zero-collapse alarm。

## 评分（100 分）

| 维度 | 分数 | 判断 |
|---|---:|---|
| 独立数值可复现性 | 96 | M70 15 配置、M71 catalog/容量、M72 Lloyd 均可复算；主要扣口径/身份错误 |
| 硬件创新性 | 69 | exact PWP + signed correction + pattern-aware training 有联合设计价值；但核心祖先接近 Phi，必须靠训练目标、调度/存储和系统闭环拉开差异 |
| 性能潜力 | 52 | M72 nominal 1.503x，比 top-frequency 明显进步；公平 optimistic 32 B port+matcher 仅约 1.259x，离 headline 仍远 |
| 实验证据完整性 | 41 | 同一 valid sequence、十窗、无 cross-sequence、无 clean training catalog、无 post-PAFT accuracy/cycle/PPA |
| PAFT 正式启动就绪度 | 25 | catalog 污染、proxy 语义和 provenance 尚未修复 |
| DATE 论文硬件就绪度 | 34 | 目前是高质量 opportunity screen，不是 cycle/RTL/PPA/full-system result |

## 正式五轮 PAFT 的解锁条件

只有以下条件全部完成才允许启动正式 5 epochs：

1. 从 **训练集** 捕获并冻结多 sequence、分层 event-density calibration；另留整条 sequence 作为训练后 hardware heldout，valid825 完全不参与 catalog/超参选择。
2. 用 M72 的 filtered weighted Hamming Lloyd 生成新 catalog/schema；M71 top-frequency 只保留对照。
3. 修复 binary support proxy、threshold carry 和负值断言；加入 tiny synthetic oracle、gradient smoke、zero-collapse alarm。
4. config/receipt pin catalog、builder、M40-equivalent trace、dataset split/list、checkpoint 和四个 operator 的 SHA/identity。
5. 同 checkpoint/seed/epoch 跑 no-PAFT paired baseline；每 epoch 做全量 packed replay并报告 accuracy、activity rate、nominal op、byte/port-aware cost。
6. 只有 clean heldout 达 `>=3x` 且 accuracy 合格，才进入 matcher/packer RTL、VCS、DC/STA/SAIF/PTPX；之后再以同资源 baseline 做 address-timed system speedup。

在解锁前，可以做最多几十 step 的 **debug smoke** 验证梯度和内存，不得把其 checkpoint、proxy speedup 或 valid825 内部结果当作训练/论文证据。
