# M705 fresh result hammer：M699 三序列 decoder payload

裁决：**GO_M699_PAYLOAD_DENSITY_AND_OBSERVED_S3_STABILITY_ONLY__P0_0_P1_0**，98/100，P0/P1/P2 = **0/0/2**。

M699 canonical 的 30 个输入样本、120 个 decoder hook 和 261,090,000 byte bitpack 全部通过独立审计。审计没有运行 GPU、模型、RTL、周期模拟器或 EDA。合法准入仅包括：同一 H67 ep35 checkpoint 下的 payload 身份、密度，以及这三个指定 S10 cohort 上观察到的跨序列密度稳定性；不准入 accuracy、cycles、speedup、system、ours、energy、PPA 或 headline。

## 身份与完整性

- canonical 为 124 个普通文件、2 个目录、0 个符号链接；顶层 complete double seal 通过，manifest SHA-256 为 `e2d7c92a...`，outer-seal-file SHA-256 为 `eaf975a9...`。
- attempt 已在 GPU one-shot 前消费，post-capture runner rehash、精确 argv/env/GPU receipt 和 exit-zero completion path 均通过；M700、M686、M692 前驱封印链通过。
- checkpoint load 为 missing=0、unexpected=0、overlay missing=0、overlay unexpected=0。30/30 输入 NPY 的 SHA-256、shape、dtype、确定性等距选择和顺序全部重算通过。
- 四个 ConvTranspose2d 的几何与 weight content identity 与同 checkpoint、已准入的 M686 payload 完全一致。

每个 bitpack 都独立重算 SHA-256、popcount、shape/bytes 和尾位；随后逐块还原原始 FP32 字节并重算 raw-content SHA-256。D0/D2/D3 使用精确 `0.0/1.0`，D1 使用冻结的 runtime theta `0.9999954104423523`（IEEE-754 uint32 `1065353139`）。120/120 均精确相等。所有记录的 thresholded/rounded/coerced 均为 false。

## 密度结果

| sequence | D0 | D1 scaled | D2 | D3 | 四模块按元素加权 |
|---|---:|---:|---:|---:|---:|
| interlaken_01_a | 17.6559% | 18.2246% | 17.6371% | 28.0182% | 23.2701% |
| thun_01_b | 17.8757% | 18.3785% | 17.5044% | 28.0808% | 23.3032% |
| zurich_city_12_a | 18.1036% | 18.4550% | 17.3228% | 28.2732% | 23.3830% |

三序列 max-min 绝对跨度为：D0 **0.4478 pp**、D1 **0.2304 pp**、D2 **0.3144 pp**、D3 **0.2550 pp**；最大值是 D0 的 **0.4477756 个百分点**。四模块按元素加权密度跨度为 **0.1129679 个百分点**。因此可以陈述“这三个指定 S10 cohort 上 decoder 输入密度稳定”，不能外推为 DSEC 总体分布或统计置信结论。

D1 不是精确二值：theta 与 `1.0` 的 bit pattern 不同。这里仅准入 30 条精确 `{0, runtime theta}` bitpack；不准入 folded-weight deployment 或 decoder numerical equivalence。

## 攻击结果与边界

bitpack 单字节修改、成员删除均被 seal 拒绝。私有副本即使一致重封，record 重排和把 D1 route 偷换为 `EXACT_BINARY_BITPACK` 仍被独立语义校验拒绝，同时改变两个外部冻结根。

两个 P2 是范围提醒，不是 payload 缺陷：样本仅为三个确定性 S10 cohort；D1 仅有 scaled-binary 表示准入。禁止由本 review 生成 performance、accuracy 或 paper headline。
