# M188 独立打铁评审

结论：**88/100，`PASS_AS_M189_INTEGRATION_CANDIDATE_ONLY`**。M188 的 K7 最弱-bank延后调度功能成立，值得进入一次 M189 真实 `8→7 compactor + 7-lane signed accumulator` 的同口径筛选；但 M188 单独不胜 M184/K8，不能替换 K8，更不能升格为 FC2、FFN、physical、system 或 headline 结果。

## 独立证据

- sealed M188 VCS input/output manifest 全通过；sealed DC input/evidence manifest 全通过。
- 用 seed `188026` 独立复跑 sealed VCS：15 header、60 descriptor、320 bitmap event、59 unique group、292 replayed result 全守恒；双窗口同时关闭 183 cycle、release/refill 1 次、window replace 3 次；21/21 SVA cover 非零，0 assertion failure。
- 新写的独立 VCS bench 覆盖 255 个非空 structural-bank mask，逐 group 核对 mask、source count 和每个 bank 的 channel；再覆盖最低 population 平局、唯一最弱 bank、stall hold 与 zero-descriptor fail-close。结果 272 group、5 次 all-eight tie/weakest deferral，0 mismatch。
- 独立 selector 程序不 import 生产 analyzer：穷举每 bank population 0..4 的 390,624 个非零向量，并检查 100,000 个 population 0..96 的确定性随机向量，全部达到 `G=max(max_bank,ceil(total/7))`，0 mismatch。
- `docs/359` SHA 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

### 为什么 weakest-bank greedy 达到 K7 下界

每组每 bank 最多取一个、全组最多取七个，因此至少需要 `L=max(max(q_i),ceil(sum(q_i)/7))` 组。当八个 bank 都非空时，延后当前最小的一个并服务其余七个：总量减少七，且被延后的最小队列不可能等于 L——否则八个队列都会等于 L，与总量不超过 `7L` 矛盾；所以一轮后最大队列和总量下界都至多为 `L-1`。当活跃 bank 不超过七个时全部服务，各队列和总量同样降到下一层下界。归纳即可在 L 轮完成。

## 同口径 DC 与密度重算

3.0 ns、flattened、ideal clock、ZeroWireload、0 macro 的 sealed DC 数字为：

| frontend | K | area (µm²) | cells | seq | levels | critical | setup slack |
|---|---:|---:|---:|---:|---:|---:|---:|
| M180 | 4 | 14,417.928053 | 22,209 | 1,882 | 161 | 2.53 ns | 0.0000 ns |
| M184 | 8 | 10,026.828029 | 14,665 | 1,915 | 136 | 2.52 ns | 0.0023 ns |
| M188 | 7 | 10,417.680032 | 15,637 | 1,914 | 148 | 2.53 ns | 0.0002 ns |

相对 M184/K8，M188 面积增加 **390.852003 µm² / 3.898062%**，而 K7 吞吐仅保留 K8 的 **99.911221%**；所以 standalone frontend throughput/area 是 **0.961627×**，即输 **3.837262%**。M188 **没有** supersede M184。

相对 M180/K4，K7 exact schedule 是 **1.305919×**，M188 frontend 面积少 **27.744958%**，两者相乘得到 **1.807375× conditional frontend schedule-throughput/logic-area**。这只是部分模块、同频、理想 memory/response 条件下的筛选比，绝不是 FC2/FFN/全网加速比。

M188 的 0.0002 ns setup slack 只有 0.2 ps，且仍是 ideal-clock/zero-wireload pre-macro；它表示“该逻辑筛选在 DC 报告中未违例”，不表示物理时序稳健。

## 为什么仍值得做 M189

K7 的潜在收益不在 M188 frontend，而在少一条 weight/arithmetic lane。按 standalone M184+M185 总面积 37,156.643801 µm² 做纯筛选，为补偿 K7 的 0.088779% 吞吐损失，`M188 + M189` 必须低于 **37,123.656593 µm²**；因此 M189 的 compactor+accumulator 必须低于 **26,705.976561 µm²**，比 M185 至少少 **423.839211 µm² / 1.562264%**。最终必须以 flattened composed island 重算，因为跨模块共享或重复会使 standalone 求和失真。

## 打铁优先级

### P0

- 不得把 M188 单独选为 K7 胜出结果；它对 M184/K8 的 standalone density 是负收益。
- M189 必须真的接八个 structural bank 并实现 `8→7` compactor 与七路 signed accumulation，不能用预先 packed 的七路输入隐藏 routing 成本。
- 做 `M188+M189` flattened island 对 matched flat K8 的 VCS/SVA/DC；若不能保持一拍一组，或总面积没有跨过 throughput-adjusted 阈值，立即退回 K8。

### P1

- 将冻结 M187 的 120 个 payload 通过 composed RTL/transaction bridge replay；现有 VCS 证明了协议和调度策略，但没有逐 payload 跑 RTL。
- 单列 selector/compactor 的 levels、fanout 和 timing；M188 比 M184 多 12 个 mapped levels，3 ns setup margin 极薄。
- M186 已发现 reset 后 delayed untagged response alias。M188 没有 response port，无法关闭它；composition 必须增加 epoch/identity 或 flush-ack quarantine。

### P2

- flat screen 胜出后再做 Formality、PT/SAIF/PTPX；12.5% 只能称 nominal lane-response width 缩减。按 replay cycles 计的 full-width bit-cycle 缩减为 **12.404103%**，有效 weight value 未减少，也没有证明 SRAM capacity/energy 降低。
- `1.807375×` 永远保持 conditional frontend 指标，不得写成硬件、FC2、FFN 或系统倍速。

机器可读评审见 `m188_independent_hammer_review_r1.json`；独立 VCS、selector proof 和重跑 sealed VCS 的日志均在本目录。

