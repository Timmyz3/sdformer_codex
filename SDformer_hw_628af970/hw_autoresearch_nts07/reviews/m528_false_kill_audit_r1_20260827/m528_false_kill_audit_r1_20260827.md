# M528 错杀独立审计：M505 单端口 product capture 有条件重开

日期：2026-08-27  
角色：独立只读审计；未运行 EDA/GPU，未修改被审文件或 `docs/359`  
裁决：**93/100，推翻“永久杀 M505 RTL”；GO 一次同账本 CPU 重算，通过后只准一个 dead-write-only 1RW RTL 版本。**

## 1. 结论

用户提供的分析在最重要的一点上成立：M505 的旧裁决把两个不同问题混在了一起。

- 算术裁决没有错：`435,293,339 / 389,974,420 - 1 = 11.6210%`，确实没有通过“距离 M473 fused 理想点不超过 5%”的合同门。
- **永久项目裁决错得过严**：M473 的 `389,974,420` 使用 concurrent-1R1W（同拍一读加一写）机会模型，physical scratch 仍为 OPEN，不能作为 DATE 对可落地实现的唯一淘汰分母。
- 在真正可用的两个冻结分母上，M505 dead-write-only 是 `760,350,133 / 435,293,339 = 1.746753x`（M468 strong-zero）和 `757,946,784 / 435,293,339 = 1.741232x`（same-coordinate bit）。这两个数值得到同一 trace、tile、bank、带宽和完成语义的支持。

因此，`NO_GO` 只应解释为“没有捕获 M473 理想 fused 上限的 95%”，不能解释为“单端口机制不值得做”。本审计允许一次 fail-closed 重开，但不把 1.74x 提前升格成 RTL、物理或系统结果。

## 2. M505 与 M468/M473 是否真正同账本

| 维度 | M468 strong-zero | M473 same-coordinate bit | M505 dead-write-only | 裁定 |
|---|---:|---:|---:|---|
| checkpoint / trace | H67 ep35, S10 | 同左 | 同左 | 相同 |
| population | 4 Conv, 51.84M rows | 同左 | 同左 | 相同 |
| task order | sample/op/chunk/partition | 同左 | 同左 | 相同 |
| row tile | 64 | 64 | 64 | 相同 |
| resident output banks | 8 | 8 | 8 | 相同 |
| external BW | 128 B/cycle | 128 B/cycle | 128 B/cycle | 相同 |
| weight DRAM | 9,069,207,552 B | 9,069,207,552 B | 继承同值 | 相同 |
| source SRAM movement | 103,680,000 B | 103,680,000 B | 继承同值 | 相同 |
| commit | 960,000 cycles | 960,000 | 960,000 | 相同 |
| arithmetic | direct bit/zero | direct bit | exact signed parent/residual | 候选算法对象不同、输出等价 |
| parent scratch | 无 | 仅 product 点使用 | 1RW SP，read XOR write | 候选已收费 |
| cycles | 760,350,133 | 757,946,784 | 435,293,339 | 1.7468x / 1.7412x |

M505 不是把 single-port latency 忽略掉：它逐 task 模拟一周期同步响应、queue+pending 容量 2、无 same-cycle consume credit、禁止 read-before-store，并限制每拍最多一次 macro read **或** write。它把该 issue window 乘八个 output banks，再放进与 M473 相同的 preprocess/work pipeline 和固定 commit。

PWP/psum 口径也没有偷换。M468 strong-zero 不生成 parent PWP；M505 的 parent scratch 是片上 store，并在端口周期中收费。八 output block 合计的 dead-write-only parent movement 是 **30,457,108,224 B**；它不是 DRAM，但后续能量必须定价。Resident psum、weight DMA 和 completion 保持原坐标。

仍需强调：这里的“same-budget”是同一 240 KiB 容量上限和相同执行资源轴，不是已经完成了 matched total-area RTL。Matcher/CAM/scheduler 尚未整体物理化，所以当前只准 CPU cycle claim。

## 3. 240 KiB 与真实 1RW 宏

M473 的 `203,008 B` macro-rounded 总量把 parent scratch 按理想 64-depth、`9,216 B` 收费。已有 generated view 是 `128x128b 1RW SP`，实现 1152-bit word 要九颗，物理 capacity 为 `18,432 B`，面积 `78,825.2454 µm²`。

保守替换并额外加一个 64-row live bitmap 的 macro-rounded slice：

`203,008 - 9,216 + 18,432 + 1,152 = 213,376 B`

仍低于 `240 KiB = 245,760 B`，剩余 `32,384 B`。所以“generated 单端口宏使点超 240 KiB”不成立。这个检查只证明容量可容纳，不证明 matcher/adapter 的面积、布线、功耗或 3 ns 全闭合。

## 4. 旧门为什么是错杀，而不是旧计算错误

旧合同把五门做成合取，其中主门为 M505 相对 M473 ideal concurrent-1R1W 的税不超过 5%。合同执行和旧 hammer 都正确地得到 FAIL。问题是后续文字把它扩成了 **永久停止所有同族 RTL**。

M473 的状态实际上是机制与实现分离：

- exact subset mapping / topological schedule：CPU GO；
- M474 fused micro-pipeline：directed VCS GO；
- M475：`37,316.29 µm²` logic-only、3 ns setup 恰为 0，宏数为 0；
- 64x1152-bit concurrent-1R1W scratch：没有 generated DP view，PPA OPEN；
- M477 双 response-slot wrapper：DRC fail，诊断面积 `42,370.65 µm²`，未准入；
- M473 `1.9436x`：仍是未物理化 fused opportunity，不是 admitted hardware speedup。

所以旧 5% 门可以继续作为“capture efficiency”诊断，却不应继续是 DATE Accept 的单一生死门。可落地的 1RW 点应对照 strong-zero/bit，并由总面积效率和能量决定是否最终存活。

## 5. 重开哪个版本

只重开 **dead-write-only**，不重开 combined PVRF。

| mode | cycles | one-block parent accesses |
|---|---:|---:|
| M504 all-write 1RW | 456,016,645 | 43,796,329 |
| M505 dead-write-only | 435,293,339 | 26,438,462 |
| M505 combined PVRF | 435,293,339 | 26,194,116 |

combined PVRF 相对 dead-write-only 只再少 `244,346` 次写，即 dead-only access 的 `0.9242%`，周期收益严格为零，却要求更丰富的 `0/1/2+` refclass。RTL 应只保留 dead/live 判定和已有 same-address RAW forward。这既保住 1.7468x 的 CPU 点，又减少元数据和控制风险。

## 6. 唯一下一门

同目录合同 `m528_single_port_same_ledger_recompute_contract_r1_20260827.json` 冻结一次 CPU 重算，要求：

1. 同时列 M468 strong-zero、M473 same-coordinate bit、M505 dead-only/combined 和 M473 ideal ceiling；
2. 同时列 logical/macro-rounded capacity、weight DRAM、source/descriptor、parent scratch、DMA、commit 与 conservation；
3. 给 per-sample/per-operator 分布及 arithmetic/geometric mean、min/max；禁止把 operator slice 简单相加冒充 sample-major pipeline；
4. CPU 门为：身份/守恒全过、物理 capacity 不超过 240 KiB、相对两个可落地分母均至少 1.50x、不得回退冻结 435,293,339；
5. M473 11.621% capture gap 只作诊断，不再参与 RTL 生死。

CPU 过门后只准一个 single-port RTL 版本。物理准入必须同时满足：generated 九宏集成、matcher/scheduler/directory/queue/psum 边界全收费；3 ns setup/hold/cap/transition/fanout 全 clean；Formality pass；trace recurrence 距 435,293,339 不超过 1%；相对两个分母仍至少 1.50x；总 throughput/area 至少 1.10x；四 Conv energy 至少降 20%或 EDP 至少 1.25x。任一失败即停止，不开第二结构。

## 7. 对用户其余“错杀”判断的校正

### FC1 M224

“1.5x 门埋掉了 1.19x 局部机会”部分成立。M224 spatial K1 相对 raw K1 为：

- D96：`1.190252x`，十样本 `1.162498--1.217234x`；
- D128：`1.176055x`。

这可以保留为 C2/FC1 的局部消融或 parent-delta 支撑，但不是独立 headline。K2/K4/K8 bank 变体在公平资源下变慢，仍应保持 KILL。

### M229 F4

“真实 trace 从未跑”已经被后续里程碑淘汰，是陈旧判断：

- M229 directed service island：F4/F1 `2.586957x`；
- M230 已把冻结 H67 100-record trace 放入 fixed-latency recurrence，latency=2 raw F4 相对 same K8/F1 为 `2.068357x`；
- M481 进一步做 full-width resource model；
- M482 用完整 96-lane RTL handshake recurrence 在同一 100-record workload 上得到 `1.359897x`，ideal envelope sensitivity 只有 `1.044983x`，因此 compact F2 点没有进入 DC。

所以 M229 不是“未完成而误杀”；它是早期 2.59x 被真实 lifecycle/银行冲突逐层压到 1.36x。M224 的 1.19x 可留作支撑，M482 compact 主点维持 NO-GO。

### G8 FFN skip

“缺 post-BN2 F(x)”只描述 M383 当时状态。M460R5 后来已抓取 120 records / 5,580,000 tokens 的 post-BN2 oracle reduction；M462R2 冻结网格结果为：

- T10 all-site gate：0 saved cycles；
- strict-token 最好只省 2,951 cycles；
- 对 620,302,905 envelope 的 ideal ceiling 为 `1.000004757x`。

因此 G8 的数据缺口已经关闭，并形成可信负结果；不应再抢 GPU 或 RTL 复活。

## 8. 打铁评分与 P0/P1/P2

评分：**93/100**。

- P0=1：旧 permanent-kill 将未物理化 ceiling 当成唯一生死分母，需按 M528 受限重开。
- P1=4：M505 capacity 总表未显式替换 generated 深度；matcher/scheduler/macro 未整体 PPA；30.457 GB scratch movement 未定价；当前仅一序列四 Conv。
- P2=2：必须区分 concurrent-1R1W 与 single-port-1RW；combined PVRF 不得包装成周期机制。

最终边界：**GO 一次 CPU 同账本重算；通过后 GO 一个 dead-write-only 1RW RTL。当前 1.7468x/1.7412x 仍不是 RTL、PPA、系统或 DATE headline。**
