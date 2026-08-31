# M336：4-bit token-dynamic cumulative-budget selector 预设计

结论：方案 B 已形成一个固定宽度、可证明数值界、能接现有 K8 fixed-bank issue 协议的预设计，评分 `84/100`，`P0=0、P1=4、P2=3`。本轮只准入下一步 CPU schedule/DSE；**不准入 RTL、VCS、GPU、DC 或论文贡献**。没有修改现有合同、RTL 和 `docs/359`。

## 机制

对每个冻结的 `(layer, destination-group, source)`，离线计算

`beta = max_j_in_group |Wq[j,source]|`

并存一个 4-bit 保守 code。全局硬连线的 16 个上界为：

`U={0,9,17,26,34,43,51,60,68,77,85,94,102,111,119,127}`。

编码取满足 `beta<=U[c]` 的最小 code。运行时每拍接收最多 8 个升序 active source ID，查 8 个 code，稳定装入 16 个 bucket。bucket 按 code 从低到高、桶内按 source ID 排序。若当前 conservative sum 为 `S`，一词有 `k` 个相同 code 的 source，则一次计算可丢前

`d=min(k,floor((B-S)/U))`

个；`U=0` 时该词全部可丢。第一次 `d<k` 就找到 cutoff，其余全部保留并送 K8。因为 `U>=beta`：

`|sum dropped Wq[j,i]| <= sum beta_i <= sum U_i <= B`。

该证明只覆盖逐 destination-row 的 INT8 accumulator。`B>0` 仍必须跑 GPU accuracy，因为反量化、BN、ATLIF 阈值和 recurrent state 都可能放大局部误差。`B=0` 默认直接 bypass selector，把全部 active source 送 legacy K8，零 selector 开销且 bit-exact；`beta=0` source 是额外无损子集，但不是默认 B0 路径。

## 静态与动态字段

- 逐权重静态：4-bit beta code、固定表 base、source/group geometry、`source_id mod 8` bank mapping、全局 U 表。
- 模型/层级静态配置：当前 M328-compatible 布局使用一个全局 17-bit B；exact-bypass、module/group descriptor 和 group-major mode 固定。若 accuracy 后改成逐层 B，仍不随 token 改变，但 11 层共 33 B CSR 必须另计。
- 逐 token 动态：active IDs、16 桶内容/指针、17-bit cumulative sum、cutoff、drop/keep 计数、bank reservoir 和 ping-pong context。

新颖性所需的 token dynamism 来自 active subset 消耗预算不同：同一个静态 source/group pair 必须在不同 token 上出现一次 drop、一次 keep；不是让 B 本身逐 token 变化。

## 固定 footprint 与端口

| 项 | 固定逻辑容量 | 端口/带宽 |
|---|---:|---|
| 持久 4-bit pair table | 498,816 B | 1R、256 bit/cycle 顺序预取 |
| 当前 group beta scratchpad | 432 B | fill 256 bit/cycle；capture 8R×4 bit/cycle |
| 16 bucket、448-active、双 context | 19,232 B | 16 个逻辑 `112×84` 1R1W bank |
| kept suffix reservoir | 32 B | 最多 8 input、8 个固定 bank output |
| B CSR | 3 B rounded | 配置口 |

持久 pair table 正好是 997,632 pair ×4 bit=`498,816 B`，相对一位 mask 的 `124,704 B` 为 `4.000x`，没有另存 source ID 或 permutation。算上 scratchpad、双 context、reservoir 和 CSR 后，逻辑总容量是 `518,515 B`，即 `4.158x`。这些数都不含 SRAM 宏对齐、ECC、decoder、spare 和多读 mux；不能当物理面积。

498,816 B 大表不能直接提供 8 个任意地址读口。可实现边界是：以 group-major 顺序，用 1R/256-bit 主表把当前 group 的最多 864 个 code 预取到 432 B distributed 8R scratchpad。最大 group 需 14 cycles；完整 2,904 个 group 的一次 sweep 是 15,696 cycles。若使用 token-major、每个 task 都切 group，这条路线立即失效。

每个 context 的 bucket word RAM 为 `16×56×84 bit=9,408 B`；84 bit 是 8 个 10-bit ID 和 4-bit valid count。每桶另有一个 partial word。每拍一个桶最多进 8 个 ID，所以 partial word 最多形成一个满词并留下 0–7 个 ID，一写口足够。两个 context 共用 16 个按 context 分区的 `112×84 1R1W` bank：一个 context capture 写，另一个 drain 读。

## selector 周期

对一个 token/group task，令 active 数为 A、bucket c 数为 `n_c`：

- capture：`C=ceil(A/8)`；
- drain：`R=sum_c ceil(n_c/8)`，空桶不计；
- fragmentation：`F=R-C`；至多 16 个非空桶、每词 8 ID 时，恒有 `0<=F<=14`。

M328 最大 A=448，因此 `C<=56`、`R<=70`，完全串行 selector `C+R<=126` cycles/task。M328 group4 population 共 1,013,760,000 task，capture 精确为 4,808,840,856 cycles，平均 `4.743569` cycles/task。由于 M328 没有记录新 code 的 16 桶 population，本轮不能伪造 drain 平均值，只能给 `4.743569–18.743569` cycles/task；孤立串行 selector 平均为 `9.487139–23.487139`。

## 用 M328 ledger 重算

冻结 group4/B1024：baseline K8 `6,681,676,272`，旧 exact-beta kept issue `3,787,146,191`，capture `4,808,840,856` cycles。旧 kept issue 只作为乐观参考；4-bit 上界会改变排序、drop prefix 和 bank population，必须重跑。

| 模型 | 乐观 cycles | 相对 baseline |
|---|---:|---:|
| 三段完全串行 `C+R+Q`，取 `R=C` | 13,404,827,903 | 0.4985x |
| 同 task drain/issue 融合 `C+max(R,Q)` | 9,617,681,712 | 0.6947x |
| 跨 task ping-pong `max(C,R,Q4)` | 4,808,840,856 | 1.3895x proxy |

1.15x 的总周期上限是 `5,810,153,280`。因此新 CPU schedule 至少要同时满足：

- 4-bit kept K8 issue `Q4<=5,810,153,280`；
- bucket drain `R4<=5,810,153,280`；
- bucket→bank reservoir→K8 的真实联合 schedule 也不超过该上限。

仅看 R，允许的总 fragmentation 是 1,001,312,424 cycles，即平均最多 `0.987721` cycle/task。再保守计 2,904 个 group 各 574-cycle pipeline fill/drain，以及精确 15,696-cycle prefetch，stage 上限收紧到 `5,808,470,688`，fragmentation 平均最多 `0.986062` cycle/task。换句话说：16 桶 padding 平均必须控制在约一拍以内，否则 1.15x 自动失败。

## 与现有 K8 共存

M184 是 FC2 descriptor-window frontend，不能原样拿来做 M336 selector。可复用的是它输出到 M186 request slot 的 fixed-bank group 协议：`valid/ready、tag、group、source_count、bank_valid[7:0]、source_channel[0:7]`。

M336 drain 后接一个 16-entry elastic reservoir，每拍为 8 个 `source_id mod 8` bank 各选最早一个 ID；同 bank 冲突留在 reservoir。route mux 在 legacy M184 group 和 M336 group 之间选择，M185/M186 后端协议保持不变。后续仍要 VCS 证明参数化 destination-lane width、Acc 语义、反压和 exactly-once。

最小 RTL 边界是 5 个模块：beta group prefetch/scratchpad、bucket16 ping-pong builder、cumulative cutoff drain、K8 bank reservoir adapter、legacy/M336 route mux。本轮没有写这些 RTL。

## VCS/DC 后续验收

VCS 必须使用 exact-SHA 合同和 SVA，至少覆盖：B0 legacy miter、16 个 code 边界、词内 budget crossing、U=0、单桶/16桶、448 active、同 bank 冲突、随机反压、ping-pong、group switch，以及 `>448`、非升序 ID、stale context、partial reset 等 fail-closed 攻击。核心断言是 accepted=`drop+keep` exactly once、`U>=beta`、sum≤B、drop 是 stable prefix、cutoff 后不再 drop、B0 不 lookup 且全量输出。

DC 必须把 8R scratchpad mux、16-way compactor、prefix budget 和 bank reservoir 全部计入；目标为同一 TSMC28/3 ns、setup/hold WNS≥0。logic-only DC 不能当 paper PPA。持久 beta 表、bucket 1R1W bank 和 scratchpad 没有目标宏前，面积、功耗、能量和 EDP 均保持 false。

## 新颖性碰撞

- Prosperity：邻近点是 product sparsity、exact/partial pattern reuse、TCAM/sort/dispatch；M336 不做 pattern/product cache，而做 token-active 的 destination-group bounded prefix。没有发现公式同构，但 bucket/sort/skip 不能单独算创新。
- Phi：邻近点是 hierarchical pattern/codebook、PWP 和 PAFT；M336 的 4-bit code 是 weight bound，不是 spike pattern。两者都有 static code 驱动近似 work removal，必须做 accuracy 和 claim chart。
- Bishop：邻近度最高，其 BSA/ECP 做 approximate activity removal 和 error reasoning。但 M336 没有 TTB dense/sparse 双核、BSA 训练或 Q/K/S/V/Y bundle pruning，而是单 K8 路径上的 per-group INT8 cumulative prefix。现有本地材料不足以允许“首次”主张。

条件可辩护的只有组合机制：**weight-static 4-bit conservative destination-group bound + token-active cumulative cutoff + same-source cross-token witness + fixed-bank bit-sparse execution**。16 bucket、双缓冲、8 lookup、bank FIFO 和 codebook 都不是独立贡献。

## 问题和下一步

- P1：没有 4-bit bucket population、Q4 和联合 schedule；1.3895x 只是 proxy。
- P1：432 B 8R scratchpad 的 3 ns timing/area 未证，且依赖 group-major。
- P1：19,232 B 1R1W bucket store 没有目标宏。
- P1：B>0 没有 GPU accuracy。
- P2：exact 4x 只对固定 11-module layout 成立；泛化 descriptor/codebook 必须另计。
- P2：只复用 M184/M186 的下游协议，不是原样复用 frontend。
- P2：Bishop 仍是高风险邻近工作，需 primary-paper formula claim chart。

下一里程碑只能先跑 CPU：按本 README 的固定 U 表，输出每 task 的 `n_c/C/R/cutoff/dynamic witness/kept-bank population/reservoir stalls`，并执行真实 ping-pong schedule。只有 bound=0 violation、B0 exact、witness>0 且总周期≤5,810,153,280 时，才允许申请 GPU modified-forward；RTL/VCS/DC 仍在 GPU gate 之后。
