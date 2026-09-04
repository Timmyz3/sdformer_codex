# M25 资源受限全系统 tiling 与 cycle 门槛

日期：2026-08-22。Canonical：`results/m25_resource_bounded_tiled_cycles_r5_rowclosed_20260822`，receipt：`contracts/m25_output_receipt_r5_20260822.json`。

## 结论先行

当前 **DATE 论文 headline 的同资源 >2x 仍为 NO-GO**。现有 L16 点在回灌 M21 FIFO4 stall/drain 与 registered-result bubble 后，Local/Motion compute envelope 为 `2.051589x/2.062975x`，但 L16 使用 `160` 个 ATLIF INT8 multiplier，不能与 `96-MAC` Fixed/RQTB 当同资源。L8/80 为 `1.636274x/1.643509x`，L10/100 为 `1.780444x/1.789013x`；即使采用忽略 temporal dependency/routing 的非可执行 flat96 算术下界，也只有 `1.754677x/1.763000x`。

因此，不能再靠增加 ATLIF lanes 宣称过 2x。Local 在当前 M4 速度下，exact96 ATLIF arithmetic lower bound 为 `128,020,500 cycles`，而 2x 门要求 ATLIF 不超过 `84,618,497 cycles`，仍需 `1.5129x` 额外 ATLIF work reduction。

## 资源与 exact tiling

M23 的 `595,968,864 B` allocator capacity 是 boundary-materialized DRAM working set，不是片上 SRAM。M25 将片上固定驻留和 tile state 分开：

- 固定驻留 `52,032 B`，全部按 `96 B` 行对齐；
- 包含 M4 descriptor/accumulator/weight-response、M21 FIFO4 payload/moment state、sideband+snapshot reserve、dynamic-BN scale/offset、normalization scratch 和 DMA/replay control reserve；
- 从 frozen H67/Local5 `tile_records.csv` 恢复 exact C4 cohort identity，检查 temporal step、chunk、row、operator、call、weight-group；
- 生成 `4,450` 条 exact row-aligned first/phase-boundary/second-pass dependency record；activation 和 final partial lane 均同时保存 payload/96B padded allocation；所有 `tile_state_bytes`、fixed footprint 和 maximum simultaneous allocation 都是 96B 的整数倍；
- M21 的 13 个 dynamic-BN operator 逐名与 frozen fanout 绑定，`ceil(fanout/96)` 合计 `123` 个 lane tile；每个 operator 只生成一次 coefficient，barrier crossing 为 0。

| identity | 96 KiB | 128 KiB | 240 KiB | 408 KiB |
|---|---:|---:|---:|---:|
| H67 frozen C4 最大 tiles/cohort | 6 | 3 | 2 | 1 |
| Local5 frozen non-attention C4 最大 tiles/cohort | 11 | 6 | 3 | 2 |

H67 abstract attention 的物理容量仍 unknown；Local5 attention 为 missing nonzero，因此 Local5 只能报告 non-attention conditional capacity，不能报告 full-system capacity/cycles/speedup。

## M21 回灌

实现态只采用 `one arithmetic slice + FIFO4`：

- Local FIFO4 phase-1 incremental stall+drain：`6,098,531 cycles`；
- Motion：`6,260,784 cycles`；
- frozen 123 lane tiles，每 tile 六个 registered-result retirement：`738 cycles`；
- 只把 phase-1 incremental 与 738 bubble 加到 M7；M21 source-without-stalls 与 phase-2 replay 不再重复加入，因为分别与 M4 source 和 ATLIF service 重合；
- `three slice + FIFO40` 仍严格标为 DSE-only，不是 RTL。

## Local/Motion、memory 与 baseline 边界

Local、Motion shared-state、Motion explicit-copy 使用相同频率、带宽、SRAM 和离散逻辑接口。M23 只提供 payload bytes 与 bank-service envelope；其 tick 从未作为 cycle。`16/32/64/128 GB/s` 结果只是 ideal-bandwidth serialized/perfect-overlap sensitivity，不是 DRAMsim3 或 measured system cycle。Motion shared/copy 的 SRAM service 差异也未冒充系统加速比。

Fixed/RQTB 只有内部 cycle envelope；Prosperity/Phi-like 缺同 checkpoint/workload/address/PPA adapter，禁止直接写 numeric speedup。

## 下一硬件点

下一候选为 `M25A_EXACT96_RANK3_FACTORIZED_T10_CO_DESIGN`：训练 rank-3 factorized T10（理论 `60 vs 100 MAC/neuron`），与 M4 time-share exactly 96 lanes，并接 barrier-indexed tile DMA/replay controller 和一项 M21 result snapshot queue。远端 checkpoint audit 的 45 个 T10 matrix 显示 rank-2 energy 中位 `0.6303`、rank-4 `0.8594`、diagonal energy 中位仅 `0.0968`，所以 diagonal/现有低秩不能假装 exact。

该候选在 M25 中获得 **零 cycle/speedup credit**。必须先 fine-tune 和 accuracy admission，冻结 105 个 ATLIF matrix identity/rank factor，再做 VCS numeric RTL、同频同面积 DC A/B 和 address-timed SRAM/DRAM schedule。

## 独立打铁

独立 agent 全量复核 r5：`94/100`，限定范围 GO，`P0=0/P1=0/P2=3`。Python 3.6.8 `9/9 PASS`，从冻结合同重生成 7 个产物逐字节一致；4,450 条 plan、2,680 个 identity/budget/barrier group、144 个 SRAM/BW sensitivity point 均独立核账通过。

三个 P2 均不改变 scoped GO：

1. 两个 4KiB sideband/snapshot 与 DMA/control reserve 是显式预算合同，尚未由 RTL/DC 导出；
2. Motion FIFO4 phase-1 只有 aggregate increment，未像 Local 一样拆到逐 operator；
3. H67 replay upper 只覆盖 observed tiling traffic，abstract attention 仍有 unknown additional traffic。

所以 r5 只冻结为 frozen non-attention C4 tiling、M21-corrected compute envelope 和 ideal-bandwidth sensitivity。系统加速、FPS、能耗、PPA、DRAMsim3、Prosperity/Phi 直接数值比较继续 NO-GO。
