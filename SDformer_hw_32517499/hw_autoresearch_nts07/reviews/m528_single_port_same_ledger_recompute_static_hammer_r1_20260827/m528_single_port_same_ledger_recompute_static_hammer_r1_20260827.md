# M528 单端口同账本重算：独立 source-only 静态打铁

日期：2026-08-27  
角色：独立技术审阅；未运行 production runner、CPU 重算、EDA/GPU，未写 RTL，未修改候选与 `docs/359`  
裁决：**98/100，P0=0、P1=0；授权 root 创建一次精确绑定的 M528 CPU admission。**

## 技术结论：一次 CPU admission 可以安全创建

M528 author handoff 已满足一次 fail-closed CPU 重算的静态准入条件。作者包、基础错杀审计、M468/M473/M505 结果与 SRAM mapping 的双封均通过；analyzer、runner、execution contract、governing contract 和 `docs/359` 的实际 SHA 与冻结值一致。canonical result 和 attempt sentinel 在本审阅时均不存在。

这项准入只允许重算冻结的 H67 ep35 十样本、四层 bottleneck Conv3x3 CPU 周期/交通/容量账本。它不授权 EDA、GPU、RTL，也不把 `1.746753x` 或 `1.741232x` 升格成 RTL、PPA、系统性能或 DATE headline。生产结果仍必须经过新的独立 result hammer。

## 同一坐标与锚点是闭合的

| 项目 | 冻结值 | 静态裁定 |
|---|---:|---|
| population | 10 samples × 4 operators × 432 partitions × 3000 rows | 51,840,000 rows |
| task grain | 47 row chunks × 432 partitions × 4 operators × 10 samples | 812,160 tasks |
| execution coordinate | row64 / 8 resident banks / 128 B/cycle / CAM64 | 与密封 M468/M473 对象一致 |
| M468 strong-zero | 760,350,133 cycles | runtime exact-anchor gate |
| M473 same-coordinate bit | 757,946,784 cycles | runtime exact-anchor gate |
| M473 fused concurrent-1R1W ceiling | 389,974,420 cycles | 仅 diagnostic ceiling |
| M505 dead-write-only single-1RW | 435,293,339 cycles | runtime exact-anchor gate |
| M505 combined PVRF | 435,293,339 cycles | 周期无增益，不提名 RTL |

静态复核算术为：

- `760,350,133 / 435,293,339 = 1.7467534301047505x`；
- `757,946,784 / 435,293,339 = 1.741232213066325x`；
- `435,293,339 / 389,974,420 - 1 = 11.62099785929549%`。

analyzer 不依赖事后除法来生成结果：它重新读同一密封 row ledger，逐 task 调用冻结 M504/M505 模型，再要求 sample-major 总数逐项等于上述锚点。M473 ceiling 的距离被明确排除在 CPU 生死门之外；实现候选的主分母是 M468 strong-zero 和 M473 same-coordinate bit。

## 213,376 B 容量与交通没有藏项

容量重算为：

`203,008 - 9,216 + 18,432 + 1,152 = 213,376 B`

其中 `18,432 B` 是九颗 `128×128-bit 1RW SP` generated macro 的真实物理 capacity，`1,152 B` 是保守 macro-rounded liveness metadata。相对 `240 KiB = 245,760 B` 尚有 `32,384 B` 余量。response queue、descriptor directory、matcher source store、resident psum/valid、ping-pong ownership 均有 capacity obligation mapping；matcher/scheduler 的 standard-cell area 被明确留给后续物理门，未折算成“免费 SRAM”。

交通表分开输出 weight DRAM、source SRAM、descriptor write/search/scan、parent read/write、DMA 和 commit。dead-only 一块的 parent accesses 为 `26,438,462`，按八个 output block 与 144 B/vector 扩展后为 `30,457,108,224 B`；combined 为 `30,175,621,632 B`。这些值被明确标为 logical access traffic，不是 SRAM/DRAM energy。

守恒门覆盖：

- arithmetic issue = residual nnz + exact-parent issue；
- parent reads + same-address forwards = parent edges；
- dead-only writes + dead elisions = active rows；
- combined writes + all elisions = active rows；
- trace rows和 commit cycles 与冻结合同一致。

因此，这次重算不会用少收 parent traffic、少算 completion 或把逻辑流量冒充能量来制造倍率。

## sample-major 与 operator-isolated 没有混粒度

analyzer 明确构造两个不同分布：

1. `sample_major`：每个样本把四个 operator 的 task 按冻结顺序连成一条 pipeline，只加一次 `96,000-cycle` commit；十行之和是唯一 aggregate 来源。
2. `operator_isolated`：每个 sample/operator slice 都重启 pipeline 且不加 commit；四十行只用于异质性，不进入 sample-major 总数。

两个粒度都分别报告 cycle 和 ratio 的 arithmetic mean、geometric mean、minimum、maximum、population CV；两个主 speedup 另报 ratio-of-sums。operator-isolated 的 M504、dead-only、combined 三个周期还逐行对照旧 M505 密封 CSV，禁止把四十个重启切片求和冒充连续四算子运行。

## runner 的一次性边界足够严格

runner 要求 caller 同时固定：runner SHA、independent admission path/SHA，以及 admission 内的 analyzer、execution contract、governing contract、author outer seal 和 `docs/359` 身份。它还执行以下门：

- canonical output 与 attempt sentinel 都必须为空；
- output/worker override 禁止；runner 固定 `3 workers / chunksize 2`；
- 本用户的 DC/Formality/PT/VCS/simv 冲突会拒绝启动；
- commit headroom、MemAvailable、SwapFree 和 user-slice OOM counters 必须连续三次达标；
- attempt sentinel 在 Python production launch 前原子消耗；
- 失败或不完整结果进入双封 quarantine；
- 成功 raw result 双封，并标注 `paper_admitted=false`、`system_speedup=false`，等待独立 result hammer。

本审阅时 canonical output 与 attempt sentinel 均不存在，`docs/359` 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## 非阻塞 P2

1. 当前密封 SHA 和 exact anchors 已令 M468/M473 selection 唯一，但 analyzer 没有对每个 selected object 再逐字段 `require(row_tile=64, banks=8, BW=128, CAM=64)`。独立 result hammer 应再次核坐标；若以后修 analyzer，可加入显式字段门。本轮不得为此改动已封候选。
2. traffic CSV 有 M468、M473 fused、M505 dead-only/combined 行，但没有冗余的 M473 bit 行。这不影响 CPU 周期裁决；最终 paper-facing energy ledger 应显式补一行，或注明 bit 与 strong-zero 的 weight/source/DMA/commit 相同且无 parent scratch。

两项均为 P2：当前 frozen object SHA、anchor、traffic identity 已消除本次运行的歧义，不阻塞一次 admission。

## 授权边界与下一步

root 现在可以创建一次满足以下字段的 admission：

- schema：`m528_single_port_same_ledger_static_admission_v1`；
- status：`AUTHORIZED_ONE_M528_CPU_PRODUCTION_RUN`；
- authorization：CPU=1、EDA=0、GPU=0、RTL=false；
- 固定 runner `a31d891ab83a8c87fa98f31cabbc7a81174362ef9b4f469fe0a3220b80711531`；
- 固定 analyzer `c611f8c98253e44ccf93743d47476da0adc9835b013b247bc4e2d821953afb8a`；
- 固定 execution contract `910c804a9a9df13395ab4f6b2ef5988ea0dee56ab7e52a21f887fa8fe0d73a34`；
- 固定 governing contract `d0e3728f3a9991cf97c6af88181cd51996e457228b120f2b706a8986caf9ca51`；
- 固定 author outer-seal file `9c29e7950b1d6563e78004acac54a858fe8d8821e784500ff8f9cabbe2d4521a`；
- 固定 `docs/359` SHA `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

准入之后只准 exact reviewed runner 消耗一次 CPU production attempt。raw result 出现后必须另做独立双封 result hammer；在此之前，M528 仍无 paper-admitted 性能数字。
