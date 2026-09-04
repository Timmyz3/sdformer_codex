# M105 bounded raster-window transpose：独立性能前置审计

日期：2026-08-24

结论：**88/100，P0=0、P1=5、P2=5。** 数据和算术层面对 M105 继续做模块设计是 GO；任何 scheduled-cycle、physical、system 或 headline 升级均为 NO-GO。

## 独立重建范围

- 直接读取 M40 heldout samples 5–9 的 20 个 raw support-mask payload，并按 `timestep → output_y → output_x` 重建自然栅格顺序。
- 每个 sample/operator 有 432 个 16-source partition；合计 8,640 phases，每 phase 3,000 rows。
- 使用 M72 的 16-center codebook和完全相同的 `(Hamming distance, numeric center, index)` tie-break。
- 从四份 exact INT8 weight payload 独立重建 M78 PWP width catalog，并应用 cap11 block-local escape。
- 仅在每个 phase/partition 的固定窗口内，把 correction/fallback events 按 `(partition-local source, output block)` 合并为最多 128 个 group。
- 条件 token 公式：每 active group 三个 32-byte weight-load token，加每 event 一个 token；最后加冻结的 cap11 PWP service ledger `226,222,255`。

## 守恒

- baseline source events：371,461,096，精确复现。
- correction/fallback events：188,148,490；其中 beneficial correction 105,998,693，non-beneficial bit-sparse fallback 82,145,384，cap escape fallback 4,413。
- cap escape rows：362；assignment rows：7,371,217。
- PWP uses：8b 11,164,284；9b 32,360,036；10b 13,936,011；11b 1,509,043。
- PWP service ledger：226,222,255。
- 每个报告窗口的 event、descriptor-event 和全量 event 三路守恒均为 188,148,490。

## 主要结果

| Window rows | Active groups | Candidate+PWP envelope | Conditional ratio | Events/window max / p95 / p99 | Events/descriptor max / p95 / p99 |
|---:|---:|---:|---:|---:|---:|
| 1 | 188,148,490 | 978,816,215 | 1.1385× | 112 / 32 / 48 | 1 / 1 / 1 |
| 16 | 89,329,137 | 682,358,156 | 1.6331× | 1,088 / 344 / 472 | 16 / 5 / 7 |
| 43 | 46,867,834 | 554,974,247 | 2.0080× | 2,672 / 888 / 1,200 | 41 / 10 / 14 |
| 64 | 35,140,002 | 519,790,751 | 2.1439× | 3,480 / 1,288 / 1,736 | 60 / 14 / 19 |
| 256 | 11,800,032 | 449,770,841 | 2.4777× | 10,416 / 4,392 / 5,664 | 179 / 41 / 56 |
| 294 | 10,395,056 | 445,555,913 | 2.5011× | 11,128 / 5,032 / 6,416 | 188 / 47 / 63 |
| 1,024 | 3,292,280 | 424,247,585 | 2.6267× | 22,992 / 14,464 / 16,736 | 340 / 128 / 160 |
| 3,000 | 1,105,920 | 417,688,505 | 2.6680× | 48,592 / 39,056 / 42,680 | 695 / 344 / 409 |

完整的 1/4/16/43/64/256/294/1024/3000 表见 `m105_window_summary.csv`。

## 2× 与 2.5× 门槛

- 2× 的最小窗口是 **43 rows**：42 rows 为 1.9976939×，43 rows 为 2.0079910×。
- 2.5× 的最小窗口是 **294 rows**：293 rows 为 2.4996327×，294 rows 为 2.5011076×。

这两个“最小”都非常脆弱：

- 43-row 相对 2× 门槛只剩 2,217,397 token，除以 604,800 windows 约 **3.67 token/window**。
- 294-row 相对 2.5× 门槛只剩 197,402 token，除以 95,040 windows 约 **2.08 token/window**。

因此 43/294 适合作为数学门槛，不适合作为 RTL 性能承诺。任何 descriptor issue、fill/drain、bank stall 或 commit 开销都可能吃掉门槛。若后续做 RTL，64 rows 是更合理的 2× 起点；2.5× 必须先补真实 accumulator/queue schedule，再决定是否接受更大的窗口。

## 硬件存储风险

- 每窗口 descriptor key space 固定上限 128；从 43 rows 起 p95/p99 已经都是 128，不能按平均 descriptor 数缩小表。
- 43 rows 最大 2,672 events/window。只存 6-bit row offset + 1-bit direction 的理论下限约 2,338 bytes；p99 约 1,050 bytes。实际还需 group boundary、destination/commit metadata、valid 和 backpressure。
- 294 rows 最大 11,128 events/window。只存 9-bit row offset + 1-bit direction 的理论下限约 13,910 bytes；p99 约 8,020 bytes。
- 如果 loop order 改成 window-major 并把 19-bit accumulator 按 3 bytes/lane 留在片上，43/294 rows 的单 buffer 下限分别约 **99,072 B / 677,376 B**，双 buffer 约 198,144 B / 1,354,752 B，尚未计 event/descriptor SRAM。
- 如果保持 partition-major 顺序，窗口并不会限制跨 432 partitions 的 live destination state；全 3,000 rows 的 raw accumulator state 是约 **6,912,000 B/operator**。若改成 window-major 来减少它，则 weight/PWP residency 和 load 次数会变化，本账本不再自动成立。

## 打铁分级

### P0

无。当前产物严格限定为 conditional service-token envelope，输入身份、事件守恒和公式未发现致命错误。

### P1

1. 43/294 的门槛余量只有约 3.67/2.08 token per window，不能抵御任何真实控制开销。
2. accumulator bank、RMW port、有限位宽、commit/dependency 顺序全部 port cut；one-event-per-token 不等于 one-event-per-cycle。
3. partition-major 需要约 6.9 MB 全 raster live accumulator；window-major 会改变 weight/PWP load/residency 假设。
4. lossless queue 必须按最大值或实现 backpressure/spill；p95/p99 不能作为无损容量。294-row 最大事件队列已超过 11k entries。
5. 三拍 group load、event token、既有 PWP ledger 只是相加，没有可执行 shared-port/descriptor/fill-drain schedule。

### P2

1. M72 centers 来自 valid825 internal calibration，heldout 也仍属于同一 valid825 population，不是独立 validation 或 train-only catalog。
2. 窗口固定从 phase row 0 对齐；不同硬件起点、跨 phase 合并或动态窗口会改变 group totals。
3. group key 不含方向，必须为每个 event 保存 add/sub direction；否则不能安全共享权重。
4. event token 当前未收费 destination offset、group pointer、count、valid、ECC 或 SRAM read/write energy。
5. 2.668× 是 3,000-row 条件上限，不能写成模块 scheduled cycle，更不能写成 physical/system/headline。

## GO / NO-GO

- raw-mask natural-order reconstruction：**GO**。
- M72/M78 cap11 work conservation：**GO**。
- bounded-window conditional service-token envelope：**GO**。
- 以 64-row 为首个 RTL microarchitecture DSE 点：**GO_AFTER_QUEUE_AND_ACC_CONTRACT**。
- 43-row 2× / 294-row 2.5× scheduled-cycle claim：**NO-GO**。
- physical/equal-area/system/headline：**NO-GO**。

本评审只在当前目录新增文件；没有修改生产文件或 `docs/359`，没有使用或引用开源工具结果。
