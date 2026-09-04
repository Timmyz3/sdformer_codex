# M528 dead-write-only 单端口 1RW RTL 复用预规格评审

日期：2026-08-27  
模式：source-only、只读证据审计；未写 RTL，未运行 VCS/DC/Formality/PTPX/GPU  
裁决：**94/100，M528 同账本 CPU 结果及独立 hammer 通过后，允许唯一一个 dead-write-only 1RW RTL；combined PVRF 禁止。**

## 1. 唯一最小实现边界

未来只允许一个顶层：`m528_dead_write_only_1rw_product_capture_island`。它必须把下列对象放在同一证据边界内：

1. 64-row 动态 exact-subset matcher；
2. 两份 ping-pong `64x32b` row directory、两份 `64b` parent-live bitmap 和当前执行 bank 的 `64b` written bitmap；
3. 按 `(popcount(original), row_id)` 稳定排序的 scheduler，以及唯一一条 earliest-parent lookahead；
4. deadline-hold 单端口仲裁、1-cycle read response、queue+pending 总容量 2；
5. M474 的 signed12 parent/residual 重建与 signed19 psum 更新；
6. 九颗 generated `128x128b 1RW` 宏并行拼成 `1152b`，只用 row 0--63；
7. 不因 dead store 被省掉而改变的 psum commit、row completion 与计数守恒。

不包含 combined PVRF、第二 lookahead、concurrent-1R1W、decoder/full-network scheduler，也不允许失败后再开第二结构。

## 2. 哪些能复用，哪些不能直接搬

| 来源 | 可以复用 | 必须修改或禁止沿用 |
|---|---|---|
| M474 | signed12/signed19 算术、overflow、同步 read response mux、same-address new-value forward | 原模块允许独立 read+write，并把 `row_complete` 绑到 scratch write |
| M476/M476r2 | 两 reserved-entry FIFO 顺序、无 consume credit、pop→response→forward 入队顺序、stalled RAW guard | packed 双槽只是功能证据，不是 clean physical block |
| M498 | queue next-state、lane 分组思路、targeted/full regression 内容 | BUFFD tree 在 DC 中被优化丢失，fanout/cap/transition 未过，不能当物理实现复用 |
| M475 | TSMC28、3 ns、报告/约束方法和算术 logic reference | `37,316.285 µm²` 是 0-macro logic-only，缺 matcher/scheduler/queue/macro/psum |
| generated macro adapter | active-low `CEB/WEB` 与并行 slice 绑定方式 | 新 wrapper 必须九 slice；`A[6]` 必须为 0 |

M498 的正确定位是“可复用语义，不可复用已失败的物理结论”。M474 的 one-slot 不能替代两 reserved-entry 模型，因为 `pending response + same-cycle forward` 可产生双入队。

## 3. 单端口与九宏拼接

九颗 `TS1N28HPCPHVTB128X128M4S` 共享 `CLK/CEB/WEB/A`，每颗承载连续 128-bit slice：`slice i -> data[i*128 +: 128]`。wrapper 只允许三态：

- idle：`CEB=1`；
- read：wrapper `enable=1, write_enable=0`，宏脚 `CEB=0, WEB=1`；
- write：wrapper `enable=1, write_enable=1`，宏脚 `CEB=0, WEB=0`。

硬断言是 `!(scratch_read_enable && scratch_write_enable)`。一次 read 在 N 边沿接受，N+1 周期带原 request ID 返回。live final 与同地址 request 同拍时，只写宏、不读宏，直接 forward 新结果。九宏物理容量为 `18,432 B`、生成宏裸面积小计 `78,825.2454 µm²`；这还不是 integrated PPA。

## 4. dead/live 元数据不是免费 oracle

M505 的 `cleanroom_subset` 对每个 64-row task 完成全部 parent 选择后，才能知道哪个 producer 被至少一个 child 引用。dead-only 不需要 `0/1/2+`，只需要 parent-live bit：matcher 每接受一个 parent descriptor，就置 `live[parent_id]=1`。

执行与下一 task preprocess 在 CPU 模型中重叠，因此 live bitmap 和 row directory 必须双缓冲。目录每 row 32 bit 足够放：`residual16 + parent_id6 + parent_valid1 + original_popcount5 + reserved4`。execution 只有在 matcher done 和 bank ownership handoff 后才能看 live bitmap；bank 未 release 前不能覆盖。

每个执行 bank 还要一份 written bitmap。只有 accepted live write 才置位；普通 read 必须满足 `written[parent]`。forward 是唯一允许在写入同拍提供新 parent 的旁路。

最容易漏掉的语义是 completion：

- dead final：只省 parent scratch write；算术 final beat、psum write、`row_complete` 全部发生，还可用空闲 scratch port 发一个合法 read；
- live final：一定写 scratch，即便这个 single-use row 已 same-cycle forward。省掉该写就是 combined PVRF，越界。

因此旧断言 `row_complete <-> scratch_write_enable` 必须拆掉，改成 scratch write 由 live 控制、architectural completion 由 accepted final beat 控制。

## 5. 尚缺的硬件

当前没有可直接实例化的动态 64-candidate subset matcher。M348/M363 是静态 Hamming matcher，只能借鉴 compare/reduction 的编码纪律，不能冒充同一算法。新 matcher 必须实现：candidate 是 current 的 subset、排除 equal-later、最大 candidate popcount、tie 取最低 eligible row ID、residual=`current XOR parent`。

另外尚缺：

- 17-popcount 稳定顺序扫描与 earliest-parent scheduler；
- ping-pong directory/live ownership 和 written bitmap；
- 九宏 1152-bit vendor wrapper；
- 能 clean DC 的两条 1152-bit response slots。M477/M498 只证明其功能需要，未证明其物理可行；
- matched strongest-zero/bit 总面积 wrapper 与 resident-psum 明确端口；
- 完整 SAIF/PTPX、宏读写/leakage、DRAM 能量账。

## 6. VCS/SVA 必须覆盖

断言至少包括：

- read XOR write、九宏 control/address 一致、slice 拼接无误；
- dead final 无 scratch write 但恰有一次 psum commit/row completion；
- live final 恰有一次 scratch write，single-use forward 也不能省；
- forward 必须 live、final、同地址、新数据，并抑制 macro read；
- one-cycle response、ID 对齐、禁止 read-before-written；
- queue+pending `<=2`、full 不借 consume credit、FIFO parent 顺序不变；
- stalled same-address final 在 psum backpressure 下不读旧值；
- directory/live bank 在 matcher done 前不可见、release 前不可覆盖；
- issue、edge、dead elision、read/write、hold/stall、commit/completion 全部与 M528 reference counter 一致；
- sticky protocol fault、overflow atomic block、stall payload stable。

cover 至少打到：dead-final+并发 read、live-final deadline hold→read→write、same-address forward、pending response+forward 双入队、queue-full consume、连续 dead 与 dead/live 交替、exact/partial parent、多 beat、psum stall stale-RAW、ping-pong 重叠、row 0/63、九个 slice 非零，以及 wrong-ID/read-before-write/stale-epoch/overflow 攻击。

## 7. DC/Formality/PTPX 硬门

只有 M528 CPU 同账本重算和独立 result hammer 先通过，才可写这一个 RTL。

- VCS：directed+random+冻结 trace recurrence 全部 0 mismatch；总周期距离 `435,293,339` 不超过 1%，并相对 `760,350,133`、`757,946,784` 均保留 `>=1.50x`。
- DC/STA：exactly 9 个 generated SRAM macro；不得把 parent scratch 合成寄存器；总容量 `<=245,760 B`；3.000 ns setup/hold 非负；max-cap/max-transition/max-fanout 全 clean；0 unconstrained/latch/multiple-driver；宏与保守互连延迟计入。
- 面积效率：matcher、scheduler、directory、live/written、response queue、九宏、adapter、matched psum boundary 全收费后，local throughput/total-area 相对 matched strongest zero/bit `>=1.10x`。
- Formality：RTL↔mapped netlist 在九宏 black-box/cutpoint 下通过；另做 invariant miter，在 live bitmap 正确前提下证明 dead-write-only 与 all-write 的 architectural psum/row outputs 等价。
- PTPX：SAIF logic + generated macro read/write/leakage + DRAM 总账，四 Conv energy `>=20%` 降低或 EDP `>=1.25x`。必须显式保留 `9,069,207,552 B` weight DRAM 和 `30,457,108,224 B` 八 block parent logical movement。

任何 identity/cycle/capacity/DRC/timing/Formality/area-efficiency/energy 门失败，永久关闭该唯一变体。

## 8. 模型—RTL 漂移 P0

1. 沿用 `row_complete=scratch_write` 会丢 dead row completion。
2. 没有在 wrapper 强制 read XOR write，会偷回 M473 concurrent-1R1W ceiling。
3. live 只做一份、边 match 边 execute、或不计容量，会破坏 task overlap。
4. 省 single-use forwarded store 会偷换成 combined PVRF。
5. tie/order/equal-later/lookahead/no-consume-credit 任一改变，435M 口径失效。
6. one-slot 无法覆盖 dual enqueue；直接搬 packed dual-slot 又会重复 M477/M498 的物理失败。
7. 把 128-depth 宏按 64-depth收费会少算 `9,216 B`。
8. 把 macro Q 当组合读或两周期读会改变所有 stall。
9. psum stall 下 same-address prefetch 未封，会重现 stale-RAW。
10. 用 M475 0-macro area 对 CPU 倍速，或隐藏 30.457 GB parent movement，都是无效 DATE 对比。

## 9. 当前 claim boundary

本包只是 source-only prespec。它没有生成 RTL、VCS、DC、Formality、PTPX、macro-inclusive PPA、full-network 或 system speedup 证据，也未修改 `docs/359`。
