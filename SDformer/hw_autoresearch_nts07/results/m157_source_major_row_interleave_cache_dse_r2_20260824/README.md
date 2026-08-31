# M157 source-major cache + row interleave DSE r2

## 结论

M155 通用四 bank RMW accumulator 的 debug DC 暴露了新热点：76,994.064 µm²、14,519 个 sequential cell、50 级逻辑，3 ns setup 余量只有 +0.0164 ns。主要成本包含 7,296-bit same-address forwarding payload、3,072-bit valid bitmap 和 384 lane 的 runtime overflow 归约。该 DC 是 debug anchor，未 sealed，不是 paper PPA。

M157 不再继续扩张这个通用数据路径，而是利用 Conv 的真实可交换性重排顺序：

`window -> partition -> source -> destination half -> active row`

- 每个 phase 只保留四个目的 bank 的 96-lane INT8 vector，单 cache 3,072 bit，ping-pong 为 6,144 bit。
- 先处理 destination 0..3，再处理 4..7；phase 内跨 active row 交织。
- 同一 phase 的相邻 descriptor 总是不同 row，两个 phase 的 `destination[2]` 又不同，因此可以从调度上避免连续同地址 RMW。

## 冻结 heldout 结果

20 条 Motion/H67 ep35 heldout：

- M152 descriptor 守恒：47,040,777 / 47,040,777。
- source event 守恒：188,148,490 / 188,148,490。
- active source key 守恒：23,522,595 / 23,522,595。
- 自然 phase-row 顺序的相邻同 bank/同地址 hazard：**0**。
- 优化后需要的 RMW bubble：**0**。
- 逐事件目的向量读取为 188,148,490 组，source-major cache 只需 8,271,296 组，组读取数减少 **22.7472×**。这是访存工作量，不是周期或系统加速。

共有 2,067,824 个 cache phase。若 ping-pong 只对每个非空 partition-window 收一拍启动费，对 M143 的局部周期 sensitivity 为 1.80370×；若最悲观地每个 phase 都不 overlap，为 1.75694×。两者都没有被 hardware admission。

## 下一硬件门

1. 用真实 signed INT8/Acc19 payload 做 source-major 与原顺序的 integer miter，确认重排在冻结数值合同下 bit-exact。
2. 实现 fused cache-to-accumulator RTL，显式带 sequence/operator/window/partition 身份，并在 VCS 中证明 no-forward II=1、tail、backpressure 和 fault drain。
3. 用新思 DC 验证删除 7,296-bit forwarding 与 runtime overflow OR 后的面积/时序；Acc19 动态检测只能在真实 trace 界已证明时删除。
4. 最后加目标 SRAM macro、cache load-to-use、accumulator commit 和 SAIF/PTPX，再决定 1.80× 是否可接纳。

当前等级：`PASS_HELDOUT_CACHE_AND_RMW_HAZARD_DSE_ONLY`。`physical_speedup=false`，`system_speedup=false`，`headline=false`。
