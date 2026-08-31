# M347：M344 fixed-64-KiB output-block tiling 独立打铁

结论：**71/100，P0=0、P1=5、P2=3**。

M344 最有价值的部分成立：用 q16/O8、q32/O4、q64/O2、q128/O1 保持 `q×O=128`，把最坏 PWP tile 固定为 18,432 bytes。加入当前 tile 的 weight 和一份 pattern table 后，单 context 依次为 30,752 / 24,640 / 21,632 / 20,224 bytes，双 context 为 61,504 / 49,280 / 43,264 / 40,448 bytes，全部能放进两个独立 32-KiB cache context。

36,000-byte descriptor SRAM 的容量也算对了：`2×3000×6 bytes`。48-bit 字段总宽准确，row12、original16、center-index7、distance5 足以覆盖 3000 rows、raw16、q128 center 和 0..16 Hamming distance。但 64 KiB 只是 PWP/weight/pattern tile cache；加上单独 descriptor 后，固定物理分配是 101,536 bytes。descriptor 的全行写入、八个 output tile 的读取带宽、bank conflict、context tag 和 fallback/zero-row 语义尚未进入周期模型。

## 一个确定的周期账本问题

pattern 已在 `matcher_packer` 中、第一次 match 前加载一次；`candidate_tile_bytes` 又把 `q×2` pattern bytes 放入每一个 output tile load。pattern 与 output block 无关，cache capacity 也只保存一份，因此后者是重复收费。

q128/O1 在 17,280 个 phase 中多计 35,389,440 bytes、即最多 1,105,920 个原始 32-byte DMA service cycles。strict 中每个 phase 至少有一个 first-tile duplicate 暴露，故只修这个问题时，275,922,812 cycles 应落在 274,816,892..275,784,572，speedup 为 1.971771x..1.978714x。overlap 的对应算术敏感区间是 265,611,599..266,717,439 cycles、2.038802x..2.047291x。这里只是去掉重复收费的区间，不是新的可执行性能数字。

## 为什么两个数字都不是 hardware bound

strict 把 next-partition first-tile load 串在当前 body 之后，但它仍让 next matcher/packer（其中含 pattern DMA）与当前 body 重叠。q128/O1 当前 tile compute 与 next-current-tile prefetch 已可能占用两个 cache context，因此 next phase 的 pattern 或 first tile 没有保证可用的 slot。

overlap 更进一步让 next matcher、packer 和 first-tile load 全部与 current body 重叠；在只有两个 tile context 时，这通常需要第三个 slot，或至少需要精确到 final-tile release 的状态机。因此 1.970784x 只是带保守项和遗漏项并存的 analytical estimate，2.038802x 是更乐观的 recurrence estimate，均不能称 executable/strict hardware bound。

`q×O=128` 固定的是最坏 PWP tile bytes，不是面积。q128 systolic 仍隐含 128 个 distance PE，是 SERIAL16 的 8 倍；WIDE144 的 PWP payload 又是 SHARED96 的 1.5 倍。当前最可信的实现起点是 **q128/O1 + SHARED96 + SERIAL16**，其 strict 递推为 389,416,990 cycles、1.396406x，但在 finite context 实现前仍不能晋级。

## 下一最小实现

先做 cycle-exact finite-context module simulator：两个 32-KiB tile slot、两个 `3000×48-bit` descriptor bank、一个 32-byte/cycle DMA、一个 SHARED96 compute port和真正的 SERIAL16 matcher。每个 cache slot 必须有 EMPTY/LOADING/READY/COMPUTING/RELEASED 状态和 phase/tile generation tag；pattern 每 phase 只能加载一次。容量或 queue 不足只能 stall，不能静默重叠。

并行准备 q128 matcher RTL/DC DSE：PE16/32/64/128 对应 8/4/2/1 passes，VCS 对齐 frozen Python nearest-center/tie/PWP rule，再用相同 TSMC28 3.0-ns pre-macro 口径比较 area、Fmax、rows/s、throughput/mm² 和 center-SRAM ports。只有 finite simulator zero mismatch、matcher RTL cycle match、同资源 baseline 和面积/Fmax 收口后，才能把 1.396406x 替换成可执行模块性能。

当前口径：**GO output-block tiling 与容量证明；NOGO M344 两个 cycle bound、面积归一、系统倍速和 DATE headline。**
