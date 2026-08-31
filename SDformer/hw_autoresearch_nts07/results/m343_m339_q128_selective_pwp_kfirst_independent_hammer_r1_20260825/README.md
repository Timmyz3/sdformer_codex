# M343：M339 q128 selective-PWP K-first 独立打铁评审

结论：**76/100，P0=0、P1=4、P2=3**。

M339 的 exact-work、runtime working-set、PWP byte、M251 q16 复现和固定递推算术都成立。未发现 sample/operator/partition 循环错位；M248 的 40 个 sample/operator 键恰好各出现一次，80 个 payload 共 40,685,343 bytes 重新哈希均通过。

## 可以保留的数字

- q16/q32/q64/q128 exact vector-work speedup：1.540642x / 1.692877x / 1.857852x / 2.043940x。
- q128 working set：mean 106.97、p90 125、p99 127、max 128。
- q128 selective PWP traffic 为 2,129,387,904 bytes，仅比 full table 少 1.1966x。
- M251 q16 legacy cycle 两种端口均 exact reproduction。
- match→selective-PWP DMA 已改成串依赖；common commit 为每样本 96,000 cycles、十样本 960,000 cycles，并同时计入两条线。

## 不能提升为 cycle headline 的原因

2.003053x（WIDE144 + q128 systolic）和 1.429893x（SHARED96 + q128 systolic）只是固定递推的乐观估算，不是严格的 cycle upper bound。full-phase match 后仍需保存或重读 raw/correction/chosen-pattern 描述符；有限 queue、容量不足时 chunk/spill、SRAM bank conflict 和端口仲裁均未进入模型。

PWP-only cache fit 的布尔矩阵本身算对了。补入每 phase 12,288-byte weight 和 2q-byte pattern code 后阈值暂未改变，但 headroom 明显缩小；再加入与 K-first 因果相匹配的最低 raw16 与 chosen-ID 留存后，q16 的 64-KiB 双 context 从 fit 变为不 fit。若这些缓冲使用独立 SRAM，也必须进入面积和流量账本。

面积公平性仍是硬伤：q128 systolic 相当于 128 个 distance PE，是 SERIAL16 的 8 倍；WIDE144 又比 SHARED96 宽 1.5 倍。当前没有 Fmax、面积或把相同额外预算交给 bit-sparse baseline 的对照。

## 下一步最小实现

先实现 **SERIAL16 + SHARED96** 的有限双 context 模块模拟器，扫 q16/32/64/128 与 64/128/256/512 KiB。接口至少包含 phase header、raw16 row stream、pattern-code read、exact match descriptor、带 kind/address/bytes/context 的 DMA、共享 compute port、common commit 和最终 done；必须断言 match_done 后才能发 data-dependent PWP DMA，容量不足只能 stall/spill，不能静默假设 fit。

在该模块得到 zero-mismatch、有限资源 cycle，并完成面积/Fmax 对齐前：**GO exact work/traffic，NOGO executable cycle、严格 upper bound、系统倍速和 DATE headline。**
