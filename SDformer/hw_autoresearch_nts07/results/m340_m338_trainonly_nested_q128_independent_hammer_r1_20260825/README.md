# M340：M338 train-only nested q128 独立打铁评审

结论：**87/100，P0/P1/P2 = 0/3/3。M338 作为 train-only catalog 与精确 vector-work 证据 GO；进入 M339 runtime/cycle、energy、system 或 DATE headline 全部 NO-GO，直到三个 P1 闭合。**

这次没有导入 M338 builder 或 M43 unpacker，而是独立重写 3x3 support 展开，逐个核验 128 个 packed payload 和 128 个压缩 value payload 的文件 SHA、解压内容 SHA，并比对 value support 与 packed positive plane。随后从 165,888,000 个 partition vector 重建全部 1,728 个 partition histogram、catalog 排序和 q16/q32/q64/q128 工作账本。

独立复算结果为：

| q | candidate vector ops/block | vs bit-sparse train calibration | 全 PWP 容量 |
|---:|---:|---:|---:|
| 16 | 193,626,934 | 1.541232332x | 31,850,496 B |
| 32 | 176,101,460 | 1.694614519x | 63,700,992 B |
| 64 | 160,138,872 | 1.863533115x | 127,401,984 B |
| 128 | 144,983,745 | 2.058327925x | 254,803,968 B |

所有 1,728 个 q16 prefix 与 M77 逐 partition、逐位、逐顺序一致；q32/q64/q128 均为严格 prefix；排序、逐 partition observation、聚合账本、work conservation mismatch 全是 0。q128 的 `2.0583x` 相对的是 bit-sparse vector-work；相对 q16 的 candidate-work 改善只有 `1.335507881x`。它不是 cycle、runtime 或 system speedup。

## 关键发现

1. **254.8 MB 算术正确，但口径必须写清。** `4×432×128×8×144 = 254,803,968 B`，即 `254.803968 MB`（十进制）或正好 `243.0 MiB`（二进制）；pattern table 另有 `442,368 B`。这是全 provision 容量，不是 working set、SRAM fit、流量或驻留量。
2. **没有 valid 泄漏。** 32 个实际消费 key 与本机 exact-SHA 的 825-key valid list 交集为 0；128 个 record 恰好是 32 sample × 4 operator。限制是本机缺完整 7,345-key train list，因此 manifest 声明的 full-train overlap=0 无法在本机从头重建，但不影响已消费 S32 的独立零泄漏结论。
3. **catalog 排序 tie 已完全确定。** 规则是依次降序 `count×distance`、distance、count，再按 packed uint16 升序；独立复现 mismatch=0。可是 runtime 最近 center 的等距 tie 没有冻结 center/index 选择，这不会改变 work 或精确性，却会改变 PWP 地址、prefetch、bank conflict 与周期，属于 M339 P1。
4. **one-hot 处理正确但有少量继承浪费。** M77 q16 prefix 中有 20 个 one-hot center，分布于 19 个 partition；为保持逐位身份而保留是正确的。新增 112 项没有 zero/one-hot。严格 `1+distance < population` 下 one-hot PWP 永远不会胜过 fallback，M339 应明确禁止为它发 PWP service/fetch。
5. **q>16 是确定性的 nested heuristic，不是 q128 Lloyd。** 它只对 frozen q16 的距离做一次 `count×distance` 排名，不在追加 center 后更新距离。优点是 prefix 公平、可复现、向后兼容；算法 novelty 只能评为低到中，不可替代同 q/equal-byte 的优化基线。

## M339 最小闭合项

- 冻结一个与 S32 calibration 严格不相交、exact-SHA 的 runtime cohort；catalog 在 replay 前固定，禁止重调。
- 冻结最近中心 tie：建议与现有 matcher 一致使用最小 `(Hamming distance, packed uint16)`，同时输出 center index；value/index/distance 端到端 miter。
- 同一 row stream 重放 bit-sparse 与 q16/q32/q64/q128，显式计入 matcher latency/II 与 q pass、PWP 地址流、驻留/cache/DMA、有限 bank/FIFO、backpressure、correction、accumulator 和 commit。
- 给 candidate 和 baseline 同资源、同带宽、同过滤口径；分别报告 vector-work、module cycles、energy/PPA、system，不能把 `2.0583x` 继承到更强口径。
- 将完整 train/valid key list 封入交接包，补齐 full split 的独立复验；增加 separately optimized same-q 和 equal-byte baseline。

完整机器可读证据见 `m340_independent_recompute_r1.json` 与 `m340_independent_hammer_review_r1.json`。本评审未运行 GPU、RTL 或新思，未修改 M338 文件、合同或 `docs/359`。
