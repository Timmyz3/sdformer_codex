# M168 H67 FC2 K-bank multi-source DSE 独立打铁评审 r1

结论：**85/100，`PASS_EXACT_DSE_PRIORITIZE_K4_RTL_NO_SPEEDUP_ADMISSION`，P0/P1/P2 = 4/5/3。**

M168 的窄口径结论成立。独立脚本未调用 production analyzer，而是从冻结 M51
归档中单独解出 120 个 FC2 bitpack，对全部 **437,760,000 bytes** 走两条解码路径，
重新核了 SHA、大小、popcount、bank 占用、逐 token 周期和 `Cout/96` 加权。结果精确复现：

| 项 | 独立复算 |
|---|---:|
| records / modules / samples | 120 / 12 / 10 |
| tokens / input bits | 5,580,000 / 3,502,080,000 |
| events | **143,894,510** |
| K1 output-block cycles | **412,900,394** |
| K4 output-block cycles | **106,536,803** |
| K8 output-block cycles | **70,657,362** |
| K1/K4 bank-service boundary | **3.875660x** |
| K1/K8 bank-service boundary | **5.843700x** |

这两个倍率仍只是已经形成八条独立 bank queue 之后的 source-service boundary，不能叫
RTL、物理、FC2、FFN、网络或系统加速。现在最值钱的下一步是 K4 RTL，但必须把
compactor、3072-bit/cycle weight response、四项 reduction、Acc24 和输出 commit 一起计时。

## 1. 身份与 packing

独立复核的冻结身份：

- analyzer `54b8f09b78e49a6ee1ea27e4133bdec1182ae1f73ada55ead5348d5e995390dd`；
- contract `93abf9e5ba4d11bd35821e01a62719600847b96c1362fde95bb0a9c1e26d3a3d`；
- published result `d203ca6bb5a59e23c8b39cd8dff116d2134efb2280ba7889781021df1f96b137`；
- M51 manifest `2a5e6e472b3897ea508f61c3727bddd97818c921b0729c0f2731b150c6d7a76e`；
- M51 archive `aa261ebe64015bbd295f65f4b734efcb6b26c11c3dd0828e9e7a659433f6c3b4`；
- `docs/359` `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`，未改。

M51 是 C-order flatten，channel 是最后且最快维；四档 FC2 `Cin` 全部是 8 的倍数。
因此每 byte 内 little-order 的 bit `b` 恰好属于 `channel mod 8 = b`，不会跨 token。
独立脚本一边用 `unpackbits(bitorder=little)`，另一边直接 shift/mask bit 0..7，全部
437,760,000 bytes 的 bank count 一致，120 个 record popcount 也全部匹配。

## 2. 最小周期公式充分且必要

令一个 token 的八个 bank event count 为 `n_b`，每 bank 每拍最多一个事件、全局最多 K 个：

```text
C = max(max_b(n_b), ceil(sum_b(n_b) / K))
```

必要性来自两个独立下界：最忙 bank 至少需要 `max_b(n_b)` 拍，总容量至少需要
`ceil(sum/K)` 拍。充分性也成立：建 C 个空 cycle，逐 bank 把它的 `n_b` 个事件放入
当前 load 最低的 `n_b` 个不同 cycle。每个 bank 在一拍最多出现一次；load 始终相差不超过
1，而总事件不超过 `C*K`，所以任何 cycle 都不会超过 K，恰好达到下界。

独立构造器穷举了每个 bank count 为 0..2 的 6,561 个向量，并加入冻结 payload 的确定性
actual witnesses；共 8,677 个不同向量、跨 K1/K2/K4/K8 共 34,708 次构造，0 反例。
但这个证明从“独立 bank queues 已经存在”开始，并没有证明 bitmap discovery 或 SRAM 能
每拍跟上。

## 3. `Cout/96` 加权与人口隔离

K1 的分 stage 整数复算是：

| stage | events | output blocks | weighted cycles |
|---:|---:|---:|---:|
| 0 | 46,809,056 | 1 | 46,809,056 |
| 1 | 33,053,865 | 2 | 66,107,730 |
| 2 | 53,067,276 | 4 | 212,269,104 |
| 3 | 10,964,313 | 8 | 87,714,504 |
| 合计 | 143,894,510 | — | **412,900,394** |

因此 production 的 output-block weighting 正确。未来 RTL 应把一个 compacted source group
保持住，依次 replay 到全部 `Cout/96` output blocks 后再退休；否则必须存整份 event list
或为每个 output block 重扫 bitmap，当前加权式就没有实现载体。

复算只读取 SHA-pinned M51 ten-sample manifest 与其中 120 个 FC2 payload。没有读取 M39
profile100，也没有借用它的 denominator。任何 profile100 或 dataset-wide 用语都仍被禁止。

## 4. 最大的隐藏成本：compactor

M168 最重要的发现不是再抠 bank 公式，而是不能把 event discovery 当免费。仅做一个
不重叠、固定宽 bitmap scanner 的敏感性账本：

| bitmap scan width | once/token scan cycles | matched serialized K1/K4 sensitivity |
|---:|---:|---:|
| 32 bit | 109,440,000 | 2.418502x |
| 64 bit | 54,720,000 | 2.899849x |
| 128 bit | 27,360,000 | 3.288058x |
| 256 bit | 15,600,000 | 3.508364x |

这些是暴露风险的 serialized sensitivity，不是实现性能。64-bit scanner 如果为每个 output
block 重扫，scan 变成 138,240,000 cycles，对应敏感性只剩 2.251604x。

应采用的硬件 trick 是：

1. little-bit byte 天然同时提供八个 bank 的同一 `channel>>3` 位置；维护八个 bank cursor；
2. 每次选最多四个不同 bank 的 next-set-bit，形成一个 source group；
3. group 内只保存最多四个 source index，保持它并跨全部 output blocks replay；
4. replay 的同时计算下一 group，用至少一项 group buffer 隐藏 compaction；
5. stage 3 有 8 拍 replay 窗口最好隐藏，stage 0 只有 1 拍，是 compactor 吞吐的最严角落。

只有 VCS 实际达到无泡 recurrence 后，3.875660x 才能开始向 standalone kernel measured
cycle ratio 演进。

## 5. K4 还是 K8

K4 达到理想 4-way 的 **96.8915%**；K8 只达到理想 8-way 的 **73.0462%**。从 K4 到
K8，weight payload 从 3072 bit/cycle 翻倍到 6144 bit/cycle，source-service cycles 只再减少
1.507795x，而且每 lane reduction 更深。因此把 K4 作为第一个 RTL 点是合理的。

这仍只是 provisional knee。没有 K1/K4/K8 的 matched DC、macro、时序和能耗，不能声称
K4 是物理 Pareto 最优。K4 RTL 后应低成本补 K2/K3/K4 parameterized DSE；K8 保留作带宽
上界即可。

## 6. M168 RTL 最小合同

建议只做一个独立 FC2 kernel，不必先做复杂全系统调度，但接口必须让所有隐藏成本可计时：

1. **输入与身份**：支持冻结四档 `(Cin,Cout)`；输入必须是 M51 语义的 exact `{0,1}`；
   bitmap beat、stage/module、token 和 output-block extent 全部带 tag，错序或越界 sticky fault。
2. **compactor**：`bank=c[2:0]`、`row=c>>3`；每 group 最多四个不同 bank，不丢、不重、
   不造 event；group 跨全部 output blocks replay 后才退休，next-group buffer 的所有 bubble
   进入 measured cycle。
3. **weight response**：八个逻辑 single-read bank；K4 一拍最多选四 bank，每 bank 返回
   96 个 signed INT8，即合计 **3072 bit/cycle**。request/response latency、ready/valid、
   output-block address 和 tag 必须显式，不能假设 combinational/free SRAM。
4. **算术**：96 lanes 各自把最多四个 signed INT8 精确 sign-extend/reduce，再更新 signed
   Acc24；冻结 overflow、wrap/saturation 和加法顺序。K1 与 K4 在任意 stall 下逐 lane exact
   miter。
5. **零事件和输出**：zero-event token 不能直接当 0 cycle；仍需产生正确 accumulator/BN2
   offset/residual response。BN2+residual 若不进模块，必须作为显式边界并禁止 FC2/FFN ratio。
6. **验证**：VCS/SVA 覆盖 zero、single、four-bank、>4 bank、same-bank worst case、full bitmap、
   四 stage、四种 output-block 数、随机 response/output stalls 和 protocol attacks；然后 matched
   K1/K4 Synopsys DC。
7. **性能收据**：从 token accept 计到 final output retire，单独报告 bitmap cycles、compactor
   bubbles、bank conflict、weight requests、lane utilization、zero-token fixed cost 和 commit cycles。

## 7. P0/P1/P2 与推进顺序

P0：

1. 独立 bank queue/compactor 被当成免费；
2. 3072-bit/cycle weight delivery、地址和 latency 未实现；
3. 四项 signed reduction、Acc24 与 K1/K4 exact-output miter 未实现；
4. zero-token、BN2、residual 与完整 FC2/FFN cycle 未组成。

P1：`Cout/96` 尚无 RTL replay 状态机；K4 尚非 measured Pareto；没有 VCS/DC；任意
backpressure 可能打破 recurrence；人口仍只有 ten samples。P2 是 bank-pressure histogram、
非 power-of-two K 点和 production scheduler witness。

建议顺序：**K4 compactor + group-held output-block replay → 8-bank weight response wrapper →
96-lane four-term Acc24 + K1 exact miter → VCS cycle receipt → matched K1/K4 DC → 接 M160
BN2/residual cycle composition。**

机器可读裁决在 `m168_dse_independent_hammer_review_r1.json`；独立复算脚本与结果分别是
`independent_recompute_m168.py` 和 `independent_recompute_result.json`。本评审只新增本目录，
未修改 analyzer、contract、RTL 或 `docs/359`。
