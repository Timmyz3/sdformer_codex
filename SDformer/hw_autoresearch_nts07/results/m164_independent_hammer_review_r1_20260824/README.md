# M164 独立打铁评审 r1

结论：**82/100，`PASS_MODULE_LEVEL_MATCHED_LOGIC_ONLY_AB_REVISE_BEFORE_NETWORK_ADMISSION`，P0/P1/P2 = 2/6/3。**

M164 这次是实质进步，不是把局部 work reduction 换名成 speedup。26/32/18/19-bit 的 bounded-width 数学正确；最大 H67 population、per-hidden-lane moment、preclear/snapshot、指定 RNE 边界和 FIFO 背压均有可复核证据。与语义有效的 M163r2 使用同一 DC 流程，**模块级 logic-only A/B 可以引用**。但它仍然只从“已经 Q8 的 fc1 输出”开始，到 rank3-right + moment frontend 为止，不能引用为 FFN、网络或系统加速。

## 1. 独立位宽复算

冻结 stage0 每 hidden channel 的 population 是 `120×160×T10 = 192000`。对任意 signed INT8 输入：

| 状态 | 精确需求 | 最小位宽 | M164 |
|---|---:|---:|---:|
| sum | `[-128N,127N]=[-24576000,24384000]` | signed 26 | 26 |
| sumsq | `128²N=3145728000` | unsigned 32 | 32 |
| count | `0..192000` | unsigned 18 | 18 |
| 10 项 signed 8×8 projection | `[-162560,163840]` | signed 19 | 19 |
| 10 项 factor row sum | `[-1280,1270]` | signed 12 | 12 |

五项均足够且在当前 universal Q8 合同下最小。`-128×-128=16384` 与 `-128×127=-16256` 两个不对称端点已经计入。

## 2. VCS、最大 population 与 RNE

sealed exact-SHA VCS 的 compile/sim rc 都是 0，没有 assertion failure。关键证据：

- `19264` tiles、`96320` accepted beats；其中最大 channel 连续跑 `19200` tiles，达到每 lane `192000` samples。
- 最大点 cover 1 次，结果为 `sum=-24576000`、`sumsq=3145728000`、`count=192000`；TB 对 16 lanes 都逐 lane 比较。
- five-beat cover 命中 `19204` 次，说明最大 channel 不是一个孤立五拍样例，而是长串连续 tile。
- 显式 RNE：shift1 的 `1,3,-1,-3 -> 0,2,0,-2`，12 次检查；shift0 的 `254,-256 -> 127,-128`，6 次 saturation；shift23 3 次归零检查。
- `rank_stall_cycles=2701`、`moment_stall_cycles=9`，已接受的 rank/moment 在 stall 及后续 sticky fault 中保持并排空。

有一个必须修正的收据错误：`96320/5=19264` tiles，故 per-hidden-lane 总样本是 `19264×10=192640`，不是 contract/pass/receipt 中硬编码的 `192650`。这不影响每 channel scoreboard 或最大 population 的功能结论，但 **192650 禁止进入表格**。

当前 RNE 可以说“指定向量已覆盖”，不能说“完整边界穷尽”。仍缺 exact `+127/+128/-128/-129`、更宽 shift 的 half±1、projection 极值和非法 shift24 fail-close。

## 3. population guard 与 preclear ownership

overflow predicate 在接受下一拍前检查 `channel_count_q > 192000-2`。因为每拍固定增加 2，独立枚举所有可达偶数 count 后得到：`191998` 可安全接受到 `192000`，`192000` 的下一拍被拒绝，没有 bounded-state wrap。

但 sealed TB 没有真的攻击 overflow：最大 channel 在 192000 正常 last、snapshot、preclear 后，又开始了一个新 channel。下一版应让 max channel 保持 non-last，再送下一拍，检查 sticky fault、state 不变与已接受输出可排空。

preclear/snapshot 的局部 ownership 正确：final beat 用 `channel_*_next` 复制完整结果，随后同一 `always_ff` 中更晚的 NBA 把 active bank 清零；若旧 moment 不能 pop，final beat 被 `moment_capacity` 反压；若 pop 与 push 同拍，旧值先 handshake，新 snapshot 再占有寄存器。下一 channel 因而不需要 `tile_channel_start` 驱动宽 accumulator mux。

这仍不是全网 extent 证明。接口没有 stage/channel-base/spatial-address/expected-tiles，无法发现 early-last、漏 tile、重复/乱序 tile 或 lane 换绑。

## 4. II=5、FIFO 与算术资源

独立复算每 accepted beat 是 `3×16×2=96` signed products 和 `2×16=32` squares；每 tile 是 480 products、160 Q8 samples。DC resource report 保留 32 个唯一 square `DW_mult_tc` 与 48 个各含两乘法的 projection datapath，闭合 32/96 结构。

对 raw/quant/output controller 做了不依赖 TB 的 over-approximation：允许任意合法 raw push 和任意 rank ready，遍历 28 个可达状态、112 条转移，没有 raw/output count 越界。另以 ready=1 连续模拟 19200 tiles，II=5 下 input stall 为 0，raw/out 最大 occupancy 都是 1。

因此可以说“standalone、合法 stream、无阻塞时 accepted tile II=5”。不能说完整 BN/FFN/network II=5；外部 barrier、地址 replay、correction/left、ATLIF 和 fc2 都未实现。

## 5. matched DC A/B

M163r2 与 M164 使用同一 TSMC28 library、3.0 ns clock、同一 uncertainty/I/O delay、flattened `compile_ultra`、ZeroWireload、ideal clock、0 macro：

| 指标 | M163r2 | M164 | 变化 |
|---|---:|---:|---:|
| cell area | 53662.139958 µm² | 42376.823933 µm² | **-11285.316025，-21.030313%** |
| cells | 60910 | 45824 | **-15086，-24.767690%** |
| sequential | 9183 | 6723 | **-2460，-26.788631%** |
| logic levels | 88 | 34 | **-61.363636%** |
| critical path | 2.42 ns | 2.02 ns | **-0.40 ns，-16.528926%** |
| setup slack | +0.1053 ns | +0.7492 ns | **+0.6439 ns** |
| hold slack | +0.0001 ns | +0.0000 ns | MET，但无可迁移裕量 |
| ports | 2689 | 1939 | -750 |

`2460` sequential reduction 可从 RTL 独立、精确重建：moment active+snapshot 收窄省 1472，两个 count 省 28，projection accumulator + raw×2 + quant copy 收窄省 960，总计正好 2460。最坏 setup 从 M163r2 的 `tile_channel_start -> channel_sum_q[4][47]` 移到 M164 的 `beat_expected_q[0] -> projection_acc_q[1][15][18]`；reported setup paths 中已无 `tile_channel_start`。

允许引用的完整口径是：

> 在相同 3 ns TSMC28 ZeroWireload/ideal-clock/0-macro logic-only DC 流程下，bounded-width M164 相对语义有效的 M163r2 减少 26.79% sequential cells 和 21.03% cell area，并将 setup slack 增加 0.6439 ns。

不得改写成 placed/routed PPA、Fmax、energy、physical speedup 或 network speedup。M164 hold 四舍五入为 0.0000 ns，接口仍有 1939 ports，也没有 Formality 收据。

## 6. M165 优先项

最大硬件降本点已经从 moment 宽 carry 转移到 raw ownership：M164 的 raw FIFO 两份是 1824 FF，`quant_raw` 又复制 912 FF。

M165 应优先做 direct-bank requant：

1. 两个 raw bank 直接供 requant 读取，先删除 912-FF `quant_raw` copy，保留 stall elasticity。
2. 再做一个 raw bank 的 no-backpressure-II5 版本；若严格服务周期闭合，可相对当前 raw2+copy 共省 1824 FF，但必须量化 `rank_ready` stall 下多出的 input backpressure。
3. 同时补 post-max overflow attack、地址/extent wrapper、exact saturation edges、invalid shift24，以及 Formality。
4. 后续 coefficient engine 不应把 946-bit moment snapshot 的超宽端口当免费资源，应评估 lane-serial/banked 消费。

PAFT 通过前，synthetic factor、fc1→Q8、BN coefficient、correction/left、ATLIF、fc2、完整周期和网络 speedup 仍保持 false。

机器可读裁决见 `m164_independent_hammer_review.json`；所有数字由 `independent_recompute_m164.py` 从 sealed 原始日志、mapped netlist 和两代 matched DC 报告独立复算。
