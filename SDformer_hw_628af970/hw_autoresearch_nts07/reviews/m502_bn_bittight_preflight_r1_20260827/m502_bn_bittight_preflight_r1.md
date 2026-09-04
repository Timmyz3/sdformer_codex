# M502｜H67 dynamic-BN bit-tight raw replay 只读预评审

## 裁定

**GO_OFFLINE_AUDIT_ONLY；暂不开发 RTL。**

M502 有一个值得继续核实的流量机会：在 M480 公平 strong fused baseline（raw 写一次、
current-batch barrier、raw 读一次、边归一化边消费）上，把每个 FFN phase 的 Q24 raw
container 换成由 M161 checkpoint `sumabs` 推得的 14--18 bit signed container。按同一
1R1W、同一 barrier、同一 coefficient/replay overlap 和 64 B/cycle 抽象重算，理想局部
schedule opportunity 是 **1.592129x**，useful raw traffic opportunity 是
**1.592179x**。

但是，这两个数现在都不是性能：M161 独立评审已经把 FFN INT8 bridge、Q24 sufficient
width、fixed binary point 和 overflow proof 列为 P0；M480 也没有物理 raw store、
bus-wide affine/consumer、VCS/DC/能量准入。更关键的是，候选需要每拍最多解出 37 个
值；若后端只有 21 lane，512-bit 下的 cycle opportunity 直接退化为 **1.000x**。

因此 M502 当前只准许生成 exact integer/raw trace 与地址化 schedule，不准许把机会数
写成 module/system speedup，也不准许先做一个运行时通吃 128/512-bit、14--18-bit 的
大 barrel packer。

## 证据与粒度

本评审不读取任何 M502 producer；脚本从两个已经独立打铁的上游重新建模：

- M161 independent recompute：12 个 H67 FFN，FC1 signed width 分布为
  `{14:2, 15:4, 16:6}`，FC2 为 `{15:2, 16:2, 17:6, 18:2}`；
- M480 independent receipt：公平基线是 fused raw replay，而不是显式 normalized
  materialization；24 个 BN phase、BN1/BN2 分别有 350,208,000 / 87,552,000 个元素；
- 所有输入都按 SHA256 固定，`docs/359` 仍为
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

可复跑脚本：`audit_m502_bn_bittight_preflight.py`。完整 64 点 DSE 和 admission flags 在
`m502_bn_bittight_preflight_r1.json`。

## 独立机会重算

### 512-bit 主参考点

| 项目 | M480 Q24 strong fused | M502 analytic-tight（未计价） | 机会 |
|---|---:|---:|---:|
| raw write+read useful traffic | 2,626,560,000 B | 1,649,664,000 B | 1.592179x |
| fused schedule | 41,048,856 cyc | 25,782,360 cyc | 1.592130x |
| peak single-phase raw retention | 210.9375 MiB | 123.046875 MiB | 1.714286x |
| 加 M159 fixed 205,384,111 cyc | 246,432,967 cyc | 231,166,471 cyc | 1.066041x |

候选每帧少搬 **976,896,000 useful bytes**。这里的 useful traffic 没把 tail/padding、
descriptor 和 SRAM burst 计入；cycle 也假设 compact stream 可以把总线带宽完全转化为
consumer service。

把 15,266,496 个理想 BN cycle 零重叠替换进 M292 的 620,302,905-cycle compute
envelope，得到的仅是 **1.025232x sensitivity**。M480 BN schedule 与 M292 envelope
尚未合成一条 integrated schedule，所以该值不得称为系统加速。

### lane 数是硬约束

以下各点都让 Q24 与候选使用相同 512-bit bus、相同 lane cap：

| downstream lane cap | analytic-tight local opportunity | 解释 |
|---:|---:|---|
| 16 | 1.000000x | producer/consumer 完全支配，少搬 bit 不少 cycle |
| 21 | 1.000000x | 约等于 Q24 每拍服务能力，packer 只能省流量 |
| 24 | 1.125236x | 尚不足以支撑主机制 |
| 28 | 1.312198x | 有机会，但需同时付 28-lane affine/consumer |
| 32 | **1.492259x** | 可作为后续同资源 RTL 的实际候选 |
| 36 | 1.584233x | 接近理想，但更宽 consumer 物理税未计 |
| 64 / unlimited | 1.592130x | 纯带宽上界 |

因此，M502 的正确设计对象不是孤立 pack/unpack，而是
`raw store + pack/unpack + N-lane replay/affine consumer`。只综合 packer 会重演
M480 “未计价 bus-wide consumer” 的 P0。

### 先收窄 width mode，避免过度设计

同一 512-bit、32-lane 条件下：

| width policy | useful traffic opportunity | local schedule opportunity | M159-serial opportunity |
|---|---:|---:|---:|
| 精确 14/15/16/17/18 | 1.592179x | 1.492259x | 1.058143x |
| 偶数 14/16/18 | 1.551020x | 1.486477x | 1.057657x |
| nibble 16/20 | 1.472868x | 1.472997x | 1.056511x |
| **byte 16/24** | **1.446701x** | **1.446835x** | **1.054233x** |

16/24 两档保留约 1.45x 的局部机会，又避免通用 14--18 bit 运行时 barrel network。
如果 exact trace 过门，第一版 RTL 应是：

1. per-phase 静态 width mode `{16,24}`，不是 per-value 动态 header；
2. `BUS_W` 编译期参数，主实现先固定 512 bit；128 bit 只作为第二次 elaboration 验证
   参数正确性，不增加运行时双总线 mux；
3. 32-lane signed replay 输出、cross-beat carry、尾拍 valid mask、任意 backpressure；
4. unpack 后 sign-extend 回同一个 Q24 边界，再进入原 M480 affine/consumer。

只有 16/24 两档的真实 PPA/能量失败后，才有理由评估更复杂的 14/16/18 三档；不建议
直接做完整 14/15/16/17/18。

## P0：M161 的 width 不能直接用

M161 independent review 的关键原文事实是：这些 width 来自 checkpoint INT8 weight
`sumabs`，其语义是 **binary-input analytic bound**。它没有证明冻结 H67 的实际
ATLIF/amplitude code 如何进入 FC1/FC2 整数点积，也没有证明：

- activation codebook 与 scale；
- dot 的 binary point、累加顺序、bias 和 saturation；
- Q24 对动态 alpha/offset、moment 和 correction 的 overflow safety；
- 当前 PyTorch/no-running 路径到整数路径的 valid825 identity。

所以 M502 现在最多能说：若 integer bridge 证明 raw dot 位于这些 signed bounds 内，
窄 container 经 sign extension 可以对 **假设的 M480 Q24 container** bit-exact。它不能
说“对模型无损”，更不能用 accumulator bound 代替 AEE 验证。

## prior art / novelty 裁定

standalone novelty：**NO_GO**。

- 本地 CICC'26 光流芯片的 BWAC 已按权重 group 选择 minimum width、紧凑排列，并用
  在线 BWADU 解压；论文还把外存流量、延迟、能量和 AEE 联合报告。M502 若只把对象从
  weight 换成 raw activation，差异不足以独立成贡献。论文 DOI：
  <https://doi.org/10.1109/CICC65509.2026.11509564>。
- Stripes（MICRO'16）已经让执行时间随表示位宽缩放：
  <https://people.ece.ubc.ca/aamodt/papers/stripes-final.pdf>。
- Loom（DAC'18）覆盖 layer/profile-derived activation precision 和更细粒度 runtime
  trimming：<https://arxiv.org/abs/1706.07853>。
- Bit Fusion（ISCA'18）覆盖 layer-wise bit-flexible compute/communication：
  <https://arxiv.org/abs/1712.01507>。

可保留的项目特异性只是一个**支撑子机制**：利用 spike/event integer domain 与
checkpoint weight `sumabs` 给 current-batch BN 的 barrier raw tensor 做可证 signed
container 收窄，同时保持 raw write/replay 与 fused normalize 的协议。这可以补动态 BN
收口和 memory-inclusive energy，但不应作为新的一条主贡献，也不能沿用 CICC 的 BWAC
命名或 71.4% 数字。

## RTL 前硬门

必须依次通过，任一失败即 NO_GO 或降为 traffic-only：

1. 冻结 H67 ep35 integer bridge：activation code/scale、INT8 weight code/scale、bias、
   dot order、binary point、round/saturation 与 pre-BN raw 边界；
2. S10 的 24 phase 全部证明 width fit，`overflow_count=0`；同时给 analytic domain proof，
   不能只拿样本最大值当证明；
3. 生成 address-timed 1R1W raw write/replay trace，显式包含 barrier、coefficient-ready、
   tail/padding、metadata、backpressure 和实际 accepted values/cycle；
4. 512-bit 同 lane cap 下，计入所有 pack/unpack stall 后仍需 local schedule
   `>=1.25x`；否则只能继续做能量评估；
5. 16/24 两档需保留 useful traffic `>=1.35x` 且 local schedule `>=1.20x`，才允许
   开 RTL；
6. VCS 必须覆盖 signed extrema、cross-beat carry、partial tail、任意 stall/reset，逐值
   对 Q24 sign-extension reference 0 mismatch；
7. 3 ns DC/STA 后 pack/unpack+control `<=15,000 um^2`；匹配 PTPX 加 SRAM/内存模型后，
   raw path 净能量至少降低 20%，不能只报 traffic。

## 所需 exact trace

最小交付不是完整 437.76M-value tensor dump，而是可流式审计包：

- 身份：checkpoint/config/data-list SHA、sample/sequence、module/stage/block/phase、tensor
  shape、`bn_policy=no_running/current-batch`、batch size 1；
- integer bridge：输入 codebook/scale、weight code/scale、bias、累加与 rounding/saturation
  顺序；
- 每 phase：signed min/max、exact required-bit histogram、analytic-bound margin、
  `overflow_count=0`、元素数、rolling/raw/zlib SHA；
- 向量：所有 extrema、正负边界、跨 128/512-bit beat 和尾拍的分层样本；
- transaction：raw address、cycle、R/W、width mode、beat valid mask、barrier、
  coefficient-ready、consumer-ready、实际每拍接收值数和 1R1W conflict。

S10 先用于 fail-fast；论文准入还需要 held-out valid 和至少三个 DSEC sequence 的宽度/流量
分层，不得把 train-only 多序列冒充 held-out。

## 最终评分

| 维度 | 分数 | 判断 |
|---|---:|---|
| arithmetic / schedule recompute | 19/20 | M480 Q24 点独立复现，64 点 bus/lane/width DSE 闭合 |
| data identity / grain | 17/20 | 上游 SHA 与 12-module/24-phase 粒度清楚，但没有 raw trace |
| numerical soundness | 7/20 | M161 P0 尚未解决，只能 conditional container equivalence |
| implementation evidence | 3/20 | 无 packer、lane consumer、SRAM、VCS/DC/PTPX |
| standalone novelty | 3/20 | CICC/Stripes/Loom/Bit Fusion prior art 很强 |
| claim discipline | 15/20 | opportunity、sensitivity、performance 三者分离 |
| **总分** | **64/100** | **GO_OFFLINE_AUDIT_ONLY；RTL 未提名** |

这条线值得花半天到一天补 exact trace 与 address schedule，因为理想流量窗够大；但在
integer bridge 和 lane/service gate 前写 RTL，风险高于收益。
