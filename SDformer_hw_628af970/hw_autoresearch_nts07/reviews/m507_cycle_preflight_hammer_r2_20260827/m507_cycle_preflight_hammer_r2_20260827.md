# M507 r2 APEC-G2 same-resource cycle fast-kill 独立 preflight 评审

日期：2026-08-27  
范围：独立静态、身份、冻结 metadata 与数学审查；没有 import/执行 production analyzer，没有解压或全量重放 raw payload，没有创建 production result，没有修改生产文件或 `docs/359`。

## 裁决

**`NO_GO_REVISE_R3_BEFORE_ONE_SHOT_EXECUTION`，78/100。**

M507 r2 已关闭 r1 的大部分方法学缺口：两臂都支付两个 destination-vector commit；candidate scratch 明确支付一次 write、两次同步 read 和两个 response tail；8-bank weight/output 代数在锁定 mapping 下无冲突；串行 1-entry queue 的 backpressure 有显式计数；M501 validation/train 的 per-record、overall 和 per-sequence 三字段对账已经编码；payload codeword、Conv geometry、sample×operator 笛卡尔覆盖也已 fail closed。

但 r2 仍有两个 P0，不能运行 production。第一个会令程序在 raw replay 前必然失败；第二个使 240 KiB “容量相等”尚不能推出“端口与周期公平”。因此本轮不得消耗唯一一次 production fast-kill 配额。

## 1. P0：旧评审 seal 锁错了对象

contract 的 `m507_r1_preflight_review_seal.sha256` 是：

`97ac421a8598e4da24f6fbd0af53ed1d2eadd8668c309c0b0b68d1b8417fc683`

这个数是旧 seal 文件**内容中记录的 `SHA256SUMS` manifest SHA**，不是 seal 文件本身 SHA。当前 seal 文件本身 SHA 是：

`4f79d4baa826249ef65686c570428c0512cfce1156a8c900619196236113f538`

production `main()` 会对每个 input 执行 `sha256(path) == spec["sha256"]`，所以 r2 必然在加载 raw payload 前报 `M507 input SHA drift: m507_r1_preflight_review_seal`。其余七个 contract input SHA 均匹配；`docs/359` 仍为冻结 SHA `dedde7ce...`。

最小修复：r3 contract 应锁 seal 文件的外层 SHA `4f79d4...`，同时把旧 seal 内层 `97ac...` 作为 seal 内容验证值另列，不能混用两个层级。

## 2. P0：destination accumulator slots 有容量、没有端口

独立重算的容量账本是：

| 组件 | bytes |
|---|---:|
| pair bitmap | 192 |
| overlap cache | 16,416 |
| two destination-vector slots | 32,832 |
| payload/weight window | 196,320 |
| 合计 | 245,760（240 KiB） |

这个**容量和式本身正确**，baseline/candidate 也从同一个字典复制，所以静态相等。但 `build_resource_ledger()` 的 ports 只有：pair bitmap read、overlap-cache 1R1W、weight read banks、write-only `output_sink`、compute lanes 和 group queue。它没有给 32,832 B `two_destination_vector_slots_bytes` 对应的：

1. scratch read 后写入 destination accumulator 的 seed 端口；
2. residual event 对已 seed accumulator 的 read/modify/write 端口；
3. final commit 从 destination slot 读出并送往 output sink 的读端口；
4. 上述访问与 compute、scratch、sink 是否串行/并行的周期守恒。

`output_sink` 当前只有 `write_banks: 8`，不能替代 destination slot 自身的 seed/RMW/readout 端口。`service_terms()` 只统计 MAC/weight 周期，`vector_transfer_terms()` 只统计最终 sink write；中间 accumulator slot 的物理访问没有在任何 cycle/byte ledger 中出现。因此 `same_capacity_and_ports_derived` 仍是构造相等，不是完整端口守恒证明。

这个缺口会改变倍率，不是文案问题。candidate 需要把两个 scratch read 的返回值写入两个 destination slots；若 seed write 不能和 scratch response 同拍完成，当前每个 read 只加一个 tail 仍可能少算。若允许同拍，r3 必须锁定相应独立写口/带宽并在两臂保留同样硬件。baseline 的 zero-init、正常 accumulation 和 final readout也必须使用同一具名资源。

## 3. 已关闭的 r1 缺口

### 3.1 两臂 destination 路径对称

源码对 baseline/candidate 分别计算 `bcommit`/`ccommit`，两者都按左、右非空位置支付相同的 final vector transfer。validation aggregate 还要求 commit cycles 和 transactions 相等。candidate 的 scratch 两次 read 被明确称为 seed，final commit 另计，没有沿用 r1 的“scratch read 兼作 commit”歧义。

建议 r3 把对称门从 validation aggregate 提升到 validation + train 的 per-record 级；这不是当前 P0，但能避免总量偶然相抵。

### 3.2 scratch 同步尾拍已补

每个 overlap group：

`scratch_cycles = write_pass + read0_pass + read1_pass + 2 response tails = 3×pass + 2`

源码同时分别统计 write/read transactions、serialization stalls 和两个 read tails。该部分与 r2 contract 一致。

### 3.3 8-bank mapping 在锁定条带规则下成立

weight 以 `bank = output_channel mod 8` 均匀条带：每个 event/tap 每 bank 恰为 `768/8 = 96 B`，16 B/cycle 下为 6 cycles；聚合 128 B/cycle 也为 6 cycles，无额外 bank conflict。

output 同样按 output channel 条带，每 bank 每 tap 为 `96×19 = 1,824 bit = 228 B`。full 3×3 vector 每 bank 为 2,052 B，16 B/cycle 下 129 cycles；聚合 bit-packed 计算也为 129 cycles。代码对锁定 768/8、128=8×16 和 per-bank/aggregate cycles 作了 require。

这证明的是给定 bit-packed、串行 tap mapping 的冲突为零，不是宏的时序/能量证明；claim boundary 已正确保持 `synopsys=false`、`energy=false`、`ppa=false`。

### 3.4 1-entry queue/backpressure 已显式但只是串行解析模型

每组完全串行执行，queue occupancy 固定不超过 1；backpressure 记为 `group_service_cycles - 1`。这与 contract 的“保守 serialized schedule”一致，并避免 r1 的恒零伪 conflict。它不是逐拍动态 queue 仿真，论文中只能称 serialized analytical cycle model。

### 3.5 M501、payload、geometry 和覆盖检查已补齐

- M40：40/40 个 sample×operator pair 唯一完整；
- M73：128/128 个 pair 唯一完整，覆盖 32 samples、18 train sequences；
- 两个 manifest metadata 均为冻结 `T=10,B=1,C=768,H=15,W=20`、Conv 768→768、3×3、stride/pad/dilation=1、groups=1；
- 每个 operator 只允许 `+0` 与合同锁定的一个正 float32 codeword，production 会对解压后 payload bit pattern 再逐元素检查；
- M501 horizontal G2 的 validation 40 records/1 sequence 与 train 128 records/18 sequences 均存在唯一 per-record、overall、per-sequence ledger；r2 源码逐项对账 baseline/candidate/overlap events。

本 preflight 没有解压 raw payload；它验证的是 production 已锁定 payload SHA、decoded SHA 与逐元素 codeword 检查，以及冻结 manifest metadata 当前满足这些先决条件。

## 4. 独立手算边界组

以下不 import analyzer，使用合同公式独立重算。96 lanes；INT8 weight 8×16 B/cycle；19-bit output 8×16 B/cycle；bitmap 2 cycles；candidate compare 1 cycle。

| case | baseline | candidate | ratio | 对称 final commit | scratch（含 2 tail） |
|---|---:|---:|---:|---:|---:|
| empty interior | 2 | 3 | 0.666667× | 0 | 0 |
| one event/position, full overlap, interior | 404 | 722 | 0.559557× | 258 each arm | 389 |
| one event/position, no overlap, interior | 404 | 405 | 0.997531× | 258 each arm | 0 |
| one event/position, full overlap, top-left pair | 225 | 454 | 0.495595× | 143 each arm | 260 |

top-left pair 的 taps 是 4/6，common kernel-offset union 为 6；interior 为 9/9/9。width=20 可被 G2 整除，没有尾组。手算确认 r2 源码的 destination symmetry、border taps 和 scratch two-tail 公式已经自洽；它不修复第 2 节缺失的 destination-slot 端口。

## 5. 唯一允许的 r3 最小补丁

不得更换 axis/G2、不得调阈值、不得增加免费并行端口，也不得先运行 r2 再修。r3 只能做以下闭合：

1. 将旧 r1 review seal **文件 SHA** 修为 `4f79d4baa826249ef65686c570428c0512cfce1156a8c900619196236113f538`，并区分 seal 文件 SHA 与其所封 `SHA256SUMS` SHA。
2. 给 `two_destination_vector_slots` 增加具名、可计算的物理端口账本。至少要锁定 bank 数、每 bank 宽度、同步读延迟、seed write、residual RMW、final readout；容量项与端口项必须一一对应。
3. 给每组输出 seed bytes/cycles、destination RMW bytes/cycles、final slot-read bytes/cycles、sink-write bytes/cycles。明确哪些周期已包含在 compute，哪些必须串行；不允许同一传输既算 scratch read 又免费写 destination。
4. baseline/candidate 必须保留相同 destination slots/ports。baseline zero-init/normal accumulation/final commit 与 candidate seed/residual accumulation/final commit 应通过同一个具名 schedule；两臂共有 final commit 在 validation/train per-record、aggregate 级全部相等。
5. 若 destination slots 实际是分布式寄存器而非 SRAM，必须从 `common_total_sram_bytes` 中移出并以相同逻辑面积/端口显式计价；不能既把 32,832 B 算作 SRAM 容量，又默认寄存器级免费多端口。
6. 重新锁 r3 analyzer/contract SHA，再做独立 preflight。只有新 preflight GO 后，才允许一次 production fast-kill。

## 6. 不变的 novelty/claim 边界

本地 ExSpike 官方仓库仍为 commit `51accc76936588705255487d101fcc80092b98ce`，官方 RTL 用 overlap accumulator seed 后续位置。M507 冻结 trace 只有零和单个正 operator codeword，exact value overlap 等于 support intersection，没有激活 signed-analog novelty。

因此即使未来 r3 所有性能 gate 全过，也只能得到：

`PASS_EXSPIKE_DERIVED_SUPPORT_ONLY_NO_STANDALONE_RTL`

不得开发独立 APEC RTL，不得写成新机制、系统实测倍速或 DATE headline。任一性能 gate 失败则永久关闭 M501/M507 hardware line；不得 post-hoc 搜 vertical/G4/G8。

可复现静态/数学审计器：`audit_m507_r2_preflight_independent.py`。它不会 import production analyzer，也不会运行 raw replay。
