# M507 r3 APEC-G2 same-resource cycle fast-kill 独立 preflight 评审

日期：2026-08-27  
范围：只读静态、身份、冻结 metadata 与独立数学审查；没有 import/执行 production analyzer，没有解压或全量 replay payload，没有启动 VCS/DC/GPU，没有修改 production、contract 或 `docs/359`。

## 裁决

**`NO_GO_REVISE_R4_BEFORE_ONE_SHOT_EXECUTION`，72/100。**

r3 确实关闭了 r2 明面上的两个 blocker：r1/r2 seal 的“外层 seal 文件 SHA”和“内层 `SHA256SUMS` SHA”在 contract 中已经分栏且数值正确；32,832 B 的 two-destination slots 也终于映射到每 slot 8 bank、每 bank 1R1W、32 B/cycle、同步读 1 cycle 的具名端口。两臂的 final destination read 和 sink write 对 validation + train 每条 record 都有对称门。

但 r3 暴露出一个更深的 P0：**candidate 的 common-event service 没有任何可计价的累加状态**。baseline/residual 的每个计算周期都向 destination slot 支付 RMW，而 common path 只支付 MAC/weight service，随后把一个已经形成的 9×768×19-bit overlap vector “一次写入” scratch。这个 vector 在写入前由谁保存、如何跨多个 common events 累加、如何 zero-init、端口是否能承受 96 lanes，都没有进入资源或周期账本。当前模型因此混用了两种 dataflow，倍率不能作为永久 GO/KILL 依据。

另有 final synchronous-read→sink 少一拍、物理 bank padding bytes 没有报告，以及 one-shot 输出非原子等缺口。不得运行 r3 production main，不得消耗唯一一次 fast-kill 配额。

## 1. 身份与 r1/r2 seal 层级

当前 analyzer SHA 为 `561d3e06...2045`，contract SHA 为 `34128e04...b88a`；contract 锁定的 9 个 inputs（含 `docs/359=dedde7ce...`）全部逐 SHA 匹配。

| review | seal 文件外层 SHA | seal 内容指向的内层 manifest SHA | contract 分栏 | 当前 manifest members |
|---|---|---|---|---|
| r1 | `4f79d4ba...f538` | `97ac421a...683` | 正确 | review 三件与 M501/docs 仍匹配；旧 r1 contract/analyzer 两条路径已不存在 |
| r2 | `a6701d3a...6d30` | `44b4ff3e...c02` | 正确 | 3/3 匹配 |

production `main()` 目前只校验 seal 文件外层 SHA，不解析 `sealed_manifest_sha256`，也不执行 inner manifest。因为外层 SHA 已钉死，**它能证明拿到的是同一个历史 seal 文件**；但不能把 r1 描述成“当前可完整重放的 seal chain”，因为 r1 manifest 的两个 mutable production 路径已被移走。r1 应标成 historical provenance，不能作为当前源码完整性门。这个问题是 P1，不单独阻止数值 replay；当前 P0 来自周期模型。

## 2. 通过项：240 KiB 容量和 destination 端口代数

独立重算：

| 组件 | bytes |
|---|---:|
| pair bitmap | 192 |
| overlap cache | 16,416 |
| two destination slots | 32,832 |
| payload/weight window | 196,320 |
| 合计 | 245,760（240 KiB） |

单个 full-9-tap slot 是 `768×9×19/8 = 16,416 B`。按 output channel mod 8 条带，每 bank 是 2,052 B；32 B/cycle 下 65 cycles。96 lanes 均匀分到 8 bank 后，每 bank 每拍逻辑需求是 `ceil(12×19/8)=29 B`，小于 32 B，所以在锁定映射下，1R1W bank 可以支撑一拍一次 read + write。

但当前输出的是 logical bytes。一次 full-slot 物理 bank transfer 是 `65×8×32=16,640 B`，比逻辑 16,416 B 多 224 B padding；19-bit RMW 每拍也是 29 B logical / 32 B physical per bank。M507 虽不报能量，required traffic ledger 仍应把 logical payload bytes 与 physical transferred bytes 分列，避免后续误用其定价。

## 3. P0：common partial sum 使用了未计价状态

源码第 515–526 行对 residual0、residual1、common 都计算 `service_terms()`，但只对两个 residual 调用 `destination_rmw_terms()`。common 的多个 active input-channel event 要形成 `union_taps×768×19-bit` vector，至少需要以下一种真实组织：

1. 在 overlap scratch 上逐拍 RMW；但当前 scratch 只有 128 B/cycle，而 96×19-bit lane 需求为 228 B/cycle，不能支撑；
2. 在某个 destination slot 上 zero-init + RMW，再复制/shift；则必须支付该 slot 的 init、RMW、read 和 scratch write/seed；
3. 使用 ExSpike 式 lane-local overlap accumulators；则必须显式保留第二套 accumulator state，锁定 block schedule，并按每 block drain，而不能先计算完整 16,416 B vector 后再做一个全局 packed write。

当前资源表只有 `compute.accumulator_lanes=96`，没有 normal/overlap 两套状态；第 546–568 行的 scratch `write_pass` 又发生在 common compute 完成之后。若只有 96×19-bit accumulators，9×768 项必须分 72 个 96-lane block；每个 228 B block 经 128 B/cycle scratch 至少 2 个 write cycles，串行 drain 是 144 cycles，不是对整个 16,416 B 一次 `ceil(.../128)=129` cycles。跨 block 的 129-cycle紧凑流需要额外 packing/buffering 和与 compute 的时序说明。

因此 r3 对 baseline/residual 采用 SRAM-RMW dataflow，对 common 却采用未定价的 register-accumulate/bulk-store dataflow。这个不公平项随 overlap groups 变化，会直接改变 validation/train 倍率，不能用一个常数修补。

## 4. zero/seed/RMW/final/sink 守恒审计

通过：

- baseline 有 event 时付 zero-init；candidate 无 common 时付 zero-init，有 common 时由两次 scratch read 分别 seed 两个独立 destination slots，zero 和 seed 互斥；
- residual RMW 的 read/write logical bytes 相等，且每个非空 residual stream 支付 1 个 synchronous tail；
- scratch 是一次 write + 两次串行 synchronous read + 两个 read response tail；seed-write bandwidth 256 B/cycle 不慢于 scratch read 128 B/cycle，因此 seed write 可以在独立端口上随 response 完成；
- 两臂对相同非空左/右 destination 支付相同 final read 与 sink write，validation + train 每 record 均被 gate。

未闭合：

- hard gate 只检查 `RMW read=write` 和 `final read=sink write`，没有检查每个非空 destination 恰有一次 `zero-init XOR seed`，也没有检查 `zero+seed bytes = final vector bytes`；
- scratch 只记 transaction count，不记 logical/physical read/write bytes；
- common vector 的 zero/init/accumulation 缺失，见 P0；
- `destination_commit_transactions` 实际只装 sink transactions，slot-read transactions 虽局部计算但没有聚合/对称门，名字会误导。

## 5. P0：final synchronous-read→sink 少 1 cycle

`vector_transfer_terms()` 当前使用：

`max(slot_cycles + read_tail, sink_cycles)`

full-9-tap vector 的 slot read 是 65 cycles，sink 是 129 cycles，read latency 是 1。第一批 sink data 只能在首个 synchronous response 到达后开始，因此无预取前提下应是：

`max(slot_cycles, sink_cycles) + read_tail = 130 cycles`

而不是 129。当前注释也写了“slower side plus first-response latency”，代码却只在 slot 侧不是 slower side 时加 tail。每个非空 destination 少 1 cycle；这个项两臂对称，预计只小幅改变 ratio，但 exact one-shot 必须先修。

## 6. 96-lane/bank、border/union 与独立边界组

weight 代数成立：每个 event/tap 每 bank 为 `768/8=96 B`，16 B/cycle 为 6 cycles；96 compute lanes 每拍对应每 bank 12 个 INT8 weight，低于 16 B/cycle。destination 每 bank 每拍 29 B，低于 32 B/cycle。冻结 H=15/W=20 下，interior taps=9；top-left horizontal pair 是 4/6，union=6；width 20 无 G2 tail。

按 r3 当前公式独立手算：

| case | baseline | candidate | ratio |
|---|---:|---:|---:|
| empty interior | 2 | 3 | 0.666667× |
| one each, full overlap, interior | 536 | 722 | 0.742382× |
| one each, no overlap, interior | 536 | 537 | 0.998138× |
| one each, full overlap, top-left pair | 299 | 454 | 0.658590× |

这些数与源码公式一致，但只复现了当前混合 dataflow，不证明其公平。源码通过“总 bytes/per-bank cycles 相等”推断零 bank conflict，却没有显式锁定 `bank=output_channel mod 8` 和 96-lane issue 顺序；r4 应把这两个 mapping 写进 result ledger，而不只留在叙述中。

## 7. validation/train gates、schema 与 one-shot 语义

通过：

- M501 validation/train per-record、overall、per-sequence 的三项 event ledger 已有 fail-closed 对账；
- final path symmetry 遍历 `validation_rows + train_rows`；
- contract/result schema 与 status 边界明确；任意 gate fail 才会产出 permanent KILL，全部通过也仅是 `PASS_EXSPIKE_DERIVED_SUPPORT_ONLY_NO_STANDALONE_RTL`；
- `rtl/synopsys/energy/ppa/system_speedup/date_headline=false`，claim boundary 正确。

阻塞：输出不是 transaction-like one-shot。第 1142 行直接创建最终 output dir，随后逐文件写入；若进程中断，会留下一个不完整目录，而下一次因 no-overwrite 永久拒绝。`RUN_COMPLETE.txt` 又在 seal 之后写入且不在 seal 内。唯一一次 production 应先写同目录下唯一 staging 路径，完成全部 invariant/self-SHA/seal 后原子 rename；最终 seal 或 sealed completion marker 才是成功条件。exact-SHA runner 还必须钉住当前 contract SHA，而不能只依赖 analyzer 自检。

另外，`locked_bank_mapping_zero_conflicts` 与 `single_entry_queue_respected` 最终 gate 只读取 validation aggregate；虽然当前 helper 对每条 train record 都按构造输出零/一，r4 仍应把 train 一并纳入显式 gate，与“train worst sequence”合同一致。

## 8. 唯一允许的 r4 最小修复

不得改 axis/G2、阈值、gate、cohort 或 claim boundary，不得先跑 r3 再修。

1. **先选择一种唯一 dataflow**，对 baseline、residual、common 一致使用。显式列出 common partial-sum 从 zero-init、每 event accumulation、scratch store 到两次 seed 的状态和端口；若使用 lane-local overlap accumulators，把第二套 accumulator lanes/packing buffer列入两臂保留资源，并按 block schedule 计 cycles；若使用 SRAM RMW，端口必须满足 228 B/cycle且所有 RMW/tail均收费。
2. 修 final commit 为 `max(slot_cycles, sink_cycles)+read_tail`，并补 slot-read/sink transaction、logical/physical bytes。
3. 加 per-record 守恒门：每个非空 destination 必须满足 `zero XOR seed`；`zero_init_bytes + seed_write_bytes = final_read_bytes`；scratch write/read logical + physical bytes可追踪。
4. 明示并 assert bank mapping 和 lane issue 顺序；validation/train 都纳入 bank/queue gates。
5. 输出改为 staging + 原子发布，完成标志纳入 seal；runner 钉 r4 analyzer/contract/input/review-seal SHA。
6. r4 再做一次独立静态审查。若仍不能在一个具名 dataflow 下守恒，**直接关闭 M501/M507，不再迭代或运行 production**。

## 9. 不变的论文边界

APEC 是 ExSpike 直接 prior art，冻结 trace 仅零加一个正 operator-constant codeword，exact value overlap 等于 support intersection。即使 r4 最终过全部性能门，也只能作为明确引用 ExSpike 的 workload/cycle supporting audit；不能开发 standalone APEC RTL，不能称 signed-analog innovation、系统实测倍速或 DATE headline。

可复现独立审计器：`audit_m507_r3_preflight_independent.py`。它不 import production analyzer、不打开 payload、不运行 production main。
