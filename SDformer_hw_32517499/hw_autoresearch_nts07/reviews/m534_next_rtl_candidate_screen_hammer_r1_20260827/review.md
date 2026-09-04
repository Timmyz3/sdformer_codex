# M534 下一条 RTL 候选筛选独立打铁 r1

日期：2026-08-27  
被审对象：`reviews/m534_next_rtl_candidate_screen_r1_20260827`  
模式：只读本地证据与一手论文；零 HDL/EDA/训练/远端运行；未修改 M534 或 `docs/359`  
裁决：`FAIL_CLOSED__DO_NOT_CREATE_PDR_CPU_EXECUTION_CONTRACT__AUTHOR_R2_SCREEN_REPAIR_ONLY__B_REMAINS_DEFERRED`  
评分：**68/100**；P0/P1/P2 = **1/7/3**

## 结论

M534 对 paper claim 的纪律总体正确：Prosperity/Phi/FireFly-T 的机制没有被改名成 `first`，
decoder dense-to-bit 的 `4.48--4.81x` 没有冒充我方倍率，B 的 `27.5346%` 也始终写成
local metadata movement **静态有利上界**，没有升级成 measured area/power/energy/cycle。
M534 的 member 与 outer seal 均可在其目录内通过；五个冻结本地输入 SHA 与当前文件一致；
`docs/359` 仍为 `dedde7ce...`。

但候选 A/PDR 当前不是一份可执行 pre-RTL 规格。最小 leaf 的 context 几何在第一个正常 interior
event 上就没有合法接收动作；公平 baseline 又排除了最关键的 output-stationary/Gustavson 强点，
memory/service 账也没有把 M218 的六个 16-lane slice、psum SRAM 和 bounded-state spill 写成周期规则。
因此本 hammer **不允许创建 M534 所述 PDR CPU execution contract，更不授权运行或 RTL**。
唯一允许的下一步是先 author 一份 M534 r2 screen repair；r2 经新 reviewer 打铁 P0/P1=0 后，
才可创建仍然 `run_authorized=false` 的 CPU contract。

候选 B 的冻结 r2 规格比 A 成熟，继续等待 M519 三轴 DC 是对的；本 hammer 不授权提前开发 B。

| 候选 | M534 分数 | 本 hammer 分数 | 裁决 |
|---|---:|---:|---|
| A｜decoder PDR | 82 | **52** | 先修规格；当前不准建 CPU execution contract |
| B｜typed-ticket tag elision | 74 | **76** | 保留为 C2 physical ablation；等待 M519 canonical |

## P0-01｜两 context/phase 无法接收 M523 的正常原子 bundle

M523 的冻结接口不是逐 lane 可独立握手：一个 `bundle_accept` 原子接收所有 valid lanes。其 RTL 明确：

- interior source event 产生 9 个 taps；
- phase-major 顺序为 `4 odd/odd + 2 odd/even + 2 even/odd + 1 even/even`；
- 因而首个 full-8 bundle 内同时含 **4 个不同 odd/odd destination**。

M534 的最小 leaf 却规定每个 phase bank 只有两个 destination context，且 partial group 只允许由
frontier/fence 解释关闭。对同一 active source channel 的首个 interior event，前两个 odd/odd context
占满后，第三个 odd/odd destination 到达；旧 context 仍可能收到之后 channel/space 的 contributor，
不能合法 frontier-close。当前规格也没有定义 capacity eviction、spill-to-psum 或 split-bundle handshake。
所以 leaf 只能丢 lane、提前提交、增加未收费 state，或永久 backpressure；四者都违反合同。

这不是“exact trace 后也许利用率低”的 P2，而是 nominal legal input 上的 transition relation 缺失。
r2 必须二选一并收费：

1. context 容量至少覆盖一个原子 bundle 的逐 phase distinct-destination 峰值，并证明跨 row 的长期
   live-set 上界；或
2. 定义可执行的 atomic spill/evict/restore 协议，把每次 psum read/write、L4 latency、port conflict、
   forwarding 和重放周期计入 A1/PDR 两点。

在修复前，`c2d_phase_banked_destination_join_leaf` 不是可实现的最小 RTL leaf，A 的 82/100 与
“唯一推荐”均不成立。

## P1

### P1-01｜A1-ISO8 不是 strongest exact baseline

M534 只允许 A1 合并 current head/相邻 exact destination，同时给 PDR 相同 bit budget 的
destination-keyed bounded join。这个限制会把 associative accumulation 的收益预先送给 candidate。
M512 已明确要求 PGPR/PDR 类机制同时对照“已有 parity banking 的强 source-centric A1”和
“常规 output-stationary A1”；若后者与 PGPR 代数同构则应 KILL。

r2 至少要有三行：source-centric K8 A1、相同 state/port 的 output-stationary or Gustavson A1、PDR。
论文倍率只许相对三者中的 strongest exact measured baseline。只用 A1-ISO8/head-adjacent 会制造
可预期的 paper-only speedup。

### P1-02｜M218 的物理 service 没有进入 CPU 周期定义

冻结 M218 不是“一拍 96 lane × 8 source”的抽象黑箱：每个 group 展开为 **六个 16-lane request**，
每 request 最多从八个 128-bit bank 各读一个 word，并受 O8/FIFO4、L4 response、slice busy、
out-of-order response 和 Acc24 context 顺序约束。M534 同时写“八个 128-bit bank”“96 Acc24 lane”
和“最多八 source”，却没有说明一个 group 的六 slice 是否收费、何时占/释 psum context、是否允许
slice overlap。若直接按一 group/拍计，等价于给 candidate 一个未声明的 6144-bit weight ingress；
若按 M218 计，则当前 cycle gate 不可计算。

r2 必须冻结唯一 service：建议直接复用 M218/M519 的 L4/O8/FIFO4 六-slice 状态机；A1 与 PDR
共享 exact 同一 service receipt。不得把“96 output lanes”写成“96 product/cycle”来隐藏八路 adder tree。

### P1-03｜psum/weight 容量、端口和 completion 物理税未闭合

“同四个 phase-psum ports、同八 weight bank”不是完整资源模型。当前缺：

- 每 phase psum bank 的 depth/width、1RW 或 1R1W、read/write latency、RAW forwarding 与 commit port；
- output tile 尺寸、live destination 上界、context spill/restore、row-boundary drain；
- 四层 weight bank 的可驻留容量、tile refill、DRAM/SRAM bytes 与 refill stall；
- PDR 同 phase 多 destination 时，是一组/拍还是四组/拍；若后者需四套 K8 service，不能称同资源；
- completion scoreboard、frontier/fence 自身的 state bits、compare/fanout 与 activity。

在这些字段缺失时，`weight_reads不得增加` 和 `psum traffic -30%` 不能推出 SRAM-aware cycle/energy。
CPU contract 需给每个 state/transaction 一条容量、端口、延迟与能量归属，禁止 0-macro/free-state。

### P1-04｜decoder share 门会混用 S10 decoder 与 S100 included scope

M511 只捕获固定 `zurich_city_09_a` 的 S10 decoder input；M513 r2 已明确将 S10 decoder 加 S100
included scope 标为 `mixed_cohort_sensitivity_admitted=false`。M534 却用 `decoder exact share>=15%`
作为 GO-to-RTL 门，未要求同十 sample 的非 decoder ledger。r2 应：

- 为同一 S10 生成 included-scope denominator，之后才算 exact decoder share；或
- 在 CPU fast-kill 中暂时删除 share 门，只用局部同资源倍率，系统 share 等 E5 same-cohort ledger。

当前 `21.57--22.83%` 只能是 aggregate S100 分析敏感性，不能替代 exact share。

### P1-05｜ELSA 的直接相邻协议没有被完整纳入 novelty 边界

M534 正确引用 ELSA 的 BAER 和 mini-batch Gustavson，但把差异收窄成“PDR 不是 NoC bundling”仍不够。
ELSA 一手论文还明确：同一 membrane row 的 spike batch 只读一次、并行累加多个 weight row、只写回
一次；其 output scheduler 还根据依赖 spine 的到达顺序决定何时输出。这已经占据“row/destination-keyed
accumulation + data-dependency completion”的大部分故事。

因此 PDR 可保留的窄 claim 只能是：**K3/S2 ConvTranspose 的 exact parity-dependent contributor
frontier，在 bounded state 与 C2 typed signed-source service 下的实现/测量**。不得声称发明 destination
rendezvous、single-RMW Gustavson 或 dependency completion。Prosperity/Phi/FireFly-T 的边界则基本准确。

### P1-06｜pre-RTL 与 post-DC 门混在同一个 GO 条件

`ratio-of-sums>=1.30x`、每 sample `>=1.10x` 和 traffic 是 CPU gate；`added area<=20% A1` 是 paired
DC 后的物理 gate。当前所谓“分析 added logic/state<=20%”没有可冻结的 cell-area 分母，不能在 author
RTL 前证明。r2 应把流程拆为：CPU GO → source-only RTL author admission → VCS → paired DC area/Fmax
gate。CPU 只报告 exact state bits/comparator counts，不冒充 mapped area。

### P1-07｜B 的 JSON 放宽了冻结的 zero-cycle-delta 合同

M534 Markdown 正确要求 cycle/traffic 0 mismatch，B 的上游 r2 规格也要求周期必须 0 变化；但 M534
JSON 写成 `cycle_change_max_percent=1.0`。这会允许 JSON consumer 接受非零调度变化。必须改为 exact
zero（整数 cycle/accept/traffic 全等），不能用 1% 容差覆盖协议差异。

## P2

1. `ratio-of-sums>=1.30x` 与每 sample `>=1.10x` 作为单序列 pre-RTL fast-kill 是合理的保守门，
   且 1.30 与 M510 一致；但最终 paper 还需逐 layer×sample 和多 sequence 分布，不能把 S10 通过写成
   robustness。
2. B 的 PTPX 方向正确：clean DC 后无论 area 是否过门都必须做 matched mapped-gate SAIF/PTPX。
   r2/CPU-screen 应显式继承上游 `exact net/leaf annotation=100%`、nonzero-toggle coverage、同 contiguous
   window、同 macro boundary 和 dynamic=`internal+switching`，不能只写泛化的 `annotation coverage`。
3. OpenEye 的 `variable-length FIFO`/row-stationary 细节在进入 related-work 正文前应引用 paper 具体节或
   official RTL 路径；当前 primary abstract 只直接支持 streaming dataflow、sparse weights/activations 与
   buffering/scalable routing。此项不影响 A 的主裁决。

## 一手来源核验

- [Prosperity](https://arxiv.org/html/2503.03379)：subset/exact-match product reuse、runtime detector/
  dispatcher 的概括成立；M534 没有把它改名成 PDR。
- [Phi](https://arxiv.org/html/2505.10909)：L1 pattern precompute + L2 residual sparsity + PAFT 的概括成立。
- [FireFly-T](https://arxiv.org/html/2505.12771)：multi-lane decoder、wide-memory load balance 与
  `P_Wo` out-of-order worker dispatch 的概括成立。
- [ELSA](https://arxiv.org/html/2605.20802)：BAER、row-aligned mini-batch Gustavson、single
  membrane read/write 和 dependency-aware output schedule 均是直接相邻 prior；M534 的 novelty 边界需收窄。
- [Transposed-conv decomposition](https://arxiv.org/abs/2205.02103)：decomposition/skip redundant
  computation 是既有机制；M534 正确禁止把 polyphase 当创新。
- [SNE](https://arxiv.org/abs/2203.12437) 与 [ESDA](https://arxiv.org/abs/2401.05626)：event-proportional
  queue/self-timed execution、memory interlacing、uniform sparse token-feature interface 的边界基本准确。

## r2 screen repair 的硬条件

只有同时满足下列项，后续 reviewer 才可允许创建 pre-RTL CPU contract；该 contract 仍不得授权运行：

1. 修复 atomic 8-lane bundle 与 per-phase context 的容量/eviction/spill transition，给 exhaustive 小尺寸
   reference 证明无 deadlock、无 early close；
2. 加入 output-stationary/Gustavson strongest A1，所有倍率相对 strongest exact baseline；
3. 锁定 M218 六-slice L4/O8/FIFO4 service，以及 weight/psum SRAM 容量、端口、延迟、refill、RMW、
   forwarding、commit 和 energy transaction；
4. 同 cohort 计算 decoder share，或从 CPU fast-kill 暂时移除 share 门；
5. claim 明确引用 ELSA output scheduler/Gustavson，只保留 ConvTranspose parity frontier + bounded-state +
   typed-source 对象差；
6. CPU gate 与 DC area/Fmax gate分层；
7. B JSON cycle delta 改为 exact zero，并继承 r2 的 matched PTPX exact coverage 条款；
8. M511 payload verifier 与 M513 production result 均有独立双封后，才可考虑一次 future CPU run。

## 身份与授权边界

- 被审 M534 `README.md` SHA256：`df46183bf80e5cdd803c60e813db5ef1d5ba8f0d0f019485f599c1f918b72fb6`；
- 被审 M534 JSON SHA256：`68a46bac57c710781931384a7d62266237bf13a683543013ccdb92c2f5b7e321`；
- 本地五个 frozen input SHA 与 M534 JSON 一致；M534 member/outer seal 均通过；
- `docs/359` 未修改，SHA256 为
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`；
- 本 hammer 只准入“r2 screen repair authoring”，不准入 CPU execution contract、CPU run、RTL、VCS、
  DC、PTPX、Formality、远端、cycle/area/power/energy/system/headline claim。
