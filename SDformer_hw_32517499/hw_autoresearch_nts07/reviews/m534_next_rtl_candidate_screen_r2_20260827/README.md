# M534｜下一条 RTL 候选筛选 r2：原子 bundle、强基线与完整存储账修复

日期：2026-08-27  
模式：screen repair authoring only；零 CPU/VCS/DC/PT/PTPX/Formality/训练/远端运行；零 RTL 修改  
状态：`R2_SCREEN_REPAIRED_FOR_FRESH_HAMMER__NO_EXECUTION_CONTRACT__NO_RUN__NO_RTL`

## 1. 裁决

r1 hammer 的 `1 P0 + 7 P1` 全部进入本 r2 的冻结边界。候选 A 不再描述为可长期保留任意
destination 的“两 context/phase PDR”，而收窄为一个有完整 transition relation 的
**PBR4（phase-banked bundle rendezvous with exact partial-RMW backing）**：

- M523 的一个 8-lane bundle 仍然只允许原子接收；
- 冻结合法 K3/S2 phase-major bundle 的最坏值为每个 phase 最多 **4 个 distinct destination**；
- 每个 phase 明确配置 4 个 256-bit context，共 16 个 context；另有收费的 8x128-bit atomic ingress；
- 整个 bundle 先在一个 edge 全部写入 ingress；context 不够时 lane 留在 ingress，禁止逐 lane input
  accept 或丢弃；在该 epoch 排空前，下一 bundle 整体 backpressure；
- join context drain 只是把 exact partial sum 做一次收费的 psum read-modify-write，**不是**语义
  destination close；只有 canonical contributor frontier/fence 才能 final commit；
- context、cache tag、bypass、psum spill/restore、weight refill 全部进入同资源周期和容量账，禁止
  early close、未收费 state 或 free macro。

这一收窄修复了 nominal interior event 的可执行性，但没有证明 PBR4 比强 baseline 更快。未来只允许
在同一个 exact S10、同一个 M218 service、同一个 240 KiB 账本下，同时测
`A1-SC8`、`A1-ISO8`、`A1-OSG` 与 `PBR4`；候选倍率只相对其中一个**固定的 strongest exact baseline**
报告。若 PBR4 与常规 output-stationary/Gustavson 代数或周期等价，就 KILL，不以改名制造贡献。

候选 B 仍是 C2 的 cycle-neutral physical ablation。其周期、accept 和 data-bank traffic 合同均为
exact integer zero delta，不是 1%。M519 三轴 canonical 之前不 author B RTL。

本 r2 只给 fresh hammer 审阅，并给未来 pre-RTL CPU contract 的必要字段；它不创建该 contract，
不授权任何运行、RTL、VCS、DC、PTPX、Formality、训练、远端或论文 headline。

## 2. 候选 A 的可执行最小边界

### 2.1 冻结输入与原子接收

输入沿用 M523 的 8-lane exact tap bundle。每 lane 至少携带
`{tag24,time4,source_channel12,kernel_index4,destination_y10,destination_x10,phase2,
event_boundary,stream_last}`；未来 adapter 还必须生成并验证
`flattened_source_key = source_channel * 9 + kernel_index`。

对一个 interior source event 的 K3/S2/P1/OP1 phase-major 序列，首个 full-8 原子 bundle 冻结为
`4 odd/odd + 2 odd/even + 2 even/odd`，所以该 event-local bundle 每 phase distinct destination 峰值
为 4。M523 还允许 equal-tag/time cross-event tail packing；r2 不假设任意 cross-event bundle 仍满足
每 phase `<=4`，其接口级安全上界是八条 lane 全落入一个 phase。收费的 8-lane ingress 先原子保存
完整 bundle，再以每 phase 最多四个 resident context 分批排空，因此不依赖未证明的 source 顺序。

PBR4 的唯一合法 accept 规则为：

1. 8-entry ingress epoch 为空、sticky fault 未置位；
2. 全部 valid lane 的 tag/time 一致，lane identity 与 M523 fence 合法；
3. 全部 valid lane 可在同一 edge 写入 8x128-bit ingress；
4. 条件不满足时 `bundle_accept=0`，整个 bundle payload 在 backpressure 下稳定；非法 payload 触发
   fail-closed fault，不能“接收合法子集”。

context 组织为四个 phase bank、每 bank 四个 `256-bit 1R1W/L1` entry。单 entry 保存 exact
destination identity、最多八个 `{flat_source_key16,sign1}` slot、valid/bank mask、lane/event fence 与
partial-state 标志。entry 宽度不足必须在 future source hammer 时 fail，不能另加未收费 side state。

accept 后，调度器按稳定 lane 顺序把 ingress lane 搬入匹配或空闲 context；每 phase 同时最多四个
distinct destination。若 cross-event bundle 在某 phase 超过四个 destination，余下 lane 继续在 ingress
保持 valid，等已有 context 做完收费的 partial RMW 并释放后再搬入。PBR4 采用**单 active bundle
epoch**；ingress 与所有 context 都排空后才允许下一 bundle。这一有限状态机对任意 8-lane phase 分布
均有容量，不靠 early close，也不需要第十七个 hidden context。它刻意不长期保留跨 bundle join
context；context drain 不生成 final output。final commit 只由冻结的 canonical source-row/channel/
timestep frontier 或 tag/time/stream fence触发。

### 2.2 progress、backpressure 与 spill

每个 context 内按 `bank = flat_source_key mod 8` 形成 source round；同 bank 的 contributor 顺序串行，
不同 bank 在同一 round 最多各一个。每个 round 必须完整执行六个 16-lane slice。group FIFO 满、O8
满、weight bank refill、psum miss/evict/restore、1RW port conflict、sink backpressure 都只能阻止整个
bundle 的下一状态，不能丢 lane 或提前 final commit。

psum cache 是 exact partial-state backing；cache eviction 不是 completion。dirty line eviction 将完整
六个 Acc24 slice 写回 external psum backing，后来 restore 后继续累加。只有 canonical frontier 证明该
destination 后续不再有 contributor 时，才把完整 96-lane Acc24 vector commit 到 output stream，并
invalidate cache line。未来 exhaustive reference 至少要覆盖：

- interior 9-tap event 的首个 full-8 bundle，四个 odd/odd distinct destination 同拍接收；
- 每 phase distinct destination 为 0/1/2/3/4；
- 合法 cross-event bundle 的单 phase distinct destination `>4`，完整8-lane ingress accept 后分批排空；
- 同 bank contributor 多 round；
- join bank/psum bank/weight bank backpressure；
- dirty eviction、restore、RAW、same-edge retire/replace；
- frontier 前 context drain 但不得 output commit；
- frontier 后 partial/full group 的 exact single commit；
- tag/time/stream boundary 与 sticky-fault accepted-work drain；
- 穷举的小尺寸 K3/S2 scatter 中无 deadlock、无 early close、无 duplicate/drop。

## 3. 同资源最强 exact baseline

未来一次 CPU fast-kill 必须在一个 executable model 内至少产生以下四点：

| 点 | exact 调度 | 同资源约束 |
|---|---|---|
| `A1-SC8` | source-centric K8；按冻结 source/bundle 顺序发 group，不做 destination-keyed lookahead | 与 PBR4 完全相同的 lane、join、weight/psum SRAM、L4/O8/FIFO4、external link 与 commit |
| `A1-ISO8` | 只合并 current head/相邻 exact destination；同 bank 顺序延期 | 同上；不得成为唯一 baseline |
| `A1-OSG` | 常规 output-stationary / mini-batch Gustavson；在同一个 16-context window 内按 exact destination 聚合并依赖 frontier 输出 | 同上；不得多给 context、port、lookahead 或 bandwidth |
| `PBR4` | K3/S2 parity frontier 驱动的单原子-bundle bounded rendezvous 与 exact partial-RMW backing | 同上；不得有 candidate-only SRAM、bypass 或 free completion state |

`A0 dense polyphase` 可单独展示 binary activation-zero opportunity，但不进入我方创新倍率。strongest
baseline 的选择规则禁止 per-sample oracle：在同一 S10 cohort 上，先选 ratio-of-sums total cycle 最小的
**一个固定 architecture point** `A1-STRONG`，随后所有 sample、traffic、energy 与 PBR4 比较均使用
该同一个 point。论文只许写
`speedup = sum(A1-STRONG cycles) / sum(PBR4 cycles)`。

若 `A1-OSG` 与 PBR4 的 group/RMW/commit 序列等价，PBR4 的性能 novelty 直接 KILL；最多保留 M523
decoder completeness。不得排除 OSG 以制造 1.30x。

## 4. 冻结的 M218 service 与 240 KiB memory coordinate

四个 exact 点共享下列唯一 service；任何字段变化都形成新 identity，不能混表。

### 4.1 计算与队列

- 96 output lane 固定拆成 **6 个 16-lane slice request**；一组不是一拍 96 product；
- request order 为 group、output block、slice `0..5`；总 request issue width 为 1；
- 8 个 weight bank，每个 active bank 每 slice 返回一个 128-bit/16xINT8 word；最大响应 1024 bit；
- weight/psum response latency 均为 `L4`；outstanding scoreboard 为 `O8`；group FIFO 为 `FIFO4`；
- response 保留 tag/epoch/generation/slot/context identity；future RTL 允许 OOO response，但 CPU primary
  point使用 deterministic L4 response 并仍执行 O8/FIFO4/context hazard；
- Acc24 饱和/舍入和最终 commit 顺序与 frozen integer reference bit exact；同一个
  `(phase,cache_line,slice)` 在前一 write retire 前不得重发；
- psum read、weight read、Acc24 reduction、psum write、result commit 均是独立收费 transaction。

### 4.2 weight SRAM

- 8 bank，每 bank `128 x 128 bit`、single-port `1RW`、read latency L4；总物理容量 `16,384 B`；
- resident tile 固定为 `CinTile=16, CoutBlock=96, K=3x3`；每 bank 使用
  `2 source/bank x 9 kernel x 6 slice = 108` rows，其余 20 rows invalid；
- `bank = (local_source_channel*9 + kernel_index) mod 8`，
  `row = floor(local_flat_key/8)*6 + slice`；stored identity 还必须绑定
  `{layer,cout_block,cin_tile,flat_key,slice}`；
- tail Cin tile 仍按 padded Cin16 refill，四点完全相同；
- external refill link 固定 `128 B/cycle`、每 burst first-beat latency 32 cycle；八 bank 同拍各写一个
  128-bit word，完整 tile 为 108 beats，因此 refill barrier 为 `32+108=140 cycle`、payload
  `13,824 B`；
- 1RW bank 上 compute read 与 refill write 不重叠；tile order 不同导致的 refill 次数必须各自实收。

### 4.3 psum SRAM、tag、RMW 与 spill

- 4 个 phase data bank，每 bank `1024 x 384 bit`、single-port `1RW`、L4；总物理容量
  `196,608 B`；一个 384-bit word 是 16xAcc24；
- 每 phase 逻辑容纳 128 个 destination vector，每 vector 六个 slice，使用 768/1024 rows；
- tag/frontier array 为每 phase `128 x 128 bit 1R1W/L1`，四 bank 总 `8,192 B`，字段至少绑定
  `{valid,dirty,tag,time,destination_y,destination_x,output_block,canonical source-row/channel frontier,
  final-fence and fault epoch}`；禁止另设未收费 completion directory；
- 每 phase 仅一个 384-bit charged RAW bypass，共 `192 B`；不允许未收费 accumulator shadow；
- 每 source round 对每个 destination 的六个 slice 都执行显式 psum read-modify-write。1RW 仲裁器对 read
  与 pending write 二选一；CPU 必须逐 cycle 模拟冲突，不能用 `6 requests/group` 直接替代周期；
- miss 使用 deterministic round-robin victim。dirty eviction：六个 data read 的 L4 drain 共 10 cycle，
  再传一个 padded `384 B` burst（32+3 cycle），共 45 cycle；
- restore：padded 384 B burst 35 cycle，再六个 1RW write，共 41 cycle；从未存在的 destination
  zero-fill 六个 write；
- final commit：六个 data read L4 drain 10 cycle，加 padded 384 B output burst 35 cycle，共 45 cycle；
- weight refill、psum spill/restore/commit 共用同一个 128 B/cycle external link并串行仲裁；所有 padded
  byte、burst、stall 与读写能量分别记账。

### 4.4 join、descriptor 与总容量

| 状态 | 容量 |
|---|---:|
| 8x `128x128b 1RW` weight | 16,384 B |
| 4x `1024x384b 1RW` psum | 196,608 B |
| 4x4x256b join RF | 512 B |
| 4x128x128b psum tag/frontier RF | 8,192 B |
| 4x384b RAW bypass | 192 B |
| 8x128b atomic ingress | 128 B |
| 18x128b descriptor FIFO | 288 B |
| O8 scoreboard + FIFO4 commands + response/result skid | 432 B |
| **总计** | **222,736 B** |
| 240 KiB 余量 | **23,024 B** |

O8 scoreboard 固定为 8x128b，FIFO4 command 为 4x256b，weight-response skid 为 1024b，result skid
为 384b。四点均固定并收费这份容量；不能让 baseline 少 state、candidate 多 state，也不能把
23,024 B 余量默认为
免费 scheduler。future CPU receipt 还必须逐项给 occupancy/high-water、read/write/evict/restore/refill/
commit 次数。future paired DC 只综合 logic/RF shell 时必须继续把所有 memory 标成 external macro boundary，
不得称 0-macro paper PPA。

## 5. 同 cohort 与 future CPU 门

M510 的 `21.57--22.83%` 来自 S100 aggregate sensitivity；M511 是 S10 decoder payload。二者不得混成
一个 `decoder exact share`。因此 r2 从 local CPU fast-kill **删除 share gate**。

任何系统/Amdahl 行只有两种合法来源：

1. 同一 M511 S10 sample ID、同一 H67 ep35 checkpoint、同一 T10、同一 exact manifest 的 decoder
   numerator 和 included-scope denominator；或
2. 另一份逐 sample 完全相等并有双封 identity 的 exact manifest。

S10/S100、T10/aggregate 或不同 sample manifest 一律只能分列为 analysis sensitivity。没有 same-cohort
denominator 时，PBR4 只报 local decoder ratio。

fresh hammer P0/P1=0 后，才允许另行 author 一个仍为 `run_authorized=false` 的 CPU contract。未来
CPU gate 与 physical gate严格分层：

### CPU functional/resource gate

- M511 payload verifier 与 M513 production result 均先有独立 member+outer seal；
- destination contributor multiset、flattened/stored weight identity、Acc24 final state、commit count、
  frontier、spill/restore 全部 0 mismatch；
- 四点使用第 4 节完全相同 service/memory coordinate；
- PBR4 相对固定 `A1-STRONG` 的四层 ratio-of-sums `>=1.30x`；每个 S10 sample `>=1.10x`；
- gate 只在修复后的 strongest exact baseline 上生效；不得引用 A1-ISO8 单点；
- PBR4 weight active reads、weight refill bytes 不得比 A1-STRONG 增加；全部 psum/data traffic 实报；
- CPU 只报告 exact state bits、比较器数、transaction/cycle，不给 mapped area 百分比。

若 ratio-of-sums `<1.30x`，不 author performance RTL；只有 measured psum read+write 或 modeled dynamic
energy 相对 A1-STRONG `>=30%` 时可保留为 decoder support/energy ablation。任何 sample `<1.0x`、功能
mismatch、额外 port/bandwidth/state、或与 A1-OSG 等价均 KILL。

### 后续独立 physical gate（CPU GO 之后才存在）

`CPU GO -> source-only RTL author admission -> fresh static hammer -> Synopsys VCS/SVA -> paired 3 ns DC`。
paired DC 才评价：PBR4 added mapped area `<=20% A1-STRONG`、Fmax loss `<=10%`、五类约束、setup/hold、
loop/TIM-209/OPT-150。再后续才允许 Formality、mapped-gate VCS SAIF/PTPX 与 macro-inclusive energy。
CPU 分析不得预判该 area/Fmax gate。

本 r2 不授权上述任何一步。

## 6. ELSA 与其他 prior 的窄 claim

ELSA 已经覆盖 bundled AER、row-aligned mini-batch Gustavson、同一 membrane row 单次 read/write 和
dependency-aware output scheduling；Prosperity/Phi/FireFly-T 也分别覆盖 product reuse、pattern/residual
sparsity和 multi-NZ/bank-aware dispatch。PBR4 不得声称发明 destination rendezvous、Gustavson、
single-RMW、dependency completion、polyphase 或多 lane decode。

只有在修复后的 A1-OSG 强基线上仍过门，才可主张以下窄对象差：

> an exact K3/S2 ConvTranspose parity-dependent contributor frontier, implemented under a bounded
> four-context-per-phase atomic-bundle interface and the typed signed-source L4/O8/FIFO4 service.

该 claim 的新意是 H67 ConvTranspose 的 parity frontier、原子 bundle 容量与 typed signed-source 资源
边界；不是通用 sparse row accumulation。若强 OSG 吃掉差值，论文只把 M523/PBR4 写成 decoder
completeness/support。

## 7. 候选 B 的 exact-zero 修复

B 继续继承 r2 typed-ticket spec，M519 K1/K8/K1x8 三轴 canonical/独立 receipt 仍是硬前置。paired
tagged/elided 两点必须满足以下**整数全等**：

- total cycle delta = 0；
- accepted transaction、request、response、active-bank read、weight/data byte、Acc24 update、result beat、
  done-cycle delta = 0；
- bank/service/fault/backpressure trace 逐 cycle 0 mismatch；
- 只有明确列出的 tag metadata bit movement、tag flop/compare/mux 可以改变。

不存在 `1%` 容差。任何非零 cycle/accept/data-bank traffic delta 都是 P0，不能用 area/power收益覆盖。

clean paired DC 后，无论 area 是否达到 promote 门，都必须做 matched mapped-gate SAIF/PTPX：exact
net annotation=`100%`、exact leaf annotation=`100%`、expected active cone nonzero-toggle coverage、相同
contiguous start/end cycle、相同 macro boundary；dynamic power 定义严格为 `internal + switching`，
leakage 单列。B 仍只进入 C2 physical/protocol ablation，不提供 cycle/system speedup。

## 8. r1 hammer finding closure

| finding | r2 修复 |
|---|---|
| P0-01 两 context/phase 无法接收原子 bundle | 单 bundle epoch、8-lane atomic ingress、4 context/phase 分批排空；interior 首包4峰值冻结；partial-RMW 不等于 close |
| P1-01 baseline 不强 | 增加 A1-SC8、A1-ISO8、A1-OSG；固定一个 strongest exact point；ours 只对它报 |
| P1-02 M218 service 缺失 | 冻结 6x16、issue1、L4/O8/FIFO4、Acc24/context hazard |
| P1-03 SRAM/physical tax 缺失 | 冻结 222,736 B、1RW/L4、refill、RMW、tag/frontier、O8/FIFO4/skid、spill/restore/commit |
| P1-04 S10/S100 share 混用 | 删除 local CPU share gate；系统行只准 same-cohort S10 或等 exact manifest |
| P1-05 ELSA 邻接不足 | claim 收窄到 K3/S2 parity frontier + bounded atomic bundle + typed source |
| P1-06 CPU/DC 门混合 | CPU cycle/traffic/state 与 post-VCS paired-DC area/Fmax 分层 |
| P1-07 B 允许 1% cycle | 改为 cycle/accept/data-bank traffic exact integer zero |
| P2 PTPX coverage | 冻结 net/leaf 100%、nonzero toggle、same window/macro boundary、dynamic定义 |

## 9. 身份与授权边界

- r1 README SHA：`df46183bf80e5cdd803c60e813db5ef1d5ba8f0d0f019485f599c1f918b72fb6`；
- r1 JSON SHA：`68a46bac57c710781931384a7d62266237bf13a683543013ccdb92c2f5b7e321`；
- r1 hammer Markdown SHA：`c87dc9ca8b007430100502c32e9c3e8563ca99c067351644d85d4ab398a16a65`；
- r1 hammer JSON SHA：`5d13fa22871c36aa2f503c70f1f2d4e7c2d824f342196e4dd8d30cb00ca76d36`；
- `docs/359` SHA：`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`，未修改。

本 r2 只是一份 screen repair。它没有生成 cycle、area、power、energy、accuracy、system speedup 或
paper-ready PPA；没有 author execution contract；没有实现 RTL；没有运行 CPU、Synopsys 或远端任务。
唯一允许的下一步是 fresh independent r2 hammer。只有其 P0/P1=0，才可讨论 future pre-RTL CPU
contract authoring；该 future contract 仍需单独授权运行。
