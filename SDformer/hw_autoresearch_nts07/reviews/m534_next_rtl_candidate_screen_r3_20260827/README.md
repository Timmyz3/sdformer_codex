# M534｜下一条 RTL 候选筛选 r3：decoder 数值身份与 persistent backing 闭账

日期：2026-08-27  
模式：screen repair authoring only；零 CPU/VCS/DC/PT/PTPX/Formality/训练/远端运行；零 RTL 修改  
状态：`R3_SCREEN_REPAIRED_FOR_FRESH_HAMMER__NO_PRE_RTL_CONTRACT__NO_RUN__NO_RTL`

## 1. 裁决与授权

r3 只修复 r2 fresh hammer 的两个 P1，不改变 PBR4 的算法、强 baseline、门槛、cohort、prior-art
边界或候选 B 的 exact-zero 定义。修复后的候选 A 仍只是一个**未来 pre-RTL CPU contract 的候选**：

- M511 的 ConvTranspose 输入冻结为 exact binary `{0,1}`；一个被 M523 发出的 source event 的数值严格
  是隐式 `+1`。`source_sign=0` 表示 `+1`，不得解释为 weight sign 或 product sign；signed INT8
  product 的正负只来自被 weight identity 选中的 INT8 weight；
- M523 每 lane 的全部 payload、cross-event event identity、canonical dense-source frontier、layer、output
  block 与 generation 均在收费的 `8x128-bit` ingress 中逐字段落位，不再允许 analyzer 自选字段；
- `16x256-bit` context 逐字段落位，八个 contributor slot 各自保存完整
  `{source_channel12,kernel_index4,source_sign1}`，因此合法 cross-event bundle 即使跨 Cin16 tile 也不依赖
  hidden active-tile state；
- 新增 macro-rounded `1024x128-bit` persistent backing-valid directory，完整覆盖四层当前
  `{tag,time}` epoch 的 110,400 个 output-block destination。resident miss 的 zero-fill/restore、dirty
  writeback、evicted restore、final commit、directory set/clear 与 epoch clear 均有确定状态和收费事务；
- weight resident tile tag/valid/refill、global frontier ownership、external-link command、directory RMW、
  psum pending write、384-B spill/restore/commit vector buffer、commit/fault state 全部进入 mandatory ledger。
  macro physical capacity 与逐项向上取整的 soft-state byte-equivalent 合计 `239,636 B`，低于
  `240 KiB=245,760 B`，余 `6,124 B`（2.491%）。

本目录只允许 fresh independent r3 screen hammer。只有 r3 hammer 得到 `P0/P1=0`，才允许另行 author
仍为 `run_authorized=false` 的 pre-RTL CPU contract。CPU run、RTL、VCS、DC、PTPX、Formality、训练、远端、
performance/energy/PPA/system/headline 全部为 false。

## 2. exact decoder 数值、source 与 event identity

### 2.1 M511 → M523 数值合同

四层冻结 shape 为：

| layer | Cin/Cout | Hin×Win | Hout×Wout | Cout96 blocks | C-order source population |
|---:|---:|---:|---:|---:|---:|
| 0 | 1536/384 | 15×20 | 30×40 | 4 | 460,800 |
| 1 | 770/192 | 30×40 | 60×80 | 2 | 924,000 |
| 2 | 386/96 | 60×80 | 120×160 | 1 | 1,852,800 |
| 3 | 194/96 | 120×160 | 240×320 | 1 | 3,724,800 |

M511 bitpack 是 `T,B,C,H,W` 的 C-order、little-bit-first exact binary plane，bias 必须为 null。每个
`1` 生成一个 source event，每个 `0` 不生成 event；禁止 threshold、ternary re-interpretation 或 amplitude
payload。source identity 定义为：

```text
source_linear_ordinal = ((source_channel * input_height) + source_y) * input_width + source_x
event_identity = {layer2, tag24, time4, source_linear_ordinal22}
```

四层最大 ordinal 为 `3,724,799 < 2^22`。producer/scanner 必须按该 ordinal 严格递增生成 event 和独立的
frontier token。M523 的 `tap_event_last` 是该 source event 的最后一个 legal tap；cross-event bundle 中相邻
event 由不同 ordinal 区分。`stream_last` 只允许在 `tap_event_last=1` 的 lane 上置位。M523 的 exact 几何
关系仍为：

```text
kernel_index = 3 * kernel_y + kernel_x
destination_y = 2 * source_y + kernel_y - 1
destination_x = 2 * source_x + kernel_x - 1
phase = {destination_y[0], destination_x[0]}
```

所有坐标均先按 M523 的 top/left legality mask 过滤；future reference 必须逐 lane 重算上述五个恒等式。

### 2.2 8x128-bit atomic ingress packing

每个 ingress entry 恰为 128 bit；未列字段不存在，reserved bit 不存在：

| bit | 字段 | 语义 |
|---|---|---|
| `[0]` | valid | entry ownership |
| `[1]` | source_sign | 固定 `0=+1`；`1` 为 malformed/fault |
| `[2]` | event_boundary | exact M523 `tap_event_last` |
| `[3]` | stream_last | exact `bundle_stream_last && tap_event_last` |
| `[5:4]` | phase | `{destination_y[0],destination_x[0]}` |
| `[9:6]` | kernel_index | `3*ky+kx`, 0..8 |
| `[11:10]` | kernel_y | 0..2 |
| `[13:12]` | kernel_x | 0..2 |
| `[23:14]` | destination_y | 10-bit unsigned |
| `[33:24]` | destination_x | 10-bit unsigned |
| `[43:34]` | source_y | 10-bit unsigned |
| `[53:44]` | source_x | 10-bit unsigned |
| `[65:54]` | source_channel | global 12-bit channel |
| `[69:66]` | time | exact M523 time |
| `[93:70]` | tag | exact M523 tag |
| `[95:94]` | layer | M511 module index 0..3 |
| `[97:96]` | output_block | Cout96 block 0..3；同 descriptor 按 block 重放 |
| `[119:98]` | source_linear_ordinal | 22-bit C-order identity，必须由 source tuple 重算相等 |
| `[127:120]` | epoch_generation | 当前 `{tag,time}` generation；禁止 stale response credit |

M523 原接口没有 source amplitude、sign、layer、output block、ordinal 或 generation。adapter 只能按上述
冻结规则生成它们：binary event 的 sign 常量为零；layer/output block 来自冻结 service loop；ordinal 由
source tuple 和该 layer 的 Hin/Win 重算；generation 来自收费的 global epoch register。adapter 不能省略
M523 的 source coordinates 后再假设可逆重建，也不能用 Python object/event counter 代替 entry 字段。

### 2.3 16x256-bit context packing

四个 phase bank、每 bank四个 context；每 entry 恰为 256 bit：

| bit | 字段 |
|---|---|
| `[0]` | valid |
| `[2:1]` | phase |
| `[4:3]` | layer |
| `[6:5]` | output_block |
| `[16:7]` | destination_y |
| `[26:17]` | destination_x |
| `[50:27]` | tag |
| `[54:51]` | time |
| `[62:55]` | slot_valid[7:0] |
| `[79:63]` | contributor slot0 |
| `[96:80]` | contributor slot1 |
| `[113:97]` | contributor slot2 |
| `[130:114]` | contributor slot3 |
| `[147:131]` | contributor slot4 |
| `[164:148]` | contributor slot5 |
| `[181:165]` | contributor slot6 |
| `[198:182]` | contributor slot7 |
| `[220:199]` | last_possible_source_ordinal22；按 K3/S2 几何和 full Cin 计算 |
| `[228:221]` | epoch_generation8 |
| `[236:229]` | source-bank mask8 |
| `[240:237]` | source-round index4 |
| `[246:241]` | slice-pending mask6 |
| `[249:247]` | context FSM state3 |
| `[253:250]` | contributor count4，0..8 |
| `[254]` | frontier-close-seen |
| `[255]` | stream-last-seen |

每个 slot 以其最低 bit 为 `base`，`[base+11:base]=source_channel12`、
`[base+15:base+12]=kernel_index4`、`[base+16]=source_sign1`。contributor slot 的
`source_channel` 是 global channel，不是 local Cin16 index，所以一个 cross-event bundle
可以横跨 Cin tile。每次 weight read 的 exact identity 为：

```text
cin_tile = floor(source_channel / 16)
local_channel = source_channel mod 16
flat_source_key = source_channel * 9 + kernel_index
bank = flat_source_key mod 8
weight_identity = {layer, output_block, cin_tile, local_channel, kernel_index, slice}
```

`source_sign` 仍必须为零；product sign 来自 signed INT8 weight。destination 与 kernel 可反算 source y/x，
并须与 ingress 中保存的 source tuple/ordinal一致。context 不允许保存 Acc24 shadow；它只保存待执行 contributor
identity。partial RMW 完成后才可释放相应 slot，context drain 仍不等于 semantic close。

## 3. canonical frontier 与 exact close

frontier 不由“最后一个 nonzero event”猜测。M511 dense bitplane scanner 另行发出单调
`frontier_ordinal22`，语义为该 layer/time 中 `<=frontier` 的每个 `{C,Y,X}` bit 已被确认为 zero 或已经作为
event 原子接收。global frontier owner 只在下列条件接受新 token：tag/time/layer/generation exact match，且
ordinal 不回退。

对 destination `(dy,dx)`，legal source 坐标集合由 K3/S2/P1 几何显式枚举；
`last_possible_source_ordinal` 是所有 channel 和 legal source coordinate 的最大 C-order ordinal。只有：

```text
global_frontier >= last_possible_source_ordinal
AND no matching ingress/context/O8/pending-write/external transaction exists
```

时该 destination 才能 final commit。`stream_last` 只能把 frontier 推到该 layer 的最后一个 dense source，
不能跳过 accepted work。epoch 切换前必须满足 ingress/context/scoreboard/pending write/external link 全空、
所有 resident/backed line 已 final commit、directory 无 valid bit；否则 fail closed。

## 4. persistent external psum backing

### 4.1 directory 容量与地址

新增一个 `1024x128-bit 1RW/L1` on-chip backing-valid directory，物理容量 `16,384 B`。它在当前
`{tag,time,generation}` epoch 内覆盖四层完整 output-block destination 空间：

| layer | base bit | extent bit | index |
|---:|---:|---:|---|
| 0 | 0 | 4,800 | `0 + ((ob*30+y)*40+x)` |
| 1 | 4,800 | 9,600 | `4800 + ((ob*60+y)*80+x)` |
| 2 | 14,400 | 19,200 | `14400 + ((ob*120+y)*160+x)` |
| 3 | 33,600 | 76,800 | `33600 + ((ob*240+y)*320+x)` |

总计 `110,400 bit`，其余 `20,672 bit` 永远 invalid/zero。directory bit 的唯一语义是：该 destination
在当前 epoch 的 external scratch backing 中存在一份完整、最近一次成功写回的 96xAcc24 vector；若同一
destination 当前 resident 且 dirty，则 resident 明确拥有更新版本，lookup 必须先命中 resident。external vector 使用
`384 B` 对齐槽（payload 288 B），地址为：

```text
external_address = PSUM_SCRATCH_BASE + directory_index * 384
```

external scratch 自身是所有四点共享、逐 transaction 收费的 off-chip window，不计入 240 KiB on-chip
budget；on-chip directory、command、skid、valid/tag/address ownership全部计入。epoch 不允许重叠，故 external
address 不需隐藏 tag RAM；tag/time/generation 由 global owner 和每个 command 同时绑定。

### 4.2 zero-fill、writeback、restore 与 commit

resident tag hit 优先于 directory。miss 时先收费 directory read：

- bit=`0`：该 destination 从未在 external backing 存在；执行六次 charged psum zero write，然后接收
  contributor。禁止 free calloc 或未收费 Python set；
- bit=`1`：进入 restore，收费 32-cycle first beat、3 个 128-B beat、六个 1RW psum write；恢复完成前
  不允许 contributor update；
- dirty victim：先六个 psum data read/L4 drain，再发 padded 384-B write。只有最终 write response 成功且
  generation/tag/address exact match 后，directory 才通过 charged read-modify-write 置一；victim 在此之前
  不得复用；
- final resident commit：六个 read/L4 drain + padded 384-B output。若 directory bit 原为一，成功 commit
  后 charged RMW 清零；若为零无需伪写；
- final evicted commit：output cursor 查到 bit=1 时必须 restore 后 commit，再清零。bit=0 且无 resident
  line 表示 exact all-zero vector，以下游已冻结 dense-shape zero token表达，不生成 data beat；四个架构点完全
  相同；
- epoch clear：新 `{tag,time,generation}` 开始前对 1024 directory word 做 1024 次 charged zero write。
  clear 不能与 query/update 共用同一 1RW edge；1024-cycle tax 由四点共同实收。

directory 单 bit set/clear 使用收费的 word RMW：L1 read 到 160-bit directory bypass，下一可用 write edge
写回 128-bit word。same-word RAW 必须由该 bypass 按 address/generation 精确 forward；不能把 1RW 当 1R1W。

### 4.3 mandatory writeback/restore state machine

唯一 shared external-link FSM 必须具有以下可观察状态；未来 CPU reference 和 RTL 不得合并成零周期调用：

`IDLE -> DIR_QUERY -> {ZERO_FILL,RESTORE_CMD}`；
`VICTIM_SELECT -> EVICT_READ_ISSUE -> EVICT_READ_DRAIN -> WB_FIRST_LAT -> WB_BEAT0..2 -> WB_ACK -> DIR_SET_READ -> DIR_SET_WRITE`；
`RESTORE_FIRST_LAT -> RESTORE_BEAT0..2 -> RESTORE_PSUM_WRITE0..5 -> RESTORE_DONE`；
`FINAL_READ_ISSUE -> FINAL_READ_DRAIN -> COMMIT_FIRST_LAT -> COMMIT_BEAT0..2 -> {DIR_CLEAR_READ,DIR_CLEAR_WRITE} -> DONE`；
以及 `FAULT_DRAIN`。

weight refill、psum spill/restore、result commit 共用同一个 `128 B/cycle` link，只能有一个 active command。
arbiter 使用固定 `final-commit > dirty-writeback > restore > weight-refill` 优先级；被阻塞请求保持 payload。
response 必须匹配 `{opcode,layer,output_block,destination,tag,time,directory_index,generation,beat}`；错误 response
置 sticky fault，accepted work 进入 drain，不得 credit 其他 epoch。

spill 的六个 48-B slice、restore 的三个 128-B beat以及 final commit 的六个 slice 都使用同一收费的
`3072-bit=384-B` vector buffer。它具有 exact command owner/generation，只有 buffer 空且前一 command 已
retire 才可复用。这样 r2 的“先六读 drain、再三 beat”不依赖 hidden 384-B staging，也不假设 external
response 可被任意 backpressure。

## 5. closed mandatory state ledger

### 5.1 macro-rounded SRAM

| state | 组织 | physical byte | owner/port |
|---|---|---:|---|
| weight data | `8 x 128x128b` | 16,384 | shared；每 bank 1RW/L4 |
| psum data | `4 x 1024x384b` | 196,608 | phase bank；1RW/L4 |
| resident tag/frontier | `4 x 128x128b` | 8,192 | phase bank；1R1W/L1 |
| persistent backing-valid directory | `1 x 1024x128b` | 16,384 | global；1RW/L1 |
| **macro subtotal** |  | **237,568** |  |

resident tag/frontier 的 128 bit 固定为：`valid1,dirty1,layer2,output_block2,phase2,destination20,
tag24,time4,last_possible_ordinal22,fault_epoch8,generation8,reserved34`。reserved 必须写零、读时检查零；
不得转作未收费 completion directory。

### 5.2 soft state（byte-equivalent 也进入 240 KiB）

| state | exact/rounded organization | charged byte |
|---|---|---:|
| 16 join contexts | `16x256b` | 512 |
| atomic ingress | `8x128b` | 128 |
| inherited descriptor FIFO | `18x128b` | 288 |
| M218 O8/FIFO4/shared-response/result skids | `8x128b + 4x256b + 1024b + 384b` | 432 |
| four psum pending-write/RAW entries | each `valid1,row10,generation8,data384,reserved13 = 416b` | 208 |
| directory RMW bypass | `valid1,op2,row10,word128,generation8,reserved11 = 160b` | 20 |
| shared spill/restore/commit vector buffer | `3072b` | 384 |
| shared external command | 128b | 16 |
| global frontier owner | 128b | 16 |
| frontier input skid | 128b | 16 |
| weight resident/refill control | 64b | 8 |
| external-link arbiter | 64b | 8 |
| directory clear/query/update control | 64b | 8 |
| sticky fault/epoch owner | 64b | 8 |
| output cursor/final-commit control | 128b | 16 |
| **soft-state subtotal** |  | **2,068** |

`weight resident/refill control64` 固定包含
`tile_valid1,refill_active1,layer2,output_block2,cin_tile7,refill_beat7,bank_fill_valid8,
generation8,shared_link_wait1,outstanding_count4,reserved23`。tile 只有 108 个 refill beat 全部完成后才能置
valid；row identity 由该 tile tag、bank、row、slice 重算。禁止用“SRAM 内已有数据”代替 valid/tag。

`global frontier owner128` 固定包含
`valid1,layer2,tag24,time4,input_height10,input_width10,input_channels12,frontier_ordinal22,
output_cursor17,generation8,stream_last_seen1,fault_epoch8,clear_pending1,reserved8`。

`shared external command128` 固定包含
`valid1,opcode3,layer2,output_block2,destination20,tag24,time4,directory_index17,slice3,
beats_remaining2,first_latency_counter6,generation8,direction1,reserved35`。

其余 64/128-bit controller word 的 reserved bit 同样固定为零；future source hammer 必须要求逐字段声明且
总 bit 数不增。任何新增 queue、directory、shadow accumulator、lookahead、prefetch tag 或 response payload
必须重新 macro-round/byte-round，不能吃“余量”而不形成新 screen identity。

### 5.3 240 KiB closure

```text
macro subtotal                         237,568 B
soft-state byte-equivalent               2,068 B
-----------------------------------------------
macro-rounded/accounted total          239,636 B
240 KiB budget                         245,760 B
headroom                                 6,124 B = 2.491%
```

该 headroom 不是授权的 free state。未来 contract/source 若任何 mandatory item 增长，使 total
`>245,760 B`，必须 fail closed 或收窄结构；禁止减小现有 macro 的 physical capacity、把 directory 放到
off-chip、或把 standard-cell state 从 budget 表中删除来过门。soft state 在 paired DC 中还必须按 mapped
cell area/power再次收费；这里的 byte-equivalent 只是同一 memory-coordinate 的容量保守账，不替代 DC。

## 6. 保持不变的公平性、门槛与 prior-art 边界

未来一次 CPU fast-kill 仍须在同一个 exact S10、同一个 M218 `6x16/L4/O8/FIFO4/Acc24` service、同一个
`239,636 B <=240 KiB` coordinate、同一个 external link 和同一个 backing FSM 中同时测：

- `A1-SC8`：冻结 source/bundle 顺序的 source-centric exact K8；
- `A1-ISO8`：仅 current-head/adjacent exact destination merge；
- `A1-OSG`：同 16-context window 的常规 output-stationary mini-batch Gustavson；
- `PBR4`：K3/S2 parity-frontier、single atomic bundle bounded rendezvous。

strongest baseline 仍是在完整 S10 上 ratio-of-sums cycles 最小的一个**固定 architecture point**
`A1-STRONG`；禁止 per-sample oracle。PBR4 只有同时满足以下条件才可继续：

1. function mismatch=`0`；
2. `sum(A1-STRONG cycles)/sum(PBR4 cycles) >= 1.30x`；
3. 每个 sample `>=1.10x`；
4. weight active reads/refill bytes 不增加；无额外 port/bandwidth/state；
5. group/RMW/commit sequence 不与 A1-OSG 等价。

未过 1.30x 时，只有 psum read+write 或 modeled dynamic energy 降低 `>=30%` 才能降级为 decoder
support/energy ablation。S10/S100 不得混用；system share 仍要求同 checkpoint、T10 sample IDs 与 exact
manifest 的 decoder numerator/included-scope denominator。

ELSA 已覆盖 bundled AER、row-aligned mini-batch Gustavson、single membrane-row RMW 与 dependency-aware
completion。这里只允许在强 A1-OSG 后主张：`exact K3/S2 ConvTranspose parity-dependent contributor
frontier under a bounded four-context-per-phase atomic-bundle interface and the typed signed-source
L4/O8/FIFO4 service`。不得声称发明 rendezvous、Gustavson、single-RMW、dependency completion、polyphase
或 multi-lane decode。

候选 B 保持 r2 exact-zero 边界：等待 independently sealed M519 K1/K8/K1x8 canonical；tagged/elided 的
cycles、accept、request/response、active-bank read、data byte、Acc24 update、result beat、done-cycle delta
必须全为整数零。它不产生 cycle/system speedup。

## 7. r2 hammer 两项 P1 closure

| finding | r3 closure |
|---|---|
| P1-01 binary source 数值/sign/event/frontier/packing 未冻结 | binary source=`+1`、sign 与 weight sign 分离；C-order ordinal 与 dense frontier定义；128-bit ingress 与 256-bit context 全字段闭合，跨 Cin tile 不靠 hidden state |
| P1-02 backing 存在性与 mandatory state 未闭账 | 16-KiB persistent valid directory、deterministic external address、zero-fill/restore/writeback/commit FSM、384-B vector staging、weight/global frontier/link/directory/control逐项收费；total 239,636 B，余6,124 B |

r2 hammer 的两个 P2 保留为 future contract 要求：输出 group/RMW/commit sequence hash；S10 只能是
single-sequence fast-kill，不能冒充 multi-sequence robustness、full-network 或 system speedup。

## 8. 身份与红线

冻结输入：

- r2 README SHA256：`fb2eeb6346e4a61b26d6a4f062e1b062fcb1e9a5f7f5e09b3c197c1b4dd64257`；
- r2 JSON SHA256：`e8d3a0050343d105b87c6b809cd7d743f7b6e133201072e9115cb6cf1b7d8b0b`；
- r2 hammer Markdown SHA256：`4088cfa20481adff53476415adc5d01eddf2c56227d1fd28127f08065691c780`；
- r2 hammer JSON SHA256：`c6eb1f3955d3241f4e4b20f2afcb90cbcaea37614378dd5678c340b756f8f062`；
- M511 contract SHA256：`e556743dd18804a7aba5be5b18f33823bbcd5e5be85d7715edcc43a4c314c28e`；
- M523 contract SHA256：`6dac33f9fe035c0ed1c14ddd7dbc7d9ebfabcdec279cf027ce07cf0774baa415`；
- M523 RTL SHA256：`ad6def7cd81e5f3cd1570ef23fd062da19ee8b2a35498d6deca1c010522a0920`；
- `docs/359` SHA256：`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`，未修改。

本 r3 没有生成 cycle、area、power、energy、accuracy 或 system speedup；没有 author execution contract；
没有实现/修改 RTL；没有运行 CPU、Synopsys、训练或远端任务。任何超过 fresh independent r3 hammer 的
动作均未授权。
