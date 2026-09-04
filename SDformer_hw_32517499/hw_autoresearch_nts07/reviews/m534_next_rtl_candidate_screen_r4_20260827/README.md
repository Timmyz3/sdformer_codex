# M534｜下一条 RTL 候选筛选 r4：block-qualified frontier 与显式 dense 输出

日期：2026-08-27  
模式：screen repair authoring only；零 CPU/VCS/DC/PT/PTPX/Formality/训练/远端运行；零 RTL 修改  
状态：`R4_SCREEN_REPAIRED_FOR_FRESH_HAMMER__NO_PRE_RTL_CONTRACT__NO_RUN__NO_RTL`

## 1. 裁决与授权

r4 是 r3 的窄化 overlay，只关闭 r3 fresh hammer 的两个 P1，并尽可能关闭三个 P2。r1、r2、r3
及其 seal 均不修改。r3 中未被本文件明确覆盖的 exact binary source、几何、ingress/context packing、
persistent backing、四个公平 architecture point、strongest-A1 选择、门槛、same-cohort 与 prior-art
边界继续冻结。

r4 选择一种、且只选择一种 descriptor replay：**block-outer deterministic replay**。frontier 的
architecture identity 加入 `output_block`；不同 block 的 frontier 永不互相 credit。不存在 analyzer
自选 source-outer/block-outer、提前关 block、隐藏 all-block barrier 或按 architecture 重排 refill 的自由。

r4 同时删除 r3 尚未定义的 dense-shape zero-token 假设。每个逻辑 output destination，包括 exact all-zero
destination，均向共同 sink 发出一个显式、收费、带反压的 padded `384 B` 向量。四个 architecture point
使用完全相同的 cursor、输出协议、cycle 与 traffic 收费。

本目录只授权 fresh independent r4 screen hammer。只有 fresh hammer 得到 `P0/P1=0`，才允许另行 author
仍为 `run_authorized=false` 的 pre-RTL CPU contract。CPU、RTL、VCS、DC、PT、PTPX、Formality、训练、远端、
performance/energy/PPA/system/headline 全部未授权。

## 2. 冻结的 block-outer replay

### 2.1 唯一 replay transcript

对冻结 M511 trace 中每个 `(sample, execution_record, layer, tag, time, generation)`，producer 必须先生成一份
与 architecture 无关的 canonical replay transcript；其循环次序固定为：

```text
for output_block = 0 .. output_blocks_96(layer)-1:
  for source_linear_ordinal = 0 .. source_population(layer)-1:
    observe the exact frozen dense binary source bit
    if bit == 1:
      emit all legal K3/S2/P1 tap descriptors for this output_block
      retire every descriptor for this (output_block, ordinal)
    advance only this output_block's frontier to this ordinal
  close and explicitly commit every destination of this output_block
  require all matching state and commands retired before output_block+1
```

`output_block` 递增、source C-order ordinal 递增、合法 tap 的 `kernel_index=0..8` 递增。top/left/right/bottom
legality 由冻结 ConvTranspose 几何逐 tap 决定。一个 ordinal 的 matching ingress/context/O8/pending-psum/
external command 尚未 retire 时，该 block 的 frontier 不得越过它。进入下一 output block 前，当前 block
的 ingress/context/O8/pending write/external transaction 必须为空，所有 destination 必须显式输出，directory
中属于当前 block 的所有 bit 必须清零。

canonical transcript 必须在任何 A1/PBR4 analyzer 运行前生成并封存 SHA256；四点读取同一个 transcript。
禁止 simulator 按 point 选择 output-block 次序、跳过 dense zero ordinal、合并两次 frontier、预知未来事件、
或从结果反推更有利的 refill。未来 contract 必须逐 sample 封存：

- transcript SHA256；
- 每 `(layer,output_block)` 的 dense ordinal 数、nonzero event 数、legal descriptor 数；
- frontier token 数与最后 ordinal；
- descriptor accept/retire 次序 hash；
- weight tile identity/refill 次序 hash；
- output commit 次序与 data hash。

### 2.2 block-qualified frontier identity

frontier token、frontier input skid、global frontier owner、context close key 与 external command owner 的 exact
匹配 key 统一为：

```text
{sample_id, execution_record, layer2, output_block2,
 tag24, time4, generation8, source_linear_ordinal22}
```

`sample_id` 与 `execution_record` 是 simulator/trace envelope identity，不作为未收费片上并发状态；硬件同一
时刻只允许一个 epoch、一个 execution record 和一个 output block active。片上 owner 的 128-bit packing 将
r3 的两个 reserved bit 改作 `output_block2`：

```text
valid1, layer2, output_block2, tag24, time4,
input_height10, input_width10, input_channels12,
frontier_ordinal22, output_cursor17, generation8,
stream_last_seen1, fault_epoch8, clear_pending1, reserved6 = 128 bit
```

frontier token/input skid 的 exact charged packing 是：

```text
valid1, layer2, output_block2, tag24, time4, generation8,
frontier_ordinal22, stream_last1, reserved64 = 128 bit
```

reserved 固定写零、读时检查零。每个 output block 的每个 dense ordinal 都产生一个 charged frontier
handshake；nonzero ordinal 的 frontier 只能在该 ordinal 全部 legal tap descriptor retire 后握手，zero ordinal
也必须握手，不能被 simulator 无周期地跨过。每个 descriptor 仍按 r3 `128-bit atomic ingress` 接口逐项收费；
重复 output block 就重复收取 descriptor/frontier ingress cycle 与 count。output block `b` 的 token只可更新 owner
`output_block==b`；任何 block mismatch、ordinal 回退或 generation mismatch 都 sticky fault，且不能产生 close、
commit 或其他 block 的 credit。`stream_last` 只把当前 block 的 frontier 推到 dense layer end，不能结束其他
block 或 epoch。

semantic close 仍要求：当前 block frontier 已到 destination 的 `last_possible_source_ordinal`，且没有 matching
ingress、context contributor、O8 response、pending psum write、directory RMW 或 external transaction。当前
block 的最后一个 frontier 不能推进到下一 block；只有 current-block dense output 全部 accepted by sink、目录
清空且所有命令 retire 后，owner 才以明确的 block-transition edge 装载 `b+1` 并将 frontier 复位为 invalid。

### 2.3 refill schedule 不得自选

weight identity 仍为：

```text
{layer, output_block, floor(source_channel/16),
 source_channel mod 16, kernel_index, slice}
```

唯一 resident tile 的命中/失配与 refill 次序由 canonical transcript 顺序机械推导；output-block 或 Cin tile
变化造成的每次 miss 都收费。每个 refill 仍是 108 个 beat 完整 retire 后 `tile_valid=1`。四个 A1/PBR4 point
共享同一个 refill request transcript、shared-link arbitration、first-beat latency、bytes 与 active-read收费；
禁止 point-local weight prefetch、隐藏第二 tile、重新排序 ordinal/tap、跨 block 常驻或用 oracle 删除 refill。

未来 CPU reference 必须至少有一个两-output-block negative：block 0 已到 layer-end、block 1 仍有未 retire
descriptor 时，block 1 的 close/commit 必须为零；将 block 0 frontier token 的 output_block bit 翻到 1 必须
触发 identity fault，不能 credit block 1。

## 3. exact dense output：取消 zero-token

### 3.1 统一输出接口

四个 architecture point 都必须按
`(layer,output_block,destination_y,destination_x)` 的 dense C-order cursor 输出，不能只枚举 nonzero/event
destination。每个 destination 的逻辑 payload 是 `96 x Acc24 = 288 B`，外部传输固定 pad 到 `384 B`，分为
三个 `128 B` beat。输出 command 与每 beat 必须携带/匹配：

```text
{opcode=FINAL_OUTPUT, layer, output_block, destination_y, destination_x,
 tag, time, directory_index, generation, beat_index}
```

协议为 ready/valid：`valid && !ready` 时 command、address、beat index、data 全部稳定；仅 handshake 后推进。
sink 不接受 implicit completion、zero-token、sparse-shape omission 或 free Python set。每个 destination 均收费
`32-cycle first-beat latency + 3 accepted beats`；sink stall 逐周期实收，且四点使用同一冻结 ready transcript。

### 3.2 never-existed zero 的 exact 路径

output cursor 查询当前 destination 时，若 resident miss 且 persistent directory bit=`0`，语义是 exact
`96 x Acc24 == 0`，但不是免费完成。控制器进入：

```text
FINAL_ZERO_BUILD0..5
  -> COMMIT_FIRST_LAT
  -> COMMIT_BEAT0..2
  -> DONE
```

`FINAL_ZERO_BUILD0..5` 每周期把共享 `384-B` vector buffer 的一个 48-B slice 显式写零，共收费 6 cycles；
之后发出三拍全零 data。若 resident 或 backing 存在，则沿 r3 的 charged read/restore/final-commit 路径填同一个
buffer，之后使用完全相同的 32+3 output protocol。任何 destination 都不得用 directory bit 直接给 sink
completion credit。

output cursor 只有在第三 beat handshake、response identity 匹配且所需 directory clear RMW retire 后才推进。
一个 output block 的最后一个 zero vector 也必须完整接受，才能执行 block-transition；因此 zero-fill cadence、
frontier cadence 与 sink cadence 被同一状态机闭合。新增状态编码使用 r3 已收费的 128-bit
`output_cursor/final-commit control` reserved bits，不新增 queue、buffer 或 port。其 exact packing 冻结为：

```text
valid1, state5, layer2, output_block2, destination_y10, destination_x10,
directory_index17, zero_build_index3, beat_index2, first_latency_counter6,
sink_wait1, generation8, tag24, time4, reserved33 = 128 bit
```

`state5` 明确覆盖 r3 final/restore states 与 r4 `FINAL_ZERO_BUILD0..5`；`zero_build_index3` 只能为 0..5、
`beat_index2` 只能为 0..2。shared external command 仍用 r3 已收费的 128-bit word，其中 `opcode3` 新增
`FINAL_OUTPUT` 编码且 `destination20={destination_y10,destination_x10}`；不增加 command bits。

### 3.3 共同收费

未来 analyzer 对四点必须逐 sample 同时报：dense destination count、zero-vector count、nonzero-vector count、
logical payload bytes、padded output bytes、zero-build cycles、first-latency cycles、accepted beat cycles、stall
cycles、directory RMW cycles。候选和 A1-STRONG 的 output cursor/ready transcript 完全相同；不得只给 candidate
跳过 zero vector，也不得只给 baseline 收取 dense output。

## 4. backing 地址范围与容量身份

### 4.1 冻结的 64-bit external aperture

以下为未来 CPU contract 的唯一合法 abstract external address map，均为 128-B 对齐、左闭右开：

| aperture | base | exclusive limit | 语义 |
|---|---:|---:|---|
| weight | `0x0000000010000000` | `0x0000000020000000` | weight/refill only |
| final output | `0x0000000020000000` | `0x0000000030000000` | dense 384-B output only |
| persistent psum | `0x0000000040000000` | `0x000000004286E000` | exactly 110,400 × 384 B |

`PSUM_SCRATCH_BASE=0x0000000040000000`，window=`42,393,600 B=0x0286E000`，最后合法 byte 地址为
`0x000000004286DFFF`。slot `directory_index` 的地址仍为 `base + index*384`；index 只能是 `0..110399`。
地址加法必须 64-bit、禁止 wrap。psum command 落入 weight/output aperture、超出 psum window、或两个
aperture alias 都 sticky fault。三者共享 128-B/cycle external link，实际 traffic/latency 全收费；aperture
大小不是免费 SRAM/DRAM 容量，也不是 energy 结果。

### 4.2 `239,636 B` 是 modeled logical closure，不是 foundry macro closure

r3 的 `16,384 + 196,608 + 8,192 + 16,384 = 237,568 B` 四项在 r4 统一改称**modeled array
organization logical capacity**；加 `2,068 B` byte-equivalent soft state 后为：

```text
modeled array logical capacity          237,568 B
modeled soft-state byte-equivalent        2,068 B
-------------------------------------------------
modeled logical on-chip total           239,636 B
logical 240 KiB budget                  245,760 B
logical headroom                          6,124 B
foundry macro closure                       false
CACTI closure                               false
```

这不是 macro-rounded physical capacity、mapped area、power 或 energy。r3 中所有把该数称为
`macro-rounded/accounted total` 或把 nominal organization 当 foundry macro 的措辞由本节覆盖。headroom 仍
不是 free state；新增 state 必须重新计费，logical total `>245,760 B` 即 fail closed。

只有未来 CPU GO 后，baseline 与 candidate 才能用**同一 foundry SRAM organization、同一 compiler/DB、
同一 depth/width banking、同一 port/latency**做 paired closure；若目标库无对应宏，则两点同时用同一
CACTI version/node/voltage/banking/port/latency 配置，并把逻辑容量、物理 rounded capacity、area、leakage、
dynamic energy 分列。禁止 baseline 用 foundry macro 而 candidate 用 logical bits，禁止一边 CACTI 一边 SRAM
compiler，也禁止用 `239,636 B` 冒充 paper PPA ready。任一点 physical rounded capacity 超过 240 KiB 时两点
均 fail closed 或共同收窄后重新筛选。

## 5. 保持冻结的公平性与门

未来一次 CPU fast-kill 仍须在 exact S10、同 M218 `6x16/L4/O8/FIFO4/Acc24` service、同 modeled
`239,636 B <= 240 KiB` logical coordinate、同 block-outer replay transcript、同 external link/backing/output
FSM 上同时测 `A1-SC8/A1-ISO8/A1-OSG/PBR4`。

`A1-STRONG` 仍是完整 S10 ratio-of-sums cycles 最小的一种固定 architecture point；per-sample oracle 禁止。
PBR4 仅在全部成立时可继续：function mismatch=`0`；
`sum(A1-STRONG cycles)/sum(PBR4 cycles)>=1.30x`；每个 sample `>=1.10x`；weight active reads/refill bytes
不增加；无额外 port/bandwidth/state；group/RMW/commit sequence 不等价于 A1-OSG。未过 1.30x 时，只有
psum read+write 或 modeled dynamic energy 降低 `>=30%` 才能降为 decoder support/energy ablation。

S10 仍只是 single-sequence fast-kill，不是 multi-sequence、full-network、system speedup 或 paper headline。
候选 B 仍要求 tagged/elided cycles/accept/request/response/read/byte/update/result/done 全部整数零；它不产生
cycle/system speedup。

## 6. r3 hammer closure

| finding | r4 closure |
|---|---|
| P1-01 output-block replay 与 frontier identity 不闭合 | 唯一 block-outer transcript；frontier owner/token/close key 加 output_block；current block descriptors 全 retire 后才推进；所有 block dense commit 后才换 epoch；refill 由 transcript 固定；两-block early-close negative mandatory |
| P1-02 dense zero-token 未定义/未收费 | 删除 zero-token；所有 destination 发显式 384-B vector；zero build 6 cycles、first latency 32 cycles、3 beat 与 sink stall 全收费，四点共同协议 |
| P2-01 逻辑容量冒充 macro closure | `239,636 B` 明确仅 modeled logical total；foundry/CACTI closure=false；未来 baseline/candidate 必须同一物理 organization |
| P2-02 scratch base/range 未冻结 | base=`0x40000000`、exclusive end=`0x4286E000`、42,393,600 B、与 weight/output aperture 明确不重叠 |
| P2-03 JSON sign 命名歧义 | 字段改名 `source_sign_bit_one_is_malformed`；numeric source value one 仍是合法且必须的 `+1` |

## 7. 身份与红线

冻结输入：

- r3 README SHA256：`a1bcc96e52fff2d06ae8bf8d34eb171bedc9308826125795711898ba590bfdad`；
- r3 JSON SHA256：`574ca08321f0cab38a6b652b70bf911cda37f26c9143027966a979f1e02dadde`；
- r3 hammer Markdown SHA256：`efa37e10ab46c0c20f24abf8c79f555649d2f11d040f6cb088f60c0c920141f8`；
- r3 hammer JSON SHA256：`239bc69655005691b994a3dd9ed123d74bd269d0fecae77ac73c76d9a0a90523`；
- `docs/359` SHA256：`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`，未修改。

r4 不产生 cycle、area、power、energy、accuracy 或 system speedup；不 author execution contract；不实现或修改
RTL；不运行 CPU、Synopsys、训练或远端任务。任何超过 fresh independent r4 screen hammer 的动作均未授权。
