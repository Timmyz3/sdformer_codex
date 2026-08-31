# M534 r4 下一条 RTL 候选筛选独立打铁 r1

日期：2026-08-27  
被审对象：`reviews/m534_next_rtl_candidate_screen_r4_20260827`  
模式：fresh independent、只读静态证据审计；零 CPU/VCS/DC/PT/PTPX/Formality/训练/远端运行；零 RTL 修改  
裁决：`PASS_SCREEN__AUTHOR_PRE_RTL_CPU_CONTRACT_ONLY__NO_RUN__NO_RTL`  
评分：**96/100**；P0/P1/P2 = **0/0/3**

## 1. 结论与授权

r4 已关闭 r3 hammer 的两个 P1。唯一合法 replay 现在是 block-outer deterministic；frontier
token、input skid、global owner、context/close key 与 external command owner 都由
`output_block` 区分。当前 `(output_block, source_linear_ordinal)` 的全部 legal descriptor
必须 retire 后 frontier 才能推进；当前 block 的全部 dense output 被 sink 接收、directory 清零、命令
retire 后才能换 block；全部 block 完成后才能换 epoch。两-block early-close 与 block-bit 翻转 fault
也成为未来 reference 的 mandatory negative。不同 block 不再能够相互 credit。

r4 同时删除了未冻结的 zero-token。每个 destination——包括 never-resident、directory bit=0 的
exact-zero destination——都必须显式构造并发送 padded `384 B` 向量。zero build `6 cycles`、first-beat
latency `32 cycles`、三个 accepted `128 B` beat 与 sink backpressure 对 A1/PBR4 四点共同收费。由此，
future analyzer 不再能在免费 implicit-zero 与 dense output 之间选择 denominator。

独立复算确认：三段 external aperture 不重叠；psum window 精确为 `42,393,600 B = 0x0286E000`，
最后合法 byte 为 `0x4286DFFF`；三个 128-bit packing 均逐位加和为 128；modeled logical on-chip total
为 `239,636 B`、低于 `240 KiB` `6,124 B`，并已明确标为非 foundry/CACTI/PPA closure。strongest-A1、
same-cohort、ELSA prior-art 边界、`1.30x/1.10x` 门与候选 B exact-zero 规则均保持。

因此本 hammer **只授权**创建一份仍为 `run_authorized=false` 的 pre-RTL CPU execution contract。
该 contract 必须关闭第 5 节三个 P2。CPU run、RTL authoring、VCS、DC、PT、PTPX、Formality、训练、远端、
performance/energy/PPA/system/headline 均仍为 false；不得由本裁决推导任何新性能数字。

## 2. block-qualified replay 与 frontier 复核

### 2.1 deterministic transcript 已闭合

r4 冻结的唯一循环次序为：execution record → `output_block` ascending → dense source C-order ordinal
ascending → legal `kernel_index` ascending。对每个 nonzero ordinal，所有 legal tap descriptor 必须全部 retire
后才允许对应 frontier token handshake；zero ordinal 仍产生一次收费 frontier handshake。进入下一 block 前，
当前 block 的 ingress/context/O8/pending write/directory/external transaction 全空、dense destinations 全部
commit。该定义同时冻结了 block replay 带来的 descriptor/frontier 重收费，禁止 source-outer alternative、
future-event oracle、跨 block 常驻和 point-local refill 删除。

canonical transcript 在 architecture analysis 前生成并封存；四点共享 transcript。mandatory per-sample
hash 已包含 replay transcript、descriptor accept/retire、frontier、weight identity/refill、output commit 与
output data。由 transcript 机械派生的单-resident-tile refill 顺序也被固定，完整 108 beat retire 前
`tile_valid` 不得置位。

### 2.2 block identity 与 transition 已闭合

frontier semantic identity 为
`{sample_id, execution_record, layer, output_block, tag, time, generation, ordinal}`；片上只允许一个 epoch、
record、block active。owner、token/skid 和已有 context/command identity 均含 `output_block2`；block mismatch、
ordinal regression、generation mismatch 都 sticky fault 且不得产生 close/commit/credit。

mandatory negative 覆盖了上一轮的关键反例：block0 到 dense end 而 block1 仍有 unretired descriptor 时，
block1 close/commit 必须为零；把 block0 token 的 `output_block` 翻为 block1 必须 fault 且不能 credit block1。
这是足以让 future contract 写出可执行 reference 的 screen-level closure。

## 3. dense output、packing 与共同收费复核

### 3.1 exact-zero 不再免费

统一 cursor 为 `(layer, output_block, destination_y, destination_x)` dense C-order。逻辑 payload
`96xAcc24=288 B`，传输统一 pad 到 `384 B=3x128 B`。never-existed zero 走
`FINAL_ZERO_BUILD0..5 -> COMMIT_FIRST_LAT -> COMMIT_BEAT0..2 -> DONE`：共享 vector buffer 每周期清
`48 B`，共 6 cycle；之后共同支付 32-cycle latency、3 beat handshake 和实际 sink stall。第三 beat 接收及
所需 directory clear RMW retire 前 cursor 不推进，最后一个 zero vector 也不能被 block transition 跳过。

候选与三个 A1 point 共享 cursor、ready transcript、output protocol、cycle 与 traffic charge。mandatory
metrics 包含 dense/zero/nonzero destination、logical/padded byte、zero-build、first-latency、accepted-beat、
stall 与 directory-RMW cycles。因此 r3 的 free zero-token P1 已关闭。

### 3.2 三个 128-bit packing 独立复算

- global frontier owner：
  `1+2+2+24+4+10+10+12+22+17+8+1+8+1+6 = 128 bit`；
- frontier token/input skid：
  `1+2+2+24+4+8+22+1+64 = 128 bit`；
- output cursor/final-commit control：
  `1+5+2+2+10+10+17+3+2+6+1+8+24+4+33 = 128 bit`。

新增 `output_block2` 使用 r3 reserved bits；zero-build/state/beat/latency 字段使用已经收费的 output-control
word；共享 `384 B` vector buffer 和 128-bit external command 均为继承状态。screen 中没有引入未收费
queue、buffer 或 port。

## 4. aperture、容量与公平门复核

### 4.1 external aperture

三段 64-bit、128-B 对齐、左闭右开的窗口为：weight `[0x10000000,0x20000000)`、final output
`[0x20000000,0x30000000)`、persistent psum `[0x40000000,0x4286E000)`，互不重叠。独立复算：

```text
110,400 slots * 384 B = 42,393,600 B = 0x0286E000
last slot base          = 0x4286DE80
last legal byte         = 0x4286DFFF
```

directory index 被限制为 `0..110399`，64-bit address add 禁止 wrap；越界、跨 aperture、alias 都 sticky fault。
aperture 大小没有被冒充片上容量或能量证据。

### 4.2 240 KiB identity

独立加和为：

```text
modeled arrays       16,384 + 196,608 + 8,192 + 16,384 = 237,568 B
soft-state byte-equivalent                               2,068 B
modeled logical total                                  239,636 B
logical 240 KiB budget                                 245,760 B
logical headroom                                         6,124 B
```

r4 已撤销 r3 的 `macro-rounded/accounted total` 物理措辞；foundry macro closure、CACTI closure、mapped
area/power/energy 全为 false。future CPU GO 后 only paired physical policy 要求 baseline/candidate 使用同一
compiler/DB/organization/port/latency，或在两边同时使用同一 CACTI 配置；logical、physical rounded、area、
leakage、dynamic energy 必须分列。任一点 physical capacity 超过 240 KiB 时两点共同 fail 或共同收窄。

### 4.3 分母、门槛与 claim boundary

`A1-SC8/A1-ISO8/A1-OSG/PBR4` 共享 exact S10、M218 `6x16/L4/O8/FIFO4/Acc24`、modeled
239,636-B coordinate、block-outer transcript、external link/backing/output FSM。`A1-STRONG` 是完整 S10
ratio-of-sums cycles 最小的一个固定 architecture point，禁止 per-sample oracle。PBR4 必须同时满足：
0 mismatch、ratio-of-sums `>=1.30x`、每 sample `>=1.10x`、weight reads/refill bytes 不增加、无额外资源、
且 group/RMW/commit sequence 不等价于 A1-OSG。否则只有 psum traffic 或 modeled dynamic energy
`>=30%` 才能作为 support/energy ablation。

S10 仍仅是 single-sequence CPU fast-kill；不是 multi-sequence、full-network、system speedup、paper headline。
Candidate B 继续要求 M519 canonical 后所有 cycle/transaction/read/byte/update/result/done delta 为 exact integer
zero，不产生 cycle/system speedup。

## 5. P2：future pre-RTL contract 必须写死的三项

以下不再要求 r5 screen，也不阻止 author contract；但 contract 未关闭它们前不得授权 run：

1. **FINAL_OUTPUT beat encoding 与 retire 语义。** r4 的 semantic identity 含 `beat_index`，现有 command packing
   只有 `slice3`/`beats_remaining2`。future contract 必须明确 `slice3` 是否直接编码 `beat_index`，并冻结
   `0..2` 的递增/hold 规则。README/JSON 的“response identity match”还必须明确是第三个 accepted outbound
   beat 的本地匹配，还是另有 sink ACK；若存在独立 ACK，必须冻结其 payload、latency、backpressure、state
   与 cycle charge，不能成为免费 response。
2. **final-output physical address 公式。** aperture 已冻结，但 destination 到地址的映射未写成唯一公式。
   future contract 必须冻结例如 `OUTPUT_BASE + directory_index*384` 的无别名映射，或明确该口是纯 ordered
   stream 且不使用 address；不能同时宣称 address stable/checked 又让 analyzer 自选地址。
3. **重复 source scan 的 traffic/energy 账。** block-outer replay 已通过每 block/per ordinal frontier handshake
   关闭 cycle denominator，但 source bitplane 被每个 output block 重读的 active-read/byte/energy 尚未列入
   common metrics。future contract 至少加入 `source_scan_bits/reads/bytes`，四点完全相同；在 paired macro/
   energy closure 前不得把 block-outer input access 当免费能量。

## 6. 最终授权矩阵

- author one pre-RTL CPU execution contract with `run_authorized=false`：**true**；
- 该 contract 必须关闭第 5 节三项后再接受独立 hammer：**true**；
- CPU run / RTL authoring / VCS / DC / PT / PTPX / Formality / training / remote：**false**；
- new cycle / area / power / energy / accuracy measurement：**false**；
- full-network / system speedup / DATE headline：**false**。

## 7. 身份、seal 与工作树

- 被审 README SHA256：`e54b9b768195721021d03fcdf41d2fdb78542d615cadde37ab86d9819abea776`；
- 被审 JSON SHA256：`a1594d8c92778269a4223bb900e5b73e7068b4db81c9a21e5a07a44297b4074b`；
- 被审 member seal 与 outer seal 均验证通过；四个 frozen r3/r3-hammer input SHA 均匹配；
- `docs/359` SHA256：`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`，未修改；
- `git diff --check` 通过；本 hammer 未修改被审目录、任何 RTL 或冻结输入。
