# M534 r3 下一条 RTL 候选筛选独立打铁 r1

日期：2026-08-27  
被审对象：`reviews/m534_next_rtl_candidate_screen_r3_20260827`  
模式：只读静态证据审计；零 CPU/VCS/DC/PT/PTPX/Formality/训练/远程运行；零 RTL 修改  
裁决：`FAIL_CLOSED__AUTHOR_R4_SCREEN_REPAIR_ONLY__NO_PRE_RTL_CPU_CONTRACT__NO_RUN__NO_RTL`  
评分：**89/100**；P0/P1/P2 = **0/2/3**

## 1. 结论

r3 把 r2 的两个主要缺口修到了很接近可执行：M511 的 decoder source 已冻结为
exact binary `+1`，`source_sign` 与 INT8 weight sign 分离；`8x128b` ingress 和
`16x256b` context 的 bit range 连续无洞；cross-event/Cin-tile 的 contributor 可由冻结
tuple 唯一重建。persistent directory 的四层映射、zero-fill/restore/writeback/commit
状态和 `384 B` vector staging 也已进入账本。独立复算的容量确为
`239,636 B`，低于 `240 KiB` `6,124 B` 。atomic bundle 在公平 memory/sink response
假设下不再存在 r1 那种 nominal deadlock P0。

但 r3 还不允许创建 pre-RTL CPU execution contract，原因是两个会直接改变
correctness/cycles 的身份仍未冻结：

1. 同一 M523 source descriptor 按 `output_block` 重放，但 canonical frontier 的 owner/token
   都没有 `output_block`。如果按 weight-resident 友好的 block-outer 顺序重放，第一个
   block 的 layer-end frontier 会让后续 block 的 destination 在其未来 descriptor 尚未进入时
   early-close；如果要用 source-outer 顺序，则必须冻结“该 ordinal 的全部 block
   replay 均 atomic accept 后 frontier 才推进”，并实收频繁 weight-tile 切换。当前
   `global_frontier_owner128` 的定义不能区分这两种行为。
2. directory bit=`0` 且无 resident line 时，r3 引用了一个未冻结输入中存在的
   `dense-shape zero token`，但没有定义 token 位宽、valid/ready、是否占 result
   edge/byte、sink backpressure 及接收端的 implicit-zero state。把它当免费调用会少收完整
   output scan/commit 成本，也不能证明对现有 dense decoder 下游的 exact 语义。

这两项都可在 r4 screen 中修复，不需要运行或写 RTL。在修复并经 fresh hammer
`P0/P1=0` 之前，本审只授权 **r4 screen repair authoring**；pre-RTL CPU contract、CPU
run、RTL、VCS、DC、PT/PTPX、Formality、训练、远程、performance/energy/PPA/system/headline
全部为 false。

## 2. 已通过的核心复算

### 2.1 decoder 数值、ingress 与 context packing

- M511 四层 shape 与 r3 一致；source population 分别是
  `460,800 / 924,000 / 1,852,800 / 3,724,800`，最大 ordinal `3,724,799 < 2^22`。
- ingress 从 bit 0 到 127 连续无 overlap/gap，完整保留 source/destination/kernel/
  layer/output-block/tag/time/ordinal/generation/fence identity。`source_sign=0` 是隐式 `+1`；
  product sign 只由 signed INT8 weight 决定。
- context 从 bit 0 到 255 连续无 overlap/gap；8 个 17-bit contributor slot 完整保留
  global `source_channel12 + kernel_index4 + source_sign1`。对合法 K3/S2/P1 tap，
  `source=(destination-kernel+1)/2` 唯一，因此 source coordinate/ordinal 可从 context 字段
  无损重建，cross-Cin16 bundle 不需 hidden active-tile identity。
- 一个 bundle 最多 8 个 lane；即使 8 个 distinct destination 都落在一个 phase，
  atomic ingress 可先整包 accept，4 context 完成第一波 charged partial-RMW 释放后再处理
  第二波。context 只保留一个 bundle epoch 的 contributor，不把 drain 解释为 semantic
  close，故未见需要第 17 个 context 的结构性 P0。

### 2.2 persistent directory 和 backing 语义

directory 索引独立复算：

| layer | destinations |
|---:|---:|
| 0 | `4*30*40 = 4,800` |
| 1 | `2*60*80 = 9,600` |
| 2 | `1*120*160 = 19,200` |
| 3 | `1*240*320 = 76,800` |
| **total** | **110,400 bit** |

`1024x128b = 131,072 bit = 16,384 B`，未用 `20,672 bit`；四层 base
`0/4,800/14,400/33,600` 和 extent 无 overlap，最后一个合法 index 为 `110,399`。
`directory_index*384` 产生的 external window 总大小是 `42,393,600 B`
(`40.4296875 MiB`)，96xAcc24 的 `288 B` payload 被统一 padding 为 3x128 B。

r3 对 bit0 zero-fill、bit1 restore、dirty writeback 后 set、resident/evicted final commit 后 clear、
1024-word epoch clear 和 same-word RAW bypass 的收费方向一致。directory 不再用 resident
tag 或 Python set 偷存 persistent existence；`3072b` shared vector buffer 也覆盖了一个完整
padded psum vector。

### 2.3 240 KiB 账本

| 类别 | byte |
|---|---:|
| 8x weight data | 16,384 |
| 4x psum data | 196,608 |
| 4x resident tag/frontier | 8,192 |
| persistent directory | 16,384 |
| **macro subtotal** | **237,568** |
| soft state | 2,068 |
| **accounted total** | **239,636** |
| 240 KiB budget | 245,760 |
| **headroom** | **6,124 (2.491862%)** |

soft subtotal `2,068 B` 的逐项加和正确；416-bit psum pending entry、160-bit directory
bypass、3,072-bit shared vector、64/128-bit owner/control 均已 byte-round。headroom 被正确标记为
非 free state，soft state 也保留 future paired-DC 二次收费。

## 3. P1 findings

### P1-01｜frontier 身份缺少 output-block replay fence

r3 ingress/context/resident/directory/weight command 都包含 `output_block`，而
`canonical_frontier.must_match` 只有 `{layer,tag,time,generation}`，`global_frontier_owner128`
也没有 `output_block`。这与 README 的“同 descriptor 按 block 重放”形成不完整复合键。

风险是 correctness 级的：block-outer 顺序在 block0 结束后把 frontier 推到 layer end，
block1 第一波 context 完成后即可被当成“无未来 contributor”而提前 commit。如果意图是
source-outer/all-block replay，则 frontier 只能在同 ordinal 的所有 output block 均已原子进入后
推进；该 barrier/credit 与 weight-tile 切换必须收费，不能由 future analyzer 自选。

r4 必须冻结唯一方案：

- 推荐：把 `output_block2` 加入 frontier token/owner/close key，冻结 block-outer 重放与每 block
  ordinal 单调性；所需 2 bit 可从已声明 reserved 中取，但须产生新 screen identity；或
- 冻结 source-outer 且明确 all-output-block atomic replay barrier，并把 barrier state、stall、weight
  refill 全部收费。

修复后必须加一个最小 negative reference：layer0 至少两个 output block，先完成 block0
的 dense frontier，再开始 block1；block1 在自身 frontier 过界前 commit count 必须为零。

### P1-02｜dense-shape zero token 不是已冻结的免费下游协议

r3 在 `directory=0 && no resident` 时允许不生成 data beat，改用“已冻结 dense-shape
zero token”，但八个 frozen input 里没有这个协议。M511 只冻结 ConvTranspose input bitplane，
M523 只冻结 descriptor bundler；两者都没有定义 decoder output 的 implicit-zero receiver。

这会改变每个 never-existed destination 的 output cursor/result-link cycle 和 byte，也会影响共同开销
下的 speedup ratio。r4 必须在四个 architecture point 共享的边界中二选一：

- 一律发送 charged 384-B exact-zero dense vector；或
- 定义一个 exact zero-token 协议，至少冻结 token 字段/位宽、valid-ready、每 token
  cycle/byte、backpressure、sink reconstruction state 及该 state 的容量/面积归属。

在此之前，future CPU contract author 仍可在两个不同 denominator 中任选一个，因此不准入。

## 4. P2 和物理边界

1. **当前是 modeled memory organization，不是 foundry macro closure。**
   `128x128 1RW/L4`、`1024x384 1RW/L4`、`128x128 1R1W/L1`与
   `1024x128 1RW/L1` 的 bit-capacity/port/latency 身份已列，但尚无 target SRAM compiler
   macro name/DB/LEF/width-depth banking 回执。特别是 1R1W tag macro 若只能用 1RW 复制或
   flops 实现，物理代价不能由 `8,192 B` 原始位容量代替。`239,636 B`
   只能写 logical/modeled capacity closure，不能写 paper-PPA-ready 或宏面积。
2. **external window 的地址归属待冻结。** `PSUM_SCRATCH_BASE` 仍是 symbolic；r4/
   future contract 应至少要求 128-B alignment、`42,393,600 B` 无别名保留区间、与
   weight/output window 不 overlap，并把 DRAM/bank mapping 敏感性留给后续 macro-inclusive
   energy/system ledger。这不阻止 fixed-latency CPU screen，但阻止把符号地址当实际
   DRAM 结果。
3. JSON 的 `source_sign_encoding.one_is_malformed=true` 语法上容易被解释为“numeric
   source value 1 非法”，与 `emitted_source_numeric_value=1` 字面冲突。Markdown 的意图清楚：
   非法的是 `source_sign bit == 1`。r4 应改名为
   `source_sign_bit_one_is_malformed=true`，避免 contract consumer 误读。

## 5. 公平性、prior art、cohort 与候选 B

- `A1-SC8/A1-ISO8/A1-OSG/PBR4` 共享 `6x16/L4/O8/FIFO4/Acc24`、相同
  239,636-B modeled coordinate、相同 external link/backing FSM，没有发现 candidate-only
  port/bandwidth/state。
- strongest baseline 在完整 S10 上选一个固定 ratio-of-sums 最快点，禁止 per-sample
  oracle；PBR4 只对 strongest 报，且与 A1-OSG 的 group/RMW/commit sequence 等价时
  KILL。这一分母规则通过。
- ELSA 的 bundled AER、mini-batch Gustavson、single-RMW 和 dependency completion 已被列为
  direct prior；只有过 strongest A1-OSG 后才保留 K3/S2 ConvTranspose parity frontier +
  bounded atomic bundle + typed signed-source service 的窄 claim。该边界通过。
- S10 明确只是 single-sequence fast-kill；S10/S100 不混用，system share 继续需要同
  checkpoint/T10/sample manifest denominator。该边界通过。
- 候选 B 继续等待 independently sealed M519 K1/K8/K1x8 canonical；cycles、accept、
  request/response、active-bank read、data bytes、Acc24 updates、result beats 和 done cycles 全部
  要求 exact integer zero delta。B 不产生 cycle/system speedup，本审不授权 RTL。

## 6. r4 最小修复单与授权

r4 只需修文档/JSON，不需要运行或 RTL：

1. 冻结 output-block replay 顺序，将 block 纳入 frontier identity 或增加 all-block accept
   barrier，并冻结对应 stall/refill 成本。
2. 将 never-existed output 改为 charged dense zero vector，或完整定义且收费 exact
   zero-token/sink protocol。
3. 把当前 macro 表标注为 modeled organization；实体 macro rounding/port/latency/area 继续留给
   CPU GO 后的 paired physical gate。
4. 冻结 external scratch alignment/range/non-overlap 要求，改名 JSON 的 sign 字段。

r4 fresh hammer 只有在 `P0/P1=0` 后才能授权创建一份仍为
`run_authorized=false` 的 pre-RTL CPU contract。本审当前授权矩阵：

- r4 screen repair authoring = true；
- pre-RTL CPU execution contract authoring = false；
- CPU run / RTL / VCS / DC / PT / PTPX / Formality / training / remote = false；
- performance / energy / PPA / full-network / system / DATE headline = false。

## 7. 身份、seal 与工作树

- 被审 README SHA256：`a1bcc96e52fff2d06ae8bf8d34eb171bedc9308826125795711898ba590bfdad`；
- 被审 JSON SHA256：`574ca08321f0cab38a6b652b70bf911cda37f26c9143027966a979f1e02dadde`；
- 被审 member seal 和 outer seal 均验证通过；八个 frozen input SHA 均与当前文件一致；
- `docs/359` SHA256 为
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`，未修改；
- `git diff --check` 通过；本 hammer 未修改被审目录、任何 RTL 或其他冻结输入。

