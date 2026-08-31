# M542｜M534/PBR4 pre-RTL CPU 合同 fresh static hammer r1

日期：2026-08-27  
被审合同：`contracts/m542_m534_pbr4_pre_rtl_cpu_execution_contract_r1_20260827.json`  
模式：fresh independent、只读静态审计；零 candidate CPU/analyzer、RTL、EDA、训练、GPU 与远端运行  
裁决：`FAIL_SOURCE_ONLY_PRE_RTL_CPU_CONTRACT__REPAIR_REQUIRED__NO_RUN_OR_RTL`  
评分：**77/100**；P0/P1/P2 = **0/3/2**

## 1. 结论

本合同已经把 r4 hammer 指定的三类局部问题大体写成了可执行规格：A1 三点在 PBR4 不可见时先跑并封存、
固定一个完整 S10 sum-cycle 最小的 `A1-STRONG`；FINAL_OUTPUT 使用 `slice3=beat_index=0/1/2`，逐拍
ready/valid 退休；final-output 地址按 layer/output-block/y/x 唯一映射；block replay 的 source bit/read/
logical byte/padded byte/base/stall/symbolic-energy 也声明为四点共同收费。身份、member seal、outer seal、
`239,636 B modeled logical` 边界与全部当前零运行计数均通过。

但它还不能给 source-only PASS。存在三个会改变未来功能或性能分母的 P1：

1. exact T10 payload 与 canonical loop/source ledger 不一致，若按合同文字执行会少算 9 个 timestep；
2. numeric binary source `1` 与 `source_sign=1 malformed` 在同一 functional reference 中直接冲突，回退了
   r4 已关闭的 sign 语义；
3. 四个 architecture point 只有名称和共同资源，没有在本合同中给出可执行算法语义，也没有声明 r4→r3→r2
   为规范性递归继承，future runner 因而仍能自行解释 SC8/ISO8/OSG/PBR4。

因此本 hammer **不授权** runner source admission、CPU launch 或 RTL。修复合同并重新双封后，需要一轮新的
fresh static hammer；只有新的 P0/P1=`0/0` 才能另建、且仅另建 runner-source author admission。

## 2. P1 findings

### M542-R1-P1-01｜T10 没有进入 canonical loop 与 source/output 共同账

合同同时冻结了：

- `cohort.timesteps=10`；
- M511 每个 module record 的 layout 为 `T_B_C_H_W`；
- 40 个 record、`696,240,000` bit、`87,030,000` byte；
- 每层 `source_population` 分别为 `Cin*Hin*Win`，也就是**单 timestep**人口。

独立整数复算为：

| layer | 单 timestep source_population | output blocks | 单 timestep block-replay bits | T10 bits/sample |
|---:|---:|---:|---:|---:|
| 0 | 460,800 | 4 | 1,843,200 | 18,432,000 |
| 1 | 924,000 | 2 | 1,848,000 | 18,480,000 |
| 2 | 1,852,800 | 1 | 1,852,800 | 18,528,000 |
| 3 | 3,724,800 | 1 | 3,724,800 | 37,248,000 |

四层合计应为 `92,688,000` replay bit/sample、`926,880,000` replay bit/S10。原始 M511 payload 复算则是
`10 samples * 10 timesteps * sum(source_population) = 696,240,000 bit`，与合同的 M511 identity 精确相符。

然而 `canonical_trace.loop_order` 没有 `time ascending`，而
`source_scan_*_formula_per_layer_sample` 全部只写 `source_population*output_blocks`，没有乘 `10`，mandatory
结果又只有 per-sample/per-layer 粒度而没有 per-time 粒度。按字面执行会把 replay bit/read/base-cycle/
logical-byte/padded-byte 以及 dense output epoch 少算十倍；这类共同固定成本会直接改变 ratio-of-sums，不能留给
runner 猜测。`time4` 出现在 owner packing 不能替代循环和 ledger 定义。

必须修复为二选一且全链一致：

1. canonical loop 明确 `record/layer -> time 0..9 -> output_block -> ordinal -> kernel`，per-sample/layer
   聚合公式显式乘 `T=10`；或
2. mandatory 输出改为 per-sample/layer/time，随后以十个 time epoch 的整数和生成 per-sample/layer 聚合。

同时应冻结每个 time epoch 的 close/clear/output 数量、ready transcript reset/continuation规则，以及 final-output
同一物理地址跨 time 重用只能发生在前一 time 的全部 beat 与 directory clear 已退休之后；否则应把 time 纳入地址。

### M542-R1-P1-02｜numeric source `1` 与 sign bit `1` 再次混为一谈

合同 `functional_reference` 一方面写：

- `accepted_source_values=[0,1]`；
- `emitted_source_value=1`；

另一方面又用一个无类型字符串写 `source_sign: "0 is +1; 1 is malformed"`。这让同一个 `1` 同时是唯一合法
event 值和 malformed 值。M534 r4 已明确关闭过这个问题：numeric source `1` 是隐式 `+1`；另一个独立的
source-sign metadata bit 必须为 `0`，其 `1` 才 malformed；product sign 只来自 signed-INT8 weight。

必须恢复结构化、无歧义定义，例如：

```text
accepted_numeric_source_values = [0,1]
emitted_source_numeric_value = 1
source_sign_encoding.positive_one = 0
source_sign_encoding.source_sign_bit_one_is_malformed = true
source_sign_encoding.product_sign_source = signed_int8_weight
```

并要求 reference、descriptor hash、weight/product miter 分别使用 numeric activity 与 sign metadata，禁止复用一个
`source_sign` 字段解释两种对象。

### M542-R1-P1-03｜四个 architecture point 没有被本合同规范性定义

合同只列出 `A1-SC8/A1-ISO8/A1-OSG/PBR4` 名称、共同 service/memory/link 和选择/门槛；没有冻结四点各自
如何形成 group、何时开/关 context、如何执行 RMW/commit、哪些 merge/lookahead 合法。虽然
`frozen_predecessors` 记录了 M534 r4 的 SHA，但合同没有声明该文件及其 r3/r2 inheritance 是**规范性语义**、
没有规定冲突时的 override precedence，future launch 也没有要求递归 pin 整条语义链。单纯把某文件列为
predecessor identity 不等于把其中算法规格导入本 execution contract。

这会使 runner 可以在同样端口/容量下自行实现一个更弱的 A1 或更强的 PBR4，随后仍满足当前 JSON schema；
`A1-OSG equivalent` gate 也无法在没有 exact OSG/PBR4 规则时被独立复核。

修复必须任选一种：

- 在合同中复制四点的 exact executable transition/group/RMW/commit 语义与 mandatory negative；或
- 明确声明 M534 r4→r3→r2 的 exact hash 链为 normative recursive import，列出 precedence/override，要求
  runner、launch admission 和 result identity 同时 pin 全链 SHA，并列出四点的 contract-level 摘要和不可变规则。

## 3. P2 findings

### M542-R1-P2-01｜output backpressure 的“无新 read”仍是间接推导

`hold_rule` 已要求 command/address/beat/data 全稳定，`retry_semantics` 也禁止新 request count；但请求要求的
“valid-not-ready 时无新 read/request/retirement”没有以计数不变量直接写出。当前只能从“nonzero read 在发送前”
和“one active command”间接推导，runner 仍可能在 sink stall 时预读下一 destination 或让 directory clear
退休而不违反现有 hold 字符串。

下一版应增加 exact invariant：每个 `out_valid && !sink_ready` 周期，output/psum read、new FINAL_OUTPUT
request、accepted beat、transfer retire、directory-clear retire、cursor advance 全部 delta=`0`；唯一 primary
class 为 `final_output_sink_stall`。对应 counter 必须进入 conservation 与 negative test。

### M542-R1-P2-02｜`docs/524` “未被 author 修改”缺少可独立验证的 pre-author identity

M534 r1--r4 与其 hammer 的现有 seals 均验证通过；`docs/359` 也有冻结 SHA。但 `docs/524` 当前是 untracked，
合同、handoff 与 request 都没有给出 author-task 前 SHA/size/mtime identity。本 hammer只能看到 handoff 的
self-attestation，无法独立证明“author task 未修改 docs/524”。这不影响本合同的算法语义，但未满足请求中的
provenance check。后续 author request 应在开始前 pin `docs/524` SHA，handoff 再复核同一 SHA。

## 4. 已通过的静态检查

### 4.1 身份、seal 与授权

- 合同 SHA256：`41e03c1b007b47822da7b5326a591f79f5af5c12bb59afb24d7a9a6e05deb53a`；
- contract member sidecar file SHA256：`091c982c990b1bb237f78ab8f253233d62ca94d68d6c41b411f464b82b3d5caf`；
- contract outer sidecar file SHA256：`5e1876d177249d553c60c04af766626507379722dc3a8bc6f4b2bfcbb89c60ad`；
- author handoff member/outer seal 均验证通过；request member/outer seal 均验证通过；
- M534 r1--r4 及四轮 hammer 当前 seals 均通过；合同列出的 frozen predecessor SHA 全部匹配；
- contract、handoff、future schema、request 均以 duplicate-key/non-finite fail-closed parser 静态解析通过；
- `run_authorized=false`；当前 CPU/analyzer/RTL/VCS/iverilog/Verilator/DC/PT/PTPX/Formality/训练/GPU/远端
  与 result-directory count 均为整数 0；future CPU count=1，其他 future run count=0；
- canonical result、attempt marker、future Python runner、future shell runner、M511 payload directory 均不存在；
- `docs/359` SHA256 仍为
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

### 4.2 fixed denominator 与同资源边界

- phase order 明确先生成/封存共同 transcript，再依次完成三种 A1，封存 A1-only metrics 并固定一个完整 S10
  sum-cycle 最小点，随后才允许构造/执行 PBR4；
- tie order 固定为 `A1-OSG -> A1-SC8 -> A1-ISO8`，禁止 per-sample/per-layer oracle；
- 所有 sample/layer/traffic/energy/gate 共用同一个 fixed A1；
- 四点均声明使用 M218 `6x16/L4/O8/FIFO4/Acc24`、相同 SRAM port/latency/state、相同 external link、
  output protocol、ready transcript 与 source replay；candidate-only queue/port/bypass/prefetch/state=false；
- `222,736 B` 已 superseded；`239,636 B = 237,568 B arrays + 2,068 B soft-state equivalent`，低于
  `245,760 B` 仅 `6,124 B`；foundry/CACTI/area/power/energy/PPA 全为 false。

### 4.3 beat 与 address 数学

- `slice3` exact 编码 beat 0/1/2，`beats_remaining=2/1/0`；beat 在 outbound ready/valid 时逐拍只退休一次，
  beat2 是唯一 transfer retirement；无独立 sink ACK/free response；duplicate/mismatch sticky-fault；
- directory interval 逐层为 `[0,4799]`、`[4800,14399]`、`[14400,33599]`、`[33600,110399]`，无重叠且
  完整覆盖 110,400 slot；
- vector address=`0x20000000+index*384`，beat address 再加 `beat*128`；最后合法 output byte 为
  `0x2286DFFF < 0x30000000`；
- 96 个 signed Acc24 小端占 bytes `0..287`，bytes `288..383` 强制为零；
- persistent-psum window 复算为 `110,400*384=42,393,600 B=0x0286E000`，exclusive limit
  `0x4286E000`、last byte `0x4286DFFF`。

### 4.4 decision gates

- mandatory output 已覆盖 per-sample/layer cycle、exclusive stall、weight/psum/source/output traffic、commit/
  directory/zero-build/refill/restore/writeback/capacity/hash；
- strict cycle conservation 与所有 function/conservation mismatch=0 为硬门；
- performance gate 是 fixed A1 ratio-of-sums `>=1.30x` 且每 sample `>=1.10x`；weight reads/refill bytes 不得
  增加；A1-OSG sequence 等价直接杀 performance novelty；
- performance 未过时只有同账本 psum read+write 降低 `>=30%` 才能保留为 decoder support/traffic ablation；
- CPU GO 也不授权 RTL，S10 不是 multi-sequence/full-network/system/headline。

## 5. 最终授权矩阵

- 本 r1 contract static PASS：**false**；
- author runner source / CPU launch / CPU analyzer：**false**；
- RTL / VCS / iverilog / Verilator / DC / PT / PTPX / Formality / training / GPU / remote：**false**；
- cycle / traffic / energy / PPA / system / paper headline claim：**false**；
- 合法下一步：只允许 author 一个修复后的 `run_authorized=false` contract r2 与新 handoff/request，并重新进行
  fresh independent static hammer。

本 hammer 只新增本 review 目录及双封文件；未修改被审合同、handoff、M534 r1--r4、任何 RTL、runner、
result、`docs/524` 或 `docs/359`。
