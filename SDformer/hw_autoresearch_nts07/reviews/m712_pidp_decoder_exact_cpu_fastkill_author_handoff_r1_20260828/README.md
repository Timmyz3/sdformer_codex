# M712｜PIDP decoder S3×10 exact CPU fast-kill author handoff

日期：2026-08-28  
状态：`AUTHOR_HANDOFF_KILL_FULL_PIDP__SELECTIVE_WEIGHT_FIT_DIAGNOSTIC_ONLY__FRESH_HAMMER_REQUIRED`

## 1. 结论先行

全 decoder 的 parity-indexed destination pull（PIDP）按冻结门槛为 **`KILL_NO_RTL`**。D0/D2/D3 exact-binary
headline 的固定 strongest A1 为 `A1-OSG`；ratio-of-sums `A1/PIDP = 0.497285529253x`，三个序列最小值
`0.495713994265x`。PIDP 即使免费消掉全部 materialized descriptor 和 psum SRAM bytes，也因 destination-major
顺序破坏 D0/D2 的 INT8 weight-tile 驻留而约慢两倍。performance gate 与 traffic-with-<=5%-cycle-regression gate
均失败，因此不授权 RTL/VCS/DC/PT/PTPX/Formality。

但本轮发现一个必须交 fresh hammer、不得直接升 claim 的新诊断：D3 的静态 weight working set 只有 13 个
tile，而同一 240 KiB 账下 PIDP 可容纳 16 个；D3 单层 `A1/PIDP = 2.178199251329x`。若只按**层静态尺寸**
选择 dataflow（D3=PIDP，D0/D1/D2=`A1-OSG`），整数行的非准入组合为 `1.474346419118x`，三个序列为
`1.473789914009x / 1.474046488751x / 1.475199126208x`。这只需要四个 layer config bits、没有
sample/sequence/runtime oracle；它仍只是 composition diagnostic，需新合同和 fresh hammer 后才能决定是否立
M718。

另做一个不改变 canonical result 的整数敏感性：把 PIDP 偏乐观的 `10 cycle/group` 提高到与 A1-OSG 相同的
`15 cycle/group`，D3 仍为 `1.544470913966x`，selective 组合仍为 `1.265320219224x`。这说明 D3 正方向并不
只来自 10-vs-15 的 group-service 优待；但它仍未收费真实 bitmap SRAM 端口、RTL 控制和物理 PPA，因此仍不
准入。

## 2. 第一性原理判定

ConvTranspose K3/S2/P1/OP1 的 exact contributor 不能消失。source-scatter 与 destination-pull 的根本差别是
数据复用对象：前者保权重、付 psum/descriptor；后者保单 destination Acc、付权重重装。M712 对 PIDP 使用了
明显偏乐观的下界：

- materialized descriptor bytes = 0；psum SRAM bytes = 0；
- `ceil(contributors/8)`，不收 bank conflict；
- 每 K8 group 只收 10 cycle，低于重建 A1-OSG 的 15 cycle；
- fully-associative LRU logical weight cache；不收 macro rounding、bank conflict、directory clear；
- 仍保持相同 96 lanes / K8 / 8 banks / Acc24 / 240 KiB / 128 B/cycle / dense 384-B commit。

在此候选有利条件下，D0/D1/D2 的 weight identities / cache entries 分别为 `384/16`、`98/16`、`25/16`，
PIDP weight refill bytes 分别达到 `1,765,978,343,424`、`1,485,899,928,576`、
`1,562,047,621,632`。D3 为 `13/16`，refill bytes 与 A1 同为 `53,913,600`，因此 dataflow 选择的分界恰好
由 working-set fit 决定，而不是 trace oracle。

PIDP 与 A1-OSG contributor multiset 相同，但执行序列不相同：PIDP 是 destination-major、无 scatter
descriptor；A1-OSG 是 source-major、descriptor join。它不是 A1-OSG 的逐字重命名；不过它也不是新算术，
而是经典 destination-stationary / weight-stationary tradeoff 在 H67 decoder 上的对象迁移。若未来保留，论文
必须引用 transposed-convolution polyphase、SCNN/ELSA/OpenEye，claim 只能落在 H67 的 static weight-fit
dataflow selection 与 typed signed K8/Acc24 protocol 上。

## 3. exactness 与 population

- M699/M705 exact S3×10：`interlaken_01_a`、`thun_01_b`、`zurich_city_12_a`，每序列 10 sample；
- 120 records、1200 record-timestep rows；D0/D2/D3 headline 900 rows，D1 diagnostic 300 rows；
- topology mismatch = 0；contributor multiset mismatch = 0；
- 9600 个 deterministic signed-INT8 Acc24 probes，oracle mismatch = 0；
- D1 使用 sealed exact `{0,theta}` mask 与 folded-theta FP32 weight 仅做 diagnostic；
- probe INT8 是从 M686 sealed FP32 weight 本地生成的 per-output-channel symmetric `rint/clip127`，只证明
  schedule equivalence，不是 checkpoint numerical admission，也不是 accuracy/quantization claim。

## 4. headline ledger

| 范围 | A1 cycles | PIDP cycles | A1/PIDP | PIDP weight refill bytes |
|---|---:|---:|---:|---:|
| D0 | 4,305,988,872 | 19,688,833,060 | 0.218702086552× | 1,765,978,343,424 |
| D2 | 4,439,367,778 | 17,819,260,790 | 0.249133105482× | 1,562,047,621,632 |
| D3 | 12,837,749,400 | 5,893,744,290 | 2.178199251329× | 53,913,600 |
| D0+D2+D3 | 21,583,106,050 | 43,401,838,140 | 0.497285529253× | 3,328,079,878,656 |

全 PIDP 的 psum+descriptor logical bytes 从 `549,590,729,344` 降到 0，但 cycle 回退约 101%，不满足
“cycle 回退不超过 5%”的 traffic gate。bitmap probe bytes 另收 `2,621,750,400`，没有藏进 0-byte 叙事。

## 5. 身份和 claim boundary

- canonical result report SHA256：`228ad20d34603a21903a34dd14be8306f18ca43c297cb3263f8b8653524cf20f`；
- result manifest SHA256：`00f042b35b64f242b5c4a19ee24fb36f9b5a8a31999d714919c61c70b727330e`；
- result outer-seal file SHA256：`f15c6d45e41e81b623982deda94c5b52f7213417f639f53d9116c457aca49806`；
- contract SHA256：`5c11add1b92dceab9fe09d22234545172ba58de74f221c29ee88688b248f3bf2`；
- analyzer SHA256：`87e559a1d249a9aacec31763c692a0da9e312bd753f11c63241b765fca16dbbc`；
- runner SHA256：`75f0abe95e7732b9dfa8d0dc10fb396e9e6d25d3e4f23f45063a1d4a8f46988b`；
- tests SHA256：`fb8e22d5afa15aece67cc2287a23308087df029775278552c9b64becb22652bf`；
- docs/359 SHA256：`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

本结果是 deterministic optimistic CPU fast-kill ledger，不是已失败 M590/PBR4 全 FSM 的替代品，也不是
cycle-accurate RTL/system simulator、PPA、energy、
accuracy、full-decoder admitted speedup 或 paper headline。全 PIDP 只能写负结果；selective 1.474× 只能写
“待独立审阅的静态组合诊断”。唯一下一步是 fresh independent result hammer；hammer 通过后再决定是否为
weight-fit selective PIDP 建新 M718 contract，禁止直接开 RTL。
