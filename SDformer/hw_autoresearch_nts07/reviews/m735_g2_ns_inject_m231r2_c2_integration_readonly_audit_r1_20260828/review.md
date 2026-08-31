# M735｜G2 / NS-INJECT 与 M231r2/C2 的只读第一性原理审计

日期：2026-08-28  
模式：read-only evidence audit；未运行 EDA/GPU，未修改作者 RTL、合同、结果或 `docs/359`  
裁决：**GO_CPU_RECURRENCE_ONLY_AS_C2_INTEGRATION_CLOSURE；NO-GO_STANDALONE_NOVELTY；NO-GO_NEW_RTL_BEFORE_FASTKILL**

## 1. 结论先行

Grok G2 的问题定位有价值，但“模拟 ATLIF 值发完即注入 C2”的对象判断不适用于冻结的 H67 ep35 deploy。正式配置
`system_handoff/received/h67_ep35_system_trace_handoff_20260821/h67_ep35_system_trace_handoff_20260821/config/deploy_q7q17.yml`
的 SHA256 是 `8be3f7bb...e6c49`，其中顶层和 `all_non_qk_binary_atlif` 均为
`output_mode: binary`、`threshold_mode: official_atlif`，后者用
`path_selection: all_non_qk` 覆盖 FFN sn2。冻结软件的
`OfficialATLIFSurrogate.forward` 返回 `out * thre`，而 M231 checkpoint screen 已逐项证明
ep35 的 12 个 FFN sn2 threshold 都是 scalar FP32 精确 `1.0`（little-endian
`0000803f`）。因此 sn2→fc2 边界的合法值是精确 `{0,1}`，不是任意 analog/float
descriptor。

M231r2 已经实现了 G2 所要求的核心“事件不落完整 feature map”桥接思想：把 ATLIF 的
`2 time rows × 16 channels` 事件字转成 M216 的 `raw4 × 96-bit` 输入；四个宽度均有
VCS 与 logic-only DC。但它没有与正式 Fixed-T10 producer、M216/M218/C2、有限 FIFO 和
120-record 顺序 trace 闭合。它只能作为 C2 的 integration/support module，不能作为第四个
novelty。

更关键的是：按现有 C2 L4/O8/II1 K8 service 的 `515,449,096` cycles，哪怕把冻结
S10 的 `875,520,000 B` binary activation write+read 全部、悲观地串行收费，32/64/128
B/cycle 下可消掉的局部周期上界也只有 `1.05308× / 1.02654× / 1.01327×`。因此
`≥1.15×` 的局部性能门从第一性原理上已不可能通过；CPU recurrence 值得做的是闭合
address/stall/traffic 表，不是寻找 headline speedup。

## 2. 冻结对象与数据粒度

| 对象 | 冻结事实 | 可引用边界 |
|---|---:|---|
| H67 deploy config | SHA `8be3f7bb...e6c49`；all-non-Q/K ATLIF=`binary/official_atlif` | 正式 deploy 语义 |
| FFN sn2 threshold | 12/12 为 scalar FP32 exact 1.0 | ep35 checkpoint-bound `{0,1}` |
| M51 FC2 payload | 120 records，10 samples，5,580,000 T-token，143,894,510 events，437,760,000 packed bytes | post-sn2 binary inputs；无 producer cycle/output claim |
| M51 spatial-token population | 558,000（T=10） | 可重建每个 ATLIF token 的 10 time rows |
| Fixed-T10 16-channel tiles | 21,888,000；17 issue cycles/tile → 372,096,000 steady issue-cycle work | 由 M518 RTL 锚点和 M51 shape 推导的 CPU schedule；不是实测全链周期 |
| 显式 materialization | S10 write+read=875,520,000 B；87,552,000 B/sample | 只允许叫 on-chip packed-activation traffic ablation |
| FC2 weight traffic | S10=39,638,437,824 B | 来自 M218 frozen-work service model |
| materialization 占 FC2 weight+activation traffic | 2.161033% | 不含其他系统流量，不能叫 DRAM/system reduction |

M51 `manifest.json` 的 120 个 FC2 row 都含
`sample_id/frozen_execution_call_index/name/input_shape/output_shape/relative_path/file_sha256`
并且对应 bitpack 在
`system_handoff/incoming/m51_capture_bundle_r2_20260823/calls/` 存在。因此一天内做
CPU address-timed recurrence 在数据层面可行，不需要 GPU。

## 3. 已闭合的 RTL/VCS/DC 与仍未闭合的边界

| 层 | 已闭合 | 未闭合 |
|---|---|---|
| M518 Fixed-T10 producer | r11 receipt-blind hammer：VCS compile/sim 0，numeric mismatch 0，96 multipliers，17 issue cycles/tile，5 result beats/tile | 无 checkpoint/real-trace producer replay；DC/PPA/energy 未准入；result tag 未冻结为 `{module, token, hidden_group}` |
| M231r2 transpose bridge island | 四宽度 VCS：每宽 3 pairs/6 tokens，ordinary+same-cycle fault attacks，0 accept/commit on fault；3 ns full-flop DC area为 5,690.916/10,641.456/20,803.482/40,851.216 um2 | 没有 M518→M231 producer adapter、120-record finite recurrence、SRAM、Formality/PT/PTPX；W3072 hold 仅 0.0000 ns |
| M216 sparse frontend | 120-record CPU replay；VCS；logic-only K1/K8 DC 20,436.696/20,587.392 um2；standalone always-ready frontend 4.764209× K8/K1 | 不含 weight SRAM、Acc24、response latency/commit；不是 complete FC2 或 same-bandwidth physical speedup |
| M218 service | directed VCS；3 ns logic-only DC 88,851.042 um2，setup +0.6872 ns/hold 0.0000 ns；18,432 Acc24 context bits | 0 macro；context 全为 FF；无 connected trace/PPA/Formality/SAIF/PTPX |
| M342/M519 connected C2 | M342 synthetic raw4→Acc24 VCS；M519 registered-release VCS。M519 equal-bandwidth K8/K1×8 仅 1.012–1.039× | 120-record producer→consumer 全链、M519三轴 DC、macro/energy 尚未闭合；不得用 M342 5.28× 作为同带宽结果 |
| M480 BN replay | 强基线已经定义为 raw capture + current-batch barrier + raw replay + normalized value direct-consume；显式 normalized tensor materialization 是弱基线 | fixed-point affine miter、address-bearing SRAM、宏与能量未闭合 |

### 3.1 当前最硬的 producer-order 缺口

M518 对一个 16-channel hidden group 在一个 tile 内按 beat 0..4 连续输出 5 个 time-pair；
M231 却要求先开一个 pair header，再按 `event_group_index=0..INPUT_WIDTH/16-1`
收齐该 time-pair 的全层宽度。这是 **group-major producer → pair-major consumer** 的
转置，不是直接连线。

现有 M231 两个 pair slot 总容量只有 `4W` bits：W=384/768/1536/3072 时分别为
192/384/768/1536 B。串行复用一个 M518 时，至少要同时保留五个 time-pair，单 token
为 `10W` bits，即 480/960/1920/3840 B；若要 producer/consumer ping-pong，则为
`20W` bits，即 960/1920/3840/7680 B。当前 M231 的 `4W` 容量无法承接正式 M518
输出顺序。

这也是为什么 M231r2 独立评审仍保留 P0：typed producer metadata、有限两槽 recurrence
和 120-record ordered execution 没有闭合。

### 3.2 动态 BN 边界

G2 不能跨 BN1/current-batch barrier。它只能位于 BN replay/`sn2` 之后、`fc2` 之前。
M480 已证明“不物化 normalized intermediate”是强基线卫生，不是新 accelerator novelty；
sn2→fc2 没有语义 barrier，因此 direct consume 更应视为强基线。论文若报告
875.52 MB，只能写成“相对显式 binary feature-map materialization 的 traffic ablation”，
并同时给出相对 fused-direct baseline 的性能比 `1.0×`。

## 4. 一天内最小 CPU integration / fastkill spec

名称建议：`M735R2 Fixed-T10→token-transpose→C2 address-timed recurrence`。不开发 RTL，
不运行 EDA/GPU。

### 4.1 输入与顺序

1. 只接受本 memo JSON 中冻结的 config、M51 manifest/310 个 payload SHA、M518 r11
   verdict、M231r2 review、M216/M218 admission 和 M519 VCS receipt。
2. record 顺序固定为 `(sample_id, frozen_execution_call_index)`；FC2 仅选 120 rows。
3. 每 record 的空间 token 顺序固定为 `(batch, y, x)`；每 token 的 hidden group 为
   `g=0..W/16-1`；每 group 的 M518 结果 beat 为 `p=0..4`，对应 time rows
   `(2p, 2p+1)`。
4. bit 值直接来自 M51 post-sn2 bitpack；不得声称这是 M518 对 pre-sn2 input 的 numeric
   replay。M518 只提供固定 schedule anchor。

### 4.2 producer 与转置状态

1. M518 steady tile service 使用 17 cycles；首次/短 burst 另用已准入 N1=29、N4=80
   anchors校验 recurrence。
2. producer result metadata 必须显式为
   `{sample, module/weight_set, spatial_token, hidden_group, pair_beat, valid32, last_group}`；
   不得只复用一个模糊 tag。
3. primary candidate 使用两个 token banks，每 bank `10W` bits，合计 `20W`；每个 bank
   为 5 个 pair 分别维护 group-valid bitmap 和 terminal。只有一个 pair 的全部 hidden
   group 到齐后才允许向 C2 发对应两条 time-row token。
4. M231 协议映射保持 header 后 raw4；一条 time-row 的 raw packet 数为 `W/384`，层宽
   384/768/1536/3072 分别为 1/2/4/8。顺序固定 pair0 row0,row1 ... pair4 row0,row1。
5. 计数 producer stall、result FIFO occupancy、token-bank full、pair-ready wait、raw4 stall、
   C2 group/request/response/context/result/done stall；任何丢重序、tag/weight-set mismatch 或
   deadlock 都 fail closed。

### 4.3 公平 baseline

必须同时报告两个 baseline：

- **B0 fused-direct strong baseline**：与 candidate 相同的 M518、`20W` token transpose、
  M216/C2，pair ready 后直接送 C2。这和 candidate 语义等价，性能预期严格 `1.0×`；
  它决定 G2 不能单列 novelty。
- **B1 explicit-materialized diagnostic**：仍使用相同 M518、相同 `20W` transpose 和相同
  C2，只在 pair ready 后把每条 binary row 写入 byte-addressed activation SRAM，并从
  同一 SRAM 读回再送 raw4。主点为独立 activation 1R1W、32/64/128 B/cycle、read-before-write
  已定义、有限 token-local capacity；允许跨 token overlap。不得假设整张 feature map
  强制落片外，也不得与 weight SRAM 偷换成同一个 port。

B1 的显式 materialization 只用于回答“省了多少中间流量”；论文主公平基线仍是 B0。
若另做 shared-on-chip-bandwidth 敏感性，必须单列，不能替代 B0。

### 4.4 门槛与必报数字

- exactness：120/120 record SHA、5,580,000 T-token、143,894,510 events 全等；candidate、
  B0、B1 的 C2 source tuple/result work 完全一致。
- performance gate：candidate/B1 在同资源下局部 `>=1.15×` 才允许提性能机制。根据现有
  L4/O8/II1 service，此门已经被 `<=1.05308×@32 B/cycle` 的悲观全串行上界否决；
  因此预期结果是 NO-GO performance。
- traffic gate：相对 B1，packed activation write+read 必须精确减少 875,520,000 B
  （100% activation-materialization traffic）；同时必须报告相对 FC2
  weight+activation traffic 仅 2.161033%。不得写成全网/DRAM traffic reduction。
- storage gate：W3072 的双 token transpose 必须显式收费 7,680 B；所有 metadata/FIFO
  另计。不得继续引用 M231 的 1,536 B 作为 composite 最大存储。
- claim boundary：CPU model、无损、binary、C2 support、非 novelty、非 RTL/PPA/energy/
  system/headline。

## 5. 论文落点

可写进 C2 的一句话：

> We fuse the checkpoint-bound binary Fixed-T10 neuron output into the typed
> raw4/K8 FC2 ingress through a finite token-local time/channel transpose,
> avoiding an explicit packed activation write/read; this is reported as an
> integration traffic ablation, not as an independent sparsity speedup.

不能写：analog ATLIF descriptor、floating-point firing injection、跨 BN fusion、完整 FFN
speedup、875.52 MB DRAM reduction、或新的 neuron-synapse fusion first/novel claim。

## 6. 最终裁决

G2 应从“新硬件创新候选”降为“C2 的必要 producer integration 与 traffic ablation”。
一天 CPU recurrence 可执行，但不应等结果才做是否开 RTL 的决定：现有 service/traffic
下界已经证明 `1.15×` 性能门不可能。只有当 recurrence 发现当前 C2 实际受 activation
port 而非 weight/context service 限制、并在 B0 强基线下出现新的非等价资源约束时，才允许
重新立项；否则不开发新 G2 RTL，直接在论文中作为 C2 完整性和负/支撑消融收口。

`docs/359_DATE终局冻结_20260813.md` SHA256 复核保持
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
