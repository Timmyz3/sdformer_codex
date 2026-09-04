# M766 decoder-complete address-timed cycle closure：只读审计

## 裁决

**GO_ONE_DAY_DECODER_COMPONENT_ADDRESS_TIMED_PLAN；Table-A 与 full-network 继续 fail-closed。**

M624 的旧结论需要拆成两部分理解：后续 M686 已补齐原 `zurich_city_09_a` 十样本的 D0--D3 decoder 输入与权重身份，M699/M705 又提供了三序列三十样本的 decoder payload；因此“本地没有 ConvTranspose payload”已经不再成立。但当前仍没有一条地址时序 SRAM 请求链、共同资源配置、D1 checkpoint 数值准入和同人口 full-network 非 decoder 补包，所以任何 decoder/system cycle、speedup 或 Table-A production row 仍然不得生成。

最小正确动作不是扩写旧 M22/M23，也不是把 M712/M722 的闭式 ledger 改名为周期仿真，而是新增一个隔离的 decoder component analyzer：以 M686 为主人口，以 M699 为跨序列副人口，复用 M672 的 exact ConvTranspose mapper、M712/M722 的 contributor/存储账本和 M218 的 request/response recurrence，但重新生成带地址、依赖、端口、bank、issue/return/commit 时间戳的统一事务。它必须在相同 96 lane、245,760 B、Acc24、3 ns、192 B/cycle 和相同 commit 序列下比较 A1-OSG、equal-service K1x8 与 typed K8。

## 数据质量与身份

| 数据 | 当前可用性 | 可用于什么 | 禁止用途 |
|---|---|---|---|
| M686R6，manifest `c06de650...` | 10 个 `zurich_city_09_a` 样本，40 records；D0/D2/D3 exact binary，D1 exact scaled-binary；M692 100/100 | canonical decoder-local 地址时序主人口；与旧 M51 样本键一致 | D1 checkpoint 数值等价；full-network 完整性 |
| M699，manifest `e2d7c92...` | 3 sequences × 10 samples × 4 layers；M705 98/100 | 多序列 decoder-component 稳定性 | 与 M624 的 `zurich_city_09_a` full-network trace 拼接 |
| M700 official Prosperity | D0/D2/D3 official product-vs-bit = 3.087586x；M739 99/100 | Table-C 外部 artifact 对标 | `ours`、同资源 local、monolithic、decoder-complete 或 system speedup |
| M712/M722R2 | contributor、traffic、storage、Acc24/stripe 与 closed-form service ledger | analyzer 单元 oracle 与资源预检 | address-timed/physical cycle 证据 |
| M522/M523 | exact coordinate mapper logic 与 8-lane transport bundler VCS/DC support | 协议、tap 顺序与逻辑成本旁证 | 把 8 transport lanes 当 M218 的 8 weight banks；性能人口 |

M705 的密度跨三条所选序列很稳定，但该事实只支持 decoder payload 的 observed robustness。它不能修补 M624 仍缺的 150 个非 decoder payload，也不能把不同 sequence population 拼成一帧全网。

## 当前阻塞项

### P1-1：不存在地址时序执行语义

M712/M722 使用聚合项（scan、bundle、`groups*15`、refill、commit）和逻辑 cache；没有逐请求地址、端口占用、bank conflict、response dependency、compute/memory overlap 或完成时刻。M22/M23 有事务字段和 bank-count helper，但其旧产物是约 596 MB boundary-materialized envelope，且明确不是 compute-overlap cycle simulator。只能复用 schema/helper，不能延伸旧结果。

### P1-2：共同资源坐标尚未冻结

主表共同坐标应遵循 M527：96 lane、240 KiB、Acc24、3 ns、64 GB/s = 192 B/cycle。M712/M722 的局部账本使用 128 B/cycle 和 41-cycle commit，不能直接继承进 Table-A。还缺 SRAM bank 数、bank function、port mode、read latency、outstanding/II、row/alignment、weight/psum/control 分区和宏 round-up。D3 的 A1 plan 为 243,200 B，只剩 2,560 B，descriptor/FIFO/metadata 或宏 round-up 任一超额都必须 fail-close，不能借用未计费容量。

### P1-3：D1 仅可 diagnostic

M686/M699 的 D1 输入是 runtime-theta exact scaled binary，但 folded-theta weight 并未获得 checkpoint numeric/accuracy admission。D0/D2/D3 可形成 exact-support component row；若 D1 没有 bit-exact fallback 或经批准的数值路径，`decoder_complete` 必须为 false。

### P1-4：尚不能形成 full-network 同人口行

M686 与 M624/M51 原十样本同人口，可用于 decoder-local 主人口；但旧 M51 仍缺 150/310 个非 decoder payload，旧 ordered trace 也没有 ConvTranspose ordinal。M699 是不同的三序列人口。缺失项关闭前，local decoder cycles 不得写入 M628 Table-A production rows，也不得与 C1/C2/C3 component 数字相加或相乘。

### P2 边界

- M523 的 K8 是 transport bundling，不含 flattened `(source_channel,kernel_index)` weight key、bank mapping、conflict deferral 或 stored-weight identity。
- M700 的 3.087586x 只描述官方 Prosperity CPU simulator 上的 support opportunity；官方 weight values 未被消费。

## 最少代码方案

只新增一套隔离实现，不改 RTL：

1. `contracts/m767_h67_decoder_a1_k8_address_timed_cycle_contract_r1_20260828.json`：冻结输入 SHA、population、common resource、A1/K1x8/K8 配置、fallback 和 fail-closed gate。
2. `system_simulator/scripts/analyze_m767_h67_decoder_a1_k8_address_timed_cycles.py`：唯一 production analyzer。M686 为 primary，M699 为 secondary；禁止读取 M700 作为候选输入。
3. `system_simulator/tests/test_m767_h67_decoder_a1_k8_address_timed_cycles.py`：小型 synthetic bank-conflict、same-cycle release、1RW/1R1W、容量 cliff、same-commit-hash 与 population-mixing 攻击。
4. exact-SHA runner、result/receipt、独立 fresh hammer。只有 hammer PASS 后才允许 component 表引用。

analyzer 可直接复用但必须重新计时的部分：

- M672R3：`iter_polyphase_tiles`、tap/phase/destination exact mapper 和 reconstruction oracle。
- M712：payload seal/unpack、runtime topology、descriptor counts、contributor multiset。
- M722R2：A1 stripe/storage plan、psum traffic 和 Acc24/order-independent bound。
- M218：fixed-latency request/response recurrence，仅作为单元 oracle；真实总周期由地址/端口 scheduler 给出。
- M22/M23：事务列名、`consecutive_bank_counts`/`cyclic_bank_counts` 思路；不得引用旧 M22/M23 数值。

不要把数亿次请求全部展开成 CSV。可用“连续地址段 + count + bank pattern + dependency token”的压缩事务行，但离散事件 scheduler 必须逐服务槽推进并输出 start/end；同时保存展开基数、地址 checksum、commit-address hash，使压缩前后守恒可审计。

## 必须冻结的共同资源

在运行前先生成并双封一个 common-resource manifest：

- 96 product lanes，Acc24，3.000 ns；primary external bandwidth 192 B/cycle。128 B/cycle 仅可作为独立 sensitivity，不能替代主点。
- 245,760 B 为所有 on-chip SRAM 的 macro-rounded 总和，不是 payload-only logical bytes。
- 明确 weight、psum、descriptor/control 的 bank 数、row width、地址映射、1RW/1R1W、read latency、write latency、II 和 outstanding 上限。
- A1-OSG、K1x8 和 typed K8 保留同一物理分区与端口；未使用容量仍保留，不能让候选独享额外 bank。
- 三种模式发出完全相同的 dense commit address/value-width 序列；`commit_sequence_sha256` 必须相同。若 commit 与 product 可重叠，三种模式应用同一依赖规则，不能把 41 cycles 作为无条件独立常数追加。
- typed K8 的 headline 对照为 equal-service K1x8；K8 对单 K1 的约 4.76x 不得作为同资源优势。

## 输出与 Table-A 接口

component result 至少输出：

- 每 population/sequence/sample/module/timestep/config 的 cycles；ratio-of-sums 为 primary，同时报 geometric/arithmetic mean、min/max。
- DRAM/SRAM read/write bytes，physical issued sources/cycle，retired destinations/cycle。
- mutually-exclusive stall：compute、weight-bank、psum-bank、memory、dependency/completion；总和与 total cycles 守恒。
- resource manifest SHA、population/workload SHA、transaction count/address hash、commit hash、fallback count、capacity peak。
- D0/D2/D3 exact-support aggregate；D1 单列 diagnostic，除非数值准入或 exact fallback 已闭合。

可生成一个 `table_a_feed` 对象，但当前必须固定：

```text
measurement_class = DIRECT_DECODER_COMPONENT_ADDRESS_TIMED_SIM
decoder_complete = false          # D1 未准入时
full_network_completion = false
logic_sram_dram_energy_closed = false
logic_macro_area_closed = false
sta_closed = false
table_a_insertion_allowed = false
```

字段应与 M628 对齐：`row_id, role, fidelity, cycles, energy_mj, area_mm2, accuracy, source_id, measurement_class, population_id, workload_id, resource_manifest_sha256, completion_receipt_sha256, decoder_complete, memory_timing_included, full_network_completion, logic_sram_dram_energy_closed, logic_macro_area_closed, sta_closed, independent_hammer_pass, blockers`。只有后续同人口 full-network completion、M527 五档配置 manifest、能量/面积/STA 和独立 hammer 全闭合，才可进入 Table-A production row。

## Fail-closed gates

1. `docs/359` 必须保持 `dedde7ce...`；M686/M692 或 M699/M705 的 member/outer seal 逐一通过。
2. primary population 固定为 M686 的 10×4 records；secondary 固定为 M699 的 3×10×4，不得混合。任何 sample/module/timestep/multiset mismatch → cycles/speedup null。
3. D0/D2/D3 route 必须 exact binary；D1 未通过 checkpoint numeric admission时只能 diagnostic 或收费 fallback，不能置 `decoder_complete=true`。
4. common resource 必须 macro-rounded ≤245,760 B；D3 2,560 B 余量不足时直接 fail，不得隐式外借容量。
5. 请求必须有显式 address、bank、issue、return、dependency、commit；各 stall 类互斥且守恒。聚合 closed-form ledger 不得冒充 address-timed。
6. A1/K1x8/K8 的 commit hash、资源 tuple 和 fallback policy必须相同；K8 只与 equal-service K1x8作性能比较。
7. local component 永远保持 `full_network_completion=false`；M51 150 个缺失 payload、全网 ordinal 与共同 completion 未补齐前，Table-A insertion=false。
8. M700 始终 `external_artifact_only=true`，不得进入 ours numerator/denominator 或与本地倍率相乘。

## 一日内执行清单

| 时段 | 动作 | 出口门 |
|---|---|---|
| 0--1 h | 冻结 M767 contract 与 common-resource manifest；主 M686、副 M699；明确 192 B/cycle | SHA 双封、容量含 metadata/macro round-up |
| 1--3 h | 写 payload normalizer、exact mapper adapter、压缩事务生成器与小型 unit tests | exact multiset/address/commit hash 0 mismatch |
| 3--5 h | 写 bank/port discrete-event scheduler；用 M218 recurrence 作单元 oracle | issue-return-commit dependency 与 stall 守恒 |
| 5--7 h | 先跑 M686 D0/D2/D3；D1 仅 diagnostic/fallback；比较 A1/K1x8/K8 | 同资源、same commit、capacity gate 全过 |
| 7--9 h | 跑 M699 S3×10 副人口并输出 sequence 分层 | 不与 M686 拼成系统；每序列结果齐全 |
| 9--12 h | 封 result/receipt，独立 fresh hammer | PASS 后仅升格 decoder-component；Table-A 仍 false |

优先级：**P0 是 M686 原人口的同资源 address-timed closure 和 D1 边界；P1 是 M699 多序列复跑；P2 才是把 decoder component 接回 full-network。** 在 M51 缺失和共同配置 registry 没闭合前，不应等待或运行任何重型 EDA。

## 冻结身份

- `docs/359`: `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`
- M624 contract/result: `ac09c044...` / `d213e9ed...`
- M686/M699 manifests: `c06de650...` / `e2d7c92...`
- M700/M712/M722R2 results: `76c8722f...` / `228ad20d...` / `363f319d...`
- M672R3/M218/M22/M23 scripts: `989094c7...` / `93d33317...` / `0e7f4a21...` / `22b2a022...`

本审计没有运行 simulator、GPU、VCS、DC/PT 或 remote job，没有修改 RTL，也没有修改 `docs/359`。
