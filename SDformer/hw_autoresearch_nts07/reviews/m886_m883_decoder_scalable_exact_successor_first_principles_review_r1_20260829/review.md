# M886｜M883 decoder 可扩展 exact successor 第一性原理审阅

结论：**GO 到一份新的 source-only candidate；NO-GO 直接扩展 M861 全量，NO-GO 用抽样冒充 decoder-complete。** M883 已经把 M861 的正确性闭合到一个完整 D0/A1/t0 行，但该实现的成本仍是生产不可用的：38,672,612 个 expanded request 得到 20,548,766 diagnostic cycle，耗时 932.078357 s、峰值 8,897,128 KiB（8.485 GiB）。若把该行机械外推到冻结的 160 records × 10 timesteps × 3 configs = 4,800 个独立冷启动行，单核“同尺寸行”等价投影为 51.78 天；这不是生产时间预测，只用于说明不能启动旧全量。

本审阅严格绑定 M883：review `ae443b36084a3361548ec6a950dbc0a962cf60ec650000c9638db61854c02f88`，manifest `3cdd7be9cde8177e4cce6dfd16fc42dda5a84ba729757c92638eb242fe6fed0d`，outer `4ddece71698ee0b83c18d039eb34205a0f2c93b4e5b95fd349f011686ab8d5a1`。`docs/359` 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## 1. 成本根因，不是模型下界

M861 已经去掉 `scheduled_requests` 和 `compressed_schedule` 两个完整列表，但仍永久保存每个 produced token。M883 的 `token_ready_entries = expanded_request_count = 38,672,612` 是直接证据。冻结的 M785 生成器只把 compressed transaction 的 terminal token 作为后继依赖；同一 transaction 的非 terminal token 不被消费。长期保存所有 expanded token 因而是实现选择，不是 M768 依赖语义要求。

第二个根因是 active-service 集合的表示。单行有 15,779,614 次乱序区间插入，最终只剩 526,732 个合并区间。对 Python list 做 bisect/merge 的代价与“周期数”无关，是数据结构税。第三个根因是对 38.67M 个请求逐个构造 dataclass、`json.dumps` 并更新 expanded-address SHA。地址 SHA 是审计表示，不影响 issue/return/commit 方程。

单行还表明，dependency-completion 在 priority 前覆盖 `[0, 20,548,765)`，但这只能解释该行，**不得**固化成全 decoder 假设；其他 module/config/timestep 仍必须精确计算六类优先级。

## 2. 建议 successor：GTLS（group/transaction-level exact scheduler）

建议新建 M887 source candidate，直接消费冻结 M785 `CompressedTransaction`/贡献组，而不是先扩成 `Request` 对象。它必须保持 M785/M768 的资源 tuple、request 顺序、地址、bank、依赖、same-cycle response-slot reuse、cold-start/drain 和六类优先级不变。

### 2.1 terminal-token liveness（性能工程，不改模型）

前端为每个 compressed transaction 发出 terminal token 和显式 last-use 标记。非 terminal token 仍是逻辑 Request 字段，但不进入 readiness map，因为冻结生成 grammar 中没有消费者；terminal token 在最后一个 consumer 被调度后立即释放。必须静态和运行时同时证明：每个 dependency 都引用已经产生的 terminal token、每个释放后 token 不再出现、长寿命 `source_done`、weight-ready/eviction、psum-chain/commit 四类路径均闭合。

预期 live set 由 source token、九个 weight slot 的 ready/last-use、在片 psum/待 commit vector 和当前 group 链决定，而不是 38.67M。这个变化只缩减主机 simulator 状态，不给被建模硬件减 SRAM，也不能算 accelerator speedup。

### 2.2 compressed-transaction closed form（性能工程，不改模型）

一个 transaction 内的 requests 共享 resource、banks、width、dependency 和 earliest cycle。先逐项处理至多一个有限 boundary（清空进入 transaction 前的不对称 bank/outstanding 状态），随后用冻结 recurrence 的精确 affine/periodic fast-forward：

`i_j = max(i_(j-1)+service_step, i_(j-q)+return_distance, dependency, earliest)`。

这里 `service_step=max(initiation_interval,beats)`，`q=outstanding_per_bank`；同拍 return slot 可复用必须保持 `return > candidate` 的原判断。任何不能证明进入稳态的 transaction 都退回逐 request reference path，不能猜测。

### 2.3 packed event aggregation（性能工程，不改模型）

active issue 用 packed bitset/strided-run 写入，等待与 inflight 用半开区间 run；finalize 时按 `active > dependency/inflight > weight > psum > memory > compute` 做精确 union/popcount。禁止继续对 15.8M 个乱序点做 Python list 插入。建议核心用编译型 C++/Rust 或 NumPy/C-level kernel；这不是开源 EDA，也不改变周期模型。

### 2.4 contributor/group IR（性能工程，不改模型）

同一 record/timestep 的 bitpack 解码和 destination contributor map 在三个 config 间相同，应只生成一次 sealed packed IR。A1、equal-service K1×8、typed K8 从同一 IR 作配置专属 grouping。IR 至少含 destination boundaries、`flat_k`、source index、module/timestep identity，并绑定 payload SHA、M672/M712/M722/M785 oracle 身份。weight LRU、dirty psum residency、refill/evict/restore/commit 仍在线执行，不能被 contributor count 公式替代。

### 2.5 hash 边界（审计表示变化，不改周期模型）

逐 expanded-request JSON SHA 是主要主机开销之一。生产 fast path 可改报 lossless compressed/group-IR SHA，加 frozen expansion function SHA 和 expanded count；这是**审计 schema 变化**，不是周期模型变化。旧 `transaction_address_sha256` 只能在小前缀 reference miter 中继续逐字节比较。论文周期准入依赖 exact schedule miter 与 sealed input，不把新旧 digest 名称混为一谈。

## 3. 100× 闸门

新 candidate 不因“理论上更快”获准全量。它必须在相同主机、相同 D0/A1/t0、相同冷启动口径下达到：

- total cycle = 20,548,766；expanded = 38,672,612；compressed = 9,582,057；
- cycle classes 精确为 active 18,502,452、dependency 2,046,313、compute 1、其余三类 0；
- commit SHA 仍为 `aa69b355efd62b428e2909ee4c1dbecdf34ec3e1e8681b0c78ace19a444ff861`；
- 端到端 wall time ≤ 9.320783571 s，才叫相对 M883 至少 100×；不能只报 scheduler kernel、排除 payload/oracle/IR 构造；
- peak RSS ≤ 512 MiB，且 `live_token_peak`、event-bitset bytes、IR bytes 分项披露；
- 若 wall time 未过 100×，可以继续作性能工程，但不得释放 4,800-row production。

9.32 s/row 的同尺寸机械投影仍约 12.43 h 单核，因此过门后还要做确定性 row sharding。原生产每个 record/timestep/config 都新建 scheduler，无跨行状态，故以 `(population, config, ordered-record, timestep)` 分片并在 merge 时做 4,800-key 完整性检查是 exact 性能工程；不得引入 warm start 或跨记录 overlap。

## 4. 小前缀 exact miter

source hammer 必须执行但限制在前缀：

1. synthetic 1K/10K 与 M861 比全部旧字段；real D0/A1/t0 1K/10K/100K 比 total、expanded/compressed、六类、commit/address SHA、port calendars 和 terminal readiness；
2. adversarial cases 覆盖 1RW/1R1W、`count={1,q-1,q,q+1}`、latency/beat 边界、多 bank 初态不对称、资源交替导致 issue 回跳、same-cycle slot reuse；
3. liveness attacks 覆盖多 consumer、长寿命 source token、weight victim 最后使用、dirty psum restore、commit 前最后依赖，以及 release 后恶意复用必须 fail closed；
4. closed-form 与逐 request 路径逐 endpoint 比较，不只比较总 cycle；任何 fast-forward 不可证明时必须 fallback；
5. 小前缀继续验证旧 expanded-address SHA；full fast path 只允许新 compressed-IR SHA，字段名必须不同。

## 5. full population 的论文准入

只有以下条件同时满足，才可写“decoder-complete address-timed cycle simulation”：

- M686 40 + M699 120 records 分开，3 configs × T10，共 4,800 个唯一 row key，0 missing、0 duplicate；
- D0/D2/D3 headline，D1 common charged diagnostic 且不进 headline ratio；所有 row 冷启动和 drain；
- 96 lane、245,760 B macro-rounded、Acc24、3 ns、192 B/cycle 与 M785 完全相同；
- legal headline 只比较 typed K8 与 equal-service K1×8；A1 只作诊断；
- 每个 shard 原子发布、各自双封，merge 按冻结顺序重算 population/config/module totals 与 paired ratio；
- 随机抽取每个 `(population,module,config)` 至少一个 row 回退到逐 request prefix miter；边界 module/config 必须覆盖；
- fresh independent result hammer 后才能进 Table A；此前 decoder-complete、production cycle、speedup、system 都为 false。

## 6. sampling 的合法位置

若 exact 4,800-row 仍超预算，可预注册按 population/sequence/module/timestep 的 paired stratified sample，对同一 row 同时跑 K1×8/K8，报告 design-weighted mean 与 95% CI。它可以成为“sampled cycle-simulation estimate”，但必须同时给 N、seed、stratum weight、finite-population correction/paired bootstrap 方法和最大 CI 半宽。它不改变单行周期模型，却改变覆盖范围，**永远不能**标为 full population、decoder-complete exact 或用 CI 中心值填 exact Table-A 行。

## 7. 明确禁止

- 用 932 s × 一行密度比例外推其他层/配置并当结果；
- 因本行 lower-priority stall 为 0 而删除 weight/psum/memory 模型；
- 省略 refill、dirty spill/restore、dense commit、D1 或 cold drain；
- 把 terminal liveness、compiled kernel、sharding 产生的主机加速写成 accelerator speedup；
- 在新 source candidate、fresh hammer、一次完整 M883 identity miter 和新的 production release 之前启动全量。

裁决：**GTLS 是当前最严谨的 successor。token eviction 单独只能治内存，bitset 单独只能治 interval 插入，sampling 单独不能闭合 decoder-complete；必须把 compressed/group-level schedule、live terminal state、packed events、sealed sharding 四者合并，且以 M883 的一个完整行作为不可变数值锚。**
