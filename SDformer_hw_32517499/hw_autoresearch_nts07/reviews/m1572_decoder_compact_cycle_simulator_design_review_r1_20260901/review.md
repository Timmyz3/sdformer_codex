# M1572｜Decoder compact cycle-simulator 只读设计审阅

日期：2026-09-01（Asia/Shanghai）  
性质：设计审阅；没有实现代码、执行、EDA、GPU、capture、commit 或 push  
目标：用固定容量数值状态替换 M1556/M1539 热路径中的 request-dict、token-string 与 JSON 临时序列化，**不改变资源、请求顺序、依赖、cache、端口或 commit 语义**。  
裁决：**GO_AUTHOR_SOURCE_ONLY_AFTER_EXACT_MITER_CONTRACT。**compact simulator 可作为 M1539 的 representation-preserving successor；任何 cycle/count/bytes/commit/address miter 不通过都必须 fail closed，不允许用“更快模拟器”产生新性能数字。

M1570 的 D0/call0 one-shot 已消费并因 RSS 超过严格 8 GiB 门失败。M1572 不是 M1570 自动重试；未来若实现，必须经历新 source、独立 hammer、新命名空间和独立 exactly-once release。

## 1. 不可改变的 M1539 语义

### 1.1 population 与配置

- checkpoint：Motion ep34 live93，SHA `4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48`；
- pilot identity：sample 10、D0、call ordinal 0、T10，immutable positive plane `576000 B`；
- 配置固定顺序：`DENSE_TYPED_K8`、`BIT_EQUAL_SERVICE_K1X8`、`BIT_TYPED_K8`；
- `PRODUCT_CAPTURE_TYPED_K8` 继续双重拒绝；
- D0/D1 positive plane 仍是 bit-times-layer-FP32-constant，不能归一成1或折叠weight；D2/D3仍exact binary。

### 1.2 common resource

资源 JSON 和 canonical digest 必须逐字节等价于 M1539/M1525：

- 96 lanes、Acc24、3.0 ns；external 192 B/cycle；macro-rounded SRAM 245760 B；
- weight 13824 B：8-bank 1R1W，16 B/bank row，read latency 4，II1，outstanding 8/bank；
- psum 221184 B：6-bank 1RW，48 B/bank row，read latency 2、write latency1，II1，outstanding 8/bank；
- external：1-bank 1RW，read latency32、write latency3，II1，outstanding16；
- compute：1 context，288 B，latency1，II1，outstanding1；
- control/descriptor 8192 B、reserve 2560 B；
- dense commit：每96-output block 384 B，所有output site按相同地址顺序提交；
- resource manifest SHA256保持 `64661d825ee8ddbdccad9c3e09ca5e41c5ea9cfc75bcea394667dcfd91b4de10`。

compact 路径不得改变这些参数来降低内存或改善周期。

## 2. M1556 为何仍会爆 RSS

M1556 已做到不物化完整 transaction list，并在每个 destination 后只保留 barrier/control token；但热路径仍反复创建：

- 每请求一个 Python `dict`；
- 每个 dependency/producer 一个长 token string；
- addresses/banks/dependencies 的 tuple/list；
- token、next_port、outstanding 的 dict/list 更新；
- address digest 的 `json.dumps([id,kind,address,bank])` 临时字符串与bytes；
- contributor tuple、tile-key tuple、group list、cache dict/age dict。

D0 dense 在一个内部 destination 最多有4个spatial contributors × 1536 channels = 6144 source/tap contributor；4个output blocks又重复大量group/request构造。即使live对象逐destination被清除，Python allocator的arena/high-water RSS仍可随海量分配增长。M1570已经证明“逻辑streaming”不等于“物理RSS bounded”。

## 3. compact 状态表示

### 3.1 禁止的热路径对象

进入request zero后，禁止：

- request dict；
- token或request ID字符串；
-按request/destination/timestep增长的dict/set/list；
- `json.dumps`/字符串format用于transaction digest；
- 完整transaction、receipt或contributor population留存。

允许固定容量、预分配并复用的数值数组，以及逐请求立即更新的hash/counter。

### 3.2 numeric dependency cycle

不再用 `token_string -> ready_cycle` 全局字典。每个producer函数直接返回整数 `ready_cycle`，consumer显式传入所需ready cycle：

- call级：`source_ready[t]`、`persistent_control_ready`；
- output-block级：`previous_psum_write_ready`，初值为timestep barrier；
- group级：最多8个descriptor ready、最多8个refill-external/weight-write ready、最多8个weight ready；
- `psum_read_ready -> compute_ready -> psum_write_ready`；
- commit依赖最后一个psum write，空group则依赖barrier。

每次调度仍计算：

`dependency_ready = max(all dependencies, earliest)`；  
`issue = max(earliest, dependency_ready, all relevant next_port)`；  
若任一bank outstanding已满，等待该bank最早合法return并对所有bank重新迭代；  
`beats = max(1, ceil(width/service_bytes))`；  
`returned = issue + latency + beats - 1`；  
`next_port = issue + max(II, beats)`。

compact实现失去M1539的duplicate-token字典检查后，必须以单调numeric request ordinal + 唯一event-coordinate assertion替代；不能简单删除duplicate-producer保护。

### 3.3 bounded port calendars/outstanding

固定上界可由资源合同直接推出：

| 状态 | 固定entry上界 | 说明 |
|---|---:|---|
| next-port calendar | 24 | weight read/write `8x2=16`；psum6；external1；compute1 |
| weight outstanding return cycles | 64 | 8 banks × 8 |
| psum outstanding return cycles | 48 | 6 × 8 |
| external outstanding | 16 | 1 × 16 |
| compute outstanding | 1 | 1 × 1 |
| **全部 outstanding slots** | **129** | 固定数组/小根堆，禁止动态增长 |

1RW端口必须让read/write共享同一calendar key；1R1W weight必须保留read/write两个calendar。每bank使用固定容量sorted ring或min-heap；在候选issue时丢弃`return <= issue`，满时推进到最早active return，并重复检查所有bank。这要逐请求miter，不能用近似queue depth。

### 3.4 9-tile weight cache

用9-entry固定数组替代`key_to_slot`与`age` dict：

- `valid[9]`、`packed_key[9]`、`age[9]`、全局单调tick；
- key仍为 `(module, output_block, tap, channel//16)`，仅编码成无碰撞整数；
- 当前K8 group最多8个unique key；先按原出现顺序去重，再逐key处理；
- hit同样增加tick并更新age；
- miss先选最小free slot；满时只从unpinned entry选victim；victim为最小 `(age, original tuple key)`；
- cache在一个configuration/call内跨destination、timestep持续存在，不得在retire时清空；三个configuration各自从空cache开始。

D0最坏每bank contributor queue为768项。`(tap<<11)|channel`可无碰撞放入`uint16`，8×768仅12288 B；建议预分配8条typed array并复用。不得为了省内存改变M1539 `bank_unique_groups`的顺序：先按原contributor顺序入`channel%8`队列，再按queue ordinal从bank0到7取一项组成group。

### 3.5 逐 destination retire

retire只释放/覆盖：

- 8条contributor queue的logical length；
- 当前group的descriptor/refill/weight ready数组；
- 每output block的`previous_psum_write_ready`；
- 临时packed tile keys与request coordinate。

retire不得重置：

- port calendars和outstanding；
- 9-tile cache/age/tick；
- global last-cycle、request/kind/byte counts；
- address/commit digest；
- timestep source barrier与persistent control ready。

output blocks之间只通过真实共享端口calendar发生时序影响；M1539里每个output block的psum dependency初值都是barrier，不得额外串联前一output block commit。

### 3.6 packed address digest

production热路径不能继续`json.dumps`长request ID。建议定义versioned fixed-width binary event，一条address-bank pair更新一次SHA256：

```text
schema_version:u8, config:u8, kind:u8, module:u8,
timestep:u8, flags:u8, destination:u32, output_block:u16,
group:u32, subordinal:u16, request_ordinal:u64,
address:u64, bank:u8, width_bytes:u32
```

- 固定big-endian；sentinel值对source/control等非destination请求必须写入schema；
- 禁止Python `hash()`、pickle、native-endian或依赖对象地址；
- 每个multi-bank request按M1539 zip(addresses,banks)顺序写多条packed event；
- 输出字段必须叫`packed_transaction_address_sha256`并携带`packed_address_schema_sha256`，不能冒充legacy `transaction_address_sha256`。

admission miter必须把M1539 row流通过只读adapter转换成同一packed event，并比较digest。若还要保留legacy SHA，只能作为prefix/debug dual-digest；不得为了legacy JSON string在full run重新引入高分配路径。

commit digest可继续精确复现M1539的`[commit_ordinal,address,width]`流，因为不依赖长request ID；同时建议输出packed commit digest作为第二重检查。

## 4. 与 M1539 必须逐项 miter 的条件

完整机器可读矩阵见 `miter_matrix.csv`。任何一项不等都判compact source NO-GO。

### 4.1 cycle

1. 每request：`earliest`、`dependency_ready`、`port_ready`、outstanding等待后的`issue`、`beats`、`returned`完全相等；
2. 每destination retire checkpoint：`last_cycle`、所有next-port calendar、每bank active return multiset相等；
3. 每timestep与每configuration：`last_cycle+1`相等；
4. total cycles必须逐config exact相等，不允许“数值接近”；
5. compact更快的host wall time不能进入simulated cycle。

### 4.2 count

必须逐config/逐timestep/逐destination核对：

- contributor count与canonical order；
- bank-unique group count、每group lane count/bank sequence；
- request_count；
- 每kind count：external_read、external_write、weight_read、weight_write、psum_read、compute、psum_write、commit；
- cache hit/miss/refill count、每次miss key/slot/victim；
- BIT_EQUAL_SERVICE_K1X8与BIT_TYPED_K8 contributor及compute-group population相同；
- destination、output block、dense commit population相同。

### 4.3 bytes

逐kind `sum(width_bytes * number_of_banks)` exact相等，并另核对：

- positive-plane/source fetch；
- typed/K1 descriptor bytes；
- 144-B common control read/write；
- 1536-B external tile refill与8×192-B weight write；
- 96-B per-source/output-block weight vector read语义；
- 6×48-B psum read/write；
- 384-B dense commit；
- external total与on-chip port total。

compact metadata本身是host表示，不得计成硬件traffic；但也不得借此删掉M1539任何descriptor/control charge。

### 4.4 commit

- 每个`timestep,destination,output_block`恰好一次commit；
- commit order、address、bank0、width384完全相同；
- dense、bit-equal、bit-typed三个config的commit sequence digest相同；
- 空contributor block仍在barrier后commit；
- commit issue/return受external 1RW calendar约束，不得视作免费；
- final commit count必须等于`T × Hout × Wout × ceil(Cout/96)`。

### 4.5 address

- 每个request的kind、numeric event coordinate、address vector、bank vector、width、顺序相同；
- psum slot/modulo和6个bank-local address相同；
- weight packed key、cache slot、`weight_bank_row` offset相同；
- refill external address与8-bank write address相同；
- descriptor/source/control/commit address相同；
- reference-adapter packed digest与compact digest相同；
- packed coordinate必须可逆映射到legacy ID grammar，抽样/前缀逐项证明，无collision。

## 5. miter层级

### L0｜static/schema

- 配置/forbidden branch、resource digest、geometry、numeric identity、request-kind enum、packed field widths；
- fixed arrays容量与地址范围assertion；
- 热路径静态拒绝request dict/token string/JSON digest。

### L1｜synthetic request-by-request

沿用M1539 8×2×2 synthetic并扩充：全零、单source、8-bank满group、K1×8、cache fill/evict、outstanding满、1RW read/write冲突、空destination、boundary tap。三个config每request比较cycle/count/bytes/commit/address和cache state。

### L2｜actual D0 bounded canonical prefix

读取同一immutable 576000-B plane，仅运行不会触发M1570内存风险的canonical prefix。prefix必须包含：

- corner/edge/interior与四种destination parity；
- cache首次装满和至少一次eviction；
- external/weight/psum/compute outstanding满场景；
- 4个output blocks；
- 至少t0与后续timestep source barrier。

每destination做完整state checkpoint miter。不得通过跳过中间destination来构造会改变cache/calendar历史的“采样”。

### L3｜full compact D0/call0 diagnostic

只有L0--L2与memory dry-run全部通过，才允许新的一次性diagnostic。full M1539 reference已经被RSS证伪，L3不要求同进程跑两份full simulator；它依赖逐transition等价与full aggregate invariants。结果仍须独立hammer，不能自动扩成120-call production。

## 6. memory 门

### 6.1 静态boundedness门

- hot path所有container容量有常数上界；
- outstanding总slot=129、cache=9、current group<=8、D0 contributor scratch<=6144 packed uint16；
- 无request/destination/timestep键控增长；
- 不保存transaction或per-request receipt；
- immutable plane仍仅576000 B且FD在request zero前关闭。

### 6.2 动态RSS门

未来source hammer需在双Python运行synthetic与actual-prefix并记录：

- post-import/post-snapshot baseline RSS；
- 每destination current RSS、每timestep VmRSS/VmHWM、ru_maxrss；
- live fixed-array capacity；
- Python allocated-block/tracemalloc只作diagnostic，不替代OS RSS。

建议新D0 pilot同时满足：

1. 旧合同hard fail仍为absolute peak RSS `<8 GiB`；
2. compact admission更严：absolute peak `<2 GiB`，且相对post-snapshot baseline增量 `<512 MiB`；
3. t1--t9的current RSS不得相对t0继续单调无界增长；`max(VmRSS[t1..t9])-VmRSS[t0] <=256 MiB`；
4. 任一fixed array超过合同容量立即fail，不等RSS触顶；
5. memory check本身固定频率，不生成per-request日志。

2 GiB/512 MiB是pilot工程门，不是硬件指标；若双Python基线环境证明需调整，只能在attempt消费前经新独立review修改，不能运行中放宽。

环境preflight还须有`MemAvailable >=16 GiB`、`CommitLimit-Committed_AS >=16 GiB`、无same-UID EDA/VCS冲突；环境富余不能替代进程RSS门。

## 7. 单次运行门

M1570已经消费且不可重试。M1572未来执行顺序必须是：

1. 新compact source/contract，双Python compile/test；
2. L0/L1独立source hammer；
3. actual D0 prefix L2、memory boundedness独立hammer；
4. 新one-shot release source与独立launch hammer；
5. preflight完整验证M1539/M1542/M1559、immutable payload、resource digest、free memory/commit/disk、输出目录不存在；
6. 先原子创建新命名空间和`WORK_STARTED`，再进入request zero；
7. 固定三个configuration顺序，各完成后原子写partial；任一失败立即停止并封`FAILED_OR_INCOMPLETE`；
8. `automatic_retry=false`，不得复用M1570 namespace/marker/token；
9. 三config全部完成后要求common resource、checkpoint、population与commit digest一致；
10. result独立hammer后仍只是一条D0/call0 diagnostic，不是production/Table-A/paper性能。

禁止在唯一run中启用profile/tracemalloc/legacy full JSON digest等会改变内存行为的debug路径。它们应在L1/L2前缀完成。

## 8. 最终设计裁决

compact successor应该被描述为：

> A representation-only implementation of the frozen M1539 transition machine, using numeric readiness, fixed port calendars, a nine-entry array cache, per-destination scratch retirement and a versioned packed address digest.

它不是新scheduler、不是新的decoder优化，也不能因为host wall time/RSS下降而改变论文cycle。只有M1539语义逐项miter、RSS门和new one-shot门全部通过，才允许重新获得D0/call0 diagnostic；production仍需另一次独立授权。

