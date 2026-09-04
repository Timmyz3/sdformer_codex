# Typed Slot Metadata与IPD选择性驻留架构闭环

## 一、结论

本轮关闭了第三轮DATE审稿中最具体的控制与存储合同缺口：Adaptive前端不再在每个output tile重放时解析一次word0，也不再把FADC24误套入IPD32W专用的warm offset和descriptor cache。

最终合同是：

```text
payload ingress
  -> commit时一次性校验magic/version/tag/保留位
  -> slot写入{valid, tag, mode, format, payload_bits, word_count}
  -> PLAN按format生成不可变事务
       RAW41  : cold replay, start_word=0, non-cacheable
       IPD32W : cold replay或resident replay, cacheable
       FADC24 : cold replay, start_word=0, non-cacheable
  -> atomic commit同时锁定route/format/cache ownership/start offset
  -> decoder projection共享term/event后端
```

`FORMAT`固定为2 bit：`0=RAW41`、`1=IPD32W`、`2=FADC24`、`3=非法`。非法header会完成payload drain，但不会发布有效slot，避免半提交状态被后续PLAN观察。

四组真实窗口trace以及两组由S3向量合成的混合格式功能用例，在同一Adaptive+residency配置下通过Icarus、Verilator/SVA和逐元素金参考。四stage周期和为`195149`，相对提交期metadata但关闭驻留的`196346`只提升`1.006x`。该数字说明驻留在当前四窗口bundle中是控制/能量辅助机制，不是周期主贡献；本轮真正价值是格式、缓存资格和生命周期的语义闭环。

## 二、为什么不能直接让三种格式都驻留

IPD32W的resident replay可以由`term_count`确定token区起始位置：

```text
token_start_word = 2 + ceil(term_count / 2)
```

FADC24的descriptor是24 bit/term，destination还可能采用8 bit list或162 bit bitmap。仅缓存`gate/lane/destination_count`不能恢复FADC的descriptor边界、destination mode和正确token offset。RAW41则根本没有CSR descriptor。

因此本轮选择“IPD可驻留、FADC/RAW不可驻留”，而不是扩大cache entry去支持所有格式。它有三个工程理由：

1. 保持bit-exact，不修改H67投影结果；
2. 避免为S3高扇出FADC引入宽cache entry和额外SRAM端口；
3. 真实trace表明FADC已经把S3 terms从`30960`降到`12888`，其主要收益来自流式表示和multicast，不依赖descriptor residency。

该选择仍需目标SRAM能量验证。没有macro和mapped SAIF前，只能称为已验证的功能合同，不能声称能耗收益。

## 三、模块与接口变化

### 3.1 Slot提交与元数据所有权

`gatestack_head_slot_sram_adapter`在最后一个payload word提交时解析word0，并把格式写入与slot tag同生命周期的metadata。inspect和replay接口都返回同一format，格式在slot有效期间保持稳定。

提交检查包括：

- RAW mode必须产生`FORMAT_RAW`；
- IPD检查`16'h4753`、version、flags/保留位和tag；
- FADC检查`16'h4641`、version、保留位和tag；
- 无效header增加错误计数，但不把slot标记为valid。

这里验证的是**格式身份与slot发布资格**，不是完整压缩payload证明。第二header、term count、格式专用长度、descriptor和destination完整性仍由对应decoder在replay时检查；论文不得写成“commit时验证完整CSR payload”。

### 3.2 PLAN与Atomic Commit

`gatestack_replay_plan_builder`把format写入plan，并执行以下资格判断：

| format | route | cache lookup | replay start | decoder配置不支持时 |
|---|---|---|---:|---|
| RAW41 | RAW | 否 | 0 | 不适用 |
| IPD32W cold | IPD | 是，miss后cold | 0 | 有界reject |
| IPD32W warm | RESIDENT | 是，hit后resident | `2+ceil(N/2)` | 有界reject |
| FADC24 | CSR cold | 否 | 0 | 有界reject |

`gatestack_replay_atomic_commit`不只比较tag，还同时验证：

- plan format与projection format一致；
- route与format匹配；
- resident route必须是IPD、必须拥有cache line；
- cold IPD/FADC/RAW的start word必须为0；
- warm IPD offset必须匹配term count公式；
- slot replay、projection start和生命周期提交保持原子性。

### 3.3 Decoder与Cache

`gatestack_adaptive_csr_replay_decoder`新增start format输入。metadata有效时从`IDLE`直接进入`START`并锁定IPD或FADC child；旧的`PEEK`路径只保留给没有format metadata的兼容入口。

`gatestack_ipd_cache_fill_adapter`新增cache资格门：FADC/RAW数据仍被正确消费，但不会发起cache begin/update/commit。外部descriptor fill也必须携带format，并经过同一IPD-only过滤，防止旁路接口污染cache。

外部descriptor fill当前是可信预填充边界：RTL验证format、容量和基本entry合同，但没有逐项证明其内容等于同tag slot中的IPD descriptor。主实验使用decoder自动fill路径；若论文保留外部接口，必须标注为trusted prefill，不能把它算作端到端自校验机制。

`gatestack_descriptor_residency_cache`把release定义为tag-qualified幂等操作。每个被residency管理的最终IPD事务都携带payload tag发送release：tag匹配才清除line；没有line时记录no-op而不是死锁；存在line但tag不匹配时保留新line、置protocol error并增加mismatch计数。这样旧生命周期的延迟release不会删除同一context/head的新payload。

## 四、真实Trace RTL结果

证据：`results/gatestack_typed_residency_fulltop_20260718/report.{md,json}`。

| Stage | 格式 | 周期 | cache hit/release | slot replay | terms | mismatch/protocol |
|---:|---|---:|---:|---:|---:|---|
| S0 | IPD32W | 2395 | 6/3 | 9 | 186 | 0/0 |
| S1 | IPD32W | 1677 | 30/6 | 6 | 0 | 0/0 |
| S2 | IPD32W | 21374 | 132/12 | 78 | 1956 | 0/0 |
| S3 | FADC24 | 169703 | 0/0 | 576 | 12888 | 0/0 |

S3的`0/0`是设计要求，不是cache失效：FADC明确non-cacheable，576次投影均从word0精确回放。

混合context结果：

| 用例 | 格式构成 | 周期 | cache hit/release | 预期 | 结果 |
|---|---|---:|---:|---:|---|
| IPD/FADC/RAW | 11/12/1 | 259138 | 253/11 | `11*(24-1)=253` | 通过 |
| IPD/FADC | 11/13/0 | 163493 | 253/11 | `11*(24-1)=253` | 通过 |

两个混合用例都实现`projection_heads=576`、逐元素零mismatch、done/protocol/abort为0。它们证明选择性驻留不会因head间格式切换而串线，也证明FADC/RAW不会产生伪cache hit。

## 五、周期收益应该如何解释

提交期metadata相对旧每replay PEEK版本的无驻留周期变化：

| 配置 | 四stage周期和 |
|---|---:|
| 旧PEEK Adaptive，无驻留 | 197857 |
| Typed metadata Adaptive，无驻留 | 196346 |
| Typed metadata + IPD-only residency | 195149 |

从旧PEEK到typed metadata减少`1511`周期，主要是消除每个replay约2周期的重复格式发现。加入IPD-only residency再减少`1197`周期，但四stagebundle仅`1.006x`，因为S3/FADC占总周期绝大多数且不驻留。

论文不能把`1.006x`包装成主性能创新。更准确的定位是：

- typed metadata是控制面与存储层次一致性机制；
- IPD-only residency是exact replay能量辅助机制；
- FADC24流式高扇出执行和GateStack共享后端仍承担主要周期收益；
- 是否保留descriptor cache必须由目标SRAM读写能量与面积消融决定。

## 六、验证矩阵

本轮新增或重跑：

- slot commit：合法RAW/IPD/FADC、非法magic不发布slot；
- planner：IPD hit/miss、FADC/RAW不查cache、固定IPD配置收到FADC后有界reject；
- atomic commit：format/route/cache ownership/start offset一致；
- cache fill：非IPD只drain、不产生cache事务；
- descriptor cache：正常release与幂等no-op release；
- stale release/refill：旧tag release不清新line，tag mismatch与protocol error可审计；
- projection：metadata直接选择child，162输出规模通过；
- full top：S0-S3与两个24-head混合context，Icarus和Verilator/SVA通过；
- RTL静态检查：相关模块Erie lint、Verilator warning/error和Yosys可读性检查通过；三种结构配置均生成网表并完成开放LEC。

仍未覆盖：随机reset/abort矩阵、长序列多context交错、完整162x32全网、mapped-netlist Formality/LEC、目标库门级时序和功耗。开放RTL-to-structure LEC见下一节。

### 6.1 开放结构消融与等价

使用同一single-context顶层、同一Yosys流程，只改变`CSR_FORMAT_FADC24`与`ENABLE_RESIDENCY`参数：

| 配置 | Yosys design cells | `$mem_v2` | `$mul` | `$mux` | RTL-to-structure LEC |
|---|---:|---:|---:|---:|---|
| 静态IPD + residency | 4191 | 13 | 43 | 1364 | 4832/4832 |
| Adaptive + no residency | 4958 | 11 | 44 | 1614 | 4762/4762 |
| Adaptive + IPD-only residency | 5249 | 14 | 49 | 1713 | 4832/4832 |

Adaptive配置加入选择性驻留后增加`291`个generic cells，约`5.87%`，并增加3个逻辑memory；相对静态IPD+驻留，完整Adaptive候选增加`1058`个cells，约`25.2%`。这些数字证明双decoder和residency都不是免费控制逻辑，也证明当前RTL可以生成结构网表并完成开放等价；它们仍不是标准单元面积、SRAM宏面积、时序或功耗。

上述网表入口已参数化：

```bash
CSR_FORMAT_FADC24=2 ENABLE_RESIDENCY=1 \
  bash dc_handoff/scripts/run_gatestack_yosys_structure.sh
CSR_FORMAT_FADC24=2 ENABLE_RESIDENCY=1 LEC_TIMEOUT_SECONDS=900 \
  bash dc_handoff/scripts/run_gatestack_yosys_lec.sh
```

## 七、对DATE架构创新的作用

该轮关闭的是第三轮审稿的P0控制一致性缺口，可以作为以下从属贡献：

> 一种tag-coherent typed slot与format-qualified residency机制，在逐head异构稀疏表示间原子携带格式、路由、cache所有权和warm offset，使IPD驻留与FADC/RAW精确回放能够共享同一投影后端且保持bit-exact。

它不能单独支撑DATE主创新，原因是typed metadata、cache qualification和time-multiplex本身都不是新概念。论文主贡献仍应建立在完整的数据流组合与实证上：

1. H67 final-gate等价类驱动的GateStack projection与multicast数据流；
2. IPD32W/FADC24异构表示对低/高扇出workload的精确映射；
3. output-tile-stationary累加避免head-major partial-sum spill；
4. typed slot把格式选择、驻留和异常恢复统一成可验证事务合同。

要达到DATE可接收标准，还必须补目标库PPA/SAIF/SRAM、physically-stripped同约束基线、多窗口profile、valid825部署合同和full-encoder Amdahl分账。当前状态仍是“架构控制闭环完成、ASIC证据未签核”，不是可直接投稿或直接流片。

## 八、学习抓手

对硬件初学者，建议用本轮数据流理解四个基本概念：

1. **metadata不是注释**：它决定后续端口、decoder、cache和offset，必须和payload一起保持tag一致；
2. **cache hit不是简单少读几拍**：只有缓存内容足以重建后续语义时才能跳过原payload；
3. **valid/ready是事务合同**：格式、tag、start offset在stall期间必须稳定，提交必须一次且原子；
4. **局部优化不等于系统收益**：S0-S2有明显warm hit，但S3占主导且不驻留，所以bundle只快0.6%。

阅读顺序建议：`head_slot_sram_adapter -> replay_plan_builder -> replay_atomic_commit -> slot_replay_word_router -> multihead_decoder_projection_top -> descriptor_residency_cache`，同时对照本文件第二至第四节。

## 九、入口

- 主回归：`ENABLE_TYPED_RESIDENCY=1 bash sim_hitflow/run_gatestack_adaptive_csr_fulltop.sh`；
- 结果：`results/gatestack_typed_residency_fulltop_20260718/report.{md,json}`；
- 无驻留对照：`results/gatestack_adaptive_csr_fulltop_20260718/report.{md,json}`；
- planner定向回归：`bash sim_hitflow/run_gatestack_replay_plan_builder_checks.sh`；
- projection严格回归：`bash sim_hitflow/run_gatestack_multihead_decoder_projection_checks.sh`。
