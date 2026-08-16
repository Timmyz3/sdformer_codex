# 完整 Projection 集成 RTL 独立审阅

**日期**：2026-07-20  
**审阅角色**：RTL 设计编排与 ASIC 独立审阅  
**主对象**：`rtl_hitflow/gatestack_builder_projection_single_context_top.sv`  
**子对象**：C0/C1 Builder、typed-slot、external-slot inspect/replay/release、replay control/lifecycle、projection backend、abort/reset  
**快照**：以 2026-07-20 18:06:42 +0800 后稳定的目标顶层为准；未修改 RTL 或测试。

## 1. 结论

**ASIC RTL 签核不通过。**

默认 H67 形状 `TOKENS=162`、`LANES=32`、`GATE_W=9`、`HEADS=3/6/12/24`、`WORD_W=64` 下，C0/C1 的正常路径接线和 slot 原子可见性总体自洽；独立 Verilator 顶层 lint 在 C0/C1 默认配置均为 0 warning/0 error。但当前还有 **4 项 P0** 和 **7 项 P1**：

| 等级 | 编号 | 结论 |
|---|---|---|
| P0 | P0-1 | Builder 阶段不在 execution watchdog 覆盖内，错误或永久停顿只能依赖外部 `batch_abort_valid` |
| P0 | P0-2 | 已接纳 group 期间接受 `batch_abort_valid` 会直接复位 execution，不产生对应的 error completion |
| P0 | P0-3 | bias 只有“请求握手当拍即数据”接口，且数据仅 `PRODUCT_W=17` 位，无法直接对接同步 ASIC SRAM/完整累加器 bias |
| P0 | P0-4 | 累加溢出在 final 握手同一拍才被置位，溢出数据可先逃逸到模块外，abort 不具备输出隔离原子性 |
| P1 | P1-1 | 顶层参数表面可配，实际隐含 `WORD_W=64/LANES=32/GATE_W=9/CLASS_SLOTS<=4` 等固定约束 |
| P1 | P1-2 | `HEAD_BITS`、`MAX_TERMS`、`HEAD_COUNT_W`、`TOKEN_ID_W`、`SIZE_W` 缺编译期一致性检查 |
| P1 | P1-3 | group 和 replay 内部使用 valid 依赖 ready 的“原子脉冲”契约，不是可任意替换的标准 decoupled 接口 |
| P1 | P1-4 | slot release 只有 context/head，没有 expected payload tag/generation，不支持 stale release 防护 |
| P1 | P1-5 | 手工 wrapper abort 与 execution 内部 abort 的 `protocol_error`/计数器复位语义不一致 |
| P1 | P1-6 | head slot、workspace bitmap、AccTile 仍是行为数组，端口形状未冻结为可交付 SRAM macro 契约 |
| P1 | P1-7 | H67 证据只到候选 INT8 权重的 `acc32` 累加，未包含 scale/requant/饱和/residual/ATLIF，不得宣称完整 H67 部署一致 |

## 2. 审阅边界与层次

实际数据流为：

```text
final-gate/K
  -> C0 单 workspace 或 C1 双 workspace
  -> format policy / serializer
  -> gatestack_head_slot_sram_adapter
  -> external-slot inspect/replay/release
  -> replay plan / atomic commit / dual-tag lifecycle
  -> IPD/FADC/RAW decoder
  -> product / multicast / banked accumulator
  -> final acc32
```

顶层确实禁用 execution 内部 payload commit，并强制 `EXTERNAL_SLOT_SERVICE_ENABLE=1`：`gatestack_builder_projection_single_context_top.sv:411-444`。external-slot 的 inspect、replay、release 与 slot 状态/计数器在 `gatestack_builder_projection_single_context_top.sv:445-481` 直连 Builder slot。execution 端在 `gatestack_single_context_execution_top.sv:465-518` 将内部 slot 适配器完全旁路。

编译层次包含 30 个 execution 文件，其中 slot、plan、atomic commit、dual-tag lifecycle、word router、done guard、abort controller 的顺序见 `rtl_hitflow/filelist_single_context_execution.f:20-30`。

## 3. P0 详细问题

### P0-1：Builder 阶段没有有界 watchdog 或自动错误完成

**证据**

- 顶层只在所有 head 已接纳、已完成且 slot 全 valid 时打开 `group_ready`：`gatestack_builder_projection_single_context_top.sv:207-216`。
- Builder 错误会置位 `builder_error_q`/一批错误，并反过来禁止 group 接纳：`gatestack_builder_projection_single_context_top.sv:219-245`。
- 这些状态只在全局复位、slot reset 或 group completion 时清除：`gatestack_builder_projection_single_context_top.sv:224-257`。出错后 group completion 不可达。
- execution watchdog 的 `group_active` 只由 execution group 接纳置位：`gatestack_context_abort_controller.sv:57-63`；超时只在 `group_active` 为 1 时生效：`gatestack_context_abort_controller.sv:34-35,74-82`。

**影响**

1. token 源中途停止、Serializer/slot ready 永久不返回或 Builder 内部活性故障时，`ABORT_TIMEOUT_CYCLES` 不生效。
2. 重复/非法 head 或 Builder 错误后，只能由外部观察 `protocol_error` 再额外驱动 `batch_abort_valid`。该反馈路径无有界时延契约。

**必须修复**

在 wrapper 内建立从首个 `head_fire` 到 group 接纳的 batch watchdog，并将 Builder 三类错误统一转成本地 flush+唯一带 tag 错误完成。不应把恢复正确性寄托给未规范的主机软件轮询。

### P0-2：活动 group 上的 wrapper abort 会静默丢失已接纳事务

**证据**

- `batch_abort_ready` 恒为 1，没有限制只能在 build 阶段使用：`gatestack_builder_projection_single_context_top.sv:191-192`。
- abort 握手直接并入 `execution_rst_core`、`slot_reset_pulse`和 `builder_rst_core`：`gatestack_builder_projection_single_context_top.sv:193-195`。
- group completion 完全来自被复位的 execution，wrapper 没有 abort completion 保持寄存器：`gatestack_builder_projection_single_context_top.sv:430-436`。
- scheduler 复位后直接回 `ST_IDLE`，不保留已接纳 group tag：`gatestack_output_tile_scheduler.sv:167-183`。

**影响**

若 group 已与上游握手，之后的 `batch_abort_valid` 会清空内部事务，但上游永远收不到该 group 的 `group_done_valid/error/tag`。这破坏一次请求对应一次完成的系统契约。

**必须修复**

二选一：

1. 将 `batch_abort_ready` 限制为仅 build 阶段可揥受，活动 group 交由 execution abort controller 处理；
2. wrapper 锁存 active group tag，flush 后保持一个 `group_done_valid=1, group_done_error=1`，直到上游握手。abort 响应寄存器不能被同一个 flush 复位。

### P0-3：bias 接口不是 ASIC 同步存储可用契约，且宽度不足

**证据**

- 顶层只有 `bias_req_valid/ready/token_id` 和同拍输入 `bias_req_values`，没有 `bias_rsp_valid/ready/tag`：`gatestack_builder_projection_single_context_top.sv:86-89`。
- projection 在 `bias_req_valid && bias_req_ready` 同拍把 `bias_req_values` 直接送入累加器：`gatestack_single_head_projection_top.sv:155-173`。
- bias 数据宽度是 `OUT_TILE*PRODUCT_W`，累加器只做符号扩展：`hitflow_banked_accumulator.sv:24-26,195-201`。默认 `PRODUCT_W=17`，而 `ACC_W=32`。
- H67 trace 生成器从 `projection_bias_acc_int64` 读 bias，并按 32 位写向量：`scripts/generate_gatestack_real_trace_vectors.py:56-58,135-143`。现有 TB 又显式裁成 `PRODUCT_W`：`tb_hitflow/tb_gatestack_builder_projection_real_s0.sv:193-198`。

**影响**

1. 同步 SRAM/ROM 的一拍或多拍返回无法表达；综合只能依赖大型组合读或在顶层外私自增加无契约的缓存。
2. 任意超出有符号 17 位的 bias 会在接口处静默截断；当前单样本刚好未触发，不是通用正确性证明。

**必须修复**

使用独立 bias request/response ready-valid，response 至少携带 token/执行 tag 和 `OUT_TILE*ACC_W` 数据；在 projection 中增加一个可停顿响应级。若确认 bias 是组合常量 ROM，必须在接口规格中明确限制并给出目标库时序证据。

### P0-4：累加溢出数据可先于 abort 逃逸

**证据**

- bank 在 bias commit 时同时驱动 `final_valid`，只要 `final_ready` 为 1 就允许写回/输出：`hitflow_banked_accumulator.sv:133-140`。
- `lane_overflow` 与 final 数据由同一个 `sum_vector` 组合计算：`hitflow_banked_accumulator.sv:142-160`。
- sticky `accumulator_overflow` 在该握手时钟沿之后才置位：`hitflow_banked_accumulator.sv:239-244`。
- execution 将 overflow 纳入 `fabric_protocol_error`：`gatestack_single_context_execution_top.sv:899-910`，但 abort controller 也要到时钟沿才生成 reset/abort completion：`gatestack_context_abort_controller.sv:65-82`。
- `final_valid/final_values` 在顶层不经 poison/quarantine 直连模块外：`gatestack_builder_projection_single_context_top.sv:495-501`。

**影响**

溢出 final 可在错误被记录之前已被下游消费。之后的 error completion 无法撤销该副作用，因此当前 abort 不是事务原子的。

**必须修复**

在 final 边界增加可撤销的一拍缓存，先检查溢出/事务错误，后对外拉高 valid；或将饱和策略冻结为正常数值语义，不再把可饱和事件当成 abort。两种方案必须选一种并写入 H67 数值契约。

## 4. P1 详细问题

### P1-1/P1-2：参数化表面大于实际可用集

| 参数 | 隐含约束与证据 | 风险 |
|---|---|---|
| `WORD_W` | execution external replay data 固定 64 位：`gatestack_single_context_execution_top.sv:86-94`；slot header 直接读 `[63:32]`：`gatestack_head_slot_sram_adapter.sv:216-227` | 必须等于 64；`WORD_W=32` 独立 lint 已出现越界、截断和 32->64 扩展告警 |
| `LANES` | execution scheduler 和 projection 硬编码 `.LANES(32)`：`gatestack_single_context_execution_top.sv:353-358,807-819` | Builder 按参数建 payload，execution 仍按 32 lane 解释 |
| `GATE_W` | typed descriptor 固定 9 位：`gatestack_typed_builder_commit_top.sv:40-45`；cache/decoder 接口也是 9 位 | 非 9 位配置会截断或扩展 |
| `CLASS_SLOTS` | workspace class count/活跃类 metadata 固定 4 位：`gatestack_canonical_head_workspace_c0.sv:132-152`，policy 实际按 4 类契约 | 大于 4 时不再等价当前格式契约 |
| `HEAD_BITS` | Builder 始终使用 `TOKENS*(LANES+GATE_W)`：`gatestack_typed_builder_commit_top.sv:204-210`；execution 使用可独立覆盖的顶层 `HEAD_BITS`：`gatestack_builder_projection_single_context_top.sv:411-413` | 覆盖后 RAW 元数据检查与真实 slot 不一致 |
| `MAX_TERMS` | 顶层只传给 execution：`gatestack_builder_projection_single_context_top.sv:411-429`；Builder commit 使用自身默认 128：`gatestack_typed_builder_commit_top.sv:6-19,130-143` | Builder 和 decoder 容量可分裂 |
| `HEAD_COUNT_W` | `HEADS` 在比较和传输前被强制 cast：`gatestack_builder_projection_single_context_top.sv:196-203,434` | `HEADS >= 2^HEAD_COUNT_W` 时可回绕成0，形成永久不 ready |
| `TOKEN_ID_W`/`SIZE_W` | token 外部路径多处固定 8 位，payload/word count 有 8 位输出：`gatestack_builder_projection_single_context_top.sv:96-102`、`gatestack_canonical_head_workspace_c0.sv:154-158` | `TOKENS>256`、`SLOT_WORDS>255` 或 size 不足时会截断 |

建议把默认 H67 特化顶层和真正通用模块分开。在未实现参数变体前，用编译期 elaboration assertion 明确拒绝不支持组合，不要留下“可编译但错解释”的表面参数化。

### P1-3：内部 ready/valid 是特殊原子脉冲契约

- wrapper 只在 `group_valid && group_ready` 时向 execution 发 valid：`gatestack_builder_projection_single_context_top.sv:214-217`。
- atomic commit 只在 projection、lifecycle、slot 同拍 ready 时产生一拍 commit/replay pulse：`gatestack_replay_atomic_commit.sv:145-156`。
- external replay begin 又直接使用这个 pulse：`gatestack_single_context_execution_top.sv:484-490`。

当前固定 Builder slot 的 ready 不依赖 valid，所以未形成现有组合环。但这不是标准的“valid 必须独立于 ready 并在 stall 期保持”边界。必须在接口文档中将其命名为 reserve/commit pulse，或改为寄存化 decoupled 揥口，否则替换 external-slot macro wrapper 时容易引入 ready-valid 组合环。

### P1-4：slot release 没有 tag/generation 保护

- lifecycle 内部已保存 payload tag，并在 cache release 上输出 expected tag：`gatestack_dualtag_replay_lifecycle_manager.sv:58-64,90-100`。
- slot release 却只输出 context/head：`gatestack_dualtag_replay_lifecycle_manager.sv:35-43`。
- slot adapter 只按 slot index 清 valid：`gatestack_head_slot_sram_adapter.sv:447-450`。

当前单 context、整 batch barrier 且最终 tile 后才释放，正常流中不会在释放前复用该 slot，因此不列为 P0。但任何后续跨 group overlap/epoch 设计都必须将 release 改为 `{context,head,payload_tag/generation}` 匹配后清除，否则延迟 release 可删除新 payload。

### P1-5：两条 abort 的状态与计数契约不一致

- wrapper abort 把 Builder 和整个 execution 都当作 `rst_core` 复位：`gatestack_builder_projection_single_context_top.sv:191-195,430-431`，因而所有子模块计数和 sticky error 被清零。
- execution 内部 abort 只用 `fabric_reset_pulse` 复位 datapath：`gatestack_single_context_execution_top.sv:946-974`；abort controller 自身 `protocol_error` 按断言故意保持到全局复位：`verif_hitflow/gatestack_context_abort_controller_assertions.sv:44-48`。
- wrapper 没有 `count_manual_batch_aborts`，手工 abort 还会把既有 Builder/execution 计数清零。

必须冻结计数器是“自全局复位累积”还是“每 batch”，并为手工 abort、内部 fabric error、timeout 分别计数。错误原因不能在恢复脉冲同拍被销毁。

### P1-6：存储仅是行为数组，不是 macro-ready 微架构

**head slot**

- payload 数组形状为 `CONTEXTS*HEADS*WORDS_PER_HEAD*WORD_W`：`gatestack_head_slot_sram_adapter.sv:86-107`。
- 同一行为模块包含一个流式写和一个注册读：`gatestack_head_slot_sram_adapter.sv:363-388,427-445`。功能上类似 1R1W，但未实例化 macro wrapper、byte/write mask、读延迟参数或 BIST/repair 端口。

**canonical workspace**

- C0 包含 `raw_record_q[TOKENS]`、`term_q[CLASS_SLOTS*LANES]`、`fanout_q[CLASS_SLOTS][LANES]`：`gatestack_canonical_head_workspace_c0.sv:138-147`。
- 默认 `EXPLICIT_BITMAP_BANK_ENABLE=0`，隐式 bitmap 路径以每 token 对所有 lane 多位更新，并以变址宽组合读 162 位：`gatestack_canonical_head_workspace_c0.sv:195-236,241-251`。这不等价于已解决的 SRAM 端口。

**AccTile**

- 每 bank 使用 `acc_mem[BANK_DEPTH]` 和按 token 位图复位：`hitflow_banked_accumulator.sv:91-103,164-205`。

默认 H67 S3 的 head slot 单独即为 `24*104*64=159744 bit`。在 SRAM compiler 宏、口延迟适配、DFT 和目标库时序完成前，开放综合 cell 数不能用于 ASIC 面积/功耗签核。

### P1-7：H67 一致性只闭环到候选 acc32

**已一致的部分**

- 真实 trace 生成器从 NPZ 解包 K bitmap、`gate_q17`、projection INT8 权重和 bias：`scripts/generate_gatestack_real_trace_vectors.py:48-67`。
- 金参考按 head/lane 计算 `gate * int8_weight` 并加 bias accumulator：`scripts/generate_gatestack_real_trace_vectors.py:69-79`。
- 四 stage 形状是 162 token、32 lane、3/6/12/24 head：`tb_hitflow/tb_gatestack_builder_projection_real_s0.sv:8-21`。
- Builder 输入确实是每 token 的 32 位 K 和 9 位 final gate：`tb_hitflow/tb_gatestack_builder_projection_real_s0.sv:126-149`。
- TB 对 `TOKENS*HEADS*32` 个 `acc32` 元素逐项比较：`tb_hitflow/tb_gatestack_builder_projection_real_s0.sv:261-295,391-416`。
- 向量 manifest 覆盖 45 head，当前格式分布是 44 IPD、1 FADC、0 RAW：`tb_hitflow/vectors/gatestack_all45_builder_20260720/manifest.json:1-10`。

**未一致/未证明的部分**

- trace manifest 明确把量化合同标为“候选，需 valid825 部署验证后冻结”：`results/gatestack_h67_real_trace_vectors_20260717/manifest.json:7-8`。
- `projection_weight_scale_exp2` 被 TB 读入，但只在结果行打印，没有参与 DUT 数据路径：`tb_hitflow/tb_gatestack_builder_projection_real_s0.sv:97-100,316-335,436-449`。
- DUT 输出是 `ACC_W` final accumulator，接口上没有 per-channel scale、RNE/requant、饱和、residual/skip 或 ATLIF 状态：`gatestack_builder_projection_single_context_top.sv:90-94`。
- 当前真实向量只选 `sample0/B0/window0`，不是 valid825 或 12 block 全覆盖：`scripts/generate_gatestack_real_trace_vectors.py:48-50,81-85`。

因此可防守表述是：**在已覆盖的 sample0/B0/window0 和候选 INT8 合同下，Builder->typed-slot->projection 的 acc32 累加结果可与对应整数金参考比较。**不可防守表述是：**完整 H67 projection 或 encoder 已部署逐位一致。**

## 5. 接口与 ready/valid 逐项结论

| 接口 | 结论 | 证据 |
|---|---|---|
| `head_begin` | 默认配置下可正确拒绝越界/重复 head；错误后是 fail-stop，见 P0-1 | `gatestack_builder_projection_single_context_top.sv:196-205,233-245` |
| token stream | Builder 内部在 ready 握手后核对连续 token id 和 last，错误可检测 | `gatestack_canonical_head_workspace_c0.sv:474-515` |
| Builder done | wrapper 绑定 `done_ready=1`，所以 `builder_done_pulse=done_valid` 在当前集成内成立；该输出不是可背压接口 | `gatestack_builder_projection_single_context_top.sv:207,279,351` |
| group | 全 head/slot barrier 正确；内部 valid 依赖 ready，见 P1-3 | `gatestack_builder_projection_single_context_top.sv:207-217` |
| inspect metadata | slot 保持 metadata valid 直到 ready，并阻止与同 slot commit/release 竞争 | `gatestack_head_slot_sram_adapter.sv:269-275,410-425` |
| replay word | 输出字寄存化，stall 时保持；最后一字只在数据已装入后结束 active | `gatestack_head_slot_sram_adapter.sv:288-297,427-445` |
| weight | 单 outstanding，response 的 tag/channel/tile 全比较；错响应不会被 ready 接收 | `gatestack_decoupled_product_engine.sv:78-102,171-179` |
| bias | 不是独立 response 通道，见 P0-3 | `gatestack_single_head_projection_top.sv:155-173` |
| final | stall 时由 bank busy 保持；溢出隔离失败，见 P0-4 | `hitflow_banked_accumulator.sv:133-160,180-205` |

## 6. slot 生命周期结论

### 6.1 正常流中成立的不变式

1. slot 只在 commit 字数、last 位和首字 header 全部合法后置 valid：`gatestack_head_slot_sram_adapter.sv:363-385`。
2. replay 只能从 valid slot 且合法 start word 开始：`gatestack_head_slot_sram_adapter.sv:257-265`。
3. commit/replay/release 对同 slot 的冲突有明确阻塞：`gatestack_head_slot_sram_adapter.sv:235-268`。
4. lifecycle 同时等 decoder payload tag 和 backend execution tag：`gatestack_dualtag_replay_lifecycle_manager.sv:70-84,139-165`。
5. 只有最后 output tile 完成时才释放 slot/cache，且两个 release 拥有独立 pending bit：`gatestack_dualtag_replay_lifecycle_manager.sv:153-179`。
6. group 完成前每个 head 的最终 tile 已通过 lifecycle，所以正常路径下 slot 会全释放后再开新 batch。

### 6.2 不能签核的边界

- build 阶段活性和错误完成见 P0-1。
- 活动 group 手工 abort 的 completion 见 P0-2。
- 输出副作在 abort 前隔离见 P0-4。
- tag-qualified release 见 P1-4。
- 当前没有顶层专用 SVA 证明“每次 slot commit 最多一次有效释放”、“每个已接纳 group 恰好一次 completion”和“abort 后无 stale final”。

## 7. reset/CDC/RDC 结论

- 本层次只有 `clk_core` 一个时钟域，未发现 CDC 跨域。
- `rst_core`、`slot_reset_pulse`、`fabric_reset_pulse` 都被 `always_ff @(posedge clk_core)` 作为同步复位条件使用，没有异步去复位 CDC。
- `execution_slot_reset_pulse -> slot_reset_pulse -> builder_rst_core -> external slot` 形成单拍同步 flush：`gatestack_builder_projection_single_context_top.sv:191-195,481`。
- 风险不在 CDC，而在复位的事务语义：completion 保留、输出隔离、错误原因保留和计数连续性尚未闭环。

## 8. 综合与 ASIC 交付风险

| 项目 | 当前判定 |
|---|---|
| 语法/lint | 默认 C0/C1 独立 Verilator `--lint-only -Wall` 均 0 warning/0 error |
| 参数 lint | `WORD_W=32, LANES=16` 变体确定产生 slot header 越界、Serializer 截断和 external replay 32->64 扩展告警，证明参数边界未受保护 |
| latch/多驱动 | 本次 lint 未发现 |
| CDC | 单时钟域，本层次无 CDC |
| SRAM | 未交付 macro wrapper、端口延迟、BIST/repair、物理布局和 macro 时序约束 |
| 时序 | bitmap 变址宽读、分段 priority search、大宽 accumulator 和 ready 联合路径未经目标库 STA |
| 时钟功耗 | 未见 ICG 插入、clock gating coverage 或 SAIF 证据；C0/C1 大量寄存数组的时钟活动未签核 |
| reset 扇出 | wrapper/fabric reset 同时覆盖 Builder、control、decoder、accumulator 的大量状态，需要复位树与 recovery/removal 时序检查 |
| PPA | 无目标 `.db/.lib`、PVT、SDC、SRAM macro、mapped SAIF，不具备 ASIC PPA 签核条件 |

## 9. 审阅时验证证据快照

1. 独立执行 C0/C1 顶层 Verilator lint，默认参数均无输出，即 0 warning/0 error。
2. 独立执行 `WORD_W=32, LANES=16` 变体 lint，在 `gatestack_head_slot_sram_adapter.sv:221,227`、`gatestack_typed_payload_serializer.sv:442,452`、`gatestack_builder_projection_single_context_top.sv:466` 得到确定告警。
3. 现有四 stage runner 显式枚举 S0-S3 和 C0/C1，并对比 C0/C1 checksum：`sim_hitflow/run_gatestack_builder_projection_real_s0.sh:26-49,52-75`。
4. 审阅快照中 S0-S2 的 C0/C1 现有日志均报 `mismatches=0/errors=0`：`build_hitflow/gatestack_builder_projection_real_allstages/s0_c0/iverilog.log:1`、`build_hitflow/gatestack_builder_projection_real_allstages/s0_c1/iverilog.log:1`、`build_hitflow/gatestack_builder_projection_real_allstages/s1_c0/iverilog.log:1`、`build_hitflow/gatestack_builder_projection_real_allstages/s1_c1/iverilog.log:1`、`build_hitflow/gatestack_builder_projection_real_allstages/s2_c0/iverilog.log:1`、`build_hitflow/gatestack_builder_projection_real_allstages/s2_c1/iverilog.log:1`。这是正常路径辅助证据，不关闭上述 P0。
5. 本次不修改 RTL/测试，不把其他代理运行中的 S3 产物冒充为本次独立动态签核。

## 10. 签核门槛

### P0 关闭条件

- Builder batch watchdog 能在有界时间内处理 token 中断、Builder 错误和 slot/serializer 永久 stall，并只返回一次带 tag 错误完成。
- 每个已接纳 group，包括活动期手工 abort，恰好返回一次 `group_done`；不允许静默丢弃。
- bias 接口可对接同步存储，身份可检查，数据宽度与 `ACC_W` 合同一致。
- 任何会触发 abort 的累加/协议错误不能向模块外提交 stale/overflow final。

### P1 关闭条件

- 对所有公开参数给出 elaboration assertion，或实现并回归真正可配变体。
- external-slot replay 契约明确为 reserve/commit pulse 或改为标准 decoupled ready-valid。
- slot release 加 payload tag/generation 匹配。
- 统一 wrapper abort 和 execution abort 的 error/counter/completion 语义。
- 以真实 SRAM macro 端口替换关键行为数组，完成目标库 lint、RDC、STA、DFT 和 mapped LEC。
- 冻结 H67 scale/requant/饱和/bias/residual 数值契约，并以多样本、多 block/window 扩大证据。

## 11. 最终判定

| 维度 | 判定 |
|---|---|
| 默认配置语法/lint | 通过 |
| 正常路径 ready/valid 保持 | 大部分通过，bias 和原子脉冲契约例外 |
| slot 正常生命周期 | 默认单 batch 结构下基本通过 |
| reset/abort 事务原子性 | 不通过 |
| 参数边界 | 不通过 |
| SRAM/macro 综合交付 | 不通过 |
| H67 候选 acc32 局部一致 | 已有有限正常路径证据 |
| 完整 H67 部署一致 | 不通过 |
| ASIC RTL 签核 | **不通过** |

## 12. 2026-07-20 同日整改复核

本节是原独立审阅后的增量复核，不删除原始发现。

| 原问题 | 整改状态 | 新证据 |
|---|---|---|
| P0-1 Builder 无 watchdog | 已关闭默认路径缺口 | wrapper 增加独立 BUILD_TIMEOUT_CYCLES；停顿定向测试只产生一次带原 tag error completion |
| P0-2 活动 group abort 静默丢事务 | 已关闭默认路径缺口 | wrapper 锁存 group tag，flush 后保持 group_done_error 至握手；manual abort 后可启动正常新 group |
| P0-3 bias 非同步存储接口且仅 PRODUCT_W | 已关闭接口缺口 | request 携带 tag/output_tile/token；response 携带 tag/token 与 OUT_TILE×ACC_W；至少晚一拍、可反压、错身份拒绝并重请求 |
| P0-4 overflow final 可逃逸 | 已关闭默认路径缺口 | accumulator 在 lane overflow 时抑制 final_valid，错误 commit 本地消费；定向测试 final_handshakes=0 |

同步 bias 接口额外覆盖：

- +200000/-300000 bias，明确超过有符号 17 bit；
- response 反压 4 cycle 时 payload 稳定；
- 错 tag/token response 不增加 bias commit，随后重新请求；
- single-head、multihead、decoder、single-context、Builder 完整 S0-S3、Direct RAW、G1 和 32/64/162 token 主线回归通过。

整改后真实 C0/C1 四 stage 为 211303/207213 cycle，每模式比较 233280 个 acc32，双模式总失配为 0。P0 已从“已知功能阻塞”降为“需要更广覆盖和目标库验证的签核风险”。

该复核不改变以下未关闭结论：P1 参数边界、slot generation release、真实 SRAM macro、scale/requant/residual/ATLIF、expanded trace、CDC/RDC/DFT、目标库 STA/SAIF/LEC 仍未签核，因此完整 ASIC RTL 仍不能宣布 sign-off。
