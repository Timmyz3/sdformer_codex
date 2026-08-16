# GateStack 公平基线与真实 Trace RTL 实施规格

## 一、文档状态

本文件是 `docs/97` P0 项目的 RTL 实施规格，不是实现报告或签核报告。

| 项目 | 当前状态 |
|---|---|
| 现有 GateStack 单 context execution slice | 已有 RTL 与既有回归证据 |
| H67 真实 Q/K、Q1.7 gate、投影权重 trace 采集和审计 | **一个样本、四stage首block首window已完成并通过审计** |
| 真实 trace 到 IPD32W/RAW41/MEMH 的 RTL 打包器 | **已实现并生成四stage向量** |
| RAW41-only 同顶层周期基线 | **主线程后续已实现并通过默认规模RTL回归；物理裁剪版仍待实现** |
| IPD no-residency 公平基线 | **主线程后续已实现并通过默认规模RTL回归** |
| head-major partial-sum spill 公平基线 | **待实现** |
| 四 stage 真实 trace RTL 回放 | **RAW/no-residency/GateStack共12组全部零mismatch** |
| 目标库 DC/STA/SAIF/LEC | 未执行 |

本规格子 agent 输出时未修改 RTL、TB 或脚本。随后主线程已完成第一批真实 trace 采集链、no-residency 编译变体和 RAW41-only 同顶层周期消融；结果与证据边界见 `docs/100_DATE审稿补强第一轮_真实Trace链与公平基线RTL消融_20260717.md`。下文中未被 `docs/100` 明确晋级的模块、断言、测试和结果仍属于待实现规格。

## 二、审计依据与边界

本规格遵循 `chip-design-rtl` 的模块规划、接口冻结、lint、CDC/RDC 和综合准入要求，并参考 `functional-verification` 与 `logic-synthesis` 的签核标准。

当前综合约束仍有以下硬缺口：

- 500 MHz 只是探索目标，尚未由目标工艺确认；
- 面积和功耗预算没有架构签核；
- 目标标准单元库、PVT、SRAM macro 和 ICG 单元未提供；
- 因此本轮不能进入正式 `synth_check` 或 `rtl_signoff`。

现有硬件边界是 projection execution slice，不包括完整 H67 encoder、在线 SCS builder、ATLIF、skip/residual、最终 requant 和外存系统。

## 三、现状审计结论

### 3.1 已有顶层接口足以建立同接口基线

现有 `gatestack_single_context_execution_top` 的功能接口可分为六组：

| 接口组 | 作用 |
|---|---|
| `clk_core/rst_core` | 单时钟域、同步高有效复位 |
| `group_*` | 提交一个窗口投影 group，给出 head 数和 output tile 范围 |
| `payload_commit_*` | 按 head 写入 64-bit IPD32W 或 RAW41 payload |
| `descriptor_fill_*` | 可选外部 descriptor 预填充 |
| `weight_req/rsp_*` | 按 input channel 和 output tile 读取 32-lane INT8 权重向量 |
| `bias_req_*`、`final_*` | bias 供给和 32-bit accumulator 结果输出 |
| `count_*`、`protocol_error` | 运行时审计计数和协议错误 |

三个基线必须保持相同参数列表和上述端口方向、宽度、握手语义。不能为某个候选增加更宽的权重端口、更深的 FIFO 或更低的反压概率。

### 3.2 当前主数据流

```text
payload commit
    -> 24 个固定容量 head slot
    -> output-tile 外层、input-head 内层 scheduler
    -> slot/cache metadata PLAN
    -> 原子 COMMIT
    -> Resident / IPD32W / RAW41 decoder
    -> 统一 TDR + multicast + AccTile
    -> bias once
    -> final stream
```

默认最大配置的主要逻辑存储为：

| 存储 | 逻辑容量 |
|---|---:|
| head slot | `24 × 104 × 64 = 159,744 bit` |
| Depth80 descriptor cache | `24 × 80 × 24 = 46,080 bit` |
| 单个 32-lane AccTile | `162 × 32 × 32 = 165,888 bit` |

这些只是逻辑位数，不代表 SRAM macro 面积。

### 3.3 当前 trace 回放的证据限制

已有 stage3 回放只继承真实 profile 的有序 term、event、class 和 max-fanout 数量，gate/lane/token 内容是确定性构造；权重固定为 `1`，bias 是 TB 生成值，没有网络 requant。

因此它可以证明控制、route、生命周期和构造整数结果正确，但不能证明：

- 真实 H67 网络逐 bit 等价；
- 真实 INT8 权重和 bias 下的 accumulator 正确；
- 四个 stage 的延迟和活动分布；
- RTL VCD 可以直接代表门级功耗。

### 3.4 真实 trace 采集代码的当前状态

主线程已经实现以下采集路径，当前应保留并继续收口：

- attention 中的 bit-trace collector hook；
- profile 入口的 `--bit-trace-dir`、sample/window/all-block 选项；
- `AttentionBitTraceWriter`；
- 真实 Q/K bit、Q1.7 gate、checkpoint projection weight、候选 dyadic INT8 编码和 bias accumulator 编码；
- 对应的基础单元测试代码。

主线程随后已经生成可引用的四 stage 真实 trace、RTL 向量和整数金参考，并完成三路径回放。结果见 `docs/101_H67真实四Stage消融与GateStack架构再冻结_20260717.md`。覆盖范围仍只有一个样本、首block、首window，不能替代100-frame/all-block功耗与尾延迟主表。

此外还需审计：projection BN 是否启用或已正确折叠、逐输出通道 scale 如何进入 bias/requant、`bias_acc` 是否超出现有 17-bit bias 端口。

### 3.5 RTL skill 视角的现有缺口

1. 现有 execution top 内仍有协议聚合、fill 仲裁和若干控制逻辑，不是纯 wiring top。新基线不得继续复制这些逻辑，应放入独立 leaf control/status 模块。
2. 现有子模块 SVA 较多，但没有三个基线和完整 execution contract 的统一顶层 checker。
3. 尚无正式 functional/code/FSM/toggle coverage 报告。
4. 当前没有 library-approved ICG。正式功耗比较前必须按真实活动分类并统一插入或统一关闭时钟门控。
5. 现有 `bias_req_values` 宽度为 `OUT_TILE × PRODUCT_W`，真实部署 bias/requant 合同未冻结。

## 四、公平比较合同

### 4.1 固定不变项

所有候选必须固定：

- `TOKENS=162`；
- 物理最大 `HEADS=24`，运行时 stage head 数为 `3/6/12/24`；
- `LANES=32`、`OUT_TILE=32`；
- `GATE_W=9`、`WEIGHT_W=8`、`ACC_W=32`；
- `EVENT_WAYS=4`、`BANKS=2`；
- 相同 head-slot 容量和 64-bit payload 端口；
- 相同 weight request/response 延迟、带宽和反压序列；
- 相同 bias、final 反压序列；
- 相同目标库、PVT、频率、IO delay、fanout 和 transition 约束；
- 相同真实 trace、权重、bias、舍入和饱和合同；
- 相同 reset、abort 和错误处理边界。

不得只对 GateStack 使用真实稀疏 trace，而让基线使用平均密度；不得把逻辑 SRAM 对某候选算面积、对另一候选只算数据移动。

### 4.2 三段延迟口径

必须同时报告：

| 指标 | 起止点 |
|---|---|
| `L_commit` | 第一个 payload commit begin 到最后一个 head payload commit 完成 |
| `L_execute` | `group_valid && group_ready` 到 `group_done_valid && group_done_ready` |
| `L_cold_total` | `L_commit + L_execute` |

GateStack 内部首 tile 的 IPD decode 和 promotion 计入 `L_execute`。不能只报告 warm tile，也不能隐藏 payload 构建或提交成本。

### 4.3 数据移动口径

每个候选分别统计：

- payload commit/replay 的 64-bit word 数；
- descriptor cache read/write word 数；
- weight request/response byte 数；
- product 数和 destination 数；
- AccTile read/write 数；
- partial-sum SRAM read/write byte 数；
- bias read 和 final write byte 数；
- FIFO stall、bank conflict、p50/p95/p99 group latency。

payload word 减少不能直接写成能耗减少。能耗必须来自同库 mapped SAIF。

## 五、基线一：Direct RAW41-only

### 5.1 目的

该基线回答：在保留 output-tile-stationary 调度和统一投影后端时，IPD 等价类压缩、descriptor cache 和 promotion 是否真正有收益。

### 5.2 数据流

```text
真实 gate[8:0] + K bitmap[31:0]
    -> 每 token 一个 41-bit RAW record
    -> 固定 6,642-bit/head slot
    -> 每个 output tile 完整 RAW replay
    -> RAW decoder
    -> 与 GateStack 相同的 TDR/multicast/AccTile
```

### 5.3 必须保持的条件

- scheduler 仍是 output tile 外层、head 内层；
- 权重、bias、AccTile、multicast、bank 数与主方案完全相同；
- 所有 head 均提交 RAW41，`payload_commit_mode_is_csr` 必须为 0；
- 每个 head payload 必须严格为 `162 × 41 = 6,642 bit`，即 104 个 64-bit word；
- 不实例化 IPD decoder、resident decoder、descriptor cache 或 auto-fill；
- `descriptor_fill_*` 端口保留，但必须通过独立 rejector 明确标记为 unsupported，不能静默接受。

### 5.4 模块方案

```text
gatestack_raw41_only_execution_top             // 纯连接顶层
  gatestack_output_tile_scheduler              // 复用
  gatestack_head_slot_sram_adapter             // 复用
  gatestack_fixed_policy_plan_builder          // 新增，RAW_ONLY
  gatestack_baseline_replay_control_plane_top  // 新增
  gatestack_raw41_only_projection_top          // 新增
    gatestack_raw41_replay_decoder             // 复用
    gatestack_raw_tail_retimer                  // 复用
    gatestack_raw_issue_adapter                 // 复用
    gatestack_tdr_multicast_backend             // 复用
    hitflow_banked_accumulator                  // 复用
  gatestack_descriptor_fill_rejector            // 新增
  gatestack_execution_status_aggregator         // 新增
```

### 5.5 时序合同

每个 head session 依次执行：slot inspect、RAW plan、atomic commit、104-word replay、RAW record decode、active lane issue、backend drain、head done。最后一个 head 后才进入 bias/final。

预期审计关系：

```text
count_cache_hits     = 0
count_cache_releases = 0
count_slot_replays   = active_heads × output_tiles
count_head_issues    = active_heads × output_tiles
```

## 六、基线二：IPD no-residency

### 6.1 目的

该基线隔离 decode-once promotion 和跨 output tile descriptor residency 的收益。它保留 IPD32W 压缩和 RAW41 容量安全 fallback，但每次 output tile 都重新解码完整 payload。

### 6.2 数据流

```text
容量选择
  IPD32W -> 每个 tile 从 header/descriptor/token 起点完整重放
  RAW41  -> 每个 tile 完整重放
            |
            v
       统一 TDR/multicast/AccTile
```

### 6.3 必须保持的条件

- 与主方案相同的 output-tile-stationary 调度；
- IPD 和 RAW 使用与主方案相同的序列化格式和 fallback 判定；
- 不实例化 descriptor cache；
- 不产生 resident route；
- 不产生 IPD auto-fill；
- CSR head 每个 tile 从 word0 开始重放，包括 header 和 descriptor；
- RAW head 行为与 Direct 基线一致；
- 空 IPD head 仍需完整解析合法空 header 并退休，不能直接跳过 session。

### 6.4 模块方案

```text
gatestack_ipd_no_residency_execution_top       // 纯连接顶层
  gatestack_output_tile_scheduler              // 复用
  gatestack_head_slot_sram_adapter             // 复用
  gatestack_fixed_policy_plan_builder          // 新增，IPD_RAW_NO_RESIDENCY
  gatestack_baseline_replay_control_plane_top  // 新增
  gatestack_no_residency_decoder_projection_top// 新增
    gatestack_ipd32w_replay_decoder            // 复用
    gatestack_raw41_replay_decoder             // 复用
    gatestack_raw_tail_retimer                  // 复用
    gatestack_raw_issue_adapter                 // 复用
    gatestack_replay_mux                        // 复用，两路配置
    gatestack_tdr_multicast_backend             // 复用
    hitflow_banked_accumulator                  // 复用
  gatestack_descriptor_fill_rejector            // 新增
  gatestack_execution_status_aggregator         // 新增
```

预期审计关系：

```text
count_cache_hits     = 0
count_cache_releases = 0
count_slot_replays   = active_heads × output_tiles
resident_route_count = 0
```

## 七、基线三：head-major partial-sum spill

### 7.1 目的

该基线只改变 loop order，用于证明 head-stacked output-tile-stationary 的价值。它必须保留与主方案相同的 IPD/RAW 表示、Depth80 descriptor 驻留、共享 TDR/multicast 和 32-lane计算后端。

### 7.2 调度顺序

```text
for input_head:
    decode/promote descriptor once
    for output_tile:
        if input_head == 0:
            clear local AccTile
        else:
            load partial sum from PSUM SRAM
        replay current head
        if input_head != last_head:
            spill partial sum to PSUM SRAM
        else:
            add bias and emit final
```

### 7.3 partial-sum 存储

逻辑容量为：

```text
active_output_tiles × TOKENS × OUT_TILE × ACC_W
```

最大 stage3 配置需要：

```text
24 × 162 × 32 × 32 = 3,981,312 bit = 486 KiB
```

不含 SRAM 控制和 ECC。每窗口理论 spill 流量为：

```text
2 × (H - 1) × H × 162 × 32 × 4 byte
```

| stage | H | PSUM 容量 | 每窗口 read+write 流量 |
|---:|---:|---:|---:|
| S0 | 3 | 60.75 KiB | 248,832 byte |
| S1 | 6 | 121.5 KiB | 1,244,160 byte |
| S2 | 12 | 243 KiB | 5,474,304 byte |
| S3 | 24 | 486 KiB | 22,892,544 byte |

这些是结构公式，不是仿真或功耗结果。

### 7.4 模块方案

```text
gatestack_headmajor_spill_execution_top
  gatestack_headmajor_scheduler                // 新增，head 外层/tile 内层
  gatestack_head_slot_sram_adapter             // 复用
  gatestack_descriptor_residency_cache         // 复用
  gatestack_replay_control_plane_top           // 复用或薄适配
  gatestack_headmajor_spill_projection_top     // 新增
    Resident/IPD32W/RAW41 frontends            // 复用
    gatestack_tdr_multicast_backend             // 复用
    gatestack_spillable_accumulator             // 新增
    gatestack_partial_sum_sram_adapter          // 新增
  gatestack_execution_status_aggregator         // 新增
```

### 7.5 partial-sum 接口

`gatestack_partial_sum_sram_adapter` 至少包含：

| 端口组 | 字段 |
|---|---|
| load request | `valid/ready/tag/output_tile` |
| load stream | `valid/ready/token_id/values[OUT_TILE*ACC_W]/last` |
| store request | `valid/ready/tag/output_tile` |
| store stream | `valid/ready/token_id/values[OUT_TILE*ACC_W]/last` |
| status | `protocol_error/count_load_words/count_store_words/count_stall_cycles` |

同一 output tile 在同一时刻只能有一个 load 或 store session。load/store 地址必须由 `{output_tile, token_id}` 唯一确定。

### 7.6 时序相位

```text
IDLE
 -> LOAD_PSUM（head>0）
 -> START_HEAD
 -> RUN_HEAD
 -> STORE_PSUM（非末head）
 -> BIAS_AND_FINAL（末head）
 -> DONE
```

禁止 load 与当前 head update 覆盖同一 AccTile entry；禁止非末 head 发出 final；禁止末 head 再写 PSUM。

## 八、真实四 stage bit trace 格式

### 8.1 覆盖集合

H67 共 12 个 attention block：

```text
S0: B0, B1
S1: B0, B1
S2: B0, B1, B2, B3, B4, B5
S3: B0, B1
```

P0 最小真实集合必须覆盖一个真实样本、每个 block 至少一个完整窗口，共 12 个 case。仅四个 `B0` 可作为 smoke，不足以关闭 P0。

论文评估集合应再增加每 block 的 p50、p95 和最大 term/fanout 窗口，并覆盖空 head、IPD/RAW 混合、Depth80 bypass 和 gate 0/256。

### 8.2 两层文件格式

第一层是软件 canonical trace，建议继续使用当前 NPZ writer：

| 数组 | 形状/类型 | 说明 |
|---|---|---|
| `q_bits_packed` | packed bit | 审计 H67 输入，不直接驱动 projection |
| `k_bits_packed` | `[2,W,H,81,32]` packed bit | 真实二值 K |
| `gate_q17` | `[W,H,162] uint16` | 真实 0..256 Q1.7 gate code |
| `projection_weight_float32` | `[Cout,Cin]` | checkpoint 权重审计 |
| `projection_weight_int8` | `[Cout,Cin] int8` | 候选 dyadic 部署权重 |
| `projection_weight_scale_exp2` | `[Cout] int16` | 每输出通道 2 幂 scale |
| `projection_bias_float32` | `[Cout]` | checkpoint bias |
| `projection_bias_acc_int64` | `[Cout]` | 候选 accumulator 域 bias |

第二层是 CPU 打包器生成的 RTL vector 目录：

```text
case_dir/
  manifest.json
  adaptive_payload_words.memh
  adaptive_payload_offsets.memh
  adaptive_payload_bits.memh
  adaptive_payload_modes.memh
  adaptive_payload_word_counts.memh
  raw_payload_words.memh
  weight_i8x32.memh
  bias_acc32x32.memh
  expected_acc32x32.memh
  requant_params.memh          // 数值合同冻结后启用
  expected_requant.memh        // 数值合同冻结后启用
```

### 8.3 展平和端序

- 所有 MEMH 使用 ASCII 十六进制，一行一个逻辑 word；
- 所有复合 word 使用 little-lane packing，lane0 位于最低位；
- `weight_i8x32` 索引为 `output_tile × (HEADS_MAX×32) + input_channel`；
- 每行 256 bit，output lane0 位于 `[7:0]`；
- `expected_acc32x32` 索引为 `output_tile × TOKENS + token_id`；
- 每行 1024 bit，output lane0 位于 `[31:0]`；
- RAW41 每 token record 为 `{gate[8:0], K[31:0]}`，token0 位于最低有效 bit；
- IPD descriptor 固定为 `{reserved[9:0], destination_count[7:0], lane[4:0], gate[8:0]}`；
- destination token 必须升序，descriptor 必须按 `(lane_id, gate_code)` 稳定排序；
- manifest 必须记录 schema、checkpoint/config/case 文件 SHA256、stage/block/sample/window、维度、量化合同和所有文件 SHA256。

### 8.4 打包器必须完成的检查

1. 从 canonical K/gate 独立构造 RAW41 和 adaptive IPD/RAW 两套 payload。
2. 解包两套 payload，逐 token/lane 与 canonical K/gate 比较。
3. 使用 INT8 权重和 accumulator bias 生成 `expected_acc32x32`。
4. 检查所有 accumulator 值不溢出 signed 32-bit。
5. 检查 `bias_acc` 是否能被最终 RTL bias 端口表示；不能表示则硬失败。
6. 如果 projection BN 实际启用，必须先正确折叠或把 trace 标记为无效。
7. 当前 requant 未冻结时只允许声明 pre-requant accumulator bit-exact。

## 九、TB 接入方案

### 9.1 TB 结构

```text
tb_gatestack_fair_real_trace
  group_driver
  payload_driver
  weight_responder
  bias_responder
  final_monitor
  integer_scoreboard
  coverage_collector
  protocol_assertion_bind
```

采用同一 TB，通过 elaboration 参数选择：

```text
DUT_KIND=0  GateStack Depth80
DUT_KIND=1  RAW41-only
DUT_KIND=2  IPD no-residency
DUT_KIND=3  head-major spill
```

四种 DUT 必须连接同一组 interface signal，不能使用不同 testbench。

### 9.2 驱动流程

1. 从 plusarg 读取 `VECTOR_DIR` 和 manifest。
2. 校验 manifest schema、维度和 SHA256。
3. RAW 基线提交 `raw_payload_*`，其余候选提交 `adaptive_payload_*`。
4. 使用相同 seed 产生 weight response、bias ready、final ready 和 payload backpressure。
5. group head/tile 数按 stage 取 `3/6/12/24`，硬件仍使用最大 24-head 配置。
6. 每个 final handshake 直接与 `expected_acc32x32` 比较。
7. 记录 commit、execute、total cycle 和所有物理事件计数。

### 9.3 必需测试

| 测试 | 目标 |
|---|---|
| `real_all12_blocks_no_stall` | 真实四 stage、12 block 基本 bit-exact |
| `real_all12_blocks_backpressure` | payload/weight/final 随机反压 |
| `raw_only_all_heads` | RAW 固定格式和 104-word 边界 |
| `ipd_no_residency_every_tile` | 每 tile 从 IPD word0 完整重解码 |
| `mixed_ipd_raw` | 容量安全 fallback |
| `headmajor_spill_load_store` | PSUM load/store 次数与结果 |
| `empty_head` | 空 IPD head 正确退休 |
| `gate_0_256` | gate 边界和有符号乘积 |
| `reset_during_commit` | commit 活跃时 reset |
| `reset_during_execute` | decoder/backend/PSUM 活跃时 reset |
| `tag_corruption` | payload、weight、execution tag 错配拒绝 |
| `ten_seed_regression` | 至少 10 个独立反压 seed，零 mismatch |

## 十、SVA 计划

### 10.1 共用顶层断言

- 所有 ready/valid payload 在 stall 时保持稳定；
- 同一时间最多一个 group outstanding；
- group accept 后必须在有界时间内 done 或 error-abort；
- weight response tag/input-channel/output-tile 必须匹配 request；
- final stall 时 tag、token 和 values 保持稳定；
- 每个 `{tile,token}` 只 final 一次；
- 每个 tile 的 bias commit 恰好 `TOKENS` 次；
- `protocol_error` 和 accumulator overflow sticky；
- reset 后所有 session、valid 和 ownership 状态清零。

### 10.2 RAW41-only 断言

- accepted payload 必须 `mode_is_csr==0`；
- payload bits 必须等于 6,642；
- route 永远为 RAW；
- resident/IPD/cache 事件永不发生；
- 每 session 恰好消费 162 个 RAW record。

### 10.3 IPD no-residency 断言

- route 只能是 IPD 或 RAW；
- resident route 永不发生；
- CSR replay start word 永远为 0；
- cache hit/fill/release 永不发生；
- 每个 output tile 都重新出现 descriptor begin。

### 10.4 head-major spill 断言

- issue 顺序严格为 head 外层、tile 内层；
- `head=0` 不发 load；
- `head>0` 每 tile 恰好一次完整 load；
- 非末 head 每 tile 恰好一次完整 store；
- 末 head不 store，只 bias/final；
- store 后才能切换到下一 tile；
- 同一 PSUM 地址不存在同拍冲突；
- load/store token 必须从 0 到 161 且无重复、无遗漏。

## 十一、覆盖率准入

| 覆盖类型 | 准入目标 |
|---|---:|
| 功能需求覆盖 | 100% |
| P0 directed test | 100% PASS |
| FSM state | 100% |
| FSM transition | 不低于 95% |
| assertion triggered | 100% |
| line coverage | 不低于 95% |
| branch coverage | 不低于 90% |
| toggle coverage | 不低于 85% |
| 随机回归 | 至少 10 seeds，100% PASS |

必须建立以下 cross：

```text
stage × block × representation
representation × empty/nonempty
representation × backpressure
gate_boundary × weight_sign
cache_depth × hit/miss/bypass
loop_order × first/middle/last_head
loop_order × first/middle/last_tile
```

不可达 bin 必须逐项记录原因和批准人，不能直接删除。

## 十二、综合与消融矩阵

### 12.1 同约束候选

| ID | 候选 | 主要隔离变量 |
|---|---|---|
| B0 | RAW41-only、tile-major | 无 IPD、无 residency |
| B1 | IPD/RAW、tile-major、no-residency | 有压缩，无 promotion/cache |
| B2 | IPD/RAW、head-major、Depth80 | 改 loop order，显式 PSUM spill |
| G0 | GateStack tile-major、Depth0 | 与 B1 交叉校验 |
| G64 | GateStack tile-major、Depth64 | cache 容量消融 |
| G80 | GateStack tile-major、Depth80 | 当前主候选 |

`B1` 与 `G0` 功能和结构应接近；若结果显著不一致，说明比较边界或综合裁剪不公平。

### 12.2 每个候选必须报告

- mapped combinational/sequential 面积；
- 每个 SRAM macro 的容量、端口、实例数、利用率和面积；
- WNS、TNS、unconstrained path、max transition/fanout violation；
- mapped SAIF 注释率；
- dynamic、leakage、clock、memory、datapath、control 功耗；
- `L_commit/L_execute/L_cold_total`；
- energy/group、EDP、payload byte、PSUM byte、weight byte；
- netlist LEC 结果；
- p50/p95/p99 和最坏 trace。

### 12.3 正式综合准入条件

在运行 DC/Genus 前必须冻结：

1. `clk_mhz`、area budget、power budget；
2. 标准单元库和 SS/TT/FF PVT；
3. head-slot、descriptor、AccTile、PSUM 的 SRAM macro；
4. 真实 bias/requant 位宽；
5. ICG 单元和 scan enable 规则；
6. 所有输入输出 delay、load 和 transition；
7. 所有候选相同的 compile 和 hierarchy policy。

## 十三、最小第一批代码改动

第一批目标不是一次实现三个基线，而是先关闭真实 trace 和两个低风险表示基线。

### 13.1 批次 A：真实 trace 收口

| 文件 | 动作 | 所有者 |
|---|---|---|
| 现有 `h67_bit_trace.py` | 补 BN/fold 审计、expected accumulator、manifest 完整性 | 软件 trace owner |
| 现有 profile 入口和 attention hook | 只做必要修复，不重写主线程实现 | 软件 trace owner |
| `scripts/pack_gatestack_real_trace_vectors.py` | 新增 NPZ 到 IPD/RAW/MEMH 打包 | trace-pack owner |
| `scripts/test_pack_gatestack_real_trace_vectors.py` | 新增 round-trip 和金参考测试 | trace-pack owner |
| `tb_hitflow/tb_gatestack_fair_real_trace.sv` | 新增共用真实 trace TB | verification owner |

GPU 任务只负责生成真实 trace；打包、TB 和 RTL 仿真均应 CPU-only。

### 13.2 批次 B：RAW 与 no-residency

| 新文件 | 所有者 |
|---|---|
| `rtl_hitflow/gatestack_fixed_policy_plan_builder.sv` | baseline RTL owner |
| `rtl_hitflow/gatestack_baseline_replay_control_plane_top.sv` | baseline RTL owner |
| `rtl_hitflow/gatestack_no_residency_decoder_projection_top.sv` | baseline RTL owner |
| `rtl_hitflow/gatestack_raw41_only_execution_top.sv` | baseline RTL owner |
| `rtl_hitflow/gatestack_ipd_no_residency_execution_top.sv` | baseline RTL owner |
| `rtl_hitflow/gatestack_descriptor_fill_rejector.sv` | baseline RTL owner |
| `rtl_hitflow/gatestack_execution_status_aggregator.sv` | baseline RTL owner |
| 对应 `verif_hitflow/*assertions.sv` 与 bind | verification owner |
| 两个 baseline filelist 和 run script | integration owner |

现有 `gatestack_single_context_execution_top.sv` 和已通过回归的 leaf RTL 在该批次冻结，未经代码所有者确认不得编辑。

### 13.3 批次 C：head-major spill

| 新文件 | 所有者 |
|---|---|
| `rtl_hitflow/gatestack_headmajor_scheduler.sv` | spill RTL owner |
| `rtl_hitflow/gatestack_partial_sum_sram_adapter.sv` | spill RTL owner |
| `rtl_hitflow/gatestack_spillable_accumulator.sv` | spill RTL owner |
| `rtl_hitflow/gatestack_headmajor_spill_projection_top.sv` | spill RTL owner |
| `rtl_hitflow/gatestack_headmajor_spill_execution_top.sv` | spill RTL owner |
| 对应 SVA、TB、filelist 和 regression | verification/integration owner |

head-major 不应与批次 B 同时修改共享模块，避免多个 agent 争用控制面和 accumulator 文件。

## 十四、文件所有权规则

1. 每个新文件只有一个写 owner；reviewer 只提交问题清单，不直接改 owner 文件。
2. 现有主线 RTL 默认只读，优先通过新 wrapper/leaf 建立基线。
3. trace owner 不修改 RTL；RTL owner 不修改网络 checkpoint、训练配置或 profiler 主逻辑。
4. verification owner 不在 TB 内复制 DUT 算法，金参考必须来自独立 Python/NPZ 产物。
5. synthesis owner 只维护 filelist、SDC、macro mapping 和报告脚本，不改功能 RTL。
6. 任何共享接口变化必须先更新本规格，再由接口 owner 合入。

## 十五、阶段退出条件

### P0-A 真实 trace 完成

- GPU 生成 12 block 真实 trace；
- manifest `four_stage_complete=true` 且 all-block coverage 完整；
- NPZ SHA256、shape、Q/K/gate 范围审计通过；
- IPD/RAW round-trip 与 canonical trace 零 mismatch；
- 真实权重/bias accumulator 金参考可生成。

### P0-B 两个表示基线完成

- RAW41-only 与 IPD no-residency 同接口顶层实现；
- Icarus/Verilator lint/assert 全通过；
- 真实 12-block trace accumulator 零 mismatch；
- 至少 10 个 backpressure seeds 全通过；
- 公平 cycle/byte 分账生成。

### P0-C loop-order 基线完成

- head-major scheduler、spill accumulator、PSUM SRAM adapter 实现；
- load/store 守恒和 bit-exact 通过；
- stage3 22,892,544-byte 理论值与 RTL event ledger 对齐；
- 与 GateStack 使用相同 trace 和后端参数。

### P0-D 综合准入

- 三个基线和 GateStack lint/CDC/RDC/coverage 达标；
- 目标库、预算、SRAM macro、ICG、requant 合同冻结；
- 才允许进入同库 DC/STA/mapped SAIF/netlist LEC。

## 十六、当前允许和禁止的表述

当前允许：

- “真实 trace 采集和审计代码已由主线程实现，待 GPU 运行。”
- “三个公平基线的接口、数据流、验证和综合规格已经冻结。”
- “现有 GateStack 是单 context projection execution slice。”

当前禁止：

- “真实四 stage trace 已完成。”
- “RAW41-only、IPD no-residency 或 head-major baseline 已实现。”
- “真实 H67 网络 RTL bit-exact 已通过。”
- “GateStack 相对公平基线获得某个 speedup 或 energy reduction。”
- “已完成 DC、500 MHz、PPA 或 DATE 投稿签核。”

## 十七、执行顺序

```text
GPU 生成真实 12-block trace
  -> CPU 打包与独立金参考
  -> 共用 TB 接入现有 GateStack
  -> RAW41-only
  -> IPD no-residency
  -> head-major spill
  -> 真实 trace 十种子回归和覆盖率
  -> 同约束 Yosys 结构审计
  -> 目标库/预算/SRAM 冻结
  -> DC/STA/mapped SAIF/netlist LEC
```

该顺序先解决 `docs/97` 最关键的“真实输入和公平基线”问题，再进入 PPA。继续增加主线控制 RTL但不补真实 trace 和 baseline，不能提高 DATE 证据等级。
