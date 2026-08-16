# TESSA 模块、接口、存储与 RTL 前规格

**日期**：2026-07-13  
**状态**：探索性 RTL 合同冻结，正式 architecture-to-RTL handoff 仍为 `NO-GO`  
**机器可读规格**：`spec/tessa_attention_subsystem_spec.json`  
**适用对象**：H67 Motion-XOR 功能超集、H68 编译期特化  
**替代文档**：早期 `docs/05_module_interface_spec.md` 已废止，不再作为实现依据

## 1. 顶层边界

第一版 RTL 顶层冻结为：

```text
tessa_attention_subsystem_top
```

覆盖：

- encoder 内 12 个 H67/H68 attention block 的 descriptor 驱动执行；
- temporal-pair 供数、精确 score、提交、class-stationary SCS 和 gated-K 输出；
- 1/2-context 实现，接口参数预留 4-context；
- block 静态配置、row completion 和性能计数器。

不覆盖：

- voxel/event 前端；
- projection、MLP、ATLIF 算术本体；
- decoder；
- AXI/DMA/APB/NoC；
- SRAM compiler 硬宏和 DFT wrapper。

因此在 projection/ATLIF/residual/S0-S2 skip 的周期和功耗模型完成前，论文只能称 **attention accelerator** 或 **encoder-attention subsystem**，不能称完整端到端光流芯片。

## 2. 时钟、复位与编译模式

### 2.1 时钟复位

| 项 | 第一版合同 |
|---|---|
| 时钟 | 单时钟 `clk_core` |
| 探索频率 | 500 MHz，不是正式签核值 |
| 复位 | `rst_n_core`，低有效、同步复位 |
| CDC | 顶层内部无 CDC |
| RDC | 单复位域；未来 SoC wrapper 负责复位同步 |
| 时钟门控 | 第一版使用 clock-enable 语义和活动计数，不实例化未知工艺 ICG |

正式 RTL skill 流程仍缺带 mW 的 `clock_power_budget` 和目标库 ICG 名称，因此不得把探索性 clock enable 写成已经完成的功耗签核。

### 2.2 编译期模式

| 参数 | H67 | H68 |
|---|---:|---:|
| `MOTION_XOR_ENABLE` | 1 | 0 |
| `SCORE_CLASS_DEPTH` | 35 | 3 |
| class scan pipeline | 2 拍/类 | 1 拍/类 |
| pair 布局 | `{Q0,Q1,K0,K1}` | 同一布局 |
| gate | 9-bit Q1.7 | 9-bit Q1.7 |

H67/H68 不使用运行时双模式 mux；分别综合编译配置，保证无用 Motion-XOR 和 32 个额外 class 逻辑可被常量传播删除。

## 3. 模块层次

```text
tessa_attention_subsystem_top                 # 只连线，不放复杂逻辑
├── tessa_descriptor_admission                # descriptor、空闲context、block barrier
├── tessa_pair_source_mux                     # 选择冻结的供数wrapper
│   ├── tessa_pair_source_128                 # 1x128-bit/拍
│   ├── tessa_pair_source_2x64                # 两个时间bank同拍
│   └── tessa_pair_source_1x64_fallback       # 两拍功能fallback
├── tessa_context_controller                  # 1/2/4 context生命周期
├── tessa_bitmap_pesf                         # 四位图到七个充分统计量
├── tessa_dual_score_q7                       # 同一pair两个score与RNE
├── tessa_pair_commit_router                  # active/hist分类、原子准入
│   ├── tessa_active_commit_fifo              # 深度2
│   ├── tessa_hist_commit_fifo                # 深度2
│   └── tessa_pccc_merge                      # 可旁路同class +2
├── tessa_context_state_bank[NUM_CONTEXTS]
│   ├── active_entry_mem                      # 162x56, 1W1R同步
│   ├── class_histogram                       # H67 35x8 / H68 3x8
│   ├── occupied_bitmap
│   └── row_state_and_descriptor
├── tessa_shared_scs_backend                  # max、occupied class、denominator、gate
├── tessa_sparse_output                       # gated-K流与completion流
└── tessa_performance_monitor                 # 架构证据计数器
```

设计边界：

- 顶层只做实例化和连线；
- PESF/score 为 datapath；
- descriptor/context/commit 为 control；
- SCS 独立于 pair front，允许 context 流水重叠；
- 第一版不加入 union-event、BMRF、OOO 或异构双核。

## 4. Descriptor 接口

采用 valid/ready：

```systemverilog
input  logic        desc_valid;
output logic        desc_ready;
input  logic [63:0] desc_data;
```

64-bit 字段：

| 字段 | bit | 说明 |
|---|---:|---|
| `stage` | 2 | S0-S3 |
| `block` | 3 | stage 内 block 0-5 |
| `head` | 5 | 最大 24 head |
| `window` | 10 | window 线性编号 |
| `row_tag` | 16 | 输入、context、输出和 completion 的唯一关联 |
| `n_pairs` | 7 | 当前固定 81，保留参数化检查 |
| `threshold_q8` | 8 | gated-K 阈值 |
| `preserve_mean` | 1 | 冻结软件合同 |
| `pccc_enable` | 1 | 消融与 bypass 验证 |
| `reserved` | 11 | 必须写零，非零触发状态错误 |

`desc_ready` 只有在以下条件同时满足时为 1：

1. 存在空闲 context；
2. 未处于 reset/flush；
3. 当前 block barrier 允许接收该 row；
4. `n_pairs==81`；
5. `row_tag` 未在飞。

第一版只允许 descriptor 输入顺序执行，不做 row OOO。

## 5. Pair 输入合同

### 5.1 逻辑 payload

```systemverilog
input  logic         pair_valid;
output logic         pair_ready;
input  logic [127:0] pair_payload; // {q0,q1,k0,k1}
input  logic [15:0]  pair_row_tag;
input  logic [6:0]   pair_index;   // 0..80
input  logic         pair_last;    // index==80
```

payload 固定映射：

```text
[127:96] Q0
[95:64]  Q1
[63:32]  K0
[31:0]   K1
```

第一版从 payload 内部精确计算 `pair_empty/k0_zero/k1_zero/motion_zero`，因此不能把 SRAM payload-read 节省记入收益。早到 metadata 必须作为后续独立 feature gate，并证明 metadata 在 payload 读取前可获得。

### 5.2 三种物理供数 wrapper

| wrapper | 行为 | 决策 |
|---|---|---|
| `tessa_pair_source_128` | 单个 128-bit word/拍 | 主候选 A |
| `tessa_pair_source_2x64` | time0/time1 两个 64-bit bank 同地址同拍 | 主候选 B |
| `tessa_pair_source_1x64_fallback` | 单口两拍 assembler | 功能 fallback，不作为高吞吐结果 |

`2x64` 推荐每个 bank 存 `{Q_t,K_t}`，相同 `pair_index` 访问两个时间 bank，避免同 bank 双读。

### 5.3 握手不变量

- `pair_valid && !pair_ready` 时 payload 和 sideband 必须稳定；
- 同一 row 的 `pair_index` 从 0 严格递增到 80；
- 只在 `pair_index==80` 接受 `pair_last`；
- row tag 必须匹配已分配 context；
- 每个 accepted pair 恰好产生两个 token 的数值语义。

## 6. PESF 与双 Score 接口

PESF 生成：

```text
q0, q1, k0, k1, overlap0, overlap1, motion
```

每项为 `0..32`，位宽 6。双 score 单元按 H67/H68 编译参数输出两个 16-bit Q7 score。

内部 pair-result 使用两个固定 64-bit slot：

| slot 字段 | bit |
|---|---:|
| `valid` | 1 |
| `kzero` | 1 |
| `score_q7` | 16 |
| `k_bits` | 32 |
| `token_index` | 8 |
| `class_index` | 6 |

两个 slot 合计 128 bit，并且必须原子接受：不能先提交 slot0、因队列满丢弃 slot1。

## 7. 双提交与 PCCC 合同

### 7.1 两条独立提交路径

```text
                    +-> active FIFO depth2 -> active bank 1W
dual pair result ---|
                    +-> hist FIFO depth2 -> class bank 1W
```

- mixed pair：一个 active、一个 K-zero，可同拍写两个 bank；
- 双 active：两个 active entry 顺序进入 active FIFO；
- 双 K-zero 不同 class：两个 histogram update 顺序进入 hist FIFO；
- 双 K-zero 同 class 且 PCCC 开启：一个 `{class,+2}` update；
- PCCC 关闭：始终保持两个 `{class,+1}` update。

### 7.2 原子 ready

定义本 pair 经过可选 merge 后需要：

```text
active_need = 0..2
hist_need   = 0..2
```

只有：

```text
active_fifo_free >= active_need
&& hist_fifo_free >= hist_need
```

时 pair-result 才能握手。禁止 partial commit 和回滚协议。

### 7.3 Histogram RMW 旁路

histogram 第一版使用小寄存器 bank：H67 `35x8`、H68 `3x8`。若当前 update class 与上一拍待写 class 相同，必须前递最新 count，避免连续同 class 的 read-modify-write 丢计数。

### 7.4 Token 守恒

每个 row 必须满足：

```text
active_committed + histogram_count_sum == 162
```

pair-empty 仍提交两个 class-2 语义；K-zero 只是不写 active bank，绝不能从 denominator 删除。

## 8. Context 生命周期

```text
FREE
 -> ALLOCATED
 -> PAIR_FILL
 -> COMMIT_DRAIN
 -> READY_SCS
 -> SCS_SCAN
 -> ACTIVE_REPLAY
 -> COMPLETE
 -> FREE
```

进入 `READY_SCS` 的条件：

- 接受 81 个 pair；
- 语义提交 162 个 token；
- active/hist FIFO 均为空；
- 无 score range/error；
- row max、histogram、occupied bitmap 已完成。

首版实现 `NUM_CONTEXTS=1/2`；代码参数和 tag 宽度必须支持 4。4-context 是否形成物理配置等待 ordered trace 和 SRAM 宏 DSE。

## 9. Context 存储

| 存储 | H67 逻辑容量/context | 端口 | 第一版实现 |
|---|---:|---|---|
| active entry | `162x56=9072 bit` | 1W1R，同步读1拍 | memory wrapper |
| histogram | `35x8=280 bit` | 1 update/拍 | register bank + bypass |
| occupied bitmap | 35 bit | 位更新/扫描 | register |
| row state | 不超过 256 bit | 本地 | register |
| descriptor | 64 bit | 本地 | register |

单 context 约 `9707 bit`，约 `1.185 KiB`，不含外围和宏向上取整。

约束：

- active bank 不允许异步 162:1 大 mux；
- SCS 状态显式加入同步读等待拍；
- 不把 35x8 histogram 强制映射成低利用率 SRAM；
- context 间不共享可写状态；
- reset 不要求逐拍清零整个 active bank，依靠有效计数和生命周期隔离。

## 10. 共享 SCS 与输出

SCS backend 顺序：

1. active bank 扫描求 row max；
2. occupied class 扫描和 multiplicity×exp2；
3. 形成整数 denominator；
4. active bank replay；
5. 产生 9-bit Q1.7 gate；
6. 发射稀疏 gated-K 描述。

输出：

```systemverilog
output logic        out_valid;
input  logic        out_ready;
output logic [15:0] out_row_tag;
output logic [7:0]  out_token_index;
output logic [31:0] out_k_bits;
output logic [8:0]  out_gate_q17;
output logic [7:0]  out_threshold_q8;
output logic        out_last_active;
```

全 K-zero row 没有 active 输出，必须通过独立 completion 流通知下游：

```systemverilog
output logic        completion_valid;
input  logic        completion_ready;
output logic [15:0] completion_row_tag;
output logic [7:0]  completion_active_entries;
output logic [3:0]  completion_status;
```

下游不得只监听 `out_last_active` 判断 row 完成。

## 11. 性能计数器

为了把 RTL 与 workload/DSE 对齐，至少实现：

| 计数器 | 用途 |
|---|---|
| `cycles_total` | frame/运行总周期 |
| `pairs_accepted` | 守恒与吞吐 |
| `pair_input_stall_cycles` | 供数或 context 反压 |
| `active_commit_stall_cycles` | 双 active 瓶颈 |
| `hist_commit_stall_cycles` | 双 class 瓶颈 |
| `pccc_same_class_merges` | PCCC 真实合并率 |
| `pccc_dual_class_pairs` | 双 class 比例 |
| `both_active_pairs` | active bank 双提交压力 |
| `context_occupancy_cycles[0:4]` | context 数 DSE |
| `scs_busy_cycles` | 后端利用率 |
| `output_stall_cycles` | 下游反压 |
| `block_barrier_drain_cycles` | block 尾部损失 |
| `max_row_latency_cycles` | 最坏 row 延迟 |

所有 counter 饱和而不回卷；frame start 清零；运行中不可由软件写。

## 12. 必须写入 SVA 的不变量

1. accepted descriptor 最终恰好产生一次 completion；
2. accepted pair 恰好贡献两个 token 语义；
3. pair-result 不发生 partial commit；
4. active + histogram multiplicity 恒为已提交 token 数；
5. `READY_SCS` 时已提交 token 数为 162 且两个 FIFO 为空；
6. K-zero 不进入 active bank，但必须进入 histogram；
7. active token 最多输出一次；
8. context 状态不跨 row tag 污染；
9. valid 被反压时所有 payload/sideband 稳定；
10. completion 前不得释放 context；
11. PCCC on/off 最终 histogram、denominator 和输出逐位相同；
12. 任意反压下无丢失、重复、死锁。

## 13. 时钟门控规划边界

目前只有 workload 机会代理，没有 SAIF mW：

| 域 | 机会代理 | 初步分类 | RTL 要求 |
|---|---:|---|---|
| Motion-XOR 后级 | motion-zero约83% | 高 | H68编译删除；H67注册 enable |
| active write | active token约11% | 高 | FIFO/bank clock-enable |
| pair payload | pair-empty约74% | 条件高 | 第一版仍读payload，不记SRAM节省 |
| histogram | K-zero约83% | 高活动 | 不可整体关，仅做无update门控 |
| 空闲 context | ordered occupancy未知 | 未定 | 每context独立 enable |
| SCS | front/backend比约0.55 | 中 | READY context为空时关闭 |

正式 ICG 实例化必须等待标准单元库和 `clock_power_budget`；第一版只输出 clock-enable 意图和活动 counter。

## 14. 三档编译配置

| 配置 | 内容 | 用途 |
|---|---|---|
| `TESSA_A1` | 128-bit fixed bitmap、1 context、PCCC bypass | 正确性/最小面积基线 |
| `TESSA_B2` | 128-bit或2x64、2 context、PCCC可开关 | 受控主线 |
| `TESSA_B4_MODEL` | 参数4、只完成elaboration和模型接口 | 等ordered trace，不做正式物理结果 |

禁止在第一版加入 BMRF、union-event、row OOO、方向 bank mapping 或双核。

## 15. 探索性 RTL 准入关闭情况

对应 `docs/56` 第 12.2 节：

| 检查项 | 当前状态 | 证据 |
|---|---|---|
| 顶层边界 | 已冻结 | 本文第1节 |
| pair input 合同 | 已冻结 | 第5节与机器规格 |
| 双结果 commit 合同 | 已冻结 | 第7节 |
| hardware-order golden | 已有 | `docs/49`及现有H67/H68参考脚本 |
| context memory map | 已冻结逻辑合同 | 第8-9节；物理宏未冻结 |
| block descriptor | 已冻结最小字段 | 第4节 |
| trace/cycle counter | 已冻结 | 第11节 |

这意味着候选 A/B 可以进入 **探索性 RTL module planning**。仍不意味着正式 handoff 或 DC 签核，因为 ordered trace、SRAM 宏、目标 PDK/PVT、clock power budget 和 encoder 级模型尚未完成。

## 16. 自动校验

运行：

```bash
/opt/conda/envs/sdformerflow/bin/python \
  hw_autoresearch_nts07/scripts/validate_tessa_attention_spec.py
```

校验：

- 162 token 等于 81 pair 的双时间片；
- descriptor 字段合计 64 bit；
- pair payload 合计 128 bit；
- pair-result slot 合计 64 bit；
- active entry 为 `score16+K32+token8=56 bit`；
- H67/H68 class 深度为 35/3；
- context 参数为 1/2/4、首版为2；
- active bank 深度和 row completion 守恒。
