# GateStack-CSR 容量安全存储与架构主线修正（2026-07-15）

> **主线替换提示（2026-07-15）**：本文的 `header192 + descriptor35` 已被 IPD32W 与 Depth=80 descriptor residency 替换。当前权威口径见 `docs/87_GateStack_IPD32W有界驻留与无损双格式架构收口_20260715.md`；本文仅保留 bitmap→容量安全 slot 的设计演化。

## 1. 为什么必须修正 bitmap 主线

旧 GateStack 为每个 head 保存：

```text
destination_bitmap[slot=4][lane=32][token=162]
```

位数只有 20,736 bit，但真实 SRAM 端口并不自然：

- SCS 输入按 token 到达，一周期带 32-bit K，需要同时更新最多 32 个 lane 的单 bit；
- replay 按 `{slot,lane}` 取目的集合，希望一次读出 162-bit token bitmap；
- 若用 token-major SRAM，写自然但 lane-major 读需扫描 162 token；
- 若用 lane-major SRAM，读自然但写需要 32 路 bit-write 或 162-bit RMW；
- 若全部用寄存器，stage3 双 context 超过一百万 destination bit，面积和时钟功耗不可接受。

因此“总 bit 数不大”不能证明存储可映射。固定 bitmap 降级为对照方案，不能进入 DC 主线。

## 2. 新主线：容量安全 TERM-CSR/RAW head slot

新方案命名为：

> **GateStack-CSR：Capacity-Safe Final-Gate Term Compaction and Head-Stacked Replay**

每个 head 的物理 slot 固定为原始流容量：

```text
RAW_HEAD_BITS = 162 × (K32 + gate9) = 6642 bit
```

slot 内有两种格式：

```text
TERM-CSR:
  header
  term_desc[] = {gate_code, lane_id, event_base, event_count}
  token_id[]  = 每个term连续的destination token列表

RAW:
  raw_token[0..161] = {gate_code, K_bits[31:0]}
```

容量判定：

```text
CSR_bits = 192 + term_count × 35 + active_K_event_count × 8

if active_gate_classes > 4 or CSR_bits > 6642:
    mode = RAW
else:
    mode = TERM_CSR
```

RAW 与 CSR 都放在同样大小的 slot 内，任何输入都不会溢出或丢弃。RAW 不是第二套核，只是同一 product/multicast/accumulator 后端的另一种 issue 表示。

## 3. 两级稀疏 compaction 数据流

### 3.1 捕获相

```text
SCS final gate/K stream
  -> 41-bit token-major scratch
  -> active_token_mask[162]
  -> gate-class table[S=4]
  -> term occupancy/count[slot][lane]
```

两个 scratch ping-pong：head `h` 捕获时，head `h-1` 可执行 pack/commit。

### 3.2 Pack 相

```text
active_token iterator
  -> 只读K非零token
  -> R=2 lane extractor（R4仅作消融）
  -> OBI枚举有效{slot,lane}
  -> prefix/base生成
  -> packed term descriptor + token-id list
  -> RAW-sized head slot
```

OBI 的角色从“每个 output tile 扫目录”改为“每个 head compaction 时枚举一次有效 term”。packed descriptor 在所有 output tile 上顺序重放，不再重复 128-bit priority 搜索。

### 3.3 Replay 相

```text
for output_tile:
    for input_head:
        if TERM_CSR:
            顺序读term_desc
            计算一次gate×weight-vector
            顺序读该term的token-id list并多播
        else RAW:
            逐活动token/lane走direct issue
        跨head累加
    bias -> requant -> final
```

下一窗口在另一个 context 内捕获/pack，当前窗口执行 replay。context 只有在最后一个 output tile 完成握手后才能释放。

## 4. 真实 workload 证据

### 4.1 存储格式选择

来源：`results/gatestack_csr_storage_20260715.{json,md}`。

| 指标 | 数值 |
|---|---:|
| head rows | 672000 |
| TERM-CSR | **97.2155%** |
| RAW class overflow | 0.0141% |
| RAW capacity overflow | 2.7704% |
| 平均有效 payload | 981.6 bit |
| p99 payload | 6642 bit |
| 相对固定 RAW 平均有效位减少 | 85.2213% |
| 相对旧 bitmap 平均有效位减少 | 95.3034% |

2.77% capacity RAW 是设计选择，不是错误：它用少量 dense head 换取所有 slot 固定 6642 bit，从物理上保证 exact 和定长寻址。

### 4.2 Compactor 逐 token 分布

来源：`results/gatestack_compactor_profile_20260715.{json,md}`。直接解码原 profile 的 `[2,B,H,N]` K-count，无需 GPU 重跑。

| 指标 | mean | p99 | max |
|---|---:|---:|---:|
| 活动 token/head | 18.344 | 159 | 162 |
| 单 token 最大 K lane | 2.112 | 14 | 19 |
| R=2 event 提取周期 | 35.818 | 470 | 862 |
| R=4 event 提取周期 | 24.200 | 275 | 469 |
| R=8 event 提取周期 | 19.305 | 175 | 282 |

R=2/4 都不能跨 token 理想打包。活动 token 跳读的完整模型中，R2 到 R4 只把 speedup 从 1.382x 提到 1.386x；叶级 Yosys 结构却从 374 cell/223 mux 增至 726 cell/459 mux。因此首版锁 R2，R4/R8 只作吞吐上界消融。

## 5. 物理容量口径

双 context、固定 RAW-sized head slot、两个 scratch、一个 162×32×32-bit AccTile：

| Stage | 双context head slots | AccTile | 双scratch | metadata | 合计 | 相对旧bitmap |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 4.86 KiB | 20.25 KiB | 1.62 KiB | 0.14 KiB | **26.88 KiB** | -26.10% |
| 1 | 9.73 KiB | 20.25 KiB | 1.62 KiB | 0.28 KiB | **31.88 KiB** | -38.30% |
| 2 | 19.46 KiB | 20.25 KiB | 1.62 KiB | 0.56 KiB | **41.89 KiB** | -49.09% |
| 3 | 38.92 KiB | 20.25 KiB | 1.62 KiB | 1.12 KiB | **61.91 KiB** | -56.86% |

该表不含 weight/bias SRAM、ECC、macro padding 和 FIFO。平均 payload 不能进一步缩小固定物理 slot，除非引入全 context 可变长内存分配；首版不承担该控制风险。

## 6. 完整窗口周期

来源：`results/gatestack_csr_full_projection_model_20260715.{json,md}`。模型允许 direct 基线同样使用双 context，加入：

- 两个 scratch 的 head capture/commit 重叠；
- 每 head TERM-CSR/RAW 容量判定；
- 逐 token 精确 R 路提取周期；
- active-token mask 跳读；
- packed descriptor replay；
- 所有 input head、output tile、bias/requant 尾相；
- 双 window context 首填充和末排空。

| 配置 | prepare/direct | 全CSR双context | 分stage最快模式 |
|---|---:|---:|---:|
| R1，全162扫描 | 1.483x | 1.267x | 1.275x |
| R2，全162扫描 | 1.395x | 1.287x | 1.294x |
| R4，全162扫描 | 1.356x | 1.294x | 1.301x |
| R8，全162扫描 | 1.341x | 1.296x | 1.303x |
| **R2，活动token跳读** | **1.150x** | **1.382x** | **1.382x** |
| R4，活动token跳读 | 1.115x | 1.386x | 1.386x |

R2 活动 token 跳读的分 stage speedup：

| Stage | CSR比例 | RAW rows | 双context speedup |
|---|---:|---:|---:|
| 0 | 93.679% | 16687 | 1.262x |
| 1 | 99.931% | 99 | 1.096x |
| 2 | 99.584% | 898 | 1.422x |
| 3 | 97.858% | 1028 | 1.667x |

stage1 仍只有 9.8% cycle 收益。descriptor 保留 `DIRECT/CSR` 编译期模式，是否让 stage1 走 CSR 由目标库 EDP 决定；按当前纯周期模型，CSR 仍略快。

## 7. 存储端口合同

### 7.1 Scratch

```text
写：1 × 41 bit/cycle，token顺序地址
读：1 × 41 bit/cycle，active-token iterator地址
容量：2 × 162 × 41 bit
```

捕获与 pack 使用不同 scratch，不要求同一宏双向冲突访问。

### 7.2 Head slot SRAM

```text
地址：{context_id, input_head_id, word_offset}
格式：TERM_CSR或RAW
建议逻辑字宽：64 bit，物理macro由工艺重打包
容量：2 × Hmax × 6642 bit
```

模型对 RAW copy 仍按保守 162 cycle，不提前使用 64-bit word packing 收益。

### 7.3 Replay

TERM-CSR descriptor/token list 均顺序读。RAW token 也顺序读。两种模式共享 weight request、product、multicast、accumulator 和 bias/requant 接口，不引入异构双核。

## 8. 模块层次

```text
gatestack_csr_projection_top
  |- gatestack_window_context_manager
  |- gatestack_pingpong_raw_scratch
  |- gatestack_class_term_counter
  |- gatestack_active_token_iterator
  |- gatestack_obi_iterator                 已完成叶级RTL
  |- gatestack_event_compactor              主线WAYS=2，消融WAYS=4
  |- gatestack_capacity_mode_selector
  |- gatestack_head_slot_sram_adapter
  |- gatestack_head_tile_replay_scheduler
  |- gatestack_product_engine               复用现有
  |- gatestack_segmented_multicast          复用并扩展现有
  |- gatestack_persistent_accumulator
  `- gatestack_requant_emit
```

顶层只做连接；context 控制、compaction、replay、accumulator 分模块。首版单 `clk_core`，CDC 为 N/A，但 reset 和 macro BIST/scan 仍需综合阶段补合同。

## 9. 架构创新重新表述

### 贡献 A：容量安全的最终门码双格式 head slot

以 RAW 大小作为硬容量上界，运行时在 TERM-CSR 与 RAW 间精确选择。它不是一般压缩率优先，而是把 H67 final-gate term 复用与“任何输入不扩容、不丢数”的 ASIC slot 合同绑定。

### 贡献 B：主动 token 跳读的 term-major compaction

通过 active-token mask、R2 lane extraction、OBI term enumeration，把 SCS 的 token-major 41-bit 流转成可顺序 SRAM 读写的 term-major destination list，解决 bitmap 的转置端口矛盾。

### 贡献 C：跨 output tile 的 head-stacked replay

每个 input head 只捕获和 compaction 一次，packed term 在全部 output tile 上重放；两个 window context 隐藏 prepare/replay，相同后端支持 CSR 和 RAW。

这些贡献必须作为一套数据流叙述。单独的 CSR、ping-pong、priority iterator 或格式选择都已有大量先验，不能拆开包装成四个创新点。

## 10. 与文献的可辩护边界

- [Prosperity](https://arxiv.org/abs/2503.03379) 已提出 product sparsity 与在线复用；本文差异必须落在 final-gate term 的 exact 编码、容量安全 slot 和跨 tile replay，而不是泛称“乘积复用”。
- [FLAT](https://arxiv.org/abs/2107.06419) 与 [FuseMax](https://arxiv.org/abs/2406.10491) 已覆盖 attention fusion 和 pass/data-movement 分析；本文不能把顺序 SRAM 或流水本身列为创新。
- [Bishop](https://arxiv.org/abs/2505.12281) 使用 TTB 和密疏异构；本方案不复制双核，而是在同一后端前选择 CSR/RAW 表示。
- [FABNet](https://steliosven10.github.io/papers/%5B2022%5D_micro_adaptable_butterfly_accelerator_for_attention_based_nns_via_hardware_and_algorithm_codesign.pdf) 与[复旦 ISSCC 2023](https://fics.fudan.edu.cn/70/b1/c22203a487601/page.htm)依赖 butterfly sparsity/zero skipper；当前 H67 无结构化权重证据，蝶形仍不进入主线。
- [ESSERC 2025 LLM-Friendly CIM](https://www.esserc2025.org/_files/ugd/aa54ce_46d0ec87f2084ab7b10f3bb6a22c0840.pdf) 已有 multi-ping-pong 和 head dataflow reschedule；双 context 只能作为 GateStack-CSR 的使能机制，不单列新颖性。

## 11. RTL/DC 淘汰门槛

| 项 | 最低门槛 |
|---|---:|
| Python/RTL完整投影 bit-exact | 100% |
| CSR/RAW forced-mode coverage | 每 stage 各至少1000组 |
| 容量边界 | `CSR_bits=6642/6643` 双边定向测试 |
| R2 event compactor | 逐token trace无丢失、无重复 |
| 完整 cycle speedup | 相对公平双context direct ≥1.20x |
| EDP | 同库同频同SRAM端口改善 ≥15% |
| stage3非weight存储 | macro padding后 ≤80 KiB |
| WNS/TNS | 500 MHz下均≥0 |
| LEC | 100% equivalent |

模型 `1.382x` 只有约 15.1% 相对 1.20x 门槛的速度余量。若 packed-slot bank conflict、priority 路径或随机 scratch 读使完整 RTL 低于门槛，必须降级为算子后端，不得继续以架构主贡献宣传。

## 12. 实施顺序

1. 写 active-token iterator 与参数化 R2/R4 compactor 的 standalone reference/RTL。**已完成。**
2. 写 capacity selector，覆盖 6642/6643 bit 和 class 4/5 边界。**已完成。**
3. 写固定 RAW-sized head slot adapter，先用行为 SRAM，接口对齐 64-bit macro。
4. 集成单 head CSR/RAW replay，复用现有 product/multicast。
5. 集成完整 H/head/output-tile persistent accumulator。
6. 加双 context，并用 ordered trace 驱动 cycle/SAIF replay。
7. 有目标库后跑 direct、bitmap-fixed、bitmap-OBI、CSR-fullscan、CSR-active 五档 DC/SAIF 消融。

叶级 RTL 结果详见 `docs/85_GateStack活动Token与R2R4_Compactor_RTL验证_20260715.md`。
capacity selector 已完成，详见 `docs/86_GateStack容量安全模式选择器RTL验证_20260715.md`。

## 13. 当前结论

GateStack-CSR 已从“逻辑上正确的 bitmap 目录”迭代为“存储端口可解释、容量最坏情况有界、周期收益仍过线”的完整候选。它目前仍是 `[prof]+[模型]+[OBI/active-token/R2-R4叶RTL]`，不是可投稿 PPA 结论；下一关键风险是 capacity selector 与 packed head slot，而不是继续扩 product engine 或加入蝶形网络。
