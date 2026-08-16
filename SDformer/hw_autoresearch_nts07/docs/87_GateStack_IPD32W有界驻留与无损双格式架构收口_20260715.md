# GateStack-IPD32W 有界驻留与无损双格式架构收口（2026-07-15）

> **后续 RTL 进展**：token-only 子区间回放、resident joiner、RAW 适配、三路径 mux 和 TDR 解耦后端叶模块已完成。最新接口、验证和剩余顶层缺口见 `docs/88_GateStack_TDR解耦回放架构与RTL进展_20260715.md`。

## 1. 本轮结论

当前 H67 硬件主线冻结为：

> **GateStack-IPD32W：容量安全双格式 Head Stack + Depth=80 双 Context Descriptor Residency + 无损 RAW41 Fallback**

它不是“把稀疏 head 丢掉”，而是对同一个 final-gate 投影语义选择两种无损表示：

```text
SCS final gate/K
  -> token-major RAW scratch
  -> active-token iterator + R2 event compactor + OBI term enumeration
  -> capacity selector
       |- IPD32W head slot + descriptor residency cache
       `- RAW41 head slot
  -> replay router
       |- resident descriptor + packed token-ID replay
       |- sequential IPD32W replay（cache overflow）
       `- RAW41 direct replay（class/capacity overflow）
  -> shared product engine -> segmented multicast -> persistent accumulator
```

证据仍未达到 DC/PPA 签核。当前等级是：

- workload 和命中率：`[prof]`；
- 端到端周期：`[prof]+[模型]`；
- 格式、存储、decoder、fallback、cache：`[rtl叶级]`；
- 完整多 head/output-tile 顶层与 ASIC PPA：**未完成**。

## 2. 为什么旧 35-bit CSR 被替换

旧 descriptor 为：

```text
gate9 + lane5 + event_base13 + event_count8 = 35 bit
```

但 token ID 列表严格按 descriptor 顺序连续保存。replay 维护滚动 token 指针即可得到下一 term 的起点，13-bit `event_base` 是冗余状态。

格式 DSE 结果来自 `results/gatestack_descriptor_format_dse_20260715.*`：

| 格式 | CSR比例 | 平均有效位 | 主要代价 |
|---|---:|---:|---|
| 旧 packed35/header192 | 97.2155% | 981.6 bit | 跨 word 提取、显式 event_base |
| IPD24/header128 | 98.0211% | 819.0 bit | 24-bit descriptor 跨 64-bit word |
| **IPD32W/header128** | **97.5015%** | **900.7 bit** | 奇数 term 补 32 bit |
| 理论 IPD22/header128 | 98.1219% | 800.3 bit | 最复杂的 bit-packed 提取 |

首版选择 IPD32W，而不是压缩率最高的 IPD24：每个 64-bit word 固定容纳两个 descriptor，前端可稳定给出至少一个 term/cycle，不需要 24-bit 跨 word barrel。IPD24 只保留为 DC 面积消融。

## 3. 精确格式合同

### 3.1 物理 Head Slot

```text
RAW_HEAD_BITS = 162 × (K32 + gate9) = 6642 bit
WORDS_PER_HEAD = ceil(6642/64) = 104
最后一个RAW word有效位 = 50
```

每个 context/head 都占固定 104×64-bit 逻辑槽。平均 payload 变小不能缩小物理槽；固定容量是无损 fallback 的硬合同。

### 3.2 IPD32W

```text
CSR_bits = 128 + ceil(term_count/2)×64 + active_K_event_count×8

descriptor[31:0]:
  [8:0]   gate_code
  [13:9]  lane_id
  [21:14] destination_count
  [31:22] reserved=0
```

128-bit header 保存 magic/version/tag、payload bits、term/event/class/active-token 计数和 token-list byte offset。token ID 列表按 term 顺序连续排列，descriptor 不再保存 event base。

容量选择边界为：

| classes | terms | events | bits | 结果 |
|---:|---:|---:|---:|---|
| 4 | 6 | 790 | 6640 | IPD32W |
| 4 | 6 | 791 | 6648 | RAW capacity |
| 5 | 任意 | 任意 | 任意 | RAW class |

因为 IPD32W 字节对齐，`6642/6643` 不再是可构造的相邻编码点；有效边界是 `6640/6648`。

### 3.3 RAW41

```text
record[token] = {gate_code[8:0], K_bits[31:0]}
```

RAW41 decoder 用 128-bit bit reservoir 从 64-bit SRAM word 重组 41-bit record。K-zero record 被扫描但不发 product；活动 K 按 token、lane 升序发 direct issue。它与 IPD32W 共用后端，不复制第二套乘法/多播/累加核。

## 4. Profile100 决策证据

来源：H67 ep19 的 100-sample ordered trace；来源和哈希审计见 `results/profile100_provenance_audit_20260715.*`。

### 4.1 Workload

| 指标 | 数值 |
|---|---:|
| token K-zero | 约 88.7% |
| pair empty | 约 74.0% |
| pair motion-zero | 约 83.2% |
| final-gate product work 减少 | 82.49% |
| 活动 token/head mean / p99 | 18.344 / 159 |
| R2 compactor cycle mean / p99 / max | 35.818 / 470 / 862 |

这些数字支持 active-token 跳读和 term compaction，但不支持“空 token 可从 Shiftmax 分母删除”。本架构只跳过已由软件/定点语义证明无 product 的 K-zero 传播。

### 4.2 IPD32W/RAW 比例

| 模式 | 比例 |
|---|---:|
| IPD32W | **97.5015%** |
| RAW class overflow | 0.0141% |
| RAW capacity overflow | 2.4844% |
| 平均有效 payload | 900.7 bit |
| 相对固定 RAW 平均有效位减少 | 86.4389% |

### 4.3 有界 Descriptor Residency

descriptor cache 内部用 24-bit `{reserved2,count8,lane5,gate9}`，head slot 仍用 IPD32W。cache 在 compaction 的 term 枚举拍旁路写入，不增加独立写相。

| 深度/head | CSR内命中 | 加权前端周期减少 | Stage3双context cache | Stage3非weight合计 |
|---:|---:|---:|---:|---:|
| 32 | 87.8274% | 54.7722% | 4.55 KiB | 66.46 KiB |
| 64 | 99.3900% | 95.4356% | 9.05 KiB | 70.96 KiB |
| **80** | **99.9826%** | **99.7473%** | **11.30 KiB** | **73.21 KiB** |
| 96 | 99.9989% | 99.9793% | 13.55 KiB | 75.46 KiB |
| 128 | 100% | 100% | 18.05 KiB | 79.96 KiB |

Depth=80 是默认点；Depth=64 是 macro padding 后面积超标时的回退点。Depth=128 没有 ECC/BIST/rounding 余量，不能作为首版。

## 5. 修正后的周期结果

周期模型先经历两次纠偏：

1. 不计 header/descriptor 前端时，Depth=0 曾得到约 1.401x，口径过于乐观；
2. 逐 head 加入 `2+ceil(term/2)` 顺序读后，Depth=0 只有 **1.257x**；
3. 加入真实可实现的 Depth=80 residency 后，恢复为 **1.400x**。

默认 `R=2 + active-token + Depth=80 + delivery efficiency=85%`：

| Stage | IPD32W比例 | RAW rows | 双context speedup |
|---|---:|---:|---:|
| 0 | 94.288% | 15081 | 1.285x |
| 1 | 99.938% | 90 | 1.097x |
| 2 | 99.674% | 705 | 1.437x |
| 3 | 98.096% | 914 | 1.692x |
| 全部 | 97.5015% | 16790 | **1.400x** |

该结果仍是 `[prof]+[模型]`。模型已计入公平双 context、active-token 精确计数、R2 逐 token compaction、RAW 比例和未命中 descriptor 前端；尚未计目标 SRAM latency、完整 top backpressure、布线和时钟树。

## 6. 当前 RTL 模块

| 模块 | 已验证功能 |
|---|---|
| `gatestack_active_token_iterator` | 162-bit mask 分段扫描 |
| `gatestack_event_compactor` | R2 主线、R4 消融；逐 token lane 无丢失/重复 |
| `gatestack_obi_iterator` | 128-bit term occupancy 枚举 |
| `gatestack_capacity_mode_selector` | IPD32W/RAW class/RAW capacity 三分流 |
| `gatestack_head_slot_sram_adapter` | 2 context、24 head、104×64/head、1W1R、显式 release |
| `gatestack_ipd32w_replay_decoder` | header/descriptor 校验、4 token/cycle reservoir、下一 term product 预取 |
| `gatestack_raw41_replay_decoder` | 162×41 exact 解包、K-zero skip、direct issue |
| `gatestack_descriptor_residency_cache` | Depth=80、双 context、overflow miss、1W1R |

严格回归结果：

| 项 | 结果 |
|---|---:|
| Python unittest | **77/77 PASS** |
| GateStack 叶级 Icarus 自检 | 全部 PASS |
| GateStack Verilator assertion | 全部 PASS |
| GateStack Verilator warning/error | **0** |
| GateStack Yosys check | 全部 0 problem |
| 原 G1 `T=162/L=32` | PASS |

Yosys `memory -nomap` 只用于结构审计：capacity selector 36 cell；head-slot adapter 228 cell + 5 个 `$mem_v2`；IPD32W decoder 447 cell + 1 个 `$mem_v2`；RAW41 decoder 293 cell；descriptor cache 234 cell + 3 个 `$mem_v2`。这些不是工艺面积。head-slot 地址生成已消除通用 `$mul`。

## 7. 架构创新点

### 7.1 容量安全的最终门码双格式 Head Stack

以 RAW41 作为任何输入的固定容量上界，运行时选择 IPD32W 或 RAW，不依赖近似 prune，也不让压缩溢出破坏语义。创新不在“用了 CSR”，而在 final-gate term 复用、固定 ASIC slot 和 exact fallback 的联合合同。

### 7.2 隐式前缀 IPD32W 与 Term/Destination 解耦

利用 head-stacked 顺序 replay 删除 event base；descriptor 两项/64-bit word，token ID 进入独立 reservoir；下一 term product command 可与当前 term destination multicast 重叠。它是针对 H67 gated-K 传播顺序构造的数据流，不是一般稀疏矩阵 CSR 的直接照搬。

### 7.3 Profile-Locked Bounded Descriptor Residency

Depth=80 cache 不是无界全复制，也不是命中失败丢计算。命中 head 跨所有 output tile 驻留 descriptor，超深 head 无损回顺序 IPD32W；深度由 672000 个真实 head row 的命中率、前端周期和双 context 存储共同锁定。

### 7.4 统一后端的表示异构，而非双计算核

IPD32W、cache miss 和 RAW41 只在 issue 表示上不同，共用 product、multicast、persistent accumulator 和 requant。与 dense/sparse 双核相比，它减少复制后端和负载均衡控制；是否优于双核必须由同库 PPA 对照证明。

## 8. 与已有工作的边界

- Prosperity 已有在线 product reuse；本文不能泛称“首次复用乘积”，差异是 final-gate term、容量安全双格式和跨 tile descriptor residency。
- FLAT/FuseMax 已有 attention fusion 与数据移动分析；本文不能把顺序 SRAM 或融合流水单列为创新。
- Bishop 已有 TTB 与密疏异构；本文当前不是双核，而是统一后端前的表示分流。
- SpAtten/Energon 有动态稀疏或裁剪；本文 fallback 全部 exact，不删除弱 attention pair。
- FABNet 和复旦 ISSCC 2023 蝶形工作依赖结构化权重；H67 尚无蝶形权重/精度/EDP 证据，因此不进入主线。

## 9. DATE 评估口径

### 9.1 面积

必须分开报告：逻辑、head-slot SRAM、descriptor cache、scratch、AccTile、weight/bias SRAM、FIFO/控制、clock tree。SRAM 报 macro 实例和实际 rounded bits，不能只报逻辑 bit。

### 9.2 功耗

使用同一 ordered trace 生成 SAIF，分为：scratch/slot/cache 读写、decoder、product、multicast、accumulator、clock/control。必须同时给泄漏、动态和每窗口能量；Yosys cell 数不能替代功耗。

### 9.3 吞吐

报告稳态 FPS、首窗口 latency、平均/p95/p99 cycles，并分账 prepare、descriptor hit/miss、IPD token replay、RAW replay、product、multicast、bias/requant、context stall。

### 9.4 存储

同时报告逻辑容量、macro padding 后容量、带宽、端口、bank conflict、利用率。Depth=80 的 73.21 KiB 只是非权重逻辑容量模型。

### 9.5 控制复杂度

报告 FSM 状态数、descriptor FIFO/cache 深度、最大 outstanding、反压路径、错误恢复、context release 条件和形式/断言覆盖。不能用“模块数少”替代控制复杂度。

## 10. DC 准入与淘汰门槛

| 项 | 门槛 |
|---|---:|
| 完整多 head/tile bit-exact | 100% |
| IPD32W/RAW/cache hit/miss 覆盖 | 每 stage 各至少 1000 head |
| 完整 ordered-trace speedup | 相对公平 direct ≥1.20x |
| EDP | 同库同频同 SRAM macro 改善 ≥15% |
| Stage3 非权重存储 | macro padding 后 ≤80 KiB |
| 目标频率 | 500 MHz 下 WNS/TNS 均 ≥0 |
| LEC | 100% equivalent |

当前不能直接做可信 DC 主表，原因不是叶 RTL 语法，而是：

1. 完整 multi-head/output-tile replay router 尚未集成；
2. cache hit、cache miss IPD 和 RAW 三路尚未接到同一 product/multicast/accumulator；
3. 双 window context manager 和最后一 tile release 尚未集成；
4. bias/requant 部署位宽合同尚未冻结；
5. 当前环境没有目标 `.lib/.db`、SRAM macro model、`dc_shell`/Genus 和 SAIF 主流程。

## 11. 下一步顺序

1. 写 `gatestack_replay_router`，统一 resident/IPD/RAW issue 接口并验证严格顺序。
2. 把 descriptor cache fill 与现有 OBI term 枚举同拍连接，证明不增加 prepare 周期。
3. 集成单 head、单 output tile 的三模式 bit-exact top。
4. 扩展到 3/6/12/24 input heads、全部 output tiles 和 persistent accumulator。
5. 加双 context manager、ordered trace cycle/SAIF 驱动和 p95/p99 分账。
6. 取得目标库后做 direct、无 cache、Depth64、Depth80、IPD24、IPD32W 的同约束 DC/SAIF 消融。

## 12. 当前判定

GateStack 已从“单行核优化”推进为具有存储层次、双格式稀疏执行、跨 tile 驻留和 exact fallback 的架构候选。Depth=80 residency 是本轮真正增加的系统架构点，并由真实 workload 锁定；但完整 top、DC/SAIF 和端到端 PPA 尚未完成，因此当前判定仍是：

> **架构候选成立，叶级 RTL 证据充分；尚未达到可直接发 DATE 或直接交 DC 主表的标准。**
