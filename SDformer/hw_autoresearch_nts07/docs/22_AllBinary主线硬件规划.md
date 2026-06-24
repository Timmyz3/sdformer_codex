# All-Binary NTS/H60 主线硬件规划

**版本**：2026-06-19  
**主线**：All-Binary ATLIF + all12 NTS/H60  
**候选命名**：DATE11-Bin / NTS11-Bin / UniBin-H60  
**目标**：把当前 all-binary valid825 结果转化为 DATE 论文可讲、可画、可估算、可验证的硬件架构方案。

---

## 1. 主线切换判断

当前应把硬件主线从 mixed NTS11 切到 all-binary NTS/H60。

| 方案 | AEE | AAE | total_spikes | energy_uj | 硬件复杂度 |
|---|---:|---:|---:|---:|---|
| NB0 ep59 | 1.4872 | 9.9300 | 44.0488G | 37638.01 | 原始复杂数据流 |
| mixed NTS11bj ep2 | 1.5159 | 9.9611 | 29.0414G | 23032.66 | binary/ternary 混合 |
| all-binary NTS/H60 ft ep2 | 1.4891 | 9.7785 | 23.8206G | 21045.91 | 全 1-bit event |

all-binary 现在同时满足：

1. 精度几乎等于 NB0，且优于 mixed NTS11bj。
2. spikes/energy 明显低于 NB0 和 mixed NTS11bj。
3. 结构上消除了 ternary rail、2-bit event SRAM、mixed-format 控制。

因此论文主张应改成：

> DATE11-Bin 将 SDformerFlow encoder 转换为统一 binary-event H60 attention 数据流，在保持 baseline 精度的同时显著降低 spike/energy，并把全网事件存储和调度统一为 1-bit packed format。

---

## 2. 顶层架构

新版顶层不再以 ternary Q/K 为核心，而以 1-bit binary event 为唯一事件格式。

```text
sensor/event voxel
    ↓
patch / conv frontend
    ↓
Binary ATLIF Event Encoder
    ↓
Encoder Stage S0/S1/S2/S3
    ├─ shared Binary H60 Attention Engine
    │    ├─ Binary Q/K Event Tile Buffer
    │    ├─ Popcount Consensus Score Engine
    │    ├─ Single Shiftmax Token Gate
    │    └─ Gated-K Binary Event Output
    ├─ Binary MLP / FFN Event Path
    └─ Binary Downsample Event Path
    ↓
1-bit Packed Skip Buffer
    ↓
Decoder Replay / Dense Projection
    ↓
flow prediction head
```

硬件模块边界：

| 模块 | 作用 | 是否共享 |
|---|---|---|
| `binary_atlif_cluster` | 105 个逻辑 ATLIF site 复用的二值事件生成单元 | 是 |
| `binary_h60_engine` | 12 个 encoder attention block 复用的 H60 score-gate-output 单元 | 是 |
| `popcount_consensus_engine` | Q/K overlap、active count、mismatch count 统计 | 是 |
| `shiftmax_gate_unit` | 对 token score 做统一 gate | 是 |
| `gated_k_unit` | gate 调制 K event / K feature | 是 |
| `packed_event_sram` | 1-bit event tile、skip、intermediate buffer | 是 |
| `ttb_scheduler` | TTB1/TTB2 bundle skip 与 work-item 发射 | 是 |
| `layer_descriptor_controller` | 按 stage/block/module 配置共享单元 | 是 |

---

## 3. 数据格式

### 3.1 事件格式

all-binary 主线只有一种外部事件格式：

| 名称 | 位宽 | 语义 |
|---|---:|---|
| `event_bit` | 1 | `0=inactive`, `1=active` |

不再需要：

- `-1/0/+1` ternary code；
- pos/neg 双 rail；
- sign+magnitude event；
- per-layer binary/ternary mode switch；
- ternary pack/unpack。

### 3.2 内部定点格式

事件是 1-bit，但内部状态仍需要定点：

| 数据 | 初始建议 | 说明 |
|---|---|---|
| ATLIF membrane | INT12 或 INT16 | 后续 profiling 决定位宽 |
| threshold | INT12 或 INT16 | 当前 all-binary official ATLIF threshold 固定 1.0 起步 |
| TX/SC score | INT12/INT16 | 由 popcount 派生 |
| Shiftmax gate | INT8/INT10/INT12 sweep | 需要 valid825 验证 |
| gated-K output | INT8/INT12/FP16 对照 | 第一版可用 fixed-point golden |

### 3.3 SRAM / NoC format

| buffer | mixed NTS11 | all-binary 主线 |
|---|---|---|
| Q/K tile | 2-bit ternary | 1-bit binary |
| activation event | 1/2-bit mixed | 1-bit binary |
| skip buffer | 1/2-bit mixed | 1-bit packed |
| NoC payload | mode-dependent | fixed 1-bit event packet |
| descriptor | 需要 `neuron_mode` | 可删除或保留为 debug 常量 |

---

## 4. H60 Attention 数据流

### 4.1 软件语义

当前 all-binary 配置仍使用：

```yaml
bsa_attention:
  enabled: true
  mode: h60
  target_blocks: all 12 encoder blocks
```

Q/K 由 binary ATLIF 输出：

```text
Q, K ∈ {0, 1}
```

H60 score 可解释成：

```text
overlap  = popcount(Q & K)
active_q = popcount(Q)
active_k = popcount(K)
mismatch = active_q + active_k - 2 * overlap
score    = TX(overlap, active_q, active_k)
         + μ * SC(overlap, mismatch)
```

这里不要求硬件逐字复刻 PyTorch float 运算，第一版目标是建立可校准的 integer/fixed-point equivalent。

### 4.2 硬件流水

```text
Step 1: load binary Q/K tile
Step 2: bit-parallel AND
Step 3: popcount overlap / active_q / active_k
Step 4: integer TX/SC score fusion
Step 5: optional score centering
Step 6: Shiftmax token gate
Step 7: K × gate 或 event-gated K output
Step 8: write back binary/fixed-point attention output
```

### 4.3 Shiftmax 的定位

P0 profiling 显示 Shiftmax gate 接近均匀，不能作为强 token pruning 卖点。论文中应把它定位为：

```text
single reusable token gate / normalization unit
```

节能主来源应写成：

- binary event sparsity；
- 1-bit packed SRAM/NoC；
- popcount consensus score；
- TTB1/TTB2 bundle skip；
- all-encoder H60 engine reuse。

---

## 5. TTB 调度规划

TTB 是 Token-Time Bundle，是硬件调度粒度，不是模型模块。

基于 mixed NTS11 P0 profiling：

| 粒度 | empty ratio | 结论 |
|---|---:|---|
| TTB1 | 约 39%-52% | 跳过潜力最大，控制更细 |
| TTB2 | 约 12%-37% | 面积/控制折中，推荐第一版 |
| TTB4 | 基本 0 | 粒度过粗，不推荐主线 |

all-binary 主线需要重新跑 P0 profiling，但第一版调度策略可以先定为：

```text
默认硬件规划：TTB2
论文敏感性分析：TTB1 / TTB2 / no-TTB
```

TTB scheduler 输入：

| 字段 | 说明 |
|---|---|
| `stage_id` | 0..3 |
| `block_id` | 0..5 |
| `head_id` | 0..23 |
| `window_id` | 当前 window |
| `token_bundle_id` | TTB group |
| `q_nonzero` | bundle 中 Q 是否有事件 |
| `k_nonzero` | bundle 中 K 是否有事件 |

跳过规则第一版：

```text
if q_nonzero == 0 and k_nonzero == 0:
    skip score engine and gated-K
else:
    issue binary_h60_engine work item
```

---

## 6. Skip / Decoder Replay 规划

skip 连接要按真实代码路径讲，不再模糊画线。

| skip 类型 | 来源 | 用途 | buffer 规划 |
|---|---|---|---|
| `stage_skip_predownsample` | S0/S1/S2 downsample 前 | decoder i=3/2/1 使用 | 1-bit packed skip SRAM |
| `stage_skip_final` | S3 final-stage output | decoder i=0 使用 | 1-bit packed retained buffer |

all-binary 后，每样本 skip buffer 口径应重新 profiling。第一版估算可以从 mixed NTS11 的 packed 口径减半推导，但论文最终必须用 all-binary profile 实测。

硬件图建议画：

```text
S0 pre-downsample ─┐
S1 pre-downsample ─┼─> Packed Skip SRAM ─> Decoder Replay
S2 pre-downsample ─┘
S3 final output ─────> Retained Bottleneck Buffer ─> Decoder i=0
```

---

## 7. 面积 / 功耗 / 吞吐模型

### 7.1 面积模型

第一版面积表按模块列：

| 模块 | 主要面积项 |
|---|---|
| binary ATLIF cluster | membrane register、threshold compare、reset logic |
| popcount consensus engine | AND array、popcount tree、small add/sub |
| Shiftmax gate | max/shift/add/reciprocal or LUT |
| gated-K unit | multiplier/shift or gated mux |
| packed event SRAM | 1-bit SRAM macro + address generator |
| TTB scheduler | nonzero detector、descriptor queue |
| controller | layer descriptor FSM |

需要强调的面积收益：

```text
remove ternary decoder
remove sign rail
remove pos/neg dual popcount
remove mixed event SRAM format
remove per-layer event-mode switching
```

### 7.2 功耗模型

功耗分解：

```text
E_total =
  E_ATLIF_binary
  + E_H60_popcount
  + E_Shiftmax
  + E_GatedK
  + E_SRAM_1bit
  + E_NoC_1bit
  + E_Control
```

每项统计来源：

| 项 | 统计来源 |
|---|---|
| `E_ATLIF_binary` | all-binary ATLIF activity / firing |
| `E_H60_popcount` | Q/K active density、overlap count |
| `E_Shiftmax` | windows × heads × tokens |
| `E_GatedK` | K active density × gate ops |
| `E_SRAM_1bit` | activation/skip bytes |
| `E_NoC_1bit` | tile traffic |
| `E_Control` | descriptor 数量、TTB issued/skipped |

### 7.3 吞吐模型

吞吐先按 frame pipeline 估计：

```text
T_frame =
  T_frontend
  + T_encoder_h60
  + T_downsample
  + T_decoder_replay
  + T_prediction
```

H60 主项：

```text
T_encoder_h60 =
  sum(stage, block, head, window, TTB)
    cycles(binary popcount + score + Shiftmax + gated-K)
```

TTB skip 后：

```text
effective_work_items =
  total_work_items * (1 - empty_bundle_ratio)
```

---

## 8. 架构图规划

### 图 1：总体架构

内容：

```text
Input Event Frames
→ Binary ATLIF Encoder Cluster
→ Encoder S0/S1/S2/S3
→ Shared Binary H60 Attention Engine
→ Packed Event SRAM / Skip Buffer
→ Decoder Replay
→ Flow Head
```

图中突出：

- 105 logical ATLIF sites 不是 105 套硬件；
- 12 H60 blocks 共享一个或少数几个 H60 engine；
- 全网 event format 是 1-bit。

### 图 2：Binary H60 Engine

内容：

```text
Q bit tile ─┐
            ├─ AND / Popcount ─ TX/SC Score ─ Shiftmax ─ Gated-K ─ output
K bit tile ─┘
```

标注：

- `overlap = popcount(Q & K)`
- `active_q = popcount(Q)`
- `active_k = popcount(K)`
- `score = TX + μSC`
- `gate = Shiftmax(score)`

### 图 3：TTB Scheduler

内容：

```text
Token-Time Bundle Queue
→ nonzero detector
→ skip / issue
→ H60 work queue
```

对比：

- no-TTB；
- TTB1；
- TTB2。

### 图 4：Skip Buffer / Decoder Replay

内容：

```text
S0/S1/S2 pre-downsample skip
S3 final retained output
→ 1-bit packed skip SRAM
→ decoder replay
```

---

## 9. 需要立刻补的统计

P0 状态更新：

- all-binary ep2 valid40 P0 profiling 已完成，见 `23_AllBinary_P0_profiling实测结果.md`。
- H60 调用 `480 = 40 × 12`，全 encoder H60 path 覆盖成立。
- binary ATLIF activity 约 `4.45%`。
- TTB2 empty ratio：S0 `27.9%`，S1 `73.8%`，S2 `63.0%`，S3 `64.5%`。
- all-binary 1-bit packed skip buffer 每样本约：S0/S1/S2 pre-downsample `1.45 MB`，S3 final retained `0.10 MB`。

P0，仍需补：

1. **all-binary P0 profiling**
   - checkpoint：`date11full_all_binary_atlif_nts_stdlr_ft_ep29_ft5` epoch 2。
   - samples：valid40 已完成；仍需 full valid825。
   - 指标：H60 gate、Q/K activity、TTB1/TTB2、ATLIF binary activity、activation/skip bytes。

2. **all-binary layer category spikes**
   - 确认 downsample 是否仍是热点；
   - 对比 mixed NTS11 的 downsample 4.1G-4.4G。

3. **module coverage**
   - 确认 105 个 binary ATLIF 是否全 forward 覆盖；
   - 确认 12 个 H60 block 仍全覆盖；
   - 确认没有原生 attention path。

4. **quantized H60 valid825**
   - 基于 all-binary ep2；
   - 跑 H60 score/gate 定点近似；
   - 先不动权重量化。

P1，建议补：

5. Shiftmax INT8/INT10/INT12 sweep。
6. ATLIF membrane/threshold 定点位宽 sweep。
7. 1-bit skip buffer 容量估算表。
8. H60 per-block score/gate histogram。

---

## 10. 接口文档需要改哪些

现有 `05_module_interface_spec.md` 还是 mixed/NTS07 风格，下一版要改：

| 旧字段/模块 | 新规划 |
|---|---|
| `nts07_top` | 改为 `date11_bin_top` 或 `unibin_h60_top` |
| `ternary_encode_unit` | 删除，或放到 mixed fallback appendix |
| `q_ternary/k_ternary` | 改为 `q_bits/k_bits` |
| `neuron_mode` | all-binary 主线中固定为 binary，可删除或只读 debug |
| `MAX_TOKENS=98` | 需要按当前 window token 统计修正；P0 中 H60 token 数约 162 |
| `token_mask=98` | 改为参数化 `MAX_TOKENS` |
| `legacy_attn_engine` | 删除主线图，只放历史说明 |
| `binary/ternary packer` | 改为 `1-bit event packer` |

新增接口：

```systemverilog
interface binary_event_stream #(
    parameter int LANES = 128
);
    logic valid;
    logic ready;
    logic last;
    logic [LANES-1:0] event_bits;
    logic [1:0] stage_id;
    logic [2:0] block_id;
    logic [4:0] head_id;
    logic [9:0] window_id;
    logic [7:0] token_base;
endinterface
```

Binary H60 engine 第一版接口：

```systemverilog
module binary_h60_engine #(
    parameter int HEAD_DIM = 32,
    parameter int MAX_TOKENS = 192,
    parameter int LANES = 128
)(
    input  logic clk,
    input  logic rst_n,
    input  logic cfg_center_scores,
    input  logic cfg_preserve_mean,
    input  logic [7:0] cfg_mu_q8,
    input  logic [7:0] cfg_n_tokens,
    input  logic q_valid,
    output logic q_ready,
    input  logic [LANES-1:0] q_bits,
    input  logic k_valid,
    output logic k_ready,
    input  logic [LANES-1:0] k_bits,
    output logic out_valid,
    input  logic out_ready,
    output logic [LANES-1:0] out_event_bits
);
```

这个接口是规划草案，正式 RTL 前要用 profiling 的实际 tile shape 重新核对。

---

## 11. 量化路线

量化要做，但不是主线切换的前提。

推荐顺序：

1. H60 score integer equivalent。
2. Shiftmax gate INT8/INT10/INT12。
3. gated-K output 定点。
4. ATLIF membrane/threshold 定点。
5. Conv/Linear 权重量化。

先不建议直接做全模型权重量化，因为它会把硬件数据流问题和训练稳定性问题混在一起。

量化验收：

| 阶段 | 验收标准 |
|---|---|
| H60 score/gate 定点 | AEE 相对 all-binary float ep2 增加 < 0.02 |
| Shiftmax INT sweep | 至少一个位宽 AEE < 1.52 |
| ATLIF 定点 | firing/energy 不反弹超过 5% |
| 全部署近似 | AEE 保持在 NB0 +5% 窗口内 |

---

## 12. 工作计划

### Week 1：证据补齐

- [x] 跑 all-binary ep2 valid40 P0 profiling。
- [ ] 跑 all-binary ep2 full valid825 profiling。
- [x] 生成 all-binary H60/TTB/ATLIF/skip 中文报告。
- [ ] 更新主线对比表，加入 all-binary。

### Week 2：架构定稿

- [ ] 重写模块接口文档为 `date11_bin` 主线。
- [ ] 画 4 张论文图：总体架构、Binary H60、TTB scheduler、skip replay。
- [ ] 写面积/功耗/吞吐估算口径。
- [ ] 明确哪些模块共享、哪些按 lane 并行复制。

### Week 3：量化验证

- [ ] 跑 all-binary H60 score/gate 定点 valid825。
- [ ] 跑 Shiftmax INT8/INT10/INT12 sweep。
- [ ] 统计 AEE/AAE/spikes/energy 变化。
- [ ] 决定 DATE 表格引用 float event 还是 quantized event。

### Week 4：论文故事整合

- [ ] 更新 related work：binary attention engine、event SRAM、TTB 调度。
- [ ] 写 contribution bullets。
- [ ] 写方法章节硬件部分。
- [ ] 写 ablation：NB0、mixed NTS11、all-binary、all-binary quant。

---

## 13. 当前决策

立即采用以下规划：

```text
主线：All-Binary ATLIF + all12 NTS/H60
架构名：UniBin-H60
事件格式：全 1-bit binary event
attention engine：binary popcount consensus + Shiftmax + gated-K
memory：1-bit packed event SRAM + packed skip replay
调度：TTB2 第一版，TTB1/TTB2 做敏感性分析
量化：H60 score/gate 优先，权重量化后置
```

mixed NTS11 不删除，它在论文中变成：

1. 证明 all-encoder H60 统一 attention 有效的前置版本；
2. 说明 ternary Q/K 并非硬件必需的消融；
3. 如果 all-binary quant 后续失败，可作为 fallback。
