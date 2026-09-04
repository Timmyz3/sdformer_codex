# TTX 完整 RTL 微架构与全套设计

> **2026-07-11 适用边界**：本 RTL 精确映射历史 dyadic TTX 的 `{0,+theta}` one-sided ATLIF 与 FGK K-carrier 路径。H63 direct Shiftmax、H64 centered symmetric、H65 signed Hamming 均未通过 DSEC 20-step/activity 门槛，未授权替换本 RTL。软件负结果和候选接口见 `40_H63对称ATLIF无GateK注意力探索.md`。

**日期**：2026-07-10  
**主线**：AllBinary TTX  
**实现范围**：从 packed Q/K event row 到 factorized gated-K 输出；不实现 voxel 构建  
**代码目录**：`rtl_ttx/`、`tb_ttx/`、`sim_ttx/`

---

## 1. 设计边界

本设计不实现 event camera 到 voxel 的构建。硬件输入边界定义为：

```text
上游 patch/QK projection/ATLIF
→ q_bits[31:0], k_bits[31:0]
→ TTX attention accelerator
→ token_idx + k_bits[31:0] + gate_q8 + threshold_q8
→ 下游 sparse projection / residual / MLP
```

这样做的原因：

1. TTX 的硬件创新集中在 all-binary attention，不在数据集专用 voxel 预处理；
2. DSEC/MVSEC 的 voxel 生成可以由 CPU、ISP、传感器前端或既有 event frontend 完成；
3. 论文中必须明确这是 **encoder attention subsystem**，不能把它夸大成完整 sensor-to-flow ASIC。

网络上下文仍保留：

```text
Patch Embed
→ S0/S1/S2/S3 encoder，共 12 blocks
→ 每 block: TTX attention → ADD residual → MLP → ADD residual
→ S0/S1/S2 downsample
→ skip / bottleneck / decoder
```

---

## 2. TTX 部署语义

TTX 的正式定义：

```text
all-binary ATLIFPSN
+ H60 no-carrier dataflow
+ TX-only score
+ bipolar_mu = 0
+ k_magnitude_alpha = 0
```

每 token/head：

```text
q, k ∈ {0,1}^32
overlap   = popcount(q & k)
same_zero = popcount(~q & ~k)
score     = (overlap + alpha0 × same_zero) / 32
gate      = Shiftmax(score over 162 tokens) × 162
out       = k_orig × gate
```

部署参数：

| 项 | 取值 |
|---|---:|
| `HEAD_DIM` | 32 |
| `N_TOKENS` | 162 |
| `alpha0` | 0.02，RTL 用 `5/256` |
| score | Q7 / INT8 部署语义 |
| gate | INT8 |
| threshold | 每 ATLIF site 一个共享 scalar，当前 checkpoint 为 1.0 |
| carrier | 无 |
| SC / mu | 删除 |
| K magnitude | 删除 |

valid825：

| 方案 | AEE | AAE | spikes | energy proxy |
|---|---:|---:|---:|---:|
| TTX float ep2 | 1.5020 | 9.8871 | 23.2395G | 20521.04 uJ |
| TTX score INT8 | 1.4971 | 9.8303 | 23.2434G | 20524.45 uJ |
| TTX score+gate INT8 | 1.5003 | 9.8266 | 23.2462G | 20526.92 uJ |

---

## 3. 系统结构

```text
                          descriptor scheduler
                 S0/S1/S2/S3, block/head/window
                                  │
                                  ▼
Q/K event SRAM ──► row loader ──► TX score ──► ZAF-Shiftmax ──► FGK stream
  1-bit packed       1 row/cfg      Q7          exact folding      K bits
                                                             + gate + theta
                                                                      │
                                                                      ▼
                                                         gate-late sparse MAC
```

物理实例不是 12 套：

| 物理模块 | 软件对应 | 复用方式 |
|---|---|---|
| TTX row engine ×1 | 12 attention blocks | descriptor 分时 |
| descriptor scheduler ×1 | 4 stages / 12 blocks | 固定 layer table |
| TX score leaf ×1 | 每 token 的 TX | 1 token/cycle |
| exp2 LUT ×2 | denominator / emit | row 内复用 |
| late-gate accumulator | 下游 projection lane | 输出通道分时/并行 |

---

## 4. 真实 row 调度

窗口为 `2×9×9=162` tokens。当前输入尺寸和 stage 尺寸对应：

| Stage | Blocks | Heads | T×H×W | Windows/block | Rows/frame |
|---:|---:|---:|---:|---:|---:|
| S0 | 2 | 3 | 10×72×96 | 440 | 2640 |
| S1 | 2 | 6 | 10×36×48 | 120 | 1440 |
| S2 | 6 | 12 | 10×18×24 | 30 | 2160 |
| S3 | 2 | 24 | 10×9×12 | 10 | 480 |
| 合计 | 12 | - | - | - | **6720** |

`ttx_descriptor_scheduler.sv` 已按该表发出 row job，testbench 验证总数为 6720。

---

## 5. 创新一：ZAF-Shiftmax

全称：**Zero-K Activity-class Folding Shiftmax**，中文为“零 K 活动类折叠 Shiftmax”。

### 5.1 关键观察

TTX 是 no-carrier：

```text
out_i = K_i × gate_i
```

当一个 token 的整个 K 向量为零：

```text
K_i = 0 → out_i 恒为 0
```

但不能直接删掉该 token，因为它的 `exp(score_i)` 仍属于 Shiftmax 分母。普通 pruning 会改变其它 token 的 gate。

### 5.2 精确折叠

对 `K=0` token：

```text
overlap = 0
same_zero = 32 - q_active
score = alpha0 × (32 - q_active) / 32
```

score 只由 `q_active∈[0,32]` 决定。因此建立 33-bin histogram：

```text
hist[c] = K=0 且 q_active=c 的 token 数量
```

Shiftmax 分母精确写成：

```text
sum_exp = Σ(K-active token) exp(score_i-max)
        + Σ(c=0..32) hist[c] × exp(score_class[c]-max)
```

这不是近似剪枝：

1. 每个 K-zero token 的分母贡献仍被保留；
2. 每个 K-active token 的 gate 与 dense 路径一致；
3. K-zero token 输出本来就是零，因此不需要发 gate/writeback；
4. 输出 SRAM 采用预清零或 valid bitmap，未发 token 保持零。

### 5.3 TTX 实测收益

TTX ep2 valid40 专项 profiling：

| Stage | K-zero token | active entries/row | fold classes/row |
|---:|---:|---:|---:|
| S0 | 78.15% | 35.40 | 2.99 |
| S1 | 97.41% | 4.20 | 1.39 |
| S2 | 93.91% | 9.87 | 2.47 |
| S3 | 89.45% | 17.09 | 2.25 |
| 按 6720 rows 加权 | **88.15%** | **19.20** | **2.43** |

每 row 的 exp 事务：

```text
dense = N sum + N emit = 324
ZAF   = A sum + C class + A emit
      = 2×19.20 + 2.43
      = 40.82
减少 = 87.40%
```

当前 RTL 为固定扫描 33 个 class，因此 cycle 模型是：

```text
dense cycles ≈ N load + N sum + N emit = 486
ZAF cycles   ≈ N load + A sum + 33 class scan + A emit
             ≈ 233.4
行级周期减少约 52.0%
```

后续将 class scan 改成 nonempty-class bitmap priority walk，可接近：

```text
N + 2A + C ≈ 202.8 cycles
```

### 5.4 与 TTB skip 的关系

TTB skip 只跳过整个 empty bundle；ZAF 处理 bundle 内部大量 K-zero token，并保留归一化贡献。

```text
TTB = coarse work-issue gating
ZAF = exact token-class denominator folding
```

两者互补，不应混写成同一个创新。

---

## 6. 创新二：FGK 因子化 gated-K

全称：**Factorized Gated-K stream**。

ATLIF official binary 输出：

```text
K = k_bits × theta
```

TTX 输出：

```text
Y = k_bits × theta × gate
```

不应在 attention core 中实例化 32 个乘法器并物化 32 路 INT16。RTL 输出：

```text
token_idx
k_bits[31:0]
gate_q8
threshold_q8
```

对下一层某个输出通道：

```text
Σ_i W_i × k_i × theta × gate
= (Σ_{i:k_i=1} W_i) × theta × gate
```

所以先做 spike-selected weight accumulation，再只做一次共享缩放。`ttx_late_gate_accum.sv` 已实现并验证该代数等价。

收益口径：

1. attention 输出保持 bitmask + scalar metadata；
2. 不写 32 路多 bit activation；
3. 将每 channel multiply 变成每 output lane/head 的 late scale；
4. 与 1-bit event SRAM 格式一致。

限制：跨 head 时每个 head 的 gate 不同，projection 必须按 head 分组累加后再合并。

---

## 7. 创新三：TTX 专用 dyadic score

TTX 不做 QK 矩阵，也不做 H60 SC：

```text
score_i 只比较同 token 的 q_i/k_i
```

RTL 只需：

```text
q_count
k_count
overlap = popcount(q & k)
same_zero = 32 - q_count - k_count + overlap
score_q7 = round(((overlap<<8) + 5×same_zero) / 64)
```

因为 `HEAD_DIM=32`、`SCORE_FRAC=7`：

```text
Q7 denominator = 2×HEAD_DIM = 64
```

所以除法被常数右移替代。相对旧 H60 core，删除：

1. SC score；
2. mu multiplier；
3. score fusion；
4. carrier；
5. K magnitude lane。

---

## 8. Center pass 消除

软件配置有 `center_scores=true`：

```text
centered_i = score_i - mean(score)
```

Shiftmax 先减 row max：

```text
centered_i - max(centered)
= score_i - mean - (max(score)-mean)
= score_i - max(score)
```

因此在同一 Q7 score 域内，center 对 Shiftmax delta 完全抵消。新 RTL 不再执行 score mean/division/center pass，直接用 raw Q7 score 与 raw row max。

这与当前硬件 Q7 语义严格一致；与 PyTorch“float center 后再 INT8 quant”仍需 golden vector 做最终误差验证。

---

## 9. RTL 模块

| 文件 | 作用 | 状态 |
|---|---|---|
| `ttx_tx_score_q7.sv` | TX-only popcount score | 已实现 |
| `ttx_zero_k_class_score_q7.sv` | K-zero class score | 已实现 |
| `ttx_exp2_lut_q8.sv` | Q7 delta 到 Q8 exp2 LUT | 已实现 |
| `ttx_ceil_log2_u32.sv` | next-power-of-two denominator | 已实现 |
| `ttx_row_engine.sv` | dense/ZAF row engine | 已实现 |
| `ttx_late_gate_accum.sv` | FGK late-gate proof module | 已实现 |
| `ttx_descriptor_scheduler.sv` | 12-block/6720-row 调度 | 已实现 |
| `ttx_attention_top.sv` | scheduler + row engine 顶层 | 已实现 |

---

## 10. 主要接口

### 10.1 Row request

```text
row_req_valid / row_req_ready
row_stage[1:0]
row_block[2:0]
row_head[4:0]
row_window[9:0]
row_n_tokens[7:0] = 162
```

### 10.2 Token input

```text
in_valid / in_ready / in_last
in_q_bits[31:0]
in_k_bits[31:0]
```

### 10.3 Sparse factorized output

```text
out_valid / out_ready / out_last
out_token_idx[7:0]
out_k_bits[31:0]
out_gate_q8[7:0]
out_threshold_q8[7:0]
```

协议要求：

1. ZAF 模式只发 K-active token；
2. destination row 必须预清零或维护 valid bitmap；
3. `out_last` 表示最后一个 K-active entry，而不是 token 161；
4. 全 K-zero row 无输出，直接 `done`。

---

## 11. 存储设计

当前原型内部存：

```text
active_score_mem
active_k_mem
active_token_mem
zero_k_hist[64]
```

综合检查使用 2 次幂物理深度：

```text
active store: 256 entries
class histogram: 64 entries
```

ASIC release 应改为：

```text
K bits 保留在 event SRAM
active FIFO 只存 token_idx + score
emit 时按 token_idx reread K bits
histogram 用 64×8 register file 或小 SRAM
```

这会把 prototype 的 K mask duplication 删除。当前 Yosys 把数组映成寄存器，不能作为 SRAM 面积。

---

## 12. 可画的论文图

1. **系统图**：event SRAM、scheduler、TTX row engine、FGK sparse MAC、skip SRAM。
2. **ZAF 图**：dense token list 对比 active FIFO + 33-bin histogram。
3. **精确性图**：K-zero token 不发输出，但 `hist×exp` 进入 denominator。
4. **FGK 图**：32 路乘法物化对比 bitmask accumulation + one late scale。
5. **调度图**：S0-S3 的 6720 row jobs 复用单 row engine。
6. **实测柱图**：四 stage 的 K-zero ratio、active entries、fold classes。

---

## 13. PPA 与 DC 口径

发文时必须分开：

| 项 | 口径 |
|---|---|
| logic area | DC 标准单元面积，不含 SRAM |
| SRAM area | compiler/CACTI，单独列 |
| dynamic power | TTX valid workload SAIF/VCD + PT-PX |
| leakage | 同一 PVT corner |
| throughput | 6720 rows/frame × 实测 cycles/row |
| ZAF speedup | dense 与 ZAF 同频、同接口、同 workload |
| output traffic | dense 162 entries/row 对比实测 active entries |
| accuracy | float、INT8、RTL golden 三列 |

现在可报告的是 generic synthesis 和事务减少，不能报告最终 ASIC mm²/mW。
