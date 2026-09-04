# H63 对称 ATLIF 与无 Gate-K 统一注意力探索

> 2026-07-11 勘误：本文件此前误把“禁止 native QKFormer carrier”写成“禁止所有 gate/value 与 K 的结合”。正确边界允许 `Shiftmax gate*K` 和 `attention weights@K`；禁止的是先构造 `K*sn2_q(sum(Q))`、再叠加第二个 gate 的原生 carrier。单边 `{0,+theta}` ATLIF 也合法，只有双边正负发放才要求阈值对称。H63 保留为错误强制对称与 direct-output 的负实验，不替代当前 TTX RTL 主线。

**日期**：2026-07-11  
**状态**：软件预注册，尚未进入 RTL 实现  
**范围**：DSEC encoder，12 个 attention block 统一公式

## 1. 不变的数据流边界

以下系统级接口和调度保持不变：

```text
PSN temporal transform
-> symmetric ATLIF event SRAM
-> window/head row scheduler
-> Q/K event score engine
-> attention normalization
-> projection/sparse MAC
```

禁止 stage 特化、TX/SC 双路径、Kmag lane 和运行时 mu 控制。已有 TTX RTL 不删除、不改语义；H63 在软件 valid40 晋级前只做接口评估。

## 2. 对称 ATLIF 编码

### 2.1 三值

```text
h >= theta   : POS
h <= -theta  : NEG
otherwise    : ZERO
```

两个比较器共享同一幅值阈值，SRAM 每 event 使用 2 bit。

### 2.2 二值 magnitude event

```text
event = (h >= theta) OR (h <= -theta)
```

两个比较器共享 `theta`，OR 后只写 1 bit，不保存 polarity。相对现有单边 `binary_atlif_unit` 增加一个有符号比较器和一个 OR；阈值 descriptor、PSN temporal transform、SRAM bitwidth 均不变。

不采用 dense `{-1,+1}` binary，因为它没有零事件，无法进行 event skip。

## 3. Direct-Shiftmax block

head_dim=32 被划分成 `G` 个连续 channel groups：

```text
Q/K bits
-> per-group overlap and silent-match count
-> dyadic TX score (64:1)
-> Shiftmax over tokens
-> subtract uniform baseline (gate - 1)
-> gate descriptor broadcast to channels in the group
-> projection
```

与当前 TTX 的差异：

| block | TTX | H63 direct |
|---|---|---|
| score front-end | one TX score/head/token | G TX scores/head/token |
| normalization | one Shiftmax context/head | G contexts/head |
| K event reread | required | removed after score generation |
| FGK late scale | required | removed |
| output metadata | token + K bits + gate | token + G gate descriptors |
| channel rank before projection | K-defined | at most G |

## 4. Candidate cost/risk

| candidate | contexts/head | gate descriptors/token | expected cycle multiplier | accuracy risk |
|---|---:|---:|---:|---|
| G1 | 1 | 1 | 1x | very high: rank-1/head |
| STC | N + D | token gate + channel gate | two serialized normalizations | medium |
| G4 | 4 | 4 | up to 4x if fully serialized | medium-high |
| G8 | 8 | 8 | up to 8x if fully serialized | medium |
| G32 | 32 | 32 | up to 32x | area/bandwidth unacceptable unless strongly justified |

STC accumulates one token score vector and one 32-entry channel score vector from the same TX evidence, then computes `0.5*((gate_token-1)+(gate_channel-1))`; the factor 0.5 is a fixed right shift. This is a single factorized TX operator, not mixed attention. The context count need not equal physical lane count. A balanced implementation can instantiate 2 or 4 Shiftmax lanes and time-multiplex group contexts. Descriptor SRAM bandwidth, not the TX popcount, is expected to become the main local bottleneck.

## 5. RTL decision gate

No H63 RTL is authorized before software valid40. RTL promotion requires:

1. all12 software path and checkpoint reload audit pass;
2. valid40 AEE <= 1.65 with a converging trend;
3. activity below 20% in the short run;
4. a measured advantage over scalar direct output that justifies G>1 context cost.

If direct output fails, the fallback is not TX/SC mixing. The only fallback is an all12 dyadic shift-value block where Shiftmax emits powers-of-two and the value/event stream is shifted or selected. That fallback reuses the current TTX scheduler and event SRAM but must be named separately because it is no longer a no-carrier design.

## 6. Required software-to-hardware observables

Future profiling must export per stage/head:

- group score min/max and histogram;
- Shiftmax denominator exponent per group;
- output gate sparsity/entropy;
- group-to-group correlation, to detect redundant contexts;
- symmetric-binary positive-trigger and negative-trigger rates before OR;
- attention output activity after `attn_sn`;
- estimated descriptor bytes and serialized cycles for G1/G4/G8.

These observables are block-local and align directly with the RTL row-engine granularity.

## 7. H64 centered-symmetric descriptor extension

H63 暴露出已有 PSN 权重的膜电位分布并不以零为中心。H64 不改变 ATLIF 对称性，而是把对称轴固化为离线校准 descriptor：

```text
lo_t = c_t - theta
hi_t = c_t + theta
NEG if h <= lo_t
POS if h >= hi_t
ZERO otherwise
```

硬件直接存 `lo_t/hi_t`，因此相对零中心 ternary ATLIF：

- comparator 数量不变，仍为两个；
- 每个分时 neuron descriptor 从一个 `theta` 扩为每 timestep 两个 bound；
- 不增加运行时减法、target-rate、统计器或反馈 FSM；
- event SRAM 和 signed TX score engine 不变。

H64-ref 只用现有 H60/TX 隔离验证 centered neuron，不代表恢复 gate-K 主线；只有 H64-STC 的无 carrier 结果达到门槛才会推动 direct attention RTL。

## 8. H65 signed Hamming 备用块

H65 对应 ICML 2025 SpikeVideoFormer 的线性 Hamming 范式在当前 no-V block 上的 K-reuse 版本：

```text
M[D,D] = K_sign^T * K_value
Y[N,D] = Q_sign * M >> 6
```

事件乘法映射为条件 add/subtract，没有 gate-K、Shiftmax、TX/SC 混合或 `N×N` attention SRAM。代价是每 head/window 需要 `D×D=1024` 个多位 accumulator state，运算量 `O(ND^2)`；若串行复用 accumulator，延迟会明显高于 TTX。只有 all105/all12 DSEC 20-step 同时通过精度和 `<20%` activity 门槛，才值得新增 RTL；否则保持文献负对照，不进入硬件主线。

## 9. 软件门槛最终结果

| candidate | activity/firing | AEE | hardware decision |
|---|---:|---:|---|
| centered signed-TX STC, step20 | 41.57% activity | 9.78 | no RTL |
| raw signed-TX STC, valid1 | 53.92% firing | 13.85 | no RTL |
| centered-symmetric H60 ref, valid1 | 59.63% firing | 11.88 | centered descriptor rejected |
| signed Hamming, step20 | 45.12% activity | 8.36 | no D×D accumulator RTL |

所有实验均通过 `ATLIF=105`、attention patch `=12`、checkpoint overlay `210/0/0` 审计，失败不能归因于漏装模块或权重未加载。没有候选进入120/360/full/valid825，因此当前不修改 TTX RTL datapath，不删除 FGK/K reread，也不新增 H63/H65 block。

硬件主线状态必须明确区分：

- `rtl_ttx/` 是已验证的 one-sided binary + factorized gated-K 数值参考实现；
- 它不满足新提出的“strict symmetric ATLIF + no K carrier”双约束；
- H63-H65 是已证伪的软件架构探索，不应写成待集成正向模块。
