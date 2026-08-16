# Motion 与 Local5 双线 DATE 硬件贡献冻结及并行路线

**日期**：2026-07-28
**目标**：Motion 与 Local5 并行推进，共享可复用硬件，但不强行统一不相同的数据流。
**范围**：只冻结能在 DATE 论文贡献列表中单独编号的硬件架构贡献。

---

## 1. 结论

建议论文最终保留四条硬件贡献：

1. **C1：Semantic-Anchor Exact Residual Execution，语义锚定精确残差执行**
2. **C2：Class-Closed Normalization-to-Term Dataflow，类别闭合的归一化到 term 数据流**
3. **C3：Topology-Stationary Bounded-Multiset Projection，拓扑驻留的有界多重集投影**
4. **C4：Set/Multiset-Polymorphic Term-Atomic Projection Fabric，集合/多重集多态的 term 原子投影互连**

对应关系：

```text
                         C1 共享 TARE
                 ┌──────────┴──────────┐
                 │                     │
Motion 前端      C2                    C3       Local5 前端
Pair/SCS/NMF(set)│                     │Stencil/Shiftmax/MFEP(multiset)
                 └──────────┬──────────┘
                            C4
                多态 term-atomic bank-local backend
```

四条贡献不是都已完成。当前状态是：

| 贡献 | 当前状态 | 是否可写成“已完成” |
|---|---|---|
| C1 | 双模式叶/复合核/Local5 行 TB 已有 | 条件可写，缺同约束 PPA 和真实收益 |
| C2 | Motion SCS/NMF/DCTF 子模块较完整 | 只能写 prototype，缺统一顶层和多 trace |
| C3 | Local5 行、MFEP、line-buffer、窗口原型已有 | 不能冻结，缺 true-mask 软件闭环和真实 trace |
| C4 | Motion DCTF-2C 已有；Local5 只有 adapter/轻量 Acc | 不能写双线多态，必须补统一 schema 和后端 |

---

## 2. C1：语义锚定精确残差执行

### 2.1 架构主张

通用复用架构需要在线搜索相似 operand。事件光流网络直接提供确定锚点：

- Motion：时间 peer 是锚点；
- Local5：self K 是拓扑锚点。

同一物理核按更新数执行：

```text
ZERO       0 lane update  -> 直接复用 anchor raw
LIST4      1..4 updates   -> 有符号 residual 修正
REPLAY     >4 updates     -> 同一 32-lane 核精确重放
```

三条路径只在最后执行一次 RNE，不引入近似。

### 2.2 借鉴与本土化

| 来源 | 借鉴 | 本工作变化 |
|---|---|---|
| Prosperity | exact product/residual reuse | 静态网络锚点，取消在线关系发现 |
| Bishop | density stratification | 不复制 dense/sparse 双核，在同一精确核中路由 |
| FireFly-T | bounded multi-lane extraction | 4-lane signed delta，用于 alpha-XNOR raw 修正 |

### 2.3 为什么能单独列贡献

它改变了 score engine 的组织、路由和共享方式，而不只是减少一个 XOR：

- 两种 attention 共享一个物理执行底座；
- 稀疏和稠密路径保持整数等价；
- 取消通用 reuse detector、预测和恢复路径。

### 2.4 必须补的证据

1. Motion/Local5 真实 trace 的 `0/1–4/>4` 分布；
2. Direct32、双 Direct32、TARE 的同 SDC/SRAM DC/STA/SAIF；
3. 每模式 cycle、切换气泡、FIFO 峰值和 replay 比例；
4. 统一 score packet 的 bit-exact 回放。

---

## 3. C2：类别闭合的归一化到 term 数据流

### 3.1 适用主线

**Motion 特有贡献。**

Motion 的离散 score、SCS-Shiftmax 和 gate-code term 形成有限类别链：

```text
score class / occupancy
        -> Shiftmax denominator
        -> gate class
        -> NMF set term
        -> projection
```

目标不是“提出 Shiftmax”，而是让归一化产生的类别元数据直接成为投影调度信息，
不写出完整 attention matrix，也不写出完整 gated-K tensor。

### 3.2 架构主张

```text
Pair Score
  -> occupied-class histogram
  -> denominator and gate LUT
  -> active {gate,K,destination} directory
  -> set-term projection
```

zero-K、empty pair 和未占用 score class 采用 closed-form exact commit，
不做近似 pruning。

### 3.3 借鉴与本土化

| 来源 | 借鉴 | 本工作变化 |
|---|---|---|
| FLAT/FuseMax | attention stage fusion | 离散类别元数据跨 Shiftmax 和 projection 原地演化 |
| Sanger | score-stationary 思想 | 从 score 驻留扩展为 gate-class 驻留 |
| SpAtten | cascade issue | 只采用 exact empty/K-zero/class-empty issue |
| Bishop | bundle-first | TTB 只承载 exact metadata，不采用 ECP |

### 3.4 为什么能单独列贡献

它是跨算子的存储和数据流架构：

- 改变 score、normalization、projection 的中间表示；
- 删除 attention/gated-K 物化流量；
- 让类别占用同时驱动计算 gating 和投影调度。

SCS 占用扫描、NMF、TTB 和 zero-K folding 都属于 C2 的组成机制，不单独计数。

### 3.5 必须补的证据

1. 统一 `TARE -> SCS -> NMF -> DCTF` 顶层；
2. materialized、token-stream、class-closed 三种数据流的 SRAM bytes/cycle；
3. 多 sample/window 的 mean/p95/p99；
4. SCS/NMF overflow 和 fallback 的完整 bit-exact 验证。

---

## 4. C3：拓扑驻留的有界多重集投影

### 4.1 适用主线

**Local5 特有贡献。**

Local5 不是普通 set multicast。同一 destination 中，多条方向边可能具有相同
`(gate,lane)`，贡献必须累加 1 至 5 次。普通 bitmap OR 会丢失重数。

### 4.2 架构主张

```text
T2 x 3-row x 9-col K residency
  -> self/N/S/E/W exact masked score
  -> Shiftmax5
  -> MFEP {gate,lane,multiplicity,destination}
  -> gate-stationary product reuse
```

核心有两部分：

1. **Topology-stationary**：K 保持在三行/双时间片存储中，按固定方向读取；
2. **Bounded multiset**：用小重数保存 1–5 次重复贡献，直接进入投影。

### 4.3 借鉴与本土化

| 来源 | 借鉴 | 本工作变化 |
|---|---|---|
| LoAS | source/temporal stationary | 固定 `T2×3-row×9-col` stencil 驻留 |
| FLAT | 融合数据流 | Shiftmax5 后直接形成 MFEP term |
| Phi | pattern+residual 思想 | 只在真实低基数成立时编码方向/重数模式 |
| Bishop | tile bundle | bundle 加入边界 mask、方向和 multiplicity |

### 4.4 为什么能单独列贡献

它同时改变：

- K 的存储层次和读取顺序；
- attention 到 projection 的中间表示；
- set projection 无法表达的重复贡献语义；
- gate/product 的复用粒度。

三行 line-buffer 本身不是创新；MFEP 单独作为计数压缩也不够。二者结合成
topology-stationary multiset tileflow 后才构成架构贡献。

### 4.5 必须补的证据

1. true-mask valid825 保持 Local5 精度；
2. 完整 `T=2×9×9`、12 module ordered trace；
3. gate cardinality、multiplicity、term、K-read 和 line-buffer hit profile；
4. naïve gather、edge-term、set-plane、MFEP 四种公平基线；
5. 真实 SRAM latency 和 backpressure 下的净 cycle/energy。

---

## 5. C4：集合/多重集多态的 term 原子投影互连

### 5.1 共享后端目标

Motion 和 Local5 的 projection 都可用窄 term 驱动，但语义不同：

```text
Motion: set term       {gate,lane,destination bitmap}
Local5: multiset term  {gate,lane,destination,multiplicity 1..5}
```

目标后端增加显式 `schema`：

```text
schema=SET:
    bitmap / PPDI destination issue

schema=MULTISET:
    destination + multiplicity

shared:
    whole-term validation
    narrow-command multicast
    bank-local weight/product/Acc
    ordered retirement
    dual-context overlap
```

### 5.2 架构主张

它不是“3-bank + 双缓冲”，而是：

> 一个保持 set/multiset 精确语义的 schema-polymorphic term fabric，
> 共享前端验证和窄命令分发，同时允许 bank-local executor 独立推进。

### 5.3 为什么能单独列贡献

只有完成以下差异后才能成立：

- 同一互连原生支持两种代数语义；
- 不通过展开 multiplicity 退化为重复命令；
- 不通过统一成宽 payload 失去窄命令优势；
- Local5 和 Motion 共用实际 reader/executor，而不是两个 wrapper 名义共享。

### 5.4 当前不能写的

- Local5 `local5_multibank_projection_top` 是单发射模型，不是 C4；
- 双 context、三 bank 和 ordered queue 单独都不新；
- 当前 Motion DCTF-2C 只能作为 C4 的 SET 基线实现。

### 5.5 必须补的证据

1. 统一 term schema、adapter、fabric 和 bank executor；
2. Central96、3×Independent32、SET-only DCTF、MULTISET-only、
   polymorphic fabric 的公平对照；
3. 同一 SRAM macro、端口数、lane 数和 SDC；
4. 多 trace 吞吐、bank imbalance、FIFO、面积、功耗和 EDP。

---

## 6. 不再单独列为创新的内容

| 机制 | 归属 |
|---|---|
| alpha-XNOR / Motion-XOR 定点化 | 数值合同，不是架构贡献 |
| Shiftmax / SCS-Shiftmax | C2/C3 的归一化算子 |
| TTB/STT | C2/C3 的 bundle 基础设施 |
| line-buffer | C3 的存储实现 |
| MFEP term builder | C3 的多重集编码器 |
| DCTF 双 context | C4 的延迟隐藏机制 |
| 3-bank | 只有真实并行和 PPA 收益时才属于 C4 |
| PPDI | C4 的 SET destination issue 优化 |
| descriptor 时间复用 | 全 encoder 实现机制 |
| bit-exact / ZUI | 正确性属性，不是架构 |
| ATLIF=93/105、Shiftmax=12 | workload/资源规模，不是创新 |

---

## 7. Motion 与 Local5 并行推进矩阵

### 7.1 Motion 分支

| 编号 | 工作 | 服务贡献 |
|---|---|---|
| M1 | 统一 TARE→SCS→NMF→DCTF 顶层 | C1/C2/C4 |
| M2 | 多 sample/window ordered trace 回放 | C1/C2/C4 |
| M3 | materialized/token/class-closed 流量对照 | C2 |
| M4 | SET schema、fallback、overflow 签核 | C2/C4 |
| M5 | Central96/DCTF 同约束 DC/STA/SAIF | C4 |

### 7.2 Local5 分支

| 编号 | 工作 | 服务贡献 |
|---|---|---|
| L0 | true-mask 软件 valid825 | C3 的算法合同 |
| L1 | TARE 参数化接入 window/linebuf 顶层 | C1/C3 |
| L2 | 完整 T2×9×9 ordered trace | C1/C3/C4 |
| L3 | MFEP 与 edge/set-plane 公平 DSE | C3 |
| L4 | MULTISET schema 接入共享 fabric | C4 |
| L5 | 真实 9-col 双时间片 SRAM/反压 TB | C3 |

### 7.3 共享分支

| 编号 | 工作 | 服务贡献 |
|---|---|---|
| S1 | 冻结统一 score packet 与 term schema | C1/C4 |
| S2 | schema-polymorphic fabric RTL | C4 |
| S3 | 统一 cycle/traffic/energy trace simulator | 全部 |
| S4 | 同 SDC/SRAM macro PPA | 全部 |
| S5 | full encoder Amdahl 与 FPS/EDP | 系统结论 |

---

## 8. 论文贡献段建议写法

在证据全部关闭后，贡献段可以写成：

1. 提出一种**网络语义锚定的精确残差 score 架构**，以共享的
   ZERO/LIST4/REPLAY 核执行时间和拓扑 attention，无需在线复用发现或近似恢复。
2. 提出面向 Motion attention 的**类别闭合 normalization-to-term 数据流**，
   使 score occupancy、Shiftmax 和 set projection 之间不物化 attention/gated-K。
3. 提出面向 Local5 attention 的**拓扑驻留有界多重集数据流**，在
   `T2×3-row×9-col` 驻留中保留 1–5 次重复边贡献并复用 gate/product。
4. 提出一个**集合/多重集多态的 term 原子投影互连**，以共享窄命令分发和
   bank-local executor 精确支持两条 attention 流，并给出同约束 PPA 和 EDP。

当前只能称这些为“冻结目标贡献”。只有对应 profile、RTL、bit-exact、PPA
全部通过后，才能把“提出”写进最终论文。

---

## 9. 当前 DATE 判断

如果只提交当前证据：

- C1：机制原型；
- C2：Motion 子系统；
- C3：Local5 合成窗口原型；
- C4：尚未形成双线多态后端。

因此仍是 **Borderline Reject**。

若 C2/C3 分别证明两种前端的独立数据流收益，C4 又证明共享后端相对
两套专用后端有面积/EDP 净收益，则论文不再是“小模块拼接”，而是一套：

> 由网络语义选择前端表示、由统一 term fabric 执行投影的双数据流事件光流架构。

这是双线并行的最终目标。
