# 第十轮双线架构创新：LS-COTT 与 Source-Owned Wavefront

## 1. 本轮架构结论

本轮不再把 ACQN、双 context、3-bank、term 编码分别列为创新，而是冻结两条
改变数据拥有者、存储组织和执行顺序的候选：

1. **Motion：Lane-Sharded Class-Owned Term Transducer（LS-COTT）**；
2. **Local5：Source-Owned Stencil Wavefront（SOSW）**。

两条线不再强行共享 score/normalizer 核，只共享 exact term/projection 接口：

```text
Motion LS-COTT ---- SET term -----+
                                  +-> product-stationary multicast
Local5 SOSW ------ MPET term -----+   -> bank-local Acc
```

其中：

- SET 表示每个 destination 对同一 source contribution 只出现一次；
- MPET 保留 Local5 可能需要的 multiplicity；
- 两者都不能删除 denominator 中的合法候选；
- 两者都不能把 work reduction 直接写成周期或功耗收益。

---

## 2. Motion：LS-COTT

### 2.1 要解决的问题

当前 Motion 链路有两个分开的 token-major 阶段：

```text
SCS active member SRAM
  -> 逐 member exp/gate replay
  -> token gate + K
  -> G1 final-gate directory build
  -> term
```

保序 ACQN 只减少 exp/gate 求值，仍保留 member replay。固定 class bitmap
虽然能取消 replay，但真实 active score class 的 p99 为13，S16 才能把全局
class overflow 压到0.1%以下，导致每 context 状态达到86,041 bit，约为当前
SCS+G1两阶段30,159 bit的2.85倍，因此淘汰。

### 2.2 核心数据流

LS-COTT 在 score 到达时直接把 active K 转成 class-owned 稀疏 term/event：

```text
candidate {score_class, K[31:0], token}
        |
        +-- K==0 --> class_count[class]++
        |
        +-- K!=0 --> class_count[class]++
                     for each active lane:
                       lane_bank[lane].lookup/alloc(class)
                       append token to class-owned event list

ACQN denominator/gate
        |
        v
class -> final gate rename
        |
        v
per-lane gate alias chain
        |
        v
one {gate,lane} product
  stays resident across all aliased score-class fragments
        |
        v
destination event stream -> multicast -> Acc
```

关键变化不是“把 member 排序”，而是：

> active candidate 从进入硬件起就不再写入 token-major member SRAM；
> score class 直接拥有下游 term fragment。

### 2.3 物理组织

```text
Global class table
  163 count + occupied + gate

32 lane-local banks
  each:
    <=16 active score-class slots
    class id
    event-list start/count
    final-gate alias next
    token-id event SRAM

Shared fallback
  current ordered SCS + G1 path
```

一个 candidate 每拍只向每个 lane bank写至多一次，因为 32-bit K 的每一 bit
固定对应一个 lane bank，不需要通用 crossbar。多 bit K 可以在32个 bank
并行 append；真正风险是每 lane event SRAM 深度和分配器，而不是算术。

归一化完成后，class table 给出最终 Q1.7 gate。不同 score class 映射到同一
gate 时，不复制 destination：

- 每个 lane 建立 final-gate alias chain；
- product generator 对 `{gate,lane}` 只计算一次；
- product 在多个 score-class event fragment 间保持驻留；
- destination 集合逐 fragment 消费；
- 同一 token 只属于一个 score class，因此 class fragment 间不产生重复 token。

### 2.4 Ordered profile100 预筛

| 指标 | mean | p95 | p99 | max |
|---|---:|---:|---:|---:|
| active score class/row | 2.167 | 10 | 13 | 31 |
| score-class/lane term/row | 27.114 | 175 | 251 | 539 |
| final-gate/lane term/row | 10.567 | 51 | 67 | 146 |
| active lane event/row | 60.357 | 395 | 863 | 1643 |

容量预筛：

| 容量 | overflow row | overflow work | 状态 |
|---|---:|---:|---|
| global active class S16 | 0.0481% | 0.0628% | 可 exact fallback |
| shared term slot T256 | 0.8302% | 0.7166% | 条件可用 |
| shared event id E1024 | 0.6512% | 1.6921% | 条件可用 |

三种 overflow 的联合比例不能由边缘分布相加得到，必须补 joint trace。

### 2.5 存储账本

| 结构 | bit/context | 结论 |
|---|---:|---|
| 当前 SCS active+hist | 9387 | 基线 |
| 当前 G1 S4 bitmap directory | 20772 | 基线 |
| 当前两阶段合计 | 30159 | 公平比较对象 |
| fixed class bitmap S4 | 23833 | overflow 18.71%，不可用 |
| fixed class bitmap S16 | 86041 | 淘汰 |
| LS-COTT sparse lower bound | 22233 | 比当前低26.28%，尚非实现值 |

22,233 bit 包含：

- ACQN class state；
- 256个 term header；
- 1024个8-bit destination id；
- alias next 与 final-gate/lane head。

它没有包含：

- per-lane event SRAM 为避免多写冲突产生的容量碎片；
- free-list、端口复制、ECC、SRAM macro 对齐；
- exact fallback 的额外状态；
- 双 context 的两倍成本。

因此该数字只允许写“存储下界可行”，不能写“面积下降26.28%”。

### 2.6 借鉴与本土化

| 工作 | 借鉴 | 本工作差分 |
|---|---|---|
| Prosperity | exact reuse、关系驱动执行 | 不做在线 subset/matcher；关系是最终 score class 与二值 K lane |
| Sanger | stationary score 元数据 | class 直接拥有 term fragment，而不是驻留预测后的稀疏 score 做 SpMM |
| FLAT/FuseMax | 跨 attention 阶段融合 | 从 normalization 直接转 projection term，不物化 gated-K |
| Bishop | metadata-first 与可证伪 fallback | 不做 ECP、不复制 dense/sparse 双核 |
| Phi | anchor/residual 与压缩格式纪律 | 不使用 learned pattern/codebook；LS-COTT处理 normalize-to-term |
| FireFly-T | 多 lane 事件执行 | lane bank是一一映射，不声称通用可重构阵列 |

LS-COTT 的 DATE 差分必须落在：

```text
class-owned sparse term formation
  + late final-gate alias binding
  + product residency across class fragments
```

单独的 histogram、CAM、链表或多播均不算创新。

---

## 3. Local5：Source-Owned Stencil Wavefront

### 3.1 当前 RTL 的瓶颈

当前 `local5_stencil_linebuf_fetcher` 是 query/destination-major：

```text
for each center destination:
  read K_self
  read K_n
  read K_s
  read K_w
  read K_e
  score -> Shiftmax5 -> MFEP
```

三行 line buffer 已存在，但同一 source K 会被相邻五个 destination 重复读取。
pre-G0 profile100 给出：

| 指标 | query-major | source-owned | 减少 |
|---|---:|---:|---:|
| K lane read | 15.870G | 3.484G | 78.05% |
| active K lane read | 188.373M | 41.222M | 78.12% |

这是读取事务/bit-work，不是周期收益。

### 3.2 关系转置

Local5 原始 gather：

```text
destination q <- {Kself, Kn, Ks, Ke, Kw}
```

SOSW 转成 source-owned scatter：

```text
source K -> {qself, qnorth, qsouth, qeast, qwest}
```

数据流：

```text
3-row Q/K stripe
      |
source K read once
      |
fixed 5-direction router
      |
up to 5 destination score contexts
      |
degree-complete mask
      |
Shiftmax5 per destination
      |
streaming edge-gate plane
      |
source K reload/read once
      |
group incoming destinations by final gate
      |
{gate,lane,destination-set,multiplicity}
      |
shared projection sink
```

每条合法 stencil edge exactly-once。boundary-invalid edge 在 denominator 前
屏蔽；一个 destination 的合法 degree 全部完成后才能归一化。

### 3.3 为什么它不是“给 line buffer 改名”

当前 RTL：

- destination 拥有执行；
- 逐 destination 串行读取五个 K；
- 只有一个 destination score/Shiftmax context；
- MFEP 在 destination 完成后生成。

SOSW：

- source K 拥有执行；
- K 一次读取后广播到五个 destination context；
- 多个 destination context 波前并存；
- 归一化结果转置为 source-indexed edge-gate plane；
- projection 再按 source K 与 final gate 形成 multicast term。

改变了数据拥有者、活动划分、状态驻留和调度顺序，属于架构候选。

### 3.4 状态下界

| 结构 | 3x9 | 3x15 |
|---|---:|---:|
| destination score/完成 context | 1485 bit | 2475 bit |
| streaming edge-gate plane | 1215 bit | 2025 bit |
| 合计下界 | 2700 bit | 4500 bit |

未计 Q/K line buffer、FIFO、SRAM 对齐和投影 Acc。若不做 stripe streaming，
完整 T450 edge-gate plane 单独需要20,250 bit，因此默认必须采用3-row波前，
不能整窗物化。

### 3.5 公平性

不能只与当前单 K 读口 gather 比周期。公平基线必须有两条：

1. **B0-current**：单读口 query-major RTL，反映当前实现；
2. **B1-equal-lane**：五读口或等带宽 query-major Direct5，反映相同 row-rate
   的强基线。

SOSW 的目标不是虚构5倍吞吐，而是在接近 B1 row-rate 时使用更少 K SRAM
端口和读取事务。必须报告：

- K/Q SRAM 端口数；
- 每 window 读 bit；
- destination context 数；
- score lane 利用率；
- source router toggle；
- p95/p99 context occupancy；
- projection term/delivery；
- Fmax 与 EDP。

### 3.6 借鉴与本土化

| 工作/范式 | 借鉴 | 本工作差分 |
|---|---|---|
| stencil accelerator | 三行驻留、波前执行 | 把 attention gather 关系转置为 source-owned gated scatter |
| Prosperity | exact source/product reuse | 无在线相似度匹配；复用由固定五点图保证 |
| Phi | anchor/residual | self/source 是拓扑锚点，不使用 learned pattern |
| FLAT | attention-to-projection fusion | Shiftmax5 edge gate 直接形成 source multicast term |
| Bishop TTB | metadata-first tile | stripe携带degree/boundary，不做剪枝或ECP |
| 复旦 ISSCC 2023 butterfly | 可作为五向投递互连消融 | 默认使用固定方向router，不声称发明蝶形网络 |
| FireFly-T | 时空事件流水 | 这里的并行轴是source与destination wavefront，不复制通用双引擎 |

---

## 4. 可写入 DATE 的候选贡献

只有 RTL/PPA 通过门槛后，主文最多列三条：

### C1：LS-COTT

面向 all-binary Motion attention 的 lane-sharded class-owned term transducer，
在 score 收集时直接形成稀疏 class/lane/destination 结构，并通过 late
final-gate alias binding 取消 token-major member replay 和 gated-K 物化。

### C2：SOSW

面向 Local5 的 source-owned stencil wavefront，把五邻域 attention 从
query gather 反转为 source K scatter，用多 destination completion context 和
streaming edge-gate plane 将 K 读取从拓扑度数缩放改为 source 数缩放。

### C3：Exact relation-transposed term interface

SET/MPET 两种 exact term 在统一 projection sink 汇合，使 Motion 的
class quotient 与 Local5 的 stencil relation 都在不近似、不删 denominator
的条件下转换为 product-stationary multicast。

C3 若只是接口复用而没有跨线面积/EDP收益，应降为系统实现，不单列贡献。

---

## 5. 晋级和淘汰门槛

### 5.1 LS-COTT

- joint overflow row `<1%`，overflow work `<2%`，全部 exact fallback；
- per-lane class/event depth profile 完成；
- 相对当前 SCS+G1，attention-to-term EDP改善 `>=15%`；
- 总 SRAM+logic 面积不超过当前两阶段 `1.10x`；
- p99 row latency不超过当前 `1.10x`；
- 最终 Acc commit零失配；
- 若 lane SRAM 碎片使状态超过当前 `1.20x`，淘汰。

### 5.2 SOSW

- post-G0/fullres ordered K-read bit减少 `>=40%`；
- 相对 B1 equal-lane query gather，完整窗口 EDP改善 `>=15%`；
- row-rate不低于 B1 的 `0.95x`；
- source router 导致 Fmax下降不超过10%；
- destination context无丢边、重复边和 overflow；
- 最终 Acc commit零失配；
- 若只赢当前单端口弱基线而不赢 B1，不能列主贡献。

---

## 6. 下一轮最小工作

1. Motion profiler 增加逐 row、逐 lane：
   - active score class count；
   - class-lane term count；
   - event-list depth；
   - gate alias count；
   - S16/T256/E1024 joint fallback；
2. 用 ordered trace 建 LS-COTT lane-bank cycle model；
3. Local5 post-G0/fullres trace到达后建立 source-major edge顺序；
4. 比较 B0单口 gather、B1等带宽 gather、SOSW；
5. 两个模型分别做独立 DATE 复审；
6. 只有通过门槛的候选进入最小 RTL。

当前架构冻结状态：

| 候选 | 状态 |
|---|---|
| Motion fixed class bitmap | 淘汰 |
| Motion ordered ACQN | RTL基线，不列贡献 |
| Motion LS-COTT sparse | 条件晋级 |
| Local5 query-major TARE | 负结果/基线 |
| Local5 SOSW | 条件晋级 |
| DCTF双 context/3-bank | 支撑模块，不列贡献 |
