# 全 Class 关系转导架构：ACRT 与 AENR

> **第十五轮强基线纠错：** ACRT 相对当前串行 FSM 有 `1.345–1.371x`
> 子流水收益，但相对 `all-class + W1 ordered replay` 仅 `0.950–0.966x`，
> 相对理想 W4 replay 仅 `0.615–0.659x`。因此 ACRT 已降级为低状态/低端口
> PPA候选，不能宣称吞吐创新；AENR阈值结果也不能作为主结果。

## 1. 架构主张

本轮提出：

> **All-Class Relation Transduction（ACRT，全class关系转导）**：
> 将 H67 的 active token 与 K-zero token 统一折叠为 score-class
> histogram；在 denominator 完成后，以第二遍 active-class scan 直接产生
> final-gate destination relation，再与独立 K-lane relation 做因子交集。

它取代已淘汰的原 FCIP 和不成立的 zero-cost NC-FIP fused 主张。

---

## 2. 从现有 FSM 得出的机会

当前 SCS 数据流：

```text
LOAD
  active K token -> active_entry_mem
  K-zero token   -> score_hist

ST_SUM_ACTIVE
  每个active token计算exp并累加denominator

ST_FIND_FOLD/ST_SUM_FOLD
  每个K-zero score class计算一次exp×count

ST_EMIT
  每个active token重新计算gate并输出{token,K,gate}
```

当前 active token 付出两次逐 token 工作：

1. denominator 求和；
2. gate/K emit并构建投影 term。

H67 的离散语义保证：

- 同一 row 内，相同 score class 的 `exp_q8` 相同；
- 完整 denominator 确定后，相同 score class 的 `gate_q17` 相同；
- K value carrier 已由 `K[lane,destination]` 独立保存；
- 因此 gate 不需要按 token 重算，destination relation 可按 class 生成。

---

## 3. ACRT 数据流

### 3.1 LOAD

每个 token 只更新：

```text
hist[class] += 1
C[class][destination] = 1       // 仅active K需要projection relation
K[lane][destination] = K_bit
```

K-zero token仍进入 histogram，因此 Shiftmax denominator 不删除任何 token。

### 3.2 Pass 1：All-Class Denominator

```text
for occupied class:
    row_sum += count[class] * exp2(class - row_max)
```

现有 active-token sum 与 K-zero fold 被统一为一个 class scan。

### 3.3 Pass 2：Class-to-Gate Relation Transduction

```text
for active class:
    gate = gate_quant(exp[class], row_sum, n_tokens)
    gate_slot = exact_alias_lookup(gate)
    G[gate_slot] |= C[class]
```

该 pass 必须存在，不能宣称零周期。T162 使用三个独立 64-bit segment bank；
T450 使用八个 segment bank：

```text
{class_slot, gate_slot}
        |
        +--> seg0: C read -> local G OR
        +--> seg1: C read -> local G OR
        ...
```

所有 segment bank 本地更新，避免集中式 192/512-bit join。控制 tag 可采用
浅层树形分发，但树形网络本身不单列创新。

### 3.4 Segment-Major Factor Intersection

```text
read one G[gate][segment]
  -> broadcast to four lanes
  -> read K[lane0..3][segment]
  -> four 64-bit AND
  -> four bounded T-bit contexts
  -> single term ready/valid sink
```

一组四个 lane 完成并排空后再进入下一组，首版不假设跨组理想重叠。

---

## 4. AENR：有界前缀双模式

纯 class 模式在极稀疏行有固定 class scan 和 context fill/drain。为避免复制
dense/sparse 双核，LOAD 期间同时保留一个有界 event prefix：

```text
prefix entry = {score_class, token_id, 32-bit K-mask}
```

如果 LOAD 期可知的原始 K lane event 数不超过 E：

```text
token-style denominator + K-zero class fold
-> prefix lane枚举
-> singleton destination-bitmap term
```

超过 E：

```text
丢弃prefix
-> ACRT all-class mode
```

称为：

> **Adaptive Exact Normalization-Relation（AENR）**。

它迁移 Bishop 的 density stratification 思想，但：

- 不复制异构核；
- 不做 ECP 或有损删除；
- 分流对象是 exact relation representation；
- 判据在 final gate 产生前已经确定，不读取未来 gate 或 term 数；
- 两种模式共享 exp LUT、gate quant、term sink 和 DCTF backend；
- prefix 有界，超过阈值后不保留完整 active list 或 B1目录。

AENR 是 ACRT 的尾延迟保护，不单列第四个论文贡献。

---

## 5. 保守全链模型

新增：

- `scripts/model_acrt_full_pipeline.py`；
- `tests/test_acrt_full_pipeline.py`；
- `results/acrt_full_pipeline_model_20260730/report.*`。

算法整数合同已有两组独立证据：

- `results/acqn163_reference_20260730/report.*`：20,000 random rows、
  40,020 preserve-mean on/off checks，expanded candidate 与 all-class count
  的 `row_sum_q8/n_tokens/gate/member record` 零失配；
- `results/fcip_integer_reference_20260730/report.*`：因子交集 projection 与
  dense gated-K Acc 零失配。

二者合起来证明 ACRT 的算法分解，但仍不证明逐拍流水和回压。

### 5.1 基线

按现有 FSM 形状：

```text
active-token denominator
+ 2-cycle-per-class K-zero fold
+ active-token emit/G1 build
+ single term sink
```

### 5.2 ACRT

```text
2-cycle-per-class all-class denominator
+ 1-cycle-per-active-class gate fold
+ 1-cycle fold drain
+ strict segment-major G∩K
+ 1-cycle relation read latency
+ single term sink
```

### 5.3 结果

| sink ready | current mean | ACRT mean | aggregate speedup | p99 slowdown |
|---:|---:|---:|---:|---:|
| 100% | 71.07 | 51.82 | 1.371x | 1.548x |
| 90% | 72.64 | 53.64 | 1.354x | 1.561x |
| 75% | 76.29 | 56.73 | 1.345x | 1.675x |

ACRT aggregate收益已超过模型晋级门槛，但极稀疏行尾部不可接受。

加入强中间基线后：

| sink ready | all-class replay W1 | all-class replay W4 | ACRT | ACRT/W1 | ACRT/W4 |
|---:|---:|---:|---:|---:|---:|
| 100% | 49.24 | 31.89 | 51.82 | 0.950x | 0.615x |
| 90% | 51.18 | 33.78 | 53.64 | 0.954x | 0.630x |
| 75% | 54.82 | 37.38 | 56.73 | 0.966x | 0.659x |

这证明前述 `1.37x` 混合了 all-class normalization 和 relation dataflow
收益，不能全部归因于 ACRT。W4 是高端口、未综合的周期上界；ACRT 后续只能
通过同宏 PPA 证明面积/能耗优势，不能靠周期模型晋级。

AENR 在当前45行上的阈值扫描：

| E | aggregate speedup | p99 slowdown | >10%慢行 |
|---:|---:|---:|---:|
| 4 | 1.387x | 1.310x | 8.9% |
| 20 | 1.424x | 1.119x | 2.2% |
| 32 | 1.424x | 1.119x | 2.2% |
| 48 | 1.419x | 1.081x | 2.2% |

阈值不能在该样本上冻结。它只证明：

1. class-only aggregate有潜力；
2. 有界prefix能显著削弱尾部；
3. 仍需 profile100/fullres 独立数据选择阈值；
4. 当前 p99 尚未达到 `<=1.05x` 门槛。

---

## 6. 逻辑状态

T162 payload：

| 结构 | bit |
|---|---:|
| B1 G4×L32×T目录，不含其他状态 | 20,736 |
| ACRT C16+G4+K32+context4 | 9,072 |
| AENR E20 prefix | 980 |
| AENR E20合计 | 10,052 |

这不是面积。尚未包括：

- histogram/count；
- occupancy；
- tags/epoch；
- allocator；
- fallback；
- SRAM外围；
- clock gate与扫描。

---

## 7. 可写成 DATE 的三条贡献

### C1：All-Class Closed Normalization

利用 H67 离散 score class 和独立 K carrier，把 active/K-zero 两套执行统一为
all-class denominator，消除 active-token denominator replay。

### C2：Normalization-to-Relation Transduction

第二遍 class scan 不恢复 token stream，而直接生成 final-gate destination
relation，使 normalization 输出成为 projection 的原生控制平面。

### C3：Segment-Distributed Factorized Projection

`G[gate,dst]` 与 `K[lane,dst]` 分段驻留；gate tag树形分发、segment-local
fold、1G+4K segment-major intersection和有界context共同消除集中式宽join。

AENR、TTB metadata、descriptor 和 DCTF 是支撑机制，不单列贡献。

---

## 8. 文献借鉴边界

| 工作 | 借鉴 | ACRT差分 |
|---|---|---|
| Bishop | TTB metadata与density stratification | 分流exact normalization/relation表示；单共享后端；无ECP、无异构双核 |
| Prosperity | exact重复利用与残差思维 | 重复来自score-class代数闭包；不做在线pattern matcher/TCAM |
| Sanger | stationary sparse relation | 驻留final-gate destination与K-lane两个因子，不驻留完整score matrix |
| FLAT/STAR | 跨attention阶段数据流融合 | normalization直接生产projection关系，而非仅重排Q/K/V |
| FireFly-T | 多lane事件抽取 | 四路AND和prefix lane枚举是实现手段，不宣称原创decoder |
| PHI-like | pattern/residual两级表示 | 不学习codebook；prefix/class模式由精确event count决定 |

不能把 histogram、TTB、树形分发、双模式或位图单独称为首次提出。新颖性来自
三者在 H67 class/gate/K 语义下形成的完整执行合同。

---

## 9. 下一阶段

先不写全顶层，做最小 `T162/S16/G4/L32/SEG64/W4/CTX4`：

1. all-class histogram与C/K writer；
2. pass1 denominator；
3. pass2 segment-distributed gate fold；
4. segment-major 1G+4K intersection；
5. 四个真实T-bit context；
6. E-prefix与mode commit；
7. 单term sink、burst backpressure；
8. S16/G4 overflow、epoch、abort/replay；
9. 与当前 SCS+G1 的同端口、同SDC对照。

在写 RTL 前还需：

- 用新增真实 segment hook 跑 profile100/fullres；
- 将 AENR 阈值在独立样本冻结；
- 补 T450 8-segment 端口/控制模型；
- 写 all-class denominator 与现有 hybrid denominator 的整数等价参考。

停止条件：

- 多样本 aggregate低于1.15x；
- p99无法压到1.10x以下；
- segment分布式fold导致Fmax下降超过5%；
- AENR prefix超过关系状态节省；
- 同约束DC/SAIF EDP改善低于15%；
- 任一 denominator、gate、term或Acc mismatch。
