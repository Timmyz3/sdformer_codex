# Term 在 Local5 中的统一语义、TTB/STT 贡献升级与全分辨率口径

日期：2026-07-28

## 0. 直接结论

1. **Term 不仅能用于 Local5，而且 Local5 已经有 term RTL。**
   `local5_mfep_term_builder` 输出的是带 `multiplicity=1..5` 的
   bounded-multiset term。它不是 Motion 的 set term，也不能把重数丢掉。
2. **TTB/STT 可以升级为 DATE 单列贡献，但不能只写“打包描述符”。**
   必须把它实现成跨 `load -> score -> normalize -> term -> retire` 生命周期的
   语义驻留 tile，并证明减少 payload fetch、中间量物化和调度开销。
3. **Term 与 TTB/STT 不重复。**
   TTB/STT 是输入侧驻留和调度单元；term 是输出侧投影复用和提交单元；
   中间由 exact tile-to-term transducer 连接。
4. **最终硬件论文必须按 DSEC 480x640 全分辨率评估。**
   DC 不需要实例化一整帧阵列，但系统周期、带宽、SRAM 和 FPS 必须累计整帧。
   crop 只能作为兼容性和早期 workload 证据。
5. 当前建议同时保留 `w9-162` 和 `w15-450` 两个 RTL 参数点；
   最终 DC 主点由 full-resolution 软件精度主线决定。当前软件队列正在先跑
   `480x640 + T2x15x15` 的 NB0，再按队列进入 H67/Local5，因此
   **w15 应作为容量设计目标，但还不是已冻结算法主点**。

当前证据分层：

| 机制 | 状态 |
|---|---|
| per-destination multiplicity-tagged MFEP | `[CURRENT RTL]` |
| Local5 小规模乘重数 projection | `[CURRENT RTL prototype]` |
| STT phase descriptor | `[PROTOTYPE]` |
| 跨 destination MPET | `[PROPOSED]` |
| RES-Tile 执行生命周期 | `[PROPOSED]` |
| multiset-aware 共享 DCTF | `[PROPOSED]` |
| w15 bit-exact/PPA/full-frame 结果 | `[MISSING]` |

---

## 1. Local5 里现在有没有 term

### 1.1 已有 RTL 语义

`rtl_local5/local5_mfep_term_builder.sv` 对一个 destination 的最多 5 条邻边
执行：

```text
m[lane, gate]
  = 满足 K(edge,lane)=1 且 gate(edge)=gate 的有效 edge 数

输出：
{tag, destination, lane, gate, multiplicity=m, last}
```

因此当前 Local5 term 是：

```text
Local5 current term
  = {gate, lane, destination, multiplicity 1..5}
```

它已经通过以下独立定向验证：

- `tb_local5_mfep_term_builder.sv`；
- `tb_local5_mfep_sparse_last.sv`；
- `tb_local5_score_gate_term_top.sv`；
- `tb_local5_zero_term_window.sv`。

最终回归见：

```text
build_local5/parity/final_regression_term_tile_postreview_20260728.log
```

### 1.2 Motion 与 Local5 的 term 差异

```text
Motion:
  同一 destination/lane 只有一次 gated-K contribution
  -> set term
  -> multiplicity 恒为 1

Local5:
  同一 destination/lane 可从 self/N/S/E/W 收到相同 gate
  -> bounded multiset term
  -> multiplicity 为 1..5
```

所以错误做法是：

```text
Local5 edge contribution -> OR 成 Motion bitmap
```

这会把两个或多个相同贡献错误地压成一次，破坏整数累加结果。

正确的统一式是：

```text
Acc[d,o] += m * gate * W[lane,o]
```

其中 Motion 是 `m=1` 的退化情况。

### 1.3 当前还缺什么

已有 Local5 term 不等于共享 term 架构已经完成：

| 层次 | 当前状态 | 缺口 |
|---|---|---|
| term builder | 有，输出 per-destination multiset term | 未跨 destination 聚合 |
| adapter | 有 sideband 和 explode 两种模式 | explode 会丢掉 command 压缩收益 |
| Local5 projection | 有小规模乘重数 Acc | 当前 multi-bank 仍是单 issue/1 IPC 原型 |
| Motion DCTF | 有真正三消费者 ordered fabric | payload 没有 multiplicity |
| 共享后端 | 尚无 | 需统一 schema、fabric、bank executor、fallback |

因此论文现在只能写“Local5 已有 multiset term 原型”，不能写
“Motion/Local5 已共享完整 term fabric”。

---

## 2. Local5 term 的目标升级：Multiplicity-Plane Term `[PROPOSED]`

当前 per-destination term 仍没有把 Local5 的有界邻域利用到底。

`docs/150` 已经提出过五个 multiplicity destination plane；本节不是重新声称
首次提出，而是把该候选收敛成 Motion/Local5 共享 term schema，并补上
full-resolution segmented destination 和共享后端的晋级条件。

在同一 `group_tag/head/absolute_input_channel/output_supertile` 上，对固定
`(gate, lane, multiplicity=m)`，所有 destination 的 product 完全相同：

```text
product[o] = m * gate * W[lane,o]
```

因此可把相同完整 key 的 destination 再聚合成一个集合：

```text
MPT = {
  gate,
  lane,
  multiplicity m,
  destination bitmap/list
}
```

五个 multiplicity plane 为：

```text
plane1: m=1  -> destination set
plane2: m=2  -> destination set
...
plane5: m=5  -> destination set
```

该变换还要求 accumulator 在 term 重排区间内使用足够宽的整数累加，并且不做
逐 edge 中间饱和；否则整数饱和会破坏加法结合律，必须保留原顺序 replay。

这得到一个统一关系：

```text
Motion set term = Local5 MPT 的 plane1 特例
```

### 2.1 为什么比当前 MFEP 更强

| 当前 per-destination MFEP | Multiplicity-Plane Term |
|---|---|
| 每个 destination 一条 command | 同 `(gate,lane,m)` 多 destination 一条 term |
| product 可能重复计算 | product 每个 key 计算一次 |
| 仅利用候选内重数 | 同时利用候选内重数和跨 destination 复用 |
| 与 Motion schema 不完全同构 | Motion 自然落在 `m=1` plane |

### 2.2 这条可以怎么写成贡献

建议名称：

> **Multiplicity-Plane Exact Termization，MPET：**
> 利用 Local5 邻域度上界 5，把多重集贡献映射为五个精确 multiplicity plane；
> Motion 的集合投影作为 `m=1` 特例，从而由同一 term-stationary 后端执行两种
> attention 数据流。

它不是把已有 DCTF payload 多加 3 bit。要成为贡献必须同时实现：

1. `(group,absolute-input-channel,output-tile,gate,lane,m)` 目录；
2. segmented destination set；
3. 不展开 multiplicity 的 fabric；
4. bank 内 `m*gate*W` 共享 product；
5. set/MPT 与 dense gated-K 的整数等价；
6. 相对 per-edge、per-destination MFEP、explode-DCTF 的周期/能耗消融。
7. 证明乘重数中间位宽、Acc 位宽和饱和边界不会改变重排结果。

---

## 3. TTB/STT 能否单独列成贡献

### 3.1 当前为什么还不能

当前 Motion 的 TTB 主要是 profile/打包/empty gating 证据；
`local5_stt_descriptor.sv` 也明确只是：

```text
issue -> score -> term -> commit
```

的薄生命周期 sideband。普通 bundle、descriptor、三行 line-buffer 都是常见机制，
单独改名不构成 DATE 架构贡献。

### 3.2 升级条件：从 bundle 变成 Semantic Residency Tile

目标上建议把 TTB/STT 统一提升为：

> **Resolution-Elastic Semantic Residency Tile，RES-Tile**

它不是数据包，而是硬件调度、驻留、跳过和结果重塑的最小原子：

```text
Frame/stage descriptor
  -> RES-Tile issue
  -> payload residency
  -> exact anchor/TARE
  -> normalization
  -> set/MPT termization
  -> term commit
  -> tile retire
```

两种前端：

```text
Motion RES-Tile / TTB:
  T2 temporal-pair payload
  empty/K-zero/delta class
  segmented active-token metadata

Local5 RES-Tile / STT:
  T2 x 3-row resident K
  self/N/S/E/W valid mask
  boundary/degree/delta class
```

### 3.3 RES-Tile 必须承载的真实行为

| 行为 | Motion | Local5 |
|---|---|---|
| metadata-first | empty/K-zero/delta | degree/boundary/delta |
| payload gating | 空 bundle 不取 Q/K payload | invalid neighbor 不取 K |
| anchor reuse | temporal peer anchor | self-K topology anchor |
| normalization state | 35-class SCS histogram | 5-way Shiftmax state |
| term output | set term | MPET multiset plane |
| resolution scaling | token segment 数变化 | row span/destination 数变化 |
| exact fallback | dense replay | direct five-edge replay |

如果同一个 tile record 只在入口出现一次，之后仅更新 phase、pointer、count 和
term commit 状态，而不重新生成 token-major score/gate tensor，才可以写：

```text
semantic tile orchestration
```

### 3.4 与外部工作的差异边界

| 工作 | 借鉴 | 本工作必须形成的差异 |
|---|---|---|
| Bishop | TTB、metadata-first、density | 不用 ECP；tile 跨 normalize/term 生命周期 |
| LoAS | temporal inner packing | T=2 pair 与 Local5 3-row topology 双模式 |
| Prosperity | exact temporal reuse | temporal/topology anchor 共用 exact residual engine |
| PHI/稀疏格式工作 | payload/metadata 分离 | set/multiset term 是网络语义，不是通用 CSR |
| FLAT/FuseMax | attention operator fusion | score class/五邻域直接转 term，不物化 A/gated-K |

必须引用来源并写“borrow/modify/not borrow”，不能把 TTB 改名后宣称首次提出。

---

## 4. Term 与 TTB/STT 的关系

两者不是同一个贡献：

```text
输入复用域                          输出复用域

TTB / STT / RES-Tile               Set / MPET term
{payload, geometry, semantics}      {gate,lane,m,destination-set}
         |                                      |
         | exact tile-to-term transduction      |
         +--------------------------------------+
```

### 4.1 Tile 解决什么

- Q/K 从哪里取；
- 哪些 payload 不需要取；
- temporal/topology anchor 如何驻留；
- 当前 window/destination 属于哪个 stage/head；
- w9/w15 如何分段；
- 何时可以 retire。

### 4.2 Term 解决什么

- 哪些 destination 共享同一个 product；
- multiplicity 是否为 1 还是 1..5；
- product 计算一次后如何 multicast；
- 三个 bank 如何独立消费；
- whole-term 如何原子提交和有序退休。

### 4.3 建议的统一架构

```text
                     Frame/Stage Scheduler
                              |
                  Resolution-Elastic RES-Tile
                    /                         \
            Motion TTB                    Local5 STT
        temporal-pair residency       3-row topology residency
                    \                         /
                     Semantic-Anchor TARE
                    /                         \
             SCS-Shiftmax                 Shiftmax5
           class occupancy              per-dest degree
                    \                         /
             Exact Tile-to-Term Transducer
                    /                         \
           SET: m=1                     MPET: m=1..5
                    \                         /
          segmented term-stationary ordered fabric
                              |
              bank-local product reuse and Acc
```

---

## 5. Crop 和全分辨率到底影响什么

### 5.1 三个明确配置

| 配置 | 输入 | attention window | 角色 |
|---|---:|---:|---|
| crop-w9 | 288x384 | T2x9x9=162 | H67/H66d 现有 full30/valid825 训练口径 |
| full-w9 | 480x640 | T2x9x9=162 | 叶核兼容、窗口数增加的硬件一致方案 |
| full-w15 | 480x640 | T2x15x15=450 | 当前软件队列中的论文几何方案 |

可复跑账本：

```bash
python3 scripts/resolution_tile_term_ledger.py
```

产物：

```text
results/resolution_tile_term_ledger_20260728/ledger.md
results/resolution_tile_term_ledger_20260728/ledger.json
```

### 5.2 关键数字

| 配置 | rows/frame | scheduled token slots/frame | 相对 crop |
|---|---:|---:|---:|
| crop-w9 | 6,720 | 1,088,640 | 1.0000x |
| full-w9 | 19,980 | 3,236,760 | 2.9732x |
| full-w15 | 6,720 | 3,024,000 | 2.7778x |

`full-w15` 与 `crop-w9` 的 row 数相同，是因为四级 feature map 与窗口尺寸
按 `480/288 = 15/9` 同比扩展；但每 row 的 scheduled token slot 从 162
增到 450。该比值不是实际有效 token、term、周期或能耗。

### 5.3 对 Motion 的影响

| per-row 结构 | w9 | w15 | 变化 |
|---|---:|---:|---:|
| token bitmap | 21 B | 57 B | 2.71x |
| 单个 Q 或 K tile/head | 648 B | 1,800 B | 2.78x |
| Q7 score 物化 | 162 B | 450 B | 2.78x |
| Q1.7 gate 物化 | 183 B | 507 B | 2.77x |
| 35-class SCS histogram | 35 B | 40 B | 1.14x |
| TTB4 bundle 数 | 41 | 113 | 2.76x |

上表只是理想 bit-packed logical lower bound，不含 SRAM bank 对齐、端口复制、
ECC 和 macro 粒度，不能当作 SRAM 面积。它提示 w15 下值得验证：

```text
per-token logical state 随 scheduled slot 数增长，
class-closed histogram 的 logical state 基本不变。
```

但现有 Motion RTL 还有以下 w15 阻塞项：

1. 8-bit token ID 不能表示 0..449，必须至少 9 bit；
2. FADC24/RAW41 等模块中存在硬编码 162-bit/162-token 路径；
3. NMF destination bitmap、iterator、Acc 深度需参数化到 450；
4. SCS occupancy count 从 8 bit 增到 9 bit；
5. 不能把 450-bit bitmap 作为长组合广播，必须 segmented stream。

### 5.4 对 Local5 的影响

Local5 的重要性质不随 w9/w15 改变：

- 每个 destination 的有效候选仍为 3/4/5；
- multiplicity 上界仍为 5；
- Shiftmax 每次仍只处理最多 5 个 score；
- MPET 仍只需要 3-bit multiplicity。

变化的是：

| 结构 | w9 | w15 |
|---|---:|---:|
| destination/window | 162 | 450 |
| candidate slots/window | 810 | 2,250 |
| valid edges/window | 738 | 2,130 |
| boundary-invalid 比例 | 8.89% | 5.33% |
| 三行 K 驻留/head | 216 B | 360 B |
| destination ID | 8 bit 足够 | 至少 9 bit |
| 每窗 MFEP/MPET 总 work | 较小 | 约随 destination 数增长 |

这说明 Local5 的叶算子复杂度比 Motion 更不敏感于 window token 数，但系统吞吐
仍会按全帧 destination 数增长。并且 w15 的边界占比更低，degree 和 term
分布也会漂移，所以不能说分辨率“不影响”，也不能直接外推 crop profile。

---

## 6. 硬件到底按 crop 还是全分辨率做

### 6.1 正确边界

```text
叶核综合：按 tile/window 参数点
系统评估：按 480x640 全帧
训练 crop：只决定现有 checkpoint/profile 证据来源
```

不应制造一个 480x640 全展开阵列。芯片应复用固定数量的 lane、SRAM bank 和
term executor，逐 stage/window/head 调度。

### 6.2 推荐双参数点

1. **兼容点 `w9-162`**
   - 保持现有 H67/Local5 crop checkpoint 和大量 RTL 回归；
   - 用于早期功能等价、公平架构消融；
   - 可评估 full-w9 的整帧 19,980 rows。
2. **目标点 `w15-450`**
   - 对齐正在训练的 DSEC paper full-resolution 几何；
   - token ID、segmented bitmap、SRAM 深度按它设计；
   - full-resolution 模型精度通过后作为 DC/STA/SAIF 主点。

若最后 full-w15 的软件精度不成立而 full-w9 成为部署主线，DC 主表再切回 w9；
但接口和 buffer 不应继续假设 8-bit/162-token。

### 6.3 论文结果口径

主表必须报告：

- 480x640 frame latency / FPS；
- 四 stage 周期分解；
- full-frame SRAM read/write bytes；
- mean/p95/p99 term、stall、overflow；
- w9 与 w15 两点的面积/频率/功耗敏感性；
- crop profile 与 full-resolution profile 的分布漂移。

以下表述不允许：

- “162-token row 周期 = frame 周期”；
- “crop valid825 能耗 = 480x640 芯片能耗”；
- “Local5 每 token 5 候选，所以分辨率不影响”；
- “RTL 参数可改，所以已经支持 w15”。

---

## 7. DATE 目标贡献候选与当前状态

建议把 `docs/164` 的四条重构为以下四条，避免 Motion/Local5 各说各话：

### C1：Semantic-Anchor Exact Residual Execution `[CURRENT leaf RTL]`

Motion 用 temporal peer，Local5 用 self-K/topology anchor；统一
ZERO/LIST4/REPLAY exact residual engine。

### C2：Resolution-Elastic Semantic Tile Orchestration `[PROPOSED]`

TTB/STT 不是打包格式，而是跨 score、normalization、term commit 生命周期的
驻留执行 tile；支持 w9/w15 segmented payload 和 metadata-first fetch gating。

### C3：Exact Set/Multiset Tile-to-Term Transduction `[PROPOSED]`

Motion 用 SCS class closure 生成 set term；Local5 用 5-way normalization 和
MPET 生成 bounded-multiset plane；两者都不物化 attention/gated-K tensor。

### C4：Term-Stationary Polymorphic Projection Fabric `[PROPOSED]`

共享窄 term 前端、segmented destination set、whole-term validation、三 bank
独立消费和 ordered retirement；Motion 为 `m=1`，Local5 为 `m=1..5`。

四条的层次分别是：

```text
C1 算术执行
C2 输入驻留/调度
C3 中间表示和数据流重塑
C4 输出互连/投影
```

在 C2-C4 没有完成同 trace RTL/PPA 消融前，投稿应将它们合并为一条
`proposed tile-to-term system dataflow`，不能作为三条已完成贡献计数。

---

## 8. 下一阶段最小实现清单

### P0：先冻结接口

1. 统一 tile descriptor：
   `mode/stage/block/head/window/time_group/window_size/token_count/segment_count`；
2. 统一 term：
   `{gate,lane,multiplicity,destination-segment,first,last}`；
3. 全部 token/destination ID 升到参数化 9 bit 能力；
4. 明确 w9/w15 的 SRAM macro 和 segment 宽度。

### P1：Local5 term 真正接入共享后端

1. 新增 MPET builder，按 `(gate,lane,m)` 聚合 destination；
2. DCTF payload 加 multiplicity，禁止主路径 explode；
3. bank executor 共享一次 `m*gate*W` product；
4. 与 dense five-edge reference 做 bit-exact。

### P1：把 TTB/STT 从薄 descriptor 变成执行 tile

1. Motion TTB4 接 payload fetch enable 和 TARE/SCS phase；
2. Local5 STT 接 line-buffer fetch、TARE、Shiftmax5、MPET phase；
3. tile 只有 term 全部 commit 后才能 retire；
4. 加 backpressure、zero-term、flush、malformed descriptor 回归。

### P2：全分辨率证据

1. full-w15 软件训练完成后跑 Motion/Local5 true-mask ordered profile；
2. w9/w15 各跑多 sample/window mean/p95/p99；
3. 统一 Central/term materialized/RES-Tile+term 三基线；
4. 同 SDC、同 SRAM macro 规则做 DC/STA/SAIF；
5. 按 480x640 全帧累计 FPS、energy/frame 和 EDP。

---

## 9. 当前判断

用户的两个直觉都成立：

1. **Term 可以且应该用于 Local5。**
   真正统一形式不是忽略重数，而是让 Motion set 成为 Local5
   multiplicity-plane multiset 的 `m=1` 特例。
2. **TTB/STT 可以单列，但必须改造成执行架构。**
   最值得做的不是再写一个 descriptor，而是让 tile 从 payload 驻留一路活到
   term commit，并在 full-w15 下验证约 315-bit 逻辑 class histogram
   相对逐 token score/gate 逻辑物化下界的优势；这还不是物理 SRAM/PPA 结果。

这会把当前“若干工程模块”收束成一条可审稿的系统故事：

> 面向全分辨率 all-binary 事件光流，以 resolution-elastic semantic tile
> 管理输入驻留，以 exact set/multiset term 重塑 normalization-to-projection
> 数据流，并由共享 term-stationary fabric 执行两类 attention。
