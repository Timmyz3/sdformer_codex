# Motion 与 Local5 论文机制本土化、架构主线及 DATE 缺口

**日期**：2026-07-26
**范围**：H67 Motion 与 H66d Local5 双线；外部论文机制、开源评估器、
架构本土化、idea 有效性和 DATE 2027 硬件缺口。
**机器可读筛选结果**：
`results/dual_line_arch_idea_screen_20260726/idea_screen.json`。

## 0. 直接结论

### 0.1 两套是否都已经包含七类论文机制

**不是。**

“机制对当前 workload 适用”“文档提出过候选”和“已经进入 RTL”是三个不同
状态。当前真实状态如下。

| 来源 | Motion | Local5 | 能否写成已实现 |
|---|---|---|---|
| Prosperity exact/product reuse | TARE 时间 anchor 已实现 | TARE 单 edge 拓扑 anchor 已实现 | 部分可以，必须写 inspired/adapted |
| Bishop TTB/stratifier | ZERO/SPARSE/DENSE 已实现；TTB 仅 profile/叶模块 | 同一 classifier 已实现；STT 只有方案 | 只能写分层已实现，不能写 Bishop 架构已实现 |
| FireFly-T multi-nonzero decoder | 32-to-4 update extractor 已实现 | 同一 extractor 已实现 | 只能作为 TARE 微结构 |
| FLAT operation fusion/驻留 | SCS/NMF/DCTF 子系统体现原则 | line-buffer/MFEP 尚未闭合 | Motion 部分，Local5 候选 |
| LoAS temporal dataflow/压缩 | T=2 连续组织，未实现 FTP/inner-join | source-stationary 仅方案 | 只能写数据组织借鉴 |
| SpAtten cascade issue | exact pair 级条件已有 profile/局部路径 | exact-K bypass 已进 TARE | 没有独立 cascade scheduler |
| Phi pattern+residual | 未采用 codebook | 未采用 codebook | 不能列为已实现 |

因此，目前两条线共同真正实现的是：

```text
Prosperity-inspired static anchor
  + Bishop-inspired exact density routing
  + FireFly-T-inspired 4-lane extraction
  -> 单实例 TARE-4
```

FLAT/LoAS/SpAtten 的思想在 Motion 后端部分出现，但 Local5 还没有形成完整
硬件。Phi 当前只是基线和新格式设计的灵感，不是已经复用的贡献。

### 0.2 当前主线

**当前投稿实现主线仍应是 H67 Motion，Local5 是有条件挑战者。**

Motion 不是理论上限更高，而是证据链更完整：

- 算法和部署结果已冻结；
- profile100、ordered DSE 和小样本真实位级 trace 已存在；
- TARE、SCS、NMF、DCTF 均已有独立 RTL；
- Local5 仍缺 mask/RNE/Shiftmax5 的统一数值合同、post-G0 profile、
  row-context 和 MFEP RTL。

Local5 提供了更高的**局部复用比例**：

- fixed stencil 提供免费、确定的 self anchor；
- source-resident K 读取理论减少 `78.0488%`；
- pre-G0 MFEP term-count 候选减少 `92.7098%`；
- 规则三行缓冲比 Motion 的全局 162-token 行更容易形成明确存储层次。

但局部压缩比例不能等同于绝对 PPA 上限。当前 pre-G0 绝对工作量为：

| 绝对工作量 | Motion | Local5 pre-G0 | Local5/Motion |
|---|---:|---:|---:|
| active-K 读取/项 | `36,507,347` | `41,221,765` | `1.129x` |
| projection term | `7,101,034` | `13,732,741` | `1.934x` |
| 原始 token/valid edge | `108,864,000` | `495,936,000` | `4.5556x` |

所以当前只能说 Local5 的拓扑复用机会更强，不能说其面积、功耗或系统上限已经
优于 Motion。

正确策略是：

```text
Motion：关闭统一顶层、fallback、SRAM 和 PPA，作为近期投稿主线
Local5：关闭 G0/G1、post-G0 profile 和 row-context，满足门槛再切线
```

## 1. 数据驱动的共同底座

双线 profile 派生得到：

| 指标 | Motion | Local5 |
|---|---:|---:|
| ZERO/exact 或 LIST4 总覆盖 | `94.0162%` | `94.6010%`，pre-G0 |
| 变化项中 LIST8 覆盖 | `95.3225%` | `87.2761%`，pre-G0 |
| delta lane density | `2.5029%` | `1.8916%`，pre-G0 |

这支持将 TARE 从“两个 wrapper 共用叶模块”提升为共同 score substrate：

```text
semantic anchor
    |
    +-- ZERO/exact -----------+
    +-- LIST4/LIST8 residual -+--> result select --> one RNE
    +-- BITMAP/direct replay -+
```

现有 `tare4_residual_composite_core.sv` 仍分别实例化 `raw32` 和 `delta4`
reduction，只共享结果选择与 RNE，并没有共享同一棵 reduction tree。真正的
多格式共享 reduction substrate 是下一版 RTL 目标，不是当前已实现事实。

本地化区别不是换 mode 名称，而是 anchor 和执行上下文不同：

```text
Motion:
  {Q0,K0} anchor -> {Q1,K1} target
  + Motion-XOR bias
  + T0/T1 atomic score packet

Local5:
  {Qself,Kself} anchor
  -> N/S/E/W probes
  + carried raw16/remainder
  + degree=3/4/5 row packet
```

## 2. 建议的新架构：Semantic-Anchor Tileflow

暂用工作名 **Semantic-Anchor Tileflow，SATF**。它是目标架构，不是当前
已完成加速器。名称不是贡献，贡献必须由下面可实现的状态、存储、端口和
数据流支撑。

### 2.1 总体结构

```text
TAB / STT descriptor frontend
           |
           v
multi-format static-anchor TARE
  ZERO / LIST4 / LIST8 / BITMAP / DIRECT
           |
           +-----------------------------+
           |                             |
           v                             v
Motion pair-SCS                    Local Shiftmax5
           |                             |
           v                             v
NMF gate-term encoder             MFEP multiset-term encoder
           +--------------+--------------+
                          |
                          v
             DCTF bank-local projection
                          |
                          v
                     Acc / final
```

该结构把外部工作本土化为三个架构层。

### 2.2 C1：Network-Semantic Anchor Execution

来源：

- Prosperity 的 exact residual/product reuse；
- Bishop 的 density stratification；
- FireFly-T 的 multi-lane extraction；
- Phi 的 pattern+residual 两级表示。

本项目改造：

1. 不用 TCAM、k-means codebook 或在线相似度搜索；
2. anchor 由网络语义免费给出：Motion 时间 peer 或 Local5 self stencil；
3. Phi 的 Level-1 改为网络语义给出的静态 anchor；anchor 本身不等于
   target，只有 `anchor + 完整 residual` 才与 direct target 零误差；
4. Level-2 residual 按
   `ZERO/LIST4/LIST8/BITMAP32/DIRECT` 选择格式；
5. 目标是 sparse 和 dense 不复制两个 core，而是复用同一个 reduction
   substrate；当前 RTL 尚未完成该物理共享。

可辩护的新点不是“提出 residual reuse”，而是：

> 利用事件光流网络中的确定时间/拓扑关系，消除通用 product-reuse
> 架构的在线关系发现成本，并在同一精确执行底座上做多格式残差路由。

### 2.3 C2：Semantic Bundle as an Evolving Execution Contract

来源：

- Bishop TTB；
- SpAtten cascade issue；
- LoAS 时间维内层组织。

Motion 的 **Temporal-Affinity Bundle，TAB**：

```text
token span / stage / block / head
pair empty / motion zero / update count
K-zero class / class multiplicity
payload format / fallback pointer
```

Local5 的 **Stencil-Time Tile，STT**：

```text
time / row / x span / halo
corner-edge-interior valid mask
self anchor occupancy
N/S/E/W delta count and format
source-K bitmap / row-end marker
```

目标本土化：

- descriptor 不只负责 skip；
- 同一 descriptor 从 score issue 演化到 normalization commit，再演化到
  projection term；
- 所有 skip 都是 exact skip，K-zero 的 Shiftmax 分母、多重边和 fallback
  语义不能丢。

它能否成为架构贡献，取决于是否证明：

```text
metadata overhead + decode + FIFO
<
payload SRAM/NoC + arithmetic + control 节省
```

当前 Motion 有 `TTB4 empty=61.0828%` 的动机；Local5 STT 尚无 post-G0
ordered profile，因此 STT 还不能写成实测贡献。当前也没有 descriptor
lifecycle FSM、依赖/版本合同和跨 SCS/NMF/DCTF 的流控实现；在这些 RTL
出现前，C2 只能作为设计方向，不能列为论文已实现贡献。

### 2.4 C3：Normalization-to-Term, Materialization-Free Projection

来源：

- FLAT 的 attention operation fusion；
- LoAS 的流式压缩/时间组织；
- Prosperity 的 product reuse。

Motion：

```text
active {gate,K,token}
 -> (gate,lane,destination bitmap)
 -> one gate product, many destinations
```

Local5：

```text
directed {source,destination,gate,K}
 -> (source,gate,lane,multiplicity,destination)
 -> one source product, multiplicity-preserving multicast
```

Local5 不能使用普通 set-OR，因为同一 destination 的重复边是多重集。MFEP
必须携带 `multiplicity=1..5` 或生成等价重复 commit。

当前 Motion 的 term-count 候选减少为 `82.4926%`；Local5 pre-G0 为
`92.7098%`。Local5 当前 `zero_gate_entries=0`、gate cardinality
`p95=1`，而边界 mask、Shiftmax 缩放和 RNE 合同尚未冻结，所以该数字只
能说明值得 post-G0 复测。这些均是 work-item 机会，不是 cycle/energy 结果。

### 2.5 两条线应采用不同驻留策略

Motion：

- temporal-pair co-resident；
- score/class metadata stationary；
- K 时间源读取复用只有 `9.9922%`，不应主打 K source-stationary。

Local5：

- 三行 K line buffer；
- source K stationary，一次读取向最多五个 destination 多播；
- 重点是 line-buffer/halo/edge-to-term 融合，而不是复刻 Motion 的 NMF。

因此可以共享 TARE 和 DCTF backend，但不能强行共享所有前端存储。

## 3. 七类机制如何在两条线完善

| 机制 | Motion 下一实现 | Local5 下一实现 | 淘汰条件 |
|---|---|---|---|
| Prosperity reuse | T0/T1 atomic packet | ANCHOR_LOAD+2/3/4 PROBE | 加 detector 后 EDP 无净收益 |
| Bishop bundle | TAB4 联合 score/SCS/term | post-G0 STT+halo | metadata 或 FIFO 抵消收益 |
| FireFly decoder | LIST4/8 与 direct fallback | 四方向 residual pack | Fmax/能量不如 bitmap direct |
| FLAT fusion | SCS-NMF-DCTF 单顶层 | Shiftmax5-MFEP-DCTF 单顶层 | SRAM bytes 无显著下降 |
| LoAS organization | T=2 pair 内层 | source-stationary line buffer | bank/halo stall 抵消读取节省 |
| SpAtten cascade | exact L0-L3 pair issue | valid/exact/degree/term issue | mapped SAIF 功耗改善不足 |
| Phi hierarchy | 只做格式基线 | anchor+residual pack 基线 | codebook/matcher 成本高于 residual |

## 4. 开源评估器审计

### 4.1 Prosperity

官方仓库：
<https://github.com/dubcyfor3/Prosperity>

审计 commit：
`6ee1c6f1cb419fcf942f2eda63db84ca28248f4b`

许可证：MIT。

实际包含：

- cycle-accurate Python/CUDA simulator；
- Eyeriss、PTB、SATO、MINT、LoAS 基线；
- `Stats` 风格的 compute/preprocess/memory stall 分账；
- tile M/K DSE；
- CACTI buffer/DRAM 能量接口；
- 激活矩阵和论文参考结果。

可以借用：

1. 每个组件显式返回 cycles、memory reads/writes、stall；
2. preprocessing 与 compute 用 `max()` 建模重叠；
3. initial memory latency 与 steady-state overlap 分离；
4. 相同 simulator 内运行所有基线；
5. DSE、消融和成本收益比同时报告。

不能直接使用：

- CUDA subset kernel针对普通 activation row，不符合 H67/Local5 score 语义；
- `energy.py` 中很多功耗是从论文填入的常数；
- 官方 README 明确说明 DC 面积/功耗脚本未公开；
- 其 500MHz、28nm 和 PE 数不能直接成为本项目结果。

本项目应借其**评估结构**，不借其结果。

### 4.2 Bishop

未检索到官方 Bishop 仓库。

论文方法是：

- 自建 analytic cycle-accurate heterogeneous simulator；
- 三层 memory hierarchy；
- CACTI 7.0；
- DRAMsim3；
- STONNE/SIGMA 模拟 sparse core；
- 商业 28nm RTL 综合。

可使用开源 STONNE：
<https://github.com/stonne-simulator/stonne>

但 STONNE 适合 dense/sparse GEMM 和 projection 基线，不能直接表达：

- K-zero denominator；
- SCS/Shiftmax5；
- TARE static anchor；
- term-atomic commit；
- Local5 multiplicity。

因此主 simulator 仍应是项目自己的 ordered-event simulator，STONNE 只承担
generic sparse projection 对照。

### 4.3 Phi

未检索到官方 Phi 仓库。论文说明：

- simulator 输入 activation 和 calibrated patterns；
- SystemVerilog 实现 preprocessor、L1/L2 processor 和 neuron array；
- commercial 28nm DC；
- CACTI 和 DRAMsim3。

项目可以按论文重建两种公平基线：

1. `codebook pattern + sparse residual`；
2. `multi-row residual pack + conflict-free reconfigurable adder tree`。

但不能写“复现 Phi 官方 simulator”，也不能把 Phi 的 PPA 数字带入本项目。

## 5. Idea 有效性评估口径

### 5.1 五级证据门

| Gate | 目的 | 必须通过 |
|---|---|---|
| A 算法/数值 | 保证执行对象正确 | valid825、定点合同、bit-exact、fallback |
| B workload | 证明机会真实 | ≥100 sample，stage/block 分位数，ordered burst |
| C cycle/traffic | 证明架构有净收益 | detector、FIFO、SRAM、bank、backpressure 全计入 |
| D RTL/PPA | 证明硬件收益 | 同约束 RTL、DC/STA/SAIF、LEC |
| E system | 证明值得发论文 | full-encoder Amdahl、FPS/energy/frame、EDP |

任何 idea 只通过 A/B，均只能写 `[prof]` 或 `[模型]`，不能写成 accelerator
speedup。

### 5.2 公平基线

Score 前端：

```text
Direct32
Direct32x2
zero-only
Prosperity-like online matcher
Phi-like pattern+residual
TARE ZERO/LIST4/DIRECT
TARE multi-format
```

Bundle/调度：

```text
no bundle
TAB/STT same-core
Bishop-like dense+sparse dual core
```

Projection：

```text
materialized gated-K
token-major
source-major
Central96
3xIndependent32
DCTF-2C
DCTF-2C + MFEP/NMF
```

Compactor：

```text
linear priority
segmented prefix
butterfly/Benes
bitmap direct
```

### 5.3 核心指标

- 正确性：raw16/Q7/gate/Acc mismatch；
- 性能：cycles/frame、p50/p95/p99、stall、Fmax；
- 面积：logic、SRAM、buffer、routing proxy；
- 功耗：clock、datapath、SRAM、interconnect、leakage；
- 数据移动：DRAM/SRAM read/write bit、metadata bit；
- 利用率：anchor、residual、direct、projection bank；
- 稳定性：fallback、FIFO max、bank conflict、overflow；
- 系统：FPS、energy/frame、EDP、Amdahl speedup。

建议晋级门槛：

- 子系统 EDP 至少下降 `10%`；
- 面积归一吞吐至少提升 `8%`；
- full-encoder energy/frame 或 throughput 至少改善 `8%`；
- 多 trace p95/p99 不出现严重失衡；
- 所有 fallback 保持 bit-exact。

这些阈值只是内部筛选门，不构成 DATE 自动录用标准。还必须满足：

- 95% 置信区间下界仍为正收益；
- 所有 stage/block 及 p95/p99 不出现不可接受退化；
- 相同 SRAM macro、端口、lane、SDC 和 PVT；
- 报 absolute cycles/bytes/energy，而不是只报 reduction ratio；
- 对稀疏度、SRAM latency、FIFO 深度和频率做敏感性；
- 与 strongest baseline 对比。

当前 `evaluate_dual_line_arch_ideas.py` 已加入绝对工作量和切线
`PASS/FAIL`，但仍只是 profile 机会筛选器，不是 cycle/PPA 排名器。逐样本
bootstrap、完整成本函数和 architecture ranking 必须在 ordered-event
simulator 中完成。

## 6. 主线切换条件

Local5 只有同时满足以下条件才替换 Motion：

1. mask、RNE、Shiftmax5 合同统一；
2. valid825 AEE 仍优于 H67；
3. post-G0 profile100 的 exact-K 不低于 `80%`；
4. source-resident 真实 SRAM 读取收益不低于 `70%`；
5. post-G0 active-K bytes 绝对量不高于 Motion；
6. post-G0 projection terms/cycles 绝对量不高于 Motion；
7. ANCHOR_LOAD/PROBE、Shiftmax5、MFEP、DCTF 完整 bit-exact；
8. 同约束 EDP 至少优于 Motion `10%`；
9. full-encoder Amdahl 后仍有至少 `8%` 系统收益。

当前切线结果为 **FAIL**。此外，Local5 当前软件、Python 参考和 SV 的边界
合同仍不一致：软件/Python 给最低分无效边非零 gate 并乘 clamp 后 K；SV
把无效边输出 gate 强制为 0；真正 masked Shiftmax 应同时移出分母并令输出
为 0。SV 还存在 gate 缩放 `x2` 和 score half-up/RNE 差异。因此现有
`AEE=1.4486` 不是已闭合的 Local5 RTL bit-exact 结果。

在这些问题关闭前，Motion 是主线，不因为 Local5 的相对压缩比例更漂亮而
提前切换。

## 7. 距离 DATE 还缺的硬件工作

### P0：语义和完整顶层

1. 冻结 Motion 或 Local5 投稿配置；
2. Motion 完成 T0/T1 atomic packet；
3. Local5 完成 G0/G1 和 post-G0 trace；
4. 完成 `score -> normalization -> term -> projection -> Acc` 单顶层；
5. 接通 NMF/MFEP overflow、dense replay 和 malformed fallback。

### P1：架构模拟和公平实验

1. 建立 Prosperity 风格 component/event simulator；
2. 多 sample/window 的 ordered replay；
3. SRAM latency、bank conflict、反压、跨 tile 生命周期；
4. 所有公平基线；
5. mean/p50/p95/p99、置信区间和负结果。

### P2：物理证据

当前环境只有 Icarus、Verilator、Yosys 和 Nangate45/OpenRoad 代理，没有：

- `dc_shell`；
- PrimeTime；
- 商业 PDK/标准单元 `.db`；
- 可签核 SRAM macro；
- 商业功耗/布线环境。

因此最终 DATE PPA 仍需补：

1. 同一 SDC/PVT 下 DC 面积和 Fmax；
2. RTL/gate SAIF 动态功耗；
3. CACTI 或真实 SRAM compiler 数据；
4. energy/frame、EDP 和面积归一吞吐；
5. post-synthesis LEC，最好补一次布局后时序/功耗检查。

开放 Nangate45 只能用于结构淘汰，不能代替论文主 PPA。

### P3：验证和系统闭环

1. 多窗口、多 block、全部 stage 真实 trace；
2. overflow、flush、malformed、长反压、epoch 回绕；
3. functional/code/assertion coverage；
4. full-encoder Amdahl；
5. ATLIF、skip、projection 和 attention 的系统周期/能量分账；
6. 至少一个端到端 FPS 或 energy/frame 结果。

## 8. DATE 2027 时间约束

DATE 2027 官方 final paper 截止为 **2026-09-20 AoE**。从 2026-07-26
计算，留给硬件闭环的时间不足两个月。

建议关键路径：

```text
第1周：冻结 Motion 投稿主线；Local5 G0/G1 并行
第2周：ordered-event simulator + 公平基线
第3周：Motion 单一顶层 + overflow fallback
第4周：多 trace/SRAM/backpressure 回归
第5周：DC/STA/SAIF/CACTI
第6周：PPA/EDP/Amdahl 主表，决定 Local5 是否切线
第7周：论文图表、related work、独立审稿
第8周：仅修问题，不再扩机制
```

如果第 3 周仍没有完整 Motion 顶层，或第 5 周仍没有目标工艺环境，应主动
降低投稿范围为可完整证明的 attention-to-projection subsystem，而不是继续
增加未实现的命名。

## 9. 本轮新增证据

- `scripts/evaluate_dual_line_arch_ideas.py`
- `scripts/test_evaluate_dual_line_arch_ideas.py`
- `results/dual_line_arch_idea_screen_20260726/idea_screen.json`
- `results/dual_line_arch_idea_screen_20260726/idea_screen.md`

脚本测试结果：`2/2 PASS`。

## 10. 独立 DATE 复审

独立审稿 agent 在修订前给出：

| 项目 | 评分 |
|---|---:|
| 问题动机 | `4.0/5` |
| 架构新颖性 | `2.5/5` |
| RTL 可信度 | `3.5/5` |
| 实验完整度 | `2.0/5` |
| 系统完整度 | `2.0/5` |
| 综合 | **`2.5/5，Weak Reject / Borderline Reject`** |

本轮已修正其指出的四处关键问题：

1. 不再用相对压缩比例宣称 Local5 架构/PPA 上限更高；
2. 不再宣称当前 TARE 已共享同一 reduction tree；
3. 将精确性严格表述为 `anchor + 完整 residual = direct target`；
4. 把 SATF、C2 和 Local5 MFEP 明确降为目标/候选，而非已实现贡献。

复审给出的 Weak Accept 最小闭环仍未完成：

1. `TARE -> SCS -> NMF -> DCTF -> Acc` 单一顶层；
2. overflow/dense/malformed/flush 无损 fallback；
3. Motion 双 score packet；C2 descriptor lifecycle；
4. 多样本、12 block、mean/p50/p95/p99 ordered replay；
5. 同约束 SRAM、DC、STA、SAIF、LEC 和 energy/frame；
6. full-encoder Amdahl 与端到端收益。
