# M532｜顶会开源工作的性能 headline 方法审计与 H67 合法包装

日期：2026-08-27  
状态：`PASS_PRIMARY_SOURCE_METHOD_AUDIT__ZERO_NEW_RUNS`  
审计对象：Prosperity、Phi、FireFly-T、Bishop，辅以 ELSA、DeltaCNN 与 CICC'26 光流芯片  
执行边界：只读文献、官方仓库和既有 H67 收据；本轮新启动 HDL/EDA/训练/性能任务均为 0；未修改 `docs/359_DATE终局冻结_20260813.md`。

## 1. 结论先行

顶会论文里“像样的纸面倍率”主要不是靠隐藏条件，而是靠四个可审计的选择：

1. **对完整架构选择清楚且合理的旧系统分母**，同时在同页补 strongest exact baseline；
2. **把局部理论机会、可执行算子周期和端到端系统性能分层**，各自在其合法作用域内给较强数字；
3. **用 waterfall 解释大倍率来自哪些台阶**，最终倍率必须由统一模拟器直接重跑，不将独立模块倍率相乘；
4. **同时报告吞吐、面积、能量、SRAM/DRAM 和精度**，让审稿人看到倍率不是用无限硬件或精度损失换来的。

H67 当前已经有两个可以写得好看但必须限界的数字：

- `M528 = 1.746753x / 1.741232x`：H67 ep35、10 个冻结样本、四层 bottleneck Conv3x3、51.84M source-row 的 exact CPU same-ledger 结果；两个分母分别是 M468 strongest-zero 与 same-coordinate bit。它是**本项目本地机制候选**，但尚非 RTL/PPA/能量/全网 headline。
- `M472 = 2.459487x`：同一 H67 四层 Conv workload 在**官方 Prosperity CPU 周期框架**中 product 相对 bit 的结果。它是 external-method mapping / opportunity，不是 ours RTL，也不是全网倍率。

C2 的 `4.7642x` 只能叫 typed-K8 相对**单端点 K1**的低带宽 throughput scaling；等服务比较必须同时给 K8 对 K1x8 的 `约 1.01--1.04x` 周期，以及待准入的面积、throughput/mm² 和 energy/source。隐藏 K1x8 会让 C2 headline 失去公平性。

当前 Table A 的 admitted system headline 仍为空。Fixed `1.442B` 对候选 `790.9--803.8M` 的 `1.794--1.823x` 只是 decoder/memory 未闭合的分析敏感性，不能提前写成实测系统加速。

这意味着 H67 获得 DATE Accept 级纸面结果的最短路径不是继续寻找可相乘的局部倍率，而是复刻 Prosperity/Phi 的证据结构：在冻结 H67 工作负载上一次性直跑 `Dense96 -> PTB-like -> K1 -> K1x8 -> typed-K8 -> Ours exact -> Ours lossy`，正文 headline 选 `Ours/Dense96` 或 `Ours/PTB-like`，同页公开 `Ours/K1x8`，并用 C1/C2/C3 waterfall 解释最终结果。

## 2. Primary-source 方法审计

### 2.1 总表

| 工作 | headline 与分母 | scope | workload | baseline 与资源公平性 | 聚合 | 性能证据 | 精度处理 |
|---|---|---|---|---|---|---|---|
| [Prosperity, HPCA 2025](https://arxiv.org/html/2503.03379)，[官方仓库](https://github.com/dubcyfor3/Prosperity) | 平均 `7.4x vs PTB`，约 `1.8x vs A100`；同架构 product-vs-bit 平均约 `3.2x` | PTB/SATO/MINT 对 Transformer 只覆盖其支持的 linear layers；A100 项是另一个 end-to-end 分母 | VGG16、ResNet18、Spikformer、SDT、SpikeBERT、SpikingBERT；视觉/NLP 数据集 | 作者统一 simulator；Prosperity/PTB/SATO/MINT 为 128 PE，Eyeriss/Stellar 为 168 PE；28 nm、500 MHz、64 GB/s；不同面积由面积/能效表补充 | 论文称 average，公式未显式写明；官方 16 行独立重算：算术 `7.461107x`、几何 `7.313885x`、总 runtime 比 `6.731836x` | cycle simulator；核心 RTL+DC，CACTI，DRAMsim3；不是流片 | product sparsity exact；无精度损失 |
| [Phi, ISCA 2025](https://arxiv.org/html/2505.10909) | `3.45x` speedup、`4.93x` energy efficiency vs Stellar；理论 pattern 相对 bit 为 `3.2--6.1x`；PAFT 再增约 `1.26x` | 整套 Phi 架构 headline；theoretical、Phi exact、PAFT lossy 分层 | VGG/ResNet、Spikformer/SDT/SpikeBERT/SpikingBERT；CIFAR、DVS、SST、MNLI | 同工艺/频率模拟；m256/k16/n32，L1/L2 各 8x32 SIMD，240 KiB，64 GB/s；跨架构面积不同，另报 area efficiency | 称平均，正文未明确 arithmetic/geomean | profile-driven simulator；关键模块 RTL+DC，CACTI，DRAMsim3；不是流片 | Phi exact 不降精度；PAFT 单独作为轻微有损档 |
| [FireFly-T, 2025](https://arxiv.org/html/2505.12771) | 主要 headline 是 energy/DSP efficiency：相对 FireFly v2、SpikeTA 分别约 `1.39x/2.40x` 和 `4.21x/7.10x`；等总带宽 microbenchmark `3.48x` | FPGA 网络吞吐/效率与一个 matched-resource memory microbenchmark；跨论文表并非同网络同器件 | CIFAR-Net、Spikingformer-4-256、Spikingformer-8-512；CIFAR10/ImageNet | FireFly v2 可做到同 KV260 平台配置；SpikeTA 为 U280 且部分行作用域不同；microbenchmark 固定总 memory bandwidth 和近似资源 | 无统一 suite mean/geomean headline；多为选定 workload/配置比值 | SpinalHDL/Verilog，Vivado 2024.2 implementation；KV260 FPGA；功耗取 implementation estimate，不是流片实测 | 全部 4-bit 量化，表中同时给 accuracy |
| [Bishop, ISCA 2025](https://arxiv.org/html/2505.12281) | 平均 `5.91x` speedup、`6.11x` energy vs PTB；ImageNet-100 hardware-only 先有 `1.39x/1.57x`，再叠 BSA/ECP | 端到端模型 headline、attention-local ECP 和 hardware-only waterfall 分列 | 4 个 pretrained model/dataset：CIFAR10/100、ImageNet-100、DVS-Gesture-128；Speech Command 另列 | Bishop/PTB 同 PE 和近似单 PE 结构；28 nm 500 MHz，面积 `2.96 vs 2.80 mm²`，功耗 `627 vs 606.9 mW` | 称 average，未明确 arithmetic/geomean | analytic cycle model + STONNE，RTL/CACTI，DDR4；不是流片 | BSA/ECP 为协同有损；阈值、精度与局部/全网结果同表，ImageNet-100 约 `0.13 pp` 损失 |
| [ELSA, 2026 preprint](https://arxiv.org/html/2605.20802) | 对 ANT `3.4x/13.6x`、对 PAICORE `2.9x/22.1x` latency/energy；部分机制用 geomean | 4-bit ResNet50 与子机制分层；部分 cross-work 数字由公开 peak 指标估算 | ResNet50 等 | baseline 异构，部分结果为 estimated；论文明确区分 reported/estimated | 子指标有 geomean；总 headline 不统一 | 模拟/综合型证据，当前为 arXiv preprint | early termination 有 `<0.2%` 和 `<3.3%` 两档 accuracy-loss Pareto |
| [DeltaCNN, CVPR 2022](https://arxiv.org/html/2203.03996)，[官方仓库](https://github.com/facebookresearch/DeltaCNN) | end-to-end video inference 加速与 frame-delta sparsity Pareto | GPU 软件端到端视频推理，不是 ASIC | 视频 CNN | 与 GPU dense/sparse software baseline 比较 | 按模型/序列给结果 | CUDA 实现与端到端 GPU timing | threshold 控误差；适合 H67 temporal-delta related work，不可当 ASIC headline 分母 |
| [CICC'26 光流芯片](https://doi.org/10.1109/CICC65509.2026.11509564) | MVSEC 上 operations `0.20x`、energy `0.12x`、latency `0.19x`，同时给 AEE `+0.03`；另报 peak/benchmark/system TOPS/W | 完整光流芯片，但 peak、benchmark、含 EMA system 三种口径分列 | MVSEC indoor1/2/3、outdoor1 | 流片 28 nm；EMA 外存按 `3.7 pJ/bit`、6.4 GB/s 估算，脚注与芯片测量分开 | 四序列值和 mean 均给 | 28 nm silicon；EMA 能量是 model，不是片上实测 | INT8 baseline AEE 与 feature-on AEE逐序列报告 |

### 2.2 Prosperity 的 `7.4x` 为什么大而不造假

Prosperity 的 headline 是一个**架构栈对旧系统**的结果，而不是一个 product matcher 对 strongest product-capable baseline 的结果。论文消融给出：

| 台阶 | 论文口径 | 含义 |
|---|---:|---|
| PTB -> unstructured bit execution | `2.28x` | 从较粗结构化 skip 到更细 bit sparsity |
| bit -> product sparsity（含高开销 dispatch） | `2.16x` | product relation 提供额外工作复用 |
| high-overhead -> optimized dispatch | `1.49x` | 提高机会到周期的捕获率 |
| 三项叙事乘积 | `7.34x` | 解释摘要 `7.4x` 的来源；不是不同 workload 平均值的数学恒等式 |

此外论文还同时给约 `14.2x vs Eyeriss`、`4.8x vs SATO`、`3.6x vs MINT`、`2.1x vs Stellar` 和约 `1.8x vs A100`。这些是**分母阶梯**：每个数字可以合法存在，但正文必须写清 baseline、作用域和证据类型。

Prosperity 的真正方法学价值有三点：

- 所有可模拟 baseline 尽量放进统一 simulator；
- 理论 product density、product-vs-bit 周期和最终 architecture-vs-PTB 分开；
- 用 RTL/DC、CACTI、DRAMsim3 给周期模型定价，而不是只报 nonzero-count。

H67 可以照搬这套评测结构，不能将外部 product mechanism 改名成自己的 novelty。

### 2.3 Phi 的 `3.45x` 为什么不是 PAFT 一招得到

Phi 的 `3.45x` 是完整架构相对 Stellar，而 pattern 理论机会与 PAFT 都有自己的作用域：

- pattern 相对 bit 的 `3.2--6.1x` 是理论工作量/潜力；
- exact Phi 负责硬件结构与无损执行；
- PAFT 只是在 exact Phi 之上追加约 `1.26x` runtime 和约 `1.10x` energy，且属于 lossy co-design。

因此 H67 的有损档也应写成 `Ours-exact -> Ours-lossy` 的增量 Pareto，而不是用 PAFT/近似稀疏去改写 exact baseline 的大 headline。只有 H67 自己的 valid 协议、checkpoint 身份和完整周期重跑闭合后，有损点才能进入主表。

### 2.4 FireFly-T 对 C2 最有用的不是跨论文 FPS

FireFly-T 最值得 H67 学的是同平台、同总 memory bandwidth 的 one-bank/crossbar microbenchmark：约 `3.48x` 的结果把“带宽相同”明确写入实验，而不是拿更多 bank 偷换资源。其跨论文主表还并列 device、频率、LUT/BRAM/URAM/DSP、精度、FPS、GOP/s、GOP/s/W 和 DSP efficiency。

映射到 C2，正确问题不是 K8 比单 K1 快多少，而是：

> 在提供八路 source service 的相同能力下，typed-K8 相对 replicated K1x8 能否以近似周期减少共享 scoreboard、partial-state、completion 与 tag movement 的面积/能量？

所以 C2 需要 iso-lane 和 iso-service 双表，而不是删除其中一张表。

### 2.5 Bishop 对 H67 有损稀疏的边界

Bishop 是最接近“有界 pruning + 强 headline”的参照。它没有把 attention-local ECP 的数字直接当系统数字，而是同时给：

1. 同资源附近 hardware-only 的 `1.39x`；
2. BSA/ECP 后的端到端 `5.91x` suite average；
3. attention-local 仅保留约 `15.5%` 计算、约 `43.92%` latency reduction；
4. 对应精度/阈值。

H67 的 RQTB/attention 份额只有约 `0.59%`，所以即使在 attention 上复制 Bishop 式大局部倍率，全网上限也约 `1.006x`。有损 headline 必须打到 Conv/FC/ATLIF/decoder 的主要工作份额，并用 `S_system = 1 / ((1-f) + f/S_local)` 明示系统敏感性。

## 3. 合法的“paper trick”清单

这里的 trick 指可复核的实验组织，不包括藏引用、换名冒充新机制或不公平造数。

### 3.1 可以做

1. **Denominator ladder**：摘要选 Ours/Dense96 或 Ours/PTB-like；正文同页补 Ours/K1、Ours/K1x8 和 Ours/external mapping。
2. **Iso-lane 与 iso-service 双表**：iso-lane 展示同 lane budget 的系统收益；iso-service 展示相同服务能力下的 area/energy 优势。两者回答不同问题。
3. **Official artifact replay**：在官方 Prosperity simulator 中重放冻结 H67 workload，明确写“external method on our workload”。
4. **Support-tile aggregation 脚注**：若官方实现按 N tile 重复，允许严格代数展开，但必须写出 tile 数、同构条件、memory equation 和 direct validation。
5. **Capture-gap 图**：theoretical opportunity -> official artifact -> unbounded oracle -> resource-bounded candidate -> RTL/system。它能把负结果转成设计洞见。
6. **Oracle ceiling**：M473 `389,974,420` cycle 可以显示理想并发上界，但不得作为 M528 的准入分母或实测 headline。
7. **Waterfall**：逐级开启 C1/C2/C3，每一级由同一 simulator 直接重跑。最终行与 B0/B1 的比值才是累计 headline。
8. **Amdahl/system sensitivity**：局部强结果旁边给工作份额和系统上限，防止审稿人替作者计算后推翻结论。
9. **多聚合并列**：arithmetic mean、geomean、ratio-of-summed-runtimes、min/max 全报；摘要采用事先冻结的默认值，不能事后挑最大者。
10. **负结果消融**：空 tile、lazy-PWP、payload residency 等负结果可以形成“为何常见稀疏技巧在事件光流失效”的消融栏，提升 soundness。
11. **固定分子 effective throughput**：所有候选使用同一个 dense-equivalent workload 或同一原始 useful work 作为分子；executed additions 另列。
12. **证据等级标签**：每个数字标 `[silicon] / [FPGA-impl] / [RTL+DC] / [cycle-sim] / [official-artifact] / [analysis]`。

### 3.2 绝对禁止

- 把 Prosperity `7.4x`、Phi `3.45x` 或 M472 `2.459487x` 写成 H67 自研 RTL 的加速；
- 把 C1、C2、C3 的局部倍率直接相乘；
- 把 M528 `1.74x` 写成 full-network、all-Conv、RTL、PPA 或 energy 结果；
- 只报 C2 K8/K1 `4.7642x` 而隐藏 K8/K1x8 的等服务对照；
- 把 K1x8/K1 的 `4.89--6.32x` 称为“稀疏收益”；它主要是八端点并行扩展；
- 把 Fixed/candidate 分析 envelope `1.794--1.823x` 提前写入 admitted Table A；
- 把 theoretical nonzero/product reduction 当 cycle speedup；
- 把 logical byte reduction、selected-slice mW 或 CICC 的 `3.7 pJ/bit` 敏感性当作本芯片 energy/frame；
- 混用 exact checkpoint 的周期和 PAFT/lossy checkpoint 的精度；
- 在跨论文异构网络/器件上用 raw FPS 排名并声称 apples-to-apples；
- 未引用原工作而仅更名机制，或宣称 `first` 而无系统性文献证据。

## 4. H67 现有结果如何进入 Table A/B/C

### 4.1 Table A｜同一 H67 workload 的 admitted system headline

当前可正式填入 admitted 数值的行：**无**。

应预留的列：

| Config | Exact/lossy | Scope | Fairness | Cycle/frame | vs Dense96 | vs PTB-like | vs K1x8 | FPS | Energy/frame | Area | SRAM | DRAM bytes | Eff. GOP/s | GOP/s/mm² | Evidence |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Dense96 Fixed-T10 | exact | full network | iso-lane | TBD | 1.0 | — | — | TBD | TBD | TBD | 240 KiB | TBD | TBD | TBD | pending |
| PTB-like structured | exact | full network | iso-lane | TBD | TBD | 1.0 | — | TBD | TBD | TBD | 240 KiB | TBD | TBD | TBD | pending |
| K1x8 | exact | full network | iso-service | TBD | TBD | TBD | 1.0 | TBD | TBD | TBD | 240 KiB | TBD | TBD | TBD | pending |
| Ours exact | exact | full network | iso-lane + iso-service companion | TBD | TBD | TBD | TBD | TBD | TBD | TBD | 240 KiB | TBD | TBD | TBD | pending |
| Ours lossy | lossy | full network | same hardware, new checkpoint identity | TBD | TBD | TBD | TBD | TBD | TBD | TBD | 240 KiB | TBD | TBD | TBD | pending |

Fixed `1,442,206,883`、candidate `790,920,000--803,774,000` 和 `1.794--1.823x` 只放正文灰色 analysis box，不填 admitted 单元格。

### 4.2 Table B｜本项目同架构局部/消融

| 结果 | 合法数值 | 两个必须公开的分母/对照 | 当前证据等级 | 可写位置 |
|---|---:|---|---|---|
| M528 single-port product capture | `435,293,339` cycle；`1.746753x` vs M468；`1.741232x` vs same-coordinate bit | M468 strongest-zero `760,350,133`；same-coordinate bit `757,946,784` | `[exact CPU cycle recompute]`；四层 Conv/一序列；非 RTL/PPA/system | C1 local candidate / capture-gap；待 RTL+物理闭合后升格 |
| M473 fused concurrent ceiling | `389,974,420` cycle；相对 M468 约 `1.95x` | 只作 oracle；不得作为 M528 失败门或 headline | `[analysis/oracle]` | capture-gap 虚线 |
| M504 all-write -> M528 dead-write | `456,016,645 -> 435,293,339`，增量 `1.047608x` | 同一 single-port ledger | `[exact CPU ablation]` | C1 liveness ablation |
| C2 typed-K8 vs K1 | `429,716,335 -> 90,196,785`，`4.7642x`；旧 logic area `20,436.7 -> 20,587.4 um²` | 单端点低带宽 K1 | `[local frontend cycle + logic]`，非 equal-service/system | C2 scaling 列，分母写进表头 |
| C2 typed-K8 vs K1x8 | directed VCS 中约 `1.01--1.04x` cycle | replicated eight-endpoint K1x8 | `[VCS directed]`；complete FC2/DC/power/headline=false | C2 iso-service 列；最终以 matched DC/PTPX 补 area/energy |
| C2 tag elision | metadata movement 静态上界 `-27.53%` | typed tag-present vs tag-elided | `[static bound]` | protocol 支撑；非 cycle/energy headline |
| C3 Fixed-T10 / rank path | 当前只有 directed VCS 与 logic-only 候选 PPA | matched Fixed-T10、相同 ATLIF identity | 非系统 | C3 microarchitecture，等待公平物理/系统行 |
| RQTB/attention | 全网份额约 `0.59%` | Fixed attention | exact 局部 | completeness/energy；不得做系统 headline |

#### M528 的推荐论文句

> On 51.84M frozen source rows from four H67 bottleneck Conv3x3 layers, the liveness-aware single-port parent path reduces same-ledger cycles from 760.35M to 435.29M, yielding 1.747x over the strongest-zero schedule and 1.741x over the same-coordinate bit-sparse schedule. This CPU cycle result is exact but remains pending RTL/PPA and full-network admission.

句子必须同时保留 scope 与 `pending` 限定。

### 4.3 Table C｜外部工作与 external-method mapping

| 行 | 数字 | 论文标签 | 禁止标签 |
|---|---:|---|---|
| Prosperity 原论文 | avg `7.4x vs PTB`；约 `1.8x vs A100`；product-vs-bit avg约 `3.2x` | original reported, simulator+RTL/DC/CACTI/DRAMsim3 | H67 ours |
| Phi 原论文 | `3.45x vs Stellar`；PAFT extra约 `1.26x` | original reported exact/lossy split | H67 ours |
| FireFly-T 原论文 | efficiency headline；matched-bandwidth microbenchmark `3.48x` | FPGA implementation / estimated power | ASIC system speedup |
| Bishop 原论文 | avg `5.91x vs PTB` | original reported co-designed lossy | H67 exact |
| CICC'26 原论文 | silicon，operations/energy/latency + AEE 表 | silicon reference | 与 H67 simulated 值直接排名 |
| M472 H67 official replay | bit `556,188,432`、product `226,140,006` cycle，`2.459487x` | `[official Prosperity artifact on H67 support tiles]` | ours RTL / same-resource / monolithic Conv / full network |

M472 的必需脚注：

> The official CPU path was executed for one complete 128-output N tile and algebraically expanded across six identical N tiles using the official `run_fc` equations. The one-time transfer and total traffic terms were recomputed, and three direct 768-output checks produced zero mismatch. Results aggregate support tiles rather than a monolithic H67 Conv implementation.

## 5. 聚合、精度与公平性的冻结规则

### 5.1 聚合

对每个 sequence/workload 同时报：

- arithmetic mean of per-workload speedups；
- geometric mean of per-workload speedups；
- ratio of summed baseline cycles to summed candidate cycles；
- min/max；
- event-density 分层的结果与样本数。

默认摘要采用哪一种必须在看结果前写进合同。若 abstract 用 arithmetic mean，必须直说 `arithmetic mean`；不能因为它最大才事后选择。`ratio-of-sums` 是总体 workload population 的总时间比，不等同于 per-workload mean。

### 5.2 固定吞吐分子

- `dense-equivalent effective GOP/s`：所有候选使用同一 Fixed dense OP numerator；
- `useful-nonzero GOP/s`：所有候选使用冻结原始 trace 的同一 useful accumulation numerator；
- `executed additions/cycle`：允许随架构减少，但必须另列，不能冒充 fixed-work throughput；
- 对积累操作统一写明 `1 accumulation = 1 OP` 或 `1 MAC = 2 OP`，全文不可切换。

### 5.3 精度

- exact 行必须引用同一 ep35 checkpoint，bit-exact/0 mismatch；
- lossy 行必须使用独立 checkpoint identity，报 AEE/EPE、per-sequence 变化、阈值和 valid protocol；
- `epsilon=0` 必须回归 exact 硬件子集；
- accuracy 与 cycle 必须来自同一个配置/身份，不交叉拼接；
- CICC 式写法值得复刻：每序列同时给 baseline accuracy、candidate accuracy、degradation、operations、traffic、energy、latency。

### 5.4 资源公平性

每行至少冻结：工艺、频率/时钟约束、lane/PE、precision、accumulator width、SRAM logical/macro-rounded capacity、bank/port、DRAM bandwidth、NoC/issue bandwidth、decoder/operator scope。`iso-lane` 不自动等于 `iso-area`；`iso-service` 需要匹配峰值 source service，并报告 area/throughput/W。

## 6. 可执行但本轮不启动的评测模板

以下是下一次系统评测的 schema，不是本轮运行命令。

```yaml
evaluation_id: h67_date_headline_r1
identity:
  checkpoint: H67_ep35_exact_sha256
  trace_manifest: ordered_full_network_sha256
  accuracy_protocol: dsec_valid_protocol_sha256
resource_manifest:
  process: tsmc28
  clock_ns: 3.0
  signed_source_lanes: 96
  accumulator_bits: 24
  sram_budget_bytes_macro_rounded: 245760
  dram_bandwidth_GBps: 64
  issue_bandwidth_Bpc: 192
baselines:
  - Dense96_FixedT10
  - PTB_like_structured_group_skip
  - exact_bit_sparse_K1
  - exact_bit_sparse_K1x8
  - exact_typed_K8
candidates:
  - Ours_C1_C2_C3_exact
  - Ours_C1_C2_C3_lossy
required_scope:
  operators: all_network_ops_including_decoder_attention_bn_atlif
  sequences: at_least_3_DSEC_sequences
  bins: event_density_quantiles
per_configuration_outputs:
  - cycles_per_frame
  - stall_cycles_by_cause
  - issued_sources
  - retired_destinations
  - sram_read_write_bytes_by_array
  - dram_read_write_bytes
  - logic_sram_dram_energy_per_frame
  - area_mm2
  - fps
  - dense_equivalent_effective_GOPs
  - useful_nonzero_GOPs
  - GOPs_per_mm2
  - exact_mismatch_or_AEE_EPE
aggregation:
  - arithmetic_mean_speedup
  - geometric_mean_speedup
  - ratio_of_summed_cycles
  - minimum_maximum
  - per_sequence_and_density_bin
direct_reruns:
  - B0
  - B1
  - B2
  - B3
  - C2
  - C1_plus_C2
  - C1_plus_C2_plus_C3
  - final_exact
  - final_lossy
headline_gate:
  same_identity: true
  direct_full_network_cycles: true
  decoder_and_memory_closed: true
  same_page_strongest_K1x8_baseline: true
  exact_and_lossy_separate: true
  independent_review_P0_P1_zero: true
```

推荐生成三份机器表：

1. `table_a_system.csv`：每 sequence 和 aggregate 的系统数字；
2. `table_b_waterfall.csv`：同一 simulator 的直接 rerun 消融；
3. `table_c_external.csv`：原论文指标与 evidence label，严禁把外部数字复制进 ours 列。

## 7. 对当前路线的具体裁定

1. **C1/M528 值得继续物理化**：`1.74x` 是目前最有希望的本项目 Conv 局部候选，且分母是可落地 same-ledger baseline，不应再拿做不出来的 M473 双口理想点错杀。下一关是 bounded RTL/VCS、新思 DC/STA、宏口/调度器面积与 memory-inclusive cycle。
2. **M472 保持 external opportunity**：`2.459x` 很有价值，适合动机、capture-gap 和 external mapping，但不得升格成 ours。
3. **C2 用“双表”救价值**：`4.7642x` 保留为低带宽 scaling；真正 DATE claim 应争取 K8 相对 K1x8 的 area/energy/source 优势。`27.53%` metadata movement reduction 不是差结果，缺的是 matched PTPX/area 转化。
4. **C3 只在 Fixed-T10 公平物理基线后进入主贡献**：rank/timestep 的局部理论数字不能代替系统结果。
5. **有损点先做 exact 主线的增量**：借 Bishop/Phi 的写法，优先非 attention 的 Conv/FC/ATLIF/decoder，报 exact -> lossy 的额外收益与 AEE Pareto；attention-only 不能成为系统 headline。
6. **系统 headline 目标可现实设为约 `1.8x`**：如果统一 simulator 相对 Dense96/B1 直接跑出 `>=2x`，可以强 headline；若最终是约 `1.8x`，配合 C1 `1.74x`、C2 throughput/mm²/energy、C3 exact neuron service 与完整 CICC 式表格，仍可达到 DATE Accept 的性能完整度。

## 8. 来源与证据身份

Primary sources：

- Prosperity：[paper](https://arxiv.org/html/2503.03379)，[official repository](https://github.com/dubcyfor3/Prosperity)
- Phi：[paper](https://arxiv.org/html/2505.10909)
- FireFly-T：[paper](https://arxiv.org/html/2505.12771)
- Bishop：[paper](https://arxiv.org/html/2505.12281)
- ELSA：[paper/preprint](https://arxiv.org/html/2605.20802)
- DeltaCNN：[paper](https://arxiv.org/html/2203.03996)，[official repository](https://github.com/facebookresearch/DeltaCNN)
- CICC'26 optical-flow accelerator：[DOI](https://doi.org/10.1109/CICC65509.2026.11509564)，本地作者 PDF：`docs/Zhang 等 - 2026 - A 28-nm Optical Flow Estimation Accelerator with Redundancy Speculation, Bit-Width-Aware Compression.pdf`

审计限制：本轮没有在 primary source 中确认 Phi、FireFly-T、Bishop 的官方可复跑代码仓库，因此只把它们作为 paper-reported 方法学与数字，不宣称已独立 artifact replay。ELSA 是 2026 arXiv preprint，不能写成已正式顶会录用。跨论文不同网络、器件和精度只用于完整度对标，不构成 H67 apples-to-apples 排名。

H67 local evidence：

- `docs/524_DATEAccept当前硬件贡献与机制迁移收口表_20260827.md`
- `docs/526_Prosperity_Phi性能Headline方法审计与H67合法包装_20260827.md`
- `results/m528_h67_single_port_same_ledger_recompute_r4_20260827/m528_h67_single_port_same_ledger_recompute_result_r1.json`
- `results/m472_h67_official_prosperity_iso_workload_r1_20260826/m472_h67_official_prosperity_iso_workload_r1.json`
- `results/m519_fc2_registered_release_k1_vs_k1x8_vcs_r2_20260827/m519_fc2_registered_release_vcs_receipt_r2.json`
- `reviews/m526_prosperity_phi_headline_method_independent_hammer_r2_20260827/`
- `reviews/m527_h67_headline_baseline_ladder_independent_hammer_r3_20260827/`

本轮操作计数：`HDL=0, EDA=0, training=0, performance=0`。只进行了文献/官方仓库阅读、既有 JSON 收据核对与报告封存。
