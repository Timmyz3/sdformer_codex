# M167 三模式 shared96 kernel 独立打铁评审 r1

结论：**81/100，`PASS_STANDALONE_SHARED96_KERNEL_BLOCK_AREA_AND_SPEEDUP_HEADLINES_PENDING_NUMERIC_COMPOSITION`，P0/P1/P2 = 5/7/3。**

M167 的核心硬件点是真的：源码只有一组 96 个 signed-INT8 main products，FRONT、BACK、PREFOLD 三种 operand mapping 通过 `issue_mode` 复用；DC postcompile 也只找到 96 个唯一 main multiply operation。FRONT 另有 32 个 square lanes，并没有从账本里消失。两次 VCS（sealed seed1 和本评审 fresh seed9173）都通过 361 个完整 scoreboard transaction、反压、358 次同拍 result replace、一个 fail-closed 协议攻击和全部六项 cover。BACK 同时返回 event bit 与 signed-Q24 threshold amplitude，已在模块接口上修复 M166 丢失 ATLIF 非零幅值的 P0。

但必须挡住两个很诱人的误用：

- **29,489 vs 83,849 µm² 的 64.830306% 不是共享乘法池面积节省。** M167 把 M165/M166 的累加器、moment/count、raw/rank/event FIFO、7704-bit coefficient config、requant、barrier 和 controller 全部外置了；DFF 从两块合计 15,172 降到 2,828（-81.36%），这已经说明面积差主要不是“少一套 96 pool”的受控 A/B。
- **1.7× 不是 measured speedup。** 它只是 `ceil(1600/96) / (ceil(480/96)+ceil(480/96)) = 17/10` 的 projection capacity boundary。只加入最小 stage 每 group 一次的 7-cycle prefold，就变成 1.696043×；rsqrt、reduction、RNE/saturation、barrier、state traffic、fc2、BN2 和 drain 都还未计入。

## 1. 共享池是否真实

| 审计项 | 独立结果 |
|---|---:|
| source main product slots | 96 |
| DC postcompile unique main mult ops | 96 |
| source/DC square sidecar | 32 / 32 |
| main datapath operator groups after DC | 16 |
| legal modes | FRONT / BACK / PREFOLD |

`main_a[0:95]`、`main_b[0:95]` 先由三模式 operand selector 选择，再经过唯一的 `main_product[0:95]` generate。因此不是三套乘法器用注释伪装成共享。

需要收窄“互斥”的含义：当前只证明了**每个 issue 由一个 2-bit mode owner 占用 main pool**。模块不包含 current-batch BN 的真实 phase controller，TB 甚至按 `index % 3` 交替发 FRONT/BACK/PREFOLD。算法上的 `全部 FRONT -> moment barrier -> PREFOLD -> 全部 BACK replay` 是合理的外部依赖，但还没有 epoch/address 协议来实现或验证。下一版只需小型 local epoch wrapper 或 cycle contract，不必做复杂全系统 scheduler。

## 2. VCS/SVA 与 fresh second seed

| 项 | sealed seed1 | fresh seed9173 |
|---|---:|---:|
| accepted issues / results | 361 / 361 | 361 / 361 |
| FRONT / BACK / PREFOLD | 121 / 120 / 120 | 121 / 120 / 120 |
| consecutive II1 hits | 89 | 89 |
| same-cycle replace | 358 | 358 |
| output stall cycles | 76 | 109 |
| BACK amplitude checks | 120 | 120 |
| protocol attacks | 1 | 1 |
| SVA cover hit | 6/6 | 6/6 |
| assertion failure | 0 | 0 |

scoreboard 对 mode/tag、48 个 FRONT projection delta、16 组 sum/sumsq、32 个 BACK event bit、BACK amplitude 和 96 个 PREFOLD product 全量比较。攻击发生前已接受的 pending result 在 fault 后继续保留并 drain；同拍 consume+replace 被大量覆盖。

缺口是 numeric payload：随机值主要在 `[-64,64]`，threshold 是小正数，没有 Q24 rail、INT8 `-128/127` 边界、真实 alpha/offset/factor/scale 或 checkpoint vector。独立整数范围审计确认现有 port 宽度足以容纳“任意 8x8 局部运算”，但这不等于 binary point、prefold requant 或 checkpoint equivalence。

## 3. ATLIF amplitude：局部 P0 已修，fc2 尚未闭环

软件 `OfficialATLIFSurrogate` 返回 `active * threshold`，即二值模式的 `{0,+threshold}`。M167 在每个 accepted BACK issue 上把 `back_threshold` 写进 `back_event_amplitude`，并由 scoreboard 检查 120 次；因此 M166 的“只吐 bit、把非零幅值丢掉”在当前接口处已经修复。

不过进入 fc2 仍需三选一并做 hardware-order miter：

1. 用 bit 选择 `{0,threshold}` 的定点激活，再乘 fc2 weight；
2. 把 threshold 与 activation scale 折进 fc2 input-column weight，证明 accumulator、RNE、saturation、BN2、residual 顺序；
3. 更有潜力的 trick：fc2 直接对 event bit 选择原 INT8 weight 做 source-major 加法，把整个 sn2 module 共享的正 threshold scale 延迟到 current-batch BN2。实数域里 scale 可通过 BN2 moments 搬移，但 epsilon 会变成 `epsilon/theta^2`，定点域仍必须证明，不能直接宣称抵消。

第 3 条值得优先打：它能把 fc2 的“event×weight”乘法变成“event 选择 weight + signed accumulate”，同时保留比逐列 weight prefold 更干净的舍入顺序。

## 4. DC 与 mapped resource 账本

| 指标 | M167 standalone kernel |
|---|---:|
| cell area | **29,489.291809 µm²** |
| cells | **34,692** |
| combinational / sequential | 31,864 / **2,828** |
| logic levels / critical path | 47 / 2.04 ns |
| setup / hold slack | **+0.4987 / +0.0221 ns** |
| critical path | `issue_mode[0] -> event_bits_q_reg[28]` |
| ports / macros | 6,782 / 0 |

全部 2,828 个 DFF 可逐项对齐：

| register payload | DFF |
|---|---:|
| projection deltas | 816 |
| moment sum / sumsq | 144 / 256 |
| event bits / amplitude | 32 / 24 |
| PREFOLD products | 1536 |
| mode/tag/valid/fault | 20 |
| total | **2828** |

`sumsq` 源码声明 17b，但两个 signed8 square 的最大和是 32768，最高第 17 位恒零，DC 每 lane保留 16b，因此 mapped 是 256 而不是 272 DFF。其余类别也完全对齐。这个结果说明 M167 几乎只是一个 wide-port combinational kernel 加 mode-exclusive output registers，不能当作带 storage/controller 的 accelerator PPA。

流程是 Synopsys DC V-2023.12-SP3、TSMC28 HPC+、3 ns request、ideal clock、ZeroWireload、0 macro、flattened logic-only；五类 constraint report 均 clear。没有 CTS、extracted parasitics、SAIF/PTPX、Formality。

## 5. 为什么 64.830306% 只能是 boundary

| 项 | area µm² | cells | DFF |
|---|---:|---:|---:|
| M165 frontend | 39,568.535846 | 42,174 | 5,811 |
| M166 backend | 44,280.053123 | 35,675 | 9,361 |
| naïve independent sum | 83,848.588969 | 77,849 | 15,172 |
| M167 kernel | 29,489.291809 | 34,692 | 2,828 |
| numerical difference | **64.830306%** | 55.436807% | **81.360401%** |

M165 缺失项包括五拍 projection accumulation、16-lane current-batch moment/count、factor/config、两份 raw bank、requant/output FIFO 和 controller。M166 缺失项包括 7704-bit coefficient config、rank FIFO、16-beat event FIFO 和 service controller。两者之间还缺 rank/coefficient epoch SRAM、barrier、rsqrt/coefficient reduction、fc2/BN2/residual。

因此安全表述只有：

> M167 standalone multiplexed arithmetic kernel 在同一 logic-only DC 流程下为 29,489.291809 µm²。相对功能不等价、资源更完整的 M165+M166 standalone 数字之和，数值边界差为 64.830306%；这不是 matched sharing saving、full-FFN area reduction 或 accelerator PPA。

如果论文需要“共享池省多少面积”，必须另综合一份**同 ports、同 output register、同功能，仅把 main pool 从一套改两套**的 duplicate-pool reference；那才是干净 A/B。

## 6. PREFOLD 现在还不是真 coefficient generator

PREFOLD 的 96 路输入是 generic signed8×signed8，输出 raw signed16 products。BACK 却直接要求 signed8 `folded_left` 和 signed24 `folded_bias`。中间尚无：

- population variance、epsilon、rsqrt、alpha、offset；
- `alpha*L` 的 16b→8b RNE/saturation；
- `offset*(L*R*1)` reduction；
- temporal bias、`-center` 的固定点相加顺序；
- coefficient write/address/epoch storage。

这并不否定共享 product pool，而是把 claim 限定为“generic product capacity 可以复用”。下一数值里程碑要利用 rank factor 的尺度不唯一性：对每个 rank 用 dyadic `s_r` 做 `L[:,r]*s_r` 与 `R[r,:]/s_r` 重平衡，目标同时压住 FRONT 的 R 和动态 `alpha*L` 的最大幅值。若 12 个 FFN 中任何一个仍不能 signed8 表示，就必须选择 wider pool 或 lane-serial schedule 并重新综合，不能靠 clip 偷过。

## 7. 1.7× 的正确口径

| path | products/tile | 96-slot capacity cycles |
|---|---:|---:|
| dense T10 temporal | 1600 | 17 lower bound |
| rank3 FRONT | 480 | 5 issued |
| rank3 BACK | 480 | 5 issued |
| rank3 total | 960 | 10 issued |

- product ratio：`1600/960 = 1.666667×`；
- ideal capacity boundary：`17/10 = 1.7×`；
- prefold：每 16-lane group 640 products，即 7 issues；
- H67 最小 group 有 300 spatial positions，只计 prefold 后边界为 `17*300/(10*300+7)=1.696043×`。

prefold 的确很好摊薄，但还没有 rsqrt、state write/read、barrier、fc2、BN2、memory 和 fill/drain；因此 1.7× 只能放 analytical upper-bound 表，不能当 measured operator speedup。

## 8. 下一里程碑裁决

**硬件优先做 M168：binary-event fc2 source-major / K-bank multi-source accumulator。** 这是当前最可能把“1.7× projection 上限”抬成更有 DATE 吸引力 operator 优势的地方，而且符合只做 standalone module + cycle simulator 的策略，无需总体调度器。

建议 M168 合同：

1. 输入直接接 M167 的 16-bit source-event bitmap 与 threshold identity；
2. 以 source-major 顺序读取 fc2 input columns，event=0 完全跳过；
3. 比较 K=1/2/4 active sources per issue，显式处理 source-bank conflict、weight load-to-use 和 destination accumulator bank；
4. weight supply 必须是真实的 K 份 source column/packed word，不能复用一份向量冒充不同 source；
5. 做 direct amplitude、threshold-folded weight、postponed-threshold-through-BN2 三种 hardware-order integer miter；
6. cycle simulator 用相同 SRAM ports 比 dense fc2 MAC 与 sparse event-add，报告 operator cycle，不越级到 network/system；
7. PAFT 后用真实 event density 重跑；当前 profile只能作为 pre-PAFT sensitivity。

并行科学门禁：

- alpha×L range + rank-wise dyadic rebalance；
- rsqrt/alpha/offset 小型 LUT/Newton 或迭代模块（它每 group 一次，先保正确，不应当主 speedup）；
- M29 rank3 factor checkpoint、M87/M162 deployment result 与 hardware-order valid825；
- 最终相同 port/memory 的 dense-vs-rank3+event-fc2 cycle simulation。

机器可读裁决见 `m167_independent_hammer_review.json`；所有 hash、VCS、DC、resource、DFF、area 和 capacity 数字由 `independent_recompute_m167.py` 只读复算到 `fresh_recompute_m167.json`。`run_independent_seed9173.sh` 使用 sealed exact-SHA `simv` 做了第二随机种子商业仿真。本评审只新增本目录文件，未修改 production、contracts 或 `docs/359`。
