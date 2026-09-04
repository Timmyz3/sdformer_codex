# TCAS-II Express Brief 独立模拟审稿与 Strong-Accept 最小收口审计 r2

- 审阅日期：2026-09-04（Asia/Shanghai）
- 审阅对象：`paper/tcasii/main.tex`、当前五页 PDF、claim/PDF checker，以及稿件引用的 C1、K8、TSBG 封存证据
- 审阅模式：只读审稿；未修改论文、RTL、实验结果或 `docs/359`，未运行 VCS/simv、EDA、GPU、许可证查询或 Git 操作
- 稿源 SHA256：`523f1f91c3a18d0826d95ec88d0e35fac13178ee36a71796a810532fcd6195c6`
- PDF SHA256：`d42904347470945457f80adbd0cc9a2fb2176dbe8b28281c1b52c02a076768de`
- 冻结 `docs/359` SHA256：`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`

## 技术结论

**两贡献叙事足以构成 TCAS-II Express Brief，不应再增加第三个稀疏机制。** C1 与 C2/TSBG 命中的是两个不同的受限资源：C1 在有限 lifetime 和单 1RW 冲突下复用 product；C2/TSBG 在保留私有 sign、destination、product 和 Acc24 ownership 时复用 weight delivery。二者可由同一个电路原则统一：判定必须在被省资源之前完成，而且被共享的数据和不能共享的完成状态必须显式分离。

按当前可投稿科学内容，我给 **3.8/5，Weak Accept，约 55--70% 外审接收倾向**。证据纪律和 C1 物理锚点明显强于普通架构草稿，但还不是 Strong Accept，原因是：

1. TSBG 的主要价值是少发 SRAM 请求，当前却没有同身份 routed hold 与 logic+SRAM energy；
2. `1.8345x` 只覆盖每层 first/middle/last 三个固定 B4 区域，不是完整 token population；
3. C1 的 `1.6945x` 仍是单序列 cycle model，九宏 PPA 与完整 214,912-B ledger 是分离证据；
4. 当前 PDF 未满足严格 4.5+0.5 页面门，且 author/ORCID/funding 仍为占位符；
5. 正文过度防御，图中文字偏小，弱化了真正的电路贡献。

若 matched post-route hold、同身份 TSBG power/SRAM energy、token 覆盖鲁棒性和最终版面同时闭合，预计可到 **4.25--4.4/5，可信 Accept / Strong-Accept 倾向，约 78--88%**。这是审稿倾向估计，不是接收保证。

TCAS-II 普通投稿**没有 FPGA 或流片硬门槛**。官方要求的是显著的 circuits-and-systems 创新、接近完稿的首次提交和严格五页格式；同时明确提醒，电路稿若不能证明相对 prior 的性能优势（通常伴随 measured results）或实际系统意义，可能不送外审。因此本稿最有效的增量是 matched P&R/PTPX/存储能量，而不是临时转 FPGA。官方依据：[submission guide](https://ieee-cas.org/publication/TCAS-II/tcas-ii-manuscript-submission-guide)、[author guidelines](https://ieee-cas.org/publication/TCAS-II/guidelines-author)。

## 第一性原理审阅：两条贡献是否真的成立

### C1：成立，但 novelty 是“有限单口执行”，不是 product sparsity

Prosperity 已经占据 subset/prefix parent、residual activation 和 dependency reconstruction 的概念位置。因此 C1 能守住的 claim 只能是：在 parent 会失效、容量有限、一个真实 1RW grant 同时承受 read/write、且 completion 必须原子退休时，如何保证 exact product capture 可执行。

当前电路细节足以支撑这一对象差：grant-time liveness recheck、deadline-aware arbiter、reserved response、forwarding、dead-write suppression 和 atomic completion 共同解决了一个具体的单口电路问题。`382.849 M` 周期相对 strongest-zero 的 `648.741 M` 周期给出 `1.694510x`；这不是把 firing sparsity 重命名，因为 same-coordinate bit baseline 只有 `1.003282x`。

残余风险是证据没有在同一物理对象上闭合：51.84 M row 比率来自 software same-ledger replay，真实 VCS 只校准一块 64-row tile；`166,514 um2` 的 mapped island只实际含九个 parent-product SRAM leaf，而不是完整 214,912-B ledger。当前稿已披露此边界，因此不是错误，但 Strong Accept 不能把这些轴合成“RTL 实现达到 1.6945x”。

### C2/TSBG：机制成立，真正的电路意义要由 energy 证明

K8 相对等带宽 K1x8 的 `1.0167x` 周期说明它不是吞吐 headline；其价值是共享 Acc24、endpoint 和控制后，logic area 从 `585,479` 降到 `131,086 um2`，得到 `4.541x` directed-throughput/logic-area。这个比较必须始终把 `1.0167x` 与 `4.541x` 同列，避免把面积共享包装成稀疏周期收益。

TSBG 的合法 specialization 不是 broadcast、bundling 或 group-major order本身，而是：common row identity 在 SRAM request 之前被证明；只共享 weight row delivery，四个 context 的 sign、destination、tag、terminal、product 和 Acc24 state 保持私有；miss 与弱 reuse 时可 exact fallback。M2018 确实实例化 M803 的八 bank typed adapter 和 Acc24，而不是一个脱离 C2 的软件调度器。

问题是目前最关键的“被省资源”只以 request count 出现。全 2,880 fixed-region population 的 128-bit scalar reads 从 `21,087,648` 降到 `8,830,176`（`-58.1263%`），但部署容量仍是同一个 288-KiB weight SRAM。只有在相同容量、相同 PVT、相同 workload 下报告 bank dynamic energy、leakage 和 logic energy，才能证明 pre-read admission 省掉了物理上昂贵的 max term。

### 统一原则：可作为论文主线，不宜冒充第三个算法

可将两条贡献统一为 **resource-preceding exact admission**：

- C1 在省 residual product work 前，证明 exact live parent，并保持 row completion 私有；
- TSBG 在省 weight request 前，证明 common row identity，并保持各 context completion 私有。

这是一个清楚的 circuits design rule，但不应额外列成第三条 contribution，也不应声称 first。它的作用是解释为什么两条机制不是随机拼接，并导出一条可验证的安全条件。

## 证据与计算复核

本次重新执行七个核心目录的 inner/outer SHA 检查，均通过；未将 M2135/M2139 或任何失败/未独立准入的 P&R/power attempt 当成正结果。

| Claim | 独立重算 | 裁决与边界 |
|---|---:|---|
| C1 speedup | `648,741,051 / 382,848,700 = 1.694510262x` | 正确；四层 bottleneck Conv、十个 `zurich_city_09_a` sample、same-ledger cycle model |
| C1 time reduction | `40.985899%` | 正确；可写 `40.99%`，不可称 RTL/system |
| bit baseline | `648,741,051 / 646,619,098 = 1.003281612x` | 正确；证明收益不是普通 same-coordinate zero/bit skip |
| C1 physical anchor | `166,514.312 um2`，setup/hold `+27.871/+1.827 ps` | 正确；九 SRAM leaf、prelayout、ideal clock、ZeroWireload、无 SPEF |
| C1 energy window | `29.0763016 mW x 253 x 3 ns = 22.0689129 nJ` | 正确；mixed-corner directed component window，不是 frame energy |
| K8 cycle | `1,945 / 1,913 = 1.016727653x` | 正确；五个 directed loads、equal-bandwidth K1x8 baseline |
| K8 area efficiency | `(1,945/1,913) x (585,479.154/131,086.241) = 4.541078x` | 正确；logic-only，不能省略周期仅 `1.0167x` |
| TSBG G48 | `12,522,876 / 5,124,365 = 2.443790792x` | 正确；1,920 fixed-region workloads，G48 subset |
| TSBG continuation | `80,129,099 / 45,381,069 = 1.765694391x` | 正确；960 G96/G192 workloads，单 compile/simv batch |
| TSBG all identities | `92,651,975 / 50,505,434 = 1.834495175x` | 正确；2,880 workloads，ratio of sums，非 full FC/network |
| TSBG all reads | `21,087,648 -> 8,830,176 = -58.126312%` | 正确；是 bank activation，不是 SRAM capacity reduction |
| matched logic | `249,710.452 -> 249,739.810 um2 = +0.0117568%` | 正确；prelayout setup-clean，但两轴 hold 均 `-16.4 ps` |
| common-memory area model | `689,593 / 1,143,986 um2`，area `-39.72%`，throughput/area `1.687x` | 算术正确；foundry-QRT common-capacity model，不是 integrated P&R |

### 数据质量与选择风险

- **高置信正项：** 所有 12 FC1 和 12 FC2 layer identity 均被 fixed protocol 覆盖；continuation 的 960 行来自一个干净 single-simv batch；aggregate 算术与逐行结果一致。
- **中等风险：** 每个 layer/sample 只取 first/middle/last 三个 B4 quartet。选择规则是预先固定的，因此不是直接 cherry-pick，但无法证明中间所有 token 的 cache/reuse 分布相同。
- **中等风险：** G48 的 1,920 行仍是同一 executable image 下的 `1,917+3` 跨 attempt composite。证据可引用且稿件已披露，但它占用大量版面并增加审稿疑虑；一个新身份的 clean one-batch rerun 比继续解释 lineage 更划算。
- **中等风险：** TSBG 的 activity 来源是真实 ep34 capture，但 arithmetic weight 是 deterministic directed INT8 verification weight；自然非零 descriptor 为 `+1`，负号由 directed protocol test 覆盖。它证明 exact schedule/control，不是用真实 checkpoint weight 测得的数值分布收益。
- **高置信缺口：** 截至本快照，没有准入的 TSBG P&R、hold-clean、SAIF/PTPX 或 SRAM energy 正结果。

## 当前 PDF 与叙事审阅

当前 PDF 是 Letter、5 页，page 5 right column 只含 references；但 submission checker 报 `page5_left_content_ymax=530.25 pt`，未达到设定的 `650 pt`，因此仍不满足严格的 4.5 页 content 目标。author block 仍是占位符。官方 binary review 鼓励首次提交接近 publication-ready，这两项属于 upload P0。

### 应保留

1. 标题中的 `Finite-Lifetime Single-Port` 与 `Context-Safe`，它们直接给出对象差。
2. C1 exact reconstruction 等式与 TSBG 私有 context 等式。
3. C1 strongest-zero/bit/concurrent ceiling 梯子。
4. `1.0167x cycles + 4.541x throughput/logic-area` 同句的公平 K8 口径。
5. direct-prior table，但应压成最直接的 Prosperity、FireFly-T/ELSA、Phi/WS-LOS 四行。
6. 七个略慢 case、两条 continuation regressions 等负尾部，作为一条鲁棒性句即可。

### 应删除或大幅压缩

1. Workload 段的 `105/93/81` ATLIF wrapper 盘点与 attention/decoder 身份说明。它们不决定 C1/C2 电路，压成“冻结 ep34、AEE、firing rate、binary real source”两句。
2. “Each result is assigned one evidence class”整段。把 `[model]/[VCS]/[P&R]/[PX]` 放在主表即可。
3. C1 Evaluation 中对 64-row quantum、1.6945/1.902 和 1RW 边界的多次重复。机制段解释一次、结果段报一次。
4. G48 `1,917+3` host-orchestration lineage 的长段。优先用 clean batch 替换；若来不及，仅在 evaluation footnote 一句披露，详细 seal 放 artifact。
5. Related Work 的逐工作防御段与 Table IV 重复。保留 compact prior/object/private-state table，加一段解释即可。
6. Discussion 与 Limitations 合并为一个 8--12 行的 `Scope and Limitations`。删除“A reader seeking silicon metrics should use...”以及“This brief refuses several claims...”整段；这些文字诚实但像答辩记录，不像结果优先的 Express Brief。
7. Figure 1/2 内部字号目前在单栏打印下偏小。删去 figure 2 的 evidence ladder，把空间给 SRAM request/read-enable、cache hit、四个 private Acc24 的可读 timing 图。

压缩后省出的篇幅应由真实 routed/power 结果、能量定义和 token-coverage robustness 填充，而不是再放免责声明。

## matched power/P&R 到位后的主表写法

不应把所有数字塞进一个“ours vs prior”倍率表。最清楚的是一张跨双栏的 paired-ablation table，所有比值只在同一 pair 内计算：

| Pair / mode | Frozen scope | Cycles | 128-b bank reads | Routed cell area | Setup/Hold WNS | Logic energy | SRAM energy `[model]` | Evidence |
|---|---|---:|---:|---:|---:|---:|---:|---|
| C1 strongest-zero | 51.84 M rows | 648.741 M | -- | -- | -- | -- | -- | model |
| C1 finite-1RW | same rows | 382.849 M | -- | 166,514 um2 island | +27.9/+1.8 ps prelayout | 22.07 nJ/window | -- | model + VCS tile + DC/PT/FM/PX |
| K1x8 | five equal-service loads | 1,945 | -- | 585,479 um2 logic | report scope | open/filled | common | VCS + DC |
| K8 | same loads | 1,913 | -- | 131,086 um2 logic | report scope | open/filled | common | VCS + DC |
| ordinary-LRU4 | 2,880 fixed regions or full-token successor | 92.652 M | 21.088 M | routed value | routed value | measured value | modeled value | VCS + P&R + PX |
| TSBG-B4 | exact same population | 50.505 M | 8.830 M | routed value | routed value | measured value | modeled value | VCS + P&R + PX |

表注必须明确：

- C1 ratio 与 TSBG ratio 不相乘；
- `4.541x` 是 K8/K1x8 directed-throughput/logic-area，不是 cycle speedup；
- C1 九宏 island 与完整 105-macro ledger 分列；
- TSBG 288-KiB capacity/area/leakage在两轴相同，只有 read activation 和执行 duration 变化；
- logic PTPX 与 SRAM QRT/model 不混成“measured total”，证据标签分开；
- 若 P&R 只覆盖 macro-free logic island，表头必须写 `routed logic island`，不可写 chip/accelerator PPA。

物理结果成立后，摘要最有效的一句话形式是：

> At 3 ns after matched routing, TSBG reduces post-load cycles by 45.5% and 128-bit bank activations by 58.1%, yielding X% logic-energy and Y% modeled weight-store-energy reductions with Z% routed-cell-area overhead; four signed accumulator contexts remain bit-exact.

其中 X/Y/Z 只有在同身份结果独立准入后才能填写。

## Strong Accept 的最小实验集

### P0-S1：matched TSBG routed identity

普通/TSBG 两轴必须使用相同 M2018 source、floorplan、pin order、clock/IO、PVT、CTS、route layer 和 hold-repair policy。两轴都需满足：library/import checks、connectivity/DRC、setup/hold WNS `>=0`、post-physical equivalence。失败的 M2135 不提供任何正物理数字；下一步先完成独立 library-import preflight，再启动新身份 full P&R。

### P0-S2：同身份 logic + weight-store energy

至少对三个预注册 activity 点（建议按 request density 取 low/median/high，且来自不同 sequence）运行 ordinary/TSBG：

- DUT-only activity 的 duration 与 cycle count 完全一致，`TX=0`；
- transformation-mapped RTL SAIF 或 direct gate SAIF 的 annotation/key-cone gate 达到冻结阈值；
- PT-PX 分列 internal、switching、leakage、total，能量按各自真实 duration 计算；
- 288-KiB SRAM 的 area/leakage两轴相同，dynamic energy 按真实 bank reads 与同一 PVT read-energy 计价；
- 报 low/median/high 和按 fixed population 权重的 aggregate，不能只选一个收益最大的 slot。

一个 fixed slot 可以成为可引用 component power anchor，但不足以单独把稿件推到 Strong Accept；三点 activity sensitivity 是最小的抗选择偏差版本。

### P0-S3：token-coverage robustness

不要求把所有 token 都跑 gate-level VCS。最低成本路线是：建立一个 cycle calculator，先逐行精确匹配现有 2,880 VCS rows，再对 40 sample、24 layer identity 的所有 aligned B4 quartets replay；报告 ratio-of-sums、p10/p50/p90、最差值、fallback 比例和略慢 case 比例。若 calculator 不能 100% 匹配封存 VCS，不得用于 full-token headline。

### P0-S4：clean G48 batch 与最终版面

用一个新 identity、一次 compile、一次 simv batch 重新覆盖 G48 的 1,920 rows，以替换 `1,917+3` composite；这不改变结果机制，只移除不必要的 provenance 风险。随后回填 routed/power/robustness，确保 page 5 left column真正到 4.5 页、page 5 right references only，并补齐合法 authors、affiliations、e-mail、ORCID 与 funding disclosure。

### P1：显著加分但不是最小硬门

1. 将 C1 same-ledger replay扩到与 TSBG 相同的另外三条 DSEC sequence，报告 ratio-of-sums 与最差 sequence；若 C1 仍约 `>=1.6x`，可消除单序列外推疑问。
2. 将 C1 的九宏 island 与 full-ledger 105-macro area model画成一个清楚的存储分解，而不是在 limitations 才解释。
3. 若 P&R 显示 TSBG routed area/clock tax显著，增加 cache-entry/B4 的小 DSE；否则不要为凑图而新增轴。
4. 对 K8/K1x8 增加 trace-derived service distribution或至少说明五个 directed loads 覆盖的边界条件；不要把 `4.541x` 当普适 workload speedup。

## 是否需要新机制

**不需要。** 第三个机制会带来新的 exactness、baseline、RTL、PPA 和篇幅债务，而当前稿的接收风险都来自未闭合证据，不来自 idea 数量不足。

唯一允许的“新改动”应是 C2/TSBG 内部的物理化，而不是新 contribution：reuse-hit-qualified SRAM read-enable、地址译码/row-register activity suppression、context-local clock enable。这些本质上是在实现论文已经声称的 hit-before-request；若现有 RTL 已经阻断请求，就只需测量，不得另起名字重新包装。只有 routed/power 结果显示 request 已减少但 SRAM/logic energy没有下降时，才根据 first-divergent power breakdown 做针对性电路修正。

## 评分

| 维度 | 当前 /5 | P0-S1--S4 后 /5 | 审稿判断 |
|---|---:|---:|---|
| Novelty | 3.55 | 3.7--3.8 | 两个 object difference 足够；第三机制反而稀释 |
| Circuits fit | 4.35 | 4.6 | 1RW SRAM、typed accumulator、pre-read gating 与事件视觉契合 |
| Soundness | 4.55 | 4.7 | 边界/封存很强；full-token 与 clean batch 可补 selection/provenance |
| Implementation | 3.65 | 4.45 | C1 强，C2 当前 hold/power/P&R open |
| Evaluation | 3.75 | 4.4 | all identity 强，但 fixed regions 与单序列仍限制外推 |
| Presentation | 3.25 | 4.35 | 5 页但 underfill；图小、免责声明过多、author 未填 |
| **Overall** | **3.8** | **4.25--4.4** | 当前 Weak Accept；最小闭环后才有 Strong-Accept 倾向 |

## 最终裁决

**保留 C1 + C2/TSBG 两条 exact reuse circuit，不开第三线。** 现在最值得推进的不是更多剪枝或“纸面倍率”，而是把 TSBG 已经真实减少的 request 转成 routed hold-clean、matched logic energy 与 same-capacity SRAM energy，并用 full-token calibrated replay证明 `1.8345x` 不是三个位置的偶然结果。完成这组最小证据后，论文才能从“边界很诚实的组件稿”升级为“性能、能量和物理代价闭合的 TCAS-II 电路 brief”。
