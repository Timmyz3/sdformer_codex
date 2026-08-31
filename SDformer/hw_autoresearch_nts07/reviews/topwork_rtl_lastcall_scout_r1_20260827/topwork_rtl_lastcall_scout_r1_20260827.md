# Motion/H67 新增性能 RTL 最后审计 r1

日期：2026-08-27（Asia/Shanghai）  
证据截点：2026-08-27 11:27 CST  
范围：Conv/ConvTranspose、FC1/FC2、dynamic BN、ATLIF、attention  
动作边界：只读本地证据与一手公开来源；未写 RTL，未运行 VCS/DC/PT/GPU/DSE，未修改 `docs/359`。

## 独立裁决

**当前立即新增的性能 RTL 数量为 0。** 裁决为：

`NO_NEW_PERFORMANCE_RTL_NOW__ONE_TRACE_GATED_TDR_OPTION__CLOSE_C1_C2_C3`

唯一仍保留的“未来新性能 RTL”候选是 decoder 的 exact T10 temporal-delta replay
（TDR），但它必须先由已经冻结的 M511→M513 链得到真实 transition ratio，再经过
数值 miter、状态 SRAM 和净能量三道门。当前不授权编码。M514 只属于 C2-D decoder
完整性适配器，不是性能候选。

这个结论与既有 r5 的方向一致，但不是照抄。独立重核发现：

- corrected analytical envelope 为 `790.920--803.774 M cycle/frame`，decoder
  sensitivity share 为 `21.57--22.83%`；旧 `620.303 M` 只能叫 included-scope；
- decoder phase scheduler 和 PGPR 均被强 96-wide、1R1W output-stationary A1 的
  `P/96` 下界否决；
- Conv、FC1、patch、attention 的剩余候选分别被 port/capture 税、真实 recurrence、
  极低整块空率和 Amdahl 否决；
- dynamic-BN 16/24-bit replay 有流量价值，但 standalone novelty 与完整系统灵敏度
  均不足，只能作为 memory/energy 支撑机制；
- 若 ATLIF PAFT/rank-3 训练最终成功，已有 M273r2/M289 数据通路已经覆盖所需
  rank-3 执行；下一步是 checkpoint-bound replay、同接口 Fixed baseline、state macro
  与 matched SAIF/PTPX，不是再写一条性能 RTL。

主线程汇总提到 `new_rtl_topwork_gap_scout r3` 的先验裁决为
`NO_NEW_PERFORMANCE_RTL`。本审计在当前盘面未找到该目录或 seal，因此没有把它当作
可核验输入，只把它当作待挑战先验；本报告的数值与结论均来自下列可定位证据。

## 修正后的 Amdahl 地图

M510 发现四层 `ConvTranspose2d` 未进入旧 operator ledger。下面份额用 M510 的
aggregate-count analytical bound 计算，只用于优先级，不是 exact S10 系统周期。

| 作用域 | 冻结/分析 cycles | 修正 envelope 份额 | 已有最好局部点 | 理想 envelope sensitivity |
|---|---:|---:|---:|---:|
| Patch embed Conv | 199.421 M | 24.81--25.21% | 假设局部 1.20x | 1.0431--1.0439x |
| 全网 ATLIF | 128.021 M | 15.93--16.19% | rank3 analytic 3.3999x | 1.1267--1.1290x |
| FC1 | 118.370 M | 14.73--14.97% | RTL recurrence 1.3599x | 1.0406--1.0412x |
| 四层 bottleneck Conv | 79.631 M | 9.91--10.07% | M473 fused 1.9436x | 1.0505--1.0514x |
| FC2 | 41.414 M | 5.15--5.24% | K8/K1 4.7642x，非同峰值 | 1.0424--1.0432x |
| Attention core | 3.656 M | 0.455--0.462% | 无限快 | 1.00457--1.00464x |
| 四层 ConvTranspose | 170.617--183.471 M | 21.57--22.83% | TDR 尚待 exact S10 | 局部 1.30x 仅 1.0524--1.0556x |

以上 sensitivity 都不准进主性能表。尤其 C3 的 3.3999x 是额外 stage1+stage2 资源下的
isolated analytical module cycle；C2 的 4.7642x 是 K8 对低带宽 K1，不是 K8 对等峰值
K1x8。

## 一手工作核验与真正可迁移的 trick

| 工作 | 已核机制/开源身份 | 对本项目能借什么 | 为什么没有产生第四条 RTL |
|---|---|---|---|
| Prosperity，HPCA'25 | 官方仓库含 cycle-accurate simulator、baselines、CACTI、time/energy reference；product sparsity | product-vs-bit 同框架、memory-inclusive energy、full-workload aggregation | M472 已完成官方框架重放；新 matcher 与 C1/M473 高度重复 |
| Phi，ISCA'25 | L1 predefined pattern/PWP + L2 residual + PAFT；论文报 3.45x/4.93x | exact/lossy 分列、训练身份与 Pareto | PAFT near-match 已有 accuracy NO-GO；无重训条件下不能复刻 L2 |
| Bishop，ISCA'25 | TTB、dense/sparse heterogeneous cores、error-constrained attention pruning | density stratification、误差预算单列 | attention 只有约 0.46%；无限快也约 1.0046x |
| FireFly-T，2025 arXiv | multi-nonzero decode、bank-aware dispatch、OOO workers、binary attention | equal-throughput bank baseline、冲突/尾部覆盖 | C2/FC2 已覆盖多 bank issue；仅 bank dispatch 的 novelty 已被占位 |
| ELSA，2026 arXiv | spine/token elastic pipeline、bundled AER、mini-batch spiking Gustavson product | compute/NoC/memory 同时计价、first/complete latency 分列 | H67 current-batch BN barrier 与无 NoC ledger 阻止直接迁移 |
| SNE，DATE'22 官方 RTL | sparse event Conv、resident state、standalone RTL+Python model | typed event protocol、state locality、两级验证 | event router/state residency 已是强 prior art；C2/M221 已覆盖 glue |
| DeltaCNN / MotionDeltaCNN | delta/update mask、非线性 state cache、moving-camera spherical buffer | TDR 必须收费 state、refresh、mask dilation、相机运动 | TDR novelty 只能来自 exact T10 deconv + analog ATLIF 数值桥 |
| FEATHER，ISCA'24 官方 ASIC RTL | reorder-in-reduction、LayoutLoop、两级 functional/cycle 验证、DC/PnR reports | adapter 必须藏在 reduction 内；同 top 多配置物理表 | 当前没有 exposed layout-reorder ledger；M499 已是 adapter |
| ESDA，FPGA'24 官方 artifact | submanifold sparse Conv、token-feature interface、网络硬件共设计 | 中间层稀疏必须实测，不能从 event input 推断 | H67 patch whole-temporal 空 site 仅 156/4,032,000 |
| LoAS | fully temporal-parallel dual-sparse SpMSpM、compression、inner join | metadata/prefix tax与 payload 分列 | H67 不是冻结的 dual-sparse binary-weight workload |

额外核到的 HPCA'26 Best Paper Candidate Focus 官方仓库提供 algorithm trace、cycle
simulator、RTL、CACTI、DRAMsim3 和 baseline 的完整链。它最值得借的是**同一 trace 驱动
algorithm/simulator/RTL/memory 的评估组织**，不是把 VLM semantic concentration 移植到
H67。当前项目的 BP 缺口也正是这一完整链，而不是缺一个模块名。

## 候选逐项 last-call

### C-NEW-1｜Exact T10 decoder temporal-delta replay（唯一条件候选）

- **机制**：逐 timestep 对 binary deconv input 做 XOR；`0→1` 发 `+weight`，`1→0`
  发 `-weight`，从前一 timestep output 增量更新当前 output。
- **已有先例**：DeltaCNN/DeltaRNN 的 delta update；MotionDeltaCNN 的移动相机 state
  管理。SNE 提供 sparse event convolution 邻居。
- **项目差异**：K3/S2/P1/OP1 ConvTranspose 的 exact signed delta、T10 内 reset、
  binary ATLIF input 与后续 analog ATLIF/current-batch 数值桥；不依赖阈值或重训。
- **冻结机会率**：尚无生产数值；M513 只接受 `P_delta/P_A1 < 0.7692`
  （ideal `>1.30x`）。decoder analytical share 为 21.57--22.83%。
- **强 baseline**：同 descriptor、同 96 product lanes、同 weight/psum SRAM ports、同
  destination commit 的 exact bit-sparse A1；t0 以 zero state 开始。
- **公平资源**：TDR 和 A1 同 weight bandwidth/ports；TDR 额外 input/output state、
  XOR/transition descriptor、signed accumulate 和 refresh 全收费。
- **预期 Amdahl**：局部刚过 1.30x 只有约 1.052--1.056x analytical sensitivity；
  局部 2x 才约 1.121--1.129x，且未扣 state traffic。
- **状态/存储税**：previous input bitmap `870,300 B`；previous output INT16
  `21,196,800 B`，Acc24 `31,795,200 B`。除非数值桥证明 output state 可被已有
  downstream state 无损吸收，否则大概率被 SRAM 主导。
- **48 小时 kill gate**：M511/M513 exact S10 首先过 `>1.30x`；再做 canonical-order
  integer miter 0 mismatch、overflow=0；计入 state SRAM 后 cycles `>=1.20x`、净能量
  `>=20%`；新增逻辑/状态面积不超过 A1 的 20%。任一失败，永久 `NO_GO_RTL`。

**当前裁决：WAIT_TRACE__NO_RTL。** 即使 ideal ratio 过门，也只升到数值与 SRAM
模型，不直接升 RTL。

### C-NEW-2｜ATLIF rank-3/PAFT 路线

- **机制**：低秩 stage1 product + stage2 shift/add，替代 Fixed 1600 INT8 product/tile。
- **已有先例**：低秩/分解计算广泛存在；SNE/ELSA 占 resident/elastic state 邻域。
- **项目差异**：H67 analog ATLIF 的 rank/phase-decoupled two-stage recurrence 与 exact
  threshold/event output。
- **冻结机会率**：M265 ideal matched-boundary Fixed `124.412 M` vs rank3 `36.593 M`，
  3.3999x；但 ep35 全 rank0，rank3 checkpoint/accuracy 未准入。
- **强 baseline**：同 96 multiplier slots、同 input/config/result ports、同 persistent
  membrane state、同 state SRAM latency 的 tile-closed Fixed engine。
- **公平资源**：rank3 独有 stage1 multiplier 和 M37-class stage2、intermediate bank、
  config bits全部收费；Fixed 的 accumulator/compare/writeback 也必须物理实现。
- **预期 Amdahl**：未扣资源的 analytical upper sensitivity 约 1.127--1.129x。
- **状态/存储税**：当前 M273 working state 是 stdcell、小深度多端口；full-network
  persistent membrane-state depth仍未冻结。M289 `102,852.29 µm²` 是 0-macro logic-only。
- **48 小时 kill gate**：若训练成功，直接用现有 M273r2/M289 重放 trained config；
  补同接口 Fixed baseline、同 state macro、Formality、matched SAIF/PTPX。要求
  rank3 throughput/mm² `>=1.50x`、energy/tile `<=0.70x`，且 valid accuracy 身份通过。

**当前裁决：NO_NEW_PERFORMANCE_RTL。** PAFT/rank3 成功只会激活现有 C3；允许写的是
公平 Fixed comparator/wrapper，属于 baseline closure，不是新机制或第四条贡献。

### C-NEW-3｜Dynamic-BN 16/24-bit bit-tight raw replay

- **机制**：current-batch barrier 保持不变，per-phase 静态 `{16,24}` container 紧凑写入
  raw store，再以 32-lane 512-bit replay sign-extend 到 Q24 边界。
- **已有先例**：CICC'26 BWAC、Stripes、Loom、Bit Fusion；DeltaCNN 也强调 BN fusion。
- **项目差异**：位宽来自 binary-event×INT8 weight 的 analytic sumabs bound，且对象是
  current-batch BN barrier raw tensor，不是普通 weight/activation precision scaling。
- **冻结机会率**：Q24 useful raw traffic `2.62656 GB` → 16/24 `1.815552 GB`，
  1.44670x；32-lane fused schedule 1.44684x。尚无 integer bridge/address trace。
- **强 baseline**：M480 Q24 strong fused raw replay，而非 normalized-materialization
  strawman；同 1R1W、barrier、coefficient overlap、bus/lane cap。
- **公平资源**：packer、cross-beat carry、tail mask、32-lane unpack/affine consumer、
  metadata、SRAM padding/ports全部收费。
- **预期 Amdahl**：按 corrected analytical envelope，零重叠替换仅约
  1.0194--1.0197x sensitivity；主要价值是 memory energy。
- **状态/存储税**：峰值 raw retention 约 `140.625 MiB`（16/24 point）；物理 macro、
  address-timed schedule 和 consumer 尚未闭合。
- **48 小时 kill gate**：integer bridge与24 phase overflow=0；useful traffic
  `>=1.35x`、同 lane local schedule `>=1.20x`；control `<=15,000 µm²`；matched
  raw-path energy降低 `>=20%`。任一失败降为 traffic-only。

**当前裁决：OFFLINE_TRACE_ONLY；即使过门也只是 C3/FFN memory 支撑，不是独立
performance contribution。**

### C-NEW-4｜第四种 Conv matcher/parent cache/queue

- **机制**：继续尝试 product/PWP parent reuse、payload residency 或更深 parent queue。
- **已有先例**：Prosperity product reuse、Phi PWP/residual、FireFly-T OOO/bank dispatch。
- **项目差异**：本项目的 signed dual-destination atomic commit 仍有一定实现差异。
- **冻结机会率**：M473 fused 1.9436x，但 unfused-sync 仅 1.0147x；M468/lazy-PWP、
  M470 payload-stationary、G15 均暴露 capacity/spill/capture cliff。
- **强 baseline**：同 128 B/cycle、同 `<=240 KiB`、同 row tile、同 parent scratch/
  psum ports 的 strongest exact bit-sparse zero/product baseline。
- **公平资源**：64x1152b 1R1W parent scratch、dual response slots、CAM、resident psum、
  completion/forwarding全收费。
- **预期 Amdahl**：即便完整捕获 fused opportunity，corrected sensitivity约
  1.0505--1.0514x。
- **状态/存储税**：parent scratch QRT exact DP fallback面积 `473,034.72 µm²`；一个
  64-row psum bank QRT `113,087.47 µm²`，远大于 selected logic slice。
- **48 小时 kill gate**：不再开新结构。只完成 C1/既有实现的 macro-aware
  energy/capture-gap；任何新增 matcher/router 自动 NO-GO。

**当前裁决：KILL_NEW_RTL；收口 C1，不复活 M468--M478 变体。**

### C-NEW-5｜FC2/FC1 再扩 bank/context RTL

- **机制**：FC2 多 bank 同拍发射与共享 Acc24；FC1 增加 factor/context 并行。
- **已有先例**：FireFly-T bank dispatch、LoAS temporal-parallel、Prosperity/Phi reuse。
- **项目差异**：FC2 的一份 signed Acc24 partial state 对比八份 scalar state 是清晰差异。
- **冻结机会率**：FC2 K8/K1 4.7642x，但 K8/K1x8 等峰值 1.000x；FC1 真实 RTL
  recurrence 仅 1.3599x，未过 1.50x 门。
- **强 baseline**：FC2 为同 8 bank/同 response schedule 的 K1x8；FC1 为同 port/lane/
  held-context 的现有 F2。
- **公平资源**：同 top/SDC/IO；weight macro相同；K8 与 K1x8 的 context/state复制税
  必须入面积和能量。
- **预期 Amdahl**：FC2 对低带宽 K1 的理想 corrected sensitivity约 1.042--1.043x；
  FC1 现有 recurrence约 1.041x。
- **状态/存储税**：FC2 公共 288 KiB weight banks约 `558,507 µm²`；K1/K8 context
  18,432 bit，K1x8 为147,456 bit。
- **48 小时 kill gate**：只允许 M496 K1/K8/K1x8 三点收口。K8/K1x8 area与seq cells
  均 `<=0.50x`、throughput/mm² `>=2.0x`、energy/token `<=0.70x`；否则降级。FC1
  禁止新 F4/F8 RTL。

**当前裁决：FC2 EXISTING_CLOSEOUT；FC1 NO_GO_NEW_RTL。**

### C-NEW-6｜Patch/submanifold/空 tile engine

- **机制**：仅对非空 spatial token/site 执行，descriptor 同时抑制 compute 与 fetch。
- **已有先例**：ESDA、SNE、ELSA bundled AER。
- **项目差异**：event-camera optical-flow 输入原生稀疏，但当前 H67 并未保持
  submanifold topology。
- **冻结机会率**：patch whole-temporal zero site 只有 `156/4,032,000 = 0.00387%`；
  strong bit-sparse baseline已跳 source zero。
- **强 baseline**：同 line buffer、same source-zero skip、same DMA/commit 的 bit-sparse
  patch Conv。
- **公平资源**：token map、coordinate metadata、line buffer、descriptor scan和
  dense fallback全收费。
- **预期 Amdahl**：尽管 patch份额约25%，该空率带来的理论收益近似1.00001x。
- **状态/存储税**：额外 coordinate/token directory，而不是免费 mask。
- **48 小时 kill gate**：冻结机会已远低于30% traffic或1.20x local门，无需再测。

**当前裁决：NO_GO。** 不改网络/重训就不能借到 ESDA 的主要收益。

### C-NEW-7｜Attention epsilon-prune / mass truncation / RQTB 扩展

- **机制**：近等 score 合并、低质量尾部剪枝、empty-Q/K-zero skip。
- **已有先例**：Bishop ECP、Phi residual pruning；本项目已有 lossless RQTB。
- **项目差异**：analog Q7 score quotienting 与 ordered K recovery。
- **冻结机会率**：attention约0.455--0.462%；现有 RQTB局部1.1865x。
- **强 baseline**：同 attention core、同 score/K-store、同 exact output的 lossless RQTB。
- **公平资源**：score bins/mass accumulator、error budget state和 accuracy验证收费。
- **预期 Amdahl**：attention无限快也仅1.00457--1.00464x。
- **状态/存储税**：K-store约是局部能量大头，但不能转成系统周期 headline。
- **48 小时 kill gate**：作为附录可做 energy/accuracy Pareto；主性能 RTL 门直接失败。

**当前裁决：NO_GO_MAINLINE；A1 只作 supporting ablation。**

### C-NEW-8｜PGPR、phase scheduler、bundled-AER NoC、reorder adapter

- **机制**：output-stationary psum residency、phase balancing、elastic packet/reorder。
- **已有先例**：FireFly-T OOO/bank balance、ELSA AER/Gustavson、FEATHER RIR、SNE
  event streaming。
- **项目差异**：C2 binary descriptor与K3/S2 asymmetric 4/2/2/1 tap mapping。
- **冻结机会率**：四层 Cout均为96整数倍，A1每合法tap正好填96 lanes；PGPR/EPD
  product issue上限均1.0x。当前也没有 exposed NoC/reorder cycle ledger。
- **强 baseline**：96-wide、1R1W、deterministic parity-bank、output-stationary A1。
- **公平资源**：相同 RF/SRAM ports/FIFO/forwarding；不能免费增加 resident RF或端口。
- **预期 Amdahl**：cycle gain 1.0x；最多有 traffic/energy side effect。
- **状态/存储税**：RF、descriptor FIFO、NoC/reorder switch都会新增；未暴露瓶颈时净负担。
- **48 小时 kill gate**：理论下界已关闭 performance RTL。M514 只做 exact address
  adapter一次 VCS收口，不独立报 speedup/PPA。

**当前裁决：KILL_AS_PERFORMANCE；M514 COMPLETENESS ONLY。**

## 最终硬件收口面

### 必收口

1. **C2 / M496 FC2 shared-state K8**：完成三点 matched DC 的 blind hammer；过门后只做
   Formality、matched SAIF/PTPX、公共 weight/context macro。主指标是
   throughput/mm² 和 energy/token，不是4.764x。
2. **C1 exact balanced sparse-Conv slice**：保留现有 VCS/Formality/PT/PTPX锚点；补
   macro-aware memory energy、capture-gap和完整 workload映射，不再改乘法器/queue。
3. **C3 ATLIF**：若 rank3/PAFT checkpoint成功，复用 M273r2/M289；补 Fixed comparator、
   checkpoint-bound full trace、persistent state macro与 matched energy。若训练失败，
   rank0只作 adapter/完整性，不硬凑性能贡献。
4. **M514 C2-D**：一次 exact-SHA Synopsys VCS后并入 decoder completeness；不单列贡献。
5. **M511/M513**：TDR只按上述 gate决定是否获得最后一个 RTL名额。

### 不再收口为独立贡献

- M468--M478 Conv变体、PGPR/EPD scheduler、FC1 F4/F8、patch token engine、dynamic-BN
  standalone packer、epsilon-RQTB、bundled-AER/NoC、FEATHER式独立 reorder。
- “2.459x Prosperity official replay”“1.9436x fused opportunity”“4.7642x K8/K1”三者
  只能分列，不能相乘，不能写成我们的系统加速。

## 对 DATE/BP 的最终影响

新增一个未闭合模块不会提高当前约3.2/5的录用证据；它会增加 prior-art 暴露、宏税和
验证债务。真正能抬升评分的是：

- corrected non-overlap full H67 denominator；
- 至少三条真实 DSEC sequence的 mean/worst；
- logic+SRAM+DRAM 的 energy/frame；
- 同 trace、同资源的 Prosperity/Phi-like/bit-sparse公开对标；
- 至少一个完整 top 的 throughput/mm² `>=2.0x` 且 energy/frame `<=0.70x`，或整网
  speedup geomean `>=1.20x`、worst `>=1.10x`。

所以本轮“最后审计”的正确动作不是继续寻找第四个 trick，而是执行上述物理与系统收口。
TDR 若失败，最终论文以 C2+C1为两条主硬件贡献，C3过公平门才作为第三条；A1/M514
只作完整性与消融。

## 一手来源

- Prosperity official artifact: <https://github.com/dubcyfor3/Prosperity>
- Phi paper and ISCA'25 program: <https://arxiv.org/abs/2505.10909>,
  <https://www.iscaconf.org/isca2025/program/>
- Bishop: <https://arxiv.org/abs/2505.12281>
- FireFly-T: <https://arxiv.org/abs/2505.12771>
- ELSA: <https://arxiv.org/abs/2605.20802>
- SNE official RTL: <https://github.com/pulp-platform/sne>
- DeltaCNN official code: <https://github.com/facebookresearch/DeltaCNN>
- MotionDeltaCNN official paper: <https://openaccess.thecvf.com/content/ICCV2023/html/Parger_MotionDeltaCNN_Sparse_CNN_Inference_of_Frame_Differences_in_Moving_Camera_ICCV_2023_paper.html>
- FEATHER official ASIC stack: <https://github.com/maeri-project/FEATHER>
- ESDA official artifact: <https://github.com/CASR-HKU/ESDA>
- LoAS: <https://arxiv.org/abs/2407.14073>
- Focus full-stack artifact: <https://github.com/dubcyfor3/Focus>

## 身份与局限

- `docs/359` SHA256仍为
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
- M496仍在运行，未将 partial log当结果；M511/M513生产 payload尚未在本报告中出现。
- corrected envelope来自M510 analytical bound；exact S10完成前不得称 measured system
  cycles。
- FireFly-T与ELSA按arXiv预印本处理；本轮未把它们描述成已核实正式venue或官方RTL。
- 本报告未找到可核验的 `new_rtl_topwork_gap_scout r3` 目录，因此没有把其先验裁决计入
  独立证据。
