# M486 顶会/开源硬件机制独立挖掘 r1

日期：2026-08-27  
范围：H67/Motion；四层 bottleneck Conv、ATLIF state、FC2 为主，attention 仅作边界核验。  
动作边界：只读研究；未修改 RTL、合同或 `docs/359_DATE终局冻结_20260813.md`，未启动 DC、VCS 或 GPU。

## 结论先行

**不建议再开发第四种通用稀疏乘法器、attention 主线或全系统调度器。** 已有负结果与公开前作共同表明，DATE 截止前最有价值的不是继续发明 descriptor/matcher，而是把已经测到的稀疏机会封闭成公平、物理可执行的模块证据。

本轮只保留三个方向，其中只有前两个允许立即写少量集成/修复 RTL，第三个必须先过算法精度门：

| 排名 | 方向 | 当前裁决 | 当前/潜在可投稿性 | 本轮是否允许新 RTL |
|---|---|---|---:|---|
| 1 | Conv fused parent-read / dual-update 的 M479 物理收口 | **GO closeout；禁止新 matcher** | 80/100 → 88/100 | 仅允许由物理报告触发的 lane-enable 分段或 slot 实现修复 |
| 2 | FC2 bank-coissue + shared Acc24 context | **GO matched wrapper/Pareto** | 70/100 → 82/100 | 允许统一 replay wrapper、canonical 8-bank adapter 和 shared-context top；不写第四种算术核 |
| 3 | PAFT near-match residual elision，`tau=0` 为 exact 子集 | **MEASURE-ONLY；当前 NO-GO RTL** | 45/100 → 74/100 | 只有 `tau=3` 精度与同资源门同时通过后才允许在 Conv 路径前加阈值门 |

关键反直觉结论：

1. **Prosperity/Phi/FireFly-T 的“技巧”大部分已与本项目 C2/M473/M479 的 source decode、bank-aware dispatch、PWP/remainder 路径同族。** 再复制一个 decoder 或 matcher 不会形成独立贡献。
2. **ATLIF 不是没有机会，而是新 arithmetic RTL 已经存在。** M273/M289 的 rank-3 两级 ATLIF 路径缺的是训练身份、协议/物理准入与同资源比较，不是再发明一条 ATLIF datapath。
3. **光流的输入稀疏不等于网络中间张量空。** H67 的空 output-site 仅 `0.1117%`；ESDA 式 submanifold 空 tile 跳算在此 checkpoint 上被真实数据直接否决。
4. **跨帧 delta 是已拥挤赛道。** DeltaCNN、MotionDeltaCNN 和 DeltaRNN 已覆盖阈值化 delta、端到端 update mask、移动相机对齐/缓存等核心思想；本项目当前真实 trace 的 local-vs-temporal 逐行选择只节省约 `2.7%` source-work，不足以直接开 RTL。

## 1. 公平分母与 Amdahl 作用域

冻结机会 envelope 为 `620,302,905` cycles。它用于优先级和 Amdahl 灵敏度，不是已经带 SRAM/DRAM 重叠、FPS 和能量的系统周期。

| 作用域 | 冻结 cycles | envelope 份额 | 对新 RTL 的含义 |
|---|---:|---:|---|
| Patch embed Conv | 199,420,620 | 32.1489% | 最贵，但强同-bank baseline 已使既有 patch 方案关停；不能因份额大就重开 |
| 全网 ATLIF | 128,020,500 | 20.6384% | 值得收口已有 M273；G12 early-stop 几乎无收益 |
| FC1 | 118,370,114 | 19.0826% | M482 的真实 RTL recurrence 仅 `1.3599x`，理想 envelope 灵敏度 `1.0450x`，已 NO-GO 新扩宽 RTL |
| 四层 bottleneck Conv | 79,630,957 | 12.8374% | 当前最值得物理闭合的稀疏主线 |
| FC2 | 41,413,997 | 6.6764% | 适合做面积/能效贡献；即使局部 `4.764x`，理想 envelope 也仅约 `1.0557x` |
| attention core | 3,656,069 | 0.5894% | 无限加速的 envelope 上限也只有约 `1.0059x`；不得作为性能主线 |

四层 Conv 若把 M473 fused 机会 `1.9436x` 完整替换进上述份额，理想 Amdahl 灵敏度约 `1.0665x`；官方 Prosperity artifact 的 `2.4595x` product-vs-bit 机会对应约 `1.0825x`。两者都**不是系统倍速**，但足以说明 Conv 比 attention 更值得付 RTL/物理化成本。FC2 份额的无限加速上限也仅约 `1.0715x`，所以 FC2 应卖吞吐保持下的面积/能效，而非“全网 4.76x”。

## 2. 公开工作核验与能借什么

| 工作 | 发表状态 | 官方开源状态 | 可借机制 | 本项目边界 |
|---|---|---|---|---|
| Prosperity | HPCA 2025 | [官方仓库](https://github.com/dubcyfor3/Prosperity)公开周期精确 simulator、baseline、CACTI/参考结果；DC power/area 脚本明确未公开 | product sparsity、PWP、bit/product 同框架对比、端到端周期/能量表 | M472 可作官方框架重放；`2.4595x` 是官方 product-vs-bit 机会，不是本项目 RTL |
| Phi | [ISCA 2025 官方议程](https://www.iscaconf.org/isca2025/program/)已核；[论文](https://arxiv.org/abs/2505.10909) | 论文公开；截至本次检索未定位官方 artifact | L1 pattern/PWP + L2 residual sparsity + PAFT | 本项目 exact pattern 与 PAFT 已直接相邻；近似 residual-elision 必须证明额外差异和精度预算 |
| Bishop | [ISCA 2025 官方议程](https://www.iscaconf.org/isca2025/program/)已核；[论文](https://arxiv.org/abs/2505.12281) | 论文公开；截至本次检索未定位官方 artifact | TTB、dense/sparse heterogeneity、error-constrained pruning | 其 ECP 主要作用于 attention；本项目 attention 仅 0.59%，不能当主线 |
| FireFly-T | [arXiv 2025 预印本](https://arxiv.org/abs/2505.12771) | 未核到正式 venue 或官方 RTL；不得写成已发表 RTL artifact | multi-nonzero decode、bank-aware dispatch、OOO worker、binary attention engine | 支撑 C2/FC2 的银行感知发射合理性，也压低“只是 bank dispatch”的创新性 |
| SNE | [DATE 2022 论文](https://past.date-conference.com/proceedings-archive/2022/pdf/0908.pdf)；官方仓库引用 DATE | [官方 RTL 仓库](https://github.com/pulp-platform/sne)，Solderpad/Apache 许可 | event stream、稀疏 Conv、resident neuron state | 是 ATLIF/state 的强 baseline；不能把“状态驻留”本身写成新贡献 |
| ESDA | FPGA 2024 | [官方 artifact](https://github.com/CASR-HKU/ESDA)，含 HLS/训练/映射工具 | token-feature interface、submanifold sparse convolution | H67 空 output-site `0.1117%`，真实 checkpoint 不支持复制其空位优势 |
| DeltaCNN | [CVPR 2022 官方论文页](https://openaccess.thecvf.com/content/CVPR2022/html/Parger_DeltaCNN_End-to-End_CNN_Inference_of_Sparse_Frame_Differences_in_Videos_CVPR_2022_paper.html) | [官方 CUDA/PyTorch 仓库](https://github.com/facebookresearch/DeltaCNN)，不是 RTL | 阈值化 frame delta、update mask、跨层稀疏传播、BN 融合 | 任何跨帧跳算都必须与其误差/状态/端到端 mask 正面对比 |
| MotionDeltaCNN | [ICCV 2023 官方论文页](https://openaccess.thecvf.com/content/ICCV2023/html/Parger_MotionDeltaCNN_Sparse_CNN_Inference_of_Frame_Differences_in_Moving_Camera_ICCV_2023_paper.html) | 论文公开；截至本次检索未定位独立官方 artifact | moving-camera alignment、spherical buffer、padded Conv、dynamic initialization | 直接覆盖“上一帧特征 + 光流 warp”故事；本项目不能把它包装成空白创新 |
| DeltaRNN | FPGA 2018 | 论文公开；本次未定位官方 RTL | thresholded delta update，减少 RNN compute/memory | 是 ATLIF/state delta 的直接前作 |

未核实的“Sparse by Command/MICRO 2026”等条目不进入本报告的已发表 prior-art 表；只保留“mask 同时抑制 compute 与 fetch”这一一般设计原则。

## 3. 候选 1：M479 Conv capture-path 物理收口

### 机制与复用

复用 M473 的 fused parent opportunity、M474 的最小双更新 pipeline、M476R2 的 backpressure-safe 两槽队列和 M479 的 lane-local 物理实验。核心不是再找一种 pattern，而是让一次 parent/PWP read 在一个原子事务中完成两路 destination update，避免 M473 unfused 模型中几乎全部收益被 parent-read/row-completion bubble 吞掉。

M473 的冻结四层 Conv 数据为：bit baseline `757,946,784` cycles，fused `389,974,420`，即 `1.9436x`；对同预算 zero baseline 为 `1.9497x`。但 unfused-sync 为 `746,979,771`，只剩约 `1.0147x`。因此论文真正要解释的是 **capture gap**，不是再展示一个理想稀疏率。

### 强 baseline

- 周期：M468 strongest-zero、同 128 B/cycle 资源与同 trace；另列 unfused-sync。
- 物理：M475 单槽/最小 fused pipeline，与 M477/M479 同 3 ns、同约束、同 scratch 边界。
- 存储：parent scratch 必须按同容量、同 1R1W 宏或同一零宏政策比较；不得把 9 KiB 外部 scratch 免费化。

### 48 小时 fast-kill 与 RTL 门

1. 先让现有 M479r2 3 ns DC 完成；检查 `keep` 后的 lane enables 是否仍只是高扇出 global accept 的别名。
2. setup/hold 均 `>=0`，五类约束 clean；functional logic area 必须 `<=1.20x M475`，即 `<=44,779.2 um2`。超门或不收敛则停止双槽方向。
3. 若且仅若 timing/fanout 报告明确指向全局 enable，允许一次真正的 8/12-lane 分段寄存 enable tree；不得改 matcher、稀疏定义或事务语义。
4. 分段版必须在原有 VCS/SVA、协议攻击、Formality 与同一 fused recurrence 下为 0 mismatch；吞吐不得低于 M474 的 `0.98x`。
5. paper 级还需 1R1W scratch macro/CACTI、SAIF/PTPX 和至少两条 DSEC sequence。logic-only 通过仍不等于 paper PPA。

### 与 prior art 的差异

Prosperity/Phi 提供 product/pattern sparsity，FireFly-T 提供多非零 decode 与 bank dispatch；M479 可以主张的是 **signed analog SNN Conv 中，把 parent/PWP reuse 转换成 fail-closed、backpressure-safe 的双 destination commit**。这是“稀疏机会捕获微结构”，不是新的稀疏定义。创新性中等，但证据链最接近投稿闭合。

## 4. 候选 2：FC2 bank-coissue 与 partial-state collapse

### 机制与复用

复用 M216 frontend、M218 K8 service、M219 K1 service、M342/M349 测试边界，只新增三个统一功能 top/wrapper 与 canonical 8-bank adapter：

- K1：每拍至多一个 128-bit bank word，一份 O8/FIFO4/Acc24 context；
- K8：每拍至多八个 bank word，一份 O8/FIFO4/Acc24 context；
- K1x8：相同 1,024-bit/cycle 峰值，但八套 M219、O64/FIFO32 和八份 Acc24 context。

M349 已证明 K8 与 K1x8 在 directed equal-peak-bandwidth 边界为 `1.000x` cycles。因此贡献应是 **throughput-preserving bank-coissue and partial-state collapse**，而不是“同带宽再快 5.28x”。M342 的 `5.2814x` 只能表示从一 bank-word/cycle 扩到八 bank-word/cycle 的资源 Pareto。

### 强 baseline 与 Amdahl

- 性能 baseline：K1x8；相同八个逻辑 weight bank、每 bank 1R×128-bit、同 response schedule。
- 低逻辑 endpoint：K1；不能称同带宽。
- 面积/能量：三个 top 使用同一 debug crop、IO、hold 策略和宏政策。
- FC2 份额 `6.6764%`；局部 `4.764x` 的理想 envelope 灵敏度约 `1.0557x`。论文主指标应是 area、energy/token 和 throughput/mm2。

### 48 小时 fast-kill 与 RTL 门

1. 同一 120-record frozen H67 FC2 cohort 做三点统一 replay，逐 token 数值、request/response multiset、active-bank reads、weight bytes、inclusive cycle endpoint 全相等。
2. K8/K1 throughput geomean `>=3.0x`；K8/K1x8 每 record `>=0.95x`、geomean `>=0.98x`。
3. 同一 3 ns top-level DC：K8 area/K1 `<=1.25x`；K8 area/K1x8 `<=0.50x`；K8 Fmax/K1x8 `>=0.90x`；setup/hold 与五类约束 clean。
4. 只有前三门通过才做 macro + SAIF/PTPX；K8 energy/token/K1x8 必须 `<=0.70x`，annotation coverage `>=95%`。
5. 失败即停止，不写第四种 FC2 arithmetic RTL。

### 与 prior art 的差异

bank-aware dispatch 本身已不新。可投稿差异在于 **多 bank 同拍读出被一个 signed accumulation service 与一份 partial state 吸收**，并用吞吐匹配的八 scalar service 作为强 baseline 暴露状态/控制复制税。相对 FireFly-T 的 OOO worker，必须强调本项目是 raw4 analog source、Acc24 signed accumulation 与 atomic token completion，而非二值 spike worker 的复刻。

## 5. 候选 3：PAFT near-match residual elision（先算法、后 RTL）

### 机制

在 exact pattern/PWP path 上，对 population 至少为二的 partition 查最近 pattern；当 Hamming distance `<=tau` 时只发一个 PWP、不发 signed correction vectors。zero/singleton exact fallback 保留，`tau=0` 精确退化到 M251。

冻结 trace 的重要数字：

| tau | WIDE144 局部机会 | SHARED96 同较窄端口机会 | mean abs Acc delta | worst row max Acc delta |
|---:|---:|---:|---:|---:|
| 0 | `1.5406x` | `1.2329x` | 0 | 0 |
| 2 | `2.0002x` | `1.4275x` | 29.14 | 254 |
| 3 | `2.4114x` | `1.5948x` | 32.65 | 378 |

`tau=2` 的 2x 依赖 WIDE144；在 SHARED96 下低于本项目 `1.50x` 新 RTL 门。`tau=3` 周期过门，但尚无 modified-forward 或 valid825 accuracy。因此当前不准写 RTL。

### 48 小时 fast-kill 与 RTL 门

1. 用同一 PAFT checkpoint 做 S10 modified-forward，再做 paired running-BN valid825；相对 `tau=0` 的 `Delta AEE <=0.02`。
2. 层/序列分别报告 accuracy cliff，不能只报均值；tie-break 必须固定为 minimum packed uint16，M283 已证明 tie 非空泛细节。
3. 只接受 SHARED96 或与 M479 同端口/同 SRAM 资源的 `>=1.50x`；WIDE144 的 `2x` 不能跨资源准入。
4. 通过后允许新增的 RTL 仅为阈值寄存器、distance compare、exact fallback 与 correction-suppress gate；area `<=1.30x tau0`，`tau=0` 与 exact RTL bit/cycle miter 0 mismatch。
5. 若 `tau=3` 精度不过或多序列收益崩塌，永久降为负结果；不得尝试更大 tau 救数字。

### 与 prior art 的差异和风险

Phi 已有 L2 residual sparsity + PAFT，DeltaCNN 已有阈值化小更新，Bishop 已有 error-constrained pruning。单独的“近似匹配后丢 residual”创新性偏低。只有同时具备下列三项才有投稿价值：

- `tau=0` 是已验证 exact hardware 子集；
- 对事件光流给出 checkpoint-bound AEE Pareto，而非分类 accuracy；
- 在与 exact M479 相同端口/存储资源上证明额外收益。

当前 accumulator delta 不是网络/AEE 上界，禁止写“provably bounded optical-flow error”。

## 6. 明确关闭或暂缓的方向

| 方向 | 真实证据/前作攻击 | 裁决 |
|---|---|---|
| attention epsilon-RQTB / pruning | attention 仅 0.5894%；无限加速约 1.0059x；Bishop 已有 ECP | 附录 Pareto，非新 RTL 主线 |
| ESDA 式空 tile/submanifold | H67 空 output-site 仅 0.1117% | NO-GO |
| ATLIF remaining-budget early stop | M386 term skip 6.577%，issue reduction仅 0.0676%，条件 speedup约 1.00008x | NO-GO |
| 新 ATLIF arithmetic | M273/M289 integrated rank3 path 已存在；M265 隔离机会 3.3999x 但训练 rank3/公平面积缺失 | 收口旧 RTL，不新开 |
| row-bundle stationary M484 | 相对同资源 K8-resident 为 1.0000x，流量略增 | NO-GO |
| FC1 扩宽/重打包 | M482 真 RTL recurrence 1.3599x，未过 1.50 门；F4 预估吞吐增幅不足以覆盖 2x lane/port | NO-GO |
| dynamic BN materialization elision | 相对强 streaming baseline 为 1.0000x，raw retention 不变 | NO-GO |
| 跨帧 feature delta/warp cache | 当前自然 delta 只约 2.7% source-work；需前帧状态 SRAM；DeltaCNN/MotionDeltaCNN/DeltaRNN 直接占位 | 只允许离线多帧 fast-kill；不写 RTL |
| 新 spatial/event token engine | 光流输入稀疏未延续为空中间输出；ESDA 已提供强 prior art | NO-GO |
| 第四种 Conv matcher/PWP | M468/M470/lazy-PWP/G15 等已经暴露 SRAM、spill、unfused capture 瓶颈 | 禁止 |

## 7. 建议的 72 小时执行顺序

1. **不等待新 idea，先封 M479。** 跑完当前物理门；通过则做 Formality/PTPX/macro 计划，失败则只允许一次由报告指向的分段 enable 修复。
2. **并行做 M485 的统一三点 replay 与 matched wrapper。** 这是唯一值得新增的独立 RTL/Pareto 模块，且不会与 Conv 线冲突。
3. **算法服务器只做 M280 `tau=3` 精度 fast-kill。** 通过前硬件侧不编码；`tau=2` 在 SHARED96 已低于门。
4. **ATLIF 只收旧线。** 补 M273r2 协议回归、rank3 checkpoint/精度身份与 matched fixed-vs-rank3 物理比较；不新建 datapath。
5. **写作同步开始。** 主贡献结构建议为：C2 signed source engine；Conv capture-gap/fused parent reuse；FC2 shared-state bank-coissue。M472 官方 Prosperity 重放只放 iso-workload 对标层，M280 仅在精度过门后成为有损 Pareto 子节。

## 8. 对 DATE/BP 目标的诚实判断

这些机制足以形成一篇有竞争力的 DATE accelerator paper，但**单靠继续增加 RTL 点不会把项目推到 best-paper**。BP 级别更依赖下面四个闭环：

1. 同一 checkpoint、同一 trace、同一资源表中，把 C2/Conv/FC2 的 cycle、area、energy、memory traffic 放在一起；
2. 至少两条 DSEC sequence，报告均值和分层稀疏度，而非 Zurich 十窗口单点；
3. 28 nm DC/STA/Formality/SAIF/PTPX + 显式 SRAM macro/CACTI，不能用 zero-macro selected-slice 冒充芯片 PPA；
4. 与 Prosperity/Phi/FireFly-T/SNE 的比较分三层：论文原报、官方 simulator 上同 workload、自己的同资源 RTL。三种倍率不得混列或相乘。

当前最危险的审稿问题不是“idea 太少”，而是“2.46x、1.94x、4.76x 分别是什么分母，为什么不能相乘”。只要上述收口完成，三点贡献已经够；若继续横向开新模块，反而会降低完整度和可信度。

## 9. 局限与待回答问题

- 本报告没有执行新的 simulator、VCS、DC、PTPX 或 GPU accuracy；所有项目数字均来自仓库现有封存证据。
- M479r2 DC 在本报告快照时尚未形成最终封存结果，本报告不预判其面积或 timing。
- Phi、Bishop、FireFly-T、MotionDeltaCNN 的“未定位官方 artifact”是截至 2026-08-27 的本次检索结果，不等于作者明确声明永不公开。
- M280 的 accumulator delta 不能替代 valid825 AEE；M473/M472 的 isolated/support-tile cycles 不能替代全系统 cycles。
- 论文最终还需作者核实 DATE 2027 页数、匿名和 artifact 规则；本报告不把未核实未来会议条目当已发表事实。

