# DATE审稿人第三轮：Adaptive CSR后硬件评估

> 第三条后续补记：`docs/110_TypedSlotMetadata与IPD选择性驻留架构闭环_20260718.md`已落实本审稿第七节P0-1/P0-2的控制面部分：payload commit一次性校验格式并写入tag-coherent slot metadata，PLAN携带格式，只有IPD允许cache lookup/fill/warm replay，FADC与RAW从word0精确回放；六组真实trace双工具/SVA零mismatch。因此下文“重复PEEK”“Adaptive与residency合同冲突”已关闭。on-chip格式builder/policy、目标库PPA、扩展trace、valid825和full encoder分账仍未关闭，历史`2.8/5 Weak Reject`不能自动上调为接收。

> 主线程后续补记：第三轮独立复审完成后，`docs/107_PhysicallyStripped_Direct_RAW41投影基线_20260718.md`新增了不含resident/IPD/FADC/Adaptive/replay-mux的Direct RAW41 projection-slice顶层，并完成双模拟器、SVA、Erie和Yosys检查。因此下文“没有physically-stripped Direct”已在projection-slice结构代理层部分关闭；target-library PPA、整single-context物理边界和head-major spill仍未关闭，不改变Weak Reject总评。

> 第二条后续补记：`docs/108_AdaptiveCSR配置合同与SelectorSVA整改_20260718.md`已对`Adaptive + 旧residency`做elaboration fail-fast，并新增selector锁存、单child启动/收字和首字exactly-once SVA；`docs/109_HeadMajor_PSUM_Spill公平下界与架构决策_20260718.md`新增可综合head-major事务调度器和四stage真实trace spill下界。format-aware residency、完整head-major算术核和目标PPA仍未完成。

## 一、复审范围与证据口径

本轮以严格 DATE 硬件架构审稿人视角，重点复核第二轮提出的两个缺口：

1. 最强的 S0/S1/S2-IPD、S3-FADC24 结果是否仍依赖不同编译配置的离线 oracle；
2. 是否已经实现一个可综合、可在同一 context 内混合执行 IPD32W/FADC24/RAW41 的统一硬件。

主要阅读和交叉核对：

- `docs/104_DATE审稿人第二轮_FADC24后评估_20260718.md`；
- `results/gatestack_adaptive_csr_fulltop_20260718/report.{md,json}`；
- `rtl_hitflow/gatestack_adaptive_csr_replay_decoder.sv`；
- `rtl_hitflow/gatestack_multihead_decoder_projection_top.sv` 的 generate 分支；
- `rtl_hitflow/gatestack_single_context_execution_top.sv` 的参数和接线；
- `sim_hitflow/run_gatestack_adaptive_csr_fulltop.sh`；
- Adaptive 向量生成、报告汇总、顶层 TB 和专用 SVA；
- 已有 Icarus、Verilator/SVA 和 Yosys 日志。

证据分档如下：

| 等级 | 含义 | 本轮可支持结论 |
|---|---|---|
| `[prof]` | H67 profile/trace 软件统计 | workload 动机、格式容量和分布 |
| `[RTL]` | RTL 仿真、SVA、整数金参考 | 被测配置的功能、计数和周期 |
| `[RTL+输入清单]` | RTL 通过，格式数量由向量 manifest 提供 | 同 context 混合格式的有效覆盖，不等于硬件内部统计 |
| `[结构代理]` | Yosys generic cell/memory | 逻辑复杂度趋势，不是物理面积 |
| `[model]` | 周期求和或系统预算 | 设计筛选，不是芯片实测 |
| `[PPA]` | 目标库、PVT、SRAM macro、STA、mapped SAIF | **当前仍缺失** |

## 二、总体审稿结论

### 2.1 第三轮判定

| 项目 | 结论 |
|---|---|
| 推荐意见 | **Weak Reject（弱拒稿）** |
| 综合评分 | **2.8/5** |
| 当前接收风险估计 | **约20%至25%**，仅表示稿件成熟度，不是会议统计预测 |
| 审稿信心 | 高 |
| 相比第二轮 | 有实质提升，但不足以跨过接收线 |
| 当前最准确定位 | 支持每 head 异构 CSR 流分派的统一稀疏投影前端原型 |
| 仍不能使用的定位 | 完整自适应 GateStack、完整 H67 encoder accelerator、已签核 ASIC |

严格结论：

> 第二轮“没有一个可综合的混合 decoder”缺口已经关闭；“每 head 格式只能通过不同编译配置选择”也已经关闭。但“根据运行时 workload 自主构建和选择格式”并未关闭，当前硬件只是解析上游已写入的 magic 并转发到两个 decoder。更重要的是，Adaptive CSR 回归显式禁用 descriptor residency，而现有 warm replay 合同仍是 IPD32W 专用布局。因此 Adaptive CSR 尚未与 GateStack 最核心的跨 tile 驻留机制形成最终架构。

### 2.2 “离线 oracle/无运行时混合格式”缺口关闭矩阵

| 子问题 | 判定 | 证据与限定 |
|---|---|---|
| 同一可综合模块是否同时包含 IPD32W/FADC24 decoder | **已关闭** | `[RTL+结构代理]`，Adaptive 叶模块同时实例化两个 decoder，Yosys 为 1496 generic cells/2 memories |
| 格式是否可在每 head 开始时切换 | **已关闭** | `[RTL]`，缓存 word0，按 magic 锁定 child 直到 done |
| 同一 context 是否实际交错 IPD/FADC | **已关闭** | `[RTL+输入清单]`，11 IPD+13 FADC 用例零 mismatch/error |
| 同一 context 是否覆盖 RAW fallback | **已关闭** | `[RTL+输入清单]`，11 IPD+12 FADC+1 RAW 通过；RAW 由顶层 route 选择，不经 Adaptive CSR 叶模块 |
| 是否不再依赖逐 stage 编译不同 decoder | **前端层已关闭** | 四 stage 都使用 `CSR_FORMAT_FADC24=2`；但每 stage 仍用不同 `HEADS` 重新 elaborate 完整顶层 |
| 格式选择策略是否由运行时硬件执行 | **未关闭** | 四 stage 格式由 runner 选择不同预生成向量，mixed 向量按 head 奇偶交错 |
| 是否有 on-chip builder 生成 IPD/FADC/RAW | **未关闭** | 顶层接收已编码 payload，未计 builder 周期、面积和能量 |
| Adaptive CSR 是否与 descriptor residency 组合正确 | **未关闭，存在合同冲突** | runner 强制 `GATESTACK_NO_RESIDENCY`；planner 的 warm token offset 是 IPD32W 专用公式 |

因此，上述缺口的正确结论是：**编译期oracle和混合解码缺口已关闭，格式决策、格式构建和residency系统缺口未关闭。**

## 三、主要审稿发现

### 3.1 【高】运行时“选择”实际是运行时“分派”

Adaptive decoder 在 `ST_PEEK` 接收首字，仅当低 16 bit 等于 `16'h4641` 时选择 FADC24，否则转给 IPD32W。有效 IPD32W 的 magic 为 `16'h4753`，因此已定义两种格式没有 magic 冲突；无效 magic/version 会由子 decoder 报错。该分派机制在当前合同下是合理的。

但硬件没有根据 term count、fanout、payload bytes 或预测 decoder 周期做决策。决策由软件向量生成器提前完成：

- 四 stage 主实验直接给 S0-S2 IPD 向量、S3 FADC 向量；
- mixed 用例按 head 奇偶性选择，不是硬件友好性决策器；
- FADC 向量生成器只在 FADC 与 RAW 之间以“能否放入 slot”选择，没有在 IPD 与 FADC 之间评估周期/EDP。

这不否定统一 decoder 的 RTL 贡献，但论文用词应限制为 **header-steered heterogeneous CSR replay**，而不是“online workload-adaptive format selection”。

### 3.2 【高/可能致命】Adaptive CSR 与现有 residency 合同不兼容

当前顶层默认 `ENABLE_RESIDENCY=1`，同时允许 `CSR_FORMAT_FADC24=2`。但 Adaptive runner 在 Icarus 和 Verilator 中都显式定义 `GATESTACK_NO_RESIDENCY`。

从代码合同看，这不只是“尚未测试”：

1. 现有 planner 用 `2 + ceil(term_count/2)` 计算 warm replay 的 token 起始 word，这是 IPD32W 的两个 64-bit header 加两 term/word 布局；
2. FADC24 descriptor 为 24 bit/term，destination 区还可能是 8-bit list 或 162-bit bitmap；
3. 现有 resident joiner 按缓存 descriptor 加连续 token byte 流回放，没有 FADC bitmap/list 格式信息；
4. cache entry 只保存 gate/lane/destination count，没有保存 FADC destination mode 和正确 offset。

因此，如果直接以默认 residency 打开 Adaptive CSR，冷 tile 可以解码并填 cache，但 warm tile 可能以错误 offset/错误 destination 格式进入 resident route。本轮未运行该非法组合，因此不直接声称已观测到数值 bug；但已足以判定当前参数合同没有被安全冻结。

这是当前最重要的架构缺口，因为跨 tile residency 是 GateStack 原主贡献之一。不能一边以 residency 作为论文主线，一边只在禁用 residency 时验证 Adaptive CSR。

### 3.3 【中高】首字分派在每次 replay 重复付出约 2 cycle

与相同单格式、无 residency 的回归比较：

| Stage | 单格式周期 | Adaptive 周期 | 增量 | session | 每 session 增量 |
|---:|---:|---:|---:|---:|---:|
| S0 IPD | 2455 | 2473 | 18 | 9 | 2.00 |
| S1 IPD | 1729 | 1802 | 73 | 36 | 约2.03 |
| S2 IPD | 22459 | 22751 | 292 | 144 | 约2.03 |
| S3 FADC | 169703 | 170831 | 1128 | 576 | 约1.96 |

这与 wrapper 的 `PEEK -> child START -> RUN` 结构一致。当 residency 关闭时，同一 head 在每个 output tile 都重新 peek magic，因而分派开销随 replay 数放大。

更合理的最终实现是在 payload commit/build 时解码一次 format，把 format 作为受 tag 保护的 slot/cache metadata，PLAN 直接分派。这可以同时解决重复 peek、format-aware residency 和硬件格式计数问题。

### 3.4 【中高】“实际识别格式”的报告计数不是 RTL 内部计数器

汇总脚本对 S0-S2 直接标记 IPD32W，对 S3 直接标记 FADC24；mixed 用例的 11/12/1 和 11/13/0 来自向量 `manifest.json`。Adaptive RTL 只求和两个 child 的 head/term/event 计数，没有独立输出 per-format selection counter。

当前良性 payload 如果路由错误通常会触发 child protocol error，因而零 error 与正确数值对路由正确性有强支持。但论文仍应把格式数量标为 `[RTL+输入清单]`，不应写成“芯片运行时统计”。下一版应增加 IPD/FADC/RAW 选择、重复 peek、非法 magic 和 fallback 计数器。

### 3.5 【高】`1.407x` 是同一前端配置的 trace-bundle 周期，仍不是整网或自主适应策略收益

当前统一前端四 stage 周期和为 `197857`，相对 GateStack `278388` 为 `1.407x`，相对 IPD-no-residency `285765` 为 `1.444x`。这比第二轮 `1.427x` 离线编译配置上界更可实现，因为已经计入双 decoder wrapper 的分派周期。

但该数字仍有严格限定：

- 每 stage 只是 sample0/B0/window0；
- 四 stage 分别以 `HEADS=3/6/12/24` 编译和运行，不是一次 full-encoder schedule；
- S0-S2 与 S3 的 payload 格式仍由 runner 选定；
- Adaptive 关闭 residency，而 GateStack 基线含 residency，因此这是最终子系统净对比，不能归因为单一格式机制收益；
- 权重和 bias 是候选 dyadic INT8 合同，未通过 valid825；
- 不包含 builder、SCS attention、ATLIF、skip/residual、外存和 stage 调度。

因此可以写“在四个被测真实位级窗口的 trace bundle 上，统一 Adaptive CSR 前端相对当前 GateStack slice 周期和为 `1.407x`”；不得写“H67 加速 `1.407x`”或“运行时策略收益 `1.407x`”。

### 3.6 【中高】同 context 混合覆盖有价值，但不是策略评估

24-head S3 中的两个用例均通过双模拟器：

| 用例 | 格式 | Verilator 周期 | terms | 结果 |
|---|---|---:|---:|---|
| mixed+RAW | 11 IPD + 12 FADC + 1 RAW | 263303 | 30960 | mismatch/done_error/protocol=0 |
| mixed CSR | 11 IPD + 13 FADC | 167665 | 12888 | mismatch/done_error/protocol=0 |

这两个用例已足以关闭“同 context 不能切换格式”的功能缺口。同时它们也暴露出策略的重要性：仅1个 RAW 高扇出 head 就使用例周期从 `167665` 上升到 `263303`。但“用 FADC 替代 RAW”是向量生成器的离线操作，不是 on-chip policy 的结果。

因此这是很好的**功能覆盖和架构动机**，尚不是 adaptive policy 的性能证据。

### 3.7 【中】Adaptive wrapper 的专用 SVA 尚未覆盖分派不变式

当前专用 SVA 覆盖 descriptor/term/event/done 反压稳定、event mask/count、head-last 和 sticky error。这些是有价值的输出协议性质。

但没有直接检查：

- 每 session 仅启动一个 child；
- `select_fadc_q` 在 start 到 done 期间保持不变；
- 缓存首字恰好转发一次；
- 未选 child 不接收 word/ready/done；
- unknown magic、wrong version、word0 index 非 0、word0 即 last 和中途 reset；
- 连续多 group、多随机 seed 和格式快速切换。

当前 good-path 双模拟器结果较强，但还不能称为 adaptive control 的验证签核。Icarus/Verilator 周期仍存在 1至4 cycle 差，功能计数一致，不影响功能 claim，但不能声称双工具 cycle-exact。

### 3.8 【致命】目标工艺 PPA 缺口没有任何实质变化

Adaptive 叶模块的 Yosys 结构为 1496 generic cells 和 2 个 logical memory。相比第二轮叶模块代理，这表明统一前端基本上付出了 IPD decoder + FADC decoder + wrapper 的成本，并未发生显著逻辑共享。

但仍没有：

- target `.db/.lib`、PVT 和 operating condition；
- 500 MHz WNS/TNS 和关键路径；
- descriptor/head-slot/AccTile SRAM macro 舍入；
- mapped SAIF 注释率、动态功耗、时钟功耗和 leakage；
- physically-stripped IPD/FADC/Adaptive 基线；
- mapped netlist LEC/Formality。

因此无法回答“同时保留两个 decoder 是否值得”。这仍是 DATE 硬件稿的直接拒稿项。

### 3.9 【高】部署精度和 full-encoder 系统边界仍未闭合

当前真实 Q/K/gate 与 checkpoint 权重进入了向量生成，这比早期 trace-shaped 载荷更可信。但 projection weight/bias 仍是候选 dyadic INT8 合同，尚未在 valid825 证明 AEE/AAE 与 H67 部署一致，也没有闭合 requant、BN folding、饱和、负数移位和 residual/skip scale。

硬件边界仍是 single-context projection slice，没有 SCS attention、ATLIF、skip/residual SRAM、stage 调度和外存。所以新结果不能改写为整 encoder 的速度、能量或精度结论。

## 四、分项评分更新

| 维度 | 第二轮 | 第三轮 | 复审意见 |
|---|---:|---:|---|
| 问题重要性 | 3.8/5 | 3.8/5 | 应用与高扇出稀疏投影问题不变 |
| workload 动机 | 4.3/5 | 4.3/5 | 真实 trace/profile 动机较强 |
| 架构新颖性 | 2.8/5 | 3.1/5 | 统一异构 CSR 流分派已实现，但 list/bitmap 与双 decoder 本身不新 |
| 架构完整度 | 1.8/5 | 2.2/5 | 运行时混合前端闭合，residency/builder/encoder 未闭合 |
| RTL 实现深度 | 4.0/5 | 4.2/5 | 双 decoder、首字缓存、同 context 混合已落地 |
| 验证可信度 | 3.4/5 | 3.7/5 | 六组 full-top 双模拟器+SVA，但缺 selector/error/coverage 闭环 |
| 基线公平性 | 2.2/5 | 2.4/5 | 统一前端周期已纳入，仍缺物理裁剪、residency联合和 head-major |
| PPA 可信度 | 0.5/5 | 0.5/5 | 无变化，仍为致命缺口 |
| 软硬件协同 | 2.8/5 | 3.0/5 | 格式语义与 H67 term/fanout 相联，但 builder/policy 和部署精度未闭合 |
| 系统边界 | 1.6/5 | 1.6/5 | 仍为 projection slice |
| 可复现性 | 3.8/5 | 4.0/5 | runner、向量、manifest、报告链清晰 |

评分上调来自“可综合统一前端与同 context 混合已实现”，而不是因为 RTL 文件数增加。PPA、系统边界与部署精度未变，因此总评仍不能进入 accept 区间。

## 五、RTL与实验 Claim 成立性

### 5.1 当前成立

| Claim | 证据与限定 |
|---|---|
| 一个可综合 Adaptive CSR 叶模块同时包含 IPD32W/FADC24 decoder | `[RTL+结构代理]`，Yosys 结构可读 |
| 每个 CSR head 可按 word0 magic 在运行时分派到 IPD/FADC | `[RTL]`，选择在 session 期间锁存 |
| 同一 context 内 IPD/FADC 交错回放可正确执行 | `[RTL+输入清单]`，11+13 用例零 mismatch/error |
| 同一 execution top 可同时覆盖 IPD/FADC/RAW | `[RTL+输入清单]`，RAW 经独立 route，11+12+1 用例通过 |
| 四个被测窗口在 Adaptive 参数下均零 accumulator mismatch | `[RTL]`，仅 sample0/B0/window0 和候选 INT8 合同 |
| 被测 trace bundle 周期和相对 GateStack 为 1.407x | `[RTL周期求和]`，不是整 encoder、不是全数据集 |
| Adaptive wrapper 引入约 2 cycle/session 分派开销 | `[RTL]，由同格式对照周期分账 |
| Adaptive 叶模块为 1496 generic cells/2 logical memories | `[结构代理]`，不是面积 |

### 5.2 仅能条件成立

| Claim | 尚缺条件 |
|---|---|
| Adaptive CSR 关闭了离线 oracle | 仅关闭不同编译 decoder 的 oracle；尚缺统一 builder/policy |
| Adaptive CSR 是完整 GateStack 架构 | 必须解决 format-aware residency 和 warm replay |
| 按 head 自适应优于按 stage 固定格式 | 需要真实 policy、多样本和同 PPA 比较 |
| 双 decoder 面积代价可接受 | 需要 target-library 面积、SRAM macro 和关键路径 |
| 格式选择降低能量/EDP | 需要包含 builder、decoder、slot/cache、multicast 的 mapped SAIF |
| Adaptive 保持 H67 部署精度 | 需要 valid825 RTL-exact 量化验证 |

### 5.3 当前不成立

- “已实现硬件运行时 fanout-aware 格式决策。”
- “已实现从 H67 活动到 IPD/FADC/RAW payload 的 on-chip adaptive builder。”
- “Adaptive CSR 已与 GateStack descriptor residency 闭环。”
- “`CSR_FORMAT_FADC24=2` 与默认 `ENABLE_RESIDENCY=1` 的参数组合已验证可用。”
- “`1.407x` 是 H67/full encoder/全数据集加速比。”
- “同 context 11/12/1 是硬件策略自主选择的格式分布。”
- “1496 generic cells 是标准单元面积或证明双 decoder 开销很小。”
- “Adaptive CSR 已证明减少功耗、能量或 EDP。”
- “四 stage 使用一个已集成全网顶层完成了连续执行。”
- “已达到 500 MHz、30 FPS 或 DATE 级 ASIC sign-off。”

## 六、剩余致命缺口

| 优先级 | 缺口 | 为什么会导致拒稿 |
|---|---|---|
| P0-1 | 无 format-aware residency，且参数组合未防误用 | Adaptive 尚未与 GateStack 原核心数据流合并 |
| P0-2 | 无 on-chip format builder 和硬件友好 policy | “adaptive”收益仍由离线编码决定，构建成本未计 |
| P0-3 | 无 target-library PPA/SRAM/STA/mapped SAIF/LEC | 无法证明双 decoder 值得，是 DATE 硬件稿直接拒稿项 |
| P0-4 | 无 physically-stripped 公平基线 | 无法分离格式、分派、residency、后端和调度收益 |
| P0-5 | 真实位级回放仅 sample0/B0/window0 | 不能支持全 workload 平均、尾延迟和格式分布 |
| P0-6 | valid825 部署量化未闭合 | 不能把 projection 整数等价扩展为网络精度不变 |
| P0-7 | full encoder 周期、存储、能量和外存分账缺失 | 仍可能被审稿人定位为局部格式/decoder 工作 |

## 七、下一轮优先级

### P0：先冻结最终 Adaptive GateStack，不再增加新格式

1. **冻结 format metadata 所有权**：在 payload commit/build 时校验 magic/version，将 `IPD/FADC/RAW` 存入受 tag 保护的 slot metadata，PLAN 直接分派，避免每 tile 重复 peek。
2. **解决 format-aware residency**，二选一并明确淘汰口径：
   - FADC 也支持正确 destination-mode/offset 的 resident replay；
   - 仅 IPD 允许 residency，FADC 显式 non-cacheable，不得让 planner 误命中。
3. **实现或精确建模 builder/policy**：至少基于 IPD bytes、FADC bytes、RAW capacity 和解码周期的确定性决策；必须把 build 周期、buffer、端口和能量纳入主表。
4. **对非法参数组合 fail-fast**：在 format-aware residency 完成前，`Adaptive + ENABLE_RESIDENCY` 必须在 elaboration 或 admission 阶段被拒绝，不得默认暴露为可用配置。
5. **重跑同 context mixed+residency**：覆盖冷/warm tile、IPD/FADC/RAW、cache hit/miss/non-cacheable、last-use release、abort 和反压，逐 token/lane 零 mismatch。

### P1：把局部周期结果升级为可投稿物理证据

1. 用相同顶层接口生成 physically-stripped IPD-only、FADC-only、Adaptive、Adaptive+residency、RAW-only 和 head-major spill 网表。
2. 冻结 target `.db`、PVT、SRAM macro 和 500 MHz SDC，报告 WNS/TNS、面积、memory rounding 和关键路径。
3. 用同一批真实 trace 生成 mapped SAIF，分开 builder、decoder、slot/cache、multicast/product、AccTile、clock 和 leakage。
4. 对比“双 decoder”与“共享 reservoir/descriptor/event emitter”，只有在 EDP 改善达到预设门槛时才保留共享重构。
5. 完成 mapped netlist LEC/Formality 和必要的门级回放。

### P2：扩展 workload、验证和系统闭环

1. 扩大到多 sample、12 attention block、多 window，报告 per-format 选择、payload、fallback、周期、p50/p95/p99/worst。
2. 在 Adaptive wrapper 增加 selector lock、single-child start、first-word exactly-once、unknown magic/version、中途 reset 和连续 group 的 SVA/coverage。
3. 完成 valid825 RTL-exact 量化精度，冻结 requant、饱和、负数移位和 skip/residual scale。
4. 把 SCS attention、Adaptive GateStack、ATLIF、skip SRAM 和外存纳入统一 full-encoder 周期/能量模型，报告 Amdahl 上限。

## 八、DATE论文贡献表述建议

本轮后可以更克制地表述为：

> 我们实现了一个 header-steered heterogeneous CSR replay frontend，使 IPD32W 和 fanout-adaptive FADC24 payload 能够在同一 context 内逐 head 交错分派，并与 RAW41 exact fallback 共享 GateStack term/event 投影后端。在四个被测 H67 位级窗口的 trace bundle 上，该统一前端在候选 INT8 合同下逐元素零 mismatch，周期和相对当前 GateStack slice 为 1.407x。

该表述必须紧跟三个限定：

- 格式由上游 payload 预先编码，当前是硬件分派而非 on-chip policy；
- Adaptive 回归关闭 descriptor residency；
- 周期和不是 full encoder 或全数据集加速比。

当前不建议把“Adaptive CSR”作为与 GateStack 并列的独立第二主创新。更稳妥的结构仍是：

> **GateStack 是一个面向 H67 final-gate 等价类的精确稀疏投影数据流；Adaptive CSR/FADC24 是其格式异构、容量安全和高扇出 multicast 执行的组成机制。**

只有当 format-aware residency、builder/policy、公平 PPA 和 full-encoder 分账闭合后，这一组合才有可能从“有质量的子系统原型”升级为 DATE 可接收的架构贡献。

## 九、最终复审意见

Adaptive CSR 本轮并非只增加了名词：双 decoder 统一前端、首字锁定、同 context IPD/FADC 交错和 RAW 无损回退都有实际 RTL 与数值验证证据。因此，第二轮对“统一混合硬件未实现”的批评应当更新为已关闭。

但新 RTL 尚未形成最终 Adaptive GateStack：格式仍由离线生成器决定，每次 replay 重复首字分派，residency 与 FADC 布局不兼容，目标 PPA 和整 encoder 证据仍为空缺。这些问题中，residency 合同冲突和无 on-chip builder/policy 是本轮新的首要架构阻塞；无 PPA 仍是会议投稿的致命阻塞。

**当前推荐仍为 Weak Reject，2.8/5。下一轮应停止增加新格式，优先把 format metadata、builder/policy 和 residency 统一到一个可进入公平 PPA 的最终顶层。**
