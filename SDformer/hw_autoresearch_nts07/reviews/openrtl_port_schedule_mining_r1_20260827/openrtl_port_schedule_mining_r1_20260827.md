# 顶会开源 RTL 的端口与调度微结构挖掘 r1

日期：2026-08-27  
状态：`ONE_CONDITIONAL_CANDIDATE / NO_RTL_YET`  
范围：Motion/H67 的 M473/M498/M504、C1/C2/C3 与 FC2 K8。  
证据性质：只读源码审计；未运行任何开源工具链、VCS、DC、PT/PTPX、Formality 或 GPU；未修改 `docs/359`。

## 1. 结论先行

本轮没有发现一个既不撞 prior art、又能绕过冻结 trace 判门而直接开始写 RTL 的新独立贡献。值得继续的只有一个**组合式物理收口候选**：把 M473/M498 的 `64x1152b 1R1W` parent scratch 改成面向单口宏的 parent-value residency frontend，并把以下机制作为一个整体而不是四个“创新点”：

1. 一周期同步 1RW scratch；
2. 两槽有序 response queue；
3. 当前完成 parent 对下一 consumer 的 write-to-read forwarding；
4. 根据 parent out-degree/liveness 省略 dead-parent 写入，并让只被近邻使用一次的值在 queue 中直达 consumer；
5. 只在合法 residual issue 空隙中预取，不引入通用乱序重排。

该候选暂定名为 **M504-PVRF（Port-aware Parent-Value Residency Frontend）**。其中单口仲裁、store-to-load forwarding、两槽 FIFO、dead-value store elision 都是经典结构，不能单独声称 novelty。可投稿的缝只可能是：这些结构与 H67 online subset parent forest、稳定拓扑顺序、exact signed parent/residual reconstruction 和真实 TSMC28 宏组织共同形成的端口感知捕获路径。

M504 目前仍不允许写 RTL。现有 preflight hammer 为 `REVISE, 73/100`：分析器改变了 M473 冻结任务顺序，并把 work-conserving greedy 误称为最优；合法反例 `[1,3,5]` 可在 4 周期完成，而现分析器报告 5 周期。必须先修正成 exact compact-state shortest path 或给可实现策略加 exact oracle gap，再跑全量 frozen trace。

## 2. 为什么这一项值得先测

M473 的 fused 机会点为 389,974,420 product cycles，相对同坐标 bit 为 **1.943581x**，相对 strongest same-budget M468 zero 为 **1.949744x**；但同步 unfused 上界只有 **1.014682x** / **1.017899x**。因此 paper 性能能否成立，几乎完全取决于 parent 数据能否以低物理税、低停顿率及时到达。

同时，M498 parent scratch 的端口税远大于 logic-only RTL：

| 组织 | 证据等级 | 面积 |
|---|---|---:|
| `32 x DP 64x36` exact-capacity fallback | foundry QRT model | 473,034.720 um2 |
| `16 x DP 128x72` over-depth proxy | foundry QRT model | 285,350.640 um2 |
| `9 x SP 128x128`，仅使用低 64 行 | generated foundry views | 78,825.245 um2 |

单口组织相对两个 DP 敏感性点分别少 **83.336%** 和 **72.376%** macro area，但 preferred `16 x DP 64x72` 尚无生成 PPA，所以这些数字只能作为 macro sensitivity，不能称 integrated PPA。M498 logic-only 的绝对面积门仅 44,779.2 um2，更说明隐藏存储税不可忽略。

### 2.1 popcount-bucket bitmap 只能是支撑机制

M473 的 `row_tile=64` 点用 8-lane popcount capture，capture 为 8 周期；product frontend 固定包含 `17 x capture = 136` 周期的 descriptor-order scan。8-bank、128 B/cycle 的 weight DMA 却为 160 周期。若用 17-bit nonempty/popcount-bucket bitmap 把 17-pass scan **全部**消除，非空 task 的 preprocess 节省为：

```text
max(146 + search_rows, 160) - 160 = max(search_rows - 14, 0)
```

冻结 aggregate 为 `search_rows=19,789,148`、`nonempty_tasks=738,054`，因此总 preprocess 节省满足：

```text
19,789,148 - 14 x 738,054 = 9,456,392 cycles  <= saving <= 19,789,148 cycles
```

即使采用绝对上界并假设这些周期完全落在关键路径，389,974,420-cycle M473 局部点也最多变为约 **1.0535x**；实际还会被相邻 compute 与 DMA overlap 进一步隐藏。故 bitmap 可作为低成本 metadata/energy 支撑机制，或帮助 M504 生成确定性预取提示，但不值得单独开发/包装成 RTL contribution，更不能把 Prosperity 模型中的低/零开销 dispatch 假设改名当 novelty。

## 3. 开源微结构与 prior-art 边界

| 工作 | 源码中可核实的机制 | 可借用部分 | 直接撞车的主张 | 对本项目的裁定 |
|---|---|---|---|---|
| FEATHER，ISCA 2024 | output-buffer RMW pipeline；complete sum 绕过 psum read/add；BIRRD 在 reduction 内完成任意 reorder | 把 scratch 读、parent add、写回分拍；对无需驻留的完成值做 bypass | “在 reduction 中隐藏数据移动/重排”及通用 RIR | 只借 pipeline discipline；不开发通用 RIR |
| SNE，DATE 2022 | memory-backed FIFO 写优先，写周期禁读；event group 全 ready 才前进并在暂停时保持 valid/data | fail-closed 单口冲突规则、exactly-once group backpressure | 通用 resident neuron state、事件路由、单口 FIFO | C3/C2 只补覆盖率和 transaction assertion，不新立 RTL |
| ActiveN，MICRO 2024 | SPM 同步延迟读；利用后续操作不占 response port 隐藏返回；request/response identity 与小队列 | 一周期 read response tag、队列 reservation、延迟返回不阻塞无关 issue | 通用 delayed-response LSU、many-core response network | M504 可借时序范式，贡献必须限定在 parent dependency |
| ExSpike，FPL 2026 | full-overlap descriptor 直接禁止 weight read；两槽 elastic FIFO；ready 由 FIFO、fanout、accumulation 完成合取 | descriptor 必须同时抑制 compute 与 weight fetch；两槽解耦作为实现结构 | 通用 event compression、zero-fetch suppression、两槽事件队列；APEC | C2 只做“零 descriptor=零权重事务”闭环；M501 不写 RTL |
| Prosperity，HPCA 2025 | prefix row 先于 suffix；consumer 读取 prefix partial sum；官方模拟器统计 psum read/write | parent 拓扑顺序与 iso-workload 算法机会对标 | product sparsity、prefix-parent reuse 本身 | M504 只能主张 physical port-aware capture，不能主张 parent reuse |
| Phi，ISCA 2025 | multi-window packer 同时检查空间与 partial-sum bank conflict，选择可发窗口 | 冲突计数和 admission 指标设计 | 通用 bank-conflict-aware pack/issue | 禁止另做通用 packer 作为 novelty |
| FireFly-T，2025 arXiv | 宽权重存储、worker/OOO dispatch、implicit layout 与多 lane sparse decoder | bank activity/idle 统计口径 | 通用 bank-conflict avoidance、OOO event dispatch、多 lane decoder | FC2 K8 不做新 OOO bank scheduler |
| LoAS，MICRO 2024 | timestep-inner temporal packing；silent-neuron bitmap/pointer | descriptor 统计与时间打包对标 | 通用 temporal parallelism/bitmap packing | C2 已相邻，不追加新 RTL |

### 3.1 FEATHER 对 M504 的真正启示

FEATHER 的 `feather_controller.v` 将 BIRRD 结果、旧 psum SRAM 返回、加法与写回分拍；当结果已完整时，`bypass_to_scale` 跳过旧 psum 读取/相加。可借鉴的是“先证明值是否仍需驻留，再决定是否触碰 SRAM”的控制原则。不能把 M504 写成 RIR，因为 FEATHER 已把重排隐藏在 reduction 内作为中心贡献，而且本项目无需任意 layout reorder。

### 3.2 ExSpike 对 C2 的强制闭环

ExSpike `weight_top.v` 的 `rd_weight_en <= event_info_vld && full_overlap == 0` 表明：稀疏 descriptor 若只跳过 MAC、不跳过权重读，硬件故事是不完整的。C2 已有 descriptor/bitmap 基建，正确下一步不是再写一个 descriptor RTL，而是对 frozen FC2 trace 给出三组同源计数：accepted nonzero descriptors、weight-bank read transactions、zero descriptors；并用 assertion 证明 zero descriptor 不产生权重请求。

### 3.3 Prosperity 模拟器不能替 M504 关闭端口周期

官方 `simulator.py` 在每个 tile 中以 residual nonzero 数计算 product cycles，并累计 `g_psum` read/write；最终周期为 compute/preprocess 与 DRAM stall 的 max/和。它没有把 output SRAM 的 1RW/1R1W 端口冲突、同步返回、response queue 占用写入 per-cycle recurrence。`sram_config.json` 的 `read-write port=1` 是 CACTI 配置字段，不是计算调度证明。因此 M472 的 2.459487x 只能继续作为官方框架 iso-workload opportunity，不能证明 M504 单口无损耗。

## 4. 唯一条件式新 RTL：M504-PVRF

### 4.1 必须先冻结的数据

全量分析必须逐 task 保留 M473 的 `[sample, operator, chunk, partition]` 扁平顺序，并导出：

- issue-order row ID、input/residual mask、每 row issue beats；
- 每个 consumer 的 parent ID、producer 完成周期和 reuse distance；
- parent out-degree、最后一次使用位置、当前完成值是否可直接 forward；
- 每周期 macro read、macro write、forward、queue enqueue/dequeue、hold；
- read/write 冲突数、同地址可转发冲突数、queue-full stall、producer-not-ready stall；
- dead-parent write 数、single-use-nearby read/write 省略数；
- depth `{1,2,4}` 的 response queue DSE；
- exact oracle、可实现静态策略、work-conserving upper bound 三列周期。

### 4.2 快速判门

先修正 M504 preflight 的两个 P0，再执行 full trace。只有同时满足下列主门，才授权 RTL：

1. 值、parent edge、transaction 和稳定 issue order 全部 exact；`[1,3,5]` oracle=4、work-conserving=5；小规模 exhaustive 与随机 64-row differential 通过。
2. 可实现策略相对 exact oracle 的周期差 `<=1%`；否则论文不能用 oracle 数冒充 RTL schedule。
3. 单口点相对 M473 fused 1R1W 的 Conv 周期开销 `<=5%` 为首选；`5–15%` 只保留为面积/性能 Pareto，且相对 strongest M468 zero 仍 `>=1.50x`。
4. 以两个 DP QRT 敏感性点都报告 macro area；在 preferred DP 生成前不得把 `>=75%` reduction 当 physical admission。
5. liveness bypass 必须至少去掉 `10%` scratch writes 或总 scratch accesses；低于此值只保留 forwarding，不增加 liveness 控制 RTL。
6. integrated macro PPA、能量和系统 speedup 保持 false，直到正式生成 DP/SP views、VCS exact-SHA 与 Synopsys macro-aware closure 完成。

### 4.3 若通过，RTL 边界

RTL 只新增一个 parent-scratch frontend，接口保持 M498 consumer/producer 协议；不改 signed12 parent/residual arithmetic，不改 signed19 resident psum，不改 row matching，不改变 checkpoint。内部只允许：1RW macro adapter、两槽有序 queue、pending-read tag、same-address forward、last-use/dead-parent write suppression 与确定性小状态 scheduler。

论文中建议称其为 **port-aware capture mechanism** 或 **macro-aware parent residency**，而不是新的 product-sparsity 算法、通用 bank-conflict scheduler、RIR 或 event-compression engine。该点适合作为 C1/Conv 主机制的物理收口子贡献；单独不足以支撑 DATE best-paper novelty。

## 5. 其余 RTL idea 的结论

| 候选 | 决策 | 数据依据 | 后续动作 |
|---|---|---|---|
| M504-PVRF 单口 parent frontend | `FASTKILL_THEN_CONDITIONAL_RTL` | DP 宏税大；fused 与 unfused 周期差巨大；现 preflight 尚有两个 P0 | 修 analyzer/order/oracle，跑 frozen full trace，过门后再写 RTL |
| dead-parent/liveness bypass | `BUNDLE_WITH_M504` | 可精确减少 scratch 写/读和端口冲突 | 先量 out-degree/last-use；不单列贡献 |
| generic multi-window bank-aware scheduler | `NO_GO` | Phi/FireFly-T 直接 prior art；控制和 bank 税高 | 只借冲突指标，不写新 RTL |
| FEATHER-style general reorder-in-reduction | `NO_GO` | FEATHER 中心 novelty 直接覆盖 | 仅借 RMW/bypass pipeline |
| C2 新 descriptor/zero-fetch 模块 | `NO_NEW_RTL` | ExSpike/SNE 已覆盖；C2 已有同类基建 | 补 transaction-level assertion 与 access counts |
| FC2 K8 OOO dispatcher | `NO_GO` | FireFly-T 已覆盖；K8 与同带宽 K1x8 已是周期 parity | 继续同容量、同带宽 DC/energy 对比，卖 state collapse |
| C3 新 resident-state/event queue | `NO_GO` | SNE/ActiveN 直接 prior；现有 C3 已验证 | 只做完整状态容量/端口与能量闭环 |
| M501 APEC RTL | `NO_GO_CURRENT_TRACE` | ExSpike APEC 直接 prior；H67 当前二级正激活退化为 support intersection | 除非冻结数据出现 signed/nonbinary 差异，否则停止 |
| popcount-bucket bitmap / 17-pass scan elimination | `SUPPORT_ONLY_NO_STANDALONE_RTL` | 理想绝对上界仅约 1.0535x 局部，且大部分可能被 160-cycle DMA/compute 遮蔽 | 可并入 descriptor metadata 或 M504 hint；不单列创新 |
| attention 新 ε/descriptor 硬件 | `NO_GO_MAINLINE` | attention 仅 0.5889% envelope；Bishop/Phi 等先占位 | 只留附录 Pareto，不占 RTL 队列 |

## 6. 对 DATE 贡献结构的影响

本轮挖掘不会凭空增加一个 headline contribution，但能把现有故事从“算法机会 + logic-only RTL”修成更可信的“机会—捕获—物理税”链：

1. C1：官方 Prosperity 同框架 2.459487x opportunity；本项目稳定 parent forest 与 online subset capture；M504-PVRF 若过门，给出 macro-aware 1RW 实现点。
2. C2/FC2：signed analog event descriptor、零 fetch 抑制、共享 Acc24 partial state；K8 与等带宽 K1x8 不声称加速，只报 state/area/energy。
3. C3：ATLIF rank/phase decoupling 与 exact early-stop/状态访问收口；不把 resident state/event routing 当新颖性。
4. A1/RQTB：无损 attention 辅助点；不得用 0.5889% 子系统包装系统倍率。

如果 M504 fast-kill 失败，应立即停止新的 Conv RTL，保留 C1 的 capture-gap 消融和外部 opportunity 对标。若通过，它仍是“物理可信度与实现巧思”的增强，不应与 C2 4.76x 局部倍率相乘，也不能独自把当前工作升到 DATE best-paper。

## 7. 方法、边界与复核

本审计固定官方仓库 commit，逐文件阅读 RTL/Chisel/Python simulator，不执行仓库脚本。优先寻找用户指定的五类结构：单口冲突隐藏、forwarding/response queue、reduction 中数据移动隐藏、bank-conflict-aware issue、event descriptor/zero-fetch suppression。每个候选按三项裁定：是否有直接 prior art、是否需要新的冻结 trace、是否有足够物理收益支持新 RTL。

局限性：Phi 与 FireFly-T 只核到论文机制，未定位到可冻结的官方 RTL；LoAS 官方作者仓库含模型/分析 artifact、未含 RTL；ExSpike 以仓库 README 标注的 FPL 2026 身份记录。以上缺失不等于作者从未发布 artifact，只表示本轮未找到可审计源码。

结构化结论见 `openrtl_port_schedule_mining_r1_20260827.json`；官方源码 commit/blob 身份见 `SOURCE_IDENTITY.md`；内容 hash 见 `seal.json` 与 `SHA256SUMS`。
