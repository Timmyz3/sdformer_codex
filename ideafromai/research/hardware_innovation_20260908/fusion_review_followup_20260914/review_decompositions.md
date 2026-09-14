# Q1 bitmap、Q2 DA、Q1 字典与 T10 方向：独立复审

2026-09-14。先读原始 SV、TB、输入生成代码，后核原报告。旧实验树只读；本轮没有修改生产目录，也没有启动 GPU。结论支持四个**固定实现点**的负性能结果，未发现本合同内的整数错误；需要修正跨叶输入口径、字典冷启动分母和“强控制”的强度表述。不能据这些结果宣布 BISMO、DA、UCNN 或时间复用家族已经被否定。

|实现|支持|修正 / 未证实|
|---|---|---|
|Q1 bitmap|完整 K864/C96/R8/N96/T10，真实八块核心 87,599→103,908，慢 18.618%；动态转换已付费|只检验 16K、逐位置逐平面的 FSM；作者流水、DMA、指令调度及调优未复现|
|Q2 DA|真实八块 87,599→100,043，慢 14.206%；构表与展开均在 RTL|未比较常驻 LUT、按位置 MAC/DA 选择；“不用乘法”不等于同面积、更低能耗或更快时钟|
|Q1 字典|完整 R8 列的 32 类 sum-first 确实执行；87,599→90,217，慢 2.989%|2.989% 是核心差；两臂额外加载同一份 897 拍 metadata，不是与实际无字典控制的冷服务差；不是完整 UCNN|
|T10 方向|mode2 相对 mode1 多 1,058 拍，真实新增 α 选择为零；raw/J20/I24 完整|局部评分没有首次 anchor 保存税；“强 Δ”可称较强局部控制，不能称完整服务费用最优控制；不同叶 real_6 的 source 不同|

## 本次独立证据

[audit.py](audit_decompositions/audit.py) 用直接卷积索引累加重建 z，没有调用旧 `verify.py` 或读取 RTL 导出的 latent。它重算旧分解组 14 个 fixture 的 53,760 个 raw 输出、方向组 19 个 fixture 的 raw/J20/I24 各 72,960 个值，全部一致。I24 使用 Python 无界整数和独立 `divmod` 的 ties-to-even，避免把 NumPy 的同一溢出错误复制到验证器。另验证 520 组覆盖 signed 宽度 1..13 的极值 DA 向量。[数学结果](audit_decompositions/summary.json)

用 Verilator 4.028 `--cc --exe` 对四个原始顶层重新构建到本目录，共跑 **78 条命令**：48 条线性核命令比较 184,320 个 raw 值；30 条方向完整服务命令比较 raw/J20/I24 各 115,200 个值。主集有确定性 source/weight/result 背压、每进程两次无 reset 命令；新增 anchor 费用反例为无背压冷/暖命令。角落包括随机图外非零污染、K=15/16 跨块、不同空间半字、Q2 整组零和整 rank 零、同类计数 864、signed3 最小值 −4、首 anchor 前零和时间中间零，以及旧非零方向残差/range_guard。均通过；−4 对前三叶是**扩展覆盖**，其原报告仅承诺 Q1∈[−3,3]。[重跑记录](audit_decompositions/results.json)、[anchor 反例](audit_decompositions/anchor_tax_results.json)。本轮未重复旧全量组合，也未重做 EDA。

TB 是真正的 RTL 比对。分解 TB 只配置 source、Q1、Q2、liveK；字典另配置静态类别。它对实际 `result_valid && result_ready` 的 480×8 个输出逐值检查并检查 hold 稳定及状态拍数。方向 TB 由 RTL 请求绝对 source 地址与 FP32 identity，分别检查 raw、实际 J20 和 I24；没有把 Python z、bitmap、计数或方向码喂进设计。[分解 TB 16–45 行][tb-decomp]、[字典配置 28–32 行][tb-dict]、[方向 TB 40–84 行][tb-temporal]。

## Q1 bitmap

**支持。** `BM_PACK` 对全部 864 个 K 写满 40 个 bitmap 银行，包括 dead-K 的零，不依赖复位清空 bitmap；54 个 16bit 块恰好覆盖 K864。三个系数平面在有序 cfg 中形成，`bp_live` 每块首 bit 清/初始化后 OR，其正确性依赖完整顺序配置。`BM_POP` 实现 `pc0+2pc1−4pc2`，负权符号平面正确；BM_STORE 的行/半字映射与后端一致。[转换与扫描 150–184 行][bm-pack]、[cfg 118–125 行][bm-cfg]、[算术 69–103 行][bm-alu]。新随机 padding poison 与 −4 全密向量均通过。

**费用与资源支持。** 真实八块的 6,912 个构造拍、17,280 次 bitmap 字读、9,012 次 128bit 平面读/发射都是真实 FSM 状态；没有把原生源直接变成免费预排布输入。source bitmap 4,320B、第二份 Q1 bitplanes 2,592B、8 棵 popcount16，以及 40-bank bit-write 权限均被资源表披露。普通 Q1 一次仅读 24bit，不能用混合 `weight_words` 估字节或能耗。[资源表 128–174 行][bm-res]。

**未证实。** 同一模块声明了额外资源，但 mode14 不使用 popcount/位平面；它证明共享资源集合中的两个调度结果，不证明最佳资源使用或各自综合后的面积。当前按 P/T 外循环重复读权重平面，也没有 fetch/execute/result 流水。BISMO 原文 §II 和 §III-A 的 AND/popcount/shift/sign 已被借入；§III-A/C 的 DMA、同步 FIFO、指令队列和跨阶段调度没有被本地重现。因此不能写成“完整 BISMO 对本任务无效”。[BISMO 原文](https://arxiv.org/html/1806.08862v1)

## Q2 DA

**支持。** 两组 R4 的 16 项 LUT 由最低置位的父项加一个 Q2 系数构造，signed18 足以容纳四个 signed16 系数之和 [−131072,131068]。最小公共 signed 宽度包含符号位；低位加、最高有效位减，零 LUT 向量不发射。DA_POS 将失活 rank 置零后编码，避免由不参与输出的 z 放大宽度。[宽度/符号 52–66 行][da-width]、[构表和执行 181–216 行][da-build]。按 signed13 全范围，全部低位的最坏正向绝对和 ≤8×32768×4095=1,073,479,680，signed32 容纳；符号修正后逐 rank 仍在合法范围。

**运行时转换支持。** 每 N8 真实 32 构表拍，全 tile 384 拍；八真块多 3,072 构表拍、2×2,484 读 z/编码拍，以及 21,960−17,556 额外发射，合 **12,444** 拍，与结果差一致。候选仍实例化原乘法器，因此只能说候选路径没有 Q2 multiply 发射。[DA_BUILD/DA_POS/DA_ENCODE][da-build]、[资源与运行时构表][da-res]。

**强控制未覆盖。** mode14 已有 rank/position 支持、Q2 零向量跳过、完整 R8×N8 cache/OS，并非弱稠密控制；但是 mode15 整体切到 DA，未让同一位置选最便宜的直接 MAC，零输出块仍构表。固定权重的 LUT 又在每 tile 每 N8 重建，只有 576B 单块容量，无跨 tile 常驻表测试。MathWorks 官方方法说明给出 LUT/shift/符号修正和 partition/radix 选择；没有运行 HDL Coder，也没有复现它的生成器优化。[官方 DA 方法](https://www.mathworks.com/help/hdlcoder/ug/distributed-arithmetic-for-hdl-filters.html)

## Q1 字典

**支持。** 离线仅按 Q1 产生静态 metadata：24bit 完整 R8 列键、频数≥2、前32类，其余 direct。运行时 source 仍由 native 窗 gather；`C_ADD` 以 bit10 carry-cut 更新两 unsigned10 计数，最多 864，`G_MAC` 用正的 count×有符号代表系数退休并保留另一 z 半字。count clear、group-live、三块扫描与逐非零位置退休都有拍数，不是理想操作计数。[静态编译 6–15 行][dict-prepare]、[计数及退休 185–225 行][dict-count]、[乘法与 carry 79–101 行][dict-alu]。本轮 864 同类、负权与 −4 扩展均通过。

**冷启动解释修正。** TB 不论 mode 都加载 32 个代表、864 个 class 和 1 个尺寸，所以两臂首配置均 4,258 拍；核心慢 2.989% 正确。若比较实际旧控制的 3,361 拍配置，真实八块冷合计为控制 `87,599+8×3,361=114,487`、候选 `90,217+8×4,258=124,281`，差 **9,794 拍（8.555%）**。这是明确包含各自配置的算术口径，不是新整机实测。Python 建类时间尚未测量；仅在权重静态、离线编译的合同下可以不计入每 tile 周期。[TB 23–32 行][tb-dict]

**作者机制与本地机制应分开。** UCNN §III-A 利用单滤波器中的相同标量权重，§III-B 在少量滤波器间复用重叠 activation group，§IV 支付/压缩 indirection 并作空间向量化。本地是“完整八 rank 同列才共享”的窄特例，没有实现其层次 indirection 数据流或作者全架构。真实完整列有793种不能推出局部重复不足；本轮同一 Q1 分为四个 R2 后，非零键仅 **26/21/28/29** 种。[UCNN 原文 §III/IV](https://arxiv.org/html/1804.06508v1)、[独立键计数](audit_decompositions/summary.json)。是否值得拆 rank，取决于新增计数写口、bank 冲突和失去 R8 广播的费用，尚未证明。

## T10 方向

**支持数值与主要负结果。** EDIFF 用共享八条宽 ALU 求差，再显式检查 [−4096,4095] 才写入 signed13；原 z/previous/anchor 的生命周期分开，每 P 首时刻重启，前一输出通过保留 acc 复用，anchor 仅在需要时保存。BASE_READ 对 ±2 的 shift 与负方向的独立 BASE_NEG 都实际执行。[编码 203–243 行][temp-encode]、[Q2/anchor 257–284 行][temp-output]。新增 −4/时间零间隙通过，旧 range_guard 与非零 residual 也重跑通过。原三臂 raw/J20/I24 的 19 个 fixture 金标准已独立重算。

真实新增缩放方向没有选择，mode2 多 529 个 EDIFF/EVAL 配对，恰多 1,058 拍；这支持停止该固定字母表扩展。mode1 已具备 prev 驻 acc、相同 prev/anchor 候选消重、有限 rank_cost 与 lower-bound 剪枝，确实强于清零后再读取 prev 的控制。负结论应限当前 native producer→线性 Q2→原 I24 的固定接口；本地没有复现任何论文的整层在线分解/跨层调度。

**“完整费用强控制”需要修正。** `candidate_cost` 只包含残差 MAC 和 base 读/取负；首次启用本 P 的 anchor 需要全12个 N8 各保存一拍，而这一边际12拍不在选择评分里。所有真实总拍仍会收费，因此不是性能计数伪造，是选择器的优化缺口。[费用 83–94 行][temp-cost]、[anchor-needed 237–238 行][temp-encode]、[保存 271–284 行][temp-output]。

本轮 [anchor_tax.py](audit_decompositions/anchor_tax.py) 从合法 source/Q1/Q2 构造 `rank_cost=[12,1,0,…]`，每 P 的 `z0=z2=(1,1,0,…)`，其余 t 为零。局部评分对 t2 选择 anchor：省13 MAC、读 base12，看似省1；该 P 首次保存再付12。完整 RTL 全4P显示 MAC **104→52**、base读取 **0→48**、保存 **0→48**、encoder **0→88**，冷总服务 **11,369→11,501（多132拍）**，warm 同样差132拍。即使忽略编码，该选择本身仍多44拍。无需再扩字母表就存在可检验的控制补强点。此反例不推翻“mode2 在真实八块不赚钱”，也不声称完整费用选择已被实现。

**输入与服务边界修正。** 分解 fixture 来自旧 `dense_q16.npz` 的 source，方向 fixture 来自 `consumer_first8.npz`；Q1/Q2/origin 相同，但 `real_6` 有 **19/1536 个 source 词不同**，其他7个相同。[分解输入来源 7–9/55 行][source-decomp]、[方向输入来源 5–7/41–43 行][source-temp]、[逐输入独立对照](audit_decompositions/summary.json)。因此分解后端 17,556 MAC 与方向 full 的17,604 MAC不能直接作跨叶融合增量；后续先统一一份原始 fixture。

分解第二命令复用同一 source/config，只证明同输入无 reset 重入；方向 wrapper 的 warm 会重新装入1536个 source、origin 和480个 identity，但 TB 仍重放同一 tile。两者都不是多 tile / 多源切换覆盖。外部 `source_allow/weight_allow` 在分解核只门控本地读；方向 top 才有真实请求/背压。全部资源是 union 模块级共同权限，未证实单独裁剪后等面积、Fmax、能耗或整帧速率。[分解 TB 23–32 行][tb-decomp]、[方向 TB 25–49 行][tb-temporal]、[方向 wrapper 304–323 行][temp-top]。

## 可接下一轮的三个接口

1. **固定容量 R2 权重分组接 native source。** 用四个 R2 分组取代完整 R8 字典键，先固定32类上限/组，输入仍一次 native gather；真实26/21/28/29类说明命中不是臆测。必须把四路 class metadata、计数银行、共享八 ALU 的仲裁、逐组退休和 fallback 全算进去；强控制是同一资源下当前 dual-P direct 与 R8 字典。只检验是否增加的分组共享覆盖足以支付更多状态，不以 UCNN 原理本身申报新机制。

2. **按 N8/位置选择 MAC 或 DA，并显式管理静态 LUT 生命周期。** 两者已在同一模块，DA 各位置可用编码后的真实发射数与非零 rank 数比较，编码本身、构表是否摊销都收费。先保留576B容量跑付费 hybrid 控制；若要固定12块 LUT 常驻，明确增加至6,912B与配置/填表费用并让控制同权。真实 DA 比直接多4,404发射，另有8,040构造/展开拍，接口痛点清晰。未测收益，不把免费择优下界写成 RTL 结果。

3. **把 T10 的 anchor 启用与候选提交联成完整 P 级费用决策。** 保留现有小字母表，记录每 P 的候选总节省和首次 anchor 保存税，连同编码/必要 z 写回作显式 fallback；对本轮132拍反例必须不再把局部 MAC 节省报成服务节省。若需要暂存选择/残差或两遍扫描，新增读写全部收费；控制同时包含 full、prev 驻 acc 和原局部评分。只有在统一后的真实输入上净省才继续接其他融合。

这些是对本轮已检查 RTL/计划未实施接口的建议，不是新颖性证明；本轮没有借主观创新分替代功能、费用与作者架构边界分析。

审阅后续已执行第1项固定32类R2接口：128条RTL命令全过，但真实核心87,599→113,475（慢29.539%），停止该布局；见[完整后续报告](pair_dictionary/REPORT.md)。第2、3项仍未实现。

[tb-decomp]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_ten_trials_20260914/decompositions/q1_bitplanes/tb.cpp:16
[tb-dict]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_ten_trials_20260914/decompositions/q1_dictionary/tb.cpp:23
[tb-temporal]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_ten_trials_20260914/temporal_direction/tb.cpp:25
[bm-pack]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_ten_trials_20260914/decompositions/q1_bitplanes/decomp_core.sv:150
[bm-cfg]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_ten_trials_20260914/decompositions/q1_bitplanes/decomp_core.sv:118
[bm-alu]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_ten_trials_20260914/decompositions/q1_bitplanes/decomp_core.sv:69
[bm-res]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_ten_trials_20260914/decompositions/q1_bitplanes/resource_contract.json:128
[da-width]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_ten_trials_20260914/decompositions/q2_da/decomp_core.sv:52
[da-build]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_ten_trials_20260914/decompositions/q2_da/decomp_core.sv:181
[da-res]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_ten_trials_20260914/decompositions/q2_da/resource_contract.json:127
[dict-prepare]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_ten_trials_20260914/decompositions/q1_dictionary/prepare_dictionary.py:6
[dict-count]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_ten_trials_20260914/decompositions/q1_dictionary/decomp_core.sv:185
[dict-alu]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_ten_trials_20260914/decompositions/q1_dictionary/decomp_core.sv:79
[temp-encode]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_ten_trials_20260914/temporal_direction/temporal_core.sv:203
[temp-output]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_ten_trials_20260914/temporal_direction/temporal_core.sv:257
[temp-cost]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_ten_trials_20260914/temporal_direction/temporal_core.sv:83
[temp-top]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_ten_trials_20260914/temporal_direction/consumer_stream.sv:304
[source-decomp]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/r0_stream_fusion_20260914/integer_factor/prepare.py:7
[source-temp]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_ten_trials_20260914/temporal_direction/prepare.py:5
