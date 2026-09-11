# R3Q01 — circuit-level candidates (independent)

Focal: After official AT-LIF `{0,θ}` with `θ` absorbed into next `W` (spike path is binary GeMM), what **one** circuit-level mechanism could be a TCAS-II 5-page letter for event-camera 2D optical-flow SNN-Transformer co-design?

Locked (copy, not paraphrase as identity): `o=θH(m-θ)∈{0,θ}`; inference `W←θW`; transmitted spikes `{0,1}`. Residual/PED/I24 is a **separate continuous tensor**, not analog AT-LIF amplitude shared by two MACs. Pre-threshold `T=10` PSN mix `H=WX` is continuous then threshold then absorb.

A (must copy fully as对照, never as this letter’s title): Prosperity product-sparsity; GustavSNN CPTB/NRV as GeMM psums not LIF membranes; LoAS FTP; FireFly-S Bitmap AND; Phi PWP; Bishop SAC/AAC; FireFly-T dual-engine overlay (sparse spike×W + binary AND-PopCount attention) with pre-neuron membrane residual; DATE 2025 hybrid = dense core for non-binary INPUT layer + sparse cores for rest (layer-split, not same-producer dual last-use); Fang CICC 3D AC spiking-only (against hetero MAC+SNN cores); ESTU TCAS-II binary SSA skip + classification mW (same-journal collision); PSN+da4ml for pre-threshold mix; SDT membrane-shortcut; Spike-IAND replaces residual ADD with IAND to stay all-spike (negative if we keep PED); SENECA/TrueNorth/plane-fit FPGA/EventShiftFlow already event-OF HW; ASNA-Flow abstract already claims OF spatial-locality sparse computing; FlexSpIM CIM = A then STOP as title; SpikePool is GPU max-pool attention not silicon; live full-domain proj BN 10×96×120×160 is hygiene not X.

Observations (not findings): source always-ready can drop (e.g. 6938→5354 fused writeback or 6914→6170 source_service — different tables, not additive); long-BP both 8088; wait-class histogram missing; serialized full-chain −6.4% is NOT closed same-port net service; FP preview+sn2 ~49% of that construction; delayed-V arithmetic_saving=0; BN hardware not closed; lifting AEE +0.013 fails +0.005 vs ordinary 1.219801338. Gates: same-port net service ≥15%; AEE abs ≤1.259 and relative ≤+0.005. No OpenROAD-as-PPA; no component-ratio FPS; no CIM title; no first-OF-HW; no deleting 35 RNE as title; one mechanism.

Perspective: digital circuits / datapath (comparators, last-use retirement, typed FIFOs, dual-width accumulators). Origin/stage/status filled per idea. No scores, no winner.

---

## R3Q01-I1 Typed dual last-use retirement scoreboard (binary GeMM vs PED/I24)

- **ID:** R3Q01-I1
- **Statement:** 在 θ 吸入下一层 W 之后，用一块带双 typed occupancy 计数器的 last-use scoreboard（比较器在两个计数都到 0 时才释放 producer 行），把同一行的二值脉冲 GeMM last-use 与连续 PED/I24 last-use 拆开记账，而不是把两条消费者合成一次 fused writeback。
- **Assumptions:** 残差/PED/I24 是独立连续张量，不是 AT-LIF 的模拟幅值被两个 MAC 共用；DATE 2025 hybrid 是 INPUT 稠密核 vs 其余稀疏核的 **layer-split**，不是同一 producer 的 dual last-use；FireFly-T 的 pre-neuron membrane residual 不是 PED；source always-ready 从 6938→5354 的 fused writeback 至少部分来自把两类 last-use 焊在同一写回拍。
- **Predicted observation:** 相对 fused writeback，same-port net service 达到 ≥15%（按「last-use 开火才占源端口」计，不把 serialized full-chain −6.4% 当净服务）；source always-ready 从 5354 类回升，且 ordinary AEE 停在 1.219801338 的 +0.005 门内（不做 lifting）。
- **Disconfirming evidence:** wait-class 一旦补上，8088 的 long-BP 主因不是 dual last-use（例如纯 BN-barrier 或源空）；拆 scoreboard 后 net service 仍 <15%；或审稿人认定 FireFly-T dual-engine overlay + membrane residual 已覆盖「双消费者」机制。
- **Uncertainties:** 6938→5354 与 6914→6170 分属不同表、不可相加；wait-class histogram 缺失，8088 尚未可证伪；事件相机 2D OF 的突发是否让两个 occupancy 长期一满一空。
- **Origin:** ai-assisted
- **Stage:** independent
- **Status:** candidate

---

## R3Q01-I2 Anti-fused 双 retirement 端口（1-bit spike FIFO vs I24 PED FIFO）

- **ID:** R3Q01-I2
- **Statement:** 在源 SRAM 端口旁放两个独立 valid/ready 的 last-use retirement 端口——窄口只退 1-bit 脉冲 bitmap，宽口只退 I24 PED——禁止把两类写回 mux 进同一拍 fused writeback。
- **Assumptions:** fused writeback 是 always-ready 掉到 5354 的电路原因，而不是 GeMM 算术本身；脉冲路径吸 θ 后是 `{0,1}`，PED 必须保持连续加法（Spike-IAND 把 residual ADD 换成 IAND 以保持全脉冲，在保留 PED 时是负对照）；Prosperity product-sparsity / FireFly-S Bitmap AND / Phi PWP / Bishop SAC/AAC 仍只是二值 GeMM 侧的 A。
- **Predicted observation:** 关掉 fused writeback 后，always-ready 掉点主要消失，且该恢复 **不能** 与 6914→6170 的 source_service 掉点做加法；same-port 占用按端口分别记账后出现 ≥15% 净服务；AEE 不变。
- **Disconfirming evidence:** 双端口后 5354 类掉点仍在（瓶颈在算力或长 BP 8088）；serialized −6.4% 在双端口下原样存在，说明它从来不是写回融合；双端口面积/布线在 5 页信里变成第二条机制。
- **Uncertainties:** 两张掉点表能否在同一 testbench 对齐；事件 OF 的 PED 行宽是否让宽口反而饿死窄口；与 I1 scoreboard 是否其实是同一机制的端口实现。
- **Origin:** ai-assisted
- **Stage:** independent
- **Status:** candidate

---

## R3Q01-I3 拍内 stall-cause 比较器 + 带标签 credit 环

- **ID:** R3Q01-I3
- **Statement:** 在 source-ready 比较器上打 3-bit stall tag（`source_empty` / `spike_consumer_BP` / `PED_consumer_BP` / `BN_barrier` / `typed_FIFO_full` / `other`），credit 环按 tag 改仲裁，从而让两端都是 8088 的 long-BP 变成可证伪的电路状态，而不是事后软件直方图。
- **Assumptions:** 当前 wait-class histogram 缺失，G1 typed last-use 在 8088 切开之前不能当标题；BN 硬件未闭合，live full-domain proj BN 10×96×120×160 是 hygiene not X，因此 BN_barrier 只作为 tag，不作为信件贡献；tag 比较器本身必须改变发放策略，否则只是观测器。
- **Predicted observation:** 8088 被拆成可加总的 wait-class；若主质量是 `PED_consumer_BP` 或 `typed_FIFO_full`，则 typed 仲裁相对无标签 round-robin 给出 ≥15% same-port net service；若主质量是 `BN_barrier`，则本机制作为 OF 信件失败（退回 hygiene）。
- **Disconfirming evidence:** tag 不改变发放，8088 数字不变；两端 long-BP 仍同为 8088 且各类均匀，无政策可调；ESTU TCAS-II 同刊分类 mW 信被审稿人拿来要求我们改做分类 SSA skip 而不是 OF stall tag。
- **Uncertainties:** 8088 是否为计数饱和/对齐假象；FP preview+sn2 约占那套 serialized 构造的 49%，tag 可能把 FP 预览 stall 误标成 PED_BP。
- **Origin:** ai-assisted
- **Stage:** independent
- **Status:** candidate

---

## R3Q01-I4 阈值边界 dual-width 累加器（T=10 连续 mix → 吸 θ 后窄 psum）

- **ID:** R3Q01-I4
- **Statement:** 用一组 dual-width 累加器：`T=10` 的 PSN mix `H=WX` 走宽字（连续），比较器过阈值并吸 θ 后，脉冲 GeMM 只走窄 psum/AND-PopCount 宽度，PED/I24 另走宽字 FIFO——宽度提升只发生在阈值边界，绝不是两个 MAC 共享模拟 `o=θ`。
- **Assumptions:** 预阈值 mix 连续、过阈后传递 `{0,1}` 是锁定身份；GustavSNN CPTB/NRV 是 GeMM psum 不是 LIF 膜，可作为窄侧 A；PSN+da4ml 管预阈值 mix 是 A；delayed-V 的 arithmetic_saving=0 说明「拖到 native BN 完再算 V」不是宽度机制，本想法不得再把 delayed-V 写成算术节省。
- **Predicted observation:** 相对全程宽累加，窄 psum 段减少源端口字宽占用并给出 ≥15% same-port net service；arithmetic_saving 只出现在阈值后 GeMM，且 integer 相对 ordinary 0-diff；AEE 不碰 lifting（+0.013 已失败）。
- **Disconfirming evidence:** delayed-V 已 integer 0-diff 且 saving=0，宽度本来就不是瓶颈；ESTU binary SSA skip 在同刊已被写成「二值跳过」；Fang CICC 3D AC spiking-only against hetero MAC+SNN cores 被用来否定宽/窄双累加（即使我们是 2D 数字、不是 3D analog compute）。
- **Uncertainties:** 预阈值 T=10 窗是否在 OF 推理热路径上足够长，值得单独累加器；窄侧一旦用 Bitmap AND/PopCount，审稿人可能判成 FireFly-S/FireFly-T 注意力引擎的复述。
- **Origin:** ai-assisted
- **Stage:** independent
- **Status:** candidate

---

## R3Q01-I5 隔离的 residual ADD 侧 ALU（保留 PED，拒绝 IAND / 拒绝双 MAC 共用 o）

- **ID:** R3Q01-I5
- **Statement:** 脉冲 GeMM last-use 之后，用比较器门控的侧边连续加法 ALU 只消费 PED/I24 FIFO 做 residual ADD；加法器不是第二条吃 AT-LIF 幅值的 MAC，也不是 Spike-IAND 那种用 IAND 替换 ADD 以保持全脉冲。
- **Assumptions:** 我们 **保留 PED**（Spike-IAND 在此为负对照）；SDT membrane-shortcut 与 FireFly-T pre-neuron membrane residual 都是膜捷径，不是 PED 张量；Fang CICC 反对的是 3D analog 上 hetero MAC+SNN cores，本侧 ALU 是 2D 数字 typed ADD；lifting AEE +0.013 已破 +0.005 门，本机制不得改算法。
- **Predicted observation:** 相对「把 residual 脉冲化 / IAND」准确率不掉，AEE abs ≤1.259 且相对 ordinary 1.219801338 的偏移 ≤+0.005；相对「两个 MAC 都看到连续 o」综合部不出现双倍宽乘加；same-port 上 PED 拍与 spike 拍错开后净服务 ≥15%。
- **Disconfirming evidence:** 去掉连续 ADD、改 IAND 或全脉冲 residual 后 AEE 仍过门（PED 不是必要）；侧 ALU 变成第二条完整 MAC，被 Fang 类 hetero 理由打回；面积故事滑向 OpenROAD-as-PPA 或 component-ratio FPS（禁止）。
- **Uncertainties:** OF Transformer 的 residual 是否每次 last-use 都真正需要 I24，还是可在软件里融合掉；与 I1/I2 是否只是 scoreboard 的 ALU 附件。
- **Origin:** ai-assisted
- **Stage:** independent
- **Status:** candidate

---

## R3Q01-I6 Last-use 开火门控的 same-port grant 比较器（净服务，不是 full-chain %）

- **ID:** R3Q01-I6
- **Statement:** 源端口 grant 比较器只在「某 typed last-use 本拍开火」时发拍：度量定义为 granted beats / source-ready beats，显式拒绝把 serialized full-chain −6.4% 和 FP preview+sn2 占该构造 ~49% 的延迟差当作净服务。
- **Assumptions:** −6.4% 尚未闭合为 same-port net service；FP preview+sn2 污染了那条链，8 条 FP32 FMA lane 的 preview-V microkernel（ordinary 10396 slots）不是硅上机制；信件只主张 grant 门控这一件电路，不报 component-ratio FPS。
- **Predicted observation:** 在同一源端口、关掉 FP preview 后，last-use-gated grant 相对「有 ready 就灌」给出 ≥15% net service；long-BP 8088 若来自无效灌拍，会随 grant 门控下降；AEE 不变。
- **Disconfirming evidence:** 门控后 net service <15%，−6.4% 与 grant 无关；8088 不变，说明 long-BP 不是无效灌拍；审稿人把该比较器看成性能计数器而非 datapath 机制。
- **Uncertainties:** source_service 6914→6170 与 fused-writeback 6938→5354 哪一张才是 same-port 表；无 wait-class 时无法证明省下的拍属于 spike 还是 PED。
- **Origin:** ai-assisted
- **Stage:** independent
- **Status:** candidate

---

## R3Q01-I7 切断 BN 假依赖的 last-use retirement（BN 不是 X）

- **ID:** R3Q01-I7
- **Statement:** producer 行的 retirement 比较器 **不等** native BN complete：spike last-use 与 PED last-use 都完成后即释放，BN 只走旁路 typed FIFO；这是去掉假依赖，不是做 BN 硬件，也不是 delayed-V 算术。
- **Assumptions:** delayed-V until native BN completes: integer 0-diff，arithmetic_saving=0，native_BN_reduction_hardware_closed=false；early_V96=55.3MB vs late_U32=18.4MB 的 bus occupancy released **不是** net service；live full-domain proj BN 10×96×120×160 是 hygiene not X。
- **Predicted observation:** 若 8088 的 wait-class 含大量 `BN_barrier`，切断假依赖后 same-port net service ≥15% 且 arithmetic_saving 仍为 0（与 delayed-V 一致）；AEE 不变。若 8088 不含 BN_barrier，本机制在 OF 信件中被证伪。
- **Disconfirming evidence:** retirement 提前后功能/数值破裂（BN 是真依赖）；净服务仍 <15%，只看到 occupancy 数字变好看；被读成「我们做了 BN 硬件」而 BN 未闭合。
- **Uncertainties:** 事件 OF 流水里 BN 是否真挡在 last-use 上，还是只在 full-domain 投影卫生路径上；与 I3 tag 强耦合，单独成信可能不够。
- **Origin:** ai-assisted
- **Stage:** independent
- **Status:** candidate

---

## R3Q01-I8 事件 tile 的双 typed 对齐 FIFO（不是空间稀疏标题，不是 first-OF-HW）

- **ID:** R3Q01-I8
- **Statement:** 为 event-camera 2D OF 放一条按 tile 对齐的双 typed FIFO：同一空间窗的事件令牌同时对齐到「二值 spike GeMM tile」与「PED/I24 tile」，用比较器决定该窗何时双 last-use 可退——主张的是突发事件下的对齐/retirement，而不是「OF 有空间局部所以稀疏」。
- **Assumptions:** ASNA-Flow abstract already claims OF spatial-locality sparse computing，故空间稀疏不可当标题；SENECA/TrueNorth/plane-fit FPGA/EventShiftFlow already event-OF HW，故不可当 first-OF-HW；FlexSpIM CIM = A then STOP as title；SpikePool is GPU max-pool attention not silicon；LoAS FTP 不是本对齐 FIFO。
- **Predicted observation:** 在事件突发窗，相对「先写完脉冲再写 PED」的串行对齐，双 typed FIFO 减少源端口空等，same-port net service ≥15%；ordinary AEE 维持；不声称比已有 event-OF 加速器「第一个」。
- **Disconfirming evidence:** 对齐 FIFO 的收益完全可被 ASNA-Flow 空间局部稀疏计算一句话覆盖；无事件突发时 FIFO 恒空，机制退化为 I1；变成又一个 event-OF 整机而超出「一个机制」。
- **Uncertainties:** 2D OF 的 tile 与 Transformer 的 token 是否同一索引；事件稀疏是否让 PED tile 长期无 last-use，scoreboard 无法释放；5 页信能否在不画整机的前提下把对齐 FIFO 讲完。
- **Origin:** ai-assisted
- **Stage:** independent
- **Status:** candidate
