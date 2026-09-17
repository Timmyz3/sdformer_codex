# 有限距离界替代 ROBDD 图读取：真实源 RTL 负结果

2026-09-15。固定 **16 个 popcount16、逐时间面/逐 witness 复用**的点功能正确，但远慢于普通 static64+下一 X 预取。32 个既有训练帧×P32，exact PDE ready 延迟是 static64 的 **7.872×**；最佳 Wz 类也为 **7.701×**。图读取确实归零、读字减少，判决税却占 exact PDE ready 服务的 **90.12%**。停止这个有限比较器布局；没有扫并行度或改 D、顺序、pair、权重、阈值，也不据此否定全部费用感知源生产接口。

| 同输入源臂 | ready 总拍 | BP 总拍 | 实际 gate (P,c,t) | 实际 X word | 全部物理读取 byte |
|---|---:|---:|---:|---:|---:|
| m1 static64 + next-X PF | 1,269,632 | 1,295,104 | 655,360 | 131,072 | 2,112,512 |
| m2 ordinary exact-code PDE | 9,994,558 | 10,165,485 | 435,787 | 115,878 | 1,870,432 |
| m3 W′ class PDE | 9,981,420 | 10,152,450 | 435,091 | 115,570 | 1,867,040 |
| m3 W″ class PDE | 9,935,492 | 10,105,298 | 432,713 | 114,102 | 1,843,552 |
| m3 Wz class PDE | 9,777,634 | 9,945,910 | 424,986 | 112,614 | 1,819,744 |

每行均为 32 帧×32 个 P，cycle 包含 start、第一 P 实际冷配置、实际 X 请求/返回、十项 MAC、判决、六个 T10 label 包及 done。每帧 P1…31 只复用静态寄存器，X/cache 每 P 清空；同一进程仅初始 reset，连续换 P/帧/mode/BP。全表共同 8 bank、每 bank 单 128 bit 请求/返回、最多一个在途；ready 返回延迟 1 拍，BP 日历及握手保持实测。**这是源 PSN→typed code/class 组件，未接后端 Y/U/gate；不是完整链或网络 AEE。** W′/W″/Wz 是三种既存不同权重函数的完整 H384 类，不能混算质量；m2 在三者输入上执行完全一致，m1 exact code 可用于三者，不消费 W。TB 的 BP 相位每 P 重启，旧 joined 整链日历没有相同分界，因此未直接拿旧整链周期作本表分母。

**A / B / 已实际实现的差分。** A 是经典 partial-distance elimination、费用感知特征求值及普通四前沿/驻留调度，见 [LITERATURE.md](LITERATURE.md)。B 是 ROBDD 图流量压缩了源早停的净服务收益。此次用真实有限比较器取代图请求，仍须先经过真实 A0×X 的完整非因果十项 PSN 才得到被查询的门；只在所有剩余码共享 full-H384 canonical 时交付类。没有 TB 注入门/候选/标签，没有把 CPU 树存入 RTL。它完成了有费生产与判决的接口验证，**没有形成有效加速 X**；公式、resident 调度和 class 相同都不是新颖点，也不宣称新颖性或接收概率。

**数学与源工作核验。** 对 k/j，未知补全下完整距离差的下界为 `dk−dj−pop((Dk xor Dj)&~known)`；仅严格正，或等零且 `j<k` 时删除 k。RTL 用明确 unsigned6-bit `dk` 和 `dj+span`，后者最大 32。保留所有 16 个 witness（包括已删码）仍不丢真正最低索引 winner。`probe.py` 对每种 code/W′/W″/Wz 的 6×65536 输入穷举通过，真实 W 的完整 H384 响应相等也逐值核验。固定 D-only entropy rank、同四槽 resident-first 调度的独立 CPU 统计与全部真实 RTL 源工作相等。exact PDE 和原 code ROBDD 的 gate/X 工作恰好相同；Wz PDE 的 424,986 gate **多于**旧 exact class ROBDD 的 414,191，说明 pairwise 支配证书比完整 class 决策图保守。免图流量是真实变化，额外早停不是普遍成立的变化。

**物理资源与税。** 一个真实 source PSN：10 个 signed16×24 multiplier、10 路 48 bit 累加、原两级乘积 pipeline。一个判决引擎：16 个 popcount16，共享给 static HAM、DIST 与 DOM；DOM 每拍一个 j，对 16 个 k 做 6 bit 加/比较，16 拍；另有 16-code OR/AND、类一致归约和固定 rank 选择逻辑，REDUCE 收 1 拍。所有十个时间面串行共用，未复制 160 个 popcount。D 的 192 B 是实读寄存器，16 路读取/扇出与组选择 mux 是逻辑成本，不冒充单端口 SRAM 的 16 次免费读；不做等面积/Fmax/EDA 声明。

共同静态寄存器容量：A 200 B、tau 60 B、D 192 B、info mask 12 B、rank 48 B、canonical 48 B，共 **560 B**。真实冷装 m1/m2/m3 为 **30/32/35 个 128 bit word**；class 比 code 多 3 个 canonical word，没有免费 metadata。共同 X resident slots 4×32 B 和 cache 128 B 保留；static 用后者预取下个已知 X，PDE 不产生图预取。新判决工作状态为 known 10×16 bit、一个 16 bit survivor mask、16×5 bit 距离、pending 10 bit 与 t/j 各4 bit，共 **34.25 B**，并保留每 t 5 bit live/label、4 bit next-variable、原 raw word/codes 及控制；不需要十份 candidate mask。旧 node lo/hi、roots 不保留。X 标签/FIFO/有效位、bank pending 地址/位、cache 标签/有效位、MAC holding/pipeline 仍在源码，未将它们计为零。TB 局部图像占一块共同 **4 KiB（8×32×128 bit）** bank 模型；X 为 192 word/P，静态地址到 229，不含被 CPU 读取但未加载到 DUT 的历史图。外存最初写入这些 bank 的装载仍未测，所有实际 bank 请求均计数。

判决恒等式（每 P）为 `E=60+produced_pairs`，`bound_cycles=18E+batches+6`，popcount 操作为 `272E`，支配比较为 `256E`。32 帧 exact 的 E=497,227，真实判决 9,007,319 拍、135,245,744 次 popcount16、127,290,112 次支配测试。少求的 PSN 工作不足以偿还此费用。

验证共 **17,408 个 P 命令、1,044,480 个输出 label**：旧 W′ 小集含两真实帧、zero、signed-extreme，各两遍 ready/BP；W″ 同小集一遍；Wz 两遍；三组 32 帧 strong code/class（static64 在相同输入另实跑一遍）。C++ 从原 X 独立计算完整门和最低索引 gold，逐次比对实际源 U48/gate、所有输出、重复门禁止、顺序和 held-request/output/done。`summarize.py` 另做 **2,120 项 CPU 源工作等式**、**4,608 项跨权重 exact 执行等式**及每条 FSM/物理字/判决计数核验，全部通过。未使用之前 Q11/后端 gate 作为当前 gold。

复现入口：[run_all.sh](run_all.sh)（Verilator 4.028、C++17、assert 开启）；[implement.py](implement.py) 从只读成熟 source 派生实际 [distance_source.sv](distance_source.sv)；[tb.cpp](tb.cpp) 不读取 CPU decision tree。数值原始记录为 `{old,adapt,zero}_{small,expanded}.csv` 与同名日志；[SUMMARY.json](SUMMARY.json)、逐记录 [summary.jsonl](summary.jsonl)、[probe.json](probe.json) 分别保留 RTL 统计与 CPU 机会。完整实验没有新训练、生产改动、Git、hash 或 EDA。
