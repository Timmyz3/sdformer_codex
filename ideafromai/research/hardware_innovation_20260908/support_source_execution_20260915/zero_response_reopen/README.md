**最终强控已推进到 [frontier_retained](frontier_retained/README.md)：双方同 ordinary partial-frontier、X驻留、activePF、批内配置复用与zero-aware后端，固定Wz的32训练帧完整链净省ready 1.7998%、BP 1.8390%；AEE仍未知。** 下文保留零pair选择、误差及较早单通道源/before/统一root地址控制。后者不是最新最强分母。[最终总账](frontier_retained/SUMMARY.json)。

**较早统一root地址、整T10单通道源控制：同一 Wz、两臂都有后端普通零响应跳过和内容去重，完整链 ready 111917→108406 拍（少 3.1371%），BP 133716→130152（少 2.6654%）。** 此链只测两训练帧、每帧 32 个抽样 P、T10、完整 H384；源单独扩大到 32 个训练帧。Wz 改变了函数，局部门误差增大，网络 AEE 未知。[统一 root 地址的最终总账](NEUTRAL_SUMMARY.json)、[原布局 before 总账](SUMMARY.json)、[源总账](SOURCE_SUMMARY.json)、[数值探针](probe.json)。

B：旧零响应试验只省后端字，未降周期，且没有接源 PSN；旧 producer-cost selector 排除了 code0。这不能判定“提前知道响应为零，是否免去真实源生产”的接口。此次只增加每组 15 个 `(0,k)`，共 **90** 个候选，复用 [旧 630 行](../source_class_adapt/pair_choices.csv)。每组至多选一对，从原 W 作对称 INT8 `[-127,127]` 投影。仍只用 frame0，score 为实际整数权重平方改变量 / 显式 pair quotient 节省的完整 T10 通道任务，再按平方改变量和 pair index 打破 tie。最终投影偶然产生的额外类只在选定后统计；frame1…31 不选 pair，也不选顺序或阈值。[计划](PLAN.md)、[实际 90 行](zero_pair_choices.csv)、[生成脚本](probe.py)。

| g | 固定 pair | frame0 显式 pair 节省 T10 任务 | 整数平方改变量 |
|---|---|---:|---:|
| 0 | (1,4) | 10 | 529449 |
| 1 | (14,15) | 9 | 594505 |
| 2 | **(0,4)** | 8 | 576666 |
| 3 | **(0,8)** | 21 | 821822 |
| 4 | **(0,8)** | 26 | 600623 |
| 5 | **(0,14)** | 19 | 559692 |

合计平方改变量 **3682757**，Wz 保持 INT8。全 H384 响应 canonical 类数为 `[13,15,15,15,15,15]`；零类分别为 `[0],[0],[0,4],[0,8],[0,8],[0,14]`。自然序/既定 D 熵序类图为 **2092/941** 个节点（含 16 个终点），exact-code 熵序图为 966 个。`zero_tables.npz` 包含 canonical、两序 fixed64 节点和 roots；每组 **65536** 个输入全部核对最低 index Hamming tie 和最终 H384 类，INT10 响应范围 `[−388,413]`。RTL 只运行原熵序 packed32＋128 B cache＋预取；未增加顺序扫描。[Wz](Wz.npy)、[静态图](zero_tables.npz)。

普通强对照也从同 720 个候选选最小**实际整数 L2**，允许零 pair，得到 `(0,5),(0,10),(9,15),(0,11),(3,5),(1,4)`，平方改变量 2565103。它在全部 32 帧的熵序完整 T10 源任务数都与 exact-code 相同，尽管确实产生三个零类。该对照也实跑源与整链，不能以“有零响应”直接声称减少源工作。[对照 W](integer_l2_W.npy)、[对照图](integer_l2_tables.npz)。

| 固定函数相对原 W 的局部误差 | Y relative RMSE（frame0 / 1） | U relative RMSE（frame0 / 1） | gate 翻转 / 245760 |
|---|---:|---:|---:|
| 旧非零 producer-cost W″ | 0.23471 / 0.20337 | 0.21795 / 0.21147 | 672 |
| 本次 Wz | 0.45615 / 0.44746 | 0.42816 / 0.43766 | **1645** |
| 普通 integer-L2，允许零 | 0.45589 / 0.47154 | 0.44767 / 0.45612 | 1174 |

以上不是 AEE，也不是留出集。后端 A14/tau/sign/constant 复用既存 `support_lut_execution_20260915/cases.npz` 的固定整数函数，未声称适配了这两帧的 BN/tau。源为实际 `u0=Σ A0_Q12 X_Q16`、`g=(u0≥tau0_Q28)`，Hamming 投影后才算 `Y=Σ D[code]Wz`、`U=Σ A1_Q14 Y` 及原正/负/常量门规则。无运行时 RNE 穿越或截断：源任意 X24 的绝对和界 74524393472，Wz 任意 spike 部分和 Y `[−2119,2097]`、U `[−66978295,67655660]`，满足 U48/Y24。[输入合同与静态界](joined_zero.json)。

| 源单独实跑范围，code→class | ready 拍 | BP 拍 | 完整 T10 源任务 |
|---|---:|---:|---:|
| Wz，两训练帧 × P32 | 71509→67998 | 87055→82825 | 3359→3179 |
| Wz，全部 32 训练帧 × P32 | 1212786→1176013 | 1476189→1432293 | 57128→55220 |
| Wz，未用于选择的 frame1…31 | 1176829→1141915 | 1432418→1390757 | 55438→53625 |
| integer-L2，全部 32 帧 | 1212786→1212960 | 1476189→1471210 | 57128→57128 |

该源表仍保留旧 root222/bank6 对 root223/bank7 布局，BP 含配置地址及后续相位差，**不是最终独立归因表**；本轮未重跑其 32 帧 neutral 版本。全部 32 帧 Wz 源净省 ready **3.032%**、BP **2.974%**。integer-L2 的微小读图收益/负差没有源码任务减少，不解释成早停收益。源独立测试沿原 harness：每例初始 reset，周期是 start 到首次 done_valid 的 index 差，状态 histogram 合计为 `cycles+1`。两个函数各含 268 次小范围、4108 次扩大范围执行，共 **8752 次 PASS**；范围有重叠，不能当 8752 个独立样本。所有真实帧的 RTL 通道数另与 CPU 图遍历逐帧相等。[源核验脚本](summarize_source.py)、[Wz 全范围 raw](source_zero_expanded.csv)、[L2 全范围 raw](source_l2_expanded.csv)。

整链是 `X → 真实 10×10 source PSN → code/class → RTL 读 D 展开 → 320 行 g′ → FC1 → 完整 temporal PSN → gate`。本目录 [wrapper](joined_core.sv) 的最终构建使用隔离 [source_classifier_neutral.sv](source_classifier_neutral.sv)；原布局构建仍只读引用父 [source_classifier.sv](../source_classifier.sv)。隔离 [support_fc1_zero.sv](support_fc1_zero.sv) 在既有 mode4 中增加 **class_map=0 就不建立 route/job**；零 D 没有 LUT payload，不能访问一个假零表项。exact-code 和 class 都真实加载相同每 H96 三字 class map，享有此权限；C++ map 搜索也从 0 开始。其余端口、缓存、在途和调度继承原模块，旧模块未改。

| 最终统一 root 地址、完整同函数、双方 zero-aware mode4 | exact-code / class 拍 | 净省 | exact-code / class 读字节 |
|---|---:|---:|---:|
| Wz，ready | 111917 / 108406 | **3.1371%** | 372032 / 361520 |
| Wz，BP | 133716 / 130152 | **2.6654%** | 371968 / 361488 |
| integer-L2，ready | 111597 / 111590 | 0.0063% | 373568 / 371952 |
| integer-L2，BP | 133341 / 133638 | **−0.2227%** | 373504 / 371936 |

审阅发现 before 模块在 boot30 分别取 root222/bank6 与223/bank7，冷配置及其相位差会进入 BP 总数。最终在模块内部统一 `boot_addr=192+boot`，请求 bank 也随地址统一；[neutral TB](joined_tb_neutral.cpp) 将当前 mode 自己的六个 root 实际装入同一全局 word6174，不只改外部返回地址。两臂图本体仍不同且所有图请求照收。before Wz BP 133716/130284（2.5666%）及原 raw 保留，不再全归为早判；neutral 为上表，普通 L2 结果恰好未改变。每行只合计两真实训练帧各一次 H384，第二遍用于无 reset 验证，不重复计性能。Wz 两臂后端均 **552 jobs、4416 系数字、5144 向量更新、17120 个 96-lane PSN MAC issue**。普通 exact-code 在路由阶段已经省掉 **480 个非零 support / 零响应的 `(H96,g,row)` 目的项**；class 输出 D0 同样不产生它们。因此独立差异只在源何时知道等价。ready 源阶段 72213→68702，后端阶段同为 39656；源 X 字 6718→6358、图字 7574→7277，确有 180 个完整 T10 源任务和总 10512 B 读量下降。neutral BP 后端阶段 46507/46544 相差 37 拍来自到达日历相位，语义工作未变；不能拿源叶子百分比代替整链百分比。[Wz neutral raw](joined_zero_neutral_cycles.csv)、[L2 neutral raw](joined_l2_neutral_cycles.csv)、[Wz before raw](joined_zero_cycles.csv)、[L2 before raw](joined_l2_cycles.csv)、[独立计数](JOINED_PROFILE.json)。

资源沿相同 joined 预算：外部可见 **256 KiB 八 bank、每 bank 128 bit、最多一个已接受在途请求**，源/D/后端分阶段路由，非复制带宽。256 KiB 指共享地址池，**不是含所有局部数组的总芯片存储**。源 10 和后端 96 个 signed16×24 乘法单元物理不同、串行活动，共 **106 个**；未称 96-MAC 同面积。桥接 g′ 为 **3840 B**、D 为 **192 B**、十码 holding 为 40 bit；源保留 128 B cache 及 A/tau/D/node/X/U 状态。后端仅一个 H96 实例复用四次：Y 92160 B、route 7680 B、U/tau 各 5760 B、payload 384 B 等均保留，完整细项见 [原 joined 资源合同](../joined_chain/README.md)。本改动只是六组 class-map 4-bit 非零判断及 route 写使能，未增加 SRAM、MAC 或服务端口；两臂同模块。没有 EDA、等面积或等 Fmax 结论。

所有实际池内请求均收费：每 P 图源配置 **21 字**，每 H96 后端配置 **397 字**，每 tile wrapper D **12 字**；每 tile 实际 1920 次 D16 查询/局部写、四块合计 1280 次 bridge96 读。主周期从接受 top start 到接受最后 gate，含源生产/配置、桥接、后端配置/计算/反压；done 另列。**初始外存装填共享池尚未建模**。局部数组和外部池的读写不能混称免费数据。

验证先 H96 小测 4 命令，再 Wz/L2 各 32 命令；最终 neutral 控制独立重跑同范围，共 **64 个完整 H384 命令、7864320 个 Y/U/gate 标量位置零差**（before 另 64 命令保留，不是更多独立输入）；包括两真实训练 tile、全零 X、signed X24 极值交替、ready/BP、两遍，另加小测 122880 个位置。整链每次运行仅初始 reset，32P、4H96、跨 tile/模式/遍次均用 done 握手重用。C++ 独立从 X 算源 MAC/g、最低 tie code、完整 H384 等价、D 展开、Y/U/gate；DUT 不接受预计算门或码。持有请求/输出、唯一在途、字计数全部断言。Python 独立从 X 算 routes/touched rows，检查 before、neutral 与 H96 小测全部 **132** 行计数，两遍周期与计数一致。[C++](joined_tb.cpp)、[核验脚本](profile_joined.py)、[Wz neutral 日志](joined_zero_neutral.log)、[L2 neutral 日志](joined_l2_neutral.log)。

A 是成熟的约束投影、ROBDD、静态排序/预取、零消除和内容去重；它们不因串接成为新算法。此轮留下的 X 证据是：**按完整 T10 生产义务选择消费者等价约束，能在双方都拥有普通后端优化后减少真实上游 MAC 和整链服务**。本次选择更偏向零类，代价是明显更大的局部响应误差。它尚未证明独立算法新颖性、网络精度收益或 PPA 优势；不能跨仍需原 g/U 的第二消费者，不能由这两训练帧推广为整网结论。旧后端负结果保留，本新接口也不能用来否定其计费事实。

复现 `bash run_all.sh`：仅本目录输出，复用旧 630 CSV，生成固定 W/图，跑源小范围和 32 帧强臂，再跑保留的 before 和最终统一 root 配置地址的同权限 zero-aware 整链及独立计数。原始来源和父 RTL 只读，无训练、参数扫描、Git、hash 或 EDA。
