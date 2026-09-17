# Post-Y T10 两半子集表＋门证书：真实 RTL 负结果

本叶完成了普通 distributed arithmetic 两组 5bit 子集和的实际迁移，以及同一核上的精确门证书适配。**32 个真实病例的 PSN 服务周期：普通 subset full 285126，cert 428572（慢 50.31%）；旧 native96MAC 为 118344，是不同资源参照。当前 H8×T10 布局停止，不接入生产者，不据此否定 DA、BitL 或门证书家族。**

这补上了[先前直接条件加法树 CPU 下界](../probe_psn_retirement_granularity.json)没有覆盖的子集查表执行单位。它不是完整 BitL 迁移；没有借旧 T5 的 17% 数字。没有训练、EDA、频率、面积、功耗或新 AEE 结论。

## 实际输入和算术边界

输入固定为[原 cases.npz](../../support_lut_execution_20260915/cases.npz)中的 32 real＋原 5 diagnostic，保持原顺序；32 real 是 8 个已投影源 tile×4 个 H96 函数，不能当 32 个独立源。本叶边界是 **post-Y**：`prepare.py` 从真实 S/W 独立计算 Y，再计算 `U[p,t,h]=sum_s A[t,s]Y[p,s,h]`，与原 Y/U/gold 全部逐值相同。DUT 只接受 Y、A、tau 和 flags，U/gold 仅在 TB；FC1 的最后一次 Y 写者/就绪协议没有接入本叶。

A 是 FC1 **后级**整数 PSN 的真实 signed16[10,10]，100 项全部非零，不能用前级源 PSN 的 A 替代。Y 使用 signed24，U/prefix/bounds 使用 signed48，无中间移位14位或 RNE。正 gain 是 `U>=tau`，负 gain 是 `U<=tau`；constant gate 优先。真实通道均正 gain，负 gain/constant/tie 来自明确诊断，最后一个诊断有 3520 个 `U==tau`，不冒充真实 BN 统计。数值与来源见 [inputs.json](inputs.json)；原函数说明见 [CASES.md](../../support_lut_execution_20260915/CASES.md)。

两个半表为 `L[j,k,t]=sum_{s=0..4} bit(k,s)*A[t,5j+s]`。按 8h×10t＝80 输出成组，共每命令 32P×12 组＝384 组。每个 P 的 T10 Y 行经真实存储读入暂存区，RTL 从每个 H8 组的实际 signed24 值产生共同幅值指数 e，零组取至少一位。符号前缀 `v=-sum_s A[t,s]*sign(Y[s,h])`，随后逐位 `v=2v+L[0,k0,t]+L[1,k1,t]`。证书使用 `lo=(v<<m)+N[t]*(2^m-1)`、`hi=(v<<m)+P[t]*(2^m-1)`；全部 80 个门已确定才退休。符号头先查表相加、再取负，也真实占用两拍共同 ALU。

## 同资源 full/cert 与异资源 native

唯一数据算术资源是 **80 路 48bit add/sub（含 carry-in）**；没有乘法器。所有表构建、P/N 构建、符号头、两个子集相加、prefix 更新和 lo/hi 都分时用这同一组 ALU。每个输出共用 signed less/equality 比较结果处理 full、lo、hi。full 使用同一个核、相同资源、相同参数/输入/输出服务，**跳过无用的证书阶段**；不是让 full 也承担 cert 税的弱对照。

| 状态/服务 | 实现与容量，full/cert 完全相同 |
|---|---|
| A 与子集表 | A 200B；表 1280B，20 个 signed16 bank×32 项，每 bank **8 个不同地址的 32:1 寄存器读 mux**，共 160 个16bit mux。存储一份，不复制8份，也不称单口 SRAM。构建复用这些 mux，没额外第9个读口。 |
| 表构建 | A 经 13 个128bit物理词加载。64 拍写 64 个160bit行：2 行零赋值、62 行用10路共同ALU；P/N 清零1拍＋20拍共同ALU累加。冷共85拍，warm保留。 |
| Y 持久存储 | 320×96×24bit＝92160B；单个共同 row 地址，写/读阶段互斥，每命令真实320次2304bit生产交接写＋320次2304bit内部读。没有任意 T10 多端口读。 |
| T10 暂存 | 10×96×24bit＝2880B，较 native 的单 row 288B **多2592B**；一次读完一个 P 的 T10 后处理12个H8组，避免把同Y读12遍。后续选H8需80个24bit的12:1 mux，不能视为无硬件。 |
| BFP 指数 | 12×5bit＝60bit；96 个 signed24 固定阈值/优先编码路径和每H8最大值选择。不用CPU指数、不加绝对值数据加法器；编码/选择逻辑仍是额外逻辑。 |
| 阈值与 flags | tau 5760B，positive/constant/constant_gate 共144B；当前组读80个tau，需要按hgroup的寄存器选择网，未声称单口tau SRAM同时完成80读。 |
| 算术 holding | v 480B、dot 480B；P/N/tail_pos/tail_neg 共240B；gate/locked/lower_hit/lower_gate 共40B。数据数组合计103664B，另60bit指数、控制、计数器和组合逻辑。 |
| 其他组合逻辑 | 80路48bit变长左移路径（复用不同阶段输入）、80路signed less/equality、地址/门组归约。查表→加法、48bit变移→加法→比较的时序均未测。 |
| 参数口 | 唯一128bit请求/响应口，一次一笔。cold503词＝A13＋tau480（每词2×48bit）＋flags10；warm490词。阈值/flags每命令仍实际重载。ready时每词请求1拍＋响应1拍；BP有请求拒绝和响应延迟。 |
| 输出 | 每组80门，384次握手，映射`(p,t,hgroup*8+j)`，out_valid/data拒绝保持，最后门完成后才done/再启动。与native每T行96门/320次握手的布局不同。 |

`out_u`与`mon_*`仅为TB观察线，不反馈计算，不提供额外存储请求或提前答案；本叶没有把这些观察线作为下一消费者接口。

native 有96路16×24乘法器、96路48bit加法器和原生H96布局；本叶是80加法器＋多读mux表＋更大holding。它们**不是等面积、同频或同端口布局**。full/cert才是本叶同资源因果对照。旧 native 的外存/输出 BP 日历也不同，BP 跨核只提供参考，ready 分摊用于诊断核心调度。

## 32real 周期

`PSN service` 从 PINIT 到最后 OUTPUT 握手，含真实 Y 读和输出等待，不含参数、表构建、Y 生产交接写、go/done。`叶 cold/warm 总计` 包含这些所有项目及每命令1拍go。warm只复用A/表/P/N，Y、tau、flags仍重装。

| 执行臂 | ready PSN service | ready cold 总计 | ready warm 总计 | BP cold PSN service | BP cold 总计 | BP warm 总计 |
|---|---:|---:|---:|---:|---:|---:|
| subset full | 285126 | 330342 | 326790 | 291982 | 371790 | 367225 |
| subset cert | 428572 | 473788 | 470236 | 434094 | 513907 | 509384 |
| 旧 native96MAC（异资源、旧完整 FC1→PSN RTL） | 118344 | 不同边界 | 不同边界 | 122550 | 不同边界 | 不同边界 |

native 原 complete ready184184/BP212665 含 FC1，不与本叶 post-Y 总计相减冒充端到端替换收益；原FC1 ready30352、PSN占118344/184184≈64.25%。原记录见 [rtl_cycles.csv](../../support_lut_execution_20260915/rtl_cycles.csv)。本叶 exact records 每命令一行：[results_all.jsonl](results_all.jsonl)，聚合与独立 trace：[SUMMARY.json](SUMMARY.json)。首病例 native PSN3753，subset full8890，cert13110。

## 失配定位：证书省的位数抵不过共享 ALU 的判界成本

真实 full planes112355 → cert61584，**减少45.19%**；11951/12288 组（97.26%）提前完成。机会确实存在，门证书没有失效。

| ready PSN 分摊（32real） | full | cert |
|---|---:|---:|
| 子集相加＋prefix更新 | 224710 | 123168 |
| tail正界、tail负界（10ALU/拍） | 0 | 122494 |
| lo、hi（80ALU/拍） | 0 | 122494 |
| 符号头两拍 | 24576 | 24576 |
| 每组初始化与输出 | 24576 | 24576 |
| P初始化＋真实Y读 | 11264 | 11264 |
| 合计 | 285126 | 428572 |

cert 节约101542拍查表/前缀，但付244988拍判界，净慢143446。四个判界状态在非末位各执行61247次；到末位时已直接比较，不强制走判界。cold参数/表/Y交接/go/done共同45184+32拍不是主要差额；warm只省3552拍，不能改变结论。

从实测状态账删掉两个 tail 生成状态的**乐观反事实**也还有306078拍，比full285126慢；这没有计所需常量表的存储/读/构建，不能当已实现更快臂。甚至免费删掉全部四个证书状态，沿本次实际退休深度也仍有183584拍，高于native118344。当前限制还包括H8串行组、符号头和逐位双拍；需要改变算术/供数布局才能成为native延迟竞争者，单看45.19%位数减少不能推出周期收益。

## 验证与未闭

Verilator4.028 `--cc --exe --unroll-count 512`＋make；[run.sh](run.sh) 可复现。唯一初始reset后连续切换病例/full/cert/cold/warm，37×2×2×2＝**296命令**全部通过。输入有效可以在参数装载期间提前到达并保持，DUT只在YLOAD接受；参数请求拒绝时地址/valid保持，输出拒绝时门、U和地址保持，done亦可背压。source在有效前有间隙、参数响应可延后，不为等待创造额外访问。

- gate 9093120、Y读回9093120、full U4546560，均零差；37个唯一病例各有30720个Y/U/gate。
- 证书上下界包络检查44624640次；RTL生成指数检查113664组；表构建实际写值独立检查94720个。
- signed dense、zero、onehot、多项逃逸、negative gain、constant优先与tie均实际过。没有使用“XOR翻转>=”代替负gain的包含等号比较。
- [summarize.py](summarize.py) 从真实S/W重算Y，用独立矩阵算术重新推导H8逐组位数/退休深度，逐条核对planes、early、所有状态、Y访问和检查数；不是用CPU工作量替代RTL周期。
- 小测捕获80bit归约退休异常；该Verilator版本生成C++把宽`&locked_next`写成错误的零判断。最终RTL用32/32/16bit显式全1比较，且TB断言每个输出的80门均锁定；最终全量均使用修正源码。该处理不改变证书数学边界。

本叶完成的是“直接树下界负→两半DA真实执行→共享资源证书适配→明确新的判界/粒度税”。未完成完整 BitL 的列编码/布局、优化后的固定A常数MVM/CSE、FC1实际Y生产者握手及跨层融合。当前80门组输出也尚未接原H96行顺序消费者；若接入，需要真实门组装/逆布局存储和读写，不能免费忽略。没有将未完整迁移的这些接口判成失败；按授权停止当前H8两半布局，不继续扫分组/表宽。
