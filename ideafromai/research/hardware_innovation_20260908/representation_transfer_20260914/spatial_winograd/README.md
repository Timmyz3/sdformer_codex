# Q11 连续 Q2 F(2,3)：完整整数 RTL 与消费者

已完成同 Q11 函数的普通空间 R16 分解与 Q2 Winograd 两臂，实际执行 native gate source → Q1 → Q2 → raw → 原 FP32 identity → J20 → wide64 → I24。**新增 18 序列、36 tile 的冷完整服务只改善 0.268%，背压下仅 0.022%；当前通用四 M 布局没有显示稳定的跨输入净收益。** 原两套同帧 64 tile 的 6.47%／7.41% 改善仍成立，但不能代替跨序列结果。当前源码、输入和结果在此锁定，后续结构改动必须另立函数和目录。

下表为从实际模型加载到最后 I24 的累计周期；冷流只在第一个命令付模型加载，暖流是模型驻留后的第二遍，source/origin/start 每 tile 均付。两遍连续换源、不 reset，只有 I24 完成后才回收本 context。

| 输入与服务 | 普通 factor m0 | Q2 F(2,3) m1 | 少用周期 | 改善 |
|---|---:|---:|---:|---:|
| **18 序列 × 2 tile，冷 ready** | **980,690** | **978,062** | **2,628** | **0.268%** |
| **18 序列 × 2 tile，冷 BP** | **1,025,097** | **1,024,872** | **225** | **0.022%** |
| 18 序列 × 2 tile，暖 ready | 979,514 | 976,694 | 2,820 | 0.288% |
| 18 序列 × 2 tile，暖 BP | 1,023,921 | 1,023,504 | 417 | 0.041% |
| 同帧 held128–191，冷 ready | 2,334,322 | 2,183,350 | 150,972 | 6.467% |
| 同帧 held128–191，暖 ready | 2,333,146 | 2,181,982 | 151,164 | 6.479% |
| 同帧 held128–191，冷 BP | 2,418,895 | 2,273,392 | 145,503 | 6.015% |
| 同帧 held128–191，暖 BP | 2,417,719 | 2,272,024 | 145,695 | 6.026% |
| 同帧 disjoint4000–4063，冷 ready | 2,510,814 | 2,324,802 | 186,012 | 7.408% |
| 同帧 disjoint4000–4063，暖 ready | 2,509,638 | 2,323,434 | 186,204 | 7.420% |
| 同帧 disjoint4000–4063，冷 BP | 2,598,908 | 2,418,406 | 180,502 | 6.945% |
| 同帧 disjoint4000–4063，暖 BP | 2,597,732 | 2,417,038 | 180,694 | 6.956% |

36 个跨序列 tile 为每个序列首个捕获帧的固定 edge128／interior9664，文件、帧号、原点和顺序见 [sequence_fixture_metadata.json](sequence_fixture_metadata.json)。这些输入重新经过 Q1/Q2 和独立展开 W 计算 raw，再从实际 FP bits 重算 J20/wide/I24，与捕获的本函数值全部相等；没有借原单帧 gold。它们覆盖 18 个序列的两个固定位置，仍不代表 825 帧全部 tile 的 RTL 性能。[sequence_results.jsonl](sequence_results.jsonl) 保留每 tile 损益；ready 时 17 个赢、19 个输，按每序列两 tile 合计为 10 赢、8 输。edge 累计慢 9,660 拍，interior 快 12,480 拍，这两个数均不含一次冷模型额外 192 拍。

实际失效条件可以按源数据精确解释，而非只看六乘积变四乘积。原 Q2 已按最终 Z／权重支持跳零，D 变换会改变支持；即使 D 完全零，本布局也付固定变换和输出恢复。每 tile 不含 BP 的周期差严格满足 `ordinary − Winograd = 普通Q2向量MAC数 − Winograd向量MAC数 − 2832`。这个式子由 RTL 每状态计数和独立 source profile 同时核过。

| 输入 | 普通 Q2 MAC | Winograd Q2 MAC | 少用 MAC 拍 | 固定新增拍 | 冷加载新增 | 最终冷收益 |
|---|---:|---:|---:|---:|---:|---:|
| 36 跨序列 | 391,776 | 287,004 | 104,772 | 101,952 | 192 | 2,628 |
| held64 | 1,174,644 | 842,232 | 332,412 | 181,248 | 192 | 150,972 |
| disjoint64 | 1,246,164 | 878,712 | 367,452 | 181,248 | 192 | 186,012 |
| 跨 row/边界 short5 | 52,332 | 42,300 | 10,032 | 14,160 | 192 | **−4,320** |

short5 固定为 tile159,160,161,19199,19040；完整消费者冷 ready 129,572→133,892（慢 3.334%），冷 BP 135,753→140,506（慢 3.501%），暖流也负。该失败臂与所有原记录保留，没有以它否定整个空间快卷积家族。36 跨序列的 Q2 实际权重向量读从 16,092 增为 21,456，BP 进一步吃掉 ready 的微小收益。这里没有增加输入预测器、扫描参数或事后择优切模式。

固定 2,832 拍来自可检查的实际 controller：普通的 cache填充／POSLOAD／STORE 合计 576+960+960=2,496 拍；Winograd 为 cache768、POSLOAD480、输入变换240、输出恢复1920、stripe1读480、stripe1加480、STORE960，合计 5,328 拍。输入变换的 240 拍已经包含 80 次实际 Z 整字读、160 次共享 ALU 加减，80 次整字写与其中两种 ALU 状态同拍；没有免费输入变换。每 tile 新增恢复 1,920 次八 lane ALU issue、stripe 累加 480 次，全部与乘加使用同八条 32bit carry 链。

raw-only 独立回放也已完成：跨序列冷 ready 893,366→890,738、冷 BP 921,427→921,188；held64 冷 ready 2,179,098→2,028,126；disjoint64 为 2,355,590→2,169,578。完整消费者 ready 每 tile 实际增加 2,425 拍，另首次加载 consumer 参数 24 拍；64 tile 共 155,224 拍。上述 I24 数字来自真实 wrapper 回放，非把这常数加到 raw 结果代替 RTL。

同函数的独立强普通 OS AAC 也已对账，来源为 [Q11 direct 报告](../spatial_r16_direct/q11/README.md)。[summarize.py](summarize.py) 对两套64和跨序列共164个 fixture 的 source/origin/raw/identity/J/wide/I24 做了数值同一性检查。冷完整服务为：

| 输入 | 直接 OS | 普通 factor | Q2 Winograd |
|---|---:|---:|---:|
| 跨序列36 ready | 1,182,084 | 980,690 | 978,062 |
| 跨序列36 BP | 1,422,253 | 1,025,097 | 1,024,872 |
| held64 ready | 2,601,944 | 2,334,322 | 2,183,350 |
| disjoint64 ready | 3,006,212 | 2,510,814 | 2,324,802 |

OS 有同八条 producer ALU 和同服务权限，但展开 W32 为 331,776B，factor 为较小因子表；两者不等面积。对 Winograd 适配本身，主分母仍是已经较强的同函数普通 factor，不能把空间低秩本身的收益一并记成 Winograd 收益。

[PLAN_RESOURCE_CONTRACT.md](PLAN_RESOURCE_CONTRACT.md) 在 SV 前写明 controller，完整资源见 [resource_contract.json](resource_contract.json)。Q1 保留 θ 已吸入权重后的二值 AAC、两个 R8 stripe、双 P15、完整最终 Z 支持；T10 及空间 padding/order 均保持。mode1 只在连续 Q2 使用经典 F(2,3)，没有新公式或新的中间 RNE：

```text
D = [d0−d2, d1+d2, d2−d1, d1−d3]
U = [2g0, g0+g1+g2, g0−g1+g2, 2g2]
Mi = sum_rank Di*Ui
y0 = (M0+M1+M2) >> 1
y1 = (M1−M2−M3) >> 1
```

每 stripe 两个恢复和必须偶数，右移1是精确除法。冻结 Q11 是一次满足现有乘法位宽的函数选择；原 Z 界 [−8562,7427] 保持 signed15，D 绝对值≤17124 用 signed16，再扩到19；U∈[−2981,2981] 用 signed13。M 任意 rank 前缀界239,174,076，恢复部分和界507,080,380均适配32bit。原普通 Q11 和 Winograd 共享这个函数；原 Q13 的周期和质量不能作为同函数分母。根代理已独立完成本 Q11 [825帧质量](../quality/q11/deployed_valid/spatial_integer_summary.json)：frame-mean AEE **1.2544305372701712**。这是另一路整网质量结果，825帧没有全部跑本 RTL。

两臂实例化共同的 union 资源：producer 8×32bit ALU、8×19×13 multiplier；source1920B、原生窗口20B、Z1280B、完整 p_mem15360B。外部 W 仍单256bit服务，Q1实用64bit、Q2实用104bit；Z 共用一个地址，向量阶段读256bit，Q2只使能选中32bit bank；psum 同地址读或写互斥。Q2 cache每拍只读一个104bit向量。consumer另有相同8×32×32 multiplier／8×64bit ALU，FP32转换与RNE/saturation实际执行，没有借给producer。

| 明确新增容量，相对原普通 factor 布局 | 字节 |
|---|---:|
| 一份静态 Q2 表 576→768 向量 | 2,496 |
| Q2 cache 24→32 向量（312→416B） | 104 |
| M1、M2、M3 三组八 lane accumulator | 96 |
| 变换第二个原 Z 字 holding | 32 |
| Q2 live 576→768 bit | 24 |
| block_live／remaining 各扩8bit | 2 |
| 上述数据和支持小计 | **2,754** |

原 acc 先用于 D 临时值、再为 M0／恢复值；原 z_hold 先保留第一个原 Z 字，后用于 stripe1 psum读回。每个 `(y,T)` 两原字都读完才覆盖成两个 D16字段，仍原地用1280B Z。M1–3在第二输出恢复前保持；p_mem所有stripe0行写完才由stripe1读加写，全部960次store后才原序drain。没有另一份 D 全数组或免费原 g 副本。tx32bit状态寄存、mode latch、扩展支持选择和地址／减法 mux 等控制另外存在；监视计数/断言不当成功能存储。m0允许同额外holding/表容量而闲置，不据此声称裁剪后绝对等面积。

Q2物理只一份9984B表：m0装原 g 的576向量，m1装 U 的768向量。两臂分别实际装 Q1+Q2 1152／1344拍；完整consumer再各24拍，总1176／1368拍。source1536拍、origin1拍、start1拍逐tile付。模式分别在独立 reset/config 进程运行，已经测试的是同模式同模型的连续换源与驻留复用；**没有测试无reset跨mode换布局重载，也没有在线权重版本失效接口。** 这不构成一拍切模式免费改权重的主张。

验证覆盖15个small（8真实、zero/one/random/tail/rank正负/图外poison）、short5、两套64和新增36；每集两mode×ready/BP×两遍连续换源。raw-only与完整consumer各 **1,472命令**，合计2,944命令；总raw比较11,304,960值，实际J20/wide64/I24分别5,652,480值，总Z3,768,320值、D1,884,160值。178个fixture从source独立重算；逐状态、端口、配置、恒等式和驻留重复核验共351,848项全过，见 [verification.json](verification.json)、[raw_verification.json](raw_verification.json)、[consumer_verification.json](consumer_verification.json)。BP在source/W/raw/identity/output真实服务生效，输出数据及地址在stall时保持，末包和原序480行逐值比较。**M没有逐值RTL monitor**；其范围／更新由每拍前缀断言和独立静态路径审阅覆盖，Z/D/raw/J/wide/I24才是逐值检查。

[独立审阅](../spatial_r16_rtl/review_new_fusion.md) 区分了经典算法与这里的接口工作，并发现/促成修正了初版 q2_live 容量错误；最终两mode所有结果来自修正后的768项支持表。实现借用连续层F(2,3)，具体完成的是既有窄乘法器内的Q11系数、D16原地格式、共享ALU恢复及完整消费者计费，不能将普通低秩+Winograd组合称为首创。仅 Verilator4.028 的周期与整数功能验证；cache/Z读→乘加及新加减选择路径没有综合、Fmax、面积、能耗或PPA数据。

复现使用 `/opt/anaconda3/bin/python3.12`。首次执行 `prepare.py`、`implement_consumer.py`（生成wrapper及完整consumer oracle）、`prepare_sequences.py`；已有源码可直接编译。运行 `run.py --stage small` 与 `run_consumer.py --stage small` 各以 Verilator `--cc --exe`＋make 构建，再逐一运行同脚本的 `--stage short/held/disjoint/sequences --skip-build`（每次一个stage）。最后执行 `verify_raw.py`、`verify_consumer.py`、`summarize.py`。逐命令结果为 `raw_{stage}_m{mode}_s{stall}.jsonl` 和 `consumer_...jsonl`，每命令一行；[comparison.json](comparison.json) 保存完整冷暖/BP与每序列汇总。无EDA、训练、生产/main.tex改动或Git提交。
