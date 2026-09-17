# 真实 mode4 FC1 → fused PSN → 逐t H96门

**32real完整冷启动周期：native184408、fused full227219、cert176448；cert比native节省7960拍（4.3165%）。BP冷启动native212997、cert204698，节省3.8963%。** 同边界时钟break-even分别为 **95.6835%与96.1037%**；不能沿用独立post-Y叶的82.32%。长组合路径仍未映射，未宣布频率、面积、功耗或ASIC性能通过。

本阶段完成了真实FC1写者、唯一Y阵列所有权、共同密排参数加载和实际120B门转排。完整444命令与36跨模式warm生命周期命令共480条全部通过；独立只读审阅见[REVIEW.md](REVIEW.md)。同新union硬件full→cert整命令减少22.34%（cold ready），但full自身比native慢；full用于隔离门证书的增量，native仍是主要强分母。cold ready只有30/32real更快，最差慢68拍；BP31/32更快，最差慢58拍。结果不是输入无关收益。

实际计算边界和普通强控制

输入是[原真实cases](../../../support_lut_execution_20260915/cases.npz)的g′门，320行×96bit；不是CPU提供Y。前级复制[原support_fc1](../../../support_lut_execution_20260915/support_fc1.sv)的**mode4普通内容去重**：完整6组支撑码匹配、W内容相同响应归一、按真实source产生routes、原24词/4描述符供数、96lane条件加法以及转发/写回。三臂固定相同FC1，没有把native换成弱mode0，也没有将普通class去重当新机制。

PSN只有native/full/cert三种选择，最终共同接口是`out_valid/out_ready/out_row/out_gate[95:0]`，按p→t的320行输出。native提供完整U观察值；fused full在实际80lane组结果上检查完整U，最终门按原H96行顺序输出；cert只承诺门，不承诺完整U。U/Y/界等观察信号只供TB，不反馈计算，不作为免费额外memory服务。

32real仍是8个投影源tile×4个H96函数，另5个原诊断；没有把它说成32个独立源。本阶段起点在g′，未接入frontier source classifier。A是真实后级signed16[10,10]，全部100项非零，Y为signed24，U/prefix/界为signed48，无中间RNE；正gain使用U>=tau、负gain使用U<=tau，constant优先。CPU独立重算每个S@W、A@Y和门，与原gold逐值相等；DUT只得到S、W/表、A/tau/flags。

唯一Y与真实转排

| 资源/服务 | 当前RTL及所有权 |
|---|---|
| Y持久存储 | **唯一**`y[320][96]`，92160B；bank-valid320B懒清零，原FC1转发和两个写回级保留。没有第二份90KiB、没有post-Y重装阶段。 |
| Y写者 | 原FC1按实际活跃bank写回，每bank12×24bit＝288bit；TB检查每个部分和写回。只有jobs/qcount/active/fv/pending全部空，才把访问权交给PSN。 |
| Y读取 | 一份2304bit共同行地址mux；FC读取、native PSN、fused FYREAD和输出观察互斥。无效bank返回零且不读物理Y；fused每P实际10次读形成T10缓冲。 |
| 门转排 | 120B，10×96bit寄存器bank。每个FSTORE给10bank各写当前H8的8bit，共80bit；12组全部覆盖后才读出10行96bit。下一P不能在上一个P输出完前覆写。无需清零，所有960位先完整写再读。 |
| 消费者背压 | FOUT拒绝时row/门稳定，pack禁止覆写，直到握手；接受320行后才done，再允许下个命令。native原生H96输出绕过无用转排，但具有同union容量。 |
| 实际读计数 | pack接受读320次/命令；另外保守计入FOUT保持期间的mux访问周期。Y同样计入有效bank的保持访问周期，未把BP保持读量藏掉。 |

共同加载与warm合同

A200B、tau5760B和flags144B各**只有一份**；fused没有第二个参数loader。原8bank×128bit外存协议保留，每bank至多一笔pending请求，拒绝时地址/valid保持。

冷命令实际397词：A13、D12、tau360、flags9、class_map3；ready请求/响应各1拍，共794拍/命令。tau密排48bit跨词解码、flags密排144B，消除了独立叶的tau480词/flags10词接口差。W/响应系数仍由共同FC1按实际jobs读取，A/tau既未免费共享，也未重复计费。

`start_reuse_config`仅用于同模型的明确复用，硬件检查`config_valid`和hblock；**没有检测A/D/W派生class/tau/flags身份的版本协议**。换模型必须cold，不能仅凭hblock相同复用。cold将LUT置无效；首次fused使用真实构建64表行＋21个P/N拍，native不付无用构表。warm在同模型保留这些状态。另36命令实际覆盖native cold→full warm（必须构表）→cert warm→native warm→full cold→cert warm；没有把“warm一律零构表”写死。

共同union资源及尚未测的物理代价

| 项目 | 结构账，不是综合单元或面积 |
|---|---|
| 第一层算术 | 原96路48bit add/sub，FC1、native累加、fused前80lane及冷表/P-N分时复用。没有复制FC1数据ALU。 |
| 新组合算术 | 额外80路第二层prefix＋160路lower/upper＋20路tail减法，共260个位置；union合计356个48bit add/sub位置，carry-in包含在逻辑ALU内。 |
| native乘法/比较 | 96个16×24乘法器及原native门比较路径保留。fused另有160路signed less/equality判界路径，**不是全module比较器总数**。所有模式在同module使用同预算，未裁剪闲置单元。 |
| LUT | 一份1280B，20bank×32×16bit，每bank8个32:1寄存器读mux，共160个不同地址16bit mux；没有8份数据副本，不称单口SRAM多读。 |
| 暂存/累加 | fused ybuf2880B、v480B、P/N/tail240B、gate/locked20B、指数60bit；原native yhold288B、U5760B和乘积流水保留。没有为full观察新建U转排数组。 |
| 增量容量 | fused LUT/暂存/累加/门pack共新增5020B数据数组，另60bit指数及控制；原native yhold保持独立，未声称已复用掉288B。主要数据数组额定合计120888B，队列描述符/地址、busy、pending、控制与计数器另列而不混入面积结论。 |
| 外存/供数 | 8192×128bit＝128KiB地址空间，8bank，每bank1pending；24词payload＝384B、4描述符。没有加W服务或source classifier的另一套资源。 |

最长候选组合链仍是LUTmux→两次48bit加法→变长移位→界加法→signed比较→80门全锁定归约，并增加了共同ALU输入选择。三臂的控制因果对照在同union RTL内成立；这不等于已经证明相对独立native核的等面积/同频收益，也没有证明native已在全部额外算术上重新优化到极限。

完整32real周期和物理服务

`cycles`从实际start握手到done握手，含go、密排配置、source、FC1、表构建、PSN、转排、输出和done背压。`psn_cycles`包括native清零/读/MAC/drain/output或fused初始化/读/符号/planes/pack写/读输出；构表单列。warm表是相同模型重复，不将不同病例换模型免费装入。

| 日历 | native完整 | fused full完整 | cert完整 | cert/native节省 | 相对独立native的时钟break-even |
|---|---:|---:|---:|---:|---:|
| cold ready | 184408 | 227219 | 176448 | 4.3165% | 95.6835% |
| cold BP | 212997 | 255588 | 204698 | 3.8963% | 96.1037% |
| warm ready | 159000 | 199091 | 148320 | 6.7170% | 93.2830% |
| warm BP | 165340 | 205189 | 154330 | 6.6590% | 93.3410% |

break-even来自C_cert/C_native：候选频率至少达到该比例才能保住对应整命令延迟优势。它不是测得Fmax；BP是固定bank/输出日历参考。当前最重要的cold ready门槛是只能承受约4.32%频率下降，远小于独立叶容许的17.68%。

| cold ready分摊，32real | native | full | cert |
|---|---:|---:|---:|
| 共同密排配置 | 25408 | 25408 | 25408 |
| g′源输入 | 10240 | 10240 | 10240 |
| 真实mode4 FC1 | 30352 | 30352 | 30352 |
| 子集表/P-N冷构 | 0 | 2720 | 2720 |
| PSN含真实转排/输出 | 118344 | 158435 | 107664 |
| go＋done | 64 | 64 | 64 |
| 合计 | 184408 | 227219 | 176448 |

fused full/cert planes112355/61584与冻结叶逐病例相同，11951/12288组提前退休。独立叶cert97424服务在这里变成107664：384组写pack已承接原叶的384次组输出，但还须真实读出320行H96，每32real多10240拍。再付2720冷构表，服务优势10680最终剩完整周期7960。这说明输出布局和cold合同确实改变了结论。

| 实际数据服务，32real cold ready | 三臂共同 | fused额外 |
|---|---:|---:|
| g′输入 / 最后gate | 各122880B | 0 |
| 配置 | 12704词＝203264B | 0重复A/tau读取 |
| 系数 | 21184词＝338944B | 0 |
| Y bank写 | 241024次288bit＝8676864B | 0 |
| Y有效bank读，含输出观察 | 307744次288bit＝11078784B | 0 |
| gatepack写 / 接受读 | native无需 | 各122880B |
| LUT冷写 | native无需 | 2048次160bit＝40960B |

三臂逐命令FC周期、30128个真实更新、jobs、系数词和Y写bank数完全一致。cold BP共同bank请求拒绝10148个bank-cycle；输出拒绝native4186/full3969/cert3854拍。相应Y读量为12032928/11974752/11971584B，差异来自同接口不同到达相位；Y写量相同。fused BP的pack读mux保持周期已计，full170508B/cert169128B等效访问量，而接受读仍各122880B。[SUMMARY.json](SUMMARY.json)同时保留accepted和held周期口径。

反例与验证

| 单独诊断，cold ready完整周期 | native | full | cert |
|---|---:|---:|---:|
| zero | 2109 | 3026 | 3026 |
| onehot escape | 5763 | 5655 | 3801 |
| multibit escape | 6083 | 6168 | 4396 |
| signed dense | 19781 | 22088 | 20984 |
| negative gain / constant / tie | 19781 | 22088 | 22027 |

zero时native直接利用未初始化Y的零支持，fused固定H8组/pack税反而更大；signed/tie诊断的证书深度较差。保留这些负结果，不将32real聚合推广为普遍优势。前一叶另有直接Y24全范围诊断，已经[冻结通过](../fused_datapath/README.md)；本阶段保持真实binaryFC1输入函数，没有把极值Y从CPU插入实际FC1链。

[run.sh](run.sh)使用Verilator4.028 `--cc --exe --unroll-count 512`＋make。先2real＋5诊断84命令小测，再原37病例444命令，最后3病例36命令跨mode warm；报告聚合以最后444＋36＝480命令为准。唯一初始化reset后持续换source/model/cold/warm；所有输出和请求的背压保持都有断言。

| 最终480命令检查 | 次数 |
|---|---:|
| 逐t H96最后gate | 14745600 |
| 输出Y及fused Y装缓冲 | 24576000 |
| native完整U＋fused full组U | 9830400 |
| cert实际lo/hi包络 | 50269440 |
| RTL从Y产生指数 | 122880 |
| 实际冷LUT写值 | 102400 |
| tail值 / 递推偶数差 | 6283680 / 6168880 |
| 实际共享Y读值 | 123564192 |
| 实际FC1部分Y写值 | 91169280 |

实际Y读/写检查由独立TB committed/issued模型跟踪每次有效bank写和转发提交；最终Y再与独立S@W全量比较，不能以DUT系数自证整个函数。新增这组观察后444条已有结果字段/周期全部不变，见[instrumentation_check.json](instrumentation_check.json)。[prepare.py](prepare.py)还独立推导mode4 class/routes/更新/MAC及高位截断包络退休；[summarize.py](summarize.py)逐条核对。native cold37×ready/BP的全部FC/PSN/config/访问/等待字段与原mode4逐条相等；总周期仅补真实done握手，未弱化普通臂。

可检查文件：[完整命令](results_all.jsonl)、[跨mode生命周期](results_swap.jsonl)、[输入与独立trace](inputs.json)、[RTL](joined_fc1.sv)、[TB](tb.cpp)。本版保留独立post-Y父结果，未改生产或其他agent文件。

这个执行接口已闭到真实FC1→最后H96门。尚未完成的是source classifier接入、更大独立源/函数验证、warm模型版本接口及组合路径物理映射。普通DA和精确证书的数学关系不作为强新颖性；当前只有固定布局上的实际周期候选，没有新训练/网络AEE/EDA/PPA通过结论，也不新增流水或另一种重排。
