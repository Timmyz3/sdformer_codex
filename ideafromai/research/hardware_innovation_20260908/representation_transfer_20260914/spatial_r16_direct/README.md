# 同整数函数：直接 AAC 的两种实际 RTL 对照

主对照采用普通output-stationary [os_core.sv](os_core.sv)。OS与原Gustav臂各通过572命令2196480raw：15 small、两套64、ready/BP、无reset两遍不同tile，合计1144命令4392960raw零差。独立source/bitmap机会账逐服务计数、逐FSM状态全部复现。[OS_SUMMARY.json](OS_SUMMARY.json) 和 `os_results_{set}_{stall}.jsonl` 单独保存；原SUMMARY与Gustav原记录保持。

| 冷64，最后raw输出 | Gustav | OS | factor | factor对OS变化 |
|---|---:|---:|---:|---:|
| held ready | 3093058 | 2446720 | 2179098 | **−10.94%** |
| disjoint ready | 3826589 | 2850988 | 2355590 | **−17.38%** |
| held BP | 3333382 | 3007749 | 2234589 | −25.71% |
| disjoint BP | 4108304 | 3559955 | 2414586 | −32.17% |

OS ready core为2337920/2742188。OS消除逐event psum RMW，表明下文Gustav对照的29.55%/38.44%明显受调度分母影响，不能再当最强原生结论。两臂是同一expanded-W32函数，均保留真实输入、所有配置/供数/背压。冷64仅首tile装10368拍direct静态W，每tile再1537源/原点配置+1启动；factor静态1152拍，其余一致。

OS从实际source构造单P×T10×K864位图，逻辑1080B，物理采用与factor Z相同的8bank×40row×32bit=1280B。135个逻辑64bit行映射相邻bank对和公共行地址；每32K收集10词后5拍写入，最多64bit写/单bank32bit读，不超过原256bit服务权限。270bit非空词支持随构表实际生成。每P重新读全部96C原生16字、完整覆盖位图，不保存四P的4320B位图，不接TB支持。40B阶段寄存器在构表期放10词，在消费期前8个作为唯一8×32acc。每og/T按实际位图定位K，单256bit W异步读与8ALU同拍相加，W stall收费；完整归约后仅写一次原psum行。组合读W加法及factor缓存乘加路径都未EDA，周期结果不证明等Fmax。

OS每ready64的source读字341376/391680（四遍，为factor两倍），W读37488384B/48317952B，bitmap读1478688B/1742064B、写276480B/276480B；psum各读写30720行，即各983040B。AAC issue与Gustav同为1171512/1509936。它以W/source复读与真实构表换取acc驻留，这些交换全部收费。OS除物理1280B bitmap外，有270bit支持、27bit剩余词、32bit当前词、40B阶段寄存器、32B输出hold及pair-bank写使能选择器。W容量仍为331776B，对factor12096B；同执行端口/同Z容量预算不等于总面积相同。也未证明穷尽所有可能的OS调度。

OS复现依次运行 `run_os.py`、`run_os.py --streams`、`verify_os.py`，Python路径同下文。最终[build_os.log](build_os.log)无Warning/Error。Q13 OS随后也用同exe实接原FP32 identity→J→wide→I24消费者，572命令2196480 I24/raw/J/wide全过，见 [CONSUMER_SUMMARY.json](CONSUMER_SUMMARY.json)。ready两64冷完整服务为2601944/3006212，BP为3192075/3744251；上表仍明确保留raw端点。Q11同函数完整分母另见 [q11/README.md](q11/README.md)，记录互不覆盖。这两个普通控制均不构成新机制。

# Gustav 共享权重臂：保留的指定控制结果

已实现 [direct_core.sv](direct_core.sv)，执行 `W=Σrank Q2·Q1` 的完整 C96×3×3→N96 binary-spike AAC，不重新量化、不穿越任何 RNE。与 sibling spatial factor 的全部 P32 输出是同一个函数。[静态界](admission.json) 证明任意原始spike归约前缀均在 `[-125666413,159593676]` 内；仿真每次加法另检查33bit结果可精确存回signed32。

实现按 K 外循环、og 内循环：每C只从1920B源库读一次原生16字，20B窗口服务9tap；每次实际读取一个8输出W32向量，然后更新其所有有效P/T，权重不按t重复读。空源跳过所有og；随实际权重配置生成的整向量live位可跳空W。本矩阵10368个向量全部非零，未把这一机制算成实际压缩收益。每个P/T更新用一拍读psum、一拍用8×32ALU加写；仅一个32B p hold，没有40×8局部acc阵列。480行先显式清零、末尾再完整读取退休。内存读写互斥断言与输出背压保持检查都在最终仿真启用。

**结果。** 15 small（原8真实、zero/one/random/tail/rank正负/图外poison）及held128..191、disjoint4000..4063各64，均执行ready/BP与无reset两遍不同tile命令，共 **572命令、2196480个raw输出，零差**。独立 Python 用原始source×expanded-W重算142个不同fixture全gold；[verify.py](verify.py) 再逐条复核源/W/psum计数、每个FSM状态、实际背压税、配置和启动、重复命令。[SUMMARY.json](SUMMARY.json) 为紧凑完整结果；`results_{small,held,disjoint}_{0,1}.jsonl` 保留每个实际RTL命令。

| 64 tile，第一次连续遍历 | 直接AAC core | factor core | 直接AAC冷service | factor冷service | factor冷service变化 |
|---|---:|---:|---:|---:|---:|
| held，ready | 2984258 | 2079514 | 3093058 | 2179098 | −29.55% |
| disjoint，ready | 3717789 | 2256006 | 3826589 | 2355590 | −38.44% |
| held，BP | 3224582 | 2135005 | 3333382 | 2234589 | −32.96% |
| disjoint，BP | 3999504 | 2315002 | 4108304 | 2414586 | −41.23% |

冷64只在第一条命令装一次静态权重：direct10368拍，factor1152拍；两者每tile另付1536源配置、1原点、1启动。`service=core+static_once+64×1538`，不把每tile都重装W误称持久配置。第二遍完全不reset，权重保持，source/origin重新加载；其core及每项服务计数与第一遍一致。表中factor来自[原始同集raw记录](../spatial_r16_rtl/results_held_0.jsonl)，没有借CPU周期或不同R8函数分母。Gustav臂的终点是最后raw输出，未接I24消费者；它不能当consumer端到端数字。

| ready64 实际服务 | held direct / factor | disjoint direct / factor |
|---|---:|---:|
| native源10bit读字 | 85344 / 170688 | 97920 / 195840 |
| 实际权重读payload B | 9728256 / 662432 | 11707008 / 694000 |
| psum读行（每行256bit） | 1202232 / 61440 | 1540656 / 61440 |
| psum写行（含初始化） | 1202232 / 61440 | 1540656 / 61440 |
| direct AAC / factor Q2 MAC issues | 1171512 / 1174644 | 1509936 / 1246164 |

held中Q2 MAC数量接近直接AAC，而factor还付了102698个Q1 issue；其速度收益主要来自不同归约状态减少psum读写和W供数，不能只说乘加次数下降。factor仍有两stripe源复读、Z扫描与cache状态。负条件也真实存在：原8真实位置有5个factor core更慢；例如tile0为22369对17734，tile19040为9023对5770。zero为8609对5665，tail为8811对5700。两64总收益不等于逐tile普适收益。

**资源界。** 两者执行权限均为8×32ALU、单256bit权重读、1920B源、20B原生窗口、15360B单256bit行读/写psum和同背压日历。factor许可的8个19×13乘法器在direct闲置。direct权重 **331776B**，factor Q1+Q2为 **12096B**，容量相差27.43倍；direct另有1296B live位、32B W hold、32B p hold、32B输出hold、各5B支持/待办掩码和控制寄存器。factor的Z1280B、Q2cache312B和支持位等另按其报告记。这里是同函数、同执行端口权限的展开反事实，**没有证明总SRAM或面积相等**。

配置总线每拍实有256bit：direct静态10368拍=331776B，源+原点每tile1537拍=49184B传输（有效source payload1920B、原点4B）。held/disjoint冷64配置总线各传3479552B，已在SUMMARY计费。源读取按实际10bit字、W/psum按实际256bit行记录；source_allow/weight_allow表示片内读服务停顿，没有未实现的外部在途队列或DRAM延迟模型。无EDA/PPA、GPU、训练、生产修改或Git操作。

复现：`/opt/anaconda3/bin/python3.12 -B run.py` 完成prepare/build/small，通过后执行 `run.py --streams`，最后执行 `verify.py`。使用Verilator4.028，最终 [build.log](build.log) 无Warning/Error。fixture只引用 sibling稳定路径，不重复归档原输入。该实现是普通强AAC控制；本身不构成新机制，新的研究主张仍需另行证明。
