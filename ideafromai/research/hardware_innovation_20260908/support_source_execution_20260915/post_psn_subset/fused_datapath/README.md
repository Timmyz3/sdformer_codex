# Post-Y T10 子集表：合并组合数据通路与寄存边界

**32real 的 PSN 服务周期为 full148195、cert97424；同一新数据通路上的 cert 节省34.26%。父80ALU布局 full285126/cert428572 的判界税已被这次真实RTL适配消除。相对异资源 native96MAC 的118344服务周期，cert少17.68%，但长组合路径、面积与频率均未测，不能称其赢得 native 整链或 PPA。** 本版已冻结，原37病例296命令及新增 signed24 全范围诊断8命令全部通过，共304命令。原32real主表未混入诊断。

A 是普通两组5bit distributed arithmetic 子集表，B 是父布局每次判界需四拍共同ALU，X 只改变固定的组合求值和寄存边界。本次完成了“已迁移布局负结果→定位判界/双拍prefix税→增加明确计费的组合算术→再次实测”。没有重新发明 DA 或精确上下界，也没有完成 BitL 全部机制迁移。没有训练、EDA、功耗或新 AEE 测量。

输入、算术与接口

沿用[父 cases.bin](../cases.bin)及其[独立算术核验](../prepare.py)：32real 是8个投影源tile×4个H96函数，另5个原诊断。Y由真实S/W独立计算并与原数据逐值相等；本核边界仍是 **post-Y**，DUT经真实2304bit输入握手将320行写入持久Y存储。未连接实际FC1最后写者。U和gold只留在TB，未进入DUT计算。

后级A为实际signed16[10,10]，100项全部非零，表中子集和范围[-19657,11516]适配signed16。Y为signed24；U、prefix和上下界为signed48，无中间RNE。正gain使用U>=tau，负gain使用U<=tau，constant gate优先。真实32病例通道为正gain，负gain/constant/tie来自标明的定向诊断。

调度组为H8×T10，共80个输出，每P有12组，每命令32P。Y单共同地址读口逐行形成T10×H96暂存，RTL从实际Y生成每H8共同指数e；不接受CPU指数或外供plane。每组以符号头v=−Llo−Lhi开始，再每位做nv=2v+Llo+Lhi。令m为本次消费位之下的剩余位数，界为lo=(nv<<m)+N(2^m−1)、hi=(nv<<m)+P(2^m−1)。80门全部锁定才能提前完成；full仅最后一位采门，输出全部完整U。cert的out_u只有部分prefix，**不承诺完整U**。

GROUP一拍完成符号头和tail初始化。每个PLANE_SUM一拍完成两次prefix加法、并行lo/hi加法、比较及门锁定；tail按(tail−P或N)>>>1递推，实际差必须为偶数。只有最终结果寄存一次，所以可在控制周期上成为每位一拍；物理代价是更长组合路径，不是将父80路ALU免费做两次。旧SIGN_SUM/SIGN_NEG/PREFIX和四个BOUND状态在全部记录中访问为零。

同新硬件 full/cert 与资源变化

| 项目 | 当前实现；两臂同权利 |
|---|---|
| prefix算术 | 80lane×两级48bit add/sub，共160个显式位置；其中10个第一级还分时构建LUT/P/N。 |
| 判界算术 | 80个lower＋80个upper，共160个额外48bit加法位置；不与同拍prefix共享。 |
| tail算术 | 20个独立48bit减法位置，按t共享给8h，GROUP用于初始化、plane用于精确递推。总计340个显式48bit加减位置、0乘法器。该数是RTL结构账，不是映射后单元数/面积。 |
| 比较与移位 | 160路signed less/equality路径；80路48bit界操作数变长移位，20路tail初始化移位；指数比较/优先编码、H8最大值选择和80门全锁定归约另计。 |
| 子集表 | 一份1280B，20bank×32×16bit；每bank8个不同地址的32:1寄存器读mux，共160个16bit mux。没有8份数据副本，也不是宣称单口SRAM能给8读。表构建复用同mux。 |
| Y持久/暂存 | 92160B持久Y，单共同2304bit行地址；每命令实际320写＋320读。T10 ybuf2880B，较native单行288B多2592B；H8选择需80个24bit的12:1mux。 |
| 参数/状态 | A200B、tau5760B、flags144B；80tau并行选择网显账，未称单口tau SRAM。v480B、P/N/tail240B、gate/locked20B；指数60bit另计。删除父dot480B和下界标志20B。数据数组合计103164B，控制/计数器/观察线另计。 |
| 冷构建 | A13个128bit词；64行160bitLUT写：2行零、62行共同10ALU相加。P/N初始化1＋20拍，冷构建85拍。full/cert共同初始化P/N，允许后续warm切模式驻留；full无逐位判界等待。 |
| 参数服务 | 一个128bit req/rsp口、一次一笔；cold503词=A13＋tau480＋flags10；warm490词。warm仅保留A/LUT/P/N，tau/flags和Y仍重装。ready每词请求/响应各1拍，BP拒绝/延迟均计费。 |
| 输出服务 | 每命令384次80门握手，顺序p→hgroup，lane映射t×8+j；拒绝时valid、地址、门、U保持。当前没有将此接口打包为逐t H96行。 |

组合关键路径为LUTmux→两次48bit加法→变长移位→界48bit加法→signed比较→全锁定归约。full/cert在同一个新核上比较；parent80ALU和native96mult16×24＋96ALU均为**不同资源参考**。不能从340/80推出面积倍率，也不能从少乘法器推出更省面积或更高频率。

32real 实测周期

PSN service从PINIT到最后OUTPUT握手，含Y实际读与输出等待；不含参数、LUT/P-N构建、Y交接写、go/done。cold/warm总计包含这些项目及每命令1拍go。BP是固定请求拒绝、响应延迟、Y有效间隙、输出和done拒绝日历；跨核接口不同，BP比较仅是该日历参考。

| 执行臂 | ready service | ready cold总计 | ready warm总计 | BP cold service | BP warm service | BP cold总计 | BP warm总计 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 父80ALU full | 285126 | 330342 | 326790 | 291982 | 291918 | 371790 | 367225 |
| 父80ALU cert | 428572 | 473788 | 470236 | 434094 | 434082 | 513907 | 509384 |
| 新fused full | 148195 | 193411 | 189859 | 161208 | 161144 | 241017 | 236444 |
| 新fused cert | 97424 | 142640 | 139088 | 103265 | 103292 | 183081 | 178598 |
| 原native96MAC，异资源 | 118344 | 不同边界 | 不同边界 | 122550 | 不同边界 | 不同边界 | 不同边界 |

新full相比父full服务周期减少48.02%；新cert相比父cert减少77.27%。关键同硬件控制是新full→cert减少34.26%。完整native FC1→PSN ready184184/BP212665来自[原CSV](../../../support_lut_execution_20260915/rtl_cycles.csv)，不能直接减去本叶post-Y总计以宣称端到端替换收益。[SUMMARY.json](SUMMARY.json)保留所有计数、父聚合和native参考，[results_all.jsonl](results_all.jsonl)每命令一行。

| ready service分摊，32real | full | cert |
|---|---:|---:|
| GROUP符号/tail初始化 | 12288 | 12288 |
| 每位一次合并求值 | 112355 | 61584 |
| P初始化＋Y实际读取 | 11264 | 11264 |
| 80门输出握手 | 12288 | 12288 |
| 合计 | 148195 | 97424 |

full112355→cert61584 planes，11951/12288组提前退休，逐病例与父真实RTL及其独立CPU退休深度相等。机会未变，本次去掉了父244988拍四状态判界以及prefix/符号的额外寄存等待。ready cold另有34912参数/表/P-N拍、10240Y交接写、32done、32go，共45216拍；warm少3552拍。BP输出等待取决于实际到达相位，已计入，不能由ready周期比例替代。

时钟 break-even 与Claude叶的边界

若对同一服务边界比较实际时间C/f，候选获益条件是f_new/f_ref>C_new/C_ref。

| 服务比较 | 需要的最低频率比 |
|---|---:|
| 新cert/native，ready | 97424/118344＝82.323% |
| 新cert/native，BP cold固定日历 | 103265/122550＝84.264% |
| 新full/父full，ready | 148195/285126＝51.975% |
| 新cert/父cert，ready | 97424/428572＝22.732% |

因此只要新核实际频率比native下降超过17.68%，ready服务周期优势就会被吃掉。**这些是条件阈值，不是测得Fmax，也没有面积/功耗条件。** 本阶段未综合或测时序，无法宣布其物理速度胜出。

只读参考[Claude cert_gate_bitl.sv](../../../claude_fusion_trials_20260914/t10_rtl/cert_gate_bitl.sv)采用H1×T10，TB外供sign/e/plane，LUT在initial离线构建；其[T10原报告](../../../claude_fusion_trials_20260914/results/T10_REPORT.md)使用不同4条trace。本核确实达到同类型“一拍符号头＋每plane一拍”的**控制叶**：每组full10.14347拍、cert6.01172拍；32real cert叶共73872拍。真实Y读/P初始化和80门输出再加23552，成为97424服务周期，尚有参数与交接写成本。没有继承Claude历史17%收益，也没有称其离线初始化/外供指数已自动解决真实接口。

验证与新增全范围诊断

[run.sh](run.sh)用Verilator4.028 `--cc --exe --unroll-count 512`＋make复现；[implement.py](implement.py)只读父源码，输出本目录独立核/TB。原37病例唯一初始reset后full/cert×ready/BP×cold/warm连续执行，共296命令；另进程的一个全范围病例同样连续8命令。加载被拒时请求稳定，Y有效提前到达时保持至YLOAD接收，输出被拒保持地址/门/U；只有输出完才done再启动。

新增[prepare_fullrange.py](prepare_fullrange.py)使用实际A、原tau作底并定向置等号阈值，直接构造signed24 Y域诊断：包含−8388608、8388607、0、±1和跨t交替。它不是从真实binaryFC1生成的分布样本，单独记录为诊断。U用独立int64矩阵乘法，范围[-125825203769,112176373235]，没有截断范围外数据；384组实际指数全部24。64正/32负gain、6constant，850个U==tau，其中非constant负gain等号240个。full9216/cert4714 planes、348提前组与独立“直接高位截断MVM＋包络”模型相等。记录见[fullrange_inputs.json](fullrange_inputs.json)、[results_fullrange.jsonl](results_fullrange.jsonl)。

| 304命令实际核验 | 次数 |
|---|---:|
| 最后gate | 9338880 |
| full完整U | 4669440 |
| 真实Y存储读回 | 9338880 |
| cert上下界包络 | 48334080 |
| runtime指数 | 116736 |
| tail值逐项 | 6041760 |
| tail递推差值及偶数性 | 5952320 |
| 冷LUT写值 | 97280 |

最后门/fullU全部零差。cert out_u未作为完整U检查。诊断覆盖signed dense、zero、onehot、多项逃逸、negative gain、constant优先、tie及完整Y24端点。TB断言所有输出80门均已锁定；父为Verilator4.028宽归约问题采用的32/32/16全1比较保留。[summarize.py](summarize.py)核对每条状态和端口账，原296条的planes/early/参数/Y/表访问逐条与父相同；新增8条退休深度与独立高位模型相同。没有将CPU工作量代替RTL周期。

仍未完成的共同接口

当前80门输出布局按H8×T10，原消费者需要逐t H96。真实接入需至少一组960bits＝120B门打包状态或有费复用，12组写入、原序读取及全部handshake；本版尚未实现，不能算免费120B适配器。实际FC1生产者的最后Y写/所有权交接和后续H96消费者也尚未连接。

参数服务亦尚未归一化：同5760B tau，当前每128bit词放2个48bit值，付480词；[native loader](../../../support_lut_execution_20260915/support_fc1.sv)密排付360词，差120词，ready下差240个加载拍/命令；flags当前10词对native密排9词。上述PSN service均排除参数加载，但整链对照必须在共同loader上处理这些差额。A/tau的共同模型驻留只能按实际装载一次计费，不能一边宣称免费共享一边重复收账。

本版只给出面积/时序待证的正服务周期候选。完整BitL、固定A常数MVM/CSE、源端到H96门消费者整链、以及映射后频率/面积均未闭；不将这些尚未完整迁移的接口写成失败，也不继续扫参数或增加第二种数据通路。
