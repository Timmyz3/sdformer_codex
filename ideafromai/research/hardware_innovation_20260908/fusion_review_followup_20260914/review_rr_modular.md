RR modular 独立快速审阅，2026-09-14。

**结论：支持当前两个context、五类共享资源的RTL功能与服务计费，没有发现新的数据/授权错误。** mode2四P低8位加high5修复，相对mode1静态证明tripleP10（超界转精确dualP13）的普通RR强控制，在作者64 tile无外部停顿冷启动实测814321→787603，减少3.281%。这是单段64 tile与指定组合资源约束下的局部结果；该真实段repair=0，不能说它证明了真实重修复流下的性能优势，更不能推成全帧或PPA结果。[64 tile结果][r64]

独立工作读取原始源码、TB和原生数据，重算16 fixtures及67个去重原生tile的raw/J/I24，各318720标量，并从原生事件序列重算Q1发放、high repair、normalize、Q2和所有grant义务；核对作者176个命令（其中64 tile仅重算数据和结果，未重复运行64次RTL）。另复制源码到`audit_rr`加入逐拍断言，新编译16个跨mode命令，raw/J/I24各138240值全部通过。两种证据的输出规模和角色分开记录。[数学审阅][mathsummary]、[独立复跑][rtlsummary]

| 检查项 | 判定 | 具体范围 |
|---|---|---|
| source/W/z/psum/ALU五类共同grant | 支持 | 每拍每类至多一个context获准，不相交请求可同时推进 |
| RR无饥饿 | 支持有条件结论 | 持续eligible的请求最多输一次冲突；永久外部停顿不在此保证内 |
| z写入和ALU结果的原子提交 | 支持 | ZADD、REPAIR_ADD、NORMALIZE_ADD和BASE_MAC都请求z+ALU，统一外层gate后才提交 |
| high repair与normalize不丢授权 | 支持 | 新断言复跑中实际经历1874拍repair拒绝、239拍normalize拒绝，状态与数据保持 |
| 双context是否复制生产者乘加链 | 未发现复制 | 只有顶层8个19×13乘法表达式和8条32位数据加法链；context只产生operand和接受结果 |
| z/source/psum物理单端口实现 | 只支持逻辑服务预算 | 两context有各自数组，grant限制同类总服务；尚未映射成公共SRAM宏或验证物理端口/频率 |
| 同函数强控制 | 支持当前两臂 | 同源、同Q1/Q2、同状态容量、同RR、同消费者、同proof配置费；RR+borrow尚未测试 |

**仍缺少的具体强控制是RR+borrow。** 当前表格只比较普通RR下的triple10与fourP8，不能声称候选优于未测试的RR+borrow；本轮审阅不增加该实验。

**仲裁与公平。** 请求位顺序为 `{ALU, psum, z, W, source}`。eligible要求非零请求，source/W涉及的外部allow有效；随后两个eligible请求资源集合不相交时同时grant，相交时只grant `rr`选中的context，并在该冲突拍翻转rr。资源请求独立于grant生成，grant不会参与自身请求的组合反馈。[仲裁][arb]、[context请求][req]

| 状态 | 申请资源 | 真实访问或提交 |
|---|---|---|
| L_LOAD且在边界内 | source | source_mem读一10bit字进入local_source；padding直接产生0 |
| QREAD / 有效VLOAD | W | 唯一公共Q1/Q2阵列读一个8lane向量 |
| ZCLEAR / ZREAD / ZSCAN / REPAIR_READ / NORMALIZE_READ | z | 一个416bit向量写/读；标量MAC以外的扫描/读修改写准备 |
| ZADD / REPAIR_ADD / NORMALIZE_ADD | z+ALU | 8lane结果与被保留位一起写z，两个资源同时获准后才提交 |
| BASE_MAC | z+ALU | 选择一个z银行中的signed13值，经共同8个乘法器与加法器累计 |
| STORE / DRAIN_READ | psum | 8lane完整psum写/读 |

若某个持续eligible请求输一次资源冲突，rr随该拍翻转到它；下一拍若仍冲突便获胜，若不冲突则两者同时获准，若只有它eligible也直接获准。因此不会被持续eligible的另一context重复挤占。source_allow/weight_allow关闭时请求仍留在原状态，但暂时ineligible；永久关闭allow或result_ready会使整体不结束，这不违反上述有条件公平。输出消费按context0、context1顺序，每批两者同时启动；context1可在DRAIN_SEND等前一tile消费，属于有序退休造成的等待。没有证据显示内部循环等待。[翻转/计费][grantcount]、[共同启动与输出选择][launch]、[批次退休][retire]

**修复和规范化的原子性。** `resource_request==0 || resource_grant`包住整个context状态case，未获grant时只有诊断周期/等待计数变化。z读总线也以grant门控。因此被拒绝的REPAIR_READ不会改z_hold，被拒绝的REPAIR_ADD不会用另一个context正在计算的共享ALU结果写回；NORMALIZE对应两个状态同理。[提交门控][commit]、[grant限定的z读][zread]。共享结果虽然广播给两context，`next_correction`也会组合计算，但只有获准ZADD把correction、低位、pending和下一状态一起寄存。修复之前先进入REPAIR_READ/ADD，完成后才能TIMESEL发下一个事件，没有提前消费pending导致修正丢失的问题。[溢出检测][overflow]、[低位提交和修复][repair]

独立数学模型直接维护精确整数前缀和，令 `low=((z+128)%256)-128`；对每个原生k/t事件计算active P的`low+Q1[k]`是否越过[-128,127]，把“任一lane/P越界”计作一repair，把越界字段数计作repair_fields。该模型不借用RTL的符号位公式。它与所有176命令的修复次数/字段数一致。四个high5字段经同一32位ALU的5bit断carry段累加±1；normalize再按low符号给high减1，将centered表示转回通常signed13表示。signed3 Q1在完整K864上的精确前缀范围[-3456,2592]意味着centered high在[-13,10]，5位足够；Q1后每tile固定10次NORMALIZE_READ和10次NORMALIZE_ADD，`normalization_issues`只计10次ADD，不是整个规范化总周期。[字段运算][fields]、[修复/规范化提交][repair]

新增副本逐拍检查：grant只能授予eligible、双grant资源交集为空、配置proof不与context后端重叠、持续eligible请求不得连续两次被拒绝；被拒绝后state/k/fp/zrow/og/qfill/row/load_xy/pending、acc/z_hold/correction、全部z数组和当前相关psum行必须保持。测试沿用原TB的完整输出和背压检查，额外把两次命令改为1→2或2→1；双context one、extreme±1，以及从159/19197开始的3个原生tile均在外部停顿下通过。实际repair拒绝1874拍、normalize拒绝239拍，证明这些保持断言不是只在零等待路径上空过。监视寄存器仅存在审阅副本，不计入候选资源或性能。[断言和复跑源码][rtlcode]

**数据链、端口和状态。** 顶层Q1为8×864×3bit=2592B，Q2为8×96×16bit=1536B，只有一套；所有W访问由weight_owner选择到同一输出总线。8个19×13乘法表达式和8条32bit加法链也只在顶层定义，支持13/10/8/5bit边界切断；配置Q1时直接复用这些链累加正负界，没有每context额外的proof加法器。[共享权重/界/后端声明][sharedstate]、[共享乘加][alu]

每context保留source=1536×10bit=1920B、z=8×10×52bit=520B、psum=8×480×32bit=15360B、Q2局部cache=128B、z_hold=52B、acc=32B、correction=8B，以及local_source和live/pending等控制状态。mode1和mode2获得同一份union状态/寄存器和416bit z服务预算。新增第二context确实增加这些存储，比较两臂时增加量相同；不能宣传为两个context只多几个调度位。[context存储][contextstate]

source/z/psum数组分别在两个context内，仲裁保证同类服务至多一个，不等同于已经综合为一个物理存储宏；BASE_MAC仅激活选中z银行，向量读/写激活8银行同一地址，未发现第二个隐藏z读地址。local_source的多位置组合gather与Q2缓存属于两臂公共的局部保持/复用，五类端口模型并未把所有局部寄存器读取都称为外部单口。context中的地址/计数器加减、诊断popcount不属于生产者数据加法链，但仍有逻辑面积，当前没有综合面积证据。完整I24消费者另外有8个32×32乘法器及8个64bit加法器，并按序共享于两个context；“8乘法器”只能指生产者。[z端口][zread]、[公共消费者实例][consumer]

**proof与冷配置。** 同一次冷配置1848拍，包括864 Q1、96 Q2、864 k_live、24消费者系数。每个被接受Q1配置拍都通过共享ALU累计正/负13bit界，共864次proof；第一拍清界、最后一拍加入第864列，之后才加载余下参数、source并启动context。bounds只保存一套8×26bit=26B；mode1在任一rank越出[-512,511]时整个tile转精确dualP13，mode2也获得同proof装载过程。warm不重复配置或proof，两个mode需要的静态项相同，跨mode无需额外参数，已被本次1↔2测试覆盖。[proof执行][alu]、[配置和启动][topcfg]

**费用与已支持的性能结论。** 独立公式令K为实际非零Q1列数，Q为有输入事件的有效Q1列数，U为当前模式合并后的Q1发放，M为Q2有效MAC，R为repair次数，N为NORMALIZE_ADD次数。每tile不含等待的context自身状态拍为：

```
5427 + K + 2Q + 3U + M + 2R + 2N
```

对多tile累加此式，再加实测source/weight/arbitration/output等待，逐项等于`core_cycles`。**core_cycles是两个context生命周期之和，不是系统经过时间。** 系统经过时间来自顶层真实逐拍状态计数：

```
total_cycles = window_cycles + launch_cycles + static_words + parameter_stalls
             + source_load_words + origin_words + source_load_stalls + 1
```

五类实际grant与数据义务的等式分别为：source=core_source_words；W=core_weight_words；z=z_vector_reads+z_scalar_reads+z_writes；psum=psum_reads+psum_writes；ALU=proof+first_issues+mac_issues+repair_issues+normalization_issues。一个冲突只有一个输家，故`conflict_cycles=core_arbitration_stalls`。本审阅既由原始source重建这些义务，也核对RTL计数之间的守恒，避免只靠两个同时写错的计数器互相证明。[独立原生重算与等式][mathcode]

| 原生128起64 tile | RR triple10 mode1 | RR fourP8 mode2 |
|---|---:|---:|
| 无外部停顿，cold total_cycles | 814321 | 787603 |
| 无外部停顿，warm total_cycles | 812473 | 785755 |
| 有外部停顿，cold total_cycles | 884397 | 858814 |
| 有外部停顿，warm total_cycles | 882629 | 856904 |
| Q1发放 | 76424 | 60113 |
| Q2 MAC | 198720 | 198720 |
| high repair | 0 | 0 |
| NORMALIZE_ADD | 0 | 640 |
| 无外部停顿，z grant | 354768 | 323426 |
| 无外部停顿，cold ALU grant含proof | 276008 | 260337 |
| 无外部停顿，冲突拍 | 180553 | 174163 |
| 无外部停顿，normalize被拒绝拍 | 0 | 693 |

这64 tile的无外部停顿cold/warm收益为3.281%/3.288%，有外部停顿为2.893%/2.915%。source/W/psum访问和Q2 MAC相同，z/ALU实际grant下降支持四P合并在这个受限RR共享执行中仍兑现部分收益。四P带来Q1发放少16311次，normalize新增640次ADD和640次READ；总体延迟仍受共有Q2、消费者与调度影响，不能用单context算术相减来替代此表。[64 tile结果][r64]

真实64 tile上没有high repair；高修复争用由角落单独验证：双context `one` 无外部停顿cold为95499→61659，候选repair300次、repair被拒绝600拍、normalize被拒绝38拍；`extreme±1` 候选各repair200次、repair被拒绝400拍，且mode1使用范围fallback。低活动输入仍有代价：real2单tile cold12213→12233，zero12102→12122，均多20拍normalize。这里支持固定策略的正/负输入依赖结果，不支持所有输入提速。[fixture结果][fixtures]

普通RR已经放进强控制，本项不应把新增两个context或RR仲裁本身当成X。现在得到的具体证据是：相同乘加与共同端口服务许可下，四P低位更新、同步义务修复和末尾规范化确实能被完整调度并产生可测局部收益。溢出拆分、carry切断、轮转仲裁的更强新颖性命题没有由这次功能审阅建立；作者PLAN记录的最近邻全文也未在本审阅补取，因此不作“文献没有同样细节”的断言。本文没有复现任何外部作者完整架构，也没有把局部资源审阅当成文献级复现。资源公平在当前固定布局与时序模型内获支持；同步SRAM时延、面积/功耗、全帧吞吐及其它强调度/表示仍未证实。[预先方案][plan]

复现：在`audit_rr`执行 `/opt/anaconda3/bin/python3.12 audit.py`（CPU原始数学核对，读取已存在结果）和 `/opt/anaconda3/bin/python3.12 rtl_audit.py`（Verilator4.028重新编译带断言副本，16个短命令）。只写本审阅报告和独占`audit_rr`目录；作者源码、生产文件和旧实验树未改。

[r64]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/rr_modular/results_64.json
[mathsummary]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/audit_rr/math_summary.json
[rtlsummary]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/audit_rr/rtl_summary.json
[arb]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/rr_modular/interleave_stream.sv:123
[req]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/rr_modular/rr_context.sv:67
[grantcount]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/rr_modular/interleave_stream.sv:291
[launch]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/rr_modular/interleave_stream.sv:188
[retire]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/rr_modular/interleave_stream.sv:403
[commit]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/rr_modular/rr_context.sv:168
[zread]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/rr_modular/rr_context.sv:82
[overflow]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/rr_modular/rr_context.sv:131
[repair]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/rr_modular/rr_context.sv:218
[fields]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/rr_modular/rr_context.sv:96
[rtlcode]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/audit_rr/rtl_audit.py:7
[sharedstate]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/rr_modular/interleave_stream.sv:75
[alu]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/rr_modular/interleave_stream.sv:132
[contextstate]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/rr_modular/rr_context.sv:25
[consumer]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/rr_modular/interleave_stream.sv:227
[topcfg]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/rr_modular/interleave_stream.sv:380
[mathcode]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/audit_rr/audit.py:13
[fixtures]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/rr_modular/results.json
[plan]: /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/fusion_review_followup_20260914/rr_modular/PLAN.md
