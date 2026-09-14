已将正count8／psum复用mode20和固定时间配对mode21接入**实际FP32 identity→J20→I24消费者与双context RR**。同一模块、同416bit z服务下，mode21在两套64tile冷启动无BP分别为 **772611／810788拍**，比当前借用consumer64主链＋RR强控制786421／830886少 **13810拍（1.7561%）／20098拍（2.4189%）**。这些是完整wrapper实测，包含静态表、原生source、identity、所有raw/J/I24及背压义务。

所有七臂预留共同class/rep与控制状态，Q1/Q2、source、z及psum容量保持原RR预算；没有复制权重、增加计数ALU或第二psum/z端口。新增class2592B、代表96B、固定排列40bit和地址/算术mux仍是真实增量，不能由共同union模块推导裁剪后等面积。原始[接口与资源合同](PLAN_RESOURCE_CONTRACT.md)先于SV修改。

|64tile完整经过时间|0 dualP13|1 三P10/精确回退|2 四P8＋repair|3 借64主链|4 借64＋宽链组RR|20 原序count|21 固定配对count|
|---|---:|---:|---:|---:|---:|---:|---:|
|128–191冷，无BP|815406|814321|787603|786421|786421|787753|772611|
|128–191暖，无BP|813558|812473|785755|784573|784573|785008|769865|
|128–191冷，有BP|885599|884397|858814|857282|857282|859072|844899|
|128–191暖，有BP|884004|882629|856904|855357|855357|856377|842204|
|4000–4063冷，无BP|879020|868255|831996|830886|830886|828782|810788|
|4000–4063暖，无BP|877172|866407|830148|829038|829038|826037|808042|
|4000–4063冷，有BP|955834|945384|909177|908472|908472|905234|888404|
|4000–4063暖，有BP|954452|943459|907252|906052|906052|902609|885709|

冷命令加载Q1/Q2/k_live/consumer系数1848拍；mode20另付897拍class/代表/ngroups，mode21再付排列1拍，合计2745／2746拍。每tile另付1536source、1origin、实际等待和launch。两遍之间无reset，参数常驻；首次由普通mode转count或21也实际补装未驻留的静态表。静态常驻仅针对同模型；换模型须reset并重载，当前没有在线模型版本失效接口。

本轮直接迁移[上轮count20/21原型](../../transfer_adapt_20260914/pair_psum_overlay/temporal_pairing/README.md)，没有新增排列或调参。顺序固定为`[8,2,6,3,9,4,1,5,7,0]`，源仍原始T10字，在RTL的L_GATHER中选位，完整Q2物化后在原psum读地址逆映射，再以原T交给identity消费者。校准只来自同帧0–31；128集合有halo共享，4000集合与校准输入不重叠，二者仍不是跨帧或跨序列验证。离线校准DP不在此RTL服务内。

原序mode20在128集合冷启动比modular慢150拍、比borrow RR慢1332拍；4000集合则比borrow RR快2104拍。这个失败臂完整保留。实际瓶颈是正计数块更新和退休占用额外psum服务，并增加两context冲突；计数减少z更新，不能自动转成完整consumer收益。已有mode21的固定配对适配减少了同一块内的重复读写，在两套完整接口上转正，没有把旧raw阶段百分比相乘。

|冷无BP的真实服务|128：mode4|128：mode20|128：mode21|4000：mode4|4000：mode20|4000：mode21|
|---|---:|---:|---:|---:|---:|---:|
|count更新块|0|52595|42026|0|64664|52711|
|count退休乘法issue|0|19140|19140|0|21549|21549|
|metadata读取|0|25237|25237|0|30383|30383|
|共享32bit ALU grant，含proof|199584|272195|261626|211248|298548|286595|
|共享W grant，含metadata|31381|33284|33284|36527|38541|38541|
|共享psum grant|61440|172038|150606|61440|196668|172484|
|共享z grant|322146|241952|241952|361094|258856|258856|
|context冲突拒绝|173079|196744|187750|193317|224829|213416|
|consumer join等待|469436|469871|454728|513901|510900|492905|

mode20→21没有改变Q2 MAC、count退休乘法或metadata数量。128集合减少10569个更新块与287个退休块，其内禀服务减少`2×10569+2×287=21712`拍；RR冲突少8994，raw holding等待多420，两context周期和净少30286。真实wrapper window少15143拍，加排列配置1拍后净少15142。4000集合对应内禀少24498、冲突少11413、raw等待少79，context周期和少35990；wrapper window少17995，服务净少17994。context周期和是重叠占用，不能再与wrapper总时间相加。

短参数生命周期仍有负例。真实八个独立单tile冷命令总计，borrow RR为131141拍，mode20为137876，mode21为136408；相应暖命令为116357／115916／114440。每次冷装897／898拍表的成本不能省略。159起跨row三tile冷无BP为41036／41858／41066；19197起跨row三tile为31832／33252／33185。长流正结果不替代这些短流负结果。

实际接口和资源实现见[rr_context.sv](rr_context.sv)、[interleave_stream.sv](interleave_stream.sv)与[resource_contract.json](resource_contract.json)：

- top只有8个signed19×13乘法表达式和8条32bit carry chain。G_MAC的逐lane scalar、coefficient和负数修正一起跟随获准owner；普通Q2广播原scalar。count的C_ADD复用同8条链的四段8bit加法，不存在私有计数加法器。
- class、代表和Q1/Q2都是top单份静态表。MREAD/G_QREAD与QREAD/VLOAD竞争同一个W服务，受实际compute_weight_allow和context RR限制；metadata读既计数又会等待，没有侧门端口。
- 每context仍520B z、15360B psum、1920B source。z每银行一个授权读地址，G_MAC完整52bit写回由此前z_hold形成，未请求bit-write口。即使count只更新26bit，也占一次共同416bit许可。
- count借各自psum前160行共5120B，八银行按class选地址；C_CHECK/G_READ付psum读，C_ADD原子申请psum＋ALU，G_MAC原子申请z＋ALU/乘法。全局仍每拍至多一次psum服务，首触无读时只返回零，并未读取旧输出。
- 每context直到自己的480个I24全部退休才释放所有权；wrapper等当前批两消费者完成才重载/重启。count阶段结束后Q2覆盖全部480行再drain。输出holding和consumer的FP32→J20、两次64bit主和、原RNE26及signed24饱和边界保持实际执行。
- consumer只有原8条64bit主链和8个32×32乘法器。mode3/4继续真实借用并仲裁这同一主链；mode20/21本轮使用producer32，不凭空占用另一套宽链。

**928条RTL命令，raw、J20、I24各17925120个值全部通过**。23小fixture涵盖真实输入、全零/全一、padding poison、signed3正负极值、FP转换与RNE/饱和边界、count255/256、high127/128、混合直接/计数、零Q2及独特原T签名；另有恒定邻接双tile、跨mode首次装表、跨row和两套64冷暖/BP。新增assert检查未获grant不能读写存储、资源不能双grant、live psum不能配置/重启、count只能访问前160行、count8不溢出、退休后不得重新写count、所有480个输出覆盖后才drain。

[verify.py](verify.py)独立重建154个去重fixture/native tile的raw/J20/I24，各591360值，与原完整gold一致；逐K重建合法class、正计数和signed退休，检查每条命令的所有公开算术/端口/周期义务。最大实测count255。328条旧RR控制记录的原诊断字段全部复现，包括旧128集合mode1/2/3/4 cold/warm/BP。TB曾拒绝把非空间常数的edge源当同质邻接窗，随后限定该额外双tile控制的适用输入；单tile边界覆盖未删，RTL数据未改写。

本结果证明该表示/布局可以在真实消费者及RR争用下保留有限净收益，尚不证明论文标题新颖性。普通RR、分组计数、psum阶段复用、packing或时间排序各自已有先验；固定排列的跨序列稳定性、模型换参接口、全19200tile、面积/Fmax/能耗和公共SRAM宏映射未验证。旧transfer_adapt和production只读，本目录未做EDA、训练、main.tex或Git提交。

结果保留为每命令一行：[results_small.jsonl](results_small.jsonl)、[results_short.jsonl](results_short.jsonl)、[results_held.jsonl](results_held.jsonl)、[results_disjoint.jsonl](results_disjoint.jsonl)，最终核验见[verification.json](verification.json)。复现使用`/opt/anaconda3/bin/python3.12`依次运行`implement.py`、`prepare.py`、`run.py --stage small`、`run.py --stage short`、`run.py --stage held --skip-build`、`run.py --stage disjoint --skip-build`和`verify.py`；构建为Verilator4.028 `-Wall --cc --exe`后独立make。
