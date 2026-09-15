# 固定自由 U3M 的共同双 context 执行合同

A 是旧 bitmap_rr 的真实共享请求/原子授权、唯一 producer 8×32 ALU 和 8×signed19×13 mult、唯一 FP32 identity→J20→wide→I24 消费者，以及已冻结 unconstrained phase3 三项函数。B 是旧空间为单 context/256bit Z、R8 为双 context/416bit Z，旧服务百分比不能跨资源排名。本任务只接自由 U3M 一个冻结函数/布局，不加其它分解或量化，不声称三乘法或RR首创。

顶层只一份 Q1/Q2 与读取 mux、八个乘法器和八条32bit carry chain。两个 leaf 是状态/存储 context，不含数据乘法器/ALU。资源请求位 source/W/Z/psum/ALU/wide 原子授权；两个 context 请求互斥资源可同拍前进，同种端口只能有一个 owner；冲突沿旧 RR 翻转优先级。每 context source1920B，p15360B；共同 Z 8bank×40row×52bit×2=4160B，candidate 只用各bank低32bit，unused20bit显式清零，R8 同40rows容量和使用权。单W256、Z416、p256，不增加旁路系数或SRAM读口。

每 context cache312B、3M累加总96B（相对普通acc额外64B）、transform tail32B、Z holding32B、Q1 holding8B、源窗口20B、源mask10B、output32B、support80B、rank8bit、block/remaining48bit、pending40bit和控制寄存器均明列。Q1 4608B + physical_coeff3 7488B、静态live各72B在top各一份。R8给予同中间和holding/cache预算，但未声称它已最佳利用扩容或各裁剪实现等面积。

唯一消费者保持原8×32×32乘法器与8×64宽链。除producer32执行外，允许相同Q1 ZADD借消费者64链，phase3低30bit切15bit场，R8切13bit场；沿旧borrowRR消费者与producer原子仲裁，不复制宽链。此为同布局的执行权限设置，所有wide冲突、consumer等待均收费。先保留不借设置，再实测借用；不根据结果扫阈值或改量化。

虚拟tile索引0..N−1来自manifest。每tile实际请求1536个10bit原源字（含边界poison由核按origin过滤），随后独立source-origin32bit握手1拍，source/origin BP全付；每batch最多两个context加载完同时launch。原始source仅来自真实捕获，TB不给Z/D/M/raw/J。消费者按原tile顺序480行运行，只有完整I24最后行退休后允许回收该context；沿旧RR先保守整batch回收，不增加免费源加载并行。cold含一次Q1 576+Q2 576+consumer24=1176配置与go/start；warm保留静态参数，连续换源不reset。

small15→两64→18seq×2tile，全部raw/Z/D/J/wide/I24、BP与无reset回放。gold引用已独立核验的自由U3M真实source/id与physical三项输出，重新核查系数/原source对应关系；36seq绝不借Q11或R8输出。R8代理独立重算同前缀R8新gold并以同接口重跑count21/bitmap7/borrow4。本目录旧树只读，无训练/GPU/EDA/Git/生产修改。周期/容量仅为模型证据，不宣称等频率PPA或最优。

## 共享许可与实际占用补表（实施期间与R8代理对齐）

共同许可cache至少384B/context（24×8×16），R8实际声明此阵列但执行只用前8向量，candidate实际Q2 payload24×8×13=312B；candidate没有把空余72B当零费新增数据阵列。共同holding必须容纳R8的52B z_hold、32B count_hold、32B bitmap_acc，以及candidate的96B三M、32B tail和32B Z hold等；各臂在原有寄存器中执行，未声称这些不同比特用途可以同周期无代价复用。

静态/辅助容量许可按两臂union提供：候选Q1 4608B、physical Q2 7488B；R8 Q1 2592B、Q2 1536B、bitmap plane2592B、class2592B、representative96B和40bit排列，consumer系数两臂各768B。总系数/辅助容量上限按17376B+consumer768B+排列5B=18149B提供（主Q1/Q2共12096B，外加两类辅助5280B）；并单列live位：candidate两个576bit在top各一份、合计144B；R8 k_live864bit/v_live96bit在每context各120B、两context合计240B，plane_live162bit在top一份；R8每context另有count_live80B、bitmap80B及live/pending等，保留其原资源账。它是同容量许可和同服务权，不是把两个函数全部参数同时驻留，也不是已经证明两种裁剪电路等面积。候选只加载所需1176向量，R8的各静态/辅助配置按原协议实付。

Z阵列声明4160B/双context，但candidate真实读/算的是每52bit bank低32bit，所以是256bit有效payload独占416bit服务授权；高20bit每次写明确补零。不得写成观测416bit有效算术吞吐。R8保留完整52bit字段服务。这一处未优化利用空字段，是当前固定布局的限制。
