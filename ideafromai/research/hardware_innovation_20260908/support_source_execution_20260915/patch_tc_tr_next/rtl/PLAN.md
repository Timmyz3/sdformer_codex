# 完整 patch U→Z→V→A / U→Z→A→V：固定有限执行点

先补 A：加载已经冻结的 ordinary_r32.npz / regional_r96_a48.npz，完整执行 P4×T10×K864 的真实 0/1 源到 U8→Zi15→Vi→Aq14→sn2 gate。B 是 A 放到 latent 域减少算术，但 Q 扩大、TC/TR gather/布局与消费者费用尚没有有限端口实证。严格同函数内比较 VA / AV；ordinary R32 同样有 AV 和 TC 权利。regional masked-dense 与 TC 同参数，不跨两学生说函数相等。普通矩阵换序及 h 字段排布都是借入底座，不是新公式。

唯一执行预算：8 个 signed16×32 multiplier 与8个48bit ALU；U 只门控 AAC，V 用8路有符号移位加，A 用同8路 MAC。不增加80路时间算术。任意顺序静态界为 Z15、Y32、Q30、最终U47以内；实际存Z16、phase scratch48和acc48，无饱和/RNE。阈值已从真实偏置/正BN gain冻结为signed48整数；theta静态吸U，不再编译/量化。

外部为**一个总128bit请求/返回 service、最多一在途**；ready/BP都实际计费。这比旧8×128吞吐窄，已获根确认，目的为该固定点下同权比较。共同256KiB外部 bank图像，U每字16个相邻signed8，V一个字同rank的H8字段（每字段16bit只用nz/sign/shift6bit，h%8定位）；source每字3个40bit/K。desc/mask/A/tau/U/V/source均来自真实请求。P4命令首加载desc、对应region mask和13个Aword；外存最初写入bank图像仍不在本组件口内，不能称完全冷外存服务。

内部共同容量：Z 8bank×480×16=7680B；phase scratch 8bank×480×48=23040B，只存VA的Y或AV的Q，不同时放第二完整输出阵列。最终acc8×48=48B；source行cache16B、U行cache16B与组装U8 holding8B。每H8使用共同96×64bit V cache=768B（每lane byte含6bit系数，2bitpadding），经实际128bit口逐rank装入，VA/AV同权；避免把AV做成逐P/T重复外读V的弱臂。A寄存200B、tau当前H8 holding48B。最终40×96bit gate pack480B逐H8写全再有序输出。TC map最多24×5bit=15B，必须由本命令实际mask逐TC扫描构造，dense用identity索引；各持有/地址/有效位另按实际源码列账。

U采用K主序：真实40bit支持码扫描、空word跳过；8latent组经真实U行cache gather，再逐有效(P,t)作Z read/add/write。TC相邻compact组可来自不相邻global TC，额外U字读取实际付费；dense具有同cache及全死组跳过权。mask四rank共用、跨T10不变。Z全初始化对应命令实际物理行；TR使用同一TC map恢复V的global rank。VA与AV均使用H8 output-stationary局部acc及同V cache；A阶段以8个rank或8个h并行。零输入/零系数可跳过对应MAC但扫描/读取费用仍计。

先首capture每区域一个真实P4（8例）加zero/dense/交错符号支持诊断；两函数×dense/TC×VA/AV×ready/BP、无reset换命令。通过后跑四份既有capture的固定64个P4（全256，不新增选择）。Python int64独立从真实source words重算Z/Y/Q/U/gate；C++不喂中间数，检查实际Z/scratch写入在阶段末全量等于gold、最终U与96bit gate，另核地址/请求/hold和TC映射。按实际分段服务去留，绝不拿少A MAC当周期。无训练、重新量化、AEE、生产改动或EDA；regional已知局部0.9552%门翻、Y约20%误差保持质量边界。
