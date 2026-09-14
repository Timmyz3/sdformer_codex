# D3共同资源与执行边界

三种调度共用同一个 `interleave_stream`，内部两个 `thread_context` 只保留执行状态/存储/控制；数据ALU和乘法都外置，不能把双实例视为双算力。原生C96/K864/R8/N96/T10、真实FP32→J20→I24函数不变。

|实体|共同预算与物理接口|
|---|---|
|Q1/Q2阵列|top唯一Q1 8×864×3=2592B、Q2 8×96×16=1536B；一个共享权重地址/拍。两个context没有Q1/Q2阵列副本。|
|source|两个private1536×10bit tile bank（3840B总计）；每个context有16×10局部窗/四10bit读mux。全局至多一个源SRAM读grant/拍。配置时两个源按序装入，未和compute重叠。|
|z|两个private8×20×26bit bank（1040B总计），每context208bit向量接口/26bit标量选择。全局统一z类仲裁，每拍至多一个获准vector读、vector写或scalar读。未获准不执行对应存储读取/写入。|
|p_mem|两个private8×480×32bit（30720B总计），各8×32acc和256bit输出holding；STORE/DRAIN_READ均请求共同psum类单口，每拍至多一笔。未结合D1旁路。|
|Q2局部块|每context8rank×8lane×16bit=128B；合计256B，填充通过唯一共享权重口。|
|producer算术|**唯一8个signed19×13乘法器、唯一8条32bit加法链**在top；ZADD在bit13断carry形成两P，BASE_MAC保持全carry。两个context经显式operand/result总线使用，需grant方才提交。|
|仲裁|source/W/z/psum/ALU五类请求；资源交集空可同时grant；冲突轮转RR；外部allow=false先不eligible。ZADD/BASE_MAC同时请求z+ALU原子grant，无只拿ALU或偷读z。共同逻辑包括5bit请求×2、grant、RR、owner选择器。|
|每context其余状态|k_live864bit、v_live96bit、position_live40×8、rank/block/pending、q_hold24bit、z_hold208bit、origin/循环FSM以及两个完成旗。所有调度相同，完整声明见SV。|
|公共I24后端|**另有**一套8×signed32×32乘法和8×signed64加法链、8lane FP32→Q20转换/RNE/sat；这些没有并入上述producer八ALU。系数768B、单256bit行口，224B向量暂存，和旧冻结消费者完全相同。|
|外部接口|静态参数256bit、source10bit、identity256bit、result256bit各自ready/valid；source为RTL绝对C/Y/X地址，identity与输出有真实tile/row身份。|

两个source载入后才发首start，每pair最多两tile。seq控制也允许tile0的整个Q2计算完成后开始tile1，同时tile0向共同消费者drain；没有人为等待consumer结束。阶段错位候选在tile0 Q1完成后开始tile1。普通双ready强控制同时发两个start，由相同RR处理所有实际冲突。消费者严格tile0后tile1，奇数末tile只启动context0；最后480beat被接收且core/consumer都done方退休。没有跨batch免费source/identity。

冷静态装一次1848beat（Q1 864+Q2 96+k_live864+consumer24），广播控制metadata不复制权重数组。每tile付1536源写（padding也付）+1origin、480identity/480I24beat。warm命令只省同参数静态配置，源/origin/identity重新请求；测试stall波形按command-relative n产生。接口为可背压即时响应模型，未模拟DDR缓存/宏SRAM延迟。

总计：`total_cycles = window_cycles + launch_cycles + static_words + parameter_stalls + source_load_words + origin_words + source_load_stalls + 1`。window包含RUN及NEXT_CONSUMER，launch每batch一拍；末FINISH一拍。`core_cycles`是两个context活动时间之和，`consumer_cycles`也与两者重叠，三者不相加作性能。shared_ALU_grants=Q1issues+Q2MAC；每次竞争只有一个失败者，所以conflict_cycles=core_arbitration_stalls。core source/W/psum实际词数与对应grant闭合。

这是一套有界多context共享执行A。两个tile私有bank和选择网络增加实际资源；不能与单tileD1/D2或旧packed核横称同面积。未测EDA/Fmax/PPA/能耗；不能把周期缩短直接称时间或能源收益。本批没有把D1/D2/D3或root相位借用拼成一个已验证组合。

最终实际状态还满足 `window_cycles=consumer_cycles+tiles+floor(tiles/2)`，因此总式可化成共同单tile wrapper的`consumer_cycles+载入/外部stall+2*tiles+1`。这里consumer join等待已反映真实双context调度；该化简不把producer时间额外相加，也不代表两个core免费。
