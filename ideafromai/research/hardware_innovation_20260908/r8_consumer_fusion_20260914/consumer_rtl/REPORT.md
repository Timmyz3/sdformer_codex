# R8 到真实 r1 I24：整数仿射、FP32 identity 完成与连续 tile 服务

本目录实现了实际 `gate→完整r0 conv2→output_scale/固定BN2→连续identity→I24` 的有限端口 RTL。固定 Q1/Q2 新函数内，expanded21 mode6、R8 V驻留 mode7、最终支持位图 mode9、再加原生源窗口 mode11 都接同一个消费者。消费者直接收 IEEE FP32 identity 位模式，在 SV 中完成 J20 量化、宽乘加和 I24 RNE/sat；raw p 由同实例完整K864整数核产生，TB不供 latent 或 psum。

消费者闭合属于普通融合强 A。mode11 相对7含最终支持枚举和原生源复用两项；相对9才是原生窗口单项。所有模式共享新增状态/端口/后端，不从相对展开矩阵的旧大倍率继承创新主张。

## 真实图与冻结数值合同

已读原生 `MS_ResBlock.forward`：identity 是 r0 block 输入，conv2 后先 norm2，再 ADD identity，随后直接进入 r1.sn1。`ParentNetwork` 安装固定 running statistics；本次实际运行也确认 BN2 `training=false, track_running_stats=true, eps=1e-5`。当前 `LiteralForward.source_forward` 先 `I24=RNE/sat(x*2^14)`，As 和连续路径读这个完成值。不能将 As 代数穿过这一舍入点。

旧整数因子只完成 `z=Q1*g,p=Q2*z`，没有中间RNE。当前新部署合同固定为：

```
a[o] = RNE(output_scale[o] * BN_gain[o] * 2^40)    # signed32
b[o] = RNE(BN_offset[o] * 2^20)                    # signed32
J    = sat32(RNE(identity_FP32 * 2^20))             # RTL实际一拍完成
wide = p*a + (b+J)*2^20                            # signed64精确整数
I24  = sat24(RNE(wide / 2^26))                     # signed24/f14
```

MUL、加bias、加identity、最终ROUND各占一拍。a/b各读一拍后跨同N8组40个P/T成员驻留，24个常量向量读/tile。p/J分别来自真实握手，全部到齐才能执行；没有gold统计、mask或I24喂给机制。FP32转换利用符号、指数和24bit significand的有限移位及guard/remainder parity；±0、subnormal及全部有限浮点溢出都定义，溢出饱和，NaN/Inf置错误。负数最终RNE用floor商和非负余数处理，包含负tie。

实际a范围540863…2113405，b范围−1009544…338446；保守 `|p|≤679477248`。对本a/b和全部J32，wide绝对界3688869010604032，小于signed64界。该整数仿射/J编码是新部署函数，**不是原FP32 BN/add逐位等价**：真实首8块30720个I24中205个与旧浮点出口差1。独立新函数评价与出口检查见[data](../data/consumer_integer_definition.json)。

## 共同资源和真实完成义务

[资源合同](resource_contract.json)列出完整数组/接口。生产者保留旧所有系数/latent/psum预算、八条32bit数据加法链和八个16×13乘法器。另将root融合的320bit最终z支持、160bit原生C1窗口、四个10bit局部寄存器读mux、差分/前缀保留状态一并预算给所有模式；原生source SRAM仍只有一个10bit词读口。

消费者共同新增八个32×32乘法器、八条64bit加法链、八个FP32→J20转换器及八个I24 RNE/sat单元。没有把原16×13乘口称成免费32×32。静态a/b存储768B，单256bit共同行读口；p/J/a/b各一向量、wide与输出共224B持久寄存器。无整块identity或消费者输出缓存，J覆盖原位的FP32输入寄存器。

SV连续wrapper一次go处理完整tile序列。静态expanded/Q1/Q2/k_live/mask/a/b共12504个256bit配置拍，只在首次装入；每新tile实际1536个源词写入+1个原点拍，图外padding由SV判定并写零。外部identity每tile480×256bit，输出480×256bit（每lane signed24符号扩展）。未做源装入/计算双缓冲；重叠halo重新读并付费。raw p和identity双路各一项缓冲，完成消费者有背压。必须收到producer done及最后一个I24接受后的consumer done，才退休并开始下一tile。

生产者与消费者cycles重叠，**不能相加**。实际总账：

`job_total = Σconsumer_cycles + static_words + parameter_stalls + source_load_words + origin_words + source_load_stalls + 2*tiles + 1`。

每tile消费者状态恒等式为 `3385 + join_wait_cycles + output_stalls`，其中包含480拍FP32→J20转换。join等待可同时等待raw和identity，两个独立等待计数不应再次相加。

## 真实八块与有界功能验证

最终FP32入口16fixtures×4模式×2背压×2同实例命令=256runs；raw p、J20、I24三个检查点各983040个值全相等。真实八块来自本轮A800同一捕获；没有拿旧3090的19bit源差异收据顶替。独立data的p/J/I24共92160值与fixture逐值相等。合成边界包含全零、全一、padding poison、全因子极值、正负乘积tie、identity tie和I24饱和、FP32±0/subnormal/最大有限值。TB检查参数、源、identity请求以及结果在背压时保持稳定，最后身份/行号/last标志、warm零静态装入均检查。

|真8块，无背压|6 expanded21|7 R8驻留|9 最终支持强A|11 原生窗口|
|---|---:|---:|---:|---:|
|消费者完成cycles|389412|193215|169263|161631|
|实测8个独立cold job总cycles|501764|305567|281615|273983|
|扣除重复静态配置后的tile服务|401732|205535|181583|173951|
|核心source10bit读|9600|23364|23364|9600|
|核心系数向量读|40908|2722|2710|2710|
|latent读|0|37129|24013|24013|
|psum读/写，各|70680|21444|21444|21444|

八独立cold case实际装了8次12504拍；第三行只是同一测量扣除静态的服务范围，**不是八个连续tile的实跑总周期**。模式11对9在此范围减少4.20%；对7减少15.37%。完整连续结果是下节的实跑口径。

同Q20旧入口的10800项稳定事务字段在新增FP转换后保持相同；完整状态账5376项、Python大整数scalar oracle61440值通过，见[checks](checks.json)。这些核验补充实际Verilator输出比对，没有以公式替代RTL仿真。

## 连续与完整帧结果

最终FP32入口已实跑完整帧。两个模式分别只发一次go，各连续处理19200tile，未分行独立实例或重装静态权重。每臂raw p、转换后J20、I24各73728000值全通过，两个模式合计442368000检查点。两job实际并行wall174.8秒，非外推帧数；其中mode9/11各174.8/167.6秒，这只是本机仿真耗时，不是硬件FPS。

|完整帧，最终FP32入口|9 最终支持强A|11 原生窗口|
|---|---:|---:|
|**实际total cycles**|**511638152**|**493321352**|
|consumer完成cycles（与core重叠）|482076847|463760047|
|静态配置256bit词|12504|12504|
|新源写/图外padding|29491200 / 214656|29491200 / 214656|
|外部源10bit读 / origin拍|29276544 / 19200|29276544 / 19200|
|核心source10bit读|65879744|29276544|
|四路局部窗口读事件|0|16550400|
|核心系数向量读|8905268|8905268|
|latent读 / 写|77488449 / 24535485|77488449 / 24535485|
|psum读 / 写|62936964 / 62936964|62936964 / 62936964|
|FP32 identity向量 / I24输出向量|9216000 / 9216000|9216000 / 9216000|
|FP32→J20转换 / 宽乘 / RNE，各|9216000|9216000|
|宽加 / 消费者系数读|18432000 / 460800|18432000 / 460800|
|core输出等待消费者拍|46444800|46444800|
|外部source/parameter/identity/result阻塞|0|0|

模式11净省18316800拍，即**3.5800%**。这一个差分来自同资源预算下原生C1窗口的普通数据复用；W、psum、latent和乘加事务没有减少。内部source读取由65879744降至29276544，付16550400次四路局部读事件。各系数向量宽度不同，不能把向量数当相等bit流量或能耗。新增消费者转化每帧9216000拍，实测总账相对旧J入口恰多这个数；绝不把它当零成本完成。

剩余主要费用仍在latent/psum服务与53720964次第二阶段八lane MAC，消费者仍需要73728000个连续identity和I24元素。当前没有证明减少这些计算/存储的独立X，也没有缩减外部源halo装入；只有实际连续服务的普通强A提升。

完整帧无背压；固定64tile ids128…191跨行已另跑两模式×背压×cold/warm共8jobs，raw/J/I24各1966080值全通过。无背压首job9/11为1853549/1792493拍，warm为1841045/1779989拍；warm只免一次12504静态配置，仍实际重读全部新源、identity并重新转换。背压测试覆盖参数、外部源、内部source/weight许可、identity、结果；所有request/result地址及数据保持检查通过。

[逐job结果](results_full.json)、[连续64结果](results_64.json)、[CSV](stream_full_costs.csv)和[SUMMARY](SUMMARY.json)均标明`identity_input=IEEE_binary32`。[stream_checks](stream_checks.json)的220项账/几何检查与72项旧入口差分检查通过。全部最终FP32实验共266jobs，raw/J/I24各150405120值；不将旧入口收据重复计入。

旧Q20输入的四臂完整帧已独立跑完，每臂一次go19200tile，共294912000 raw p及同量I24全通过；代码/日志冻结在[q20_input_snapshot](q20_input_snapshot/README.md)。该收据不含FP32→J20转换，因此不用于最终消费者性能。不会把两种入口的周期或检查数合并成同一实验。

## 质量和边界

新固定整数消费者已重新评价diverse10，AEE 1.3903315401317662，低于历史NB0 1.45460286107；它读取本函数实际I24并逐帧核对当前LiteralForward接收到完全相同的I24。与原浮点出口首帧468058/73728000个I24差1。质量来源为[data部署记录](../data/deployed_diverse.json)。同一新函数的[完整825帧](../data/deployed_valid.json)亦已完成：48152523有效像素、frame-mean AEE **1.3276350226079938**，低于历史NB0 **1.44535253468097**。每帧当前I24读取器均核对实际新出口；这不是旧R8浮点后继AEE或其他剪枝臂825。

本实验没有推进r1 As/阈值/PED后继RTL，也没有外部DDR/缓存延迟、SRAM宏映射、Fmax、面积能量或全网FPS；完整帧指本边界全层，不是完整网络。最终I24到原下游的浮点精确重注入经过逐值检查，整网其他算子仍非全网定点RTL。完整帧无背压和warm重复，二者只在固定64tile及小块中实测。保留普通融合强A和可重复收益，不以本布局负点或倍率概括整个分解家族。

[Root独立代码审阅](../REVIEW_CONSUMER_ROOT.md)已核最终FP转换、双输入保持、共同端口及完整帧总账；这是团队内独立审阅，不是外部论文或工具签核。后续cachedOS接口的只读整理见[NEXT_CACHED_OS_INTERFACE](NEXT_CACHED_OS_INTERFACE.md)，本轮结果不因该后续接口而改写。

[data独立审阅](../data/REVIEW_CONSUMER.md)进一步核对5332项周期/工作量/地址守恒，全部一致，未重跑RTL。[同环境质量复核](../data/quality_validation.json)确认同825帧有效像素群，部署I24也优于同A800 NB0 1.447936665574317。对原R8的逐帧差并不为零（425帧变差、400帧改善），均值接近不能称任务无损。
