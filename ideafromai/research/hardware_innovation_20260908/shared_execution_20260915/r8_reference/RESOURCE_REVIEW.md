# 共同执行接口的独立审阅

审阅对象为本目录 R8 参考与 `../spatial_rr/{PLAN.md,interleave_stream.sv,rr_context.sv,wide_phase_alu.sv,stream_tb.cpp,verify.py}`。结论：允许比较同输入、同主执行资源许可下的服务周期；不能据此宣称两个裁剪网表等面积或同频率。算法函数不同，必须同时列质量。小集分别12/15个fixture，不比较其总周期。

共同约束已核：两个context各source1536×10bit=1920B、psum480×8×32bit=15360B、Z40×8×52bit=2080B；总Z4160B。每拍各source/W/Z/psum只有一个授权owner，同种资源请求不可同时接受；相互独立资源可由不同context占用。W服务256bit，Z服务416bit，psum服务256bit，psum每bank允许自己的地址。candidate的Z有效数据仅低32bit/银行，读表达式取低32bit，写高20bit为零；它仍独占整个共同Z授权拍，不能把256bit有效payload说成416bit有效数据或面积节省。R8真实声明40行，现算法只使用前10行。

两端都在top仅实例化8条32bit数据carry chain与8个signed19×13乘法。candidate的源变换、3M前缀、恢复、stripe合并全部通过同一ALU授权；leaf只导出操作数、接收共享结果，没有复制数据乘法器。R8 pop16×8是额外组合归约树，必须单列，不能因计数更新使用共同ALU便隐去pop。R8 modular修正和count更新也使用同一32bit授权。

消费者各为唯一8×32×32乘法器与8×64宽链，FP32→J20→wide→I24、最终RNE26语义保持。R8 mode4与candidate borrow设置均可申请该宽链，consumer和producer互斥真实争用，未见第二条宽链。R8 carry cut13，candidate cut15，各核只实例化自己用的切点，联合可配置切点mux的时序/面积未实现和测量。共同许可涵盖两种切点功能，不能冒称已合并成可切换两函数的单一网表。

状态与静态数据的共同许可应取以下并集，而非把仅Z扩容当成全资源等价。R8 cache已实际扩为24×8×16=384B，使用原8行；candidate cache声明24×8×13=312B，在同384B上限内。剩余寄存器预算是允许使用并集，不通过添加无效dummy寄存器制造面积相等。candidate与R8均没有利用所有剩余预算的最优性主张。

| 条目 | R8实际 | 空间candidate实际 | 共同许可/说明 |
|---|---:|---:|---|
| 共享Q1数据 | 2592B | 4608B | 4608B |
| 共享Q2/physical3数据 | 1536B | 7488B | 7488B |
| class / representative | 2592B / 96B | 0 | 各一份允许，全经同W服务 |
| bitmap plane / plane live | 2592B / 162bit | 0 | 各一份允许，非额外W口 |
| 固定time排列 / groups | 40bit / 6bit | 0 | 允许，首次加载付费 |
| 共享consumer a/b | 768B | 768B | 同容量、24配置拍 |
| 每ctx cache | 384B声明，128B用于Q2 | 312B | 384B许可 |
| 每ctx source window / masks | 20B / 5B | 20B / 10B | 20B / 10B许可 |
| 每ctx Z hold | 52B | 32B | 52B许可，真实写入和占用明确 |
| 每ctx accumulators | acc32B＋count_hold32B＋bitmap_acc32B | 3M共96B | 全部组允许；R8组用途与candidate不同 |
| 每ctx transform tail | 0 | 32B | 32B许可 |
| 每ctx Q1 hold | 3B | 8B | 8B许可 |
| 每ctx position support | 40B | 80B | 80B许可 |
| 每ctx bitmap / live / pending / hold | 80B / 5B / 5B / 2B | 0 | 原状态保留，无免费旁路 |
| 每ctx count_live | 80B | 0 | 允许；count数据使用原psum并在输出前完成退休 |
| 每ctx R8 k/v live | 120B | 0 | 原每context副本均计；candidate共享Q1/Q2 live共144B另计 |

双方其余rank/block/remaining、输出hold、source pending、indices、RR状态、proof bounds和统计寄存器仍按源码存在，未宣称上表等于完整综合面积。静态Q1/Q2许可合计12096B；额外R8 class/rep/plane/permutation/group合计5306B，另有shared live和consumer参数。仅驻留同模型跨mode/换源已测，没有跨模型失效接口；换模型需reset重新配置。

加载协议已核为每tile全部1536个source字、随后独立32bit物理origin握手；没有在TB直接喂latent/raw。所有三主集合逐次序比较source、origin和FP32 identity位：251904 source字、629760 identity字、328 origin标量全等，见 `cross_inputs.json`。candidate和R8都同一r0.conv2挂点、同load_parent前缀，输出gold各自重算。输入框架每帧reset_net，未凭一个前序替换后的状态跨帧继承前缀。

生命周期已核：先加载整batch最多两个源与origin再同时launch；只有完整I24最后行被接受且consumer_done后清除owned；下一batch加载断言owned为零。candidate的stripe1只读stripe0已写psum，两个stripe完成才能drain。R8 count阶段使用psum，但未把消费者仍持有的输出覆盖；原inverse时间读地址仅在完整p已形成之后生效。请求未grant时candidate的Z/p读写使能为零，数据状态停顿；top有逐资源重复owner、宽链重复owner和提前复用断言。现未发现明确算术或资源复制bug。

candidate TB有逐值raw/Z/D/J/wide/I24、accepted请求顺序与输出保持、独立逐state工作量。独立审阅建议后，DA已补parameter/source/origin/identity被拒期间的请求保持断言，并以单独连续换源回放核验，不改变主RTL/周期。本目录TB已经逐拍检查这类保持，全部测试通过。candidate完整最终覆盖以其SUMMARY/verification_swap为准，本文不把其未完成结果算作通过。
