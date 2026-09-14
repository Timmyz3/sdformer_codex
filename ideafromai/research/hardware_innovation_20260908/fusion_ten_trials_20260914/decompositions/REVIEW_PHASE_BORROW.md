# phase_borrow独立审阅

2026-09-14，审阅者为decompositions实施代理，未参与phase_borrow实现；只读SV/TB/收据并执行独立有限数学检查，没有重跑RTL。**未发现推翻功能或同口周期收益的阻断问题。** 范围为[phase_core](../phase_borrow/phase_core.sv)、[shared ALU](../phase_borrow/wide_phase_alu.sv)、[consumer](../phase_borrow/i24_consumer.sv)、[wrapper](../phase_borrow/consumer_stream.sv)、两TB和128小run/8个64tile命令/2个full命令。

1. 宽链确实替换了消费者原`wide_hold+add_rhs`：与旧consumer_packed逐行差分，只增加请求/授权/等待并把该表达式换成外部结果。wrapper只实例化一次8lane×64bit主加法链，消费者bias/identity两阶段都用它，producer原8条32bit链仍用于Q2。既有RNE增量器、FP32转换及控制算术仍存在，不能把“8条共享主加法链”写成整模块只有这些加法逻辑。
2. 两臂z均8bank×10row×52bit=520B，读写416bit。mode2用32bit链更新其中两个P并保留另两个；mode3用64bit链在13/26/39位断carry，四个P字段独立。各13bit字段界±2592，不会溢出；bit52以上不影响返回的52bit。读出的单rank标量按`fp%10`和`13*(fp/10)`解码，最终40位置scan/Q2缓存/完整psum义务相同。不能与旧208bit口作同面积比较。
3. `cons_add_req`优先，borrow grant条件为请求、无consumer请求且wide_allow。拒绝时producer保持ZADD和pending/active/z_hold，consumer保持ADD_BIAS/ADD_IDENTITY和wide_hold；成功才写回/前进。外部暂停在小run实际产生4533个producer借用等待和4557个consumer宽链等待。自然串行相位中producer Q1与consumer加法没有同时请求，因此双请求仲裁仅静态检查，不冒充已动态覆盖争抢；未来重叠调度须补该覆盖。
4. FP32→J20与rne26两函数和旧版逐字一致。独立用FP64表示精确FP32值计算round-to-even/clamp J，再用整数64式`p*a+((b+J)<<20)`和floor/remainder RNE26，16fixture共61440个J及61440个I24全部与gold一致。两个分阶段64加法不跨舍入边界；32×32积及两个32bit<<20项之和仍在signed64内。NaN/Inf继续error，有限极值/负tie语义不变。
5. 强控制没有偷换回scalar：无背压mode2与旧consumer_packed15比较，核心/consumer/总周期及z写仅每tile少10次clear；独立140项对比通过。新416bit布局两臂都只清10行。full mode2=324025349，mode3=310814282，减少4.07717%；差13211067=3×(19773592−15369903)。每臂raw/J/I24各73728000值，配置1848拍共同。128小run和8个连续64tile命令带双背压/无reset重启，full两臂是无背压单go19200tile，不能写成full背压覆盖。

[PHASE_AUDIT_CHECKS.json](PHASE_AUDIT_CHECKS.json)记录额外50000组四字段carry边界组合、函数对比和上面独立公式检查；它是数学/收据审计，不是新增RTL向量。[audit_phase.py](audit_phase.py)可复核。未跑PPA/Fmax，416bit银行、跨块宽链路由及borrow仲裁的物理代价未关闭。普通精度分段和资源共享属于强A，本次评定是完整真实边界下的功能/周期证据，未据此承诺论文新颖性或接收。

附：joint_selected组合静态核查。只读[build_joint.py](../joint_selected/build_joint.py)及生成的[core](../joint_selected/phase_core.sv)/[wrapper](../joint_selected/consumer_stream.sv)，未重跑RTL，未预认尚在运行的64/full结果。两臂2/3共同配置、读取都用XOR2的source_phase，reset及每次go（含参数驻留的warm命令）将phase/reuse清零并完整装首tile；合法横邻才翻phase并从逻辑列2开始，列3后步进3，159→160跨行恢复全装。相位仅在core/consumer都完成且480个输出已接收后改变。STORE对2/3只写共同result_data holding，DIRECT_SEND期间保持数据及og/fp地址，ready后才前进；这两模式无法进入p_mem写或DRAIN_READ，旧mode4物化分支与p_mem数组仍声明，不能据不访问声称综合移除了面积。消费者宽链和RNE未改，两臂仍同416bit口，仅dualP/quadP有差分。该组合将已审halo/forward/phase实际接起来，有工程筛选意义，不计第11种独立X，也不能用既有倍率相乘替代最终事务测量。
