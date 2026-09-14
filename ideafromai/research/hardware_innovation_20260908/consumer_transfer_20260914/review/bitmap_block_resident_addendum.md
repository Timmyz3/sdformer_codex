# K16条带驻留增量静态复审

2026-09-14。只读 `bitmap_block_resident/PLAN.md` 及 decomp_core.sv 相对已核 bitmap_pipeline 的差异；未运行实验、EDA或改实现。性能仍待负责人收口。

**未发现半字保存、完整字提交或条带切换的功能错误。**

- BM_POS 读取当前208bit z，完整存入既有z_hold，目标signed13半字符号扩展到bm_acc；随后pop/native累计从此前条带的和继续，不是重新从零计算。
- BM_STORE把bm_acc[12:0]与z_hold的对侧13bit拼成每lane完整26bit字写回；期间没有其它路径修改z_hold。
- 同一z行的另一个P在后续BM_POS重新读该行，因此可以保存此前P已经提交的新值。每个P/T条带只拥有当前一次读改写，没有跨P复用陈旧z_hold。
- 最后pop或BM_NATIVE_ADD在转入BM_STORE前一拍提交bm_acc；没有漏掉最后plane或native项。若live与bitmap不一致而扫描到零，写回的是原读值，仍保持功能。
- 40×16bit条带的每个bit由全部16K逐一覆盖；k%16==0直接重置40bit live，其后OR。全空条带仅经过BM_GROUP并前进，不读改写z。
- BM_GROUP在第16个k写完后才锁存live/pending及bm_block=k/16；bitmap消费期间不再构造。BM_STORE清当前最低置位，完成全部P/T后才覆盖下一条带。
- K16边界通常截在9tap内部：local_source在消费期间保持，advance只在k%9==8时转L_START，其他情况继续旧窗口的下一k，映射正确。
- 最后k=863必为完整条带；空条带或最后P/T提交后advance才进入ZSCAN。无漏尾、额外k或越界K16地址。
- bitmap声明容量4320→80B，live270→5B，pending54→40bit；bm_z_word是组合总线，不是新增26B持有寄存器，保留完整字使用原z_hold。
- 每个非空P/T条带真实多一次208bit读与一次完整208bit写，计入z_vector_reads/z_writes；不能拿声明字节减少直接宣称面积、能耗或性能改善。
- z口证据边界：新BM_POS表达式和旧ZSCAN/scalar/ZREAD使用在功能状态上互斥，可通过统一地址/enable实现原服务；当前源码未显式统一这些z读路径，本审阅不能保证综合映射不复制读口。该限制已发给父任务。
- 新模块对所有mode!=14均走条带路径，故旧mode10必须引用旧bitmap_pipeline模块作为对照，不可把新模块的mode10也当旧完整bitmap实现。PLAN的14/10/9比较需保留模块来源及不同声明资源量。

这是常规分块驻留与额外部分和访问之间的取舍，尚无独立新颖性增加；应作为已有位平面底座的资源适配消融保留。路径相对 `consumer_transfer_20260914/`。

## mode8：复用闲置Q2缓存的后续静态增量

父任务新增mode8后再读当前源码与PLAN；未运行实验。**未发现生命周期、半字提交、符号或BP控制错误。** 此版已用统一z_read_addr/enable/read_word覆盖ZREAD/ZSCAN/BASE_MAC/BM_POS，前述“未统一z读表达式”限制在代码结构上已经解除。

- 每非空K16条带在BM_GROUP确定bm_block并置bm_plane=0；BM_WFILL逐拍覆盖qblock[0..2]。活plane按weight_allow读取原bp_q，不活plane写零；三行均完成后才BM_POS/消费，没有引用前条带旧权重。
- 缓存仍是原8×8×16bit qblock=128B，只借前三行48B，不另增权重阵列。qblock_read_index统一选择bm_plane或selected_rank，唯一128bit表达式供pop或Q2系数；这两个消费者在状态上互斥。
- 每个多活动位P/T的BM_SCAN把plane归零；BM_CACHED_POP依次处理0/1/2，位型保持且plane2经共享ALU减4×pop。每个plane最多一次128bit缓存读取和一组pop，没有新增pop树/producer乘法器。
- 最后cached plane在进入BM_STORE之前提交bm_acc，原z_hold全字保存和全字写回规则仍成立。onehot继续走原Q1读取+共享加法，不因缓存预取换成免费的原生权重。
- Q1完成后才ZSCAN→VLOAD；VLOAD覆盖qblock全部8行（不活秩写零），到qfill==7后才进入POSLOAD/BASE_MAC。Q2不会把Q1 bitplane误当signed16系数；新命令也会按上述顺序重新建立各阶段数据。
- 缓存内部消费不应受weight_allow阻塞，当前确实只让真实prefill、onehot原生Q1和Q2 VLOAD承担该BP。BM_WFILL被拒时地址/plane/缓存保持，既不计cache_writes也不锁存读数据。
- 所有非空条带都预填三行，即使后续全是onehot也付费；cache_writes包括不活plane的置零。无BP相对mode9的理论core净省应为M−3S：M为多活动位P/T条带数，S为非空K16条带数；M次四槽变三槽，S次各付三拍预填，onehot两拍不变。此式是代码费用推导，非本审阅新增实测。
- 新资源主要是qblock读地址与pop数据选择、写地址/数据选择及FSM控制；数据阵列、qblock容量、ALU和pop树不增加。两处状态互斥的qblock写入可共原写服务，且没有要求与缓存消费同拍读写。

两处读使能问题已闭环：父任务修改后，本审阅再次静态确认qblock_read_enable仅在BASE_MAC或有效BM_CACHED_POP置位；z_read_word在BASE_MAC只使能selected_rank bank，向量状态才使能全部8bank。因此缓存有效读、z标量/向量读与当前协议计数一致，不再列为未改问题。父任务正在重建并重跑最终small/short→64；本审阅未提前声称该轮动态验证已经完成。物理时序/Fmax仍未测，不从这些协议计数推导能耗。

mode8是把条带内系数复用补回借入底座，并利用已证明阶段互斥的缓存容量。评价为**更完整的weight reuse适配，单独增量约2/10**；不因复用48B或通过raw验证而上升为新位串行理论/完整BISMO架构。
