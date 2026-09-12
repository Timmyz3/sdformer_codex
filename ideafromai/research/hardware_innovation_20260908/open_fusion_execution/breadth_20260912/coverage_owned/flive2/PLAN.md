# 固定 F_live=2 接口

旧 `gustavsnn_resident_core.cpp` 的 F 是驻留W行数，逐行完成突触和PSN，F_live=1；Packet只有 selected/mask，不能直接改队列数就宣称新数值执行。旧 `gp_slice_pe.sv` 有真实双NR4、56字S15、实际CSD微码，但活跃S只有P4×R≤28。本次取它的私人压紧/包执行/消费语义，新增有界CPU数值排程。

**输入和范围：** 不把事务公式冒充全层。使用已有GPS1真实完整C384、16位置、两W行、全T10 fixture；两W行在本试验映射为一个tile×4PE上的两个f任务。旧fixture在2tile运行过，本次是新的一个tile共同资源点，不沿用旧RTL周期。全部44表示用例、F1/F2×private/merge×scalar/member，共352次。正常ready主环境，不扫F4或新增GPU。

**B与强普通控制：** F_live=1仅一组S存活但仍拥有完整56字容量、两NR4包、同两条驻W行、源bank、程序和tau容量。允许F0最后W返回后先扫描/压紧下一行，把最多两NR4包预取到既有缓冲，并与F0 PSN消费重叠；不得强行把整个后续前端也串行。F1与F2均有dense同址请求合并/旁路，以及标量/成员归约权限，主分母取最强F1，不把较差private或scalar单独当唯一分母。

**新增接口：** F2在同56字S内保留2×P4×R7，NR4总包数仍2且每包只属于一个f。一次源码扫描产生当前NRV行，先后服务两个f权重；每ID最多一条W请求在途，四ID争用每tile两个W8口。源bridge等待两个f处理后才前进。f/context标签和有限仲裁控制两臂均预留。只此固定row-broadcast布局，失败不关掉所有F_live>1。

**共同资源：** 4×1KiB源bank，1R64，一拍响应，128bit滑窗；每bank576B实际源码。每tile2KiB驻W/tau/控制池：两条dense行768B、两份tau60B、控制预留128B。4份256×16程序ROM，一读/PE/拍，均驻留，配置在端点外。每PE 56×S15+valid、两包4×(code12,W8)、单S读/写口、W10局部加法、共用Acc24提交/PSN、7×S15消费缓存与一个10bit输出保持。成员加法器在两臂同样可用，未声称它与仅标量等面积。只允许一个syn packet执行或一个consumer程序；不免费并发使用同Acc24。

**收费端点：** 驻源码/W/程序/tau start到最后输出门握手。源码真实位打包/请求/响应/扫描、W响应驱动压紧、包启动/逐成员或成员合并、S提交、S加载、逐条CSD/tau、单10bit输出口全付服务。源生产、外部冷DMA、FC2/BN2和物理宏不在本边界。检查每个S和门及包/事件/写入守恒。程序使用fixture原微码，未另造算术降成本。

强对照补齐：已读原 `class_reconstruct_control.json`，b0/b4 的普通类和→原位恢复→A_eff 尾比直接B更便宜。最终两F臂都采用该静态已有选择，先在全部8个码上验证函数完全相同，再实际执行LOAD_U/add_store及重映射CSD，不把恢复当免费。7字缓存、一个U24及U→cache写回mux两臂同权；每条恢复操作占一个程序/issue槽，最终程序仍在原256×16bit容量内。首轮仅direct-B的结果不作最终主表。
