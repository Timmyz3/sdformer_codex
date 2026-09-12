# 完整K864源地址接口：固定实测结果

这一轮闭合了已有掩码的物理地址跳过接口；普通读合并、静态掩码和地址迭代均归公共底座，没有建立独占创新X。采用原始固定参数和真实已发出的sn2门图，连续raw I24与PED义务全部保留。

范围是四个固定窗口（两学生×corner/interior），每窗16个锚点，完整K864→U16→F/BN2+raw I24→projection gate及PED U32/V96；另保留窗口内非锚点raw I24门消费者。不是全帧、RTL、PPA或新AEE。

|学生/窗|掩码|原扫描|共同读合并|普通静态predicate|地址iterator|iterator比predicate少|
|---|---|---:|---:|---:|---:|---:|
|ordinary/corner|global_group2|636,224|635,016|628,325|628,013|0.050%|
|ordinary/corner|phase_joint|636,220|635,012|631,977|631,209|0.122%|
|ordinary/corner|row_phase_joint_pair|636,142|634,934|631,987|631,219|0.122%|
|ordinary/interior|global_group2|677,294|676,158|668,719|668,887|-0.025%|
|ordinary/interior|phase_joint|677,356|676,220|672,429|672,141|0.043%|
|ordinary/interior|row_phase_joint_pair|677,048|675,912|672,217|671,929|0.043%|
|lifting_raw/corner|global_group2|626,390|625,182|618,491|618,179|0.050%|
|lifting_raw/corner|phase_joint|627,138|625,930|622,927|622,159|0.123%|
|lifting_raw/corner|row_phase_joint_pair|627,172|625,964|623,017|622,249|0.123%|
|lifting_raw/interior|global_group2|669,310|668,174|660,735|660,903|-0.025%|
|lifting_raw/interior|phase_joint|668,110|666,974|663,215|662,927|0.043%|
|lifting_raw/interior|row_phase_joint_pair|668,012|666,876|663,181|662,893|0.043%|

地址/掩码费用：四phase相同mask自动得到uniform常量编译：每H8仅选一次实际metadata bit，零组直接跳过；这条相同规则适用于任意候选。普通predicate不支付无用位图构建。静态metadata为48bit，冷填32B并逐目录经过SR64读取、拆为四个12bit相位行；源phase取全局sy/sx。每H4以9个现有RF向量收集P2×9的18个SR64字，144B逻辑门载荷占432B现有RF物理位。几何、边界、掩码查表、H8 offset位图构建、每次选择/地址控制、解包写回与NRV写出均收费。H8跨两个H4；不新增读端口。门图外部DMA没有减少。

强对照权限：三种mask共同得到相同coalescer、metadata格式及resident MAC；同mask比较的NRV顺序/字节/live完全相同，后继Acc48/RNE、更新I24、门和真实PED逐值符合独立整数gold。iterator比predicate少的访问须再扣它独有的位图构建；corner还含普通图像边界空项，summary已分开计数。global/interior中iterator反而多168槽。这不是增加一个新稀疏数学原语。

验证总数：12个固定负载×4臂，384次完整K目录；检查7,418,880个后继输出值，差异0。不同掩码改变网络输出的误差仍单独保留，不能将同mask实现的0差读成剪枝无损。

当前结论：给uniform-global完整静态权限后，phase/水平P2在本次完整局部消费者上均慢于最强global约0.3–0.7%。iterator相对同mask predicate只再减少约0.04–0.12%（global interior反而略慢）。本控制布局保留为可用公共编译/供数底座，不以它恢复phase-H8标题。相对同iterator的phase与global差异见summary，不以原始重复读取版夸大新意。当前mask只是先前10帧评价的同一参数；本轮没有新增训练、量化或valid825。没有改生产或旧执行程序。

文件：execute.py为隔离wrapper，PLAN.md为执行前B/A/X与强对照，results.json含全部收费分项与目录轨迹，summary.json含相同负载与不同mask两个分母。

固定压力复核：ordinary/interior/phase_joint四臂在同一period32读写背压下完成，所有NRV与后继输出检查同ready一致；服务值见summary.fixed_stress。
