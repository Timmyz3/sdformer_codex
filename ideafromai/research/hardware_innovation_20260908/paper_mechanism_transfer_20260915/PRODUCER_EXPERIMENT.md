# Gustav真实源码字生产接口

问题B：现有eager/lazy GP→PSN已经处理真实NRV∩W，但源bank由TB预填；首次非空NRV才请求W支持，稀疏W须补扫此前列的压缩rank。没有从实际源写入到末T10门的比较。现有原始源是每bank72个64bit packed-code3字，不得免费改成按12bit行输入。

A：复用2tile×4ID、P4、C384、两W8口/tile、双NR4和完整CSD/Acc24 PSN；新增真实4bank×72×64bit RAM，每bank单读写口、读响应保持，写入优先。只读已提交字。端点计首个accepted源写入→末门；程序/τ配置另列。它仍是packed-code生产者接口，不是前级量化器RTL，也不是官方完整Gustav芯片。

候选X：每个accepted64bit字在RTL处理跨字3bit码，累计8bit seen-code；当前bank完成后与decode非空表相交，提前准入W metadata。普通强控制：相同码提取直接累计1bit nonempty，不保存通用seen-code。当前每代decode不变，二者理论上应同效果；实验应明确检验这个等价性，不能给8bit表示自封新颖性。所有模式共用同声明状态和端口；不是等面积映射证据。

四臂：3=eager；4=首次NRV lazy；5=完成bank的seen-code准入；6=完成bank的1bit nonempty准入。四臂均读取相同288个源字，无整bank跳读特权。已有每PE rank游标复用，5/6与4同样付补扫；无免费前缀表。

工作负载：保留全部48组现有真实/定向case，class/time、scalar/member、两种固定producer日历及ready/BP。补码重标记使code0为活、非零code为空，另补最后一行才非空的边界；不复位换源/W/program以验证本代状态。核对S、valid、完整T10、源/W逐地址、字节、未完成响应、输出背压以及seen-code与bank内容的一致性。

判断：性能只在该生产接口成立；5若等于6，归入普通producer-aware预取底座A，停止将seen-code当独立X。原协议A有实测落后时仍保留原因，而非否定Gustav家族。不会用本组小切片结果冒称整层、整网或ASIC PPA。
