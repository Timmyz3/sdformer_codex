# 独立审阅：同表H8融合与跨SRAM z表示

固定范围是 ordinary/corner/ready 两臂，复用真实完整局部生产者及原U/V和外送。已静态读 `phase_execution.py` 与 `PHASE_EXECUTION.md`，并核对最终 `phase_execution.json`；未发现未解决的数值或共享资源问题。审阅只读实现与已有JSON，不重新运行实验，只修改本文件。结论限于这两条CPU有限资源模型实际执行。

## 数值与资源合同

同函数为 `b=Dg+c; q=clip8(RNE((x-b)/2048)); xhat=sat24(b+2048*q)`。定义 `a=floor(b/2048)`、`phi=b mod2048`、`z=q+a`，则 `2048*z+phi=2048*q+b`。phase臂必须先保留原q的RNE/clip8，再求z；不能直接对z做通常的偶数ties舍入，也不能用原IRNE冒充负数floor。a/phi须从实际RF里的b经有费运算产生。

已只读核对既有全门码普查记录：ordinary在全部1024门字中b范围[-247277,260141]、a范围[-121,127]，与全部signed8 q组合后z为[-249,254]，故通用signed9必要。phi为unsigned11 [0,2047]。同函数重建范围[-509421,520237]不触signed24饱和；这不允许删除原q8 clipping、U/V原RNE/sat或bias边界。普查并非本轮真实执行结果。

完整LUT放在系数池[98304,131072)，1024×32B，每行十个signed20 b字段共200bit；固定全表不能被当前halo少数实见门类代替。cold fill与每次真实CR256、门SR、选择、符号扩展和RF写回全部计费。表不放入既有source ROM，也不免费提供十个预测向量。

布局合同为Uacc RF0..29、b RF40..49、gate RF50、临时RF51、当前H8 x/q/z/重建 RF60..69，共52活向量。b仅保留至当前H8 U消费结束，下一H8覆盖；进入共同H48 V前b/source已死，V可使用RF30..89。每条计算保持acc+source最多两个RF读，系数来自实际CR响应。

phase码按signed16放入状态池[114688,116608)，逐H8/T两次SW64、两次SR64后重建。不能因为当前halo实际z恰落signed8便改用8bit。source收集[0:24]与z收集[24:40]同属既有64B staging；gate collector[0:16]在当前H8输入读取前完成，未覆盖仍活的源收集值。LUT与状态码分属原独立系数池/状态池，均不增加容量。

## 实码核查

`table_image` 对全部1024行、十个signed20字段做静态打包/解包一致性验证，随后 `load_table` 经实际DMA将全部32768B填入原系数池。`table_predict_h8` 每H8先实际读row0，付费广播c到十个b RF；门经两个实际SR64、collector及ILOAD写RF50，再逐lane付费选择和零判定。非零门字选择实际CR256响应，逐t用 `ILUT20_LANE` 更新一个b RF lane并保留其他lane。没有以host预测数组替代实际表响应，也没有给一次CR响应十个免费RF写。

`encode_decode_h8` 从真实updated逐t读取I24，执行原宽减法、IRNE(11)、clip8；控制臂随后IRNE(-11)、加b和独立ISAT。phase臂 `IFLOOR11` 使用对负数向下的 `floor_divide`，z=q+a后真实写入signed16，再从SRAM读回；`IPHASE11` 取b的低11位得到非负phi。IRNE(-11)、加phi和独立ISAT完成重建，没有改变原q的tie parity或clip8边界。

`store_z16` 实际RF读到原staging并发两次SW64；`load_z16` 取两次实际SR64响应、付collector和ILOAD/等待。两函数断言[24:40]以外staging保持。U逐h=0,8,…与块内k按原顺序消费实际RF重建，保留原零旁路、权重响应与acc+source两读口；原U complete后直接调用共同 `kernel.v_stage`，RNE/sat/bias顺序不变。

q、重建与物理CODE的观察数组仅用于最终assert，没有再供U/V算术；独立oracle只用于binding最终检查。四个新增操作逐issue断言沿原单issue/两槽RF写回，实际共享仲裁与等待继续执行。signed20位选择、broadcast/lane更新、floor与低位提取的两槽延迟仍是模型假设，尚无RTL/STA/PPA证据。

## 强控制与归因

控制臂使用相同全表、实际gate选择、H8融合与预测RF保留，在原q RNE/clip8后直接于RF重建 `2048*q+b`，随后消费U；没有RECON或Q8中间spill。phase臂有额外z的signed16 SRAM写读，并在RF重建后消费同一U。两臂都没有执行显式Ug或D(Ug)图，这项省费是共同LUT/H8融合所得。

phase专属增量只能相对上述同表RF重建控制计算。与原Dg/带spill CSE路径的总差包含查表、融合、移除旧中间存取和消费方式变化，不能整体称为phase收益。若phase额外跨SRAM表示落后，只停止当前表示放置，不否定精确商余恒等式、查表融合、门预测或低位家族。

## 最终两条数据核读

JSON两臂均完成，实际前缀的服务/count/check逐项复现原 `ordinary_corner_final.json`。每臂192个H8事件、1536次实际gate选择、1275次零门旁路；192次row0广播加261次非零lane查表，共453个表事件，实际417次CR请求。两臂LUT及相关CR/CW完全相同，cold fill均6144槽，LUT预测阶段均8525槽。

每臂15360个实际RF q、15360个RF重建、3840个U与15360个PED均0diff，updated/projection gate与实际PED SRAM外送字节也精确。phase额外比较15360个物理CODE16值，1920个向量写与1920个向量读均实际执行；新增操作逐条两槽检查，控制臂4530次、phase8370次。阶段和等于总时钟，前缀加消费者等于总服务，端口字节逐项等于实际计数乘字宽。

| 同函数ordinary/corner/ready | 全共同前缀总槽 | 消费者槽 | SR64字节 | SW64字节 | CR256字节 | CW256字节 |
|---|---:|---:|---:|---:|---:|---:|
| table_RF_reconstruct | 2057099 | 584909 | 2727360 | 1043360 | 3018048 | 198336 |
| phase_z16_SRAM | 2095499 | 623309 | 2758080 | 1074080 | 3018048 | 198336 |

phase相对同table控制多38400槽，即总服务+1.8667%、消费者+6.5651%；SR/SW各多30720B，CR/CW增量均为0。38400槽为额外z的floor/add、signed16物化/重读与phi解码的合计，不应全部归为总线spill。原RF重建阶段17280槽，被phase的z编码/写出17280槽、实际重读15360槽、重建23040槽替代，净差正好38400槽；共同U阶段均50720槽。

该结果只支持停止当前signed16 z跨SRAM、再重建接U的放置。两臂均先在RF重建完整I24再乘U，本轮没有执行普查中另提的 `U*z + U*delta_phi` 因子化消费者；不能据此评价后者。LUT/H8融合相对旧Dg/CSE的差异也不能算成phase专属收益。

本halo未触发q8 clipping，实际z恰在signed8范围；signed9义务来自全域整数证明，不是本次运行遍历了所有边界。完整表的静态打包验证不等于实际CR路径遍历全部1024门码。仍未覆盖其他父、interior、压力、训练、新AEE、native/globalBN、整层/整网或RTL/PPA；不据此否定条件表示或低位家族。
