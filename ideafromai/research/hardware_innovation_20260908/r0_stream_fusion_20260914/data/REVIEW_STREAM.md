# 连续整帧 wrapper 独立审阅

2026-09-14。只读 [stream_wrapper.sv](../stream_rtl/stream_wrapper.sv)、[tb.cpp](../stream_rtl/tb.cpp)、原leaf及最终结果。没有修改硬件作者文件或重跑RTL。

**约定的固定参数集、单tile缓冲、请求/有效响应接口内，未发现阻断功能或计费错误。** 原leaf与前轮已审pair_parent_merge.sv的文本diff为空。最终24个跨行64tile job加6个完整19200tile job，共448,266,240个整数输出已由作者回放核对；其中完整帧部分442,368,000个输出。独立源/几何公式对全部30条结果的540项计数全部相等，见 [review_stream_counts.py](review_stream_counts.py)、[review_stream_counts.json](review_stream_counts.json)。该脚本只读真实source/live和最终实测，不导入作者预测模型，没有把预测送给RTL。

## 状态、地址和工作守恒

- SV以tile id除/模160形成原生源origin，按`C*76800+y*320+x`请求完整源存储。load_index0..1535给出C和4×4局部坐标；TB只按请求地址返回一个低10bit词，未提供预拼tile或origin。首64tile固定id128..191，确实从首输出行末32块跨到次行首32块。
- signed16 origin含(-1,-1)，图外先由SV判断、写本地0且不发外部读。所有图内请求地址落在0..7372799；参数W行0..10367、mask行0..287。参数和源响应无效时地址、分类和请求保持，load_index不前进。
- 首个job才LOAD_W与LOAD_MASK，resident置位后在相同参数集上继续hot job；两模式配置权限相同。**resident没有重新配置参数命令，本合同只覆盖reset之间同一W/mask集**，不能将同参数warm restart写成任意参数热切换。
- 全1536个源位置装入后付一拍origin和一拍LAUNCH，再进入RUN_TILE。所有tile仍在原leaf内清空480行psum，输出480个8lane beat。wrapper只在看到leaf_done且accepted_beats为480时退休；最后输出与leaf_done之间有原leaf FINISH状态，不存在同沿计数漏更新。
- 结果tile_id保持到leaf_done，tile_last依据addr479，job_last同时判断最后tile。TB在valid握手时同时核tile/address/两last字段，并在输出背压时检查data及身份稳定。递增tile发生在完整退休后；新tile重置load_index，避免尾块/下一行错位。
- 输入非法模式、零tile_count或first+count越过19200进入FAILED。合法模式5/6和合法范围是本次正式合同；边界保护不等于已对所有非法输入进行了仿真覆盖。
- wrapper只增加参数驻留标志、地址/控制、退休/诊断状态，没有新增数据adder、psum端口或第二tile缓冲。原八条数据加法链和source/W/psum物理端口保持。实际地址除模、mux和握手控制尚未STA，不能由相同数据链数推出相同Fmax。

TB二进制读取source/gold的尺寸与dtype，只有访存响应及完整结果核对。整帧6job为三臂×mode5/6，各冷启动、无背压；跨行64tile另外覆盖两种背压和第二个同参数不reset warm job。不能说整帧背压/整帧warm也运行过。

## 独立计费

每job完整账单：

`total = Σleaf_core + source_load(1536F) + origin(F) + 2F + 1 + cold_parameters(10656或0) + external_parameter/source_stalls`。

leaf_core已经包含内部source/W/output stalls；单独从leaf_core扣除这些stall后，与前轮完整pair-parent公式独立核对。两个额外每tile状态拍及最终job FINISH均计入，warm只取消W/mask重新装入，未取消新源、clear或drain。

完整F=19200时，各模式/各掩码均装入29,491,200个native词，其中29,276,544次外部读取和214,656次图外置零，再付19,200次origin。几何式为 `96*(4*120−2)*(4*160−2)` 次外部源读；跨行64tile对应85,344次。每层初次W/mask仍为10368+288个配置事务。

**完整Cin退休目前只取消leaf中的计算源读取及消费者，不取消wrapper对这些通道的外部源装入。** wrapper没有根据mask跳过源装载，也没有跨tile halo复用。总装入已真实计费，因此不能把leaf source节省扩写成端到端上游生产或外存词退休。

每个live目的，令两pair时间并集u/v、H=popcount(u&v)、K=Iu+Iv。相同源/W请求、相同父和构造下，mode6多H拍MERGE，少H次读写/时间枚举，以及K−1次pair结束检查；`mode5−mode6=Σ(2H+K−1)`仍成立。两模式装入和wrapper控制相同，故此差也等于本轮无背压总job差。

## 完整帧实测

| 固定掩码 | mode5总拍数 | mode6总拍数 | 少拍数 |
|---|---:|---:|---:|
| dense Q16 | 1437874585 | 1361527081 | 76347504 |
| 普通块幅值25% | 1110577614 | 1053569256 | 57008358 |
| 完整Cin费用25% | 1094526589 | 1034824441 | 59702148 |

全部对应一帧、完整C96/N96/T10/K864、19200个连续输出tile，并包括静态参数、新源和wrapper账单。它把旧8tile证据扩展成真实完整帧连续执行；属于既有mode6的范围补证，未新增代数或结构新颖性。

这些是bank/寄存器模型下的周期实测，不是DRAM、AXI burst、缓存层级或物理SRAM时序。外部参数/源接口是每拍最多一词、可背压的响应模型；未引入异步多笔事务、隐藏双缓冲或预取。没有完整网络的BN/残差/PSN输出退休，未综合、STA或EDA，不把模拟器wall time当硬件时延。

三臂官方825 AEE已独立完成，结果及逐帧核对见 [README](README.md)。此审阅的功能/周期结论与浮点消费者质量分别取证，不把连续wrapper本身称新X。
