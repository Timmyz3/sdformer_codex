# 完整K864地址接口独立审阅

2026-09-12。核读 [PLAN](PLAN.md)、最终 [execute.py](execute.py)、[results.json](results.json)、汇总脚本与固定压力记录，未修改被审源代码。**功能与当前计费对照通过；没有测出可归于 phase 的新增 X。** 只停止把此地址开关当标题，不以本结果否定 phase/lifting 的训练或表示家族。

**强对照已纠正。** 原 `directory` 仅有最后一个SR64响应缓存，c-major跨9邻域会反复读取同字的4通道；因此原扫描只作诊断，H4整字收集必须共享。审阅发现并修正了两处偏弱分母：普通predicate不再支付它不用的offset位图构建；四phase掩码相同者按同一规则获得uniform特化，零H8组直接越过。现在每个实际已masked负载分别比较 scan/coalesced/predicate/iterator；三种mask均给相同端口、RF、静态许可及驻源MAC。跨mask差异属于不同剪枝结果，不能叫同一个网络的无损优化，也不能从其原始AEE继承恢复训练版本的精度。

**功能/费用核对。** 我独立从JSON重算12负载×4臂的384次目录和7,418,880个后继值，全部0差；每次目录的positions、记录数、live一致，源码还逐字比较保持c×9顺序的完整NRV。原整数gold保留K864、Acc48、RNE/sat、原I24、projection门及真实U32/V96，非锚点连续门消费者也在。各同mask臂的NRV写字节、后继系数读及网络变化指标相同；静态臂仅新增32B metadata写，阶段槽之和与总服务一致。固定ordinary/interior/phase压力四臂也过，但没有包含原生projection卷积、全域BN或整帧。

H4的18个SR64以实际响应拆成uint16，占RF64–72的9向量：**144B有效载荷、432B物理RF位**，不是只付144B RF。几何RF80–88、mask92、NRV93不重叠；目录返回后才开始消费者，未跨消费者保留被覆盖的cache。48bit掩码拆四个12bit行，避免FP32丢位；源phase取sy/sx，P2单侧许可/边界先清另一半，再合20bit门字。NRV更新读cache和live两RF；掩码许可读几何和mask两RF，原单写回等待保留。地址、边界、H8查表/位图、选择、解包、写出及metadata冷填均收费，高水位已修为122912，仍在原128KiB状态内。新增整数操作的延迟是CPU机器合同，尚无RTL/频率/PPA结论。

| 同窗口、各自最佳已实现静态控制 | global服务槽 | phase_joint服务槽 | phase慢于global |
|---|---:|---:|---:|
| ordinary / corner | 628,013 | 631,209 | 0.509% |
| ordinary / interior | 668,719 | 672,141 | 0.512% |
| lifting / corner | 618,179 | 622,159 | 0.644% |
| lifting / interior | 660,735 | 662,927 | 0.332% |

**增量归属。** 同mask的predicate和iterator在最终记录中**所有物理端口字节完全相同**，后者仅省控制访问；没有新增权重或源请求压缩。phase/row-phase留出普通边界后，内窗省1152次遍历却付864个位图构建槽，只剩288槽；corner的768槽还含边界空项。相对同mask predicate仅约0.043–0.123%，global内窗反而多168槽。固定压力为765338/757402/753050/752666，iterator对predicate仅省384槽（0.051%）。普通NRV、字复用、静态跳过、优先编码都归已有供数/编译底座；本次没有支持“phase地址接口超出这些底座”的性能或新颖性主张。

**不杀家族的一项明确限制。** 当前phase仍逐H8×offset运行查表，而本次锚点全为偶/偶：固定mask可在装载时编为每H8的9bit offset表，单目的相位共108bit，再与运行时边界valid相交。此普通预编译接口尚未执行，若以后采用也须同时给predicate/iterator并计表读/选择；不能把当前逐offset控制费当作phase表示不可改善的证明。这不是新的标题或本轮追加任务。新精度门已放宽为同设置优于SDformerFlow baseline；它不会把上述相同费用的小增量自动变成硬件创新。
