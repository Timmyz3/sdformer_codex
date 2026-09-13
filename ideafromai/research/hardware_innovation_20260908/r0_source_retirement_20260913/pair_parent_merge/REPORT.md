# 父和转发与按时间归并：保留共同底座后得到真实净收益

2026-09-13，本阶段最后一个有界接口已封口，停止新增实验。**368 runs、1,413,120 个输出全绿；4,048 项独立源/几何/完整周期核对通过。** mode5 作为同模块强控制，周期及所有既有请求逐记录完全复现上一目录；mode6 在五个真实分支净减 4.92%–5.55% 核心拍。[完整结果](results.json)、[汇总](SUMMARY.json)、[费用表](benefits.csv)、[共同资源](resource_contract.json)。

## 完整继承与本次唯一执行变化

前一目录 [单 scratch C4 支持码归约](../rtl/REPORT.md) 已完成且未获益，本目录不修改其数据、结果或构造点。继续完整继承 mode5 原生 C4 四源读取、按需四 W 暂存、静态 O8×C4×3×3 mask、48bit 合法消费者枚举、相邻两源父和、无空对选择气泡、八条数据加法链、psum 分拍读改写、origin 图界、clear/backpressure/drain。

新增 mode6 仍先按真实 T10 AND 判断每对是否需要 `W0+W1` 或 `W2+W3`，各最多构造一次，暂存两个 signed17×8 父和。mode5 获得完全相同父和寄存器权限，仍分两对更新 psum；它也实际从相应父和寄存器读取 11 情形，保持旧周期，不被额外选择/复制拍弱化。

mode6 对四源的完整 T10 OR 逐时间枚举。若当前 t 仅一对活动，直接选择原 W 或该父和作为 psum RHS；不复制到 scratch。若两对均活动，支付一个 `PAIR_MERGE` 拍，用共同八条 signed32 加法链将两对值合成 signed18 scratch，再分别付原 `PS_READ` 和 `ADD_WRITE`。最后一次 W 读或最后父和构造直接进入 `NEXT_TIME`；没有父和挑选空状态。支持码、父和需求、合法目的、时间队列和 merge 决策均来自 SV 中真实源字，TB 只发送原生配置并检查独立完整卷积 gold。

## 完整真实叶实测

固定 C96/N96/T10、3×3 全 K864、连续4×4源→2×2输出、八个真实 tile；每行模式5/6共用同 source/Wq16/mask/gold。下表是 command0、无外部背压、包含 clear/drain 的实测总和。

| mask | mode5 core | mode6 core | 核心减少 | mode6 含每tile新源/origin | 新增MERGE＝少掉的psum更新 |
|---|---:|---:|---:|---:|---:|
| dense | 391,120 | 371,056 | 5.13% | 383,352 | 6,360 |
| block_magnitude25 | 303,052 | 288,134 | 4.92% | 300,430 | 4,743 |
| cin_magnitude25 | 302,380 | 287,080 | 5.06% | 299,376 | 4,944 |
| cin_fullcost25 | 294,316 | 277,996 | 5.55% | 290,292 | 5,268 |
| mixed_retirement25 | 300,848 | 285,212 | 5.20% | 297,508 | 5,035 |

每次 MERGE 取代第二次 psum pass，因此同样各少一次 256bit psum 读和写。source 与 W 请求逐模式完全相同：dense 为 9,600 个10bit源词、41,088个128bit W词；Cin fullcost 为7,200和30,840。父和构造数也不变：dense2,496、Cin fullcost2,052；mode6的 `sum_issues` 是父和数加 `merge_issues`，不能把两者重复计作不同 ALU。

每个新tile的source/origin配置另付1,537拍，已包含在表第五列；全八块另加12,296。W+mask静态配置10,656拍可驻留一次；TB的每个新实例实际完整配置12,193拍。若按八个cold实例算总量，核心上再加97,544拍。两条 command 使用同一已加载的tile：首条实际配置12,193，第二条0；输出JSON重复显示的 `configuration_cycles` 是一次配置描述，不应两次相加。Cin静态退休在内部source访问前取消对应C4执行；本TB仍发送所有1,536源词，未宣称上游输入搬运已同步退休。

source/W/result三类command-relative背压下同样全过，dense mode5/6核心为408,283/388,875；Cin fullcost307,711/292,120。两模式状态时间改变后遇到的外部许可相位不同，因此带背压总差由实际握手计数报告，不能直接套无背压净差。

## 全部净差的物理解释

对每个实际C4源及目的，设H为两对同时活动的t数，K为有活动的pair数，L为四源OR的活动t数，Q为需要的W行数，J为真实需要的pair父和数。共同C4供数外，每目的mode5费用为 `Q+J+3(L+H)+K+2`；mode6为 `Q+J+3L+H+3`。三拍psum pass包含时间选取、读、写；最后三项常数来自目的选择、空pending终止和ADVANCE。差为：

`mode5 − mode6 = 2H + (K−1)`。

dense实测H=6,360，少掉的pair终止扫描共7,344，净省`12,720+7,344=20,064`拍。一次真实MERGE支付一拍，换回被省的三拍pass；单pair直通不增加数据拍。这个式子来自输入工作量与最终SV状态，并由[独立检查](verify_ledger.py)/[检查输出](ledger_checks.json)逐fixture对齐，包括source图界、静态全删、父和数、merge数、clear/drain和握手阻塞。

## 同资源、公平性与边界

源仍一个10bit读口；W仍8个16bit bank每拍一个共同行；psum仍8×480×32bit、读与写分拍；数据仍八条32bit加法链。两个父和额外272bit（34B）为两模式共同功能状态，merge计数器另32bit；原四W向量与18bit scratch全部保留。允许同时选择两个父和是这34B寄存器的连线，未增加W SRAM或psum读口。旧 `wab` 也保留在联合模块，未用寄存器复用假设降低费用。没有EDA、Fmax或能耗数据；这里只对同模块的周期和事务作结论。

测试由40个正式真实tile和6个固定功能输入构成，每模式两个许可波形、每次两条同模式不复位重启。`all_supports_extreme`覆盖0..15全部码、重复时间、−32768/+32767及混合符号；另测全零、全一、角点、全mask及图外毒值。所有480beat逐lane对gold，并检查输出被阻塞时addr/data保持；mode5对上轮每条记录的core、source/W/psum/父和等完全一致。当前结果只验证模式5/6；SV中保留的旧模式不被自动宣称已在此新模块重跑。

完整Cin mask来自独立训练帧校准，未按mode6收益重新选择；所有AEE沿[数据报告](../data/README.md)的共同mask/precision评估。该RTL变化是同函数整数线性叶，未仿真全图输入流、后续BN/PSN/残差、动态源最后使用通知或提前psum完成/输出；也未把先前浮点AEE冒充逐层bit-true全网结果。

父和共享、按消费者归并和product reuse属于完整借入的A。本次实测说明“逐时间父和转发→一次真实MERGE→一次psum更新”在当前数据上比两pair分别更新合算，且避免了上一固定码构造点的复制成本；**这约5%的执行净增量本身不升级为标题X，也不宣称完整Prosperity、ELSA或Phi迁移。** 不进行新的容量/组宽/回退扫描。复现为`/opt/anaconda3/bin/python3.12 run.py`后`verify_ledger.py`，Verilator4.028使用`--cc --exe`加独立make。
