# 两父和按时间归并：独立审阅

2026-09-13。只读最终 [pair_parent_merge.sv](../pair_parent_merge/pair_parent_merge.sv)、[tb.cpp](../pair_parent_merge/tb.cpp)、[run.py](../pair_parent_merge/run.py)、[results.json](../pair_parent_merge/results.json) 和 [SUMMARY.json](../pair_parent_merge/SUMMARY.json)。本审阅没有改硬件作者文件、重跑 RTL、GPU、训练或扫描参数。

**在约定的完整 C96/N96/T10、原生4×4输入至2×2输出整数线性叶范围内，功能与计费核对通过，未发现具体未修复 bug。** 46 个 fixture、368 条命令、1,413,120 个 signed32 输出全部通过作者回放。独立从 source/mask/origin hex 推导的 3,680 项周期与事务检查全部相等；全部 184 条 mode5 命令的 2,392 项对应计数与上一版也相同，未降低强控制权限。

独立审核入口和逐 fixture 预测见 [review_parent_merge_counters.py](review_parent_merge_counters.py)、[review_parent_merge_counters.json](review_parent_merge_counters.json)。它没有导入作者 ledger 或运行其周期模型；读取其最终实测记录作比较。作者另有 4,048 项闭式检查，不能把这两组数相加当独立样本量。

## 共享资源、位宽与状态

- 源仍是1536×10bit单读口，权重8×10368×signed16、每拍同行128bit，psum为8×480×signed32，读和写分拍。`add_result=lhs+rhs` 仍只有原八条32bit数据链；父和构造、PAIR_MERGE和psum累加分时使用它们。两个 `pair_value` 是寄存器转发的组合mux，没有隐藏数据加法。
- 新增两个8lane signed17父和寄存器共272bit，双方mode5/6都保留并使用；merge诊断计数器另32bit，原signed18 scratch、原wa/wb/wab等存储没有撤销。mode5具备共同C4供数、完整死列跳过、消费者枚举和无空pair气泡。把这版mode5与旧版逐条核对，其输出及周期/读写/加法完全一致。
- signed16两数和范围[-65536,65534]，signed17精确；两个父值的和最多四项signed16，范围[-131072,131068]，signed18精确。SV在加法前分别符号扩展至32bit，落父和[16:0]或scratch[17:0]不会丢有效位；完整864项最坏绝对累加不超过28,311,552，signed32足够。没有新中间舍入、乘法或饱和。
- 最后W读只根据明确的四源AND决定第一项MAKE_SUM；最后W的非阻塞写在下个计算状态已可见。若两pair都需父和，先写parent0再切idx构造parent1；否则跳过不需要的父和。未生成的旧parent只可能对应从未出现11的pair，转发mux不会消费它。
- `NEXT_TIME` 用 `selected_time` 决定是否进入PAIR_MERGE，而不是尚未更新的 `time_idx`。随后time_idx已锁存；两pair活跃付一拍合并再PS_READ，单pair直接转发原W/父和给ADD_WRITE。合并没有读写psum，后面的PS_READ和ADD_WRITE都仍付费。
- 每个目的重新设置按需W pending，父和构造结束才装入四源OR pending；ADD_WRITE清最低时间位。完成一个目的以后再取下一个目的，源全四词在C4_CHECK前覆盖。正常完成后第二命令CLEAR全480行；辅助父和无需每命令清零，只要使用前上述覆盖条件成立。TB覆盖同配置两次无reset重启，未声称中断恢复或任意运行中重配置。
- cp按0,2,…,46前进，mask列cp/2对应完整Cin4，source最大1535、weight最大10367、psum最大479。signed origin图外读被SV置零且不计source事务；全死Cin4在任何源读前跳过。源/W背压期间地址与pending不推进，输出DRAIN_SEND在ready=0时保持data/address。最终功能fixture包括图外非零毒值。

数据加法器和存储端口相同不等于面积、时钟或能耗相同。新增父和同时读取、mux和priority路径都可能影响实现时序；没有综合或STA，不能将核心拍数比直接叫PPA提升。

## 独立完整周期等式

对每个图内native位置、live输出组及合法空间目的，四个10bit时间源字是a,b,c,d。令：

- `u=a|b, v=c|d`，`K=I(u≠0)+I(v≠0)`，只在四源OR非零的目的求和，所以K为1或2；
- `J=I(a&b≠0)+I(c&d≠0)`，为实际需要构造的普通父和数；
- `H=popcount(u&v)`，`L5=popcount(u)+popcount(v)`，`L6=popcount(u|v)`；
- Q为四个非零时间源字数，即该目的所需128bit W行数。

两模式每目的费用分别是：

`mode5 = 2 + Q + J + 3L5 + K`

`mode6 = 3 + Q + J + 3L6 + H`

完整每tile还包括固定1442拍，以及每live Cin4的112拍源/控制扫描、每dead Cin4的2拍跳过。图外赋零仍占其状态拍但不计bank读。全部clear、时间终止检查、消费者推进和drain均在其中。背压运行先减去实际source/weight/output stall数，再与同一绝对公式核对。

因此完整无背压节省恰好为：

`core(mode5) − core(mode6) = Σ_valid_destination [2H + K − 1]`。

H次合并各减少一个三拍NEXT_TIME/PS_READ/ADD_WRITE序列，但自己付一拍MERGE，净省2H；K−1来自少做pair时间集合的结束检查。mode6不是免费得到四源和，也没有模式4的按码copy/build费用。`sum_issues`已包含 `merge_issues`，控制周期分解不能把merge再减一次。

事务守恒同时成立：source/W请求不变，普通父和数J不变；mode6少H次psum更新、少H次读和H次写，新增H次计费merge。该关系对全部46个fixture成立；模式4先前的复制/构造负结果保留在 [REVIEW_C4.md](REVIEW_C4.md)。

## 固定真实评价结果

下表均为旧8个真实评价tile、每tile第一命令、无背压**核心拍数**。校准是独立train帧336栅格，这里的8tile未参与本轮mask选择；八块仍只来自一个评价帧，不是整层或多帧硬件统计。

| 固定掩码 | mode5 | mode6 | 少拍数 | 相对mode5 | 计费merge H |
|---|---:|---:|---:|---:|---:|
| dense | 391120 | 371056 | 20064 | 5.130% | 6360 |
| 普通块幅值25% | 303052 | 288134 | 14918 | 4.923% | 4743 |
| 完整Cin幅值25% | 302380 | 287080 | 15300 | 5.060% | 4944 |
| 完整Cin费用25% | 294316 | 277996 | 16320 | 5.545% | 5268 |
| mixed72 | 300848 | 285212 | 15636 | 5.197% | 5035 |

dense父和构造仍2496次，psum更新73428→67068，总sum2496→8856（其中merge6360），源读9600和W读41088均不变；20064=2×6360+7344。完整Cin费用父和2052次不变、更新54816→49548，源读7200、W读30840不变，节省16320。

新tile源/origin装入1537拍，八块统一另加12296；首次W/mask装入10656拍再计。dense含八次新源装入403416→383352，完整Cin费用306612→290292。不能把TB第二条复用配置命令当新的输入免费装入。背压测试的调度相位依赖不同执行周期，实际stall数可以不同；功能和扣除实测stall后的core都已检查。

## 可接受的结论范围

该兄弟接口在固定布局里兑现了psum合并收益，而且完整计费后仍比无空对气泡的共同供数mode5快约4.9%–5.5%。原模式4失败针对按码复制/构造接口；当前成功针对两个普通父和转发、逐时间归并接口，二者都应保留。模式6的父和共享、bitmap和稀疏迭代本身属于已有执行A，不能单凭拍数收益宣布新论文X。

掩码和Q16线性函数保持不变，六臂diverse10见 [README](README.md)：完整费用AEE1.2603504201296971、普通块1.18599666968428，均低于历史NB0门1.45460286107；质量是相同Q16系数解码后的浮点/TF32消费者，不是全网bittrue。没有valid825、上游PSN生产退休、全层halo账单或频率/PPA结果。本次最终接口审核完成，不追加新实验。
