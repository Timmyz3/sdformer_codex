# Normalized certificate：周期不变的组合路径适配

**本版496命令全部通过；480条真实整链/生命周期记录的周期和访问计数，与冻结joined_fc1逐条完全相同。没有新的cycle收益。** 32real完整cold ready仍为native184408、full227219、cert176448（cert省4.3165%）；cold BP仍为212997/255588/204698（省3.8963%）。对应整命令时钟break-even仍是95.6835%/96.1037%，没有回到独立post-Y叶的82.32%。

本次仅针对旧LUT→两次prefix加法→可变左移→界加法→比较这条尚未映射的长组合链。固定等价变形将移位放在寄存阈值支路，删除tail递推，不增加GROUP等待、不改变分组、退休策略、native或前级FC1。新增860B净状态，160个阈值右移器也显账；未测面积、Fmax、能耗，不能由算术位置减少推出PPA收益。数学等价变形属于已知方法，不作为强新颖性。

真实接口保持

从[冻结joined_fc1](../joined_fc1/README.md)独立fork，沿其mode4内容去重强前级、真实g′源、唯一92160B Y及共享读口、密排397词loader、实际120B门pack和逐t H96消费者。A/tau各一份，cold费用完整；warm仍要求host保证同模型，硬件只核config_valid/hblock。原native96mult、96门比较路径、U存储及共同数据预算全部保留。没有把CPU的Y替代真实FC1，也没有接入source classifier。

[normalized_fc1.sv](normalized_fc1.sv)保留原状态/握手，80个[normalized_bound.sv](normalized_bound.sv)实例构成新判界数据通路。独立Y24诊断的[normalized_leaf.sv](normalized_leaf.sv)也实例化**同一份normalized_bound实现**；该诊断叶的direct-Y边界明确分列，不作为FC1可达分布或整链性能样本。

等价式与实际两输入加法

记P为A每行的正系数和，N为负系数和，m为剩余位数，nv为本plane更新后的prefix。原界可写成：

```
lo = (nv + N) * 2^m - N
hi = (nv + P) * 2^m - P
delta = positive_gain ? 1 : 0
qN = tau + N - delta
qP = tau + P - delta
lower_hit = (nv + N) >  floor(qN / 2^m)
upper_hit = (nv + P) <= floor(qP / 2^m)
```

lower赋positive，upper赋!positive，constant gate先锁定。正gain对应U>=tau，负gain对应U<=tau；delta使m=0的严格/非严格比较和所有tie准确。signed算术右移给出负数floor。tau为signed48，q用signed49，不能将tau端点加P/N后截回48。

GROUP仍只用两个bound加法位置。P/N缓存改存P−1、N−1：FPNINIT写−1，原20个累加拍不变。GROUP实际执行`q=tau+stored+(1−delta)`，PLANE同位置执行`norm=nv+stored+1`。每个都是两个signed49操作数加carry-in，没有把第三个加法藏在GROUP里。GROUP一拍同时产生符号prefix并寄存q；后续每plane一拍比较。full只在末位采门，cert可提前锁定，两者同新数据通路。

q不需要额外reset清零：每个FPLANE路径都先经过相应FGROUP写入两份q；跨模式warm测试包含native未构表→首次full构表及随后cert。没有保留旧tail、旧`nv<<m`或旧bound观察电路；TB只观察真实norm、q和右移阈值，再由CPU独立计算旧数学包络进行核验。

| 路径 | 旧joined | 本版 |
|---|---|---|
| prefix到判定 | LUTmux→48add→48add→可变左移48→界add48→比较 | LUTmux→48add→48add→界add49→比较49 |
| 阈值支路 | tau选择与旧界比较 | q寄存器→算术右移49→比较49，与prefix并行 |
| GROUP阈值建立 | tail初始化变移/相减 | tau选择＋49bit两输入加carry-in→q寄存器 |
| 所有门退休 | 80门锁定归约 | 相同归约/状态转换 |

这只说明RTL拓扑变化。q支路、49bit比较、fanout、布线和寄存开销都可能成为新瓶颈；实际关键路径及频率必须经后续映射检查，当前没有速度通过结论。

资源差额

| 项目 | 旧joined | normalized |
|---|---:|---:|
| 共享第一层48bit ALU | 96 | 96 |
| 第二层prefix48bit | 80 | 80 |
| 界加法位置 | 160×48bit | 160×49bit |
| tail减法位置 | 20×48bit | 0 |
| union加减位置总数 | 356×48bit | 176×48bit＋160×49bit＝336 |
| P/N缓存 | 120B | 120B，内容为P−1/N−1 |
| tail寄存器 | 120B | 0 |
| qN/qP寄存器 | 0 | 160×49bit＝980B |
| 数据状态净差 | — | **+860B** |
| prefix后的可变左移 | 80×48bit | 0 |
| tail初始化可变左移 | 20×48bit | 0 |
| 参数可变算术右移 | 0 | 160×49bit |

q每个GROUP实际写160个49bit寄存器，无额外memory端口声明；不是免费常量表。fused增量数据数组从5020B变为5880B；原主要数组额定120888B变为121748B，指数/控制等口径保持。单份1280B LUT及160个32:1读mux、2880B ybuf、120B gatepack、native96mult/比较/U均不变。这里只比较结构位置和位数，没有将336/356解释为面积降低。

周期与负例保持

| 32real完整周期 | native | full | cert | cert/native节省 |
|---|---:|---:|---:|---:|
| cold ready | 184408 | 227219 | 176448 | 4.3165% |
| cold BP | 212997 | 255588 | 204698 | 3.8963% |
| warm ready | 159000 | 199091 | 148320 | 6.7170% |
| warm BP | 165340 | 205189 | 154330 | 6.6590% |

每条state_cycles、真实FC更新、W请求、Y bank读写、gatepack、output/backpressure、cold/warm构表及planes/early均与joined相等。原cold ready30/32获益、BP31/32获益，以及zero/signed/tie负例一并保留；没有新挑数据或将负例消失归于此等价变形。[SUMMARY.json](SUMMARY.json)保留逐字段比较结论和原聚合，[results_all.jsonl](results_all.jsonl)、[results_swap.jsonl](results_swap.jsonl)为当前真实执行记录。

Y24与tau48端点的独立RTL诊断

[prepare_diagnostic.py](prepare_diagnostic.py)使用实际A与signed24混合Y，独立int64矩阵乘法计算完整U/门，CPU不做截断救场。两案均含−8388608、8388607、0、±1、跨t翻转，384组指数全部24，混合正负gain和constant。每案full/cert×ready/BP×cold/warm共8命令。

| 案例 | tau范围 | q最低/最高关键值 | 超signed48的q项 | ties | full/cert planes |
|---|---|---|---:|---:|---:|
| 原Y24 full-range | [-125825203769,112176373235] | qN min−125825226356；qP max112176394096 | 0 | 850 | 9216 / 4714 |
| Y24＋tau48端点 | [−140737488355328,140737488355327] | qN min−140737488387014；qP max140737488377789 | **480** | 360 | 9216 / 4154 |

原Y24案的8条周期也与冻结fused叶逐条相同。新案确实越过48bit中间范围，验证49bit必要性；360个tie与负数q的floor右移均实际通过。两案U范围都是[-125825203769,112176373235]，完整U适配原signed48。它们经诊断叶的真实Y写入/读出/运行时指数路径，不从TB直接馈入norm或q。

另外对实际A的P/N、m=0..23、两gain、tau端点和每个lo/hi邻接阈值做73920个独立代数谓词检查。CPU探针不代替RTL周期或实际判定；数据见[diagnostic_inputs.json](diagnostic_inputs.json)、[results_diagnostics.jsonl](results_diagnostics.jsonl)。

验证结果与复现

[run.sh](run.sh)使用Verilator4.028 `--cc --exe --unroll-count 512`＋make；先84小命令，再444主、36跨模式warm及16直接Y诊断。最终报告按496命令计，小集是重复功能预检。没有新训练、生产修改或EDA。

| 最终496命令核验 | 次数 |
|---|---:|
| 最后gate | 15237120 |
| 输出Y/暂存读回 | 25067520 |
| native完整U＋fused full U | 10076160 |
| 实际normalized49和 | 55944960 |
| 实际寄存q及算术右移阈值 | 111889920 |
| 两个判定谓词与旧数学包络恒等 | 55944960 |
| runtime指数 | 129024 |
| 实际LUT写值 | 107520 |
| 整链实际Y读 / 部分写值 | 123564192 / 91169280 |

`normalized_value_checks`等新字段替代旧bound/tail观察计数；其余480条整链结果逐字段完全相同。[summarize.py](summarize.py)执行这些断言，不从周期近似推测一致。完整普通native、full和cert函数均保留，cert没有完整U承诺。

最终版本还把prefix生产与normalized结果消费拆成独立always_comb，以消除Verilator4.028跨cell的大组合过程产生的UNOPTFLAT假环。没有增加寄存或算术级；[comb_split_check.json](comb_split_check.json)核相同496条的全部字段/周期不变。最终两核构建均无UNOPTFLAT、LATCH、MULTIDRIVEN或错误；未使用压制这些警告的选项。

本版冻结在此固定归一化布局。它为后续有限物理映射提供了可运行的另一种组合路径，但未证实Fmax提高，也没有打开额外MAC尾部调度、流水或分组扫描。真实新接口的必要性及物理效果仍须用对应强对照实测，不能用这个等价式本身宣称强新颖性。
