# 固定资源的真实源/PED交织

B：当前source/preview/双消费者串行；13个活跃RF本身没有兑现服务收益。普通34已过825，普通两项dense也有13RF低状态控制，故不能把小活跃集直接当X。

A：完整常量编译/既有source ISA、普通resident MAC、P1留Z、有限状态与就绪调度。此次不声称完整复现某颗Gustav/LoAS芯片。

候选X：在真实updated-I24之后，源与连续PED的不同就绪空槽能否互填；给普通函数相同重算、分块、常量编译权限后仍有净服务。单纯两项量化、分块、CSE、双流排队均不独占X。

固定函数：新dense两项、新lifting两项、原同预算34。两量化函数十帧已过NB0，自己的825并行执行；34已有新825。每函数同一输入、完整数值及终点独立比较，质量不互继承。

共同边界：同帧A=corner完整源halo→K864 preview→非因果sn2→第一对实际anchor的Conv2/merge/投影门；完成updated后，将这对anchor完整PED U24/V96与B=interior完整source halo交织。计入共同前缀、整数系数替换、全部B输入DMA、B门存储、A连续输出及最终排空。B后续preview未包含，这是有界局部流水实验，不是两窗口完整推理或整层。

共同资源：单96×8×48RF、单ready/pending、单issue、SR64/SW64、CR256，128KiB状态与系数，源ROM512字；现有24B gather、3B标量collector、16B源门collector及单SR/CR响应，不新增第二RF/响应缓存。使用协程只暂停同一个Machine的操作请求，不从两条独立时间线取max。SR响应及gather有显式所有权；槽级等待可由另一流合法工作填充。

预定强控制：
- 完整CSE源＋原P2/H32、4RF供数串行；完整CSE源＋P1/H48留Z串行。
- 完整CSE源＋2RF供数交织；最大输出分组由(95−源活RF−2)//20推导，上限H32。dense为H16，34为H24，lifting为H32；不扫描分块。
- dense/34另给固定逐行两链CSD(13RF)＋H32交织；保留相同程序/同2RF供数的串行消融。lifting保留原35次norm，不强行变成dense矩阵。
- 所有串行/交织都可选择源先或PED先中的既定源先共同顺序；交织固定round-robin，阻塞时让另一就绪流进展。没有源队列深度扫描。

先跑所有固定ready对照；再把每函数ready最优交织与其最强普通串行放到原SR末8/SW末4的32槽压力形状核对一次。输出必须等于相同新函数独立CPU金值，并保留参数已核对的GPU窗口。正负结果只裁决这个工作组合及控制点；整体source/PED比例不合适不能写成所有交织失败。
