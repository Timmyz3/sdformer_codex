# 独立审阅：双空间P的signed13打包

审阅冻结 `packed_rtl/packed_r8.sv`、prepare.py、tb.cpp与results.json。没有重跑RTL或模型。独立从所有14份fixture原生source/Q1/Q2重建53,760个gold，核对112条收据（430,080输出）的3,808项状态/工作/周期，全部一致。代码与结果：[review_packed_counts.py](review_packed_counts.py)、[review_packed_counts.json](review_packed_counts.json)。未发现固定函数、已列共享资源与同周期接口中的阻断。SIMD分割进位和普通缓存输出驻留均是已有技术，结果不提供单独新颖性结论。

## 真正打包的维度

`row=(P//2)*10+t`，低/高半分别P%2，P按2×2行主序。mode15枚举(P0|P1)与(P2|P3)的两个10位时间并集；selected_time%10同时控制这两个空间成员各自源位。不是跨t配对，也没有伪造连续时间。mode14同一packed物理字中每次只更新一个P半；它享有同520B状态、208bit向量读写、160bit原生窗口、相同Q1/Q2供数与相同第二阶段。

每条32位加法链在ZADD的bit13切断进位，低13位溢出的模进位不能进入高13位。两半各以3位Q1符号扩展到13位，不活动半加0；高半可能产生到bit26的进位，但截取低26位不影响两半。合法逐步Q1部分和绝对值≤864×3=2592，因此每半signed13不溢出。离开ZADD时bit13连接普通进位，用同八条32位链做最终MAC；没有额外八条独立13位数据加法器。

所有z由source逐K计算。TB只配置source/origin/Q1/Q2/k_live并读gold，不提供z或支持位；prepare复用旧独立gold，本审阅又以`patches@Q1.T@Q2.T`及完整展开矩阵双重重建。不存在TB提前计算低秩状态再喂核。

## 完整消费者与共同强控制

20个packed行清零；全K结束后40个P/T位置真实扫描非零rank，形成position_live及rank_live。最终ZSCAN更新非阻塞寄存器后才进VLOAD，最后位置也计入rank支持。每O8组最多缓存8个Q2向量（qblock共1024bit），仅rank活跃且对应Q2向量非全零才取W，但所有96个VLOAD控制状态仍收费。

POSLOAD逐位置清8个本地acc并按最终非零rank×静态Q2非零向量枚举，BASE_MAC累加，STORE覆盖全部480个psum行。空位置也STORE零，因此不额外清psum仍无旧命令残留。最后全部480个8lane结果由DRAIN_READ/SEND连续送出，稳定持有直到ready。两次命令不复位及源/W/输出背压已覆盖。

关键时序边界：BASE_MAC在单周期组合完成selected_rank选择、同一208bit latent字的半字/银行mux、scalar读取、signed19×signed13乘法和32位加法再写acc。Q2实际仍为signed16符号扩展到19位。该异步组合MAC路径对mode14/15完全相同；没有独立Fmax/宏SRAM读延迟证明。不能拿旧mode11分拍scalar/psum接口的频率假设直接套用本核，也不能把旧→mode14全部差额归于双P打包。cachedOS、显式读写状态变化和更宽latent端口都是共同控制的改变，已经体现在baseline14。

## 独立计费

设K为活Q1列，Q为至少一次P4×T10活动的活K列，A为全部P4T10脉冲数，U为两个相邻P对的时间并集数；D为这些对同t重叠数，逐源恒等`A−U=D`。令M为最终实际非零z与活Q2输出组形成的MAC向量次数，V为最终活rank×活Q2组权重词数。

```
core14 = 5437 + K + 2Q + 3A + M
core15 = 5437 + K + 2Q + 3U + M
z_vector_reads = first_updates + 40
z_writes = 20 + first_updates
z_scalar_reads = MACs = M
weight_words = Q + V
psum_reads = psum_writes = 480
```

5437包含20次z清零、96次通道启动、1536次原生加载、864次gather、864次KNEXT、40次支持扫描、96次VLOAD控制、480次POSLOAD/STORE/DRAIN_READ/DRAIN_SEND各一，以及FINISH。K付CHECK；2Q付QREAD与TIMESEL空出口；每更新TIMESEL/ZREAD/ZADD三拍。完整observed cycles另加实际source/weight/output stalls；TB逐状态和也等于cycles。配置首命令3361拍，第二命令0，未漏持续参数驻留口径。

八个真实tile：A=6413、U=5245、D=1168，所以节省严格为3D=3504拍；实测91103→87599。仅上述1168个共同活动合并带来3504周期差。背压实测94863→91295，额外差异来自请求/完成时间改变后与同确定性背压的相遇位置；工作量不能据此多算。

范围为完整C96/N96/K864/R8/T10的单tile核及上述14种输入，不是整层已测结论；连续消费者结果由另外wrapper交付。本审阅不提供训练、AEE变化或面积/能耗/时钟收益。
