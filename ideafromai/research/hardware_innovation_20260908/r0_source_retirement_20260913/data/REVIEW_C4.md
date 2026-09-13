# C4 支持码执行的独立审阅

2026-09-13。审阅data分支只读 [c4_execution.sv](../rtl/c4_execution.sv)、[tb.cpp](../rtl/tb.cpp)、[run.py](../rtl/run.py) 和最终结果；没有改作者文件或重新跑RTL/GPU。

**功能、资源权限和完整周期核对通过，未发现具体未修复bug。** 54个fixture、648条命令、2488320个signed32输出通过。独立公式对全部648条运行的9个事务/周期字段共 **5832项**检查全部相等（含背压及第二次不reset命令），见 [review_c4_counters.py](review_c4_counters.py)、[review_c4_counters.json](review_c4_counters.json)。该执行点mode4比强mode5更慢，是应保留的布局负结果。

## 静态检查

- 源读确为同一个10bit单口，每native位置四拍读四通道，cp只走0,2,…,46。其source最大地址1535、W最大10367、psum最大479；与Cin4结构掩码cp/2对齐。signed origin边界在每次源读前生效，图外直接赋0且不计bank读；全死Cin4在读前SKIP_BLOCK。
- mode4从四个源hold在SV内形成十个4bit码、15bit非零支持集合、4bit所需W集合。TB没有输入支持码、中间和或动态索引答案。C4_READ_W只清最低需要位，四W同一128bit端口逐行取；缺席W不会因遗留寄存器而被使用。
- C4_PATTERN付一拍选择和复制首W至唯一八lane signed18 scratch，再对remaining逐项使用原八条signed32链，加`popcount(k)-1`拍。sum没有由TB计算，没有15项向量和表，也没有预构造未出现的码。
- 清当前pattern bit后，`pending`已经锁存旧pattern的matching_times，后续BUILD只看锁存remaining；组合priority变成下一pattern不会把时间集合串走。最后一次BUILD的scratch非阻塞更新在后继PS_READ/ADD_WRITE前完成。
- mode5获得同样四源、四W缓存和一次消费者枚举。最后W拍直接用明确的第一非空pair源设置pending/AND；第一pair结束时也明确用第二pair源设置pending/AND，下一拍索引已更新。没有强制pair选择气泡或空pair扫描，也没有读旧pair的新pending的时序错误。
- mode5 MAKE_SUM只在某个实际t存在pair AND时执行；单活源直接用W，双活源用signed17父和。mode4与mode5都共用原8条`lhs+rhs`数据链，未新增数据adder/乘法器或psum多读口。新增scratch/四W暂存和组合mux是双方同一SV中的共用预算；这不等于已证明面积/Fmax相同。
- 四项signed16和范围[-131072,131068]精确可容signed18。BUILD的32bit结果取[17:0]在合法界内不截断有效值，随后正确符号扩展；pair和signed17也足够。完整K864最坏累加约28311552，signed32安全，无新中间舍入。
- 合法完成路径清空time、destination、pattern、remaining和W pending；新的source四词在CHECK前全部改写，新的目的W pending重新设置，scratch在每pattern消费前重写。虽IDLE不重复清所有辅助寄存器，正常再启动不消费旧值。TB两命令检验psum清零和旧状态；中途非法中断及6/7未定义模式不是本版本协议。
- 背压期间源索引/W pending不变，已选destination被执行状态持有；输出沿用固定DRAIN_READ/SEND，ready=0时data/address保持。全部clear、copy、构造、NEXT_TIME末检查与最终send都计入core。

TB与前轮相比主要为模块名和pattern_copies计数，仍核完整480beats×8lane；追加全支持/重复码/极端signed16fixture覆盖signed18两端。无软件先算动态码/和来替代RTL执行。配置12193拍，后续新tile source/origin1537拍的边界保持；重复命令不是新数据样本。

## 独立事务与周期等式

对一个Cin4/native位置及一个live几何目的，四个源字为a,b,c,d。令：

- `u=a|b, v=c|d`；H=`popcount(u&v)`，是两pair同t都活跃次数；
- K=`I(u≠0)+I(v≠0)`，J=`I(a&b≠0)+I(c&d≠0)`；
- D为T10出现的不同非零4bit码数；B=`Σ_k(popcount(k)-1)`。

mode5更新`popcount(u)+popcount(v)`次，mode4更新`popcount(u|v)`次。因此mode4少H次psum读取和H次写回；源/W请求完全相同。mode4的构造加法B和复制D均付费，mode5父和J也付费，双方有各自实际的NEXT_TIME末检查。

完整无背压差为：

`core(mode5) − core(mode4) = Σ_valid_destination [3H + J + K − 2D − B]`。

3H来自每个被合并时间项的NEXT_TIME/PS_READ/ADD_WRITE三拍；不能只报2H访存节省而漏掉支持复制和终止控制。独立脚本另外推导每个mode完整core：固定1442拍，live C4的source/循环mode3为160拍、mode4/5为112拍，dead C4各2拍；再加上述实际目的费用。全部正常点与扣除实际stall后的背压点都精确闭合，不以公式替代作者的Verilator周期。

## 最终实测与解释

下表每臂是原8个真实评价tile、每tile第一次命令、无背压**核心周期**；每新tile的source/origin另加1537拍（8块共12296），W/mask首次装入10656拍另计，三个模式相同。

| 掩码 | mode3 原两源枚举 | mode5 共用C4供数、两pair | mode4 按支持码合并 |
|---|---:|---:|---:|
| dense | 415024 | 391120 | 422008 |
| 普通块幅值 | 323132 | 303052 | 326364 |
| 完整Cin幅值 | 320116 | 302380 | 325960 |
| 完整Cin费用 | 312796 | 294316 | 316660 |
| mixed72 | 321196 | 300848 | 323513 |

mode5是必要强A：它单靠共用C4供数与循环合并已降低原mode3费用。主增量应看mode4对mode5，不能把全部mode3差额归给支持码复用。

dense mode4把更新73428→67068，少6360次读改写，但构造加法2496→7572，新增40596次付费pattern复制，最终比mode5多30888拍。完整Cin费用也把更新54816→49548，仍比mode5多22344拍。**支持码路径精确、事务节省真实，但新增构造与控制吃掉收益**；各正式臂约慢7.5%–7.9%核心周期。全一合成源能展示另一分布下的复用收益，不替代当前真实分母。

两个完整Cin控制确实把源读9600→7200，普通块和mixed仍9600；与掩码全死列吻合。质量另见 [README](README.md)：所有臂过十帧NB0条件，但原生输入组剪枝本身仍是普通A，未因source词减少就构成新机制。

未综合或STA，source/W/psum仍是所述RTL bank和寄存器/mux模型；没有SRAM/Fmax/PPA或完整层、PSN/BN后继结果。本审阅接受本固定点的功能和负周期结果，不能扩大为整个product-reuse家族无效，也不将模式5普通控制改进称为论文X。
