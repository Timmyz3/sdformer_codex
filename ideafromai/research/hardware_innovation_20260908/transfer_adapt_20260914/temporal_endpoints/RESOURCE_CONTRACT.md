六mode在同一`decomp_core.sv`内执行，保存共同union状态。mode0直接、mode1原序全端点、mode2每有效K付费选择且两域提前清零、mode3必要条件过滤并首次端点时清第二域、mode4再用一个固定cal顺序并按逆地址退休原T raw。mode5与mode4采用相同排列、配置和逆地址，但所有K固定用端点，只使用单域并付prefix，不做SELECT、第二域clear或merge。

|资源|六mode共同硬件与服务权限|
|---|---|
|producer ALU|8条显式32bit carry chain，2P13在bit13切carry；下降端在同一链的bit0/13注入独立carry-in，没有专用负权重加法器|
|producer乘法|唯一8个signed19×13表达式，原Q2完整MAC复用；Q1为权重加减|
|z状态|8银行×40行×26bit=1040B；两个域各520B。相比旧native direct增加520B；直接mode同样预留此状态，但不能声称旧单域同面积|
|z端口|全局一个208bit向量服务；每银行仅一个共同地址读表达式。scalar Q2仅使能选中rank银行，所有写都在互斥状态执行|
|source|原1536×10bit=1920B，16字局部窗，4个T10 gather；实际边界padding与source_allow保持原协议|
|Q1/Q2|原Q1 8×864×3bit=2592B、Q2 8×96×16bit=1536B；Q1同列权重跨全部P/T复用一次，Q2同128B局部块复用|
|psum|原8×480×32bit=15360B，完整480次向量写与480次读；原256bit输出holding|
|前缀保持|复用原8×32bit acc寄存器；prefix时存两字段endpoint前缀，merge后仍保留endpoint前缀，不把完整sum重复积入下一时刻|
|z holding|原8×26bit=26B，读与加写隔拍；不存在两域同时读的隐藏口|
|新增控制|四个10bit XOR端点掩码、两位置OR/两个20位popcount与比较、每字段负号、域选择、any_endpoint位、必要条件的相邻bit AND/OR、前缀/合并/lazy clear状态；这些逻辑共担，但未测面积/Fmax|
|固定T顺序|10×4bit=5B配置寄存器，首次使用mode4或5付一完整配置拍；源在L_GATHER用位选择mux重排，psum drain通过同40bit表组合查逆位置并在已有单读地址选原T。没有第二份40bit逆表或额外psum搬运，mux/索引面积与时序未测|

mode0、mode1和mode5只使用一个z域，清20行。mode2固定清40行。mode3/4先清20行，若确实选过端点再通过同一端口清20行，首个端点的k/source/pending保持；无端点时不读取或清理未用域。mode5仍处于同union RTL，不据此声称其裁剪后的单域面积与混合相同。

mode1/5前缀固定20次读＋20次共享ALU/写，随后Q2读该单域。混合模式若存在endpoint，前缀20次读＋20次ALU/写，另外直接域20次读＋20次ALU/写合并，共80个状态拍；所有费用包含在core cycles。每一行前缀写回endpoint域，随后单独读取直接域再相加，最终Q2只读完成后的直接域。mode0/5没有被强迫运行无用的clear/merge来制造候选收益。

每tile配置3361拍：1536source＋1origin＋864Q1＋96Q2＋864k_live。所有模式相同；连续warm命令复用配置。core timing从start后到done，包含源读取、计算、prefix/merge、完整psum、drain和外部source/weight/output背压。报告同时给出配置与核心的边界，不把CPU数学模型当性能结果。

mode4/5在上述费用外首次使用多1拍40bit排列配置，跨mode暖启若首次进入4或5也付这一拍；4和5之间已加载相同表时不重复配置。64流每tile真实装1536source＋1origin；第一tile另装1824静态Q1/Q2/k_live，mode4/5再加1排列；第二遍保留静态配置。总service另外包含每tile1个start。mode3/4的direct、endpoint域与prefix/merge服务完全相同，mode4只改变已加载门字的执行T位与最后原单psum读口地址。全部Q2完成后才drain，原来完整psum物化已付费。离线校准DP不在RTL服务中，当前不是在线逐帧自校准架构。

排列固定为[8,2,6,3,9,4,1,5,7,0]，是0..9双射。通用cfg入口没有合法排列检查，离线配置方须保证这一前置条件；本轮未支持重复或越界排列。两套64使用同一表，4000–4063不与校准源重叠，但仍是同帧验证。

令K为非零Q1列数、Q为活动源列数、U为实际成对位置更新数、M为Q2 MAC、C为清零行数、S为SELECT拍数、P和G为prefix与merge ADD数：

```
core_cycles = 5437 + K + 2Q + 3U + M + (C−20) + S + 2P + 2G
            + source_stalls + weight_stalls + output_stalls
z_vector_reads = U + 40 + P + G
z_scalar_reads = M
z_writes = C + U + P + G
first_issues = direct_issues + endpoint_issues
shared_ALU obligations = U + P + G + M
```

加法域为signed13，Q1合法signed3含−4。任意K部分端点和的绝对界≤864×4=3456；完整endpoint前缀等于其选中列的二值dot product，直接和endpoint域按互斥K分区合并，最终仍在signed13内。Q2保守绝对界8×3456×32768=905969664，signed32充足。没有中间RNE、FP消费者或PSN时间改变；本叶最终交付完整raw p。

这只是RTL逻辑服务合同，没有证明映射到公共SRAM宏后的单端口时延、面积、频率或能量。未EDA、训练、生产修改或commit。
