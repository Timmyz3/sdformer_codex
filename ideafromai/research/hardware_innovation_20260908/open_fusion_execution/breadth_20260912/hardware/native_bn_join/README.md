# 完整native→onepass BN→PED后段已闭合

**完整一帧、同一Engine完成了此前缺失的后段义务。** 使用既有ordinary R24＋onepass组合的实际整域projection gate、连续PED和真实权重作为外部起点；Engine先生成全部native raw，再自己计算全域统计和normalize＋PED输出。没有读取捕获统计充当计算，也没有相加旧Engine费用表。

| 同一个实际native＋统计前缀 | BN物化再读PED相加 | 普通normalize＋PED融合 |
|---|---:|---:|
| 总服务槽 | 533,087,251 | 457,055,251 |
| 相对前者服务减少 | — | 14.2626% |
| 外部读取字节 | 332,437,568 | 258,709,568 |
| 外部写出字节 | 221,184,000 | 147,456,000 |
| SR64实际读取字节 | 624,685,176 | 477,229,176 |
| SW64实际写入字节 | 581,139,576 | 433,683,576 |
| CR256实际读取字节 | 426,912,512 | 426,912,512 |

两臂的完整Machine前缀实际跑到305,375,251槽，然后复制RF、SRAM、系数、时刻、待写回和所有端口状态，各自真实执行后缀。14.26%来自消除中间BN的73,728,000B写出与同量重读，属于普通BNFF式融合A，**X=0**。这不是新增机制加速或RTL测量。

数值域为T10、C96、120×160，即192,000个96维向量。完整native的18,432,000值与独立逐输出位置/时间/原K升序的`fmaf(1,W,acc)`参考逐位相同；实际计算的mean/variance/rsqrt/scale/bias共480值与无调度算术companion逐位相同。规范化中间量和两臂最终结果各18,432,000值也逐位一致。native实际W先执行与既有控制相同的TF32舍入，且该实际层theta=1、无bias；因此现有FP32 ADD即可实现保留项的相同算术，不增加FMA/gather ISA。

对原GPU卷积raw仍有36,729个FP32位模式不同，最大绝对差3.06368e−5。经过重新计算的全域BN，与捕获onepass输出有1,746,467个值位模式不同；相同PED相加后有1,045,039个不同，最大绝对差均为2.0599365e−4。全域统计会传播局部native差异，不能把它们说成最终影响为零。完整输入和参数属于已评组合，**新native浮点执行函数不等于其GPU卷积归约**，不能继承该组合825质量。这里只报告函数参考和接口闭合，无新AEE。

实际数据路径和费用如下：

- 三个H32系数tile各110,592B，各自在128KiB coefficient内遍历完整帧。固定4×4输出块先实载9×9×C96的T10门字，边界填零也写SRAM。H4的18个SR64字装入9个现有RF后，按原c-major顺序生成完整P2/K864目录。三个权重tile共支付55,624,896B门重读、24,883,200次目录decode、3,335,247个NRV写出；没有隐藏跨tile目录保存。
- native占256,663,590槽，其中系数冷填62,208、门加载15,981,390、目录84,234,447、计算121,825,545、raw写出34,560,000。16,071,288次8lane有效ADD都执行真实载荷。raw按T-major目标地址逐次收费写到外部73,728,000B缓冲。
- 此后仍在同Engine上支付73,728,000B raw读回计算双moment，再读同量raw作规范化。两stripe和paired256树保留原T-major顺序；variance为`E[x²]−mean²`，原seed/三Newton及分开的MUL、ADD保留。native没有偷换统计次序来省缓冲。
- 实际PED输入为spatial/T10/C96 signed24/f14，共55,296,000B；每个位置支付地址、32B/5槽DMA、SRAM存取、24bit解包、转换和f14缩放，最后FP32 ADD并完整写出73,728,000B。

片上仍为96×8 RF、128KiB state＋128KiB coefficient、SR64/SW64/CR256。native目录在0..55295，头在56000..56063，门在65536..81087，state最高占址81,088B；输出暂存在已消费目录的32..63，只有读完该目录后才覆盖。目录阶段用RF64..72存H4门，后续native累加才占RF0..79，权重用RF80，阶段不重叠。BN树16384..24063，后续PED暂存24576..24863；它们在native结束后复用同一state。73.7MB raw、55.3MB PED以及可选BN中间量明确属于有偿外部缓冲，没有被装进128KiB。

此版本是完整、可再优化的普通分母，尚未补跨tile目录保留、source-code0压缩raw或相应BN默认传播。尤其目录已占融合后总服务约18.4%，这是具体待优化接口，不能将其直接命名为新X。外部传输继续沿用既有32B/5槽合同，未建DDR行时序或片外能量模型。输入门/PED自身的生产费用仍在组件边界之外；本轮没有闭合完整I24到整帧或整网。

计划与边界：[PLAN.md](PLAN.md)。执行器：[native.hpp](native.hpp)、[pipeline.cpp](pipeline.cpp)、[run.py](run.py)。原始表：[results_ready.json](results_ready.json)。只读旧stage/生产目录，无训练、EDA或RTL/PPA结论。

复跑输入为已有完整捕获，须先从A800只读取回`/root/private_data/work/hardware_innovation_20260908/open_fusion_execution/stage_20260912/algorithm/captures/ordinary/000_zurich_city_09_a_0001.npz`，再运行`python3.12 run.py --capture /实际下载路径/ordinary.npz`。本轮保存在`/tmp/native_bn_join_20260912/capture/ordinary.npz`；该临时输入不是仓库内可再生工件，runner不会训练或自动新采集。小型权重/BN参数读取仓库既有`stage_20260912/algorithm/hardware_exports/ordinary/live_parameters.npz`。
