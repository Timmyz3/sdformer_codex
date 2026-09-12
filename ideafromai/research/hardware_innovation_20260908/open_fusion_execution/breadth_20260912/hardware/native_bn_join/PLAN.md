# 实际闭合native→全域BN→PED后段

B：已有完整局部I24/双消费者、native窗口、全域BN和join分别有程序，但未由同一个Engine生成整帧raw再完成BN统计；73.7MB raw与55.3MB PED不能藏在128KiB里。此前服务不可相加。

A：沿用原FP32 Engine、96×8 RF、128KiB state/coef、SR64/SW64/CR256与32B/5槽外部传输。完整K864 native使用真实TF32舍入权重和升序K；gate为1、theta=1时已有ADD与fmaf(1,w,acc)等价。onepass严格保留T-major、两stripe与paired256树、seed/三Newton、MUL后ADD。最后沿用实际signed24/f14 PED读取与ADD。

固定外部起点是实际ordinary R24+onepass的一整帧projection-g及PED，非本轮模型免费生成。三个H32系数tile各110592B；每个tile遍历全帧，按4×4输出块加载9×9×C96门，构造8个付费完整P2目录。H4的18个门字放9个RF，保留c-major NRV顺序。目录0..55295，头56000..56063，门65536..81087。native raw实际写外部T-major缓冲，之后同Engine重新收费读取统计与规范化；不把主机数组算片上容量。

强对照为同一个实际native＋统计执行前缀之后的两臂：普通BN物化后单独PED相加；普通normalize＋PED融合。完整Machine状态分叉，仍真实执行每个后级，不加历史表。本轮先用dense raw格式与普通dense onepass，不含跨tile目录保留/code0压缩优化；它是可执行分母，未宣称最强性能或新X。

数值参考独立从真实门/W进行升序K native运算，再用无时序的既有onepass算术companion重建统计/规范化和真实PED相加。不得声称等于GPU卷积归约或继承825精度。完整192000×96域，固定ready；本轮不扩ISA、不扫布局、不跑训练/EDA。
