# 结构T10源进入真实双消费者：本阶段B/A/X

B：可学习lifting40已得到原固定学生的完整825帧质量，然而此前源服务、mask生产者与整数消费者常被分开计费。一个源程序少节点/省拍，不代表经过完整K864、非因果sn2、raw I24残差以及连续PED后仍有净服务。必须让真实门字在同SR64/SW64状态里交接，并让ordinary享有相同编译和供数能力。

A：复用共同的da4ml式常量CSE/寄存器分配、两阶段int48源ALU、普通Gustav式完整K稀疏供数、共享SR64四通道coalescer和resident MAC。这些是底座，不分别构成X；不会将地址开关或少写一次中间门图单独包装成机制。

候选X：可学习lifting时间结构在真实两消费者义务下仍减少净服务。对照先固定ordinary vs lifting_raw、corner/interior四窗；两者从各自真实I24起，以各自固定参数执行全部source load/addsub/norm24(RNE+饱和)/gate，再执行完整K864 preview U32/V+BN1/noncausal sn2、完整K864 Conv2 U16/F+BN2/raw I24、projection gate与连续PED U32/V96。门缓存原址交接，无外部门输出再输入；raw I24后继重读仍实价。前级和后级coefficient replacement实际发生且收费，不假装同时驻留。

先完成共同R32；R24和global/phase/水平P2是随后固定消融，不抢生产接口。空间范围为原四个捕获窗口及其真实halo，完整K864但不是整帧，native projection/global动态BN不在本阶段模型范围。质量来自原固定参数已有评价，本轮不训练、不创造AEE。两学生功能各自与原捕获逐值核对，不将它们不同输出声称成等价。

物理预算：一台96×8 RF(48bit lane)、128KiB state、128KiB coefficient，1R64/1W64 state、1R256 coefficient、同32B/5slot外部DMA。两源各有相同8192B静态指令ROM预算。coalescer使用已有RF且与源/消费者错峰复用；所有边界/地址/冷填/元数据/写出及背压收费。先ready四窗，再一个固定压力例。

隔离RTL目标：共同8lane int48两阶段写回源执行器，96RF，load/addsub/norm24/gate/commit指令，两种真实编译程序皆可执行；从真实capture I24生成真实T10门字，在输出ready背压下不丢失/重复提交。RTL验证与上述CPU服务分开报告，不将单叶Verilator视为完整链RTL或PPA。生产nts07、主稿、H81和docs359只读。
