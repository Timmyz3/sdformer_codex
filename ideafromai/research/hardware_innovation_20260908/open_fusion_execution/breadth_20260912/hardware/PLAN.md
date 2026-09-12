# 压缩权重真实供数：先补完整强底座

B：上阶段 W4/W8 已有真实十帧 AEE，但 GPU helper 离线展开 q16，尚未实现压缩权重端口。少字节可能被解码、尺度与临时态的成本抵消，不能用其静态体积写硬件提升。当前真实 W4/W8 只作用于 PED U32×96；V96×32、bias及原U e16/V e15舍入均不变。它们的父臂是各自原R32/CUDA学生，不继承R24+onepass825。

A与强对照：完整既有source→preview/sn2→K864/BN2/rawI24→gate/PED同Machine底座。每窗真实执行公共前级一次，再复制全部Machine端口/状态/时间供独立分支，不相加旧表。分支固定为original32、W8-expanded16、W8-packed+row-scale、W4-expanded16、W4-packed+row-scale；expanded对照享相同源驻留/零旁路/强目录和量化参数。比较低位码接口时必须对同一函数的expanded臂。

实际接口：U按k/H顺序存signed8或成对signed4码，32个q16行尺度独立存放；所有头/尺度/code通过相同32B/5slot冷填和CR256读取。解包先写RF84，再经一次付费RF读放入既有64B staging中的16B权重暂存；源gather最多24B，因此共同64B预算不增加。MAC读acc+源两RF，权重来自付费暂存，不能偷第三RF口。RF80..83持4组尺度；完成时将code点积拆signed24 low/high，使用共同16×24乘法器两次乘法和48bit移位合并，再执行原U RNE/sat。每一步与真实载荷、等待、暂存边界一起收费，不假定免费宽乘法器。

X：本轮不将低位量化或普通解包称为新意。这是ReverB/MiLo类后续融合必须面对的可运行强底座。只有该表示真实减少共同资源服务才保留这个执行布局；若只省字节而更慢，保留低位参数/质量，明确停止当前解码/尺度布局，未试接口继续列明。

只固定两个学生×corner/interior ready；ordinary/interior最多一组压力。不训练、不EDA、不改旧stage/生产树。之后检查native projection→全域onepassBN→join的现有真实接口，若缺全域生产回放就写清缺口，不相加历史费用。
