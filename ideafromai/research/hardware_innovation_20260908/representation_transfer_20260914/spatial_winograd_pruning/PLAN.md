# 固定 Q11 的一次结构剪枝导出

只读 spatial_winograd_inputs，独占本目录。只做两个冻结模型，不训练、不用活动/valid拟合、不扫比例，不跑GPU/RTL。

moment：192个(rank,N8输出组)各选一个 m1 或 m2。对组内每条三tap，求 signed11 对称范围[-1023,1023]内满足(1,±1,1)·g'=0的精确整数最近点；用同输出固定(BN_gain×output_scale)^2加权系数平方误差总和选组模式（平局m1）。目标为768个U向量中指定192个整N8零组；附带零组如实另计。native_tap：每个相同组清一个共同native tap，按同加权误差选，平局较小tap；目标删192/576原生tap组。两臂不是同稀疏百分比，不把25%U和1/3native直接比较为等工作量。

q1、theta、BN、a/b、output_scale全部保持Q11母模型；Q2变更后重导U、expandedW32和所有bounds。135真实输入只用于导出各自factor/Winograd/direct同函数gold和事后局部误差，不能回流参与选择。每臂保留原输入/identity/J，输出新p/wide/I24及M；证明各rankstripe重建偶数、无中间RNE。网络diverse10/valid825由根单独测，不继承母模型或Q13质量。

此项是空间/Winograd约束结构剪枝的有界移植，不声称新WINS；系数加权投影是普通误差控制，不把下一篇创新预支给本次结果。
