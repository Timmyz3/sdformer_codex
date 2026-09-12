# 固定两项有符号幂和：执行前计划

**A（借入底座）**：PoT/shift权重是公开方法；[DeepShift原论文](https://openaccess.thecvf.com/content/CVPR2021W/MAI/html/Elhoushi_DeepShift_Towards_Multiplication-Less_Neural_Networks_CVPRW_2021_paper.html)研究幂二权重及移位/符号操作。本次仅借“降低常量乘法复杂度”这一方向，不复现其训练。常量网络仍交给既有da4ml0.6.0完整CSE及相同last_use_pressure编译，再执行公共源RTL。

**B（强对照）**：当前同父320步端点dense与lifting40各自未修改的常量源；输入均为同两真实ordinary上游I24 halo，双方使用同编译器/ROM/96×8×48RF/流水/SR64-SW64/ready与stress。报告量化前后同结构变化，也保留原强普通dense、34、lifting源结果，不能只与旧复杂源比较。

固定投影：对每个signed16整数系数q，取集合 `S={0, ±2^a, ±2^a±2^b}∩[-32768,32767]` 中离q最近的整数（a,b=0..15），距离相同选绝对值较小者；无验证数据选择。该范围足以覆盖signed16域所有至多两项表示；更高指数的两项若相减仍在域内只能归为已有单项/零。分别只改新dense的`As_q16`和新lifting的`lifting_q12`。各自指数、cutoff、配对/排列、完整40次字面半步RNE与编译35次实际norm、所有下游常量保持；不训练、不扫位宽/项数/指数。

**X**：单纯两项量化及CSE不是独占X，只作为性能使能或普通强对照。系数/门改变即新函数；CPU门差、编译及RTL正确性与网络AEE分开，绝不继承父825。先生成LiteralForward可直接加载的完整两小NPZ及CPUgold，再执行公共RTL。只有实际成本降低的固定臂交算法代理，在当前825/主halo队列之后按授权做至多两臂diverse10。
