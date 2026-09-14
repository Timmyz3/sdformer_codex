2026-09-14：只在本子目录写。父 spatial_winograd 的 Q11 普通 m0／通用四 M Winograd m1 源码、wrapper、TB 和原结果锁定只读，不增加任何 controller 或调度。

分别加载 spatial_winograd_pruning/moment 与 native_tap 的冻结系数。moment 的192个组均满足 g1=g0+g2、U2=0；native_tap每组一个原tap为零。两者是不同模型函数，必须各自独立重算 raw/wide/I24；同一函数内 m0/m1 才可作等值对照，不能借母Q11输出或把跨函数周期差直接称等质量加速。全网质量由根独立运行。

沿用单context、8×32ALU／8×19×13mult、同source/W/Z/psum服务和完整FP32→J20→wide64→I24。通用m1仍有四M保持、完整四项D变换和固定恢复税；不会因U零组而免费删状态。m0也根据真实Q2向量支持跳零。静态Q2加载1152／1344拍，加consumer24拍，source/origin/start逐tile收费；所有模型表/额外holding容量沿用父资源合同。每函数独立配置，所有回收在I24完成后。

从父已回放的178个fixture只借source/origin/FP32 identity与同Q1的Z/D；按各自Q1/Q2及独立expanded W重算所有raw，另核F(2,3)完整/分stripe偶数重建，再重算J/wide/I24。原135真实tile与各函数export gold逐值比对；36跨序列只使用母capture的source/identity，不拿其raw/I24作新函数gold。相同输入文件可链接，输出gold实际另存。

先15small，再两64和18序列36tile；每函数两模式ready/BP、两遍无reset换源；raw与完整consumer均实跑。共享已锁定SV进行一次独立Verilator --cc --exe＋make构建，逐状态/端口/配置/周期由父独立profile验证。此实验只隔离结构投影与通用布局贡献，DA另立moment3，不在本目录新增第三核。无EDA、训练、生产、main.tex或Git。
