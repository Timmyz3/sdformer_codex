# 大卷积整数因子：固定 R8，实际原生供数与两种收缩顺序

B：9月13日的无残差 SVD8 在真实网络上通过 NB0，但只有浮点质量及算术费用；窄第一因子如何喂连续第二因子、后者 W 复用与 psum 物化相互冲突，尚无完整 RTL 结果。过去 selected-pair 是每输出的局部等权近似，并非本条共享 R8 因子。

A：普通 SVD、小整数第一因子、尺度吸入后因子、常规两级收缩、weight/output stationary 都先归入已知底座。延续 deep_target_research_20260913 的 D1，而不是新命名一张卡。UCNN/SmartExchange/StrassenNets 限制新颖性；此阶段先补实际端点，不预填 X。

固定数值：已存 flat_svd_r8_w32；Q1[r,k]=RNE(U[r,k]/s1[r])∈[-3,3]，s1=maxabs(U)/3；C=V*s1；Q2[o,r]=RNE(C/s2[o])∈[-32767,32767]，s2=maxabs(C)/32767。z=ΣQ1*g（signed13）；p=ΣQ2*z（signed32）；输出意义为 p*s2。无中间舍入；真实后继照常执行，新增diverse10独立评价。量化布局只试这一点，不从质量结果调码宽。

硬件同函数控制：expanded Wint=Q2@Q1，需signed21而非原W16；完整K864/C96/N96/T10原生4×4→2×2。延续最快pair-parent direct为mode6；mode7两因子、V按r/输出组驻留跨T/P；mode8完整latent后按输出块局部归约，减少psum读写。三者同一SV、同源端口、共同常量/状态预算、八条数据加法链；mode7/8需要乘法器，mode6亦保留同资源，不能与原无乘法叶直接报同面积收益。

每候选至少真实旧八块＋零/全一/边界/符号控制，两种背压、两次不reset重启；完整raw输出与Python int64参考一致。source gather、padding、Q1/Q2/expanded系数读、z读写、psum清零/读写、drain均由RTL执行/计费。最终scale及BN/残差不在本叶周期之内，不报完整r0 block或ASIC PPA。品质桥与raw核分别明确，不能将旧SVD AEE挪给新Q1。

潜在 X 只有在共享窄源表示后仍出现普通同权限调度解决不了的义务或物理事务，且实际性能/质量支持时才成立。单纯因子化提速是A；mode8对mode7若赢，也先是更强融合A。根负责本目录，另外两代理分别做连续tile执行和完整捕获/质量，避免互相改文件。
