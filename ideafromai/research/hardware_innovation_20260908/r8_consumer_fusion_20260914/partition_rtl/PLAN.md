# 固定 R8 signed magnitude partition：先声明合同

2026-09-14。B是窄整数R8已经生成完整z，但后因子仍逐非零rank更新输出。当前旧八块1463个非零z，可形成1015个每(P,T)绝对值组；这是代数项数，不是周期预测。该端点此前未实装。

mode12强A：全K864原生源→Q1→完整z；最终z扫描生成rank/位置活性；每N8预取整个R8×N8 Q2块，psum驻留寄存器，逐有效rank一拍MAC。该分母比旧反复读Q2的mode8更强。mode13在同底座上付一次完整z partition生成，跨全N重用描述符，使用固定8项signed支持缓存，epoch按N8清空，roundrobin替换。禁止无限模式表。

组a选择最低rank leader，令v=z_leader、s_r=sign(z_r)sign(v)，则贡献v·Σs_rQ2[:,r]；leader符号归一为+1，使最坏系数界[-262144,262143]，signed19充分。若某Q2整向量为零，删除该项不增大此界。共同八个19×13乘法器、八条32位加减链、128B Q2块缓存、同最大latent/descriptor/8项cache/psum状态。mode12亦获得全部资源权限，不以分别裁剪面积相等作声明。

first-stage严格保留旧完整Q1逐活动源更新，不让TB输入latent。无中间RNE，gold为旧Q2@Q1 raw p。全14旧fixtures，2 modes×2背压×2不reset命令；数据/地址保持及所有3840输出检查。静态k_live、Q2整向量活性与最终rank零同权。descriptor最多320×29bit，start/count 40×(9+4)bit；8项cache保存有效、16bit支持tag、8×19bit系数。所有前缀生成、abs、系数组和、cache miss/epoch/最终物化与drain纳入实测。

UCNN的重复值分组、Phi pattern-weight缓存及常规驻留均属于A；本次完整小核允许失败，不事先立X。还没有消费者/训练/825/PPA/Fmax或全帧结论。只在本目录实施。
