# Moment3：根据固定四项税失败的有界适配

2026-09-14，先合同后RTL，只写本目录。A是约束FIR空间分解与普通快卷积；不声称新恒等式。B为先前跨18序列general Winograd每tile2832固定税几乎吃掉MAC减少，净收益仅0.268%。现在只采用根已冻结的 `spatial_winograd_pruning/moment` 模型，不改剪枝比例、不重训；明确这是新的投影函数，不能借旧Q11 gold或AEE。

逐系数检查 `g1=g0+g2`（192个N8/rank组，1536个scalar三元组）。实际使用 `D=[z0-z2,z1+z2,z1-z3]`，每rank三项 `M=[g0D0,g1D1,g2D2]`，归约后 `y0=M0+M1,y1=M1-M2`。Q2只保存原3tap系数，无general变换系数副本、无第四M/第四D、无/2。保持Q1 signed8、Z signed15、D signed16、Q2原signed13硬件容器、P signed32及原I24合同，重新静态验证M前缀/重建/consumer wide界。

从已锁定general Winograd RTL复制需要修改的core/TB/wrapper到本目录。两R8 stripe，每stripe原1280B单端口banked Z原地变换：先实读两行保留4个Z，再同8ALU做3次加减，写D0/D1一行、D2/零一行；最后unused高半为零。Q2cache缩32→24词/输出lane，Q2表缩768→576向量；3个M accumulator共96B，读取原Z保持、transform_tail、所有额外holding逐项记费。唯一8×32ALU与8个19×13乘法器，不复制算术。输出逆变换只两拍，第二stripe psum实读后同ALU累加，所有480行最后按原序退休。

完整raw+原FP32 identity→J20→wide64→I24连线，与ordinary/general控制共用消费者资源/端口/背压。先small真实/角落/BP/两遍无reset，再两64；36跨序列只复用r0之前不变的actual source/identity，使用moment系数CPU重算全部新gold，明确尚不是该新模型36序列完整网络帧capture。即便网络质量失败也至少完成真实RTL评价。所有源/权重/变换/缓存/psum/配置/启动/末输出费用由实际仿真及独立profile核验，无EDA/GPU/训练/生产/Git。
