# 三结构共同父、共同预算

**B与强对照。** 旧lifting40和ordinary各有64＋256步GT恢复，旧contiguous34仅是免训遮罩。新比较统一从当前ordinary原排列R24＋onepass父开始，共享其旧320步历史，不载入已恢复lifting作第三臂。比较dense100、连续3/3/4结构34系数、lifting40，消费者权限、训练帧顺序、seed和步数相同。

**A及缺项。** 复用原模型、冻结preview、固定RNE辅助函数、旧train16＋train128顺序、真实GT损失、八个lifting半步和单遍BN。旧fast训练是FP32；现有paired-QAT只覆盖Conv2 U/F。因此必须补source/half-step、真实preview输入与单遍BN的梯度代理，同时保留实际硬前向，不能直接称为已有source QAT。

**固定预算。** 三臂同获旧train4输入矩的1024步初始化拟合，再用原64＋256顺序、seed912、每阶段fresh Adam1e-4做320步GT恢复；不挑checkpoint、不用验证拟合。全网其余参数和θ冻结。可更新项统一以已编译函数定义：源系数、两组cutoff、Conv2 U/F、PED U24/V24、BN2常量与PED偏置。系数网格、每处RNE/sat24、signed48累加均保留。cutoff作为bias/readout的等价部署参数，不改变AT-LIF的{0,θ}身份。普通结构同样开放固定输出排列的编译权限，不能由lifting独享。

**候选X与交付。** 训练结构是否改变真实生产/消费依赖寿命尚待测；lifting/CSE/稀疏本身不作标题。先各做真实TRAIN帧的硬前向与部署一致、源和消费者梯度smoke，再实际训练。最后重新加载部署常量评diverse10与holdout9；完整825先与根代理同步选一个新学生，不自动三臂全跑。输出完整参数、活动、共同父及累计预算。只写本目录，无生产/EDA/main.tex修改，不建hash。

并行表示研究的原两父TRAIN捕获已完成：thun_00_a_0012.npy，固定64×64 anchor网格4096位置、完整T10×C96实际updatedI24与projection_g。脚本capture_train_parents.py核两父部署常量与上一阶段逐项一致；捕获不替代三结构训练数据。
