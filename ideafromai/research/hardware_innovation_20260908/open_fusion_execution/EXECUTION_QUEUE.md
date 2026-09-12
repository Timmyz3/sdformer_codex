# 家族执行队列：先试适配接口，不等待证据完备

2026-09-12。完整逐文献去留/未试项在[catalog/works.csv](catalog/works.csv)；完整候选视图在[catalog/idea_views.csv](catalog/idea_views.csv)。下表是这些材料通向实验的入口，不是把305个视图都升级成独立硬件模块。最新完成了同Machine源至局部双消费者、公共源RTL、R24＋onepass两组合真实825及11个附加小集配置；下列保留接口不代表全部在后台运行。

**当前精度按[新规则](ACCURACY_POLICY.md)：候选优于同口径原SDformerFlow本地复现即可继续考虑，取消+0.005淘汰门。** 同diverse10的NB0为1.454603；lifting/剪枝/单遍BN因此恢复比较资格。原学生差值用于权衡，不能再作为精度禁令。性能负结果和最近邻重叠仍分别保留。

最新结果见[扩大执行阶段](breadth_20260912/README.md)：同父dense/34/lifting匹配训练和新825完成，AEE为1.208330/1.421461/1.225495；源RTL491/303/446周期。普通affine/diagonal/full-D的十帧、CR/RF两种查表、真实W4/W8执行、NRV共享索引、固定F_live2均已实试；不能继续把这些完整列为“尚未开工”。另完成两项源常量、同函数低RF普通控制、native全域BN后段、两块目录复用及W8全驻留。原始结果与动态GPU端点收口见最新目录；下方旧数值是对应历史端点，不可移给新训练函数。

下一批优先真实96RF源/消费者交织、当前学生完整I24→native/BN链、因果帧间完整尾部；普通低状态dense、普通34、普通量化和分块都必须同权。其他未试接口逐一列在[剩余批次](breadth_20260912/coverage_owned/REMAINING_BATCHES.md)。没有把305个视图全部做成实验，没有用旧负布局判整个家族失败；MiLo/ReverB联合训练及网络替换保持后续队列。

| 家族/可借底座 | 网内缺口/拟融合X | 已试与停的具体版本 | 保留的下一执行接口 |
|---|---|---|---|
| HiNM/FlexHiNM、DepGraph、Bishop、CGNet | 后继stride/H8供数约束下免去整条T10生产，真实gate+PED共同评价 | phase/H8恢复后AEE1.201610，行相位两学生1.217386/1.247094，均优于NB0；请求前地址接口已实测，当前phase比强化global慢0.327–0.658%，同mask端口字节相同 | 保留源结构/精度权衡；地址开关归公共底座。已经接lifting至完整局部消费者；下一比較补共同量化/训练与native后段；静态偏移表未试但共同授予，普通全局/减宽/2:4保持同权限 |
| LoAS、BBS/BitVert | 真实连续PED的时间位字而非又一份二值共享 | 新逐位默认/校正比强MAC慢4–5倍，停该串行布局 | 更便宜bit-PE的面积点或形成低位/低校正密度的训练；不免费假设硬件 |
| 普通驻RF供数/Gustav式局部状态、Vecim | 多H8反复读同一I24；这是执行底座遗漏 | 已接两边完整消费者，并重载恢复后U/F执行；完整消费者约省15–16% | 固定容量内load/MAC重叠与完整生产链；驻留、块MVM和队列不作独占新标题 |
| Prosperity、Phi、Transitive Array、SPIDER式包含关系 | 动态重复/子集与有限父值完成 | 已补全54个K16和允许重叠TCAM：源项少34–47%，H8局部/跨K税使服务比GP慢78–83%；列关系仅比TCAM省约0.1% | 停M20/K16/H8布局。保留不同跨K状态/生产接口；不扫描当前参数，不将负结果外推完整PR |
| 全支持等价、Finch默认表达、BNFF、FlexAcc | 完整结果等价能否穿过动态BN保留代码表示 | 单遍统计+真实PED已同Engine闭该后段，普通融合少35–39%服务；三种BN函数六臂AEE均优于NB0；第三default臂X=0 | R24＋onepass两学生组合已各跑825，1.211716/1.232391；下一接完整native生产，不能将后段旧服务表加到当前局部时间线 |
| Maestro LoD、FLRC、SVD-LLM；LoopTree/TwinQuant/LowRank-SSM | 连续PED不能只因gate完成而停；原潜空间缺乏可截断性 | 两学生八个表示AEE均优于同十帧/九帧NB0；同驻RF双方同权限后R24对R32端点少11.4–14.7%；固定门字D残差也实试，仍全密并多算 | ordinary原R24及lifting白化组合已完成825；不申领blocking为X。Dg＋Q8残差新函数已过真实小集，整字空率为0；下一补普通affine/diagonal控制和付费执行 |
| da4ml/CSE、可学习lifting、结构化PSN | 非因果T10真实连续算术与二值/连续两个消费者 | 当前lifting R24＋onepass真实825为1.232391；源RTL少17.505%周期、局部CPU少7.954–8.127%服务。普通source34更快但免训AEE1.686686失败 | 同预算训练普通结构源与lifting比较；接native/globalBN完整后段；共同96RF内检验结构小活跃集，不把节点数当服务或把8%全归ALU |
| Gustav CPTB/NRV、ELSA、HYTE | 供数、分块、局部状态、有限psums | 时间类别/NR4旧开关无净增量；不是Gustav整个底座无效 | 实际来源∩W与物理组跳过、参数固定排列；完整元数据/重叠控制继续补 |
| VENOM/HighLight/CRISP、QP-SNN | patch/S2仍贵，权重侧缺真正结构零 | 原FP32无精确零块，不能凭稀疏愿望跳；旧S2掩码太不敏感 | 敏感patch的联合权重/激活训练，与普通减宽、2:4和实际任务损失比较 |
| SparseInfer/BitFair、FGIE、Precision Gating | 只要门有消费者时，早判是否能取消整个共同请求 | 粗L∞证书0命中；普通末成员早停净约1% | 有损可训练整组预测或更细可组合证书，必须付完整误预测/继续费用 |
| Avalanche、RISCSparse、HyMM、LoopTree | 双消费者不同完成边界造成状态驻留/阻塞 | 改last-use名词不删真实首读；generic defer-V净增量小 | 图/布局改变依赖寿命而非只加计数器，真实producer→consumer时间线 |
| Motion-XOR、α-XNOR、FireFly-T | 非传统QK算术的首份针对性数字映射与跳输出乘积 | 叶小/系统份额未闭；整窗memo脏率高 | 行内memo、三pop分项、K零只跳输出；用ep34/实际学生身份单独计，不与主岛倍率相乘 |
| PSN短时间阵、DeepShift/PoT | 连续T10与T2算术本身有结构 | plain快变换/移位常量已是先验；固定±草图失败不杀可学习版本 | 训练源结构与真实舍入边界同编译、同端口比较 |
| DeltaCNN/MotionDeltaCNN/CBinfer、S3Net、稀疏stem | patch大头与事件帧间局部变化 | 已试同推理T10本块前T位移参考，完整K有符号校正0数值差；免费任意邻居源项上界不足2%，停止该接口 | 已有motion/motion_probe.py对4个实际相邻推理帧、3个pair做过因果运动机会探针；未闭完整数值/端口执行与新学生，跨tile/训练支持改变另列。稀疏stem仍需训。不能把T10切片当视频帧、普通运动对齐当X |
| OP-STW、二维运动见证、ERAFT式运动先验 | 光流特有的因果计算选择 | Grok比较器用TB/最终flow喂先验不能作机制；空间局部性已有先验 | 只用事件或上一帧/TDE构建因果wake，训练后评价；不是当前最廉价并行任务 |
| ReverB、MiLo；普通W4/W8 | 连续PED权重是否能更简单，且保留真实舍入/消费者 | sign＋rank0/4/8/16免训布局被普通低位压过；W4/W8四臂真实小集均过NB0 | 先把普通低位code/scale/RNE接同执行器；联合训练保留第二队列，不从当前未训失败杀家族 |
| ZipServ、Atalanta、Shannonic | 普通融合后仍需搬运的连续值 | 真实全域块费用和字节往返已做；I24复杂codec仅比signed-width再省约0.1%，已消失的BN输出事务无可压缩增量 | 当前接口停止标题，其他必需spill仅在真实布局存在时重开 |
| HBG双轨、幅度/门分型 | 连续PED/膜与二值发放要分工 | 把AT-LIF当任意int8逐事件幅度的旧前提已撤销 | 双轨思想改挂连续PSN/PED；二值g侧继续吸收θ，不能偷换检查点身份 |

是否“借全”按每篇工作公开范围逐项说明：算法、格式、匹配、供数、状态、训练与硬件实现分别列。当前局部移植不会自动升级为完整原作复现。**不用完整证据挡住第一次代码尝试；也不用一次小草图失败抹去尚未试过的接口。**
