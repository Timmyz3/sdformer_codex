# 至少十条新增融合的执行批次

2026-09-14。用户明确要求本轮至少十条完成后再作阶段汇报，或提前形成强创新与性能共同证据。**上轮五项、重复背压、补测和同布局参数扫描不计入本轮十条。** 本文件先写问题/控制，结果以后用真实收据填入。

共同对象优先为已经有真实I24和825的R8完整r0：C96/N96/K864/T10。普通native source、最终支持、cachedOS/dualP与真实FP32残差均是强A。有损函数须各自评价，当前基线门为同A800原NB0 valid825 1.447936665574317；不得继承旧函数精度。生产与上轮提交只读。

| 数量 | 接口/负责目录 | 本网剩余问题与必须给予的强控制 |
|---|---|---|
| 1 | decompositions/Q1三位平面与bitmap-popcount | Q1仍逐活动源更新；完整源→bitmap布局、宽popcount与存储口同预算，不能免费转置 |
| 2 | decompositions/Q2分组distributed arithmetic | 连续z仍做大量Q2乘积；LUT装入/读取、符号和移位全收费，完整R8缓存MAC为控制 |
| 3 | decompositions/Q1重复列sum-first | 同Q1向量的g能先归约；组状态/退休与完整K864必须实现，对照不重复取已驻留的W |
| 4 | algorithm_sparse/完成z整组剪枝 | 源稀疏未必删除全部后因子义务；普通逐rank剪枝同精度控制，完整consumer质量 |
| 5 | algorithm_sparse/latent原型＋有限修正 | 连续latent的重复近似可减少后因子服务；真正的encoder/表/残差，不用TB原型索引 |
| 6 | algorithm_sparse/T10整latent分段保持 | 完整T10中近似相似向量仍重复Q2；整组保持/刷新必须与逐rank deadband比较，identity/BN/I24每T仍执行 |
| 7 | dataflow/输出直接完成 | 完整p STORE/DRAIN仍收费；直接送I24与物化控制共享holding/背压权限 |
| 8 | dataflow/跨tile halo轮转 | 相邻tile仍重新装重复输入；容量/主读口相同，地址轮转与行边界全由RTL完成 |
| 9 | dataflow/双tile共享算术交错 | Q1选择/读拍有空闲ALU，Q2可填；双context同资源，单套八ALU/MAC，真实端口req/grant仲裁 |
| 10 | phase_borrow/借用I24宽加法链 | Q1时消费者宽adder空闲；借作四P窄累加，后续恢复真实64位仿射，不能另复制宽adder |

每项先PLAN、后实际RTL/完整gold/周期，再作新颖性判断。每个单独资源点都明确端口、状态及算术权限；新增资源可以用，但对照也得到相同预算。不用普通先验存在阻止动手，也不把普通模块组合直接称新贡献。负结果只停止已测布局。

使用 brainstorming-research-ideas 的问题/边界/组合框架。三个原有代理分工实施，root负责第十项及交叉审阅/整合。前三组不共享写入目录；GPU由algorithm_sparse代理独占。此表是任务，不是十项已经完成的声明。
