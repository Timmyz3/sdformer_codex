# 同一个两项dense函数的固定低RF普通控制

仅把上级已生成的`dense/deployed_constants.npz`交给既有`hardware/dense_low_state_source/run.py`的`compile_fixed()`。保留逐行两条CSD累加链、RF0..9输入/10..11链/12行和/95门、所有依赖NOP和原阈值前像；不改生成器、不换策略、不扫参。原未量化dense的524字拒绝结果保留。

这是同函数编译控制，无新参数或新AEE臂。先验证完整矩阵积/门、596组原RF标签和readiness，再仅在完整程序≤512字时运行同公共RTL的两窗ready/stress。分母为上级两项dense的完整CSE144字/319周期；13工作RF并非免费收益，应报告新增执行费用。目的只在检验该函数是否能用13工作RF＋门RF装进原ROM，为未来同权交织提供普通控制；本次不执行交织。
