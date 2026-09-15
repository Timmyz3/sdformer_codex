# 固定 Prosperity 父图 × Phi 精确求值，2026-09-15

本轮确实实现并运行了八臂 RTL，再依据查询税做一次同权控制适配；同时完成 Claude 三份新交接的代码级审阅。

**结果：当前窄 N8、冷 K16 条带布局未胜原森林。** 在同一小输出 tile 的全部 K864（54条带）上，原森林为6340拍，按需控制后的森林＋Phi为6644拍，慢4.795%；有背压时6829→7403，慢8.405%。这是Verilator叶周期，未接跨K硬件累加与I24，不是完整层或准入RTL加速比。保留原森林底座，不把这次负结果外推成Phi/Prosperity家族失败。

- [执行结果、费用与去留](RESULTS.md)：八臂、两套查询控制、两种背压、4128次任务，1,320,960个整数结果逐项零差。
- [先行问题与执行合同](EXPERIMENT.md)、[原始捕获与校准/留出边界](CASES.md)。
- [Claude审阅](CLAUDE_REVIEW.md)：保留局部门核和真实分桶计数；反驳4.2拍下界、全库穷尽、T6包络闭合三个过度结论。
- [独立RTL审阅](RTL_REVIEW.md)、[逐任务结果](results.tsv)、[汇总](summary.json)。

运行：`/opt/anaconda3/bin/python3.12 prepare_cases.py`，随后同解释器执行`run.py`。需要已有相对路径中的真实R8 fixture、NumPy、Verilator4.028和C++工具链。`run.py`重生码本/激励、编译RTL并生成逐任务结果。二进制输入与构建物忽略，不复制进Git。Claude数学反例另运行`check_review_math.py`。

没有训练、量化变化、生产RTL修改、论文修改或EDA。借入算法与当前新颖性仍需分别评价；本轮没有形成“强接收已成立”的证据。
