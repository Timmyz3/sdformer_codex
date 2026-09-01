# M1763 source-author receipt

结论：**PASS source-only；可进入 M1764 不同作者审阅，但不授权分析。**

M1762 已证明 M1756 的唯一分析尝试在异构 FC1 层间共享 S2 witness 数组时因 `G16=6→12→24→48` 发生 shape drift；该失败未发布 result/work，M1707 capture 仍有效，而且 TSBG 算法未被牵连。

M1763 只修 diagnostic witness 的身份与聚合顺序：`(epsilon, scope_type, scope, layer_id)` 内 OR drop/keep，各层先乘自己的真实 `output_blocks=24/48/96/192`，之后才聚合到 sequence/all。禁止 padding 后跨层 OR。TSBG pair math/finalize 与 S2 keep/drop、decision payload/hash 都维持 exact M1747 语义。

带 NumPy 的 Python 3.10 与 3.6 各 9/9 tests PASS；Python 3.12 的 4 个无 NumPy 静态/授权测试 PASS、5 个 NumPy fixture 明确 SKIP。Python 3.12/3.6 双解释器 source self-check 2/2 PASS。没有 capture verify、analysis、result write、GPU、EDA、network，也未创建 M1764/M1765。
