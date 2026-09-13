# r0 窄整数 R8 因子：完整线性叶 RTL

2026-09-14。新增 [integer_factor.sv](integer_factor.sv)，并完成完整 K864/C96/N96/T10、原生 4×4→2×2 的三模式比较。168 条命令、645120 个 raw 输出通过；新 Q1/Q2 链独立 diverse10 AEE **1.353326**。保留 V 驻留模式作为分解底座，尚未建立标题级新机制。

- [结果与边界](REPORT.md) · [资源](resource_contract.json) · [结果汇总](SUMMARY.json) · [完整记录](results.json)
- [实施前问题与对照](PLAN.md) · [数值定义](definition.json) · [新链 AEE](../data/integer_factor_diverse10.json)
- [独立源码审阅](../novelty/REVIEW_INTEGER_FACTOR.md) · [独立函数/周期复算](../data/REVIEW_INTEGER_FACTOR.md)

在本目录复现：

```bash
/opt/anaconda3/bin/python3.12 prepare.py
/usr/bin/python3.12 run.py
/opt/anaconda3/bin/python3.12 ../data/review_integer_counts.py
```

需要原阶段固定 SVD8 因子和八块真实捕获，来源由 prepare.py 明确引用。生成的 factors.npz、fixtures/ 和 obj_dir/ 留本地，不入 Git。run.py 使用 Verilator 4.028 的 `--cc --exe` 再 make；最终 SV 是实现入口，没有另行代码生成步骤。prepare.py 只重建固定函数与测试数据，质量结果由独立评价脚本产生。

模式6为同函数 expanded21 父和直接执行；模式7为 R8 后因子权重驻留；模式8为 R8 输出部分和驻留。当前公开验证范围只含6/7/8，同实例重复同模式；旧模式0–5、运行中换配置及不reset切换模式不属于本次覆盖。
