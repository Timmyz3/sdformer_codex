先读[REPORT](REPORT.md)、[PLAN](PLAN.md)和[resource_contract](resource_contract.json)。

- 完整Q1→z→Q2→rawp，C96/N96/T10/K864，112runs /430080输出通过。
- 新同208bit口：14 scalar-packed 91103→15双P 87599拍（真实八块、无背压，−3.846%）。
- 原生160bit源窗、cachedOS、rank/位置零支持均共同；每个双活动pair精确省3拍，Q2 MAC不变。
- `packed_r8.sv`为唯一核；`/opt/anaconda3/bin/python3.12 prepare.py`、`run.py`、`verify.py`复现。Verilator4.028严格-Wall。
- 资源口宽不同于旧partition12；无跨核等面积、Fmax/PPA/本叶新AEE声明。作者新颖性自评不等于独立审阅。
