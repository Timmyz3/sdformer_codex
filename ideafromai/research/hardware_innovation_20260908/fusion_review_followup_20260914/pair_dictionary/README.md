固定四个R2×32类的精确Q1 sum-first。完整RTL结果为真实八块核心慢29.539%，停止该布局。入口：[报告](REPORT.md)、[实现前计划](PLAN.md)、[资源合同](RESOURCE_CONTRACT.md)。

在本目录依次执行：

```bash
/opt/anaconda3/bin/python3.12 prepare.py
/opt/anaconda3/bin/python3.12 run.py
/opt/anaconda3/bin/python3.12 verify.py
```

Verilator4.028 `--cc --exe`；所有构建和结果仅写本目录。原冻结树只作为source/fixture/control证据读取。prepare另读同级audit_decompositions中已生成的两个合法Q1角落；不训练、不调用GPU。
