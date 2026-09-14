先读[REPORT](REPORT.md)及[resource_contract](resource_contract.json)。

- 120 runs /460800输出通过；mode12→13真实98895→102757，分组端点负3.91%。
- 最终为29bit descriptor hold版；跨N共享描述符、8项cache与epoch/构造完整计费。
- `/opt/anaconda3/bin/python3.12 prepare.py`、`prepare_targeted.py`、`run.py`、`verify.py`可复现；Verilator4.028 `--cc --exe`，无其他依赖安装。
- baseline本身是更强cachedOS，不能把对旧mode8的改善归给分组。
