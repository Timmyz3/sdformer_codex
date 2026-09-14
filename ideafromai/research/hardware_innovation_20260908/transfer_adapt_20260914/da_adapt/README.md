# Q2 DA有费适配

[报告](REPORT.md)：最终模式11真实八块87,599→94,823拍，仍慢8.247%；540条实际RTL命令、2,073,600个输出全过。原14/15、保守13/12和修正11全部保留。

重现（本目录，Python `/opt/anaconda3/bin/python3.12`、Verilator4.028）：依次运行`implement.py`、`prepare.py`、`run.py`、`audit.py`。原实验树仅作为只读源码/fixture/记录入口；产物都在本目录。[实现合同](PLAN.md)、[资源](resource_contract.json)、[完整结果](results.json)、[独立检查](checks.json)。

`run_cost_correction.py`是仅补测mode11的增量入口；`run.py`直接重现全部五模式。`review_pair.py`只读复核另一条路径的packed退休算术及服务差，不修改其文件。
