先读[REPORT](REPORT.md)、[PLAN](PLAN.md)、[resource_contract](resource_contract.json)。

- 固定完整核112runs/430080输出通过；结果见[SUMMARY](SUMMARY.json)。
- `decomp_core.sv`与`tb.cpp`为最终唯一实现源；本目录`/opt/anaconda3/bin/python3.12 run.py`回放。
- 上级`prepare.py`仅复制fixture/静态字典，上级`verify.py`独立核验；无RTL生成脚手架。
- 这是本固定布局的负控制，无训练/消费者/PPA声明。
