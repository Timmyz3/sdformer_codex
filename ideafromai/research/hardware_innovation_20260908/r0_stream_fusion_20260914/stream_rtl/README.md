# Continuous native r0 stream RTL

完整结果与边界见[REPORT.md](REPORT.md)，费用见[full_costs.csv](full_costs.csv)。

- 六个整帧单go连续作业，442,368,000输出全绿。
- 24个跨行64tile作业，5,898,240输出全绿，含背压/参数驻留/同实例重启。
- `stream_wrapper.sv`拥有全部tile/源地址/origin/启动/输出身份与完成状态。
- 原样`pair_parent_merge.sv`保持mode5/6同底座。
- `resource_contract.json`列出真实接口和未实现的重叠/halo复用/DDR/PPA边界。
