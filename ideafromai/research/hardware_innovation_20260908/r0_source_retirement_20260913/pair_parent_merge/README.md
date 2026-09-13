# Pair-parent forwarding and per-time merge

最后一个有界执行接口已封口：[REPORT.md](REPORT.md)。

- `pair_parent_merge.sv`：共同mode5强控制与mode6逐时间父和转发/双pair归并。
- `tb.cpp` / `run.py`：368runs，1,413,120输出全匹配，真实五臂+固定功能控制。
- `verify_ledger.py` / `ledger_checks.json`：4,048项完整事务/周期核对。
- `resource_contract.json`：共同额外34B父和、单源/W/psum端口、配置计费。
- `SUMMARY.json` / `benefits.csv`：同mask/precision下4.92%–5.55%核心周期净减。

前轮单scratch负结果完整保留在[rtl](../rtl/REPORT.md)，不再追加实验。

独立审阅：[数据与完整计数审核](../data/REVIEW_PAIR_PARENT_MERGE.md)、[结构与公平性审核](../novelty/REVIEW_PARENT_MERGE.md)，均无阻断发现。
