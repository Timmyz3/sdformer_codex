# Native C4 demand sum RTL

固定单 scratch 构造已封口；结论与边界见 [REPORT.md](REPORT.md)，三模式合同见 [resource_contract.json](resource_contract.json)。

- `c4_execution.sv`：mode3 来源追踪、mode5 共同 C4 供数两源强控、mode4 仅实际码归约。
- `tb.cpp`：原生配置、逐输出检查、背压与不复位重启。
- `run.py`：完整编译/正式五臂/开发与定向功能回归。
- `verify_ledger.py`：原生几何与闭式计数核对。
- `results.json` / `SUMMARY.json` / `benefits.csv`：648 runs、2,488,320 输出全绿。

独立审阅：[数据与完整计数审核](../data/REVIEW_C4.md)、[结构与公平性审核](../novelty/REVIEW_RTL.md)，均无阻断发现。
