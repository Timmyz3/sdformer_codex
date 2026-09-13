# r0 Winograd完整线性核

先读[REPORT.md](REPORT.md)，资源边界见[PLAN.md](PLAN.md)与[resource_contract.json](resource_contract.json)。最终[cycles.csv](cycles.csv)/[SUMMARY.json](SUMMARY.json)为共同64项static-support successor版本；旧逐项扫描版在old_scan_control/。

`bash reproduce.sh` 生成三个合成控制、接同级data_and_quality真实八块NPZ、Verilator4.028编译并运行88case；337920输出逐值检查。完整 C96/N96/T10，每条命令3840值；不含整图tile供数或下游PSN。所有写入限本目录。

结果：原函数direct在真实块更快；masked Winograd只胜过同函数phase-expanded控制。十帧质量由同级diverse10.json提供，未有valid825/fullnetbittrue/PPA。
