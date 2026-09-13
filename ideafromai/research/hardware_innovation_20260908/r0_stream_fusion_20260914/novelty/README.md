先读[REPORT.md](REPORT.md)：第一候选为全K/R8/N96离散因子计数与连续收缩，第二为一维Winograd＋另一维direct。两者在本阶段起点均无对应完整原生RTL，前者延续9/13 D1；后续实际实现见下面的独立审阅，尚无独立X结论。

随后实施的窄整数R8核见[独立静态审阅](REVIEW_INTEGER_FACTOR.md)：共同 k_live 版168 runs通过，mode7对同函数expanded21控制真实八块核心周期减少53.07%；实际未实现按系数码分桶，收益归入强A，未立X。

一维Winograd的[最终独立审阅](REVIEW_ONE_AXIS.md)：52 runs通过，真实八块比保留共同强控制的direct慢70.93%；只停止优先采用该布局，不据此关闭变换家族。

[source_table.csv](source_table.csv)及[JSON](source_table.json)区分本次primary重读、旧全文复用、仅摘要及建议文件。Comperity只有新核到的出版者摘要/元数据，不能称完整A已读。
