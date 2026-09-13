# C4 / 源退休差分审阅

先读[REPORT.md](REPORT.md)；[source_table.csv](source_table.csv)严格区分本轮primary重读、旧记录复用和仅摘要定位。唯一两原生源fanout连接候选经过几何核验降为BASE_ONLY，尚未立X。

新C4核的完整独立静态审阅及最终648条结果记录核对在[REVIEW_RTL.md](REVIEW_RTL.md)，未重跑RTL或GPU。可复核的小数学产物为[geometry_bound.json](geometry_bound.json)；只读记录审计为[result_record_audit.json](result_record_audit.json)。
