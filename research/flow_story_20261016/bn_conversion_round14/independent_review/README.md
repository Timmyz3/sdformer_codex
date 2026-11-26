# R14 独立工程审计

本目录不启动 GPU、不 SSH、不读取或排名 AEE。只验证固定两份原 W 的 ALL78_CURRENT/native 原生 16 次执行来源、状态、实际 FC 服务及原预测工件。

- `SOURCE_STATES_V1.json`：真实两原 PT 的 CPU 独立读取与完整 711 / 移除全部 234 BN buffers 后 477 状态重建。
- `IMPLEMENTATION_REVIEW_V1.md`、`INDEPENDENT_CPU_REVIEW_V1.json`：保留首次源码/4 项小测试结果；真实远端 CPU preflight 随后发现 10 键成员来源字典的整体比较错误，零 GPU 推理。
- `IMPLEMENTATION_REVIEW_V2.md`、`INDEPENDENT_CPU_REVIEW_V2.json`：V2 最小修复的逐行增量审查，5 项独立测试包含真实 10 键清单。实际执行必须绑定 V2 源、CPU receipt、root `EXECUTION_FREEZE_V2.json`。
- `audit_conversion.py`：独立审计，不导入作者 runner/core，只复用旧冻结独立审计器的几何、张量摘要与原生 hook 校验。重新读取两原 PT、16 新预测、16 既有缓存 native 预测；从全部记录重新算每 FC 8,928,000 行。
- `test_auditor_cpu.py`：审计器的独立合成正例与拒绝测试；不读取实际 R14 结果。

审计器代码、测试及输入源在读取实际 R14 结果前由 `PRE_RESULT_FREEZE_V1.json` 冻结（其 V1 是本审计 freeze 的版本，执行源仍为 runner V2）。执行命令：

```
python -B independent_review/audit_conversion.py --stage prepare --output independent_review/PRE_RESULT_FREEZE_V1.json
python -B independent_review/audit_conversion.py --stage audit --freeze independent_review/PRE_RESULT_FREEZE_V1.json --result run_v2/results.json --output independent_review/ACTUAL_EXECUTION_AUDIT_V1.json
```

第二条的结果路径以 root 实际回收目录为准；来源由 SHA 绑定。禁止覆盖已存在审计文件。若冻结审计器自身失效，保留失败记录并另建版本，不能原位修补后冒充原版通过。

执行 PASS 只说明所记录的 16 次原生对照来源与工件一致。完整中间张量未逐一留存或 CPU 重演；78 个 BN 的即时归约数学亦未独立 CPU 重演。原数组/GT/全部分区的精度比较由另一位内部审者独立负责。既有 54 BN 部署模型不是全 78 current-stat 原生模型的整网周期分母，本审计不报硬件周期或能效结论。
