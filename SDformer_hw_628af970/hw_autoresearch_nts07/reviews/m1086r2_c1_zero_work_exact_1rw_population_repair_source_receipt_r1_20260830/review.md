# M1086r2：C1 zero-work contract population 修订

本修订只纠正 M1086 contract 的 population：canonical workload 是 **812,160 tasks × 3 designs = 2,436,480 task-design work values**，因此 exhaustive work-domain preflight 的 `values_checked` 必须精确等于 2,436,480。

M1086 driver 与 tests 字节保持不变；zero-work、positive-work delegate、dependency 和 zero-argument production interface 语义均未改。旧 M1086 contract 与原 M1087 GO 已被 M1087r2 STOP 撤销，不得用于 release。

本 author 没有运行 bounded tests、exhaustive preflight 或 full replay，没有消费 attempt，也没有创建 M1092/runner。下一步只允许不同作者对 repaired population、source loop、冻结身份和 bounded attacks 重新 hammer；新的 sealed GO 之前，M1092 继续禁止。
