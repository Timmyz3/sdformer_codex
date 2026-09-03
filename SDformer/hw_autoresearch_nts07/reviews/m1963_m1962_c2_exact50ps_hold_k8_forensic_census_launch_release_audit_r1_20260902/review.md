# M1963：M1962 C2 exact-50 ps K8 forensic-census release 独立审计

结论：**PASS（只授权一次 M1960 K8 pilot）**。

- M1962 release SHA-256 为 `eef9babb0834a754e8704ae9e8dada02a76da4239553921e0ad25106aa36a28f`，release sidecar 与 sidecar seal 均通过。
- release 精确绑定 M1960 runner `b6a8d63d...` 与 M1961 review `8c709fb7...`；M1938/M1940/M1944/M1953、M1811/M1830、M1939 Tcl、K8 DDC/SDC 与 docs/359 的传递身份均通过封存校验。
- 从冻结 M1960 runner 原样抽出的第一段 Python authority parser（SHA-256 `a47da3ab...`）：正向 authority tuple 通过；对 release schema/status/identity/budget/axes/gates 及 audit schema/milestone/reviewer/status/count/identity 的 12 组独立负向变异均 fail-closed。
- 审计没有调用 runner，没有查询许可证，没有创建 attempt，没有运行 DC/Formality/PrimeTime。
- 审计时 M1960 attempt/result/failure/work/lock namespace 均为空；同 UID blocked EDA 进程为 0。资源状态仅是时点观测，不替代 runner 自己的启动前门控。

授权边界：只允许冻结 M1960 runner 消耗一次 K8 `lmstat`/DC pilot；禁止自动重试。即使 raw PASS，也必须经过异作者 result hammer、Formality 与 PrimeTime，才可讨论 hold/PPA 准入。
