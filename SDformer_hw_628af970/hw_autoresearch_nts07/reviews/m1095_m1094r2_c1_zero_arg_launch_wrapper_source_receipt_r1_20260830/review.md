# M1095 zero-argument launch-wrapper source receipt

**GO 仅限不同作者 M1098 launch hammer；本 receipt 不授权执行。**

M1095 新增了一个 Python 3.10 `-I` 零参数 launcher source。M1094r2 engine/contract/receipt、M1095a library hammer、M1087r3、M1086/M1086r2、Python 和 docs/359 身份全部写死在源码中；源码不读 metric/authority environment。

静态次序是 identity → process/resource/freshness → lock → freshness recheck → atomic attempt → M1094r2 `execute_full` → no-replace publish。`execute_full` 的首个 payload 动作是零参数 2,436,480-value preflight，通过后才会零参数调用一次 full iterator。post-attempt 失败进 quarantine，attempt 不可重试。

本阶段仅做 py_compile 与隔离的 read-only authority/resource 检查；没有调用 launcher main、preflight 或 full replay，没有消费 attempt。由于 M1096 已被 C2 占用，C1 后续编号为 M1098 launch hammer 和 M1099 result hammer；冻结 M1094 输出中的 legacy `M1096` token 只是 schema 别名，不再构成授权。
