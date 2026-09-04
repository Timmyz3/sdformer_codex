# M1123r4：M1122r4 C2 DC-selector engine 独立静态打铁

裁决：**GO，但只授权下一位作者创建 zero-argument launcher；不授权启动 launcher、消费 attempt、运行 DC/VCS 或产生任何性能结论。**

## Selector 与运行时身份

正向调用路径精确固定为：

```text
/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell -f <pinned Tcl>
```

`dc_shell` 必须是 raw link text 为 `snps_shell` 的符号链接，解析后的 wrapper SHA256 必须为 `23a4101...`。该路径启动后，同一个 Popen PID 必须 exec 成 SHA256 `bf91e6ab...` 的 `common_shell_exec`，其 NUL-safe argv 必须精确为七项：

```text
common_shell_exec -shell dc_shell -r /opt/synopsys/syn/V-2023.12-SP3 -f <pinned Tcl>
```

engine 对同一 PID 的 starttime、UID、`/proc/<pid>/exe` 和 `/proc/<pid>/cmdline` 做联合检查。PID/exec 过渡期间读到 wrapper 状态可以继续轮询；starttime 或 UID 改变、backend 路径错误、argv 任一 token 漂移、子进程提前退出或四秒内没有捕获 backend 都会终止子进程并 fail closed。

独立 mock 没有调用任何工具：合法的 same-PID exec 序列通过；错误 `-shell` selector、永不进入 backend、starttime 变化、UID 变化、捕获前退出五类攻击全部拒绝。静态攻击还证明正向 Popen 只能使用 `dc_shell`，不能退回直接 `snps_shell` 或直接 backend。

## Namespace 与 no-retry

M1112r3 的 consumed attempt 和失败 quarantine 原 seal 均重算通过。其唯一 attempt 仍为一次，失败仍在 `FRESH_DC_M1112R3`，`m1112_retry=false`，canonical result 不存在；M1112r3 永久禁止 retry 和 namespace reuse。

M1122r4 使用完全不同的 attempt/result/work/failure/lock 名称。目前这些对象均不存在。engine 在任何 future authority 之后仍会检查 attempt、result、当前 work、lock、所有 work glob 和 failure glob；预存在的 result 也会拒绝执行。未来最大 attempt 数为 1，任何失败都必须 quarantine，automatic retry 始终为 false。

## 身份、封存与状态边界

engine 是 direct regular file，SHA256 为 `f278052d...`。contract primary/sidecar/outer 三层一致，contract outer-seal-file SHA256 为 `373e6b86...`。author receipt 的 exact-member manifest 和 outer seal 一致，outer-seal-file SHA256 为 `c36311a8...`。M1121、旧 attempt 和旧 failure 的 sealed authority 也全部复核。

286 项检查、38 个攻击全部通过/拒绝，覆盖 selector、backend、PID/argv race、旧新 namespace 混淆、自动重试、两次 attempt、预存 result、未来 hash fixed-point、mapped/performance/system/PPA 状态升级、duplicate key、NaN/Infinity、live extra、sealed symlink 和 contract-primary symlink。

下一阶段只能由独立作者创建 caller-blind、zero-argument launcher 和 launch receipt。它必须绑定本 review 的 outer seal，且完成另一轮独立 launch hammer 后，才可能授权一次 fresh execution。M1123r4 本身不授权执行。

`docs/359_DATE终局冻结_20260813.md` 未修改，SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
