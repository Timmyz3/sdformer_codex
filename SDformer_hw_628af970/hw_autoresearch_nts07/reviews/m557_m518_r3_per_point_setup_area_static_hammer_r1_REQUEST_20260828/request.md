# M557 M518 r3 fresh static hammer request

请由未参与本 r3 源码编写的 reviewer 作 source-only 盲审。禁止调用 DC、VCS、
runner、远端或大 CPU 任务，禁止创建 launch admission/result/attempt，禁止修改
`docs/359`。

评审对象是 r3 runner、Tcl、双封 contract、M557 author handoff，以及冻结的
M555/r2 quarantine/attempt 依据。必须核查：

- Fixed/rank3 的 result、attempt、quarantine、future admission 全部独立；
- paired comparison 仅为 schema，且要求两点同身份、各自 clean receipt review；
- preflight 64/128/32 GiB 与 runtime 48 GiB×3、40 GiB immediate、
  Mem/Swap/cgroup/collision immediate、runtime-final gate；
- actual `common_shell_exec` 的 PID/starttime/UID/parent/exe/cmdline 捕获，PID
  复用和 capture failure fail closed；
- 结构化 check-design/check-timing、area macro=0，不存在宽泛 dc.log
  black-box grep；
- source tuple 50 与 DC bit-port 1175 两口径分离；
- 恰好一次 compile_ultra，零 incremental/hold fix/hold-only，且不声称 hold/STA；
- bash syntax、JSON、所有 SHA/双封和 docs/359 身份。

只有 P0=0、P1=0 才可建议 root 后续分别创建 point admission；本 request 本身
不授权任何运行。
