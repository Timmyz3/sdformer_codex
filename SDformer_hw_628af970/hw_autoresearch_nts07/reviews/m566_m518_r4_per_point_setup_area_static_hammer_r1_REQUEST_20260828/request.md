# M566 M518 r4 fresh static hammer request

请由未参与 r4 编写的 reviewer 作 source-only 盲审。禁止调用 DC、VCS、
runner、远端或大 CPU 任务，禁止创建 launch admission/result/attempt，禁止
修改 docs/359。

除复核 r3 已通过的 per-point、paired schema、50/1175、结构化门、一次
compile_ultra/no-hold、PID/失败双封门外，必须对 M563 两项 P1 定向打铁：

- 证明 final snapshot 会更新/清零 <48 GiB 计数，且 ordinary+ordinary+final
  连续三次低于 48 GiB 会在 final gate 拒绝，同时 <40 GiB 仍 immediate；
- 证明三个 predecessor 的冻结 outer SHA 比较和递归 member/outer seal 校验，
  全部发生在任意 point preflight mkdir、result/work mkdir、attempt marker 前；
  任一失败必须 pre-attempt 退出。

只有 P0=0、P1=0 才可建议 root 后续分别创建 point admission。本 request
launch_now=false，本身不授权任何运行。
