# M701｜M519 R9 pre-EDA shell failure receipt

裁决：`PRE_EDA_SHELL_FAILURE__NO_DC_STARTED__M519_R9_NOT_CITABLE__ADDITIVE_R10_REQUIRED`。

M694 授权的唯一命令在 fresh 同 UID/资源复核后调用一次，但在 shell 解释函数定义时即以
`line 64: payload: unbound variable` 退出。原因是 `set -u` 下把 `payload` 的赋值和依赖它的
`sidecar` 放进同一个 `local` 声明；右侧展开时新的局部值尚不可见。

失败发生在 admission 验证、preflight、attempt 消耗和 `dc_shell` 启动之前。复核确认本 UID
没有 DC/FM/PT/VCS/simv，R9 canonical、attempt、work 和 preflight staging 均不存在。另一个 UID
的 PID 580855 `simv` 只作 P2 共存披露，未发信号或终止。

R9 永久不可引用。唯一允许的下一步是用全新的 R10 runner/contract/admission/attempt identity
分离局部变量声明，再走 fresh static hammer；不得原地修改或重用 R9。

`docs/359` 未修改，SHA256 仍为
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
