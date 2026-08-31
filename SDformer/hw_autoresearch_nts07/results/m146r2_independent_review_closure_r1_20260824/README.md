# M146r2 review-fix 独立 closure audit

结论：**96/100；P0=0、P1=1、P2=0。** r2 已完整关闭原评审的 reset-release H02，同时如实保留原 32-bit identity ABA P1，没有误报关闭。

## 核验结果

| 检查 | 结果 |
|---|---|
| Sealed VCS inputs/outputs/runner SHA | 全部通过 |
| Sealed compile/sim RC | 0 / 0 |
| 独立 exact-source VCS rerun | PASS |
| 正常流量 | 40 fills、40 PWP、40 correction、40 releases |
| Bank reuse | 36 |
| 协议攻击 | 4/4 |
| Reset 期间 matching correction completion | 组合检查和 reset 边沿后均 `release_valid=0` |
| Assertion failure | 0 |
| `docs/359` SHA | `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4` |

RTL 已把 release 写为 `!rst_core && correction_done_valid && ...`。SVA 新增的 `ap_no_release_during_reset` 使用局部 `disable iff (1'b0)`，不会被模块默认的 reset disable 掩掉。Directed test 先形成真实 active correction identity，再在 `rst_core=1` 时输入完全匹配的 completion，并分别在组合延迟后、同步 reset 边沿后检查 release 均为 0。因此 H02 closure 可信。

4 个 directed attack 分别为：reset 期间 matching correction completion、错误 fill sequence、live-bank refill、错误 PWP completion sequence。独立复跑再次得到 `protocol_attacks=4`，且所有既有 cover 非零。

## 仍冻结的 P1

r2 没有 epoch bit，也没有从硬件上消除 full identity reuse：

- `maximum_allocations_per_identity_epoch=4294967295`
- `external_reset_flush_required=true`
- `full_identity_reuse_aba_unconditionally_closed=false`

contract 与 sealed receipt 对这三项一致。因此 M146-H01 仍是 P1；只有实现不可混淆的 epoch，或提供可执行/可断言的 response-lifetime 与 reset-flush 约束后才能关闭。当前禁止宣称 unconditional stale-response rejection。

本次只做 VCS closure；没有理由重跑 DC，因为 r2 的 production 修改限于 release reset gate、对应 SVA/TB/contract，且任务未授权改变 production。机器结论见 `m146r2_independent_review_closure_r1.json`，`manifest.sha256` 覆盖本目录保留证据。未修改 production、sealed run 或 `docs/359`，未 commit/push。
