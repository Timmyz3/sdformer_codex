# M765/r3 macro-integrated DC source/candidate hammer request

请做一次全新的只读 source/candidate hammer。不得执行 production runner、DC、VCS、
simv 或任何 EDA；本 request 不授权 launch。

本次 additive r3 只重基准硬门：M746/r12 在建目录前已失败，不能继续作为未来 DC 的
功能准入。新的固定硬门是：

- `results/m758_m533_m528_dead_write_only_1rw_unit_delay_vcs_r13_20260828`
- `reviews/m766_m758_m533_r13_unit_delay_vcs_result_hammer_r1_20260828/review.json`

两者在 authoring 时都不存在，所以 `launch_now=false`。请核对 M757、M761、M758 runner、
M758 true release 和 M763 final-hammer request 的精确身份；未来 M766 必须独立 PASS/100，
P0/P1/P2 全零。只有 M758 真实 PASS 与 M766 hammer 都闭合后，才可另 author DC release。

宏约束保持不变：slow DB SHA `cd8c205...`；DC 禁止读取行为宏 `.v`；pre/post/netlist
宏数都必须恰好为 9；unresolved/inferred parent 直接失败；setup/hold 必须 MET；面积必须
为正。final-review SHA 仍只由 caller 环境 pin，release 不得内嵌未来 review SHA。

当前 claim 仅 source-only。macro-integrated DC、PPA、能量、系统倍速和 headline 全部为
false。

建议 fresh hammer 输出到
`reviews/m765_m528_macro_integrated_dc_source_candidate_hammer_r1_20260828/` 并双封存。
