# M756/r2 macro-integrated DC source/candidate hammer request

请做一次全新的只读 source/candidate hammer。不得执行 production runner、DC 或任何
EDA；本 request 本身不授权 launch。

重点审查：

1. M750 自审 `NO_GO` 的哈希环是否成立，旧 runner/contract/candidate 是否保持原 SHA，
   且没有补写 release、result 或 attempt；
2. M756 runner 是否只固定 final-review 路径，不内嵌未来 review SHA；调用者是否必须用
   `M756_EXPECTED_DC_FINAL_RELEASE_REVIEW_SHA256` 独立 pin payload；final review 是否被
   要求反向绑定已经存在的 release SHA；
3. future release 是否只允许绑定 source/candidate hammer（以及既有 source/M746 gate），
   并明确禁止内嵌 final-review SHA；
4. 宏规则是否保持：slow macro DB SHA `cd8c205...`、禁止行为 `.v`、pre/post/netlist
   均恰好 9 个宏、无 unresolved/inferred parent、setup/hold 必须 MET、面积必须正数；
5. M746/r12 VCS PASS 及其独立 result hammer 是否仍是 release 前硬门；
6. `static_no_eda_selftest.sh` 是否真的只做静态检查，不能触发 runner/EDA/attempt。

建议输出到
`reviews/m756_m528_macro_integrated_dc_source_candidate_hammer_r1_20260828/`，并双封
`review.json`、`review.md`、mechanical checks。只有 `PASS/100` 且 P0/P1/P2 全零，才允许
另一个 agent 单独 author `launch_now=true` release；该 release 仍不能运行 DC。

当前 claim 仅 source-only。macro-integrated DC、PPA、能量、系统倍速均为 false。
