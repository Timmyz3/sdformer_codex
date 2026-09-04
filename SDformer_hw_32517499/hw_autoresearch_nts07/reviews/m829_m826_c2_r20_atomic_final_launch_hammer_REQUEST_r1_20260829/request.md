# M829 fresh final-launch hammer request for M826/C2 R20

请由未参与 M826/M828 authoring、也未执行本 release 的 receipt-blind reviewer 完成最终 release hammer。本请求本身不授权 reviewer 运行 VCS、simv、license query 或任何 EDA；reviewer 只能做 source-only、synthetic temporary chain、collision/no-clobber 和 zero-side-effect dry-run。

## 必须闭合的 authority

- release 固定为 `contracts/m826_c2_r20_atomic_vcs_launch_admission_r1_20260829.json`，SHA256 必须是 `52606ff5...c830f`，outer-seal-file SHA256 必须是 `a6986135...e0dc`。
- release 的 frozen 6-key schema authorization 必须与 M826 guard 完全匹配；其 `final_hammer_authorization_exact` 必须是以下 15 键 typed closed set。
- future `review.json` 的 `authorization` 必须逐键、逐值、逐 Python 类型等于同一个 15 键集合；缺键、额外键、`true/1` 或 `false/0` 混淆、错误值、duplicate key、NaN/Infinity 均拒绝。
- future review target 必须精确绑定 release/runner/contract/candidate SHA；review 必须双封，caller 必须 pin 其 outer-seal-file SHA。

最终授权集合只允许一次 VCS、一次 simv 和必要的 license query；Icarus、Verilator、DC、Formality、PT、PTPX、CPU/GPU workload、remote/network job 全部禁止。

## 必须重放

1. M827 PASS100、M828 handoff、M823 negative review、release/runner/guard/contract/candidate 双封与 live SHA。
2. contract 的 40 source SHA；M803 RTL/SVA/TB/filelist；五组 exact cycle；numeric/tuple/weight/stall/full8/out-of-order gate。
3. Python 3.6/3.12 atomic 12/12、final-authorization 8/8、closure positive/undefined negative、wrong-SHA rc3、source dry-run rc86。
4. 四份临时 CLI 双封 receipt `false/false/true/true`，pre-existing exact collision 双侧 no-clobber，postrename damaged conservative consumed。
5. 合法 synthetic future chain 必须 PASS；run_vcs=false、run_simv=false、query_license=false、max_attempts=0、extra key、所有 missing key 和所有 bool/int confusion 必须拒绝。
6. formal attempt/result/failure quarantine 在评审结束前仍为 0，`docs/359` 仍为 `dedde7ce...`。

只有 PASS100、P0/P1/P2=`0/0/0` 可发布固定路径 `reviews/m829_m826_c2_r20_atomic_final_launch_hammer_r1_20260829`。reviewer 不得直接 launch；PASS 后由 caller 在 live collision/resource/license preflight 下使用固定 clean-env 命令执行一次 runner。
