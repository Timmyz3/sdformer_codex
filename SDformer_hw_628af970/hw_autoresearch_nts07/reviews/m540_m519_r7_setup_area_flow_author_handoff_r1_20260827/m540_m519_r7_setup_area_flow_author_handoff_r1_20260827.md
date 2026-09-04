# M540 / M519 R7 setup-area 三轴流程 bounded repair 作者交接 r1

日期：2026-08-27  
状态：`AUTHOR_SOURCE_ONLY_COMPLETE__NO_EDA_RUN__FRESH_STATIC_HAMMER_REQUIRED`

## 交付与边界

本轮只新建 R7 Tcl、runner、recovery contract、作者交接和 fresh static-hammer request；没有运行 DC、VCS、PT、PTPX、Formality、CPU 大任务或远端任务。没有修改 R5、R6、`docs/524` 或 `docs/359`。

- runner：`dc_handoff/scripts/run_dc_m519_r7_setup_area_three_axis_exact_sha.sh` (`fcc98b44666b163e54b1075d87326dc342b4e73c21ea1e8819479592bbfd2b43`)
- Tcl：`dc_handoff/scripts/run_dc_m519_r7_setup_area_three_axis.tcl` (`700a22770c2328e558e89a14aa7308971d2bf89fc314eb0b038afc6f05c54f9f`)
- contract：`contracts/m519_r7_setup_area_three_axis_recovery_contract_r1_20260827.json` (`c0b1ac57c3f80ac16ef59067d7e7d30adc9d60ef97993a07551ece00e4c902a2`)

## 对 M538 P1/P2 的 bounded repair

1. runner 在第一次 preflight 和 attempt 消费之前，对 contract `exact_files` 使用 closed path set 并逐项校验 SHA；future admission 的 identity 也使用 closed key set，并把 runner/Tcl/filelist/SDC/DC/两角库/R5 seals/`docs359` 的 path+SHA 逐项交叉到 contract。runner SHA 同时受调用者 pin、admission、contract 和 `exact_files` 四方一致约束。
2. runtime loop 与 `runtime_final` 调用同一个 snapshot gate。final 会更新连续 commit 计数并检查 MemAvailable、SwapFree、cgroup/OOM、外部 EDA 与 campaign identity；写出同步 `runtime_final_gate_ack.txt`，父 runner 同时检查 monitor rc 和 PASS ack。
3. DC root 在 exec 后冻结 `(PID,/proc starttime,UID,resolved executable)`。祖先链用每个 `(PID,starttime)` 二次读取校验；liveness、外部 EDA 排除和 TERM 都要求精确 tuple。PID 重用会锁存失败，且不会排除碰撞或 signal 新进程。
4. 每次 preflight PID-tree 与每次 runtime descendant 样本记录 PID/PPID/UID/starttime、完整 comm/executable 和 NUL-preserving cmdline hex；高水位以 `PID:starttime` 为 key 并保留同一 provenance。所有证据进入嵌套与根级双封。

## 保持不变的综合口径

R7 Tcl 仍是 K1/K8/K1x8 同 shell、两角库、filelist、SDC、3 ns 和同一 Tcl；只有一次 `compile_ultra`，没有 incremental 或 pre-CTS hold-only optimization。hold 仅诊断，`hold_not_closed_at_dc=true`。

## 静态自查

- `bash -n` PASS；contract `jq` PASS；`exact_files` 17/17 当前匹配。
- 独立 Tcl 命令计数：`compile_ultra=1`、incremental=0、hold-only=0。
- contract 内外双封通过；future launch admission、R7 canonical、R7 attempt 均不存在。
- `docs/359` SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

下一步只能是 fresh independent static hammer；本交接不授权任何运行，也不证明面积、时序、功耗、能量、吞吐/面积、完整 FC2 或系统加速。
