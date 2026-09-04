# M546 / M519 R8 setup-area 三轴流程 bounded repair 作者交接 r1

日期：2026-08-27  
状态：`AUTHOR_SOURCE_ONLY_COMPLETE__NO_EDA_RUN__FRESH_STATIC_HAMMER_REQUIRED`

## 交付与边界

本轮只新建 R8 Tcl、runner、recovery contract、作者交接和 fresh static-hammer request。没有运行 DC、VCS、PT、PTPX、Formality、CPU 大任务或远端任务；没有创建 launch admission；没有修改 R5-R7、`docs/524` 或 `docs/359`。

- runner：`dc_handoff/scripts/run_dc_m519_r8_setup_area_three_axis_exact_sha.sh` (`bd830577a7f31413189c78355c3e9467a567e0b90c1e0edcd6d1707d1b7e73c2`)
- Tcl：`dc_handoff/scripts/run_dc_m519_r8_setup_area_three_axis.tcl` (`c9da61c9a483487b3d1157538481a6c940d7277534e2acef634c4b1a1ff7adbe`)
- contract：`contracts/m519_r8_setup_area_three_axis_recovery_contract_r1_20260827.json` (`33273e1411cff09f793906a61d4c68964c299aad8dceae91921a5229bdf5acf4`)

## 对 M540 R7 缺陷的 bounded repair

1. 工具身份不再把 `realpath(dc_shell)=snps_shell` 当作长期 DC executable。R8 冻结并在 launch time 校验入口 `dc_shell`、wrapper `snps_shell`、实际 `common_shell_exec` 与 slow/fast DB 的路径和 SHA。
2. capture 只接受同一 fork birth 的 `(PID,parent,starttime,UID)`，且 `/proc/PID/exe` 必须为冻结的 `common_shell_exec`，NUL-safe argv 必须精确为 `common_shell_exec -shell dc_shell -r <install-root> -f <exact-R8-Tcl>`。capture 失败不会 `wait` 一个无 monitor 的综合：它立即只对精确 birth tuple 发 TERM，限时后仍存活才对同一 tuple 发 KILL，然后 wait 并进入 quarantine。
3. future admission closed key set 新增 wrapper/actual executable、R6 failed review 和 R7 disqualified review。五个 R5 basis 不仅自校验内外 seal，还将实际 `SHA256SUMS.seal.sha256` 文件 SHA 与 contract/admission 逐一交叉；R6/R7 review 同样双封并钉死 outer-seal SHA 与状态。
4. 外部 EDA collision 不再只写 `pid:comm`：每个候选在独立 TSV 中记录 timestamp/label/kind/PID/PPID/UID/starttime/state/comm-hex/exe-hex/完整 NUL-preserving cmdline-hex。PID 身份变化也保留变化前完整 tuple。
5. exact birth 的 zombie 被视为已完成而非 PID reuse；任何 starttime/UID/parent/exe/cmdline 不一致仍 fail-closed，且绝不 signal 重用 PID。

## 保持不变的综合口径

R8 Tcl 仍是 K1/K8/K1x8 同 shell、两角库、filelist、SDC、3 ns 和同一 Tcl；只有一次 `compile_ultra`，没有 incremental 或 pre-CTS hold-only optimization。hold 仅诊断，`hold_not_closed_at_dc=true`。

## 静态自查

- `bash -n` PASS；contract `jq` PASS；`exact_files` 17/17 当前匹配。
- Tcl 命令计数：`compile_ultra=1`、incremental=0、hold-only=0。
- contract 内外双封通过；future launch admission、R8 canonical、R8 attempt 均不存在。
- R5 basis outer seal 5/5、R6 failed review 与 R7 disqualified review outer seal 当前字节均匹配 contract。
- `docs/359` SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

下一步只能是一个未接触本轮作者工作的 fresh independent zero-EDA static hammer。只有其得到 `P0=0 && P1=0` 后，主 agent 才可另建一次性双封 launch admission。本交接不授权运行，也不证明面积、时序、功耗、能量、吞吐/面积、完整 FC2 或系统加速。
