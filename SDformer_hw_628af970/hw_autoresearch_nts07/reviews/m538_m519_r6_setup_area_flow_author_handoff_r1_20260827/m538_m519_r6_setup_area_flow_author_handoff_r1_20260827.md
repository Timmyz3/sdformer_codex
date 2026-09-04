# M538 / M519 R6 setup-area 三轴恢复流程作者交接 r1

日期：2026-08-27  
状态：`AUTHOR_SOURCE_ONLY_COMPLETE__NO_EDA_RUN__FRESH_STATIC_HAMMER_REQUIRED`

## 交付

本轮只新建 R6 流程源文件，没有运行 DC、VCS、PT、PTPX、Formality、CPU 大任务或远端任务；没有修改 R5 quarantine、`docs/524` 或 `docs/359`。

- Tcl：`dc_handoff/scripts/run_dc_m519_r6_setup_area_three_axis.tcl`
  (`b5c56877e8fdb920cfaf916e7f93783277557f3a00010a4eb259a89f1f463ba1`)
- runner：`dc_handoff/scripts/run_dc_m519_r6_setup_area_three_axis_exact_sha.sh`
  (`7a7cebe33c9e078bd341cd93009b3a313edf194da3bf04607c93186d8ae643d7`)
- recovery contract：`contracts/m519_r6_setup_area_three_axis_recovery_contract_r1_20260827.json`
  (`205203bd9f3c3d8bac3187d66b94fae6d2bb7af99d460ca98cf427d46c24e576`)
- fresh static request：`reviews/m538_m519_r6_setup_area_flow_static_hammer_r1_REQUEST_20260827/`

## R5 两个 P1 与一个 P2 的修复

1. 主 Tcl 是统一 setup/area flow：K1/K8/K1x8 使用同一 shell、库、Tcl、filelist、SDC 和约束；只有一次 `compile_ultra`。没有 incremental 和 pre-CTS hold optimization；hold 报告明确仅诊断，`hold_not_closed_at_dc=true`。
2. 每个待启动轴均有三个相隔 10 秒的 preflight 样本；每个样本都要求 commit headroom >=64 GiB、MemAvailable >=128 GiB、SwapFree >=32 GiB、cgroup 三项为零且同 UID 没有外部 DC/FM/PT/VCS。K1 结束后重新准入 K8，K8 结束后重新准入 K1x8，K1x8 后再做最终三样本恢复检查。
3. 运行期 commit headroom 只有严格低于 32 GiB 连续三个 10 秒样本才锁存，任一恢复样本把计数清零；MemAvailable、SwapFree、cgroup/OOM 或新同 UID 外部 EDA 碰撞则立即锁存。锁存只终止本 campaign child。
4. 每个 runtime 样本记录全局 H0 差值、碰撞和 campaign descendant 的 PID/PPID/command/VmPeak/VmSize/VmRSS/VmSwap；另给每 PID 高水位表。VmSize 不从全局 commit 中做不精确抵扣。
5. 所有 preflight 目录都独立生成 `SHA256SUMS` 与 `SHA256SUMS.seal.sha256`，并作为嵌套文件进入最终 canonical/quarantine 的根级双封。第一次 preflight 若失败，也会独立双封为 `preflight_rejected`，且不消费 DC attempt。

## 一次性与失败语义

未来 runner 必须由调用方同时 exact-pin 自身 SHA 和一份尚不存在、经独立准入后才可创建的 launch admission。R6 使用全新 canonical 和 attempt sentinel；第一条 `dc_shell` 之前原子消费 attempt。任何轴、轴间恢复/preflight、runtime 或结果门失败都会停止后续轴，并将整个 work tree 双封进 quarantine。

R5 的状态永久保持 `FAILED_OR_INCOMPLETE_DO_NOT_CITE`；R6 不会读取其 K1 QoR 作为结果，只冻结 R5 失败收据/隔离封印作为修复 provenance。

## 静态自查

- runner `bash -n` 通过；contract `jq` 解析通过；contract `exact_files` 17/17 匹配。
- Tcl 精确匹配一条独立 `compile_ultra` 命令；无 `set_fix_hold`、`only_hold_time`、incremental mapping 或 incremental compile 命令。
- future launch admission、R6 canonical 和 R6 attempt 均不存在。
- 未运行任何 EDA；本交接不证明 loop-free、DC、面积、setup timing、hold closure、power、energy、吞吐/面积、完整 FC2 或系统加速。
- `docs/359` SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

下一步只能由新独立 reviewer 完成 source static hammer；只有 P0/P1=0，才能另建一份双封 launch admission。当前交接本身不授权运行。
