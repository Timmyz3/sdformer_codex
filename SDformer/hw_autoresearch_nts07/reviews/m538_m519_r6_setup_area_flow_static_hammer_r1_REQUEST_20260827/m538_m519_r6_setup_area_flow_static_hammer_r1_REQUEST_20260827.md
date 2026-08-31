# M538 / M519 R6 setup-area flow fresh static hammer request r1

请由未参与 R6 authoring 的独立 reviewer 做 source-only hammer。**禁止运行 DC、VCS、PT、PTPX、Formality、CPU 大任务或远端任务；禁止创建 launch admission；禁止修改 R5、docs/524 或 docs/359。**

## 冻结输入

- Tcl `dc_handoff/scripts/run_dc_m519_r6_setup_area_three_axis.tcl`
  SHA256 `b5c56877e8fdb920cfaf916e7f93783277557f3a00010a4eb259a89f1f463ba1`
- runner `dc_handoff/scripts/run_dc_m519_r6_setup_area_three_axis_exact_sha.sh`
  SHA256 `7a7cebe33c9e078bd341cd93009b3a313edf194da3bf04607c93186d8ae643d7`
- contract `contracts/m519_r6_setup_area_three_axis_recovery_contract_r1_20260827.json`
  SHA256 `205203bd9f3c3d8bac3187d66b94fae6d2bb7af99d460ca98cf427d46c24e576`
- author handoff `reviews/m538_m519_r6_setup_area_flow_author_handoff_r1_20260827/`（必须先验双封）
- R5 final failure hammer `reviews/m519_r5_final_failure_receipt_hammer_r1_20260827/`（必须先验双封）

## 必查问题

1. Tcl 是否在三个 ARCH_MODE 上严格共享同一 shell/Tcl/filelist/SDC/library，并且动态可达的综合主序列只有 `ungroup` 加一次 `compile_ultra`？是否完全不存在 incremental 或 pre-CTS hold optimization？hold 是否只作为诊断，绝不覆盖或门控 setup/area checkpoint？
2. 每个轴启动前是否真的采三次、相邻样本真的 sleep 10 秒，并且三次都独立要求 commit >=64 GiB、MemAvailable >=128 GiB、SwapFree >=32 GiB、cgroup 三项为零、同 UID 外部 DC/FM/PT/VCS 为空？H0 是否取三次 headroom 的最小值？
3. runtime commit 的 `<32 GiB` 是否严格连续三个样本才锁存，任一恢复样本是否清零计数？MemAvailable、SwapFree、cgroup/OOM 和新外部 EDA 是否单样本立即锁存？比较符号边界是否正确？
4. runtime 是否只排除本 campaign DC 子孙，而不会把另一个同 UID EDA 误当作本 campaign？锁存是否只 TERM 本 child，不会杀其他用户/外部进程？
5. 是否逐样本记录 campaign descendant 的 VmPeak/VmSize/VmRSS/VmSwap、全局 H0 差值、碰撞身份，并生成高水位？是否明确不把 VmSize 从全局 commit 做不精确抵扣？
6. K1 后到 K8、K8 后到 K1x8 是否均经过资源恢复和全新三样本 preflight？K1x8 后是否也有 final recovery？任何失败是否停止后续轴并 quarantine？
7. 第一次 preflight 失败是否不消费 attempt 但仍双封？全部轴的 preflight manifest/outer seal 是否又进入根级 canonical/quarantine 双封，真正关闭 R5 P2？
8. runner 是否在第一条 DC 前原子消费全新的 R6 attempt；是否拒绝 canonical/attempt 冲突、路径 override、未 exact-pin runner/admission、未封依赖？future admission 当前是否确实不存在？
9. setup pass gate 是否只门控 max-delay/max-cap/max-transition/max-fanout，而不会错误要求 pre-CTS hold closure？TIM-209/OPT-150 是否仍 fail-closed 且 compile 只在 PASS branch？
10. contract `exact_files`、R5 functional seals、R5 failure/quarantine seals、DC/库/SDC/docs359 身份是否完全闭合？R5 是否永久 `FAILED_OR_INCOMPLETE_DO_NOT_CITE`，无任何旧 QoR 回填路径？
11. 做 shell 的 set-e、trap、monitor/child rc、signal、空数组、PID 消失/重用、首次与轴间 preflight failure 的边界审计。任何可能把 runtime latch 写成普通 child failure、或让已消费 attempt 不 quarantine 的路径，至少 P1。

## 裁决门

- 只有 `P0=0 && P1=0` 才可建议主 agent 另建独立 launch admission；本 hammer 自身永不授权运行。
- 必须给 0–100 分、P0/P1/P2 数量、逐项证据与 exact SHA，并双封结果目录。
- 任何 P0/P1 都返回 author 修复；不得自行运行工具验证。
- claim boundary 固定：`dc=false`、`ppa=false`、`hold_closed=false`、`power=false`、`energy=false`、`system_speedup=false`、`headline=false`。
- `docs/359` 必须保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
