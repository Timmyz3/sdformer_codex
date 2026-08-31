# M553 / M519 R8 DC launch-admission candidate 作者交接 r1

日期：2026-08-27  
状态：`CANDIDATE_SOURCE_ONLY_COMPLETE__LAUNCH_NOW_FALSE__NO_DC_AUTHORIZED`

## 交付结果

本轮只创建并双封 M519 R8 三轴 setup/area-only DC 的 `launch_now=false` 候选准入、作者交接和 fresh candidate-hammer request。没有运行 runner、DC、VCS、PT、PTPX、Formality、CPU/GPU 大任务或远端任务；没有创建 R8 result 或 attempt identity；没有创建 `launch_now=true` 最终释放。

- 候选：`contracts/m553_m519_r8_setup_area_three_axis_dc_launch_admission_candidate_r1_20260827.json`
- 候选 SHA256：`43e601df0d20754d5e7f65033b0958c42f2dd0b99b4abe3336d051bf22f7ad59`
- 候选 outer-seal-file SHA256：`81b2176b8cf121241a82ff9e421cf4565ad463d0f6a5580af5db67fd70ccba9f`
- `launch_now=false`，`authorization.run_dc=false`，`authorization.max_attempts=0`。

## 冻结闭包

候选以 runner 实际闭键集冻结 36 个 identity key，并冻结 recovery contract 的 17/17 `exact_files`。R8 runner/Tcl/contract、`dc_shell` 入口、`snps_shell` wrapper、实际 `common_shell_exec`、slow/fast DB、SDC/filelist、全部 RTL 和 `docs/359` 当前字节均保持冻结身份。

候选同时绑定：

- M546 作者交接的 handoff/verdict、manifest 和 outer-seal-file SHA；
- M550 fresh PASS review 的 review.md/review.json、manifest 和 outer-seal-file SHA，其中 M550 outer-seal-file SHA 为 `a6fff5def6c655cdf6f32b2f33a8430eb485fc3eed18db0b35c4cb14fc35d585`；
- R5 五个永久失败/验证 basis 的实际 outer-seal-file SHA；
- R6 failed static review 和 R7 disqualified review 的状态、member/outer seal provenance；
- 唯一 canonical result 与 attempt sentinel，候选创建不消费 attempt。

## P2 与当前串行阻塞

M550 P2-1 没有被隐藏：外部 collision TSV 有完整 tuple，但 `descendant_identity_faults.log` 当前只记录 timestamp/sample/PID/status。该 fault 仍 fail-closed 并迫使结果失败/隔离。candidate hammer 与最终释放 hammer 必须明确复核这条边界；若实际发生 fault，post-run receipt review 必须能重建完整 tuple，否则结果维持 noncitable。

作者快照观察到 M518 matched-rank3 DC 的 `common_shell_exec` 仍活跃，M518 attempt sentinel 已存在而 canonical result 尚不存在。因此当前 M519 R8 明确禁止启动。后续即使完成释放链，也必须等待 M518 终态退出且资源连续稳定，再由 runner 自身逐轴执行 3×10 s 的 64 GiB CommitHeadroom、128 GiB MemAvailable、32 GiB SwapFree、零 cgroup/OOM 和零同 UID EDA collision 门。

## 唯一合法后续

1. fresh independent candidate hammer；
2. 只有 candidate hammer 通过后，另建双封 `launch_now=true` final release；
3. fresh independent final-release hammer；
4. 等待 M518 释放、同 UID EDA collision 清零且资源稳定；
5. 才可由 root 以 runner SHA 和 final-release SHA 双 pin 发起唯一一次 runner preflight/attempt。

任何 candidate 或 hammer 都不产生面积、时序、hold、功耗、能量、吞吐/面积、完整 FC2、系统加速或 headline 证据。`docs/359` 未修改，SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
