# M611｜M610 M579 production true-v4 contract + one-shot true release hammer

## 裁决

**PASS，100/100；P0/P1/P2 = 0/0/0。** 精确双封的 production contract 与 true release 通过 fresh independent read-only true-launch hammer。contract schema 是 frozen analyzer 接受的 `m579_paft_control_single_port_product_capture_execution_contract_v4`；授权闭合为 `launch_now=true / execution_release=true / run_cpu=true / max_attempts=1`，唯一冻结 invocation 使用 3 workers、80 formal records，GPU/EDA/remote 全 false。release 仍诚实标注 `still_not_executed=true`。

本评审只直接运行了 analyzer `--validate-contract-only`，没有调用 runner `--execute`，没有运行正式 80-record CPU、GPU、EDA 或 remote，也没有创建或修改 result/attempt/consumed/quarantine/PID staging。被审 M610/M609/M605/M601 与 `docs/359` 均未修改。

## 冻结身份与链

- production contract SHA `29a471dc489da4895e38b01700a4e101a5055bcbfd37323025a0762958011bb0`，member-sidecar-file SHA `1db3c402366074f3cc7bd578d744ed3463855d8c4a16a23abdba657750095405`，outer-seal-file SHA `faf54c358b28967b33823ef53c95c03293e20790f8248e7ef268882c55299d79`。
- true release SHA `b26bcb2ed9665e561ea84cad8038ff97f2406ac3b33be90538c88d4240c7c1f6`，member-sidecar-file SHA `a161846bac2ab117af9133e725c4724dd7cd848e81f7e3cd4cc5adf9e6913d0e`，outer-seal-file SHA `baa860bdcf6c9143348ff0f645a80b2ab893408f5ebec6ede5328645f32b5e52`。
- M601 source/candidate、M603 PASS100 manifest、M605 template/admission candidate、M609 PASS100 receipt/manifest、M610 contract/release/handoff 与 M611 request 的所有 member/outer seals 均独立重算通过。12 份链上 JSON 均通过拒绝 duplicate keys 与 non-finite constants 的 strict parse。
- contract 与 release 精确绑定 M601 → M603 → M605 → M609 → M610 单向 SHA 链；M603/M609 都是 100/100、P0/P1/P2=0/0/0。

## Production contract 与 validator

- contract `.inputs` 与冻结 M601 launch-now-false candidate、M605 nonproduction template 的 `.inputs` canonical object 完全相等，精确 15 keys；15 个 live path 均是非 symlink 普通文件且 SHA 匹配。
- analyzer/runner SHA 分别为 `ba8fc032...b115195` / `8c0fcbea...ad53fe`；冻结 Python 3.10.16、NumPy 2.0.1 的 path/SHA/version 全匹配。M43/M504/M505 与 r1/r2/r3 依赖通过 15-input rehash 和 frozen validator。
- 允许的 analyzer validate-only 返回 `required_inputs_rehashed=15`、`packed_payloads_rehashed=80`、`formal_trace_records_processed=0`、`result_or_attempt_created=false`，rc=0。
- 冻结 coordinate 保持 chunk-major `sample_operator_row_chunk_partition`、anchor `[0,47,94,141]`、20,304 tasks/operator、末 chunk 56 rows、DMA 160、tail 2、commit 96,000/sample、8 output blocks。六个 canonical result/attempt/consumed/quarantine/PID-staging 坐标精确、唯一且同父。

## One-shot 状态机

- validate-only 前后，result、attempt、consumed 在 `os.path.lexists` 口径下均 absent；quarantine staging/final 与 PID staging 前缀命中均为零。
- runner 在首次 canonical mutation 前安装 cleanup trap；result/attempt/consumed/staging/quarantine 坐标使用 lexists/no-symlink guard。失败路径把 attempt/staging 搬到同父 quarantine，生成 member/outer seals，再以 `RENAME_NOREPLACE` 发布。
- 成功路径在发布前重跑相同 15-input/80-payload terminal rehash，再生成 result member/outer seals，以 `RENAME_NOREPLACE` 发布 result；attempt completion 同样双封后以 `RENAME_NOREPLACE` 变为 `.attempt.consumed`。
- runner 在 attempt mkdir 前要求 consumed absent；一次成功 attempt 产生 consumed 后，第二次 invocation 会在 attempt 前被拒绝。

## 资源门与披露边界

M610 作者三次样本间隔均为 2 秒。重算最小值：commit headroom `83,647,820 KiB`（48-GiB 门 `50,331,648`）、MemAvailable `416,265,720 KiB`（128-GiB 门 `134,217,728`）、SwapFree `57,212,156 KiB`（32-GiB 门 `33,554,432`）；session/user cgroup 的 failcnt/under_oom/oom_kill 全零，UID-local collision 全零。作者快照通过，但不是当前 launch admission。

冻结 runner 源码没有 memory、cgroup 或 collision gate；release 已如实声明这一点。root 必须在实际 invocation 紧前重新做 3×2 s live resource/cgroup/UID-local collision 检查，并运行 exact runner `--preflight-only`。任何 live gate 不通过都不得执行。

accuracy 披露与 M255 一致：valid825 单 seed PAFT +0.5730215096601543%；十帧 5 win/5 loss；完整 64 帧 `zurich_city_09_a` PAFT 退化 1.0189020311889285%；无 multi-seed significance、无 same-evaluator-runtime 双臂绑定、无 Pareto。M528 容量 ledger 精确 9 rows，213,376 B / 245,760 B，margin 32,384 B；integrated macro PPA/energy 仍 open。formal CPU result、RTL/VCS/PPA/energy/system-speedup/headline 与 ratio multiplication 均不准入。

## 授权边界

本 PASS 只建议 root：先重验 exact M610/M611 双封与 canonical absence，再做 fresh live resource/cgroup/collision check 和 exact runner preflight；全部通过后，最多执行一次 release 中冻结的 3-worker invocation。raw result 即使产生也不可引用，必须经过 fresh independent result hammer。

`docs/359` SHA 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
