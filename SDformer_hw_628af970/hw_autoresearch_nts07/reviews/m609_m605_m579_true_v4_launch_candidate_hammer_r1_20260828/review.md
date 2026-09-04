# M609｜M605 M579 true-v4 non-runnable template / admission-candidate hammer

## 裁决

**PASS，100/100；P0/P1/P2 = 0/0/0。** M605 的两份对象当前都不可运行：template schema 故意不是 production true-v4，admission candidate 的权限严格为 `launch_now=false / run_cpu=false / max_attempts=0 / execution_release=false`。本 PASS 只允许 root 下一步同步编写 production true-v4 contract 与 true release；仍不授权 80-record CPU，true release 必须再经 fresh independent hammer。

本评审没有运行正式 80-record replay、GPU、EDA 或 remote，没有创建 production contract、result、attempt、consumed、quarantine 或 staging，也没有修改被审对象或 `docs/359`。

## 不可运行性实证

- template schema 为 `m605_m579_true_v4_execution_contract_template_candidate_v1`，不是 analyzer 唯一接受的 `m579_paft_control_single_port_product_capture_execution_contract_v4`。
- 对 frozen M601 runner 实际执行 `--execute --contract <M605 template> --workers 3`：先完成 lightweight preflight，`formal_trace_records_processed=0`，随后在 analyzer contract validation 报 `execution contract schema drift` 并以 rc=1 退出。
- schema gate 位于 runner attempt validation 与 canonical coordinate mutation 之前。攻击前后 production contract、result、attempt、consumed、quarantine 和 PID staging 在 `lexists` 口径下均为 absent。
- admission candidate 不被 runner 消费；其 closed authorization dictionary 还额外冻结 `create_result=false / create_attempt=false / run_gpu=false / run_eda=false / run_remote=false`，不存在当前 live release。

## 身份、输入与执行义务

- template/candidate 的 member sidecar、outer sidecar，M605 author handoff，M603 PASS100 与 M606 request 双封全部通过。
- M601 analyzer/runner/source-contract/source-candidate SHA 精确匹配；M603 manifest/outer 精确匹配。
- template `.inputs` 与 M601 launch-now-false candidate `.inputs` canonical JSON 完全相等，精确 15 keys；15 个 live path 均为普通文件且 SHA 匹配。
- future validator 义务保持 `15 required inputs + 80 packed payloads + 0 formal records before attempt`。
- M601 的 same-parent result/attempt/quarantine、`lexists`/no-symlink、terminal rehash、member/outer seal 与 `renameat2(RENAME_NOREPLACE)` 状态机未被 M605 修改或弱化。

## 资源、accuracy 与容量边界

- future one-shot 固定 3 workers / spawn；root live gate 固定 3×2 s、commit headroom ≥48 GiB、MemAvailable ≥128 GiB、SwapFree ≥32 GiB、clean cgroup、UID-local collision=0。
- candidate 诚实声明 `runner_enforces_memory_or_collision_policy=false`；author snapshot 不是 launch admission。
- M255 证据重新核对：valid825 单 seed PAFT +0.5730215096601543%，十帧 5 win/5 loss，完整 64 帧 PAFT **退化 1.0189020311889285%**；`accuracy_performance_pareto=false`。
- M528 candidate capacity ledger 为 9 rows、213,376 B / 245,760 B，margin 32,384 B；integrated macro PPA/energy 未准入。
- formal CPU result、RTL/VCS/PPA/energy/system-speedup/headline 与 ratio multiplication 全为 false。

## 授权边界

允许：root 另立 production true-v4 contract 与 true release，并双封后交给不同 fresh reviewer。

不允许：本 M609、M605 template/candidate 或 author snapshot 启动 CPU；不允许创建 production result/attempt；不允许将任何预期周期或 PAFT 精度包装为已准入 Pareto。

`docs/359` SHA 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
