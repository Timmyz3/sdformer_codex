# M800：M519 R15 K8 TIM-209 三轴 DC 失败审计

结论：**失败原因闭合，PASS_FAILURE_AUDIT 100/100；P0/P1/P2 = 1/0/0。** 这里的 100 分只评价失败审计完整度，不是架构或论文分数。R15 attempt 已消费；K1 子点虽完成并产出三件映射工件，但 K8 在 `compile_ultra` 前触发 `TIM-209=1`、退出 36，K1x8 未启动。因此三轴 campaign 及其中 K1 均不可引用。只授权一个新身份的 M519 R16 source 修复，不授权立即运行。

## 身份、封存与生产边界

- quarantine 的 `SHA256SUMS.seal.sha256` 与 92 项内层 manifest 全部通过；attempt 与 K1/K8/license 三份 preflight 也全部双封通过。
- `input_sha256.txt` 的 31 项 live replay 全过。runner、contract、release 分别仍为 `9ad15627...`、`cdb74d...`、`278eb851...`，并与输入账本一致。
- attempt 明确为 `CONSUMED_AT_FIRST_DC_LAUNCH`；canonical production 目录不存在，quarantine 根没有 `RUN_COMPLETE.txt`，只有 `FAILED_OR_INCOMPLETE_DO_NOT_CITE`。
- `docs/359` 保持 `dedde7ce...`。

## K1 是完整 raw 子点，但不是论文结果

K1 `dc.rc=0`、runtime monitor `rc=0`，`TIM-209=0`、`OPT-150=0`，确实执行一次 `compile_ultra` 并原子发布 mapped Verilog、mapped SDC、DDC。原始观察为：cell area `124620.173180 µm²`，153287 leaf cells，setup WNS/TNS `0.00/0.00 ns`、0 条 setup violation、0 条 DRC net violation；但 hold 最差 `-0.02 ns`、29160 条 hold violation，且 0 macro、`paper_ppa_ready=false`。

这只证明 R15 同一次 attempt 内的 K1 child 字节完整。release 明确要求 K1/K8/K1x8 三轴同 attempt 全部完成，禁止跨 attempt 拼表；所以 K1 只能用于复现性调试，不能成为 Table-A production 行。

## K8 失败事实

- K8 `dc.rc=36`，`TCL_EXPLICIT_FAILURE=FAIL_PRECOMPILE_LOOP__EXPLICIT_EXIT36`。
- precompile gate 为 `TIM-209=1`、`OPT-150=0`、`NO_COMPILE`。日志没有 mapping-optimization 执行标记，没有 mapped Verilog/SDC/DDC；仅有 precompile SVF，不能冒充映射工件。
- K1 与 K8 的 runtime final gate 都 PASS，external collision 为 none，campaign identity mismatch、cgroup OOM/failcnt 均为 0。因此资源、外部碰撞、license 都不是根因。
- K1x8 目录不存在，未启动。

## 最小组合反馈根因

DC 报出的 loop 同时穿过 `g_k8.implementation/core/g_k8.service`（66 个 arc mention）与 `g_k8.implementation/memory_adapter`（20 个）。冻结 RTL 给出闭环：

1. M218 service 的 `mem_req_valid` 受 `!protocol_error` 组合门控（274–278 行）。
2. M490 的 `illegal_request` 直接依赖该 `core_req_valid`（194 行），并进入 adapter `protocol_error`（218 行）。
3. M490 的 `core_rsp_valid` 又受 `!protocol_error` 组合门控（225–226 行）。
4. M519 K8 top 把 `core_rsp_valid` 接回 M218 的 `mem_rsp_valid`。
5. M218 的 `protocol_error` 包含 `mem_rsp_valid && !response_identity_legal`（315–320 行），闭环回到 `mem_req_valid`。

K1 采用 M499；它把 `response_channel_open` 与当拍 `illegal_request` 隔离，K1 的 `TIM-209=0`。这组正反对照支持“请求故障组合抑制响应完成”是最小根因。

两个近邻不是最小根因：top 的 `consistency_fault_now` 只合成到 top-level `protocol_error` 与寄存 sticky bit，没有反馈进 service/adapter channel gate；TIM-209 路径也不经过它。M490 的 same-edge slot reuse 则通过 `req_slot_open <- core_rsp_accept`，但上述 request-fault→response suppression→service fault 闭环不需要经过 `req_slot_open` 就已成立。slot reuse 是修后必须复核的次级边界，不是本次最小根因。

## R16 最小修复门

新身份允许做的最小修复是：clone 新的 R16 adapter/K8 top，让已拥有的合法 response 完成不再依赖当拍 request validity/illegality。可采用 M499 式双通道布尔边界：response channel 只受 sticky fault/illegal response 控制，request channel 再叠加 illegal request；`core_rsp_valid` 与 response ready 只用 response channel，request ready/forwarding 只用 request channel。允许保留 M490 的 same-cycle slot retirement/reuse，但这不是默认放行：必须先过 legal-cycle no-bubble regression 与“非法 request + 合法 completion 同拍”attack。任一失败就退回 registered/no-reuse 版本，并诚实计入延迟。若布尔隔离仍无法证明，也应加明确寄存边界。

禁止用 `set_disable_timing`、false path 或 TIM-209 waiver 隐藏真实环。禁止原地修改/重跑 R15，禁止把 quarantined K1 与后续 K8/K1x8 拼接。

R16 在 EDA 前必须新增 directed VCS/SVA：非法 request 与合法 completion 同拍、last-bank cut-through、same-slot retirement/reuse、response backpressure、stale response、sticky fault。独立静态审计还要证明 current request illegality 到 response completion 不可达。两关通过后，仍须以新 attempt 从零跑全 K1/K8/K1x8，三轴齐全且无 TIM-209/OPT-150 才能进入论文表。

本评审未调用任何 EDA/VCS/许可证，也未改现有源、合同、release、结果或 quarantine。
