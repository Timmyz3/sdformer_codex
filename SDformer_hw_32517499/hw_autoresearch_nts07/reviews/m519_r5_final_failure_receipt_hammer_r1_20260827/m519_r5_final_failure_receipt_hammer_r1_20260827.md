# M519 R5 最终失败收据独立打铁审阅 r1

## 裁决

**93/100，P0=0、P1=2、P2=1。** R5 的失败/隔离语义正确且双封完整：runner 最终返回 42，DC 子进程返回 0，资源监控返回 1，未收到信号，`runtime_resource_latch=1`。因此整个 attempt 必须保持 `FAILED_OR_INCOMPLETE_DO_NOT_CITE`，不能把 K1 中间 QoR、网表或报告转成论文结果。

失败原因不是 DC 工具崩溃，而是资源合同错把启动前的 64 GiB commit reserve 原样复用为运行期 floor。完整 898 个时间样本显示，K1 运行期间 `CommitLimit-Committed_AS` 最低为 **40,696,172 KiB（38.810894 GiB）**；与此同时 `MemAvailable` 最低仍为 388.195 GiB、`SwapFree` 最低仍为 54.568 GiB，cgroup `failcnt/under_oom/oom_kill` 全程为 0。运行门在 19:56:29 首次低于 64 GiB 时已经不可逆锁存；R5 不能事后按新阈值重解释为 PASS。

另一个独立 P1 是 DC hold 策略病态：Tcl 连续执行 `compile_ultra`、`compile_ultra -incremental`、`compile -incremental_mapping -only_hold_time`，后两步各触发一轮 `min_path`。第一轮约 60 分钟从 min-delay cost -1533.30 探索到 -0.25、临时面积升至 176,383.0，随后几乎完整回滚到 cost -1530.80、面积 128,535.8；紧接的显式 hold-only 又花约 63 分钟重复修复并提交到 cost -0.01、面积 176,572.6。仅第二轮相对其入口的诊断面积膨胀 37.372%，相对首轮 `compile_ultra` 末值膨胀 41.689%。这些数值只用于审计流程病态，绝非可引用 PPA。

## 身份和封印

- quarantine：`dc_handoff/runs/m519_r5_channel_local_fault_three_axis_logic_only_dc_3p000ns_r1_20260827.failed_or_incomplete.4165439.quarantine/`。
- quarantine 内 `SHA256SUMS` 和 `SHA256SUMS.seal.sha256` 均独立 `sha256sum -c` 通过；其覆盖失败 sentinel、合同、输入身份、K1 日志/报告/网表、资源时序和返回码。
- attempt sentinel：`dc_handoff/runs/.m519_r5_channel_local_fault_dc_attempt_consumed/`，内外两级 SHA 校验通过，记录 `CONSUMED_AT_FIRST_DC_LAUNCH` 和 19:55:59 启动时间。
- 10 项 `input_sha256.txt` 全部与当前 runner、合同、admission、Tcl、filelist、SDC、dc_shell、两角库和 docs/359 实体匹配。
- live-prefix 审计 `reviews/m519_r5_runtime_gate_live_audit_r1_20260827/` 双封通过；它只证明前缀假杀，最终数字由本次完整 sealed quarantine 重算取代。
- 启动前资源日志 `.m519_r5_resource_preflight.4165439.log` 位于 quarantine 外且**没有独立 seal**。其三行与 runner 结构和第一个 sealed runtime 样本一致，足以辅助定位，但 R6 应把 preflight 复制进最终结果并封存。该缺口记 P2。
- docs/359 SHA256：`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`，未改变。

## RUN_FAILED 语义与三轴覆盖

失败 sentinel 的六项字段自洽：

| 字段 | 值 | 独立解释 |
|---|---:|---|
| status | `FAILED_OR_INCOMPLETE_DO_NOT_CITE` | 整个 attempt 不可引用 |
| runner_exit_code | 42 | runner 的 runtime-latch 专用返回码 |
| child_exit_code | 0 | 仅 K1 的 dc_shell 正常退出 |
| monitor_exit_code | 1 | 至少一次 runtime snapshot 未过门 |
| signal | none | 无 INT/TERM 来源 |
| runtime_resource_latch | 1 | 运行期失败不可逆锁存 |

实际只创建了 `k1/`。K1 有 `dc.rc=0`、`runtime_monitor.rc=1` 和 Tcl terminal；runner 在 `m519_run_point k1 0` 返回 42 后受 `set -e` 进入 quarantine，故 `m519_run_point k8 1` 与 `m519_run_point k1x8 2` 均未执行。不存在 K8/K1x8 目录、报告或 terminal，也不存在根级 `RUN_COMPLETE.txt`。

严格结论是：**K1 仅有被隔离的工具中间产物；K8、K1x8 缺失。** 不得把 K1 推断到另外两轴，不得计算三轴面积、吞吐/面积或能效对比。

## 完整资源时序复算

运行期文件共有 898 个 `timestamp=` 样本，覆盖 19:55:59 至 22:25:48（约 2:29:49），末尾另有 `runtime_resource_latch=1`：

| 指标 | 完整数值 |
|---|---:|
| preflight commit headroom（同秒三样本） | 72,314,884 KiB = 68.964848 GiB |
| runtime commit headroom 最小值 | **40,696,172 KiB = 38.810894 GiB**，21:56:34 |
| 首次低于 64 GiB | 66,207,320 KiB，19:56:29 |
| 低于 64 GiB | 817 / 898 样本 |
| 首次低于 48 GiB | 49,851,360 KiB，21:02:57 |
| 低于 48 GiB | 13 / 898 样本 |
| 低于 40 GiB | 1 / 898 样本 |
| `MemAvailable` 最小值 | 407,052,012 KiB = 388.195049 GiB |
| `SwapFree` 最小值 | 57,219,068 KiB = 54.568356 GiB |
| cgroup 任一非零样本 | 0 / 898 |

所以 live audit 的“48 GiB runtime floor”已被完整数据证伪：当前合法 K1 自身/全局并发组合曾低至 38.811 GiB。不能把 48 GiB 直接写进 R6。日志是全局 commit 会计，不能把从 68.965 GiB 到 38.811 GiB 的全部下降归因给 DC；但 false-kill 仍然成立，因为 runner 不做归因，而且物理内存、swap、cgroup 均未显示危险。

## hold 流程审计

Tcl 在理想时钟、ZeroWireload、0 macro 的逻辑综合里先设置 0.100 ns hold uncertainty 和 `set_fix_hold`，然后执行三步优化。日志中只有两次明确的 `Beginning Design Rule Fixing (min_path)`，不是三次：

1. 首个 `compile_ultra` 约 24:24 完成，作为后续 hold 膨胀的诊断基准。
2. `compile_ultra -incremental` 的第一轮 min-path 从 1:20 到 1:00:09。它曾把 cost 从 -1533.30 推到 -0.25、面积推到 176,383.0，却在结束前回滚到 -1530.80 和 128,535.8；即花费约一小时而没有保留 hold 修复。
3. 随后的 `compile -incremental_mapping -only_hold_time` 再从 -1530.84 开始，耗时 1:03:04，最终 cost -0.01、面积 176,572.6。之后 Tcl 才把 hold uncertainty 从 0.100 改为 0.090 ns，最终报告出现约 +0.0101 ns hold slack。

DC 会话总 elapsed 8,986 秒（2.50 小时），其中两轮 min-path 约占 2:03。最终网表含 66,921 个 buffer/inverter；但它是理想时钟、0 macro、失败隔离下的中间产物。该组合把“综合比较”变成“用大量标准单元修理 pre-CTS 理想时钟 hold”，对三轴公平 PPA 既昂贵又可能失真，判为病态流程。

## R6 联合准入建议（不构成运行授权）

### 资源门

1. **每个轴启动前重新准入**，而非整场只做一次：三次样本间隔 10 秒，均要求 commit headroom >=64 GiB、`MemAvailable >=128 GiB`、`SwapFree >=32 GiB`、cgroup 三项为 0，并且无外部 DC/FM/PT/VCS 碰撞。将三样本、runner PID 树和 `H0=min(headroom)` 复制到最终封存目录。
2. **运行期使用 emergency floor，不复用 prelaunch reserve**：本轮证据支持的临时下限只能设在观测最小值以下。建议首版采用 commit headroom `<32 GiB` 连续三次（30 秒）才锁存；`MemAvailable <128 GiB`、`SwapFree <32 GiB`、cgroup fail/under-oom/oom-kill 增加或新外部 EDA 碰撞仍立即/严格锁存。32 GiB 是恢复尝试的保守工程门，不是“已证明安全值”。
3. 每 10 秒记录 campaign 子孙 PID 的 `VmPeak/VmSize/VmRSS/VmSwap` 高水位、全局 headroom 相对 H0 的下降和外部碰撞；本轮先诊断，不把不精确的 `VmSize` 直接从全局 commit 中抵扣。
4. 每个轴结束后必须等资源恢复并重新满足三样本 prelaunch 门，才可启动下一轴；任一点失败就停止后续轴并双封 quarantine。R5 继续保持失败身份，不可回填。

### hold/综合策略

1. 论文主用的三轴 logic-only DC 应统一为 **setup/area flow**：移除 `set_fix_hold` 和显式 `compile -incremental_mapping -only_hold_time`，只运行一次 `compile_ultra`；只有当预先定义的 setup/area 改善门成立时才允许一次不带 hold-fix 的 incremental。三轴必须完全相同。
2. 将 hold 单列为非 headline 敏感性。真正的 hold closure 应在有布局/时钟树的 Synopsys P&R/CTS 后进行；pre-CTS ideal-clock logic-only 结果明确标注 `hold_not_closed_at_dc`，不能因此宣称 paper PPA ready。
3. 若工程上必须保留 DC hold sensitivity，只允许**一轮** bounded hold pass，并设置 wall-time 和面积膨胀 fail gate（建议相对 setup/area checkpoint <=10%）。失败只隔离 hold sensitivity，不得覆盖 setup/area checkpoint，更不得拿 41.7% 膨胀后的网表做 K1/K8/K1x8 headline 比较。

## 严格 claim boundary

- 本审阅确认的是“失败收据完整、失败原因可定位”，不是 DC/PPA PASS。
- R5 的 raw K1 area/timing/netlist、最终 +0.0101 ns hold、任何 cell count 均不可写入论文表格。
- K8/K1x8 未运行；三轴比较为空。
- R6 未创建、未授权、未运行。任何 R6 必须重新做 contract、static hammer、launch admission 和一次性 attempt 封存。
- `paper_ppa_ready=false`、`headline=false`、`system_speedup=false`。

## 严重度

- **P0：0。** quarantine 没有冒充成功，封印未破，docs/359 未变。
- **P1：2。** 64 GiB 运行期假杀；重复 min-path/理想时钟 hold 导致两小时与 37.4% 诊断面积税。
- **P2：1。** 外置 preflight 日志未被独立封存；R6 应纳入最终收据。

