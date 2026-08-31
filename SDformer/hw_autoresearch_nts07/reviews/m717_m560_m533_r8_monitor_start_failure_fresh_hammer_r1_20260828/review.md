# M717｜M560/M533 r8 runtime-monitor-start failure fresh hammer

日期：2026-08-28  
模式：receipt-blind、只读；未调用作者 runner、VCS 或 `simv`。  
裁决：**失败回执 ADMIT；functional VCS / RTL 功能无结论；r8 身份永久 consumed；允许修复后新开唯一 r9 身份。**

## 1. 结论

`results/m560_m533_m528_dead_write_only_1rw_vcs_r6_20260828` 是完整、双封、fail-closed 的失败结果。独立复核确认：

- phase：`runtime_monitor_start`；
- runner exit：`1`；
- child：`not_started`；
- monitor status：`cleanup_wait_rc_1`；
- failure message：`monitor startup`；
- preflight cleanup：`0`；
- `functional_vcs_only/speedup/PPA/energy/headline` 全为 false。

失败发生在 VCS compile 之前。结果目录中没有 `compile.log`、`sim.log`、`simv`、`resource_runtime.log`、`RESOURCE_HEARTBEAT`、final request/ack 或 violation；所以不能从本身份得出任何 RTL 功能结论。

## 2. 双封与 inventory

结果 member manifest 含 7 个成员，全部 SHA 通过：

```text
ARTIFACT_INVENTORY.json
FAILED_DO_NOT_CITE
RUN_FAILED_OR_INCOMPLETE.json
collision_final.json
collision_initial.json
collision_postmkdir.json
resource_prelaunch.log
```

- `SHA256SUMS` SHA：`6061f952794dd8e30b734e123566a2b58aa6fd017f86821af6ca114c505e1d91`；
- outer-seal file SHA：`5a3f607edf6d0021b4e45ef8eb941465dd45ffe4b145549465b1888ed472eb4b`；
- member/seal 验证：全部通过。

独立重建 inventory 后，非 terminal artifact 恰好只有 4 个普通文件：三次 collision JSON 与 `resource_prelaunch.log`。路径集、文件类型、bytes 和 SHA 与 `ARTIFACT_INVENTORY.json` 全部一致，无 symlink、目录或未封额外成员。

三次 collision scan 都为 `PASS`、`matches=[]`，runner PID 一致，scanner PID 三个不同。prelaunch 恰有 3 个样本，failcnt、under_oom、oom_kill 全为 0。说明 preflight 与 atomic result publication 已通过，失败确实位于其后的 monitor startup。

## 3. 精确根因

冻结 runner SHA 为：

```text
176c14d35bf170f75b3097d832b2a39cd97ef7869263c1a0a019d99af0f8746e
```

runner 启用了 `set -euo pipefail`，第 691–693 行是：

```bash
resource_monitor() {
  local output=$1 violation=$2 heartbeat=$3 request=$4 ack=$5 seq=0 tmp="${heartbeat}.tmp.$$"
  : >"${output}"
```

在 Bash 中，同一条 `local` command 的所有 RHS 在该 command 的赋值生效前展开。因此展开 `tmp="${heartbeat}.tmp.$$"` 时，新 local `heartbeat=$3` 尚未建立；全局也没有 heartbeat。`set -u` 使后台 monitor 在执行第一条重定向和第一次 heartbeat 之前终止。

独立、隔离的 shell 微测试（没有调用 runner/VCS/simv）：

| 声明方式 | RC | 结果 |
|---|---:|---|
| 原同一条 `local ... heartbeat=$3 ... tmp="${heartbeat}..."` | 127 | `heartbeat: unbound variable` |
| 拆成两条声明 | 0 | 正常得到 `h.tmp.<pid>` |

runner 的控制流也吻合：`CURRENT_PHASE=runtime_monitor_start` → 后台启动 monitor → 等待 heartbeat → `require_monitor_live`；只有这些通过后才设置 `CURRENT_PHASE=vcs_compile` 并执行 VCS。回执的 `child_rc=not_started` 与 artifact 缺失共同证明 VCS/simv 未开始。

因此根因精确为：**`resource_monitor()` 的同一 `local` 声明在 `set -u` 下提前展开 `${heartbeat}`，不是 RTL、SVA、testbench、宏模型、license 或资源不足。**

## 4. P0 / P1 / P2 与评分

| Severity | 数量 | 内容 |
|---|---:|---|
| P0 | 0 | 无封印、身份、inventory 或虚假 PASS 问题。 |
| P1 | 1 | `M717-P1-01`：monitor 的同声明 local 展开 bug 在 VCS 前消费唯一 r8 attempt。 |
| P2 | 0 | 回执 claim boundary 已正确 fail-closed。 |

- **失败回执质量：100/100，ADMIT_FAILURE_RECEIPT。**
- **r8 functional admission：0/100，NO_CONCLUSION。**
- runner 缺陷严重度属于 P1，不是 RTL P1。

## 5. r8 是否还能重跑

不能。

release 明确 `max_attempts=1`，唯一消费点是 runner 的 atomic result `mkdir`；当前 exact result path 已存在并已双封为失败。runner 本身也在 preflight 后明确拒绝任何已存在的 result path。因此：

```text
r8 identity = PERMANENTLY_CONSUMED
```

禁止删除、覆盖、续跑、追加或复用该 result。`FAILED_DO_NOT_CITE` 必须原样保留。

## 6. 最小 r9 修复与是否可以新开

最小源修复只有两条 declaration：

```bash
resource_monitor() {
  local output=$1 violation=$2 heartbeat=$3 request=$4 ack=$5 seq=0
  local tmp="${heartbeat}.tmp.$$"
```

第二条执行时 heartbeat 已绑定，因此在 `set -u` 下安全。无需改 RTL、SVA、TB、macro adapter 或 binding plan。

**可以新开且只开一个 r9 唯一身份**，但必须满足：

1. 新 runner 路径与新 SHA；
2. 新 source-only contract；
3. fresh source-static、candidate、candidate hammer、launch release、final hammer；
4. 新 result path，绝不复用 r8；
5. source-static 必须包含本次两组 `bash -u` micro-test，且复核 monitor 真能产出首个 heartbeat；
6. live collision/resource gate 重新执行；
7. 仍只授权一次 VCS、一次 simv；失败同样终端双封。

如果 r9 只作上述 shell 修复，功能 RTL 可以继续绑定现有 frozen SHA；若改任何 RTL/TB/SVA/macro，则必须另行扩展评审范围。

## 7. Claim boundary

本 fresh hammer 只准入：

- r8 failure receipt 的真实性与完整性；
- failure phase、child/runner/monitor status；
- exact shell root cause；
- r8 permanent consumption；
- conditional permission to author one new r9 identity。

不准入 functional VCS、RTL correctness、trace recurrence、cycle、speedup、PPA、energy 或论文 headline。

`docs/359` 未修改，SHA 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

