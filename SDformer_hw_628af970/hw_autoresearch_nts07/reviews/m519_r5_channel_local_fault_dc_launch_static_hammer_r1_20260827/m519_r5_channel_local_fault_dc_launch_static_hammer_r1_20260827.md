# M519 R5 三轴 DC launch 独立静态准入 r1

## 裁决

**99/100，P0=0、P1=0、P2=1。** 准许在 caller 同时 exact-pin runner 与 admission 的前提下，启动恰好一次 K1→K8→K1x8、3.000 ns、TSMC28、logic-only DC campaign。本文未运行 DC/VCS，因而不准许声称 `TIM-209=0`、无组合环、面积、时序、功耗、能效、完整 FC2 或系统加速。

本轮在准入前打掉了三个 shell-only 阻塞：已封 VCS receipt review 的目录身份错误；raw `dc.log` 对 Tcl source echo 的假阳性；以及未冻结 SDC/双角库与 VCS 隔离范围不清的缺口。最终 runner SHA 为 `ec20959d83c7e3e7f027e9cf34792b73361871644348e8aabbaaca5904473519`。

## 已封功能前置条件

- R5 static review outer-seal 文件 SHA：`61ac10d46be82989aca702bae079510f872b50badad99926dd68e0972a68a8e9`。
- R5 VCS result outer-seal 文件 SHA：`6c180a8a5c97d5f05042a0534e68e179899c57e2e025db14ecf72eebced77286`。
- R5 receipt-blind review outer-seal 文件 SHA：`99cc43fe1fb86999eb329642ca5eee0066dea6d2b0a44723d6b783a38498d7d8`；评分 98/100、P0/P1=0/0，三个 VCS phase、12 类 attack 与冻结 r2 cycle rows 均通过。
- VCS 只证明 channel-local 功能修复和正常周期回归；它不证明 DC 中 `TIM-209=0`。

## 日志门核查

旧 gate 对整份 `dc.log` 搜索任意 `TIM-209|OPT-150`，会被 `dc_shell -f` 回显的 Tcl regexp token 假杀。最终 gate 仅拒绝行首 `Warning|Information` 且行尾诊断码为 `(TIM-209|OPT-150)` 的工具诊断，以及任意行首 `Error:`/`Fatal:`。机械 truth table 证明 Tcl source echo 不命中，而历史失败日志的
`Warning: timing loops detected. (TIM-209)` 与 `Information: Timing loop detected. (OPT-150)` 均命中。

这不是放宽真实环门。每个 ARCH_MODE 仍必须同时满足：

1. `precompile_loop_gate.rpt` 精确包含 `TIM-209=0`、`OPT-150=0` 与 PASS sentinel；
2. 不存在 Tcl explicit-failure sentinel；
3. `dc_shell` rc=0，runtime monitor rc=0，且 PASS-only terminal 存在；
4. setup/hold 无 `slack (VIOLATED)`，五类 constraint 均无 violation；
5. 七份核心报告和 mapped netlist 非空。

Tcl 的非零环分支写 failure sentinel 后 `exit 36`，`ungroup` 与全部 compile 命令仅在显式 PASS 分支中。K1、K8、K1x8 依次各走完整门，任一点失败即封存 quarantine，不能形成 canonical PASS。

## 身份、一次性与资源隔离

Admission/runner 会共同冻结 recovery contract、三份已封 review/result outer seal、Tcl、filelist、SDC、slow/fast DB、DC binary 与 docs/359。SDC 和两角 DB 在启动前均 exact-SHA 校验，并写入结果 `input_sha256.txt`。

Runner 在创建 attempt 前要求：无 canonical、无 attempt、无同 PID work/quarantine；全局无 `dc_shell`、`dc_shell-t`、`fm_shell`、`pt_shell`；当前 uid 无 `vcs`、`vcs1`、`vlogan`、`simv`；连续三次满足全局 commit/memory/swap 与 cgroup 门。这样不干扰或永久等待其他用户的 VCS，同时任何跨用户资源压力仍会被全局资源门和 runtime latch 捕获。Attempt 在第一次 `dc_shell` 调用之前原子落盘。运行期间每 10 秒监测资源，任一次失败都会锁存并使 campaign quarantine。

审阅时 canonical 与 attempt 均不存在；当前 uid 无 VCS、全局无 DC/FM/PT，但 commit headroom 低于 64 GiB 门。因此 admission 可以签发，runner 当前仍会在未消费 attempt 前以资源预检拒绝启动；必须等资源门自然满足，不得绕过。另一用户的长期 idle `simv` 不属于当前 uid，不应被杀死或作为永久阻塞。

## 唯一 P2

Immutable recovery contract 保留修复前 runner SHA `55b02f...`。这是有意保留的作者阶段 provenance：已封 receipt review 明确允许修 DC runner 后再做独立 admission。本文与 admission 仅以最终 SHA `ec2095...` 替代 DC runner 身份；RTL、Tcl、filelist、VCS 结果均未改变。

## 允许与禁止

允许：按 admission 中给出的两个 caller-pin 环境变量，在资源/隔离门满足后运行一次 runner。

禁止：运行第二次；覆盖 canonical/attempt；从本准入声称 DC PASS、`TIM-209=0`、PPA、吞吐/面积、功耗、能量、完整 FC2、系统倍速或 DATE headline；修改 docs/359。
