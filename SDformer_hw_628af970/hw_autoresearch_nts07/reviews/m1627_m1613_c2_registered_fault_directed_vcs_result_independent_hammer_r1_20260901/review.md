# M1627｜M1613 C2 registered-fault VCS 结果独立 hammer

日期：2026-09-01

状态：`PASS_M1627_M1613_C2_REGISTERED_FAULT_DIRECTED_VCS_RESULT_HAMMER`

评分：99/100；P0=0，P1=0，P2=0。

## 裁决

M1613 已发布结果通过不同作者、纯读结果 hammer。内层 manifest 精确包含 95 个普通文件，外层 seal 一致；17 个目录与 VCS 生成的 2 条内部相对 symlink 均按路径和目标精确固定，目标也落在已封存成员内。

VCS compile 和 simv 的数值退出码均为 0，`runner.log` 精确记录一次 compile、一次 seed-1613 simv。编译日志只包含 M1609 successor RTL 与 M1613 TB，冻结 predecessor 未进入 filelist 或 compile 证据。仿真日志只有一个精确 PASS token，且达到正常 `$finish`：

`legal_terminal_no_false_pulse=1 legal_descriptor_accepts=1 illegal_header_latched=1 illegal_raw_latched=1 sticky_checks=3`

无 assertion failure、error、fatal 或 watchdog。attempt 回执内容精确为无自动重试；其 mtime 早于 compile 和 sim，runner 源码顺序也是先消耗 attempt，再启动 VCS，后启动 simv。

## 反向变异

CPython 3.6 和 3.12 均拒绝 24/24 类攻击，包括：extra flat/nested/empty 成员、特殊节点、新增 symlink、用 symlink 替换已封文件、已知 symlink 目标漂移、重复或乱序 manifest、重复 JSON key，以及 compile/sim 次数、seed、身份、PASS、RC、声称和日志注入。

## 论文边界

该回执证明的是 compactor-local registered-fault 有向功能：合法 terminal 不再产生假 public error，真非法 header/raw 仍被阻断并锁存 sticky fault。它不证明外层 error OR-chain，也不证明周期、加速比、面积、时序、功耗、能耗或论文 headline。本 hammer 没有启动 VCS/simv/EDA，`docs/359` SHA 仍为 `dedde7ce...`。
