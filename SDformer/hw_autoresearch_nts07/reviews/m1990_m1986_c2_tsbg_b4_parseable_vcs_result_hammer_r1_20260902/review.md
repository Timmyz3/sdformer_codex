# M1990｜M1986 TSBG B4 可解析 VCS 结果独立 hammer

日期：2026-09-02

裁决：**PASS；99/100，P0=0，P1=0，P2=0。准入为组件级 directed behavioral VCS 功能证据。**

## 身份与一次性执行

M1986 结果、attempt、M1987 runner review、M1988 release、M1989 release audit 和 M1985 source review 均通过各自的双 seal。runner/review/release/audit SHA 链闭合，并绑定 M1982 的格式失败和 M1984 的唯一修复。实际编译 filelist 包含 M803 adapter、M1880 RTL、M1880 SVA 与 M1984 TB，top 为 `tb_m1880_c2_tsbg_b4_real_channel_signed_frontend`。

一次性账本为 1 次 license query、1 次 VCS compile、1 次 simv、0 retry。simv 以 `-assert global_finish_maxfail=1` 运行，并受 180 秒 wall timeout 保护。

## 精确机器回执

下列完整行逐字出现且只出现一次：

`PASS_M1880_C2_TSBG_B4_REAL_M803_TYPED_SIGNED_DIRECTED rows=48 issues=576 products=9216 commits=24 bundles_base=576 bundles_tsbg=144 scalar_base=4608 scalar_tsbg=1152 stale=1 retired_replay=1 replay_accept=0 reset=2 recovery=1`

PASS prefix 也只有一次，正常 `$finish` 一次。独立 parser 进一步拒绝 8/8 种变异：错误 rows、截断、额外 suffix、重复行、字段重排、缺字段、错误 stale 值和 M1978 旧式损坏行。

## 功能与覆盖

十个 phase 的 begin/complete 共 20 token，均各一次；load 为 52 begin、52 complete、0 timeout。日志没有 compile/sim error、fatal、`$error/$fatal`、原生 SVA failure、`SVAA-RNF`、SVA error、watchdog、directed timeout 或 ignored/unknown runtime maxfail。

PASS 前的 fatal-gated 检查覆盖逐 lane 算术 0 mismatch、work conservation、精确 LRU4、独立 bank stall/reorder、`-(-128)=+128`、retired legal identity replay、bogus stale response、两次 reset recovery、恢复后合法服务，以及 directed local ratio 至少 1.15 的门。TSBG 的 11 项 required SVA cover 全部非零；base 侧未注入攻击，因此 stale/reset-recovery cover 为预期 0。

## 准入边界

可以引用的只有：**在该 directed workload、精确 RTL/SVA/TB 身份上，M1880 TSBG-B4 与 M803 typed-signed 路径通过 VCS 功能、算术、协议攻击和恢复验证，并产生上述精确工作账本。**

该结果没有输出 exact cycle ratio，`same_area=false`，也没有 DC/PT/功耗/系统重放。因此不得声称精确加速比，不得把 CPU premodel 的 2.533808× 升级为 RTL 倍率，也不得推出同面积、面积、时序、hold、功耗、能耗、全系统或论文 headline 结论。

本 hammer 没有启动 EDA；`docs/359` SHA 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
