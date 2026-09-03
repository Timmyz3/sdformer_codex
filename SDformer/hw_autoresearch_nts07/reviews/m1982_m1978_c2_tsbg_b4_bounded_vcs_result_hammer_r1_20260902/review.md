# M1982｜M1978 TSBG B4 VCS 结果独立 hammer

日期：2026-09-02

裁决：**结果准入 FAIL；P0=0，P1=1，P2=0。原始功能执行证据保留，只允许 additive PASS 格式修复。**

## 已通过的门

M1978 结果目录与 attempt 目录均通过内层 manifest 和外层 seal；M1979 runner review、M1980 release、M1981 release audit 的 SHA 身份链一致。静态 runner 与已封 receipt 均表明一次 license query、一次 VCS compile、一次 simv，自动重试为 false。compile/sim 日志中没有 `Error-`、`Fatal:`、`$error`、`$fatal`、原生 SVA failure、`SVAA-RNF` 或被忽略/未知的 `global_finish_maxfail`。

十个 phase 的 begin/complete 均各出现一次；load 为 52 begin、52 complete、0 timeout。正常 `$finish` 和唯一 PASS token 均存在。TSBG 的 11 项 SVA cover 全部非零；base 侧只有本来不应注入攻击的 stale/reset-recovery cover 为零。

TB 在 PASS 前以 `$fatal` 封闭检查了：48 rows、576 issues、9,216 products、24 commits、baseline/TSBG 576/144 bundle beats、4,608/1,152 scalar responses、精确 LRU4、算术逐 lane 0 mismatch、独立 bank stall/reorder、`-(-128)=+128`、retired-identity replay、bogus stale response、两次 reset recovery 和一次恢复后合法服务。它也执行了 directed local ratio `>=1.15` 的门，但没有输出可引用的精确 ratio。

因此第一性原理结论是：**M1978 的 behavioral 功能执行确实跑通内部全部门；不是 RTL、SVA、load handshake 或 scoreboard 失败。**

## 唯一 P1：终端账本不可机器解析

TB 第 834 行把四个 format string 用逗号作为四个 `$display` 参数。VCS 因此把后续两个字符串当成数值参数：

- `rows=<巨型十进制>` 解码后实际是字符串 `bundles_base=%0d ... scalar_tsbg=%0d`；
- `issues=<巨型十进制>` 解码后实际是字符串 `stale=%0d ... recovery=%0d`；
- `products=48`、`commits=576` 已分别错位成 rows 和 issues；
- `9216 24 576 144 4608 1152 1 1 0 2 1` 只作为无标签尾数输出。

现有 runner 只要求唯一 PASS token，并没有解析上述 terminal ledger。内部 fatal gate 可以证明执行正确，却不能把这个损坏的 PASS 行升级成独立、机器可解析的结果回执。因此 M1978 继续保持 raw evidence，不能成为 admitted VCS receipt，更不能引用其标签字段、精确局部周期或任何 speedup/area/timing/power/system/headline claim。

## 最小修复

只创建 additive TB successor，把 PASS 改为**单一连续 format string**，所有数值作为其后参数。DUT、SVA、scoreboard、攻击、watchdog、expected ledger、load handshake 和 claim boundary 全部不改。修复后仍需新的 source hammer、one-shot release/audit、一次新 VCS 和新的 result hammer；不得修改或重试 M1978。

本审阅没有启动任何 EDA；`docs/359` SHA 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
