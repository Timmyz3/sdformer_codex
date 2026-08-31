# M591｜M590 / M559 PBR4 r6 repaired immutable CPU source author handoff

日期：2026-08-28  
状态：`R6_REPAIRED_SOURCE_ONLY__RUN_AUTHORIZED_FALSE__FRESH_STATIC_HAMMER_REQUIRED`

本交接响应 M588 对 M578 r5 的 `42/100, P0/P1/P2=3/2/1` 裁决。M578 冻结字节未覆盖；新增
M590 r6 analyzer、runner、source contract 与 future schema。没有运行正式 CPU/runner，没有读取真实 M511
或 decoder weight payload，没有创建 result/attempt、launch admission/authorization/wrapper，也没有运行
RTL、VCS、DC、PT、PTPX、Formality、训练、GPU 或远端。

## M588 P0 修复

- ready 改成冻结的 `low3 != 000`；terminal 从 bulk counter 改为显式 owner/state/committed-block bitmap/
  clear index+count+hash FSM，directory port 在 1024 个 word clear 期间保持独占。正式 preflight 用 production
  classes重放四个 resident-hit与两个 terminal golden，逐 event/cycle 比较；18/18/22/21 保持。
- descriptor identity显式携带 typed `numeric_activity=1` 与 `source_sign_bit=0`。candidate保留 signed-INT8
  `+1` 与 per-contributor Acc24 wrap；direct reference 使用独立 mmap、独立 offset/signed decode，并按原始
  source/kernel-order直接卷积，不调用 candidate `event_taps()` 或 `WeightSet.get()`。
- 每行补 source/descriptor/group/refill/psum/backing/output/directory/occupancy/capacity账本与执行守恒；
  696.24M raw、每架构 926.88M replay、11.04M dense commit和总1600行均是 production assertion。
  GO/support现在显式合取 exact、全部 conservation/common ledger、无 hidden resource/capacity、weight/refill、
  speed与OSG non-equivalence。

## M588 P1 修复

- shell 在 exec 前比较 analyzer exact SHA；analyzer再验证 execution/source/schema、四个阶段 review 的精确
  schema/status/100/0/0及 md/json/manifest/outer、wrapper path/PID/starttime/cmdline和输入双封。
- candidate/reference 两个 mmap 均在任何 result byte 写入前关闭。若 publish 后 final verify 或任一
  post-attempt edge异常，failure FSM会把 staging 或已出现的 canonical output重封并移动到唯一 quarantine，
  并断言 canonical output absent。

## 本 author 的轻量静态检查

- `/usr/bin/python3` 3.6.8 compile/AST PASS；`bash -n`、future schema `jq` PASS。
- 内置 synthetic ready/xorshift/Acc24/M523/terminal/四架构检查输出
  `PASS M590 M559 r6 repaired immutable analyzer static self-test`。
- 独立调用 production-bound six-golden preflight helper PASS。

这些都是 source/static 证据，不是正式 cycle/traffic 结果。下一门必须由未参与 authoring 的 reviewer 做
fresh read-only source hammer；P0/P1=0/0才可允许后续 launch-candidate review authoring，PASS仍不授权执行。

