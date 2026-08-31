# M557｜M518 r3 per-point setup/area DC source-only handoff

日期：2026-08-28  
状态：**SOURCE ONLY；fresh independent static hammer required；launch_now=false**

本包只修复 M555 锁定的 r2 流程缺陷，没有调用 DC、VCS、runner、远端或大
CPU 任务，也没有生成 admission、result、attempt 或 comparison receipt。

## 最小修复边界

1. Fixed 与 rank3 使用独立 canonical/result/attempt/quarantine 和未来独立
   one-shot admission。任一点失败不会消费或污染另一点。
2. 后置 paired comparison 只能在两点各自 canonical PASS、各自独立回执盲审
   P0/P1=0，并证明 source/filelist/SDC/Tcl/DB/clock/flow 完全相同后生成。本轮
   只有 schema，没有 comparison admission。
3. preflight 保持 64/128/32 GiB，三次间隔 10 秒；runtime 是新 r3 策略：
   commit headroom <48 GiB 连续三次或 <40 GiB 立即终止，MemAvailable <128
   GiB、SwapFree <32 GiB、cgroup/外部 EDA 碰撞立即终止。它不回改 r2 结论。
4. Tcl 用结构化 `check_design_ok`、`check_timing_ok`、DC bit-port=1175 和
   area report macro/black-box=0 作门，禁止扫描完整 dc.log 的宽泛 `black box`
   字样，因此正常 PWR-24 不再误杀。
5. 50 是有序源码 direction/width/name tuple；1175 是 DC 展开后的 bit-level
   port object。两种口径独立冻结，禁止互比。
6. 映射只做一次 `compile_ultra` 的 setup/area-only；没有 incremental、hold
   fix、hold-only mapping 或 hold report。未来 PASS 也不构成 hold/full-STA。
7. runner 冻结 wrapper 与实际 `common_shell_exec`，记录 PID/starttime/UID/
   parent/exe/NUL-safe cmdline；PID 复用不发信号；失败目录 member manifest+
   outer seal 双封，并要求同步 runtime-final gate acknowledgement。

## 冻结身份

- runner: `b9d87bd71642c06c7c34a1abc82f3d4531cbfb0c8b9cd3bf20beaad8db63de28`
- Tcl: `8f189fc861722f4d5e9005c9301cd01d4fd3c515f5942287df0c58e8e00119e6`
- contract: `f2120e245d88b7de68d999d34307e55e528bb38462359dc80d00c78a5f902ddd`
- contract outer-seal file: `bd1e620bd479fde802a5fe92b24dc92922136e494e01af8120f93896c639e16a`
- M555 outer-seal file: `3f06400f521a68a1b42ffb9315408b3bb0d00e01ae9a81eb459491125b5dcc9b`
- docs/359: `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`

## 静态评审门

fresh reviewer 必须保持零 EDA/零 runner，至少检查：bash syntax、JSON 和双封、
per-point 隔离、admission 缺失、result/attempt 缺失、PID actual-exec 捕获、
runtime 连续/立即门与 final gate、结构化 postcheck、50/1175 口径隔离、单次
compile_ultra/no-hold 流程，以及 paired schema 不允许事后拼不同约束结果。

任何 P0/P1 都保持 launch_now=false。即使 static PASS，也只能由 root 分别创建
Fixed 与 rank3 两份双封 launch admission；两点不得捆绑启动。
