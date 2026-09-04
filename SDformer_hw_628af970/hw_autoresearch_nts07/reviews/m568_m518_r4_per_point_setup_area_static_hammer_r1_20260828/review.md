# M568｜M518 r4 per-point setup/area DC source static hammer

日期：2026-08-28  
模式：fresh independent、source-only、read-only  
结论：**PASS_STATIC__POINT_ADMISSION_MAY_BE_AUTHORED_SEPARATELY__NO_LAUNCH_NOW**  
评分：**100/100；P0/P1/P2 = 0/0/0**

本审阅完整读取 M566 request/handoff、r4 runner/contract、M563 final 以及
M555/r2 冻结失败链；没有调用 DC、VCS、runner、远端或大 CPU 任务，没有创建
point admission、result、attempt 或 paired comparison，也没有修改 `docs/359`。

## 定向裁决

### M563 P1-1 已闭合：runtime-final 是第三个受门控样本

r4 runner 的普通 runtime 路径在 476--487 行更新/清零 `soft_bad`，并在
`soft_bad >= 3` 时拒绝。final 路径在 515--520 行用同一 48 GiB 阈值再次
更新或清零该计数，并在 523--527 行先执行 `<40 GiB` immediate，再执行
`soft_bad >= 3`。因此 ordinary-low、ordinary-low、final-low 会在 final 得到
`runtime_final_commit_below_48gib_three_consecutive`，而 final-high 会清零连续
计数；final 的 Mem/Swap/cgroup/collision 门也仍存在。final ACK 记录计数、
reason、latch 和 PASS/FAIL，runner 只接受 `PASS_FINAL_GATE_ACK`。

### M563 P1-2 已闭合：前序冻结链在任何 point 路径创建前验真

r4 runner 先校验 future point admission、contract/exact files、tool/DB，再在
170--180 行对 M555 review、r2 quarantine、r2 attempt 三个 live outer-seal
文件逐项比较 contract 冻结 SHA。`m518_r4_recursive_sealed_dir_ok` 对每个包内
`SHA256SUMS` 与 `SHA256SUMS.seal.sha256` 递归执行 `sha256sum -c`。任一不符
均以 rc=3 pre-attempt 退出。第一个路径创建是 305 行的 point preflight
`mkdir`；work/result 与 attempt marker 分别更晚。因此验真严格先于 preflight、
result/work 和 attempt 创建。三个 live 包当前均递归通过，outer SHA 分别为：

- M555 review：`3f06400f521a68a1b42ffb9315408b3bb0d00e01ae9a81eb459491125b5dcc9b`
- r2 quarantine：`0583bb2ac0022965c7d8441a37ee92d830cef8525a4405ef4bcf73d252c44e31`
- r2 attempt：`1cec4b639327eae6386e4bc46f772b64cb91336ce9cddd0a74f4519d5cd42e71`

## 冻结回归门

- runner `bash -n`、contract/request/handoff JSON、M566 request/handoff 双封均通过。
- runner/Tcl/contract/contract outer seal 与 `docs/359` 均命中冻结 SHA；contract
  的 7 个 exact files、DC wrapper/actual executable、slow/fast DB 全部命中。
- Fixed/rank3 为独立 canonical result、attempt、failure quarantine 和未来
  admission；point runner 不生成 paired comparison，paired schema 只允许两点
  各自 receipt review 后另行准入。
- 两个顶层的有序 source declaration tuple 均为 50 且逐项相同；DC bit-level
  port 口径独立冻结为 1175。
- Tcl 只有一次命令级 `compile_ultra`，无 incremental compile、hold fix、
  hold-only optimization 或 hold report；结果仅可称 setup/area-only。
- 结构化 postcheck 要求 `check_design=1`、`check_timing=1`、1175 bit ports、
  setup MET、四类 constraint clean、macro/black-box exact zero；不存在宽泛
  `dc.log` black-box grep。
- preflight 为 64/128/32 GiB 三样本、10 秒间隔；runtime 为 48 GiB 连续三次、
  40 GiB immediate，Mem/Swap/cgroup/collision immediate，且 final ACK 必须通过。
- exact child 绑定 PID/starttime/UID/parent/actual-exe/NUL-safe cmdline；PID reuse
  不发送信号；失败结果按 point 双封隔离，不消费另一点。
- 两份 point admission、两个 result、两个 attempt 和 paired admission 当前均
  不存在；r2 quarantine 与 attempt 保持冻结，r2 Fixed 中间 QoR 继续
  `DO_NOT_CITE`。

## 授权边界

本 static review 只允许 root **另行**为 Fixed 与 rank3 分别创建双封 one-shot
point admission；本包 `launch_now=false`，不授权运行。任何 raw point 仍须
独立 receipt review；paired throughput/area 只有两点都通过后才能由第三份
admission 生成。当前 DC/STA/area/power/energy/system-speedup/paper-PPA/headline
均为 false。

`docs/359_DATE终局冻结_20260813.md` SHA256：
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

