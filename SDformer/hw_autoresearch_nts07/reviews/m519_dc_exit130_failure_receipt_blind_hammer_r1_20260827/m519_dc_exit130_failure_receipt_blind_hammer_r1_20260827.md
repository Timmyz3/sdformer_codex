# M519 DC exit130 失败回执独立打铁 r1

日期：2026-08-27  
结论：`TRUE_K1_COMBINATIONAL_LOOP__TCL_GATE_FAILED_OPEN__ATTEMPT_CONSUMED_QUARANTINED_NO_PPA`  
评分：**98/100**（对失败证据完整度评分，不是对 M519 PPA 评分）  
P0：**2**；P1：**2**；P2：**2**

本审阅只读取唯一 attempt 的 quarantine、attempt marker、冻结 launch admission/static review 和
对应 RTL。没有运行 DC/VCS/PT/PTPX/Formality 或开源 EDA，没有修改作者 RTL、Tcl、runner、既有
结果或 `docs/359`。未封存的 quarantine 由本审阅的 exact-SHA fingerprint 间接钉死；若原始树
随后改变，本 review 的 `evidence_identity.sha256` 会立即失败。

## 一、裁决

M519 r1 的 K1 不是扫描假阳性，而是**真实结构组合环**。precompile `check_timing` 报
`TIM-209=1`，明确列出 14 段 GTECH cell arc，层级从
`core/g_k1.service` 进入 `memory_adapter`，经 `core` 再回到 service。RTL 可重建为：

```text
adapter core_rsp_valid
 -> service mem_rsp_valid
 -> service protocol_error
 -> service mem_req_valid
 -> adapter core_req_valid / illegal_request / protocol_error
 -> adapter core_rsp_valid
```

M519 的 registered-release 确实去掉了 response 当拍释放 slot/context 后立即复用的旧路径，但
没有切断 response/request 两个独立通道之间的组合 fault gating。正常 workload 可能满足
`core_req_valid -> payload_legal`，这只是 reachable-state 不变量，不能抹掉 RTL 组合图；封存 VCS
的两态 directed PASS 也不能替代无环证明。

当前 attempt 已在首次 `dc_shell` 启动时永久消费。最终 canonical path 不存在，只有 K1
quarantine；K8/K1x8 未启动，mapped netlist/area/QoR/point `RUN_COMPLETE` 均为 0。因此本次没有
任何可引用面积、时序、吞吐每面积或三轴 Pareto 数字。

## 二、两个 P0

### P0-1：K1 RTL 仍有真实组合环

原始 `check_timing_precompile.rpt` 的 TIM-209 与后续 `compile_ultra` 中的 OPT-150/OPT-314 指向
同一个 service↔adapter↔core 反馈。OPT-314 只是优化器临时断弧，不能让断弧后的中间图获得 PPA
资格。修复必须切断 RTL 的跨通道组合依赖，而不是把 TIM-209 加白名单或引用断弧后面积。

### P0-2：所谓 precompile hard gate 实际 fail-open

冻结 Tcl 在发现 `TIM-209=1` 后写入
`FAIL_PRECOMPILE_LOOP__NO_UNGROUP_OR_COMPILE` 并调用 `error`。但 Synopsys 顶层 `-f` 会话记录表明
该 `error` 没有终止后续脚本：日志随后实际执行了 `ungroup -all -flatten`，进入
`compile_ultra`，并在 mapping phase 才被中断。冻结 static review/admission 中“非零即在
ungroup/compile 前退出”的描述已被实测证伪，不能复用为下一 identity 的 launch authority。

下一版 Tcl 必须使用经静态审计的**显式非零进程退出**，并让昂贵命令处于 PASS 分支：例如关闭
报告文件后 `exit 36`，且 `ungroup/compile_*` 仅存在于 `else`/通过路径。runner 还必须在 DC
退出后要求一个只可能由 PASS 路径写出的 terminal sentinel。单纯再次调用 `error` 不准入。

## 三、中断和隔离边界

artifact 与“发现 fail-open 后约 15 分钟人工 Ctrl-C 止损”一致：attempt marker 时间为
15:23:12，`dc.log` 末尾出现 INT-7 和 `Process terminated by hangup`，quarantine 在 15:37:56
落盘，runner 的 EXIT trap 写 `runner_exit_code=130` 后将 work tree 原子移动到
`failed_or_incomplete...quarantine`。当前没有活动 DC 进程，canonical path 缺失，旧 identity
无法因 attempt marker 再次启动。

但原始树不能自证“由哪个操作者按下 Ctrl-C”，所以该原因只记为 operator-reported；artifact
能够独立证明的是 signal/hangup、exit130、attempt consumed 和 quarantine 成功。runner 在
`wait dc_pid` 期间被中断，未写出 `dc.rc`、`runtime_monitor.rc` 或 runtime-final latch；这不影响
`NO PPA` 裁决，却意味着本次只能封失败边界，不能宣称工具正常按 hard gate 返回。

资源不是主因。preflight 三点全部过门；89 个 runtime sample 中 cgroup failcnt/under_oom/
oom_kill 始终为 0，MemAvailable 最低仍约 395 GiB。commit headroom 一度降至 61,725,536 KiB，
低于 launch 的 64 GiB 门，但 runner 的 runtime policy只 latch cgroup OOM，且组合环在资源下降前
已经由 precompile 报出。

## 四、P1/P2 和修复准入

P1：

1. 冻结 launch admission/static review 的 hard-gate 语义已失效；保留它们作历史证据，但新
   attempt 必须使用新 Tcl、runner、contract、static review 和 launch admission identity。
2. runner 只有 EXIT quarantine trap，没有把 INT/TERM 的来源、child rc、monitor rc 和 final
   resource latch完整收口。下一版应显式 trap INT/TERM，先向 child 传播、等待 child/monitor，
   再写 signal provenance 与两个 rc，最后封失败树。

P2：

1. quarantine 自身没有 manifest/seal；本 review 已 pin 15 个 raw file，但下一 runner 应在失败
   child 全部退出后给 quarantine 本体生成 inner manifest + outer seal。
2. runtime commit headroom 可以低于 preflight threshold而 runner继续；这不是本次根因，但新
   runner 可在低于阈值连续 N 次时做受控停止并同样封失败回执。

新身份的最小准入顺序：

1. 先修 K1 跨通道 fault gating。优先采用 channel-local fault decoupling：非法 request 当拍
   0 accept/0 bank side effect并 sticky fault，但不能撤回独立 response channel 上已完成且 identity
   合法的旧 response；response accept 与 retirement 必须使用同一 enable。若无法证明，再用
   真注册 request slice，并把延迟/面积税完整计入 K1。
2. VCS/SVA 必须新增“合法 complete response 与 malformed request 同拍”、非法 bank response 与
   合法 request 同拍、pending drain 时攻击、backpressure+attack、sticky fault/reset、四级守恒；
   正常 B1/B2/B4/B8/zero/stall/numeric 全部重跑。
3. 修 Tcl fail-open并独立 static hammer：通过反向一位变异证明 TIM-209 非零时进程非零退出，且
   command trace 中 `ungroup/compile` 计数为 0。此类验证只准用新 identity，不重跑当前 attempt。
4. 新 DC 仍按 K1→K8→K1x8；每点 precompile TIM-209/OPT-150 均为 0且最终报告齐全，三点才允许
   计算 matched throughput/mm²。K1 clean 不外推 K8/K1x8。

停止条件：channel-local 修复若仍有环，只允许一次 registered-boundary fallback；fallback 再失败
或三轴同资源 Pareto NO-GO，则停止 M519 物理线，不再开第三种协议结构。

## 五、claim boundary

本 review 只准入：attempt 已消费、K1 存在真实结构组合环、Tcl gate fail-open、exit130 后隔离
成功、当前 identity 无 PPA，以及下一版修复/验证门。它不准入 M519 loop-free、K8/K1x8 clean、
DC area/timing、throughput/mm²、Formality、power/energy、完整 FC2/FFN、system speedup 或 DATE
headline。

`docs/359` SHA256 保持：
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

