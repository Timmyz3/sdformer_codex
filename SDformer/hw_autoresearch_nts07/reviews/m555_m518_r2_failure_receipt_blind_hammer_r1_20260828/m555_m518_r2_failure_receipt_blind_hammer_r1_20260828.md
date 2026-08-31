# M555｜M518 matched Fixed-T10/rank3 DC r2 失败回执盲审

日期：2026-08-28  
角色：independent receipt-blind、read-only failure hammer  
结论：**FAIL_MATCHED_DC__SEALED_QUARANTINE__DO_NOT_CITE**  
评分：**90/100；P0/P1/P2 = 0/3/2**

本审计没有调用 DC、VCS、Formality、PT、PTPX、runner、远端任务或大规模
CPU 任务，没有修改隔离结果、RTL、合同或 `docs/359`。

## 1. 一句话裁决

M518 r2 的身份链和失败封存是完整的，但结果不是 matched DC：Fixed 子进程
`rc=0` 并写出 Tcl terminal，运行期资源监视器却返回 `rc=1`，runner 因而以
`rc=42` fail closed；rank3 点完全没有启动。此次 one-shot attempt 已消费，
当前 `dc_completed=false`、`sta_completed=false`、`area=false`、
`throughput_per_area=false`、`paper_ppa_ready=false`。隔离目录中的 Fixed QoR
只能用于诊断，**不得引用、不得补写为 PASS、不得进入论文表格**。

## 2. 身份、封存和 inventory

审计对象：
`dc_handoff/runs/m518_matched_fixed_rank3_logic_only_dc_3p000ns_r2_20260827.failed_or_incomplete.1433205.quarantine`

- 隔离 manifest `SHA256SUMS`：
  `2fbb0300a05a40c7816c23fb4ecefb894d0492df76d4ed3017cf5ea8481b8116`；
  34 个 member 全部通过。
- manifest 外封文件 SHA256：
  `0583bb2ac0022965c7d8441a37ee92d830cef8525a4405ef4bcf73d252c44e31`；
  `SHA256SUMS.seal.sha256` 对 manifest 的检查通过。
- 目录共有 36 个文件，恰为 34 个被封 member 加 member manifest 和 outer
  seal；没有漏封普通成员。
- `input_sha256.txt` 中 runner、r2 contract、launch admission、Tcl、filelist、
  SDC、DC binary、slow/fast DB、两份 RTL 与 `docs/359` 的 12 个 live identity
  全部复核通过。
- r1/r2 static review、r2 launch-admission review、Fixed VCS result/review、
  rank3 VCS result/review 的 member manifest 与 outer seal 均递归通过。
- 隔离包中的 `contract.json` 和 `launch_admission.json` 分别为
  `18ae1c4...` 与 `08df8b2...`，与冻结 live identity 相同。

一次性 marker 存在于
`dc_handoff/runs/.m518_matched_fixed_rank3_logic_only_dc_r2_attempt_consumed`：

- `ATTEMPT_CONSUMED.txt` 时间为 `2026-08-27T23:36:48+08:00`；
- runner/contract/admission 三身份被写入并递归双封；
- attempt manifest SHA256 为
  `8b138f3cb40933de3fa88ae0a75722387b0fe6a966ce22917a31867683755c8e`；
- attempt outer-seal 文件 SHA256 为
  `1cec4b639327eae6386e4bc46f772b64cb91336ce9cddd0a74f4519d5cd42e71`；
- canonical result 路径不存在，attempt marker 存在，因此 r2 attempt 已消费，
  不能重用同一身份。

`RUN_FAILED_OR_INCOMPLETE.txt` 的独立重算一致：runner/child/monitor =
`42/0/1`，`signal=none`，`runtime_resource_latch=1`。

## 3. 运行序列和资源触发

三次 preflight 均在 `2026-08-27T23:36:48+08:00` 采样，最小值为：

| 项 | preflight 最小值 | 合同阈值 | 结论 |
|---|---:|---:|---|
| Commit headroom | 80,054,156 KiB | 67,108,864 KiB | PASS |
| MemAvailable | 415,455,080 KiB | 134,217,728 KiB | PASS |
| SwapFree | 57,218,812 KiB | 33,554,432 KiB | PASS |
| cgroup failcnt/under_oom/oom_kill | 0/0/0 | 0/0/0 | PASS |

Fixed 的 `resource_runtime.log` 有 **171 行，但只有 170 个资源快照**：169 个
`runtime`、1 个 `runtime_final`，最后一行是 `runtime_resource_latch=1`，不能
算作第 171 个样本。快照从 `23:36:48` 到 `00:05:01`：

| 项 | 独立复算 |
|---|---:|
| commit headroom <64 GiB | 16/170 个快照 |
| 首次 <64 GiB | 23:39:58，63,298,492 KiB |
| 最小 commit headroom | 56,818,268 KiB = 54.186123 GiB，23:47:39 |
| 最大连续 <64 GiB | 3 个 10 秒采样 |
| 最小 MemAvailable | 406,231,220 KiB = 387.412281 GiB，23:47:49 |
| 最小 SwapFree | 57,212,156 KiB = 54.561764 GiB |
| cgroup max failcnt/under_oom/oom_kill | 0/0/0 |
| runtime_final commit headroom | 76,357,448 KiB |

合同的任一 runtime 样本低于 64 GiB 即锁存失败；首次触发发生在 Fixed 开始约
190 秒后。monitor 没有立即终止子进程，而是在 Fixed 完成后返回 1；Fixed
DC log 给出的 elapsed 为 1686 秒，随后 runner 在任何 Fixed postcheck 和
rank3 launch 之前返回 42。该行为符合冻结 r2 的 fail-closed 合同，不能用
事后较宽阈值改判本次结果。

## 4. Fixed 中间产物：仅诊断，DO_NOT_CITE

Fixed 子进程确实 `rc=0`，`TCL_PASS_TERMINAL.txt` 为对应 design 且
`TIM-209=0`、`OPT-150=0`。隔离目录还包含 mapped Verilog/SDC/DDC/SVF 和
报告。下列数值只用于定位 runner 问题，**不是 admitted DC/STA/PPA**：

- total cell area：66,778.235814 um2；cell/sequential cell：79,768/10,573；
- combinational/noncombinational area：45,470.501474/21,307.734340 um2；
- setup/hold 首个 slack：+0.0008/+0.0000 ns，报告为 MET；
- QoR logic levels/critical path：98/2.77 ns；
- area report 的 macro/black box 数为 0；
- DC `port_count.txt` 为 **1175**，不是 50；
- Fixed 没有 runner `RUN_COMPLETE.txt` 或 structural-cleanliness receipt。

rank3 目录、log、rc、terminal、netlist、report 与 QoR **全部不存在**。因此
没有 Fixed/rank3 同运行面积比，没有 throughput/area，也没有 matched PPA。

## 5. 严重度发现

### P1-1｜matched 结果缺失

Fixed 被资源门拒绝且 rank3 未启动；只有一个隔离失败包，没有 canonical raw
result 或 author receipt。任何 matched area、timing、throughput/mm2、PPA
主张均为 false。

### P1-2｜冻结 runner 的 black-box grep 会误杀正常日志

runner 的 postcheck 对完整 `dc.log` 搜索任意 `black box` 字样。此次正常的
`PWR-24` 信息四次包含 `other than black boxes`，即使 runtime monitor 没有
失败，该宽泛 grep 也会返回 48。新身份必须匹配真实 unresolved/black-box
诊断，或以 area/reference/check-design 的结构化零计数为权威；不得删掉
black-box 检查本身。

### P1-3｜50 个 source tuple 与 1175 个 DC port object 被混为一谈

静态 parser 的 50 是顶层 direction-width-name declaration tuple 数；DC Tcl
的 `sizeof_collection [get_ports *]` 是 bit-level port object 数，本次实际为
1175。冻结 runner 又硬要求 `ports == 50`，因此即使资源与 black-box gate
修复也会返回 49。新身份应继续在 launch 前验证两份 RTL 的 50 个 tuple，
运行后只比较两点的 bit-level port count 相等，并把预期 bit count 单独冻结。

### P2-1｜“171 个样本”口径不准确

文件是 171 行，资源快照是 170 个。未来 receipt 必须分别报告 snapshots 和
terminal latch line，避免把 latch 元数据计入样本。

### P2-2｜preflight 三样本无时间间隔

三次 preflight 同秒、数值几乎相同，只证明三个连续读取调用通过，不能证明
稳定 64 GiB headroom。新身份可在预消费阶段加入有界间隔并保持失败不消费
attempt。

## 6. 新身份可采用的最小修复，不改变本次裁决

允许在**新 runner、contract、admission、result path 和 attempt sentinel**
中做以下收敛；必须重新独立 static review，不能覆盖本隔离包：

1. 将 Fixed 与 rank3 分成两个独立 one-shot point attempt；二者仍冻结同一
   RTL/Tcl/SDC/DB/clock/flattening 身份。只有两个 point 各自 canonical PASS
   且独立 receipt review 后，第三份 comparison receipt 才能计算 matched
   比值。这样保留公平性，同时避免一个已完成点因另一点未启动而消失。
2. 保留 launch 前 64 GiB 门，但让三次 preflight 有实际时间间隔。runtime
   使用有语义的 hard/soft 两级门：cgroup OOM/failcnt 或硬内存门立即失败；
   commit headroom 的软门要求连续低水位后终止并隔离。若选 64 GiB×3，本次
   trace 仍有三组连续 3 次低水位，仍会失败；不能声称换成“连续”即可救活。
   若改为更低阈值（例如 32 GiB×3），必须在新合同中给出保护理由和独立审阅，
   不能回溯本次。此次最低仍有 54.186 GiB、MemAvailable 387.412 GiB、cgroup
   全零，只能作为新策略设计证据。
3. 修复 P1-2/P1-3 两个确定的 postcheck 误杀点，并保留所有其他 structural、
   timing、constraint、macro 和 identity gate。

这些建议仅说明下一身份怎样少浪费一次性任务；它们不把本次 Fixed 中间 QoR
变成可引用结果，也不授权任何 EDA 运行。

## 7. Claim boundary

| Claim | 当前状态 |
|---|---|
| failure quarantine integrity | true |
| one-shot attempt consumed | true |
| Fixed child Tcl terminal exists | diagnostic only |
| Fixed admitted DC/STA/area | false |
| rank3 DC/STA/area | false |
| matched throughput/area | false |
| macro-inclusive PPA | false |
| power/energy | false |
| trained-rank3 accuracy | false |
| system speedup/headline | false |
| paper_ppa_ready | false |

`docs/359_DATE终局冻结_20260813.md` 最终 SHA256 仍为
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

