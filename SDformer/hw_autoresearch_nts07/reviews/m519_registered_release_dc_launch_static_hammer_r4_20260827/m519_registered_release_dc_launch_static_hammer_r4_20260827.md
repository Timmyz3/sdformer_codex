# M519 registered-release 三轴 DC launch 独立静态打铁 r4

日期：2026-08-27  
结论：`STATIC_GO__AUTHORIZED_ONE_M519_DC_ATTEMPT__RECEIPT_BLIND_REVIEW_REQUIRED`  
评分：**96/100**  
P0：**0**；P1：**3**；P2：**3**

本审阅没有运行 runner、DC、VCS、simv、Formality、PT、PTPX 或开源 RTL/EDA 工具，也没有
修改作者 RTL/Tcl/runner、既有结果/review 或 `docs/359`。唯一新建的授权对象是独立 launch
admission r2；它与本 review 均已双封。

## 裁决

M519 的最终 launch chain 可以授权**一次且仅一次**三点 logic-only DC。调用者必须同时 pin：

```text
runner SHA256    = 7d4049dbf21ea6850776ca47b66634da996600fd98c7b6f09e6762aba033278a
admission SHA256 = d43e1997841fb4df494a8717a65f782bd20fd848117de12ca3586884d871508f
```

授权只允许 runner 固定的 K1 → K8 → K1x8 三点，不允许重定向 canonical path。attempt marker
在首次 `dc_shell` 启动前原子发布；一旦工具启动，无论后续成功、NO-GO 或失败，该 identity
的 attempt 均永久消费。任何失败只准进 quarantine，三点全部闭合且最终 receipt/manifest
双封后才原子发布 canonical。

## 身份和路径修复

请求中的 runner、Tcl、contract、VCS receipt、VCS result outer seal、prior static outer seal、
receipt-blind VCS review outer seal、M496 failure outer seal 和 `docs/359` 九个 SHA 均由当前文件
重算匹配。四套既有证据的 inner manifest 与 outer seal 全部通过，JSON strict parse 且数值
finite。

最终 runner 对旧的不存在目录
`reviews/m519_registered_release_vcs_hammer_r2_20260827/` 的 literal 引用为 **0**；唯一实际
目录变量指向已双封的
`reviews/m519_registered_release_vcs_receipt_hammer_r2_20260827/`，并在 launch 前、每点前后和
最终发布前反复验证其 outer-seal file SHA。不存在复制 review 或改名绕过。

r3 recovery contract 中仍保留了 DC 之前的旧 runner/future-review forecast；该合同本身明确
`run_dc=false`，不是本次 launch authority。本 review 和 admission r2 只覆盖最终 launch
identity，不重写冻结的历史合同。

## registered-release 断环重建

M496 的失败环是：

```text
M219 mem_rsp_valid
 -> legal_response_accept
 -> 当拍 slot/context release
 -> mem_req_valid
 -> M499 illegal_request/protocol_error
 -> core_rsp_valid
 -> mem_rsp_valid
```

M519 三点分别切断该反馈：

- K1 的 `m519_fc2_k1_registered_release_service_island` 搜索 free slot 只看注册态
  `!sb_valid_q`，context open 只看注册态 `!ctx_busy_q`；本拍 response 在时钟沿清除，下一拍
  才能复用。
- K8 仍使用 M218 的同拍 service 旁路，但 M490 已把 slot-open 从 `illegal_request` 中移除；
  slot availability 只做 flow control，不能再通过 `protocol_error` 反馈到 `core_rsp_valid`。
- K1x8 是八个 M519 registered-release scalar service 直达八个外部 bank endpoint，不经过失败的
  M219→M499 内环。

封存的 M519 VCS 已在新 identity 下重测，非零四行聚合周期为 K1/K8/K1x8 =
11718/1899/1931；这些只能作 component directed cycles。VCS 不证明组合图无环，最终无环仍
必须由本次每点 precompile `check_timing` 的 TIM-209/OPT-150 门确认。

## 三点公平性和 hard gates

三点使用同一个 top、同一 12-file filelist、同一 Tcl/SDC、同一 slow/fast TSMC28 DB、
`ssg0p9v125c`、3.000 ns、ideal clock、ZeroWireload、IO delay/transition/load/fanout/uncertainty。
唯一架构参数依次为 `ARCH_MODE=0/1/2`。每点都执行相同：

```text
ungroup -all -flatten
compile_ultra
compile_ultra -incremental
compile -incremental_mapping -only_hold_time
```

`check_timing` 位于 `ungroup` 和任何 compile 之前。它的报告中 TIM-209 或 OPT-150 非零立即
`error`；runner 同时对完整 `dc.log` 再拒绝任何 TIM-209/OPT-150、ELAB-312、Error/Fatal。
必须有三点 mapped netlist、最终 area/QoR/setup/hold、post-check、hierarchy/resource/reference/
port 报告，setup/hold 均非负，五类 constraint 各自 clean，才计算面积和 throughput/mm² 门。
即使 DC 三点 clean，只要 logic Pareto gate 失败，也只能封为 NO-GO，不能升成 headline。

## 资源、碰撞和发布

每点启动前连续三次检查：commit headroom ≥64 GiB、MemAvailable ≥128 GiB、SwapFree ≥32 GiB、
cgroup failcnt/under_oom/oom_kill 全零；同时拒绝 DC/FM/PT/VCS、同用户或活跃 simv、项目 CPU
DSE。运行期间每 10 秒采样并 latch cgroup OOM。输入身份在首次 preflight、每点前后和最终
发布前重验；caller 必须 pin runner 与 admission SHA。

## 非阻塞问题

P1：

1. r3 contract 的 final-DC 字段是历史 forecast，最终引用必须以本 review/admission 为准。
2. OPT-150 只有在 precompile `check_timing` 已出现时才是 compile 前立即退出；若工具只在后续
   phase 发出，runner仍拒绝该点，但会等 `dc_shell` 退出。不得写成“任何阶段 OPT-150 都在
   compile 前发现”。
3. 动态 DC 输出没有预先 filename/symlink 白名单，receipt builder 也没有显式
   `allow_nan=false`；后续 receipt-blind review 必须核 exact topology、symlink confinement、
   strict finite JSON 和原始 reports。

P2：process gate 不是 host-global lock；final `mv -T` 前仍有极窄的外部 writer race；port
fairness 只自动比较 count 而非 normalized schema。它们不阻塞唯一一次受控 DC，但必须在
receipt review 复核资源日志、canonical topology 和三份 `ports.rpt`。

## Claim boundary

当前只准入：launch chain 静态正确、M519 directed VCS/component cycles、一次 logic-only DC
尝试。当前仍**不**准入 combinational-loop-free、DC measured、matched Pareto、完整 FC2、macro、
Formality、power/energy、paper PPA、system speedup 或 DATE headline。

`docs/359` 仍为：
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
