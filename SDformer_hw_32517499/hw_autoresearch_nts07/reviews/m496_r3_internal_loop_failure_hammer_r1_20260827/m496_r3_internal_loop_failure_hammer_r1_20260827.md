# M496 r3 internal-loop failure receipt-blind 独立打铁

日期：2026-08-27  
结论：`PASS_FAILURE_ATTRIBUTION__RTL_HANDSHAKE_LOOP__M496_R3_PERMANENT_NO_GO__ONE_NEW_RTL_RECOVERY_CONDITIONAL`  
评分：**97/100**  
M496 r3 阻塞项：P0 **2**、P1 **3**  
VCS / DC / PT / Formality / DSE 新任务：**未运行**  
生产文件与 `docs/359`：**未修改**

## 裁决

M496 r3 不是第二次 host/OOM 失败。它的直接终止表现是 DC
V-2023.12-SP3 在 `compile -incremental_mapping -only_hold_time` 开始 timing update
时遇到 `Error id=263749` 和 `Fatal: Internal system error, cannot recover`；但根因证据早于
该命令：`check_timing_precompile.rpt` 在 `ungroup`、`compile_ultra` 和任何 mapping
之前已经报告 **两条 timing loop**。

两条 precompile GTECH 路径都跨越：

```text
g_k1.implementation/core/g_k1.service
  -> g_k1.implementation/memory_adapter
  -> g_k1.implementation/core
  -> g_k1.implementation/core/g_k1.service
```

同一组 service loop breakpoint 又在第一轮 `compile_ultra` 和
`compile_ultra -incremental` 的 timing update 中被 DC 发现并通过 `OPT-314` 临时断弧；
hold-only 阶段只是最后无法继续处理这个已存在的环，不是它内部优化新造了环。

因此归因是：**RTL ready/valid response→request 组合环为主因（0.93）；DC 对该环以 internal
fatal 而非干净报错退出是次要工具健壮性问题（0.06）；host/OOM 仅 0.01。**

exact r3 attempt 已消费且整个三轴 run 永久 NO-GO。允许设计一次新的 RTL 身份恢复，
但本审阅不直接授权运行它。恢复不得删 hold-only、断 timing arc、设 false path 或放宽任何
约束。

## receipt-blind 证据链

本结论没有使用 `RUN_FAILED_OR_INCOMPLETE.txt` 或 contract 内的 failure 描述来判断根因；
它们只用于核对 runner 的 fail-closed 状态。根因来自以下原始证据：

1. K1 `dc.log` 逐行显示 `check_timing`、两轮 compile、hold-only 命令边界和 fatal 栈；
2. `check_timing_precompile.rpt` 在 compile 前明确给出 TIM-209 与两条完整 cell-arc loop；
3. 590 个 runtime resource sample 和完整 meminfo 证明没有 cgroup OOM；
4. 源 RTL 静态依赖可重建与 GTECH 层级一致的 response/request 反馈链；
5. 必需的 final area/timing/constraint/netlist/receipt 均不存在，K8/K1x8 目录也不存在。

隔离目录的全文件树 fingerprint 为
`ad98e1aee985707e33676d1bb0e35c76fe568f16602fdd3b9accfccd922fb683`。
关键原始证据 SHA 另列于 `evidence_identity.sha256`。

## 资源与 host 排除

| 项 | 独立观察 |
|---|---:|
| K1 `dc.rc` | 1 |
| runtime monitor rc | 0 |
| runtime samples | 590 |
| min commit headroom | 37,540,292 KiB |
| min MemAvailable | 413,838,948 KiB |
| min SwapFree | 57,335,548 KiB |
| max cgroup failcnt / under_oom / oom_kill | 0 / 0 / 0 |
| `runtime_cgroup_oom_latched` | 0 |
| crash process VmSize / VmRSS | 4,817,700 / 2,865,672 KiB |

DC log 没有 `out of memory`、`failed allocating`、`cannot allocate`、SIGKILL 或
kernel OOM 文本。与 r2 的 8 KiB allocation failure 不同，r3 资源门通过，运行末端资源
仍充足，fatal 前紧邻的是 “Timing update failed because design has loops”。所以资源波动
不能解释这次退出。

## RTL 组合环重建

### M219 response 当拍释放又重发

`m219_fc2_k1_cropped_tagged_slice_service_island.sv` 中：

- `legal_response_accept` 由 `mem_rsp_valid && mem_rsp_ready` 与 response identity 形成；
- `free_slot_found` 把 `legal_response_accept && mem_rsp_slot == slot` 当作当前拍 free slot；
- `head_context_open` 把 `response_releases_head_context` 当作当前拍 context free；
- `mem_req_valid` 同时依赖 `free_slot_found` 与 `head_context_open`。

因此 M219 存在 `mem_rsp_valid → legal_response_accept → free/context release →
mem_req_valid` 的纯组合路径。

### M499 request fault 又反向控制 response valid

`m499_fc2_bundle_to_8bank_no_reuse_adapter.sv` 中：

- `illegal_request = core_req_valid && !req_payload_legal`；
- `protocol_error = fault_q || illegal_request || illegal_response`；
- `core_rsp_valid = complete_found && !protocol_error`。

顶层把 M219 `mem_req_valid` 直连为 adapter `core_req_valid`，并把 adapter
`core_rsp_valid` 直连回 M219 `mem_rsp_valid`。于是完整环为：

```text
M219 mem_rsp_valid
 -> legal_response_accept
 -> same-edge slot/context release
 -> M219 mem_req_valid
 -> M499 illegal_request/protocol_error
 -> M499 core_rsp_valid
 -> M219 mem_rsp_valid
```

M499 已有的 “no same-edge adapter slot reuse” 只消除了 adapter 自己的 registered slot
复用，没有消除 M219 scoreboard/context 的 response→request bypass，也没有切断
`illegal_request → protocol_error → core_rsp_valid`，所以原注释所称三层环仍有残余。

VCS/SVA PASS 证明在被测 two-state 时序和采样点上 transaction/arithmetic 正确，不等于
组合 timing graph 无环；一个逻辑上有稳定点的组合反馈可以通过 directed simulation，却仍
无法形成合法 STA 图。

## K1 中间面积和 hold 数字全部不可引用

DC 在 fatal 前打印过 optimizer 过程表，例如：

- 第一轮末尾 area `125282.7`；
- incremental 末尾 area `129182.3`；
- table worst-negative setup `0.00`；
- min-delay cost `-1533.26`；
- hold-only header leaf/sequential count `155319 / 31161`。

这些都不是合同要求的 `report_area`、`report_timing -delay_type min/max` 或五类
`report_constraint` 结果，而且计算图已经被 DC 通过 OPT-314 禁用 loop arcs。随后 DC
exit 1，以下全部缺失：

- `area.rpt`、`qor.rpt`；
- `timing_setup.rpt`、`timing_hold.rpt`；
- `constraint_violators.rpt`；
- postcompile `check_design/check_timing`；
- mapped Verilog、DDC；
- K1 `RUN_COMPLETE` 和 evidence manifest。

K8 与 K1x8 从未启动。因此 r3 没有 K1 PPA，更没有三轴面积比、throughput/mm² 或 Pareto
结果；禁止与 r2、旧 VCS cycle 或未来修复点拼表。

## P0

1. **precompile RTL handshake timing loop。** 两条 TIM-209 在任何 compile 前存在；带
   disabled loop arcs 的 timing/area 图不可准入。
2. **完整三轴证据为零。** K1 fatal、K8/K1x8 未启动，所有 required final reports/netlist/
   receipt 缺失；任何 r3 PPA 引用均为 fail-open。

## P1

1. runner 保存了 precompile check，但没有在 TIM-209 时立刻退出，浪费约 98 分钟后才在
   hold timing update fatal。恢复 runner 必须把 precompile loop 计数设为立即硬门。
2. 现有 VCS portability/equivalence gate 没有证明 combinational graph acyclic；新 VCS 应
   覆盖断环后的一拍 bubble、response/request backpressure 和无丢失，但最终无环仍由
   precompile DC check gate 证明。
3. Synopsys `stack_trace` 与 `crte` 文件存在并已由本审阅 fingerprint，但位于 quarantine
   外，未被原 failure package 自身封存。

## 唯一允许的受控恢复

exact M496 r3 不得重跑：attempt seal 已明确
`CONSUMED_AT_FIRST_DC_LAUNCH`，不能删除 marker、换目录或把新尝试继续叫 r3。

允许建立**一个新 RTL/contract/runner 身份**，首选最小断环是修改共享 M219：

1. `free_slot_found` 只接受 registered `!sb_valid_q[slot]`，删除
   `legal_response_accept && mem_rsp_slot == slot` 的当拍 slot reuse；
2. `head_context_open` 只接受 registered `!ctx_busy_q[...]`，删除
   `response_releases_head_context` 当拍 bypass；
3. accepted response 在时钟沿清除 scoreboard/context，下一拍才允许新 request。

它直接切断 `mem_rsp_valid → mem_req_valid`，不修改 payload、算术、SRAM 端口、协议合法性
或约束。代价是一拍 release/reissue bubble；因为 M219 同时被 K1/K1x8 使用，旧 cycle
比率全部失效，三点必须在新身份下重新测量，不能利用更慢 baseline 包装倍率。

在 full DC 前必须完成：

1. exact-SHA VCS/SVA，要求 transaction multiset、数值、result/done、protocol attack、
   response/request stalls 均 0 mismatch，并显式覆盖“response 接受后下一拍才能复用”；
2. runner 在 compile 前解析 `check_timing_precompile.rpt`，任何 TIM-209/OPT-150 立即退出；
3. 同一新 identity 下重测 K1/K8/K1x8 cycle；
4. full DC 仍使用原 3.000 ns SDC、slow/fast DB、ideal clock、ZeroWireload、点序、
   `ungroup -all -flatten`、`compile_ultra`、incremental 和 hold-only 命令；
5. 三点 final reports/netlists、setup/hold、五类 constraint、port count 全部通过后才计算比率。

禁止的“修复”：

- 删除 `compile -incremental_mapping -only_hold_time`；
- 对 loop arcs 加 `set_disable_timing`；
- false path、降低 clock/hold uncertainty 或其他约束放宽；
- 降 compile effort、仅对某点保 hierarchy；
- 复用 r3 K1 optimizer 中间数字；
- 新 M219 与旧 K1/K1x8 cycle 混算。

若新 identity 的 precompile 仍出现任意 TIM-209/OPT-150，或 VCS 等价、三个 DC point、原
logic Pareto 门任一失败，则 M496 three-axis 物理线永久 NO-GO，不再进行第三次结构修补。

本审阅只允许准备和独立审查上述新身份，不解锁实际 VCS/DC、Formality、SAIF/PTPX、
paper PPA、完整 FC2/FFN、系统倍速或 DATE headline。

`docs/359` SHA 未变：
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
