# M1611｜M1609 C2 registered-fault additive successor 独立审阅

日期：2026-09-01

状态：`PASS_M1611_M1609_ADDITIVE_SOURCE__GO_AUTHOR_EXCLUSIVE_FILELIST_AND_VCS_DC_SOURCES__NO_EXECUTION`

评分：98/100；P0=0，P1=1，P2=1。

## 裁决

M1609 source 与静态测试通过。允许下一步 author 新的二选一 filelist、VCS testbench/runner 和 matched DC runner 源码；本审阅不授权直接执行 VCS/DC，更不授权 PTPX。新 filelist/runner 必须先 exact-SHA 独立审阅，随后先 VCS，VCS PASS 后才可 DC。

## 唯一 executable delta

冻结 M214 SHA 为：

`e278da8b0deaa0dda07b0477930453daa40b0331399a3941b743d604d0b102a5`

M1609 successor SHA 为：

`7ee28b3912ae34c99c795a48e80be29df2b59b363e5de2d2b359175ec9dda931`

删去 additive identity comment，并将新 assignment 还原后，M1609 与冻结 M214 逐字节相同。唯一 executable diff 是：

```systemverilog
// frozen M214
assign protocol_error = fault_q || illegal_request;

// M1609
assign protocol_error = fault_q;
```

模块名、参数和完整 port contract 未变；冻结 M214 本体未改。

## 本地 fail-closed 语义没有丢

- `illegal_request` 的 header/raw 判定表达式保留；
- 非法 header 仍使 `header_ready=0`，因此 `header_accept=0`；
- 非法 raw 仍使 `raw_ready=0`，因此 `raw_accept=0`；
- `if (illegal_request) fault_q <= 1;` 保留，fault 为 sticky；
- reset 仍清零 `fault_q`。

语义差异只有 fault 的公开时序：非法请求出现的组合周期内仍被 ready/legal gate 阻止，但 `protocol_error` 不再组合拉高；它在采样上升沿后由 `fault_q` 公开，并保持到 reset。

Python 3.6 与当前 Python 分别 9/9 静态测试 PASS。

## m216/service 错误没有被遮蔽

M1609 没有吞入任何外层 error source；未修改的组合层仍保持：

- M214/M216 wrapper：`local_fault_q || current illegal header || m202_protocol_error || m204_protocol_error`；
- M519 service top：`adapter_fault_q || current illegal header || fe_protocol_error || svc_protocol_error`；
- `numeric_overflow = svc_numeric_overflow`；
- `stale_response_seen = svc_stale_response_seen`。

所以，只替换 compactor source 不会在源码结构上遮蔽 m204、m216 或 service error。不过静态存在性不等于运行时证明；下一轮 VCS 必须逐项 fault injection。

## filelist 必须二选一

M1609 故意沿用 M214 module name。当前扫描到 38 个 filelist 使用 predecessor，0 个使用 successor，0 个同时使用两者。不得静默改写旧 filelist。

下一步必须新建 filelist，并满足：

1. predecessor path 与 M1609 successor path 恰好选一个；
2. 两者不得同时存在；
3. elaboration 后该 module 只能有一个 definition；
4. successor、M216 wrapper、service top、testbench 和 runner 全部 exact-SHA pin；
5. 除这一项 source substitution 外，top、参数、库、约束和 workload 不变。

## P1 与下一步

`P1_RUNTIME_AND_EXCLUSIVE_FILELIST_PENDING`：normalized diff 只能证明 source locality，尚未证明 registered fault latency、integration error OR-chain 和唯一 elaboration。

VCS 必测：非法 header/raw 当前周期不 accept、下一沿后 fault sticky；reset 清 fault；合法流量相对 M214 cycle-identical；m204/m216/service/overflow/stale 错误均不被遮蔽；backpressure、terminal、close 行为不变。

VCS PASS 后，才允许同 top/参数/库/约束做 predecessor-vs-successor matched DC，报告 setup、hold diagnostic、area、FF/cell 和边界路径。PTPX 仍不授权。

## P2

新注释写了 “accepting clock boundary”，但非法 request 实际不会被 accept。它应理解为 sampling rising edge；这是注释措辞问题，不影响 executable semantics。

本审阅没有运行 VCS/DC/PTPX，没有修改 M1609、冻结 M214 或 docs/359。
