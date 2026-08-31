# M227 independent hammer review

**Score: 71/100. P0: 2. P1: 6. P2: 3.**

M227 的功能 RTL 可以保留：M225/M226/M226-review/M227 VCS 四层 seal 均通过，三档 F1/F2/F4 的 exact-SHA VCS 都是零数值 mismatch、零守恒 mismatch、零 assertion failure。独立重建三个 directed group 后，每档均精确得到 38 个唯一 source、119 次 context update、24 个 result beat；empty、mixed-sign、source-383 tail、full fanout、request/result stall、duplicate scan 与 illegal tail 均有非零覆盖。

但性能/PPA 不能准入。M226 的 `1.568695x/2.112902x` 是 M225 trace recurrence 的 prior，不是当前 M227 串行 FSM 能执行出的周期。RTL 对每个唯一 source 严格串行 `ST_REQUEST -> ST_WAIT -> ST_REPLAY`，请求、响应和更新不能重叠。即使零 stall、最短一拍 response，相对 M225 service recurrence 也必须为每个 768-bit weight read 加两个非更新周期。将该结构代回封存的 391,666,724 次 weight-vector read，F2/F4 上界只剩 **1.262957x/1.433702x**，不是 `1.568695x/2.112902x`。directed VCS 的 1.112601x/1.220588x 又混有测试平台逐 beat gap、响应延迟和不同 stall 相位，只能作功能观察，不能作性能数。

## Verdict

| 项目 | 判定 |
|---|---|
| parameterized K8 F1/F2/F4 functional RTL | **GO** |
| exact-SHA directed VCS / signed arithmetic / conservation | **GO** |
| same K8 state/mask/scanner/768-bit port fairness | **GO，限 equal-context/storage** |
| `1.568695x/2.112902x` as achieved M227 RTL speedup | **NO-GO** |
| M227 current serial FSM as performance architecture | **NO-GO，需流水修复** |
| matched DC / throughput-area / SAIF-PTPX | **NO-GO，未完成** |
| complete FC1/FFN/system/headline | **NO-GO** |

## P0 findings

1. **M227-P0-01 — cycle recurrence 与 RTL 不同构。** M225 将首次 weight service 包在 source service recurrence 内；M227 每个 source 单独消耗 request、wait/response、replay/update 三个互斥状态，且没有 next-source prefetch 或 response skid。必须先做 walker/request/replay 解耦、至少双 held-weight buffer 与显式 credit，再用同一 H67 100-record ledger 生成可执行周期；在此之前不得引用 M226 prior 为 RTL speedup。
2. **M227-P0-02 — DC/PPA evidence absent.** matched DC 的 F1 只完成 elaboration/precompile，Pass-1 mapping 时收到外部 `SIGTERM`，封存目录正确留下 `FAILED_OR_INCOMPLETE_DO_NOT_CITE`；F2/F4 未开始，也没有 postcompile、area、setup/hold 或 throughput/area。外部 kill 不是 DC 判出的结构/时序失败，但它仍然阻断全部 PPA admission。

## DC structural ruling

F1 映射时间长本身**不是结构 P0**。现有证据只说明 elaborated cone 较重：49,582 leaf、22,006 sequential，含两个 `8x1824`、一个 `8x384` 动态 mux；precompile 有 387 个 `LINT-1` non-driving cells，尚未经过优化。日志没有 DC mapping/timing failure，进程是外部 SIGTERM 终止。

不过这是明确的 P1 结构警报。下一版应在再跑三档 DC 前先做：扫描时增量构建 union，分层 bitmap walker；request/replay 双级流水和 next-weight prefetch；presence/sign 采用完整覆盖或 generation-valid，避免无谓 broad clear；accumulator 用 per-context one-hot bank enable/first-touch-zero，避免动态 `8x1824` read-modify-write mux。完成后必须重新 VCS，并从 F1 先做 3 ns 快速 compile sanity，再放大 F2/F4。

## P1 findings

1. directed population 只有三组、其中两组非空；满足合同最低门，但不足以替代随机/trace-replay miter。
2. wrong tag/epoch/source response、early scan-done、busy begin、sign-not-subset 等 fail-closed 路径虽有 RTL 逻辑/SVA，未被 directed attack 命中。
3. `numeric_overflow` sticky fail-close 没有边界/故障用例；现有低占用组也不能证明 Acc19 极值。
4. 模块固定 96 lanes，没有 partial-output lane-valid/tail；因此只能叫一个 FC1 slice，不能叫完整 FC1。
5. F1 elaboration 的全 FF mask/accumulator、broad clear 和宽动态 mux 是面积/时序风险；需结构化后再 DC，当前长 mapping 不足以证明失败也不足以证明可行。
6. DC elaboration 有多处 `VER-318` signed/unsigned conversion；VCS 数值通过但尚无 mapped netlist/Formality，综合等价仍未准入。

## P2 findings

1. TB 的 `full8_sources` 实际按“含 full-eight 的非空 group”累加，数值 2 不是 full-eight source 个数；cover 本身有效，但收据字段名易误读。
2. done backpressure 与合法 out-of-order scan 未覆盖。
3. DC fail marker 记录 `runner_exit_code=0`，但日志明确为 SIGTERM/kill；fail-closed 状态正确，退出码收据需要修正以免自动化误判。

## Required next gate

下一里程碑应是 M227 的 **pipelined recurrence repair**，不是直接重跑同一大 cone：

- one-entry current + one-entry next source/weight，walker、request、response、replay 解耦；
- 同一拍允许 replay 当前 weight 并申请下一 source，credit/identity 全 SVA；
- 用 exact H67 100-record trace 证明实际 RTL recurrence，报告 first-source、held replay、bubble、scan、drain 分项；
- VCS 增加 wrong-response、overflow、done-stall、out-of-order、partial-output wrapper 与至少一组高密度随机 miter；
- 再做 F1/F2/F4 matched DC，只有 setup/hold 均过后才比较 throughput/area，再对获胜点做 SAIF/PTPX。

本 review 未修改 M227 RTL/合同、论文或 `docs/359`。
