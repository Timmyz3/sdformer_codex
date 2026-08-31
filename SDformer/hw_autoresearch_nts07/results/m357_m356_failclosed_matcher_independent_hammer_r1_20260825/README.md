# M357：M356 fail-closed q128 matcher 独立 VCS 打铁

结论：**93/100，P0=0、P1=0、P2=4；GO exploratory logic-only DC。**

M356 的 exact-SHA runner 已在 fresh 目录重新执行，Synopsys VCS V-2023.12-SP1 编译/仿真均为 0。fresh receipt 和 assertion report 与原 M356 逐字节一致，原/fresh RUN_MANIFEST 及二级 seal 全部通过。M356、M348、M350 和 `docs/359` 均未修改。

## 数值、顺序和 II1

fresh 结果为 3000 accepted/retired、0 numeric/order mismatch：use-PWP 2885、fallback 115、正负 residual 同时非零 1643、exact-pattern use 278。scoreboard 同时比对 original、center、center ID、distance、population、use/fallback、plus/minus mask 和队列顺序。

lowest-ID 语义成立：M348 从 ID0 初始化，后续只有严格更小 distance 才替换 winner；相同 M348 SHA 已在 M350 做过独立 trace reconstruction，本次 wrapper 又直接透传全部 numeric output。signed mask 为 `plus=original&~center`、`minus=center&~original`；fallback 为 `plus=original, minus=0`。

610 个 output-stall cycle 中 payload/valid 稳定，minimum latency=128，backpressure maximum=167；连续 input accept=518、output retire=402，支持 directed II1，但不是 Fmax 或 throughput/mm²。

## sticky error 与内部复活攻击

原 M356 五类攻击和 40 个 post-error presentation cycle 全部通过，0 handshake。M357 另外跑了一个层次化 VCS 攻击台，直接观察 wrapper 内部：

- 五类攻击：out-of-order first、early group0 commit、missing final commit、duplicate group0、skip group1；
- 40 cycle 中 external cfg handshake=0；
- core-facing cfg handshake=0，core-facing input handshake=0；
- internal `core_cfg_active` revival=0；
- `cfg_next_group_q` 和全部 128 个 `pattern_q` mutation=0；
- core 自身 ready 为高、但 wrapper ready 被屏蔽的 observation=40；
- reset 后完整合法八拍配置成功=1。

因此 M356 不是只 mask 外部 `cfg_active`：它同时切断 core-facing `cfg_valid/in_valid`，内部状态不会复活。error 只能在 numeric pipeline empty 时通过 cfg handshake 产生，所以也不存在 error 后 stale output 逃逸路径。

## 四个 P2

1. 原 TB 的 40 是 backpressured presentation cycles，不是 40 个被接受的合法 beat；ready 为低时它还逐周期改变 group/payload。建议重命名为 `attempt_cycles`，并补一个稳定保持 group0 直到 reset 的 compliant-master 测试。
2. receipt 将 handshake=0 写为常量；TB 用 fatal 间接保证正确，最好打印并解析显式 external/core handshake counter。
3. shipped SVA 只观察 wrapper-facing ready/active；M357 辅助攻击补查了内部，但 production SVA 仍应 bind core-facing valid、active 和 config-state stability。
4. 尚未覆盖 active catalog 后 bad reload、mid-pipeline/reset 和完整 drain-then-reload。这不重开 M350-P1-01，但限制完整 PWP executor 的 protocol claim。

## DC admission

现在可以对 `m356_failclosed_q128_signed_residual_matcher` 做 TSMC28、3.000 ns、ideal-clock/zero-wire 的 exploratory pre-macro logic-only DC，报告总/组合/时序 area、flop、setup/hold、128-stage global-advance critical path，并拆出 wrapper 相对 M348 的增量。还应与 SERIAL16 或 PE16/32/64/128 做 throughput/mm² 对比。

这不是 complete PWP Conv 或 paper PPA admission。M350-P1-02、finite-context cycle、PWP address/memory、Formality、SAIF/PTPX、energy、system speedup 和 DATE headline 仍是 NO-GO。
