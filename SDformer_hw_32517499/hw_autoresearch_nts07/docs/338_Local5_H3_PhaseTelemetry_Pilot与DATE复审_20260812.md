# Local5 H3 Phase Telemetry Pilot 与 DATE 复审

> 日期：2026-08-12  
> 正式正证据：`results/local5_phase_telemetry_pilot_h3_sample2_w249_v3_canonical_20260812/`  
> 证据：`[rtl-direct]+[完整 identity trace 逐事件对齐]`  
> formal G0：**DENY**

## 1. 结论

H3 telemetry-only pilot 已完成真实 RTL、完整 identity trace、compact telemetry、
Acc32 和 fail-closed 身份验证闭环：

| 项 | 结果 |
|---|---:|
| 身份 | sample2/stage0/block0/window249/H3，requested=actual |
| 局部 semantic phase | 52 |
| compact resource event | 330,627 |
| 完整 identity trace | 862,507 行 |
| Acc32 | 43,200，mismatch=0 |
| telemetry/完整 trace 字节比 | 0.3947 |
| verifier 单测 | 7/7 PASS |

52 条只表示 H3 Direct pilot 的局部语义 phase，**不是** formal 462,600-phase schema，
也不提供 formal 全序账本。

## 2. 已实测的事件

被动 monitor 只读取 DUT/TB 内部信号，没有输出端口，不驱动 ready/valid、状态、存储
或结果。已实测：

- group/tile/head transaction 边界；
- head weight/frontend/readout/release 状态 phase；
- tile drain 状态 phase；
- relation request/response accepted cycle 与 identity；
- weight request/response accepted cycle 与 identity；
- final accepted cycle 与 identity；
- cross-head 1RW Acc command cycle/address/read-write；
- TCFM5 term commit cycle/source/lane/五位 bank update mask。

独立审稿代理未调用仓库 verifier，直接解析两份 CSV 并比较完整有序
`(cycle, identity)` 向量：

| 资源 | 事件数 | 独立比较 |
|---|---:|---|
| relation request | 4,050 | 完全一致 |
| relation response | 4,050 | 完全一致 |
| weight request | 9,216 | 完全一致 |
| weight response | 9,216 | 完全一致 |
| final accept | 43,200 | 完全一致 |

它还从完整 trace 的 `head_state/tx_state` 独立重建 52 条 semantic phase，集合与
telemetry 一致。由于两边的写出顺序不同，该检查采用排序后的语义边界集合，不应写成
formal 全序等价。

## 3. Acc32 独立闭环

本地 verifier 对 43,200 个 Acc32 零失配。独立复审又直接读取
`ordered_term_items.npz` 的 destination-major 项和 theta-folded INT8 权重，未导入
仓库金参考函数，重新计算 `coefficient @ weight.T`：

```text
mismatch = 0 / 43,200
range = [-11,104, 10,400]
sum = -1,099,744
```

因此数值证据不是只比较同源摘要。

## 4. 身份 P0 负结果与修复

v1/v2 请求 `window94`，实际 payload 是 `window249`，却仍写出 PASS complete。两包均
是 P0 负结果，不得作为任何正证据：

- `results/local5_phase_telemetry_pilot_h3_sample2_actualw249_v1_20260812/`
- `results/local5_phase_telemetry_pilot_h3_sample2_actualw249_v2_sealed_20260812/`

v3 改为 canonical `window249`，requested/actual 严格 MATCH。专门把请求篡改回
`window94` 时：

```text
runner exit code = 3
forbidden output dir = absent
PASS receipt/verification/complete = absent
```

负结果位于
`results/local5_phase_telemetry_pilot_identity_tamper_w94_v3_20260812/`。

## 5. 证据绑定

v3 绑定：

- monitor/bind/verifier/test/runner 的运行前源码快照；
- v10 baseline release 和仅追加被动 monitor 的 compile argv；
- executable、run argv、完整 trace、telemetry 和 Acc32；
- task plan、identity table、profile payload、软件 expected；
- run receipt、verification、complete 与最终 SHA 清单。

独立复审确认顺序为“运行产物 -> run receipt -> verification -> complete/最终 SHA”，
未发现验证后改写造成摘要陈旧。

## 6. 显式未覆盖

- epoch-slot 1RW accepted command；
- FIFO2 push/pop；
- EREP fill/execute primitive；
- 五个 Acc bank 的逐 bank 地址；
- formal prepare/drain resource code；
- 1,200 window/13,800 head/462,600 phase archive；
- C0-C4 独立底层重算；
- 100/100 Acc32 numeric、DC/STA/SAIF 或 ASIC PPA。

## 7. 独立 DATE 裁决

```text
4.2/5，Conditional Accept for H3 telemetry pilot
Pilot P0 = 0
Formal G0 = DENY
```

准入边界：

- H24 单窗口 pilot：Conditional GO；
- 少量多窗口 pilot：Conditional GO；
- 1,200-window 逐事件 CSV：NO-GO；
- formal G0：DENY。

H24 前需关闭：

1. verifier 和 runner 的 H 参数化，去除 H3 固定计数；
2. 状态号分类的独立规格检查，避免 monitor/verifier 共同硬编码；
3. CROSS_ACC_CMD 与 TCFM5 bank update 的第二观测源或 SVA ledger；
4. v1/v2 machine-readable denylist 和聚合器 `identity=MATCH` 门槛；
5. digest/RLE 表示与磁盘配额；当前 20.7 MB telemetry/窗不能直接扩全量。

P2 为 Icarus/Verilator 交叉验证、随机反压/密度极值和外部不可变信任根。

## 8. DATE 表述边界

允许表述：

> 一个 H3 Direct 窗口的被动 semantic telemetry 已与完整 identity trace 对五类资源
> 逐事件对齐，并完成 43,200 个 Acc32 独立零失配验证。

禁止表述：

- 52 条 semantic phase 等价于 formal 462,600-phase schema；
- 本 pilot 已覆盖 EREP candidate 或完成 formal G0；
- telemetry 验证周期、文件压缩比是架构性能或 DATE 创新；
- v1/v2 是正证据。
