# M104 r2 literal serial token correction 独立打铁复审 r1

日期：2026-08-24  
评分：**94/100**  
严重度：**P0=0，P1=2，P2=2**

## 结论

M104 r2 已正确关闭上一轮 token-ledger P0。独立从 M103 audit 读取 `E/G`、从 M102 ledger 读取 PWP/baseline，再按当前 production RTL 的互斥 load/event 协议计算，得到：

```text
E = 188,148,490
G =   1,105,920
correction = E + 3G = 191,466,250
combined = correction + 226,222,255 = 417,688,505
ratio = 1,114,383,288 / 417,688,505 = 2.6679769126038075
```

contract、analyzer output、RUN_COMPLETE 和独立复算四者一致。上一轮 `2.6750597075487446` 已明确标记为 `implemented=false` 的 `E+2G` fusion/overlap target，不再冒充当前 RTL literal model。

结论为：

- **GO**：引用 M104 r2 作为当前 RTL 在 perfect phase/key batching 条件下的 literal serial token ledger。
- **GO**：继续 production-only logic DC，仍保持 pre-macro/module-only 边界。
- **NO-GO**：将 `2.6679769×` 或 `2.6750597×` 写成 scheduled cycle、physical、equal-area、system 或 headline speedup。

## 身份和收据

| 输入 | SHA256 |
|---|---|
| r2 contract | `b88ec871b84342a39257497c4803db240f6898b0d5f748bb31d51966deb836c8` |
| r2 analyzer | `01736afedc74b4f77182931769966ef1657577cedb4916e4d7827a7f593e54d0` |
| r2 result | `2c59c7c8836a5f7bf802f6b5eff1ccb8e2d1e3fecc074e307458cd8c08d3538e` |
| r2 RUN_COMPLETE | `5ef52d0370ec2f558c34be2b6c2cde5aa390d5efcabf025bc666937fcc031ec9` |
| r2 result manifest | `44ba839026fba21fdd1ab06bd27e31fd39d65ca7b0d01f8f3c51406e3ba73fe3` |
| prior M104 r1 review | `22ce5342980f53429ab4a3bf1dff8f21df0f874730910556c196e58354e10860` |
| prior M104 r1 audit | `afdcbf92cdbd2514b4afe5f0b6454ee5eb404a269e8f708e6bc540d9ab8bbe3e` |
| M103 audit | `935119fab809e15f49089926550f89b3c84c2b13c0be58c96b0ea8709ed683fe` |
| M102 ledger | `a5d465b7d3361ed2ff176b4230d9051c29137aee86211cec9c3eb9ee8131aad5` |
| production RTL | `37f86144563d45ea96f594847828a00c7d872602419d81a070738f12b4417f6a` |

r2 result manifest 4/4、prior M104 r1 review manifest 12/12 均独立通过。contract、result、M103、M102、prior review/audit 都以拒绝 duplicate key 和非标准常量的 strict JSON 读取。

## 独立交叉复算

本复审没有调用 production analyzer，而是交叉读取被冻结证据：

| 量 | 独立来源 | 数值 |
|---|---|---:|
| correction/fallback events `E` | M103 order-independent grouping | 188,148,490 |
| phase weight groups `G` | M103 order-independent grouping | 1,105,920 |
| PWP token term | M102 candidate ledger | 226,222,255 |
| fixed8 baseline token denominator | M102 baseline ledger | 1,114,383,288 |
| 当前 RTL correction | `E+3G` | 191,466,250 |
| 当前 RTL combined | correction + PWP | 417,688,505 |
| 当前 RTL条件 token ratio | baseline / combined | 2.6679769126038075 |

production RTL 的文本与冻结 SHA 同时确认：

- 每个 key 需要三个独立 `load_accept`；
- 每个 destination 需要一个独立 `event_accept`；
- event 仅在 `held_valid_q` 已成立时合法；
- `load_valid && event_valid` 是 protocol collision；
- `load_ready` 与 `event_ready` 互斥另一类 valid。

所以当前 single-held-vector 端口的 literal 串行模型确实是 `E+3G`。

## r1 target 已正确降级

r2 将原数值保留为设计目标：

| 项 | 数值 |
|---|---:|
| target formula | `E+2G` |
| target correction | 190,360,330 |
| target combined | 416,582,585 |
| target conditional token ratio | 2.6750597075487446 |
| implemented | false |
| implicit free overlap | 1 token/key |

它需要 third-load/first-event fusion、last-event/next-key preload overlap、ping-pong held vector 或等价机制。当前 RTL 没有这些机制。r2 result 与 RUN_COMPLETE 都显式写出 `r1_undercharge_tokens=1,105,920` 和“r1 ratio 是未实现 target”，降级完整。

## Claim boundary

r2 仍假设 perfect phase/key batching，且以下字段全部为 false：

- ordered bounded schedule；
- scheduled cycles；
- physical speedup；
- equal area；
- macro-inclusive PPA / paper-PPA-ready；
- system speedup；
- headline。

`2.6679769×` 的正确称呼是“同 denominator、同 PWP term、perfect batching 条件下的 literal service-token ratio”。它不代表真实 queue、SRAM、bank conflict、accumulator、frequency 或端到端执行。

## Findings

### P0

无。上一轮 `E+2G` 与当前 RTL 协议不一致的问题已由 r2 修复。

### P1

- **M104-R2-P1-01-PERFECT-BATCHING-STILL-UNSCHEDULED**：r2 只修正记账，没有新增 ordered trace、bounded transpose queue、spill/fallback、phase drain 或 actual-record replay。任何 scheduled-cycle 结论仍不准入。
- **M104-R2-P1-02-ACCUMULATOR-AND-BANKS-STILL-PORT-CUT**：destination tag→bank/address/port、PWP dependency 和有限位宽更新等价性仍缺失；token ratio 不能推导 physical/system speedup。

### P2

- **M104-R2-P2-01-ANALYZER-DOES-NOT-USE-EVENT-TOKEN-FIELD**：producer analyzer 第 74 行直接写 `events + load_tokens_per_group*groups`，没有乘 `event_tokens_per_event`。当前字段为 1，所以结果完全正确；建议改成字段驱动，防止未来字段漂移时仍错误通过。
- **M104-R2-P2-02-FUSED-TARGET-NOT-PRODUCER-ASSERTED**：analyzer 将 fusion target 从 contract 原样复制，没有独立断言 `E+2G`、combined、ratio 与 `implemented=false`。本复审已经独立验证，但 producer 应补 fail-closed target assertion。

## GO / NO-GO

| 决策 | 结论 |
|---|---|
| r2 当前 RTL literal conditional token ledger | GO |
| 关闭 prior M104 r1 ledger P0 | GO |
| 将 r1 `2.6750597×` 作为未来 fusion/overlap target | GO，必须写 implemented=false |
| production-only logic DC | GO，module-only/pre-macro |
| ordered/scheduled performance | NO-GO |
| physical/equal-area/system/headline | NO-GO |

## 下一步

1. 后续合同统一引用 r2 的 `E+3G` 数值，停止把 r1 result 当作当前 RTL ledger。
2. 对 analyzer 做两个小型 fail-closed 加固：使用 `event_tokens_per_event` 字段，并断言 fusion target 的公式、数值和 `implemented=false`。
3. logic-only DC 可继续；其结果只能说明当前 broadcaster 的 area/timing，不会自动准入 token ratio。
4. 若要实现 `E+2G` target，先设计并 VCS 验证 fusion/ping-pong preload，再重新生成 ledger。
5. ordered trace、bounded transpose、accumulator miter 和 bank schedule 完成后，才能进入 scheduled cycle simulator。

本复审只写 `reviews/m104_r2_literal_serial_token_correction_independent_hammer_r1_20260824/`，未修改 production、contracts、results 或 `docs/359`。
