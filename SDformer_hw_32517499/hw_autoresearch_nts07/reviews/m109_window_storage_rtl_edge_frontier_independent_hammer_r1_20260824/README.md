# M109 window/storage/RTL-edge frontier：独立打铁复审

日期：2026-08-24

结论：**82/100，P0=1 / P1=4 / P2=5。raw work、group、commit/flush、storage 算术全部复现，W64 也确实逐字段等于封存 M108-r2；但当前 M109 cycle/ratio NO-GO，因为它连同数值一起继承了 M108-r2 已知的 prior-drain serialization P0。**

## 四窗口独立重建

独立脚本没有 import 或执行 M105/M108/M109 producer analyzer，而是从冻结 M40 support planes、M72 centers 与 M41 INT8 weights 重建自然行 event/PWP work，再分别聚合 W43/W64/W294/W384。随后对相同 ordered descriptors 同时运行 producer 的 published recurrence 和包含 prior-drain dependency 的 dual-timeline recurrence。

| W | groups | descriptors | flush | commit | storage lower bound | published cycles / ratio | prior-drain-aware cycles / ratio | undercount |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 43 | 46,867,834 | 604,800 | 1,400 | 480,000 | 101,864 B | 556,896,367 / 2.001924872× | 556,942,442 / 2.001759255× | 46,075 |
| 64 | 35,140,002 | 406,080 | 940 | 480,000 | 151,592 B | 521,238,438 / 2.138875698× | 521,264,186 / 2.138770048× | 25,748 |
| 294 | 10,395,056 | 95,040 | 220 | 480,000 | 696,232 B | 446,208,991 / 2.498523182× | 446,212,276 / 2.498504788× | 3,285 |
| 384 | 8,271,296 | 69,120 | 160 | 480,000 | 909,352 B | 439,707,315 / 2.535467139× | 439,708,199 / 2.535462042× | 884 |

四点都严格守恒 `events=188,148,490`、`PWP tokens=226,222,255`。fixed8 baseline 也从 raw masks 独立重建为 `371,461,096 events × 3 = 1,114,383,288 tokens`。

## P0：继承 M108-r2 漏算

M109 的 recurrence 仍使用：

```text
dispatch_ready = fill_end + 1
```

但当前 M106 RTL 只能在 `!drain_active_q && selected_bank==READY` 时 dispatch。正确的软件下界至少需要：

```text
dispatch_edge = max(fill_end, prior_controller_drain_release) + 1
```

这不是新发现的推测：M108-r2 的独立评审已经将其列为未关闭 P0。M109 只 pin 了 M108 producer result，没有 pin/消费该独立评审，因此 W64 与 M108-r2 的逐字段 exact match 同时也证明了缺陷被原样传播。

所需修复：用 prior-drain-aware dual timeline 重生 M109 result/contract，pin M108 独立评审；然后补合同早已要求的 commercial small-stream M106/PWP/commit cycle miter。修复前不能称 current-M106 cycle-exact 或 scheduled RTL ratio。

## 2.5× 分界与公平 baseline

分界没有被本次 recurrence repair 推翻：

- W294：`2.498504787887× < 2.5×`
- W384：`2.535462041726× > 2.5×`

因此“W294 未过、W384 已过”的方向可以保留为 **same-clock precompacted software bound**。W294 距 2.5× 仅约 `0.0014952×`，不能把 crossing window 当成物理实现结论。

commit/flush 算术正确且对称：commit 固定为 `20 records × 3000 rows × 8 blocks = 480,000 cycles`；flush 为 `20 × ceil(3000/W)`。但 baseline 只有 fixed8 service tokens 加相同 tail，没有自己的 edge-aware descriptor/controller ingress schedule，所以这是可复现的 service-island denominator，不是 equal-controller end-to-end baseline。

## Storage 与 RTL/projection 边界

storage 整数和 byte ceiling 全部正确：

```text
descriptor = 2 banks × 128 keys × W rows × 2 bits
accumulator = W rows × 8 blocks × 96 lanes × signed24
combined = descriptor + accumulator + 314 minimum payload metadata bits
```

这些数字只是 lower bound，排除了 controller/grace、valid/epoch tags、ECC、SRAM macro rounding/ports/RDW。signed24 full-lane accumulator 也没有 VCS numeric miter，不能称 SRAM size 或 macro area。

非 W64 没有冒充 VCS：result/contract 的 admission 明确令 `controller_geometry_vcs=false`。同时必须注意 production M106 自己硬冻结 `WIN_ROWS==64`，所以 W43/W294/W384 只能叫 parameterized architecture projection。JSON 字段虽统一名为 `rtl_edge_recurrence`，下游表格必须保留 projection 标签。

## Admission

- exact raw work/group：**GO**。
- commit/flush/frozen baseline 算术：**GO（conditional service baseline）**。
- storage：**GO（lower bound only）**。
- W64 等于 M108-r2：**GO（identity only）**。
- 当前 published M109 cycle/ratio：**NO-GO，P0**。
- prior-drain-aware 数字与 W294/W384 分界：**GO（software bound only）**。
- non-W64 RTL measured、physical、equal-area、macro-inclusive、system/full-network/headline：**NO-GO**。

机器证据见 `m109_window_storage_rtl_edge_frontier_independent_audit.json` 与 `m109_window_storage_rtl_edge_frontier_independent_hammer_review.json`。本评审只写本目录，未修改 production、contracts/results 或 `docs/359`。
