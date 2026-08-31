# M231r2 independent hammer review

**Score: 88/100. P0: 1. P1: 7. P2: 4.**

结论是 **M231r2 的同拍故障原子性修复可以在“当前 snapshot + directed RTL/VCS + full-flop logic-only DC”边界内准入**。旧 M231-P0-01 已关闭；旧 M231-P0-02（M167→M231→M216/M218 可执行链）原样保留，因此不能把本次修复写成 FFN 或系统性能结果。

本评审绑定并核验了纠错合同、当前 RTL/SVA/TB、r2 VCS seal `f49d2b...`、r2 DC evidence seal `cc1d46...` 与旧 M231 review seal `09dc...`。所有 seal 校验通过；`docs/359` 仍为 `dedde7ce...`。

## 独立故障攻击

我没有只复用生产回执：重新用 Synopsys VCS V-2023.12-SP1 编译并运行当前 W384 production TB/SVA，结果是：

```text
PASS M231r2 W=384 pairs=3 tokens=6 packets=6 header_stalls=28 raw_stalls=3 full_hits=9 attacks=2 fault_atomic=1 cycles=94
```

随后用独立 TB 重放旧 P0 的精确场景：bridge 正在 `DRAIN_RAW`、`raw_ready=1`，同拍注入 wrong-tag event。观测为：

```text
ATTACK_CYCLE protocol_error=1 pair_accept=0 event_accept=0 header_accept=0 raw_accept=0 raw_valid=0 raw_ready=1 packet_before=0 packet_now=0
AFTER_EDGE protocol_error=1 packet_before=0 packet_after=0 full_before=1 full_after=1 drain_state_before=2 drain_state_after=2 drain_row_before=0 drain_row_after=0 drain_packet_before=0 drain_packet_after=0
PASS_INDEPENDENT_M231R2_OLD_ATTACK_QUARANTINED accepts=0 packet_delta=0 pending_state_preserved=1
```

这直接确认旧攻击现在变成 `protocol_error=1`、全部 accept 为 0、raw packet counter 不动，且 pending full/drain state 保持不变直到 reset。production 四宽 sealed VCS 也全部报告 fault-cycle accept/state-commit 为 0。

## 组合顺序与环路审计

当前组合顺序是 `event_shape_legal → illegal_event → protocol_error → ready/valid → accept`，时序块再由同一个 `if (!protocol_error)` 阻断提交。`illegal_event` 只依赖寄存状态与 event 输入，不依赖本模块 ready/valid；RTL 内部没有组合环，四宽 DC `check_timing` loop 检查也通过。

但集成约束不能省略：`event_ready` 依赖当拍 event valid/payload 合法性，未来 M167 adapter 必须保证 valid/payload 不组合依赖 `event_ready`。修复还把 `event_pair_tag` 拉进 output quarantine 和 counter enable 的当拍控制锥；DC 的 critical path 已从该输入出发，并出现 high-fanout warning。这是集成时序风险，不是本模块内部组合环。

## 四宽 DC 复算与 r1 对比

| width | r2 area (um2) | r2 cells / seq | setup / hold (ns) | area vs r1 | setup slack delta |
|---:|---:|---:|---:|---:|---:|
| 384 | 5,690.916 | 5,428 / 1,721 | +1.1122 / +0.0010 | -0.267% | -0.4759 ns |
| 768 | 10,641.456 | 9,603 / 3,257 | +0.9390 / +0.0010 | +0.094% | -0.3169 ns |
| 1536 | 20,803.482 | 18,643 / 6,328 | +1.2705 / +0.0001 | -0.728% | -0.1784 ns |
| 3072 | 40,851.216 | 36,360 / 12,473 | +1.2280 / +0.0000 | +0.005% | -0.0269 ns |

修复在该 screen 中可称为 **面积中性**：面积变化范围 -0.73% 到 +0.10%，四宽 sequential-cell count 完全不变。不能称它提高性能：四宽 setup slack 都下降；W3072 hold 还四舍五入到 0.0000 ns。W1536 cell count 下降 7.58% 而面积只降 0.73%，这是 mapping decomposition，不是可引用的 gate-count 优势。

这些仍是同一 3 ns、TSMC28、ZeroWireload、0 macro、全 flop、pre-layout DC 数字。没有 SRAM、PT、SAIF/PTPX 或 paper PPA。

## Supersession 与准入边界

| 项 | 判定 |
|---|---|
| 原 M231 r1 functional/DC admission | **REVOKED；只保留历史证据与旧 review 的根因/trace 算术** |
| M231r2 exact snapshot，普通路径与两类 directed attack | **GO** |
| M231r2 四宽 DC | **GO only as full-flop logic-only diagnostic** |
| M167→M231→M216/M218 finite trace | **NO-GO** |
| dynamic-BN accuracy / achieved traffic elimination | **NO-GO** |
| SRAM / energy / complete FFN / system / headline | **NO-GO** |

纠错合同是跑证据前的 immutable snapshot，因此其中 `corrected_vcs=false`、`corrected_synopsys_dc=false` 没有被回写。本次 sealed receipts 加独立评审足以支撑上述 scoped admission；最终 citation 最好另建只追加的 admission overlay，避免读者误用合同中的预运行布尔值。

## P0

1. **M231R2-P0-01 — 可执行 producer-to-consumer 链仍缺失。** 旧 M231-P0-01 已被修复；但 M167 rank3/dynamic-BN accuracy、typed producer metadata、M216/M218 consumer、两槽 finite recurrence 与 ordered 120-record trace 都未闭，旧 M231-P0-02 仍成立。

## P1

1. 攻击仍是 directed：每宽只有 idle orphan 与 concurrent wrong-tag 两类，未系统覆盖 wrong group/last、各状态 reset、counter rollover 和 simultaneous slot turnover。
2. 没有 Formality、mapped-netlist VCS 或 r1→r2 ordinary-path 等价证明。
3. 全 flop、0 macro、ZeroWireload；没有 SRAM bank/port、PT STA、SAIF 或 PTPX。
4. 同拍 quarantine 引入 timing-sensitive high-fanout control cone；setup slack 全部下降，W3072 hold 为 0.0000 ns。
5. 本模块内部无组合环，但未来 producer 的 valid/payload 必须禁止组合依赖 ready；typed adapter contract 尚未冻结。
6. correction contract 的 admission flag 是预运行状态；还缺独立 immutable admission overlay。
7. fault 后只能 reset，尚未覆盖各阶段 reset recovery 与 downstream retry/liveness。

## P2

1. `cp_fault_while_raw_would_accept` 实际只覆盖 `protocol_error && raw_ready`；独立 W384 hierarchical TB 才确认 pre-quarantine 状态确实是 `DRAIN_RAW`。
2. DC 有 signed/unsigned、parameter-fold unreachable branch 与 ignored parameter-guard `$fatal` warning，支持宽度下无功能阻断，但终局 RTL 应清理。
3. production data/stall 是 deterministic，换 seed 不扩展状态空间。
4. W1536 raw cell count 变化与 area 不成比例，禁止把 cell count 当优化指标。

## Allowed claims

- exact M231r2 snapshot 在四宽 directed VCS 中通过普通数据路径及两类攻击，同拍故障时零 accept/零 state commit。
- 独立 W384 旧攻击复现确认 `protocol_error=1`、accept 全 0、packet delta 0、pending drain state 保持。
- 四宽 3 ns pre-macro DC 的精确 area/cell/slack 可按全 flop、ZeroWireload、0 macro 边界报告。
- 相对 revoked r1，修复面积变化 -0.73% 到 +0.10%、seq count 不变、setup slack 均下降。

## Forbidden claims

- 禁止继续引用 M231 r1 VCS/DC 作为当前 RTL 准入。
- 禁止写 exhaustive protocol proof、Formality、mapped equivalence、PT/SAIF/PTPX、SRAM、energy 或 paper-ready PPA。
- 禁止写 M167→M231→M216/M218 已执行、finite trace 已闭、875.52 MB 已实现消除。
- 禁止写 PAFT/M167 dynamic-BN accuracy、complete FFN、system speedup、headline speedup。

最小下一里程碑是 typed M167 BACK adapter + M231 + M216/M218 exact consumer miter，在 120-record ordered payload 上执行有限两槽 backpressure，并把 cycles、stall causes、transactions、deadlock/liveness 分开封存；同时补系统化攻击和 Formality/综合网表等价。

本 review 只新增本目录，未改 production RTL/脚本/合同、论文或 `docs/359`。
