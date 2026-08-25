# M106 r2 standard-streaming grace 独立打铁评审 r1

日期：2026-08-24  
评分：**89/100**  
严重度：**P0=0，P1=4，P2=4**  
结论：**功能 RTL GO；accumulator、DC/PPA、scheduled/physical/system/headline 均未准入。**

`docs/359_DATE终局冻结_20260813.md` 的 SHA256 在评审前后均为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。本评审只写本目录，未修改 production RTL、SVA、TB、contracts、results 或既有 sealed evidence。

## 一句话结论

M106 r2 已把 r1 的真实硬件 P0——exact close 在 bank switch 后被二次接受——修好，并由独立 commercial VCS 在 cross-bank 可用场景关闭。另一个 r1 P0 是旧 sampled-low 协议的产物；r2 明确改成标准流式规则：**只有完全相同的已接受 payload 属于 grace，不得再收；下一拍 changed legal payload 是新事务，可 II=1 接收；changed illegal payload 必须同拍 fail-closed。** 独立 8192-event 无气泡流和 changed-close 连拍均通过。

这使 controller functional RTL 和下一阶段 accumulator miter/logic-only DC 启动可以 GO；它没有把任何 token ratio 提升为 scheduled、physical 或 system speedup。

## 冻结身份

| 项目 | SHA256 |
|---|---|
| M106 r2 contract | `984ca6558ebbf3a58135e60b4aa889b7726532b8a4fc872acf7156f50d7d8196` |
| production RTL | `a6937765aea87269c3d38123b656c72b7ee400e36b0d634f21ab9c7dbdefc0b7` |
| production SVA | `db98dc72b18aa789088bdbea40ab1b5a6cd7399b2bd8d373b37d17a5bcfba227` |
| production TB | `cb581e8b02fdf86a68fc95197c96c91673a6b6c5499829c2441a168cc6c544bb` |
| r2 production sealed RUN_COMPLETE | `45db2f7ae514f7afbafff93dddbd272076181e2a3f1aa6bbcb25f24f71710999` |
| independent assertions | `8c71ca574a3dcd049f8b6730e708ad6441bb56cfbb702888b4aaf813a6dd8dea` |
| independent TB | `c185ebc0a9250ef3959ab6a80627026cf79c69092087f10ff406edf9766d1b5e` |
| independent filelist | `52f2e189cb10622d0e38644dd9bf6cfc77996235875ccad7f19f9996efce6f35` |
| independent sealed runner | `0fc88795f2f00c64e6c7ca9af2b42cc53c9b6ff06a21ae0444dcc15a34f40e9b` |
| M107 explicit cycle-exact revocation | `2fc3b195d172898341076f3b90c537013e446d1385b0dbb63213046d818cb7f1` |

完整输入身份见 `input_manifest.sha256`，所有条目在 sealed VCS 前通过 `sha256sum -c`。

## Commercial VCS 独立结果

工具：Synopsys VCS V-2023.12-SP1。主测试与 generic range probe 独立编译、独立仿真，全部 RC=0；主编译 warning/error pattern=0，assertion failure=0。

| 场景 | 结果 |
|---|---:|
| changed legal event 连续接受 | 8192 |
| `event_accept ##1 event_accept` cover | 8191 |
| full keys × rows | 128 × 64 |
| load / event / total service token | 384 / 8192 / 8576 |
| exact event grace，不二次接受 | 1 |
| exact cross-bank close grace，不二次接受 | 1 |
| changed legal close 连续接受 run | 2 |
| 主测试非法攻击 | 4 |
| service stall（测试计数） | 3 |
| sticky fault cover | 16 |

主测试逐 token scoreboard 检查了排序后的 load/event 类型、source、block、load beat、row、destination、negate、last 和 context。攻击 campaign 检查 duplicate、changed context/base、event-close collision、两 bank 不可用；每项均同拍 fail-closed、sticky quarantine，并且只能 reset 恢复。

生产参数 `ROW_W=6, WIN_ROWS=64` 覆盖全部 0..63 编码，因此 production row range attack 在二进制上不可表达。评审另用 `WIN_ROWS=63, ROW_W=6` 的 review-only probe 验证 row 63 的 comparator、同拍 fail-closed、sticky 和 reset-only recovery。这个结果只证明 generic logic，不伪装成可达 production range stimulus。

## r1 P0 的处置

| r1 P0 | r2 处置 |
|---|---|
| exact held close 在 bank switch 后 reaccept | **CLOSED**：RTL 在 `event_ready`/`window_close_ready` 上抑制 exact accepted-grace match；独立 cross-bank VCS 无 ready、无 accept、无 phantom window。 |
| held-valid payload 改变但未经过 valid-low | **由 r2 contract 明确 supersede**：changed legal 是新事务并保留 II=1；changed illegal 同拍 fault。独立 8192-event 与两次 close 连拍通过。 |

第二项不是在旧 r1 合同下“修复通过”，而是有意修改接口语义。这样做是合理的：如果强制每个 descriptor 之间都出现 sampled-low bubble，冻结的 event II=1 根本无法成立。

## 仍需注意的接口边界

r2 的 changed-legal streaming 是标准的吞吐语义，但整个入口仍不是通用 backpressure-tolerant ready/valid：当两 bank 都不可用时，producer 若继续 assert `event_valid`，RTL把它当 illegal request 并永久 quarantine，而不是简单 `ready=0` 等待。因此集成时必须有 ready look-ahead、上游 credit，或加 adapter；论文/接口文档应称其为 **fail-fast ingress contract**，不能笼统声称可直接连接任意 ready/valid producer。

## 存储与 accumulator 边界

bitmap 数字无误，但必须带 qualifier：

- presence：2 × 128 × 64 = 16,384 bit；
- direction：16,384 bit；
- raw bitmap payload：32,768 bit = 4,096 byte；
- active-key、base、context、identity-valid 至少再加 314 bit；
- control/ECC/macro rounding 前总计至少 33,082 bit，向上取整 4,136 byte。

当前 macro count=0。24-bit accumulator 也只是端口切口：M41 dense magnitude bound `877,824` 至少需要 21-bit signed，24 bit 有 3 bit 数值余量，但 M106 尚无 finite-width accumulator、八 bank 地址/RMW/forwarding、clear、commit、目标宏 latency/read-during-write 证明。

因此下一步 admission 是：**允许实现并证明 accumulator miter，也允许启动 logic-only DC；不允许把它写成 accumulator 已闭合或 DC/PPA 已完成。**

## M107 cycle-exact 降级复核

M107 原始 contract 仍含 cycle-exact admission，但后续独立评审证明它漏了当前 M106 的 READY→service dispatch edge 与 blocked-bank EMPTY→FILL reacquire edge；selected window-major 少算 384,097 cycles。仓库已有显式 revocation，SHA256 为 `2fc3b195d172898341076f3b90c537013e446d1385b0dbb63213046d818cb7f1`。

本评审只保留 M107 的 exact raw work ledger 和 fluid software service-island recurrence。edge-aware service-island `2.037842934×` 也不是 scheduled accumulator、physical 或 system speedup；它离 2× 只剩 10,347,101 cycles，不能当稳健 headline。

## Findings

### P1

- **FAILFAST-NOT-GENERAL-BACKPRESSURE**：resource unavailable 时 valid 会触发 sticky fault，需上游 credit/adapter，不能作为任意 ready/valid 接口使用。
- **CONTEXT-ADDRESS-SCHEMA-OPEN**：context bit allocation、partition/operator identity、base alignment/overflow、partial-window legality未冻结。
- **ACCUMULATOR-RMW-NOT-IMPLEMENTED**：24-bit 数值宽度合理，但 accumulator、bank/address/RMW/forwarding/commit 尚未实现或 miter。
- **PRE-DC-PRE-MACRO**：无 M106 logic-only DC sealed result，无目标 SRAM macro mapping，不能准入 frequency/area/energy/PPA。

### P2

- production range code 不可表达；range 证据来自 review-only generic probe。
- production directed coverage 比本次独立 full-capacity/II1/scoreboard campaign 浅，建议把关键 assertion/stress 回灌生产回归。
- M107 原始 contract 与后续 revocation 并存，下游工具必须解析 supersession，最好建立单一 claim registry。
- 本次 full-capacity 是 adversarial synthetic traffic，不是 actual ordered M105 heldout record 的 RTL replay。

## GO / NO-GO

| 项目 | 决定 |
|---|---|
| r2 exact accepted grace | **GO，P0 closed** |
| changed legal event/close II=1 | **GO** |
| full 128×64 capacity、order、metadata | **GO** |
| duplicate/context/collision/unavailable quarantine | **GO** |
| bitmap 32,768 bit | **GO，必须注明另有至少 314 bit metadata** |
| controller functional RTL | **GO** |
| next accumulator miter | **GO to implement and prove；尚未准入功能** |
| logic-only DC launch | **GO** |
| DC/macro PPA | **NO-GO，未运行/未映射** |
| M107 r1 cycle-exact | **NO-GO，已显式撤销** |
| actual-record RTL replay | **NO-GO** |
| scheduled/physical/equal-area/system/headline | **NO-GO** |

复跑命令：

```bash
cd hw_autoresearch_nts07/reviews/m106_r2_standard_streaming_grace_independent_hammer_r1_20260824
./run_vcs_m106_r2_independent_streaming.sh
```

runner 会拒绝覆盖既有 `vcs_sealed/`；若需复跑，必须另存评审目录，不能破坏本次收据。
