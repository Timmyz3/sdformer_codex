# M104 r3 accepted-last-event grace：独立打铁复审

日期：2026-08-24

结论：**92/100，P0=0 / P1=4 / P2=5。r3 exact SHA 的 directed functional gate 通过，并准入 module-only logic pre-macro DC 下一门。** 这只是允许为当前 SHA 建立并运行 exact-SHA 的 M104 专用 DC launch，不代表已有 DC/PPA 结果，更不准入 scheduled、physical、equal-area、system 或 headline 指标。

## 独立商业 VCS 结果

我用 Synopsys VCS V-2023.12-SP1 和独立 SVA/TB 重新建立了边沿级 witness，未复用生产 TB 的 pass 判定：

1. accepted last request 保持 exact valid 穿过完整下一 active edge：不误 fault、不二次 accept，1152-bit result 在 stall 下稳定并可正常 retire；
2. accepted non-last request 做相同 exact linger：同样不误 fault、不二次 accept，held key 保持；
3. accepted last 后分别单独 mutation `source/block/negate/last/tag`，同时把 `output_ready` 拉高：五项都在故障沿前组合 fail-closed，沿后进入 sticky fault，内部 buffered result 不被误 retire；
4. 另建 older stalled output，再注入错误 source 并同拍打开 sink：旧结果同拍被隔离，registered buffer 在 sticky fault 下保留；
5. 每项 fault 都验证 reset-only，累计 11 个 sticky checks、10 次 reset recovery；reset 后 fresh legal load/event/output 正常；
6. exact last request 在两个 active edge 之间发生 low-high 且下一边沿前恢复完整 identity：同步逻辑保持原 grace，不把它重接收为新请求。

独立 PASS：

```text
PASS M104 r3 independent VCS last_linger=1 nonlast_linger=1 between_edge_low_high=1 identity_mutations=5 older_buffer_quarantine=1 sticky_checks=11 reset_recoveries=10 accepted_events=10 macros=0
```

独立 SVA covers：last exact linger=2、non-last exact linger=1、same-cycle buffered quarantine=6。compile/sim RC 均为 0，无 assertion failure、compile warning、fatal 或 watchdog 签名。

## Complete identity 与边沿语义

r3 保存并比较完整 41-bit accepted identity：`source[3:0] + block[2:0] + negate + last + tag[31:0]`。五字段只在 `event_accept` 边沿捕获；exact linger 被 `accepted_event_grace_match` 明确压成 non-ready/non-accept。accepted last 同拍释放 `held_valid`，因此此后的任一字段 mutation 同时失去 grace 与 live key identity，组合 `illegal_request` 立即拉高，并在下一边沿锁存 sticky fault。

non-last 的边界要准确表述：exact identity linger 被 grace 压住；如果 source/block 仍匹配 live held key，而 negate/last/tag 改成一个新的合法 descriptor，RTL会把它当下一 transaction，并非非法 mutation。这保持了合法 II=1 流，不应把 post-last mutation 的 fail-closed 规则错误扩大到 non-last 新 descriptor。

## Sealed 证据与 DC filelist

- sealed input manifest 9/9、output manifest 4/4、runner 1/1，全部重新校验通过；compile/sim RC=0，PASS 行和九项 contract cover 完全一致。
- production-only DC filelist 只有 `rtl_m104/m104_held_weight_correction_broadcaster.sv`，不含 SVA/TB，SHA 为 `4507f6af3f41cae8c1c26f6779f3c33803d30e03dcbaeef36348ee905f99fd36`。
- 该 logic-only filelist 尚未被 r3 contract/sealed input manifest pin，且当前没有 M104 专用 SDC/exact DC runner；下一 launch 必须补齐这些身份。
- `docs/359` SHA 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## Findings

### P0

无。

### P1

1. 当前是 bounded directed functional evidence，不是 formal exhaustion 或 frozen workload actual-record replay。
2. descriptor transpose、SRAM response、accumulator、full schedule、DMA 仍是 port cut，且宏数为 0。
3. M104 专用 SDC/DC runner 尚不存在，production-only filelist 也未由 r3 seal pin；本轮只准入下一 DC gate。
4. same-cycle quarantine 形成 identity compare → `protocol_error` → output mask 的组合路径，必须由目标 STA 证明并保持 request 同步。

### P2

1. exact valid/identity 一直不变时 grace 可无限持续；安全但没有 liveness timeout。
2. 边沿间 low-high 仅是 synchronous digital 结论，不是 analog/CDC/metastability 证据。
3. production SVA 未分别 cover non-last linger 和五字段 post-last mutation；独立 witness 已补证，但建议并回生产回归。
4. fault/quarantine cover 含同一攻击的多周期 occupancy，不能当独立攻击数。
5. `2.6679769126038075×` 只能继续称为 M104 r2 frozen conditional same-clock service-token work ratio。

## DC admission 与性能边界

- exact-SHA sealed VCS/SVA：**GO**。
- last/non-last exact linger、no-double-accept、stall stability：**GO**。
- post-last 五字段 mutation fail-closed、older buffer quarantine、sticky/reset-only：**GO**。
- production-only filelist 内容：**GO**。
- 当前 SHA module-only logic pre-macro DC 下一门：**GO，launch 时必须 exact pin**。
- `2.6679769126038075×`：**GO（analytical conditional service-token work only）**。
- scheduled/actual-record、physical Fmax/energy、macro-inclusive/equal-area、system/headline：**NO-GO**。

DC launch 必须固定 reviewed RTL/filelist SHA、M104 SDC、TSMC28 setup/hold DB、clock period、I/O delay、uncertainty、load/fanout、compile recipe 和工具版本，并报告 cell/sequential area、FF、setup/hold、violations、unmapped cell 及 precompile operator/resource audit。

机器评审见 `m104_r3_accepted_last_event_grace_independent_hammer_review.json`。本评审只写本目录，未修改 production、contracts/results 或 `docs/359`。
