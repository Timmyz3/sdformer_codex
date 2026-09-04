# M102 r4 symmetric accepted-request grace：独立打铁复审

日期：2026-08-24

结论：**93/100，P0=0 / P1=4 / P2=5。当前 exact SHA 准入 production-only common-period logic-only DC 下一门。** 这只是允许启动匹配的 TSMC28 ideal-clock、ZeroWireload、common-period 综合，不是 DC/PPA 已通过，更不是 physical、equal-area、system 或 headline admission。

## 核心结果

r3 的非对称 P0 已关闭。独立 Synopsys VCS V-2023.12-SP1 + SVA witness 分别从 baseline/candidate 公共接口重建四类顺序：

1. final request 刚接受后继续保持 exact valid 跨越一个完整 active edge：不 fault、不 double accept，旧结果保持可见并能正常接受；
2. baseline 的 source/block/beat/tag 与 candidate 的 kind/pattern/source/block/beat/negate/tag 分别逐字段单独 mutation，同时把 `output_ready` 从 0 拉到 1：两侧均在故障沿之前组合隔离旧 M82 output，故障沿后 M82 仍保留；
3. registered request fault 持续隔离，candidate 的 phase reload 连续两个边沿均不 ready，只有同步 reset 恢复；
4. valid 严格在两个 active sampling edge 之间低再高，且 identity 完全相同：两侧都保留原 grace，不误判为新请求，也不重复接受。

独立 PASS：

```text
PASS M102_R4_INDEPENDENT_BASELINE grace_full_edges=1 glitch_low_high=1 identity_mutations=4 sticky_checks=8 reset_recoveries=4 no_double_accept=1 result_visible=1
PASS M102_R4_INDEPENDENT_CANDIDATE grace_full_edges=1 glitch_low_high=1 identity_mutations=7 sticky_checks=14 phase_reload_blocks=1 reset_recoveries=7 no_double_accept=1 result_visible=1
```

独立 SVA covers：

| side | grace | between-edge low/high | identity mutation | fault occupancy | reload block | reset recovery |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 1 | 1 | 4 | 12 | 0 | 4 |
| candidate | 1 | 1 | 7 | 22 | 2 | 7 |

两次 compile/sim RC 均为 0，无 assertion failure、compile warning、fatal 或 watchdog 签名。

## Sealed 证据与 filelist

- sealed input manifest：15/15；output manifest：7/7；runner：1/1，全部重新校验通过。
- baseline/candidate 主 suite PASS 行和合同 cover 逐项一致。
- baseline 主 covers：II3=70、stall=30、signed=120、fault=21、same-cycle=1、grace=1、buffer quarantine=3、reset recovery=14。
- candidate 主 covers：PWP=8、正/负 correction=1/1、fallback=2、stall=4、fault=46、buffer quarantine=6、same-cycle=1、grace=1、fault reload=1、metadata error=1、PWP→correction seam=2。
- 两份 production-only filelist 均只包含共同 M82 与对应 production top，不含 SVA/TB，并已被 r4 contract 与 sealed input manifest exact-SHA pin。
- `docs/359` SHA 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## Findings

### P0

无。

### P1

1. 当前仍是 directed functional evidence，不是 frozen workload population 的 actual RTL replay。
2. SRAM、response mux、decoder/ECC、memory enable、matcher/enumerator、DMA、accumulator仍是 port cut。
3. production top/filelist 已准入，但 common-period grid、SDC、setup/hold DB、compile recipe 与 precompile resource audit 应在下一 DC launch manifest 中 exact pin。
4. candidate 合法路径仍只有 8 vectors/一个 metadata context，广泛 metadata 边界与 seam population 未穷举。

### P2

1. exact grace 在 valid 一直保持且 identity 不变时可无限持续；不会 double accept，但没有 liveness timeout。
2. 边沿间 low-high 结论只属于 synchronous digital observation，不是 analog/CDC/metastability 证据。
3. fault/quarantine cover 包含同一攻击的多周期 occupancy，不能当独立攻击数。
4. runner 虽 exact 校验，仍使用独立 one-entry manifest。
5. `1.4094204844392757×` 只能叫 analytical service-token work ratio。

## DC admission 与性能边界

- exact-SHA sealed VCS/SVA：**GO**。
- A/B exact accepted-request grace、no-double-accept、result visibility：**GO**。
- A/B 全 identity mutation same-cycle quarantine：**GO**。
- sticky reset-only recovery：**GO**。
- production-only filelists：**GO**。
- 当前 SHA common-period logic-only DC 下一门：**GO**。
- `1.4094204844392757×`：**GO（analytical service-token work only）**。
- scheduled/actual-record runtime、physical Fmax/energy、macro-inclusive/equal-area、system/headline：**NO-GO**。

DC launch 必须固定同一 TSMC28 setup/min DB、common period grid、ideal clock、ZeroWireload、I/O/uncertainty/load/fanout、compile recipe 和 tool version，并对两侧报告 area、FF、setup/hold、violations 与 precompile operator/resource audit。

机器评审见 `m102_r4_symmetric_accepted_request_grace_independent_hammer_review.json`。本评审只写本目录，未修改 production、contracts/results 或 `docs/359`。
