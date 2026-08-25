# M102 r3 same-cycle fault quarantine：独立打铁复审

日期：2026-08-24

结论：**84/100，P0=1 / P1=5 / P2=5。sealed r3 matched SHA 对 production-only common-period DC 为 NO-GO。**

r3 对 candidate 的两个 r2 P0 修复是真实的：独立 Synopsys VCS witness 确认，同一组合窗口内注入 reserved request 并把 `output_ready` 从 0 拉到 1 后，时钟沿前已经是 `protocol_error=1 / output_valid=0 / output_accept=0`，故障沿后 M82 仍保存旧 tag/payload；其后把 `phase_load_valid` 连续保持 3 个故障沿，`phase_load_ready` 始终为 0，request fault 与 M82 payload 均未被清除，只有同步 reset 恢复。

但 matched denominator 的 frozen baseline SHA `29862d...` 没有同样修复。第二个独立 VCS counterexample 在 baseline 公共 lookup 接口重建相同状态，得到：

```text
COUNTEREXAMPLE M102_R3_BASELINE_PREEDGE semantic_valid=0 protocol_error=0 output_valid=1 output_accept=1 m82_valid=1
COUNTEREXAMPLE_CONFIRMED M102_R3 baseline_old_output_retired_on_invalid_edge=1
```

因此 r3 的全局 `same_cycle...quarantined=true` 只对 candidate 成立，不能推广到 frozen matched A/B。r3 production-only DC 不准入。

## Exact-SHA 与 sealed 证据

- sealed input manifest：13/13 通过；output manifest：7/7 通过；runner：1/1 通过。
- baseline/candidate compile 与 sim RC 均为 0；PASS 行和全部合同 cover 逐项一致；未发现 compile warning、assertion failure、fatal 或 watchdog 签名。
- baseline covers：II3=70、stall=28、signed boundary=118、protocol fault=12、reset recovery=12。该 suite 没有 same-cycle release attack。
- candidate covers：PWP=7、正 correction=1、负 correction=1、fallback=2、stall=3、protocol fault=46、buffer quarantine=6、same-cycle=1、faulted reload=1、metadata error=1、PWP→correction seam=2。
- `docs/359` SHA 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

独立 candidate witness：

```text
WITNESS M102_R3_PREEDGE_QUARANTINE output_valid=0 output_accept=0 protocol_error=1 m82_valid=1 m82_accept=0
WITNESS M102_R3_POSTEDGE_RETENTION request_fault=1 m82_valid=1 old_output_retired=0
WITNESS M102_R3_PHASE_RELOAD_BLOCKED edges=3 request_fault=1 m82_valid=1
WITNESS_CONFIRMED M102_R3 reset_only_recovery=1 post_reset_phase_load=1
```

两个 witness 均使用 Synopsys VCS V-2023.12-SP1，compile/sim RC=0；没有引用开源工具结论。

## 账本复算与 claim 边界

- baseline：`371,461,096 × 3 = 1,114,383,288` service cycles。
- PWP：`11,164,284×3 + 32,360,036×4 + 13,936,011×4 + 1,509,043×5 = 226,222,255` cycles。
- correction/fallback：`188,148,490 × 3 = 564,445,470` cycles。
- candidate service：`790,667,725` cycles。
- same-clock service-slot work ratio：`1.4094204844392757×`。
- 8,640 phases 的 parser 为 `1,105,920` edges，另计每 phase 一个 load edge后为 `791,782,285` candidate cycles，对应 `1.407436500047485×` upper bound。

数值与 r2 ledger 一致，但 r2 JSON 的 functional identity 指向 r2 logs，且其中的 `fault_quarantine=true` 已被 r2 hammer 反驳。因此本评审只复用并独立复算 workload arithmetic，不导入 r2 functional admission。上述比率仍是 analytical service-island 数字，不是 actual-record RTL replay、physical/frequency-normalized、equal-area、system 或 headline speedup。

## 新 findings

### P0

1. `M102-R3-H-P0-01`：frozen baseline 仍有 same-cycle invalid-request acceptance window；旧 stalled M82 output 会在登记 request fault 的同一沿被接受并清除。

### P1

1. r3 contract/RUN_COMPLETE 的全局 quarantine wording 与测试不对称：candidate 有 same-cycle attack，baseline 没有。
2. 新增两份 production-only filelist 内容正确，各只含 M82 与对应 production top，但它们在 r3 seal 之后产生，未被 r3 contract/input manifest pin。
3. r3 复用的 r2 ledger functional identity 已过时，只可复用算术。
4. candidate 仅 8 个合法向量/一个 metadata context，baseline 仅 90 个 directed vectors；没有 frozen population actual replay。
5. SRAM、response mux、decoder/ECC、bank enable、matcher/enumerator、DMA、accumulator仍是 port cut；`bank_select_pwp` 也不是 fault-qualified memory enable。

### P2

1. `buffer quarantine=6`、`protocol fault=46` 包含同一攻击的多周期 occupancy，不是对应数量的独立攻击。
2. runner 精确校验，但仍在 primary input/output manifest 外单独记录。
3. phase-reload attack 针对 request fault；metadata poison 下连续 reload 的同序 witness 仍缺。
4. `1.409420484×`、parser/load `1.407436500×` 与 M88 bounded `1.409375695×` 口径不同。
5. same bandwidth 不等于 equal area；存储、外部 mux、宏端口与功耗尚未对齐。

## DC admission

- frozen r3 exact-SHA provenance：**GO**。
- candidate same-cycle quarantine：**GO（bounded directed + independent VCS）**。
- candidate faulted phase-load block / reset-only recovery：**GO（bounded directed + independent VCS）**。
- frozen baseline same-cycle quarantine：**NO-GO（独立反例确认）**。
- frozen r3 matched SHA production-only common-period DC：**NO-GO**。
- 两份 production-only filelist 的结构：**GO**，但必须在下一合同/launch manifest 中 pin。
- analytical ratios：**GO（仅对应 analytical boundary）**。
- actual-record、physical、equal-area、system、headline：**NO-GO**。

反例报告后，工作树 baseline 已进入 draft r4 repair：RTL `597746db...`、SVA `7c49fd7c...`、TB `2ee13e9e...`。这些不是本 r3 sealed identity，本评审不准入它们；r4 必须用新合同、exact-SHA VCS/SVA、独立 witness 与 pinned production-only filelists重新封存，然后才可准入 logic-only common-period DC。

机器评审见 `m102_r3_same_cycle_fault_quarantine_independent_hammer_review.json`，算术复算见 `recomputed_m102_r3_ledger.json`。本评审未修改生产 RTL、contracts/results 或 `docs/359`。
