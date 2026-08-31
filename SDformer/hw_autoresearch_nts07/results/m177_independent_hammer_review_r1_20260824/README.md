# M177 r2 独立打铁评审

结论：**86/100，模块级通过，但 producer P0 未关闭。** M177 r2 已经把 r1 的
结构 timing loop 拆掉；exact-SHA VCS/SVA、独立 fresh seed 和 3 ns logic-only DC
均通过。M177 相对 M175 只增加 `0.413820%` cell area，在 native/preindexed
前提下得到的条件性 logic-only throughput density 是 `1.088167x`。这仍不是物理、
完整 FC2、系统或 headline 加速。

## 独立核验结果

- r1 DC：runner exit `21`；日志明确出现 `OPT-150`、`OPT-314` 和 precompile
  `TIM-209`，所以 r1 数字不可引用。
- r2 RTL：`residual_candidate_load` 与 `scan_candidate_load` 已分离，
  `residual_will_clear` 不再依赖 `scan_accept/scan_ready`。r2 DC `rc=0`，日志中
  `OPT-150=0`、`OPT-314=0`、Error/Fatal `=0`。
- sealed VCS：15/15 cover，0 assertion failure，四个 protocol attack 均 fail-close；
  合法 done+EOT 同拍接受。
- fresh seed `177991`：复用 exact-SHA sealed binary，`rc=0`，仍为 15/15、0 failure，
  四攻击与合法 rearm 均通过；随机 stall 从 124 变为 135，证明不是原 backpressure
  时序的机械重播。
- M175/M177 matched DC：同为 TSMC28、3 ns、ideal clock、ZeroWireload、0 macro。

| 指标 | M175 raw96 | M177 indexed96 r2 | M177/M175 |
|---|---:|---:|---:|
| Cell area (µm²) | 1309.266002 | 1314.684003 | 1.004138x |
| Cells | 1783 | 1838 | 1.030847x |
| Sequential cells | 236 | 235 | 0.995763x |
| Logic levels | 55 | 61 | 1.109091x |
| Setup slack (ns) | 0.4731 | 0.3470 | — |
| Hold slack (ns) | 0.0003 | 0.0002 | — |

## P0

M177 只消费已经生成的 sparse index，并没有实现 index producer。M176 独立审计已经
严格区分两种来源：

- ATLIF-native/preindexed：raw96/indexed96 K4 analytic ratio `1.092670x`；
- posthoc raw96 scanner：即使按乐观 release-aware 模型也要 `159,902,252` cycles，
  比 raw96 的 `157,504,597` 慢 `1.522%`。

因此 producer、finite FIFO 和 descriptor memory 未落地前，M176/M177 不能升级成
physical/complete-FC2/system/headline speedup。

## P1/P2

- P1：没有把 120 个冻结 payload 生成的完整 indexed stream 重放进 RTL。
- P1：合法 done+EOT 只检查同拍接受和无 fault，没有 scoreboard 新 zero token 后续
  done tag/had-event。
- P1：r2 mapped netlist 尚无 Formality receipt；当前 DC 还是 ideal/ZeroWireload/0 macro，
  且 hold margin 仅 0.0002 ns。
- P2：DC 有两条 VER-318 signed/unsigned conversion warning。
- P2：官方 runner 的 PASS regex 锁死 seed-specific stall 计数；应另加 invariant-based
  多 seed 门禁。
- P2：基础 contract/修正 overlay 是 pre-run 状态，pass admission 目前由 RUN_COMPLETE
  receipts 承载，可再补一份只引用 sealed SHA 的 admission receipt。

## 下一里程碑

不要继续优化 consumer selector。下一步应做最小 ATLIF-native index tap + 两项 FIFO，
把冻结 payload materialize 成 descriptor stream，与 M177 做端到端 conservation miter，
并在相同 port bandwidth 下对比 producer+consumer 与 M175 raw96。producer-composed K4
仍有优势后，再接 weight response、M169 arithmetic 和 accumulator context。

所有 `complete_fc2/physical_speedup/system_speedup/headline` admission 均保持 `false`；
`docs/359` SHA 仍为
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
