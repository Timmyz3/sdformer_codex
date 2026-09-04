# M1129C 对 M1128C weight-service addressed-ledger 的不同作者静态打铁

结论：**GO 仅授权下一步 additive iterator instrumentation source；STOP canonical ledger 与 51.84M 全量重放。**

真实 M1102/M1016 权威接口只有 receipt 的 count/首 beat/区间摘要，以及 `weight_task(start, beats, half_slot)`。它们不含原生 READ/WRITE、logical bank、native slices、local row、bytes/byte-enable、native-macro activation 或 service-beat 到 store transaction 的 exact-once 关系。因此不得从 count、`weight_beat_first` 或容量几何反推事务。

Canonical iterator 在 row reader 打开前 fail closed：canonical rows = 0，real weight transactions = 0。

有界 synthetic 独立重构了 24×128×128b 1RW 映射：9 事务，其中 6 store、3 read、3 个显式 stall，6/6 service beat exact-once，最终 native conflict=0，half-slot overlap=0。18 类字段、冲突、exact-once 和越权攻击全部被拒绝。这只证明 schema/映射/仲裁机械性，不是 H67 canonical 流量、周期或能量证据。

下一步唯一合法的开发是：在不改动冻结 M1102/M1016 语义的前提下，添加内部真实 service/refill 事件的 addressed instrumentation iterator；每个字段都必须来自真实事件，不得从聚合计数还原。
