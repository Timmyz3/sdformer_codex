# M1132C additive upstream weight-event producer author receipt

新增 source 提供 `PerBeatAddressedWeightRefillProducer.emit_refill_event()`：17 个字段全部为 keyword-only producer 输入；每次调用在 schema、地址映射和 exact-once 校验后向 sink 发出一个精确 M1130C WRITE event。接口不接受 count、`weight_beat_first`、start/beats、capacity 或未知字段，不存在 aggregate fallback。

Bounded synthetic 在三个 axis 各发两个 refill beat：共 6 writes、6 unique exact-once IDs；接 M1130C 1RW scheduler 后 3 个显式 stall、0 最终 conflict。33 个缺字段、aggregate、映射、重复、sink 异常与 canonical escape 攻击均被拒。

冻结 M1016/M1102/M1130C 未修改，真实 callsite 尚未集成；canonical 保持 `rows=0/events=0/STOP`。本轮没有运行 51.84M、EDA、RTL、GPU 或 remote，也没有修改 docs/359。

唯一授权是不同作者静态 hammer M1132C source/contract/author receipt；不授权真实 hook、canonical row open、full replay 或性能/能量声明。
