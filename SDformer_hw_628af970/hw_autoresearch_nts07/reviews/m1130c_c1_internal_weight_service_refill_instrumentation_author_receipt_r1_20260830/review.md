# M1130C internal weight service/refill instrumentation 作者回执

结论：源码接口、合同和有界 synthetic 通过作者侧检查；canonical 仍在 row reader 打开前 STOP，只授权不同作者静态 hammer。

深入 M1016 真实产生点后确认，`PackingAudit.weight_task(start, beats, half_slot)` 仅保留区间与 half-slot overlap 计数，没有逐 beat addressed event。M1056/M1102 的原生 port event 是 psum 事件，不能冒充 weight refill。

M1130C 已定义严格的 producer-supplied event 接口：op、bank/half-slot、logical/local row、native slices、bytes/BE、native activation、service beat、transaction ordinal、exact-once ID 和 row provenance 必须在真实事件产生时传入，禁止从 aggregate count、first beat、interval 或容量几何反推。

有界三轴 synthetic 结果为 9 events / 6 writes / 3 reads / 6 unique exact-once IDs / 3 stalls / 0 conflicts / 0 half-slot overlaps。这不是 H67 canonical 事务证据。
