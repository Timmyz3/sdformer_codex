# M1133C｜M1132C upstream weight-event producer 独立 hammer

结论：**PASS，且 canonical 继续 STOP**。

独立检查确认：

- `emit_refill_event` 仅接受 17 个 keyword-only producer 字段；
- 每次调用只构造、校验并下沉一个精确 M1130C WRITE event；
- transaction、service beat、exact-once ID 三类身份均独立去重，跨 axis 的数值复用不会误冲突；
- 受控三轴 trace 共 6 event，one-call delta 全为 1；同周期 1RW 冲突被显式化为 3 个 stall，调度后残余冲突为 0；
- 地址、slice、byte enable、activation、ordinal、provenance 和 op 边界全部 fail-closed；
- sink 异常原样传播，且不会提交 ID/beat/transaction 或 emitted count；
- 不存在 aggregate/geometry fallback，canonical iterator 在 hook 缺失时于 yield/row-open 前停止。

机械 hammer 共 206 checks、53 attacks。冻结 M1016、M1102、M1130C 与 docs/359 均未改变。

本里程碑仅授权下一步撰写 additive real producer hook source；不授权现在集成 hook、打开 canonical row、跑 51.84M、EDA/GPU/remote 或升级性能指标。
