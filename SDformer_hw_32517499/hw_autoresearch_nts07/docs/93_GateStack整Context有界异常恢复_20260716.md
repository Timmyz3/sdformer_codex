# GateStack 整 Context 有界异常恢复

## 1. 为什么需要恢复结构

原来的 sticky `protocol_error` 只能说明发生了错误。如果 slot word 的 payload tag/mode 不匹配，router 会停止向 decoder 送数；decoder 和 backend 可能永远不产生 done，scheduler 也会永久停在当前 head。对长 trace 和芯片运行而言，“检测到错误但无法退休”仍是死锁。

## 2. 采用的方案

当前单 context 顶层采用整 context flush：

1. abort controller 从 group accept 开始计时并保存 group tag；
2. 任意 fabric sticky error 或 watchdog timeout 触发单周期 `fabric_reset_pulse`；
3. scheduler、slot/cache、control、word router、done guard 和 projection 在同一同步 reset 域清空；
4. abort controller 不随 fabric reset 清空，继续持有原 group tag；
5. 对外返回一次 `group_done(error=1)`；
6. response 被接收前，`group_ready` 保持为 0。

## 3. 设计取舍

整 context flush 的优点是状态边界清楚、可证明有界、不要求给所有旧 decoder 增加独立 cancel 端口。代价是健康 payload/cache 也被清除，需要上游重装。因此它适合作为第一版 ASIC 的 fail-stop 恢复，不应包装成低代价细粒度容错。

## 4. 验证结论

- 叶模块覆盖正常完成、fabric error、watchdog timeout 和 normal-error cleanup；
- 顶层覆盖 missing-slot 触发的真实集成 abort；
- abort completion 在反压下保持 tag/error；
- abort response 未退休前禁止新 group；
- 所有相关 Icarus、Verilator+SVA、Yosys、Erie 检查通过。

详细数字见 `results/gatestack_context_abort_20260716/report.md`。

## 5. 下一步

IPD descriptor 自动捕获与 cache fill 已在下一阶段完成。当前下一优先级是用 H67 ordered trace 驱动默认 162×32 full-top 长回归，并冻结 SRAM/DC 约束。
