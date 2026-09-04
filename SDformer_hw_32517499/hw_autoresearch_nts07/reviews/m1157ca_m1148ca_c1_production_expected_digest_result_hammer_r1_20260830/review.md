# M1157CA 异作者 production expected-digest result hammer

## 结论

**PASS，但只准入 expected-digest authority artifact。** M1148CA 的外封、attempt 和输入 schedule 身份一致；三轴计数、字节数、激活数、零 stall 和 authority-ID 均可独立复核。

这个 PASS **不是** real producer replay，也不是 traffic/cycle/energy/speedup 或论文性能证据。后续仍必须让真实 producer 流以本 authority 作对比。

## 独立复核

- 全量扫描 `836,268,740 B` / `2,436,480` 条 schedule record；原始 SHA-256 为 `4d4e0e6396ac1061aca7ada142bc2761bf12a785e5373640a28503e3d73a0a81`。
- 严格 JSON、字段集、task 坐标、每 task 的 candidate/strongest-zero/same-coordinate-bit 顺序与 schedule provenance 全部通过。
- 每轴 `70,853,184` events、`9,069,207,552 B`、`566,825,472` native activations；三轴共 `212,559,552` events。
- 全 schedule 的每轴 requested interval 不重叠，因此 24-slice scheduler 的 stall 为 0。
- authority-ID 按 counts、三轴 digest、schedule SHA 和 M1135 语义 SHA 独立重算，命中 `a53f0141ff9f01b32ed8920c0c3fc10a2d70848773e9b99e02b8905ea05a6fbf`。
- 未导入/调用作者 compiler。独立实现在 24-event bounded case 命中三个 frozen golden digest，并对首/中/尾 6 个 task 的 18 条 record 生成独立 event fingerprint。
- AST 只读审计确认 compiler 保留的是固定三轴统计/digest 与 `3×24` scheduler slots，`consume_schedule_record` 无 per-event append；输出目录无 event-like 文件。
- 唯一 result namespace 和唯一 consumed-attempt namespace 存在；`automatic_retry=false`。
- 11 个边界攻击全部被拒绝，包括重复 JSON key、额外字段、轴顺序、provenance、计数/字节混淆、digest 篡改、history/event-output/retry 与 claim upgrade。

## 明确限制

本 hammer 没有重新序列化全部 2.125 亿 events。production digest 的验证由“封存 authority 身份重算 + bounded golden + 抽样 fingerprint”组成，不得说成第二次 full replay。

详细机械证据见 `mechanical_checks.json`。`docs/359` 保持 SHA-256 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
