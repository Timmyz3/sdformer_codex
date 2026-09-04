# M893 M890 decoder GTLS source fresh independent hammer

## 裁决

**PASS100（仅 bounded source exact）。** M890 在封存身份下通过 synthetic 1K/10K、real D0/A1/t0 1K/10K 的 M768↔M861↔GTLS 全 endpoint 比对，以及 real 100K 的 M861↔GTLS 比对。可授权不同作者编写一份 **inert** full-row release request；本评审不授权 full-row 执行。

## 数值结果

- pytest：9/9。
- real 100K：100,000 expanded requests，24,852 compressed transactions，terminal live peak 50，packed events 5,594,880 B，全 endpoint / commit digest / terminal readiness / port calendar / six cycle classes 与 M861 一致。
- closed-form：31,680 组 `(count, base, service, distance, q)` 与逐 request recurrence 一致；端口忙、outstanding 未清空、count<4 均禁止 closed form。
- 资源攻击：33 组 1RW/1R1W q−1/q/q+1 × latency/beat 通过；非对称 bank 正确回退；same-cycle response-slot 复用与冻结调度一致。
- liveness / packed events / shard / hash-domain 攻击全部通过。

## 缩放预检是红灯

独立 GTLS-only real 100K bounded 进程（`retain_details=false`）用时 3.58 s，峰值 RSS 904,388 KiB，已是未来 512 MiB 门的 1.725×。按 38,672,612 request 线性诊断外推：

- packed events 约 2,063.45 MiB，为 512 MiB 门的 4.03×；
- packed IR 序列化字节约 5,953.03 MiB。

这些是缩放诊断，不是 full-row 实测；但足以禁止直接执行。下一版必须同时修复 payload 输入常驻、compressed transaction 常驻和 per-request packed-event 常驻，再由新鲜 hammer 重新判定 100×/512 MiB gate。

### 最小 successor：RUN-GTLS

不改冻结调度算法，只把 per-request packed-event endpoint 换成 **maximal half-open runs + counted arithmetic progressions**，在 six-class priority reduction 时在线消费/合并，不再物化每个 request 的 issue endpoint。successor 必须先在 real 100K 上对每个 endpoint、port calendar、terminal readiness 和 six cycle classes 与 M861 exact miter，并证明 full-row **combined packed-state projection ≤512 MiB**。在这两道门通过之前，仍然禁止 full-row 执行。

## 边界

full-first-row=false，full-population=false，production=false，decoder-complete=false，cycles/speedup-citable=false，paper-citable=false。`docs/359` SHA 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
