# M1541｜M1540 ep34 剪枝、稀疏与跳过候选总表独立打铁

裁决：**PASS_WITH_THREE_P1_GATE_REPAIRS**。M1540 的证据身份、静态数字、候选去重与“不把 event / bytes / static opportunity 写成 cycle、energy、AEE”边界成立；它只授权三条 fast-kill，没有产生新性能或 RTL claim。三个 P1 都是下一版 fast-kill 合同必须恢复的公平性门，不推翻候选总表。

## 1. 独立核验通过项

- M1540 内层 `SHA256SUMS` 与外层 seal 通过；`review.md` / `review.json` 当前 SHA256 分别为 `f1d5754d5e5b5fbb5cad8724d41041e8feb3be2236a343b351aa1d4fe89c3d5d` / `218e3d23fae126ddc4a8655f8e9cd7cb762276ab87c7494b7ad05f6e469730bb`。
- M1529、M1534、M1535、M1537 与 M1538 的被 pin source SHA 均与各自 sealed artifact 一致；ep34 checkpoint SHA `4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48` 与 M1458 manifest 一致；受保护 `docs/359` SHA 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
- M1537 的 N:M 静态数字被正确转述：`4:8` 删除 `23.00%--25.00%` L1 mass，`8:16` 删除 `21.23%--23.01%`；M1540 明确没有把它们写成 `2x`、cycle、traffic、energy 或 AEE。
- M1538 的两个 P1 已正确传递：patch scope 必须分离混入的 ATLIF temporal coefficient，且 flat grouping 必须改为真实 hardware-row-local grouping。M1540 的 `local cycle >=1.20x` 比 M1538 的一般 `>=1.15x` 门更严格，同时与 M1535 的 S5 专用门一致，不构成放宽。
- 用户要求暂缓的 M501 `27.5167%` 已同时从 human/machine-readable 实现队列排除；M1540 只称其为 source-event reduction，并明确写明“不是时间”。它没有出现在 priority list、fast-kill list 或 paper slot。
- claim boundary 完整关闭 `cycles/speedup/traffic/energy/AEE/RTL/headline`。`26.506%`、`3.154%` 和 N:M mass 只用于 profiling / candidate priority，没有冒充 skip rate、latency 或系统收益。

## 2. 去重与优先级核验

- S1 只打 ep34 的 non-binary raw ingress / analog residual，和已死的 bottleneck `{0, layer-constant}` G7 分开；其 `epsilon=0` exact 子集与 fetch-before-compute 条件保留。
- S2 只有在“一个 metadata read 控制整块 fetch + metadata 至少缩小 `8x` + dynamic keep/drop witness”成立时才不退化为旧 G11；这个 collision gate 已写入 M1540。
- TSBG 只复用 FC1/FC2 跨 token 的 weight row，不删除 contributor，不冒充 C1 product capture 或 M501 adjacent overlap；普通同容量 row buffer 被列为强 baseline。
- S1/S2/TSBG 同列 T0 合理。machine-readable `priority_order=[S1,S2,TSBG]` 只能解释为 fast-kill 执行顺序：M1535 推荐先测 S1/S2，而 M1534 认为 TSBG 是最值得测的无损候选；现有 aggregate trace 尚不足以证明三者的科学 ROI 排名。
- ACES、LBWC、ARPE、PSTP 与 SD-N:M 均被压到 supporting / blocked / training-only，没有被包装成第四个并列 novelty。

## 3. 三个 P1 公平性缺口

1. **P1-S1：显式 metadata veto 未完整传递。** M1535 要求 `metadata + beta read >= 25%` 被省 weight bytes 时一票否决，并要求 beta 端口造成的 slowdown `>5%` 时 NO-GO。M1540 只写了“metadata、debt、bank/port conflict 全收费”，但 human/json gate 没保留 `25%` veto。S1 fast-kill 合同必须恢复该门，避免以未计 metadata 的 byte/energy proxy 晋级。
2. **P1-S2：metadata 总容量门丢失。** M1535 除“比旧 G11 per-source metadata 小 `8x`”外，还要求 total metadata `<=2%` of weight bytes。M1540 human/json 只传递前者。S2 fast-kill 合同必须同时恢复 `<=2%`，并对 pointer、bank、read energy 和 debt state 收费。
3. **P1-TSBG：每序列 cycle floor 丢失。** M1534 的 cycle 分支是 FC1+FC2 ratio-of-sums `>=1.15x` 且每 sequence `>=1.05x`。M1540 只保留 aggregate `>=1.15x`。TSBG fast-kill 必须恢复 per-sequence floor，防止均值掩盖某条 DSEC sequence 的倒退；energy-only 分支仍须满足 cycle regression `<=5%`。

## 4. 最终许可边界

M1540 可作为候选总表引用并授权一次共享增量 capture。任何 production fast-kill 合同须先修复上述三个 P1；在得到同资源 cycles、物理 bytes、memory energy 与 paired AEE 前，不授权新 RTL，也不得使用“时间减少”“系统加速”或“节能”措辞。独立评分：**94/100，P0=0，P1=3**。
