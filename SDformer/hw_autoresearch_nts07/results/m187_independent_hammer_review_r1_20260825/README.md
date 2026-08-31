# M187 独立打铁评审

结论：**91/100，`PASS_AS_K7_RTL_SCREEN_CANDIDATE_ONLY`**。建议进入 K7 RTL，但只能作为一次与 sealed M186 K8 完全同口径的筛选；现在不应取代 K8，更不能升格为 physical、FC2、FFN、system 或 headline 结果。

## 独立重算

本评审没有 import M172/M179/M187 的任何生产 analyzer。我从冻结 bit-packed payload 重新解码 bank population，重写 ping-pong 有限窗口递推，并对 120/120 FC2 payload 重做 SHA、extent 和 popcount 校验。

- 源 manifest SHA：`2a5e6e472b3897ea508f61c3727bddd97818c921b0729c0f2731b150c6d7a76e`
- M187 主 manifest：5/5 从仓库根目录校验通过
- payload：120/120，437,760,000 bytes，143,894,510 events，全部 SHA/size/popcount 通过
- `docs/359` SHA：`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`，未改
- 独立 scalar/vector recurrence：48,000 例，0 mismatch

K=1..8 全部和主结果、contract 一致：

| K | exact wall cycles |
|---:|---:|
| 1 | 430,917,270 |
| 2 | 227,554,255 |
| 3 | 160,289,465 |
| 4 | 127,581,198 |
| 5 | 109,553,951 |
| 6 | 100,331,395 |
| 7 | 97,694,539 |
| 8 | 97,607,807 |

## group 公式审计

公式 `G=max(max_i(n_i),ceil(sum_i(n_i)/K))` 成立，每轮服务剩余 population 最大的 K 个 bank 可达到该下界。

两个下界分别来自“每 bank 每 group 最多一项”和“每 group 最多 K 项”。设当前下界为 G，top-K 贪心一轮后，剩余总量不超过 `K(G-1)`，最大 bank population 也不超过 `G-1`；否则会出现至少 K+1 个大小为 G 的 bank，与总量不超过 KG 矛盾。归纳即得贪心恰好 G 轮完成。

另外对 8-bank、每 bank population 0..12 的 125,970 个排列等价 multiset、K=1..8 共 1,007,760 例穷举，0 mismatch。

## K7 数字的准确语义

- K7 比 K8 多 86,732 cycles，以 K8 为分母是 **0.088857646%**，数字正确。
- K7/K8 的总吞吐因子之比是 **99.911221%**；主材料的数字正确，但建议将文字改为“保留 K8 总 schedule throughput factor 的 99.911%”。如果“保留 speedup”指相对 K4 的净增益，正确数字是 **99.622115%**。
- 6,144 → 5,376 bit 确实是 **12.5% nominal weight-lane response width** 缩减。考虑 K7 多出的 issue，full-width replay bit-cycle 缩减是 **12.404103%**。有效 weight value 总数并未减少，也没有证明 SRAM capacity 或 energy 减少；八个 physical bank 仍在。

## 进入 K7 RTL 的理由与门槛

应该进入，因为同频的 schedule-throughput/area 只需 K7 相对 K8 节省超过 **0.088779% area** 就能打平。以 sealed M186 K8 的 37,144.673821 µm² 作为纯筛选参考，K7 需低于 37,111.697240 µm²。

但 K7 不是“直接删一条 lane”：它比 K8 新增了动态遗漏最小 bank 的 selector、8-bank 到 7-lane 的 compactor/crossbar，而 7-input tree 的 mapped depth 也可能和 8-input 相同。因此必须用一个 flattened K7 issue island 与 M186 K8 在同一 3.0 ns、同一端口、同一排除项下做 VCS/SVA + DC。若 selector/compactor 增加 cycle、破坏时序或无法节省面积，就应回退 K8。

## 打铁优先级

### P0

- M187 只允许作 K7 RTL screen；不得将 1.305919×、99.911221% 或 12.5% 当作 physical/system/headline 结果。
- K7 RTL 必须包含 top-seven selector、8-to-7 compactor、response protocol 和 7-lane signed accumulator，并与 M186 K8 做同口径 VCS/SVA/DC。

### P1

- 修正 99.911% 的文字语义，并将 12.5% 限定为 nominal weight response lane width。
- 单独报告 selector/compactor 逻辑深度、fanout 和时序，不允许被 lane 减少数字掩盖。
- 增加 weight-response latency 和 finite outstanding-request 敏感性，再决定是否保留 exact DSE cycle 口径。

### P2

- 把冻结 source manifest 和 payload archive receipt 直接加入 M187 checksum manifest。当前 analyzer 硬锁了 manifest SHA，但 README 声称 result identity 记录了路径，实际只记录了 hash。
- K7 相对 K8 多出的 86,732 cycles 中，stage2 占 53,587，后续如需查找非理想调度应优先看 stage2。

机器可读评审见 `m187_independent_hammer_review_r1.json`；完整 payload 检查和重算账本见 `independent_recompute_result.json`。
