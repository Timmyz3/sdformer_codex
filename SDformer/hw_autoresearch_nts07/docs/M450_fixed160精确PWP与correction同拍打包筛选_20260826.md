# M450 fixed-160B 精确 PWP / correction 同拍打包筛选（2026-08-26）

## 结论

当前 M430 payload 合同下，该方向 **NO-GO**，不投 RTL：

- strong-zero：`742,148,386` cycles；
- M430 separate：`517,041,352` cycles；
- fixed-160B atomic co-pack：`517,041,352` cycles；
- 相对 M430：`1.000000x`，低于冻结门槛 `1.10x`。

这只是四个冻结 H67 ep35 bottleneck Conv3x3 的 exact packing/cycle 筛选，不是 RTL 测得倍速、系统倍速、资源归一化结果或论文 headline。

## 为什么 160 B 放不下

M433 的当前 PWP 数据面为：

- low：`96 lanes × 8 bit = 96 B`；
- wide high：`96 lanes × 4 bit = 48 B`，物理 sidecar 为 `64 B`，因此 wide 只空 `16 B`；
- narrow：high side 全零，最大空 `64 B`；
- 物理 payload 总预算固定 `160 B/cycle`，本轮禁止扩端口。

M104 correction 数据面为：

- 96 个冻结 INT8 weights：`96 B`；
- 每个 correction vector 共享一个 negate bit；
- event metadata 为 `source4 + block3 + negate1 + last1 + tag32 = 41 bit`；
- arithmetic output 为 `96 × signed12 = 144 B`，但输入 payload 仍按冻结 INT8 的 `96 B` 收费。

本轮对 metadata 采用最有利假设：允许复用既有 sideband，payload 收费为 0 B。即使如此，完整 correction payload 仍大于 narrow 的最大 64 B slack。

进一步对四层全部 96-lane weight block、正负两种 correction 方向做最小 signed fixed-width 审计，共 `442,368` 个 sign-conditioned vector case：

| signed bits/lane | cases | payload/vector |
|---:|---:|---:|
| 6 | 6 | 72 B |
| 7 | 70,724 | 84 B |
| 8 | 371,638 | 96 B |

没有任何 vector 小于等于 64 B；因此 atomic co-pack 候选为 0。权重实际范围 `[-127,127]`，`-128` 数量为 0。

## 单遍 heldout 账本

合同在读取 heldout ledger 前冻结。分析只顺序读取一次 M430 已封存的 `per_phase_heldout_dual_replay.csv`，不再读取 raw M40 payload，也不改 catalog：

- phases：`17,280`；
- source rows：`51,840,000`；
- PWP rows：`15,909,646`；
- PWP output-block issues：`127,277,168`；
- correction ops/block：`38,055,489`；
- correction output-block issues：`304,443,912`；
- runtime narrow/wide PWP issues：`18,267,843 / 109,009,325`。

完整 96 B correction 的全局 payload-fragment pooling 即使忽略 row/block/destination/phase identity、buffer、assembly、compute port 和 accumulator 冲突，最多也只有 `1.062353x`。它不是可执行点。

更激进地假设所有 correction 都等于全库最小的 72 B，同时仍允许上述不合法的全局 pooling，则最多隐藏：

`floor(2,913,291,152 / 72) = 40,462,377` 个 correction issue，

得到 `476,578,975` cycles、相对 M430 `1.084902x`，仍低于 `1.10x`。这是 post-seal 的纯代数 generosity ceiling，不是新的 heldout pass，更不是硬件点。

## 决策边界

关闭的是“当前 dense/fixed-width correction payload 在同一 fixed-160B PWP issue 内 atomic co-pack”方向。若未来改为 weight-stationary 本地持有、只传 source/sign descriptor，那是新的 SRAM/调度/多更新端口架构，必须重新收费，不能借用本轮结果。

证据：

- contract：`contracts/m450_fixed160_exact_pwp_correction_copack_contract_r1_20260826.json`
- analyzer：`system_simulator/scripts/analyze_m450_fixed160_exact_pwp_correction_copack.py`
- sealed result：`results/m450_fixed160_exact_copack_screen_r1_20260826/`
- docs/359 SHA256：`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`
