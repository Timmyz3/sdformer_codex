# M460R5 唯一一次结果独立 fail-closed 评审

唯一结论：`PASS_CAPTURE_ONLY_NO_EXECUTABLE_OR_PERFORMANCE_ADMISSION`。

评分 100/100；P0/P1/P2 = 0/0/0。M460R5 可以准入为冻结 H67 ep35、`no_running/current-batch`、Zurich S10 的 reduction-only post-compute oracle capture。它不准入可执行跳过、Delta-AEE/valid825、硬件周期、能量、PPA、全网/系统倍速或 headline。

## 独立重验结果

- R5 contract、30-leaf 启动清单及外封通过；上游 R4 的 14-leaf launch、3-leaf remote result、5-leaf independent review 三组授权根全部通过。
- R5 top 131 leaves、payload 123 leaves 逐项物理哈希通过，两个 outer seal 均与内层 manifest 一致。
- 120 个 NPZ、120 个 sample/module record、10 个样本、12 个 FFN、5,580,000 个 token 构成完整笛卡尔积；所有 token finite。
- 120 个 NPZ 物理 SHA 和 1,440 个数组逻辑 SHA/dtype/shape/elements/bytes 全部通过。
- `rho = ||F||1 / max(||x||1, 2^-24)`、exact-zero、7 档 strict/equal/inclusive、source-work 与 dense-MAC receipts 全量重算为 0 mismatch。
- checkpoint SHA 为 `4f33e086...45158`，load missing/unexpected 均为 0；BN 策略为 `no_running/current-batch`，源配置为 `no_running`，共切换 78 个 BN module。
- one-shot marker 显示仅消耗 1 次；capture 前 4 个和 capture 后 1 个 idle snapshot 均为 0 GPU context / 0 ML process，时间序满足 preflight ≤ marker < postflight。
- 23 类对抗篡改均被拒绝，无 unexpected acceptance。

## 数值观察与红线

从 tau=0 到 `2^-8`，strict skip token 全为 0。最大档 `2^-6` 只有 87 / 5,580,000，即 0.00155914% 的 post-compute oracle token；对应 receipt 中 2,910,144 source-work units。这个量级是捕获后的机会统计，既没有 pre-compute predictor，也没有 Delta-AEE 或 cycle model，因此不得换算或表述成任何可执行/周期/系统加速。

## 可复现证据

- `independent_m460r5_result_hammer.py`: `7d6f616d9bd084a1f139ec8949a63ad289c3aba6976580f1f9f734f51d1aa773`
- `m460r5_independent_recomputation.json`: `86429afd9c85ceb04fc95d3406ff7c4d484025da94a43cca06f46ae04965230a`
- `m460r5_independent_attack_matrix.csv`: `0eb83ae03e2f5258982c4435c792426b11780804f98418f40a9b46c673120d4a`
- `docs/359`: `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`

本评审未运行 GPU/SSH/训练/模型/checkpoint 反序列化/RTL/新思，未修改 R5 payload、contract、seals 或 docs/359。
