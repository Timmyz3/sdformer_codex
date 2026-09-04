# M475：非 Conv 稀疏/复用机会独立打铁审计（r1，2026-08-26）

## 0. 结论先行

本轮只建议继续两个 **CPU 离线、同资源 DSE**，暂不新增 RTL：

1. **P0：FC1 全宽 context-factorized held-weight DSE**。它是非 Conv 中唯一同时具备较大全网份额、精确小宽生命周期模型、VCS 和 28 nm DC 依据的性能候选。
2. **P1：动态 BN 精确 materialization-elision DSE**。不要继续优化 reciprocal-sqrt/系数生成器；应测 BN1→ATLIF、BN2→residual commit 融合后是否真正省掉中间 SRAM 写回和重读。

FC2 新控制/合并、ATLIF remaining-budget early-stop、patch-embed N=0 跳过均应维持 NO-GO。ATLIF 已有的 exact phase decoupling 是另一条既有微架构线，可以保留，但不应伪装成新的稀疏机会。

本审计不产生 cycle/system-speedup admission，不修改 `docs/359`。独立打铁总裁定为：

`GO_TWO_CPU_DSES_ONLY__NO_GO_NEW_RTL`

## 1. 冻结身份与证据状态

- `docs/359_DATE终局冻结_20260813.md` SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`；本轮未修改。
- M160、M230、M232、M240、M262、M355、M375、M386 的现有 manifest/SHA seal 均已重新执行并通过。M355 的 `SHA256SUMS` 必须从 `hw_autoresearch_nts07/` 根目录校验；在结果子目录执行会因相对路径产生假失败。
- 全网尺度采用 M462r2 的冻结 envelope `620,302,905`；其中 FC1 为 `118,370,114`，即 `19.0826%`。该 envelope 是机会/Amdahl 尺度，不是最终带 SRAM/DRAM 重叠的系统周期。

## 2. P0：FC1 全宽、同资源 context-factorized held-weight DSE

### 2.1 为什么它排第一

现有证据给出了两组互补、但不能直接相乘的精确结果：

- M230 在 100-record 冻结 trace 上将 F1/F2/F4 held-context 微架构映射为 fixed-latency recurrence。raw、latency=2 时，F2 相对同模式 K8/F1 为 `1.5516×`，逻辑吞吐/面积为 `1.1772×`；F4 为 `2.0684×`，逻辑吞吐/面积为 `1.0551×`。F1/F2/F4 的 28 nm logic-only DC 面积分别为 `18,219.22 / 24,013.21 / 35,715.08 µm²`，已有 exact directed VCS 与 3.0 ns DC 绑定。
- M262 把同一 raw binary-FC1 population 映射到 serialized 8-lane、same-port 生命周期。context-factorized 相对 bit-sparse 的生命周期为 `1.6722×`，权重请求下降 `2.5801×`；但它明确不是 96-lane RTL、不是 physical SRAM、也不是完整 FC1/FFN。

这两组结果分别证明“并行 held-context recurrence 有潜力”和“descriptor/context factorization 可减少生命周期与权重访问”，却使用不同宽度和周期边界。因此正确下一步不是把 `1.5516× × 1.6722×` 相乘，而是把二者放进一个全宽、端口与容量一致的模型里。

FC1 的全网份额为 19.08%。仅作为理想 Amdahl 灵敏度：若最终完整 FC1 模块达到 `1.6722× / 2.0684× / 2.1555×`，对应 envelope 上限约为 `1.0831× / 1.1093× / 1.1140×`。这些是选择 DSE 优先级的上界，不是可投稿系统倍速。

### 2.2 需要直接跑的 DSE

在 M230/M262 的同一冻结 raw population 上统一扫描：

| 轴 | 建议点 |
|---|---|
| 物理 lane 数 | 8、16、32、96 |
| held context factor F | 1、2、4 |
| source chunk | 16、32、64 |
| weight bank/读口 | 1、2、4 bank，显式 2-cycle response |
| descriptor/accumulator | 同容量、同读写口、同响应延迟 |
| 固定开销 | header、empty bypass、prefetch fill/drain、commit、stage3 fallback 全计入 |

必须同时报告：完整 FC1 cycles、descriptor/factor/weight/acc 请求、buffer bits、端口数、面积代理、吞吐/面积、stage3 conventional fallback 占比。最终候选只能与相同 lane、相同 bank/port、相同 SRAM 容量的 bit-sparse baseline 比。

固定 32-source active-chunk skip 单独只有 `1.0666×`（raw）/`1.0671×`（spatial）的机会，不足以做主贡献；它只能作为上述 DSE 的从属轴，不能预写成新倍速。

### 2.3 决策门

只有同时满足以下条件才进入新 RTL：

1. 完整 binary-FC1、全宽、同资源模块周期相对 bit-sparse `>=1.50×`；
2. 按 19.0826% 份额折算的理想 envelope `>=1.08×`；
3. weight traffic 不增加，descriptor/accumulator 容量和端口均在冻结预算内；
4. F2/F4 均报吞吐/面积，不能只挑绝对吞吐点；
5. stage3 两个 non-binary FC1 明确走 conventional fallback，且 fallback 周期计入。

## 3. P1：动态 BN 精确 materialization-elision DSE

### 3.1 目标必须改成“中间数据不落地”

M232 已经说明：12 个 FFN 每帧只有 `22,080` 个动态 BN 系数，II16 ping-pong 模型的首 tile 暴露仅 `21,504 cycles/frame`，占 620.303M envelope 的 `0.00347%`；BN1/BN2 的最小生产消费裕量分别为 `5.859×/1.953×`。所以继续压 reciprocal-sqrt 或系数 II 不会产生系统性能故事。

真正的机会来自 M160 的精确代数：

- BN1 affine 可以直接并入 ATLIF 的 lane-local affine/threshold；
- BN2 affine 可以直接并入 residual commit；
- 每帧有 `437,760,000` 个 standalone BN 元素具备“无需物化”的代数资格。

这不是 static-BN fold。冻结网络使用 current-batch 统计，而且 BN offset 会让零输入产生非零输出；任何“输入为零便跳过 BN”的逻辑都是错误的。

若每个中间元素发生一次写 SRAM、一次读 SRAM，则仅作为带宽灵敏度，16/24/32-bit 数据分别对应 `1.751 / 2.627 / 3.502 GB/frame`。在 32/64/128 B/cycle 下，未重叠传输成本分别为：

| 位宽 | 32 B/cyc | 64 B/cyc | 128 B/cyc |
|---|---:|---:|---:|
| 16 bit | 54.72M cyc | 27.36M cyc | 13.68M cyc |
| 24 bit | 82.08M cyc | 41.04M cyc | 20.52M cyc |
| 32 bit | 109.44M cyc | 54.72M cyc | 27.36M cyc |

这些是“1 write + 1 read、无重叠”的条件上界。当前 620.303M envelope 没有给出 BN 物化流量的独立归属，因此不能把表中周期直接从 envelope 扣除并宣称 speedup。

### 3.2 需要直接跑的 DSE

比较两个完全相同计算资源的 schedule：

- baseline：FC output 写入中间 SRAM，BN current-batch moment/barrier 后读出、归一化，再写入下一阶段；
- candidate：保留 current-batch moment 与 barrier，BN1 normalized value 直接流向 ATLIF，BN2 normalized value 直接进入 residual commit；系数流、raw FC retention 和 replay 均收费。

扫描轴：元素位宽 16/24/32、SRAM 总带宽 32/64/128 B/cycle、bank/1R1W port 数、moment barrier、coefficient II9/II16、raw-FC retention 容量、replay 顺序。报告 saved bytes、实际 exposed cycles（含 overlap）、moment/raw/coefficient SRAM bits、bank conflicts 和 spill。

决策门：

1. current-batch BN 逐元素 exact，禁止换成 running-stat/static fold；
2. moment state、raw FC retention、coefficient tile、barrier 与 replay 都计入；
3. intermediate traffic 至少下降 30%；
4. 在至少一个现实带宽点上，含重叠同资源 schedule 的系统灵敏度 `>=1.05×` 才升性能贡献；否则只保留为 energy/traffic 优化；
5. 不再为 coefficient arithmetic 单独开 RTL/DSE。

## 4. 其他非 Conv 方向的负面收口

| 方向 | 证据 | 裁定 |
|---|---|---|
| FC2 | 冻结 FC2 输入是 exact unit event；M216 K8 相对 K1 的局部 `4.7642×` 来自并行峰值差异，M349/M355 等峰值带宽对照为 `1.000×` | 保留既有 exact K8，实现不再开新的 coalescing/control 稀疏 RTL |
| FFN token/site skip | M462r2 冻结 tau 网格的最佳可执行 savings 为 0；达到理想 1.15× 需要 post-hoc `tau > 0.8713`，且无 ΔAEE、不可执行 | NO-GO；不能拿 oracle cliff 当性能点 |
| ATLIF G12 | M386 独立复算：代表点 term skip 6.577%，issue reduction 0.0676%，条件 speedup 约 1.00008×；所有 site 均未过 25% gate | kill remaining-budget sparse RTL |
| ATLIF phase decoupling | 既有 M258/M265 是 exact、非稀疏的模块调度候选，尚未 integrated RTL/area/accuracy | 可保留既有路线，但与本轮两个 DSE 分表，不叠乘 |
| patch-embed N=0 | M375：zero receptive field 17.999%，但 whole-temporal zero site 仅 0.00387%；bit-sparse baseline 已跳过零 MAC/DMA，极端 scan/commit 上界也仅 1.0617× | 不做 N=0 新 RTL；N>0 有损门留作后续算法 Pareto，非本周 P0/P1 |

## 5. 独立打铁评分

### 5.1 审计质量

| 维度 | 分数 | 评语 |
|---|---:|---|
| 身份、seal、可追溯性 | 19/20 | 主证据已复核；冻结文档 SHA 未变 |
| 算子覆盖与负面控制 | 19/20 | FC1/FC2/BN/ATLIF/patch 均有明确结论 |
| Amdahl 与口径纪律 | 19/20 | 局部比、机会尺度、系统 admission 严格分开 |
| 同资源公平性 | 17/20 | 已提出端口/容量门，但全宽/宏模型尚未执行 |
| 可执行性与止损门 | 18/20 | 两个 DSE 可直接做，且未先写 RTL |
| **总分** | **92/100** | **PASS_AUDIT__NOT_PERFORMANCE_ADMISSION** |

### 5.2 候选执行优先级

- **FC1 全宽同资源 DSE：90/100，GO_OFFLINE_DSE。** 优点是 19.08% Amdahl 份额、已有 exact VCS/DC 和两种互补模型；最大风险是小宽生命周期收益在全宽 SRAM/端口下消失。
- **BN materialization-elision DSE：82/100，GO_OFFLINE_DSE。** 优点是 exact 融合代数和可能很大的中间流量；最大风险是 raw retention/barrier 使流量只是搬家，或 baseline 已用 overlap 隐藏。
- **FC2/ATLIF-G12/patch-N0 新稀疏 RTL：<=35/100，NO-GO。** 现有独立负面结果足以止损。

## 6. 可复查证据路径与 SHA256

- `results/m230_h67_fc1_m229_fixed_latency_trace_recurrence_r1_20260825/m230_h67_fc1_m229_fixed_latency_trace_recurrence_r1.json` — `6110dff1cac748ca934e05033ddabe39f06e8b54286699a7843c209ddfe4a6ca`
- `results/m262_fc1_descriptor_lifecycle_trace_r2_exact_20260825/trace_payload/m262_fc1_descriptor_lifecycle_trace_r1.json` — `9aa24e2ef8889e6e697121817e5e27ca028db81e9e0dee4206fbc34394ec103a`
- `results/m229_fc1_dual_held_prefetch_replay_directed_vcs_r1_exact_20260825/RUN_COMPLETE.txt` — `e3a6c9a9b01950e50a12d38b2b258ebdc7c047c83f5ca56aedc3bb3bb589db2f`
- `dc_handoff/runs/m229_fc1_dual_held_prefetch_replay_matched_dc_3p000ns_r1_20260825/RUN_COMPLETE.txt` — `6dcfeed9d0ea478e9d6af7b9785f7ffb28a4c8edde72430e5e9d8c02b210dbeb`
- `results/m160_h67_ffn_bn_atlif_fusion_r1_20260824/m160_h67_ffn_bn_atlif_fusion.json` — `7581ccfdfc6bffc198b4e4dabfad04269a0fc58031d743704a487c21e8aeb96d`
- `results/m232_dynamic_bn_coefficient_stream_screen_r1_20260825/m232_dynamic_bn_coefficient_stream_screen_r1.json` — `51175ae37085cef00efce57ecabeff34099ff3e05ed468f1da1da2f17f823a5d`
- `results/m240_bn_pareto_independent_hammer_r1_20260825/m240_bn_pareto_independent_hammer_r1.json` — `52a06d6c24c5369978048631c840a28fca01d7ffa6a384df2ba1ee78a24d52cc`
- `results/m281_m276_bn_protocol_ii_independent_hammer_r1_20260825/m281_m276_bn_protocol_ii_independent_hammer_review_r1.json` — `a68debd51a191f5f1ff99dd9b175294cd55c19e917b4c4c3c079568e15cdb152`
- `results/m355_m349_fc2_equal_bandwidth_independent_hammer_r1_20260825/m355_independent_hammer_review_r1.json` — `7f5325bb6d74805091c5cb08b0c87075d900b4f85f1bdbeb14c95cf04c2782a8`
- `results/m375_m374_patch_zero_rf_independent_hammer_r1_20260825/m375_m374_patch_zero_rf_independent_hammer_review_r1.json` — `1804b6b2013bb2440cf2aab6eab7d5ab9d95059c87e9e0f03c95d73048eeb5da`
- `results/m386_g12_atlif_s10_gate_recompute_independent_hammer_r1_20260825/m386_g12_atlif_s10_gate_recompute_independent_hammer_r1.json` — `84652be51d1b66bbe9e750809b4963ec565ddbc366c0dfaa8f6d083353f7e3b7`
- `results/m462r2_h67_g8_site_gate_postcompute_oracle_cycle_audit_r1_20260826/m462r2_h67_g8_site_gate_cycle_oracle_audit.json` — `3c80bb7c037f58a3f6dbabd553a822ebc4b4f887bd076530dc29a52f53c5c09f`

