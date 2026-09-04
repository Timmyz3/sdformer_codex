# M527 H67 headline baseline ladder 独立打铁 r1

日期：2026-08-27  
评审边界：只读审阅 `contracts/m527_h67_headline_baseline_ladder_contract_r1_20260827.json`，并与已准入的 M526 headline-method 边界和 M520 metric-registry 边界交叉核对。未修改被审合同，未运行 EDA，未修改 `docs/359_DATE终局冻结_20260813.md`。

## 结论

评分 **88/100**，`P0=0 / P1=4 / P2=2`。

裁决：`CONDITIONAL_GO__EVALUATION_STRUCTURE_SOUND__R2_MEASUREMENT_FREEZE_REQUIRED_BEFORE_RESULTS`。

M527 已经正确学到 Prosperity/Phi 最重要且合规的论文组织方法：允许用 B0/B1 形成明确分母的 headline，但要求同页披露 B3 强基线；将 replicated K1x8 baseline 与 typed K8 candidate 分开；禁止乘局部倍率；要求统一 simulator 重跑 waterfall；外部 artifact 不得改名为 ours；partial scope 未闭合前不得称 full network。这个方向公平，也比只报 K8/K1 的 `4.76x` 更能经受 DATE 复算。

但 r1 仍不是可以直接驱动 Table A/B 的机器合同。四处剩余自由度可让同一硬件得到显著不同的带宽、GOP/s、均值或 waterfall 归因。它们必须在见到 H67 结果前一次性冻结；否则“仿照 Prosperity/Phi”会从合法实验结构滑向事后选择口径。

## 已通过的关键检查

| 检查 | 结果 | 审阅意见 |
|---|---|---|
| supported/full scope | 基本通过 | 列出 10 类完整算子并设置 full-network gate；外部 adapter 的 supported scope 仍需机器化，见 P2-01 |
| B0/B1 headline | 通过 | 允许 dense/structured baseline 作为 headline 合法，但必须写清 `iso-lane` 与本地定义 |
| B3 强 baseline | 通过 | `b3_exact_bit_sparse_k1x8` 被强制紧邻披露，避免隐藏 equal-service 对照 |
| C2 K8 分离 | 通过 | typed K8 是 candidate，不再被误写成 strongest baseline |
| 固定 OP 分子原则 | 原则通过、实现未冻结 | 禁止 architecture-reduced ops 作跨配置分子；精确计数和 OP convention 仍缺 |
| 四类聚合 | 原则通过、population 未冻结 | geomean/arithmetic/ratio-of-sums/min/max 全列出，未允许事后选均值 |
| waterfall | 原则通过、资源归因未冻结 | 要求逐级重跑并禁止相乘，正确；step 身份和服务资源仍不完整 |
| external 边界 | 通过 | 明确 external mapping 不是 ours，跨论文不构造伪直接倍率 |
| 冻结边界 | 通过 | `h67_system_speedup=false`、`paper_headline_generated=false`、`docs359_modification_allowed=false` |

## P1 findings

### M527-P1-01｜64 的单位存在 8 倍歧义

合同字段为 `dram_bandwidth_gbps=64`，而 M526 正文冻结的是 `64 GB/s`。在 `3.0 ns` 时钟下，十进制 `64 GB/s` 等于 `192 B/cycle`；若按字段常见含义 `64 Gbit/s`，则只有 `24 B/cycle`。该差异足以改变 memory stall、FPS、capture gap 和 headline。

**唯一修复建议：** r2 将字段改为无歧义的 `dram_bandwidth_bytes_per_second=64000000000`，同时冻结 `clock_hz=333333333.333...` 和派生值 `dram_bytes_per_cycle=192`，simulator 只读这三个一致性检查后的值。

### M527-P1-02｜固定 numerator 与 population 仍是自然语言，不可复算

`dense_equivalent_ops` 与 `original_useful_nonzero_ops` 只有说明，没有 scalar、源 ledger/SHA、每次 accumulation 算 1 OP 还是 2 OP、非 MAC 算子如何计数、per-frame/per-sequence grain。聚合也没有冻结 sequence/frame manifest、每序列内部先求和还是先平均，以及 low/medium/high density 的阈值。`selection_after_measurement_forbidden=true` 不能替代这些身份。

风险是同一周期结果可以通过 OP convention、frame weighting 或 density bin 边界得到不同 effective GOP/s 和平均倍率。

**唯一修复建议：** 在首次运行前生成并 SHA-pin 一个 `measurement_population_and_numerator_manifest`，一次性列出 sequence/frame IDs、每个 frame 的 trace SHA、固定两类 OP scalar 与明确 OP convention、非 MAC 计数规则、density 指标和预注册阈值、per-sequence 汇总公式以及 ratio-of-sums 的匹配 population；任何配置只能引用该 manifest。

### M527-P1-03｜area-normalized 不等于 iso-area

合同正确承认相同 96 lane/SRAM/BW 仅为 `iso-lane`，但 `ours_c1_c2_c3_exact` 的 fairness label 又写成 `iso_lane_plus_area_normalized`，且没有定义真正的 iso-area 配置生成规则。`GOP/s/mm2` 是 area-normalized metric，不是“在相同总面积预算下的速度”。如果论文把两者互换，审稿人可以直接否定公平性。

**唯一修复建议：** r2 把所有配置的原生 fairness label 只写 `iso_lane`、`low_service` 或 `iso_service`；将 area-normalized 指标放入独立 derived-metric block。除非另行冻结“以 post-DC logic + target SRAM macro 总面积为预算、怎样增减 lane/bank、怎样重跑周期”的 iso-area sweep，否则显式设置 `iso_area_claim_allowed=false`。

### M527-P1-04｜waterfall 直接重跑了，但还不能唯一归因

`exact_unstructured_bit_skip`、`c1_exact_parent_product_capture` 和 `c3_exact_atlif_service` 不是已注册 configuration ID；从 bit-skip/C1 到 C2 时又会从 scalar service 切到 K8 service。即使每步重跑 unified simulator，incremental ratio 仍可能同时包含机制收益、8 倍服务资源和面积变化，不能称单一机制贡献。

**唯一修复建议：** r2 为每个 waterfall step 注册完整配置 ID，并冻结该步的 lane、service width、bank/port、SRAM、DRAM、逻辑面积与 checkpoint。只有资源身份相同的相邻行允许标 `mechanism incremental gain`；资源发生变化的行必须标 `architecture-stack scaling`，并把 `B3 K1x8 -> C2 typed K8` 设为 C2 唯一 equal-service 机制消融。

## P2 findings

### M527-P2-01｜external supported scope 仍应机器化

`prosperity_official_adapter` 已正确写成 external method，但没有要求输出 exact supported-operator manifest、unsupported/fallback cycles 和 `full_network=false`。`phi_like_adapter` 也应明确是 locally reconstructed Phi-like model，不是 official Phi artifact。

建议 r2 给每条 external row 增加 `scope_manifest_sha256`、`supported_ops`、`fallback_policy`、`full_network`、`official_artifact` 五字段；只有 exact same scope 才允许 ratio。

### M527-P2-02｜B1 的命名应避免被简写成 “vs PTB”

B1 是项目定义的 `PTB-like structured time-group`，并非已证明与 PTB 官方 simulator 周期相同。执行语义本身足够清楚，但摘要/图例若缩写成 `PTB` 会越过 external 边界。

建议显示名固定为 `Locally defined structured time-group baseline (PTB-like)`，并设置 `official_ptb_equivalent=false`。

## 对 DATE 呈现的裁决

在 P1 未关闭前，M527 只准作为**方法骨架**，不得生成 H67 headline。关闭后，以下写法合法：

1. 摘要报 `Ours vs B0` 或 `Ours vs B1` 的 multi-sequence geomean，并同句写 baseline 名称；
2. 主表紧邻给 `Ours vs B3`、area、SRAM、energy/frame 和两种固定 numerator 的 effective GOP/s；
3. waterfall 只使用 simulator 直接重跑值，不乘 C1/C2/C3 局部倍率；
4. Prosperity/Phi 行只保留原网络/原分母 reported metric，H67 adapter 另标 external supported-scope mapping；
5. 只有 10 类 required operator 全部有 compute、memory、preprocess、completion 时才写 full-network。

仍禁止：把 B1 称为 official PTB、把 area-normalized 称为 iso-area、把 supported-scope external ratio 放进 full-network headline、用可变 executed-additions 作 GOP/s 分子，或选择测后最有利的均值/序列。

## 准入边界

本评审只准入 M527 的实验组织方向。`h67_system_speedup=false`、`paper_headline_generated=false`、`energy_per_frame=false`、`iso_area_result=false`。机器裁定见 `m527_h67_headline_baseline_ladder_independent_hammer_verdict_r1.json`；机械核查见 `mechanical_checks.txt`。本目录以 `SHA256SUMS` 与 `SHA256SUMS.seal.sha256` 双 seal 封存。
