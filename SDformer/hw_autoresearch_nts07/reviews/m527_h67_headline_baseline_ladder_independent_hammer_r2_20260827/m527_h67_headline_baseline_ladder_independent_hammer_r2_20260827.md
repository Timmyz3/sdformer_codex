# M527 H67 headline baseline ladder 独立增量打铁 r2

日期：2026-08-27  
评审边界：以双封 r1 review 为基线，只读复核 `contracts/m527_h67_headline_baseline_ladder_contract_r2_20260827.json` 对 r1 `P0=0/P1=4/P2=2` 的关闭情况。未修改被审合同，未运行 EDA，未修改 `docs/359_DATE终局冻结_20260813.md`。

## 结论

评分 **94/100**，当前 `P0=0 / P1=2 / P2=1`。

裁决：`CONDITIONAL_GO__P1_NOT_ZERO__TWO_CLOSED_TWO_PARTIAL__HEADLINE_REMAINS_BLOCKED`。

r2 对方法公平性有实质进步：带宽单位的 8 倍歧义已经数学闭合；area-normalized 与 iso-area 已分开；B1 明确是 project-defined PTB-like；external rows 明确 `full_network=false`；waterfall 已锁 K1x8 八 source/cycle，并将 C2 唯一公平增量限定为 `B3 K1x8 -> typed K8`。

但 r1 四个 P1 中只有两个完全关闭。固定 numerator 的可复算 schema 和 waterfall 的机器资源身份仍未闭合，因此 P1 不能清零，r2 仍不得生成 H67 headline。

## r1 findings 增量裁定

| r1 finding | r2 证据 | 裁定 |
|---|---|---|
| P1-01 DRAM 单位 | `64 GB/s decimal = 512000000000 bit/s = 192 B/3ns-cycle`；独立恒等式检查通过 | `CLOSED` |
| P1-02 numerator/population | population/trace/bin/weight 设为 null placeholder，带 `required_before_headline=true`；但 numerator 没有 scalar、OP convention、receipt schema 或自己的必填门 | `PARTIAL__P1_REMAINS` |
| P1-03 area-normalized vs iso-area | 明确 `area_normalized_is_not_iso_area=true`、iso-area 需 matched total area；Ours label 改为 separate normalization | `CLOSED` |
| P1-04 waterfall resource identity | 锁定 K1x8 service width、port/queue/precision 同一性并规定 C2 唯一公平增量；但五个 step 仍非已注册 config ID，“same_*”未绑定具体值/manifest SHA | `PARTIAL__P1_REMAINS` |
| P2-01 external scope | `supported_scope` 与 `full_network=false` 已加入 | `CLOSED_AT_PRINCIPLE_LEVEL` |
| P2-02 PTB-like naming | execution 明确 project-defined 且不是 official PTB | `CLOSED` |

## 已闭合项

### 带宽身份闭合

独立复算：

- `64 x 8e9 = 512e9 bit/s`；
- `64e9 byte/s x 3e-9 s = 192 byte/cycle`；
- r1 的 `192 vs 24 B/cycle` 八倍自由度已消失。

这项现在足够作为 simulator common-resource 常数，但实现仍应在入口断言三值恒等，防止未来只改其中一个字段。

### area 公平性闭合

r2 不再把 GOP/s/mm2 当成 iso-area。允许的三个不同口径已经能分开：

1. B0/B1/Ours：`iso-lane`；
2. B3/K8：`iso-service`；
3. area：derived normalized metric；只有 matched total area 才能另称 iso-area。

### B1 与 external 边界闭合

B1 明文为 project-defined PTB-like，不能伪装成 official PTB。Prosperity/Phi-like 均为 `full_network=false`，外部数不能进入 ours full-network headline。这两项关闭了 r1 的命名风险。

## 残余 P1

### M527-R2-P1-01｜numerator placeholder 能阻止 population headline，但不能唯一解释 GOP/s

`measurement_identity` 有 9 个 null 身份字段并设置 `required_before_headline=true`，这一部分是 fail-closed 的。可是 `fixed_throughput_numerators` 的四个 receipt path/SHA 只是 null；对象内没有 `status` 或 `required_before_headline`，也没有：

- `dense_equivalent_ops_scalar` / `original_useful_nonzero_ops_scalar`；
- accumulation 算 1 OP 还是 2 OP；
- ATLIF、BN、Shiftmax、address/preprocess 是否进入 numerator；
- per-frame/per-sequence unit；
- receipt 必须包含这些字段的 schema/version。

因此“receipt 存在且 SHA 正确”仍不足以证明两个 effective GOP/s 可跨配置和跨论文复算。

**唯一修复建议：** r3 给 `fixed_throughput_numerators` 增加 `status=BLOCKED`、`required_before_headline=true` 和一个强制 `receipt_schema`；该 schema 至少要求两个非负 integer scalar、`op_convention`、`included_operator_scope`、`excluded_control_ops`、`population_manifest_sha256`、`unit=ops_per_frozen_population`，并规定任一缺失即拒绝生成 GOP/s/headline。

### M527-R2-P1-02｜waterfall 的资源锁仍是声明，不是可绑定配置

`service_width_sources_per_cycle=8` 与 `c2_unique_fair_increment=B3->K8` 是正确修复方向。但五个 ordered step 均不在顶层七个 configuration ID 中；`same_queue_depth=true`、`same_weight_and_state_sram_ports=true` 没有具体 depth/bank/port 数，也没有共同 resource manifest SHA。C1/C3 的额外逻辑、state SRAM 和各阶段 fallback 是否计入 unified model 仍不能由机器唯一确定。

这意味着 direct rerun 虽然避免“乘局部倍率”，但仍可能在两个 simulator config 中以不同资源解释相邻 step。

**唯一修复建议：** r3 将 waterfall 改成对象数组，每步必须包含唯一 `configuration_id`、`base_configuration_id`、`resource_manifest_path/sha256`、`service_width=8`、明确 queue/bank/port/SRAM/BW/Acc24 数值、额外 state/area charge 和 fallback policy；validator 要求所有 mechanism-increment 行引用同一个 resource SHA，只有 `B3 K1x8 -> C2 typed K8` 获得 `equal_service_mechanism_gain=true`。

## Residual P2

### M527-R2-P2-01｜external supported scope 仍是宽泛占位

Prosperity 的 `supported_scope=["mapped_Conv_or_FC_kernel_only"]` 关闭了 full-network 混用，但还不是 exact operator/trace manifest；Phi-like 也未明确 `official_artifact=false`。在 adapter 实现前不影响安全，因为两行均 `full_network=false` 且当前 headline 被阻塞。

建议实际 adapter 合同时补 exact operator IDs、trace manifest SHA、fallback/unsupported cycles、official-artifact boolean，并只允许 same-scope ratio。

## 最终边界

r2 已足够准入“baseline ladder 的论文组织方向”，不准入测量输出。`h67_system_speedup=false`、`paper_headline_generated=false`、`effective_gops=false`、`waterfall_mechanism_gain=false`、`iso_area_result=false`。

机器裁定见 `m527_h67_headline_baseline_ladder_independent_hammer_verdict_r2.json`，机械复算见 `mechanical_checks_r2.txt`。本目录以 `SHA256SUMS` 与 `SHA256SUMS.seal.sha256` 双 seal 封存。
