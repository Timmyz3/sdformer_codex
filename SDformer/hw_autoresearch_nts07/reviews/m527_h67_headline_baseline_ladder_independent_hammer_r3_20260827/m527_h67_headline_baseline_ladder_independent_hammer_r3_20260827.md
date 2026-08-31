# M527 H67 headline baseline ladder 独立增量打铁 r3

日期：2026-08-27  
评审边界：以双封 r2 review 为基线，只读复核 `contracts/m527_h67_headline_baseline_ladder_contract_r3_20260827.json` 是否关闭 r2 的两个残余 P1。未修改被审合同，未运行 VCS、DC、PT、Formality 或性能测量，未修改 `docs/359_DATE终局冻结_20260813.md`。

## 结论

评分 **98/100**，当前 `P0=0 / P1=0 / P2=1`。

裁决：`GO_CONTRACT_SEMANTICS__R2_P1_ZERO__ALL_MEASUREMENTS_AND_HEADLINES_REMAIN_FAIL_CLOSED`。

r3 已把 r2 的两个残余 P1 从自然语言意图改成可由未来 validator 拒绝的 schema 与独立 gate：固定 numerator 的标量、单位、OP convention、operator scope 和身份 SHA 均有唯一规则；waterfall 的每一行均由唯一 registry `configuration_id` 解析到 configuration manifest SHA，并共同绑定 non-null common-resource manifest SHA。合同内不再使用 `same_*` 布尔替代具体资源身份。

这不是测量准入。所有 receipt、configuration manifest、common-resource manifest 和 population identity 当前仍为 `null`，三个独立 headline gate 当前均为 `false`。所以 r3 只准入测量合同语义，不准入 effective GOP/s、waterfall、H67 system speedup 或 paper headline。

## r2 残余 P1 增量裁定

| r2 finding | r3 证据 | 裁定 |
|---|---|---|
| M527-R2-P1-01 numerator 不唯一 | 独立 receipt gate；两个正整数 scalar；唯一 `ops_per_frozen_population` 单位；七类机器可读 OP convention；included/excluded scope 完整分割；checkpoint/trace/population/aggregation 四类 SHA 对齐；任一失败拒绝 GOP/s 和 headline | `CLOSED_AT_CONTRACT_SEMANTICS` |
| M527-R2-P1-02 waterfall 只靠 same 布尔 | 9 个唯一 registry ID；每个配置要求 source/config/simulator/trace/common-resource SHA；完整 queue/bank/port/SRAM/BW/Acc24 resource tuple；完整 added-area/energy charge；fallback 全计费；step manifest 必须等于 registry；所有 step 绑定同一 non-null common-resource SHA | `CLOSED_AT_CONTRACT_SEMANTICS` |

## Numerator 可复算性

### 标量、单位与 population

`fixed_throughput_numerators.receipt_schema` 强制两个正整数：

- `dense_equivalent_ops_scalar`；
- `original_useful_nonzero_ops_scalar`。

二者单位均唯一为 `ops_per_frozen_population`。receipt 还必须给出正整数 `population_scalar`、唯一 population unit `frozen_frames_across_frozen_sequence_population`、`frame_definition`，并绑定 frozen sequence population 与 aggregation-weight manifest SHA。这消除了 per-frame、per-sequence、全 population 混用的自由度。

### OP convention 与 operator scope

OP convention 不是自由文本：必须机器化给出 multiply、add、MAC、comparison、state update、normalization、address/control 七项，值域锁为 `{0,1,2}`，并要求 `MAC = multiply + add`。同一 convention 必须同时用于两个 numerator 和所有配置。

included 与 excluded operator scope 必须不相交，二者并集必须精确等于合同冻结的 required operator scope；每项 excluded 都要有非空理由。即使某工作不进入 numerator，其 cycle、energy 与 traffic 仍必须收费。`executed_additions` 只能作为 architecture-reduced count，禁止充当跨配置 throughput numerator。

### 独立 headline gate

`M527_FIXED_NUMERATOR_RECEIPT_READY` 独立于 measurement-identity gate，并被 `headline_policy.independent_required_gates` 显式引用。路径/SHA、schema、标量、单位、OP convention、scope partition 或 identity match 任一失败，合同均拒绝 effective GOP/s 和 headline。当前值为 `false`，行为 fail-closed。

## Waterfall 配置身份与资源公平性

### 唯一配置身份

registry 共 9 个 `configuration_id`，独立机械检查确认 9 个均唯一。5 个 cumulative waterfall step 及其 4 个非空 base ID 全部存在于 registry。每个配置 manifest 必须包含：

- `configuration_id`、configuration source SHA、simulator source SHA、complete-trace SHA；
- common-resource manifest SHA 与 mechanism-enable map；
- 明确的 resource tuple、charge policy 和 fully charged fallback policy。

step 的 manifest path/SHA 必须等于 registry 对应项；所有 step 必须绑定同一个 non-null common-resource manifest SHA。因而未来每一行可由 `configuration_id + configuration_manifest_sha256 + common_resource_manifest_sha256` 唯一复算，而不是由 `same_queue`、`same_port` 一类布尔解释。

### 资源字段与收费

resource tuple 强制显式给出 3 ns、28 nm、96 lanes、source service width、240 KiB、64 GB/s、192 B/cycle、Acc24，以及 source/completion/parent queue depth、weight/state/parent bank count、三类 SRAM port mode、外部读写端口数。matcher、scoreboard、control、state bits、额外 SRAM bytes/ports、logic/memory dynamic energy 均须收费；unsupported operator 必须在同一 unified model 中执行并计入 cycle、traffic、energy、area。

### C2 唯一公平机制对

唯一允许标记 `equal_service_mechanism_gain=true` 的机制对为：

`b3_exact_bit_sparse_k1x8 -> c2_exact_typed_k8`

ordered cumulative waterfall 中 C1 context 下的 typed-K8 行已显式标记为非 C2 唯一公平 claim。K1 单服务仍只能作为带宽 scaling baseline，不能替代 K1x8 强基线。

## 既有公平边界保持

- 带宽恒等式独立复算通过：`64 GB/s decimal = 512000000000 bit/s = 192 B / 3 ns-cycle`。
- `area_normalized_is_not_iso_area=true`；只有 matched total area 才允许 iso-area claim。
- B1 明确是 project-defined PTB-like，且明确不是 official PTB。
- Prosperity 与 Phi-like external rows 均保持 `full_network=false`。
- 禁止把 isolated speedup 相乘；每个 waterfall 配置必须重跑 unified simulator。
- 三个独立 headline gate 当前均为 `false`；真实 manifests/receipts 未产生前没有任何测量结果。

## Residual P2

### M527-R3-P2-01｜external adapter 的 exact operator manifest 仍待实现

Prosperity 的 supported scope 仍为宽泛的 `mapped_Conv_or_FC_kernel_only`，Phi-like scope 仍为空。由于两者都明确 `full_network=false`，且当前所有 headline gate 均关闭，这不构成 P1。未来实现 adapter 时仍需冻结 exact operator IDs、trace manifest SHA、official-artifact 身份、unsupported fallback 收费，并只允许 same-scope ratio。

## 最终边界

r3 准入的是 **headline baseline ladder 的合同表达**，不是结果。当前必须继续保持：

- `numerator_receipt_admitted=false`；
- `configuration_registry_admitted=false`；
- `waterfall_admitted=false`；
- `effective_gops_admitted=false`；
- `h67_system_speedup=false`；
- `paper_headline_generated=false`。

机器裁定见 `m527_h67_headline_baseline_ladder_independent_hammer_verdict_r3.json`，机械检查见 `mechanical_checks_r3.txt`。本目录以 `SHA256SUMS` 与 `SHA256SUMS.seal.sha256` 双 seal 封存。
