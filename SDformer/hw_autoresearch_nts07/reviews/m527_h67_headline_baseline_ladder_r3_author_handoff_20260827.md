# M527 H67 headline baseline ladder r3 作者交接

日期：2026-08-27  
作者边界：只根据 M527 r2 独立评审修订测量合同；未运行 VCS、DC、PT、Formality 或其他 EDA；未生成性能结果；未修改 `docs/359_DATE终局冻结_20260813.md`。本文不是独立评审。

## 产物与身份

- r3 合同：`contracts/m527_h67_headline_baseline_ladder_contract_r3_20260827.json`
- r3 SHA256：`83ea25e43b53d12800ac64e971069a682e3077411ff10851a7861636ef77355b`
- 被增量修复的 r2 SHA256：`22af355786223ba80266438f861ecd4d3a6832eb6d284b7eff178189191dd185`
- `docs/359_DATE终局冻结_20260813.md` SHA256：`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`

## 只修的两个残余 P1

### 1. 固定 numerator 成为独立、机器可拒绝的准入门

r3 新增 `fixed_throughput_numerators.admission_gate`，当前值明确为 `false`。headline 和 effective GOP/s 都必须先得到一个 SHA 绑定的 numerator receipt bundle；receipt schema 强制要求：

- 两个正整数标量：`dense_equivalent_ops_scalar` 与 `original_useful_nonzero_ops_scalar`；
- 唯一单位：`ops_per_frozen_population`；
- 机器可读 OP convention，包括 multiply、add、MAC、comparison、state update、normalization、address/control，且 `MAC = multiply + add`；
- included operator scope 与带理由的 excluded scope；二者不相交且并集覆盖冻结的 required operator scope；
- checkpoint、complete trace、sequence population、aggregation weight 四类身份 SHA；
- 被 numerator 排除的控制/地址工作仍必须计入 cycle、energy 和 traffic。

所有 receipt path、SHA 与实际 scalar 均未伪造，仍为 `null` 或不存在。任一路径、SHA、schema、单位、scope 或 identity 检查失败，合同要求拒绝 GOP/s 与 headline。

### 2. Waterfall 改成“注册配置 + SHA 资源身份”，不再使用 `same_*` 布尔

r3 新增 `configuration_manifest_schema` 与 fail-closed `configuration_registry`：

- 每个 headline/waterfall 配置有唯一 `configuration_id`；
- 每个配置必须绑定自己的 source/config manifest SHA，并共同绑定同一个 non-null resource manifest SHA；
- resource tuple 必须显式给出 queue depth、bank count、SRAM port mode、external port count、240 KiB、64 GB/s、3 ns、Acc24 等字段；
- matcher、scoreboard、control、state、SRAM bytes/ports、logic/memory dynamic energy 均须计费；
- unsupported operator 必须在同一 unified model 中 fallback，并完整计入 cycle/traffic/energy/area。

`waterfall.ordered_cumulative_steps` 的每一行都引用 registry 中的配置 ID，并保留空的 manifest path/SHA 槽。合同明确规定：只有共同资源 SHA 与每个配置 SHA 全部存在并校验后 waterfall 才能运行或出表。

C2 的唯一公平机制增量仍锁为：

`b3_exact_bit_sparse_k1x8 -> c2_exact_typed_k8`

只有这一对可设置 `equal_service_mechanism_gain=true`。C1 context 中的 typed-K8 累积行只用于完整 waterfall，不得冒充 C2 的唯一 iso-service 增量。

## r2 已闭合项保持不变

- `64 GB/s decimal = 512000000000 bit/s = 192 B / 3 ns cycle`；
- area-normalized 明确不等于 iso-area；
- B1 明确是 project-defined PTB-like，不是 official PTB；
- Prosperity/Phi-like external rows 均为 `full_network=false`；
- 禁止乘局部倍率，所有 waterfall 行必须重跑统一 simulator；
- 有损 PAFT 保持独立 checkpoint/Pareto 身份。

## 静态自检

仅执行了 JSON/文本/身份检查：

- `jq empty` 通过；
- registry configuration ID 唯一；
- 所有 waterfall configuration/base ID 均可在 registry 中找到；
- 仅 B3 K1x8 到 typed K8 一对拥有 `equal_service_mechanism_gain=true`；
- 合同中不存在 `same_*` 资源布尔字段；
- 64 GB/s 三值恒等、外部 `full_network=false`、三个准入门当前均为 `false`；
- `git diff --check` 通过。

## 当前可声明边界

r3 只关闭“合同表达不唯一”的问题，不补不存在的实测证据。当前仍为：

- `numerator_receipt_admitted=false`
- `configuration_registry_admitted=false`
- `waterfall_admitted=false`
- `effective_gops_admitted=false`
- `h67_system_speedup=false`
- `paper_headline_generated=false`

建议下一步由不同评审者只读打铁 r3；在评审通过且真实 manifests/receipts 生成前，不得手工把任何 null 槽填成推测值。
