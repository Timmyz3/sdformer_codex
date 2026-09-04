# M638 fresh independent hammer：M634 / M527 configuration payload builder

## Verdict

**NO-GO，61/100，P0=1 / P1=3 / P2=1。**

当前版本不得生成 production payload，也**不得接入 superseding M624 R6**。M634 的正向骨架是有价值的：它冻结了恰好五档 B0/B1/B2/B3/Ours、共同 28 nm / 3 ns / 96 lanes / 240 KiB / 64 GB/s / Acc24 tuple，B2 只把执行上限降为 K1 而没有删除物理 K1x8 pool；八项 added-resource/energy charge 均强制为 `true`，fallback 明确收费 cycles/traffic/energy/area；作者 9/9 测试、Python 3.6 `py_compile`、输入 symlink/duplicate-key/non-finite 拒绝均通过。

但 fresh adversarial tests 证明：攻击者修改已经存在的**已知字段**并重做 member/outer seal 后，`verify_payload()` 仍会返回 `PASS_M634_CONFIGURATION_PAYLOAD__M527_HEADLINE_BLOCKED`。可被接受的漂移包括 `waterfall_admitted=true`、M527 registry gate=true、receipt `system_speedup=true`、配置 optimized/fallback 重叠、三个 configuration path 指向不存在文件、以及 common manifest 内嵌 measurement checkpoint/operator universe 漂移。这违反 fail-closed 验证器的核心边界，故为 P0 阻断。

## 独立复核与攻击结果

### 通过项

- frozen M527 r3 SHA：`83ea25e43b53d12800ac64e971069a682e3077411ff10851a7861636ef77355b`。
- frozen `docs/359` SHA：`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`，未修改。
- author handoff member seal 与 outer seal 在其目录内重验通过。
- builder/test 当前 SHA 分别为 `b53429d9444e44f33cb9a240f696a3d847323da1af7929ed43e473e87fa564fa` / `bc60e3bd5f6689677883b9f04d3b860b4e0774c0846fbd7a0c6c8cdbcddaa5e1`。
- `python3` 为 3.6.8；builder 与 test `py_compile` 通过；作者测试 9/9 PASS。
- exact five-row ID/order、19 个共同 resource fields、八个 charge fields、全收费 fallback、B2 physical K1x8 + execution cap K1 均在未篡改 payload 中成立。
- repo 外输入、输入路径任一 symlink component、重复 JSON key、`NaN` 等 non-finite token 均 fail-close。

### Fresh adversarial matrix

| Attack | 重封后 `verify_payload()` | 影响 |
|---|---:|---|
| A1：把 Ours `optimized_operator_ids` 改成全 universe，与 frozen fallback 重叠 | **ACCEPTED** | 输出 manifest 不再满足 operator partition |
| A2：把 Ours `claim_boundary.waterfall_admitted=true` | **ACCEPTED** | 已知 admission 字段可过验 |
| A3：registry contract SHA 置零、common path 不存在、payload-ready=false、M527 gate=true | **ACCEPTED** | registry identity/gate 不可信 |
| A4：receipt status 改为 HEADLINE、IDs 清空、source/seal checks=false、system_speedup=true | **ACCEPTED** | PASS verifier 可伴随虚假 receipt |
| A5：configuration simulator/trace/measurement paths 均改为不存在路径 | **ACCEPTED** | path/SHA pair 不再一致 |
| A6：common 内嵌 measurement checkpoint 置零、operator universe 清空 | **ACCEPTED** | 同一 common manifest 内出现两套身份 |
| A7：trace rows=0/wrong schema、population=999、aggregation 仅 1 weight，全部自哈希 | **ACCEPTED** | payload builder 不能证明 upstream population/aggregation 语义 |
| A8：output parent 是 symlink | **ACCEPTED** | 输出祖先路径未受 secure-file policy 保护 |
| A9：`os.replace` 后最终验证失败 | output 仍存在 | 无效 production 路径会残留并阻塞安全重试 |

A7 本身与作者声明的“上游语义由 superseding availability analyzer 独立验证”边界一致；问题是当前 payload 没有绑定该 upstream verification receipt，因此不能被 R6 单独视为 registry-ready 证明。

## Findings

### M638-P0-01：重封后的已知 admission/receipt 漂移仍返回 PASS

证据：`verify_payload()` 对 registry 只检查 schema 与 `headline_admitted=false`；没有锁定 `m527_contract_sha256`、`configuration_payload_ready`、`m527_contract_admission_gate_current_value` 或 registry identity SHA/path。对 per-config claim boundary 只检查 `system_speedup=false` 与 `paper_headline=false`，未检查 `waterfall_admitted` 与 registry-gate-change。对 receipt 只检查 schema 与 `paper_headline=false`，未逐值检查 status、五个 IDs、四个 proof booleans 和其余 claim boundary。A2--A4 均重封后被接受。

影响：一个被 verifier 报 PASS 的 payload 内部可以同时宣称 M527 gate/waterfall/system speedup 已准入，后续消费者若读取 registry/receipt 会得到与 verifier 返回值冲突的结果。

唯一修复：验证时按 schema 对 registry、每个 claim boundary 和 receipt 做**精确值相等**，不能只检查字段集合和两个 false；最稳妥方式是由 live frozen inputs 重建期望 common/config/registry/receipt 文档，再与 payload 做完整结构相等比较。增加 A2--A4 为负向回归。

### M638-P1-01：configuration 与 common 的派生字段未回绑 live sources

证据：config 的 `optimized_operator_ids` 从未与 live configuration source 比较；三个 source path、measurement identity SHA、common path 也没有做 path+SHA 成对比较。common 内嵌 `measurement_identity` 没有与现场 `verify_measurement_binding()` 返回对象比较。A1、A5、A6 均被接受。

影响：source JSON 本身可以保持合法，而 sealed executable manifest 已经变成另一套 operator partition、路径或 checkpoint/operator universe。

唯一修复：对每个 config 将 optimized IDs、configuration/simulator/trace/measurement/common path+SHA、mechanism/resource/charge/fallback/claim 全部与 live source 派生期望值做完整相等比较；common 的内嵌 measurement 必须等于 live canonical measurement。增加 A1/A5/A6 回归。

### M638-P1-02：没有绑定 upstream semantic-verification receipt

证据：measurement binding 只证明三个任意 repo 文件 SHA 存在，并检查 caller 自报 frame/density/operator IDs；A7 的零行 wrong-schema trace、999-frame population 与 1-weight aggregation 仍构建并验证 PASS。

影响：M634 可以作为配置身份子步骤，但不能单独关闭 M625 对 trace/schema/population/aggregation/decoder 语义的要求；直接把其 PASS 映射成 `m527_configuration_registry_ready=true` 会过度陈述。

唯一修复：superseding M624 analyzer 先完整验证 decoder-complete trace、sequence population、aggregation weights、operator universe 和 checkpoint identity，并产生 SHA-bound upstream receipt；M634 measurement binding 增加该 receipt path/SHA，并现场复核其 schema、status、population。否则只能命名为 `configuration_identity_payload_ready`，不能置 registry-ready。

### M638-P1-03：最终验证失败不会清理已发布 output

证据：`os.replace(staging, output_dir)` 在 `try` 内结束后才调用 `verify_payload(output_dir)`；A9 注入最终验证失败时，`output_dir` 仍存在。任何 live-source TOCTOU 漂移也可触发同类路径。

影响：失败产物占用 production canonical path，下一次因 “output must be absent” 被拒；操作员可能误把残留目录当成功结果。

唯一修复：先在 staging 完整验证，再原子发布；发布后若必须做 live reverify，则失败时原子移入显式 quarantine（不得留在 canonical output），并写 fail receipt。增加 post-rename failure/TOCTOU 回归。

### M638-P2-01：output ancestor symlink 未拒绝

证据：input 的每一层 symlink 会拒绝，但 output 仅检查 leaf 不存在和 parent `is_dir()`；A8 通过 symlink parent 在真实目标生成并验证 payload。

影响：受信 operator 正常使用时风险较低，但与作者对输入采用的 repo-local/no-symlink 写入纪律不一致。

唯一修复：output 必须位于 repo 内；逐层拒绝 symlink ancestor，并显式拒绝 dangling output symlink。增加路径回归。

## 对 superseding M624 R6 的明确裁决

- `READY_TO_INTEGRATE_INTO_SUPERSEDING_M624_R6=false`
- `PRODUCTION_PAYLOAD_ALLOWED=false`
- `M527_CONFIGURATION_REGISTRY_READY=false`
- `CURRENT_BUILDER_POSITIVE_SKELETON_REUSABLE=true`
- 修复顺序：P0-01 -> P1-01 -> P1-03 -> P1-02 upstream receipt binding -> fresh hammer；P2 可同批关闭。
- 即便下一版全部通过，也只准入 configuration-registry payload；fixed numerator、unified cycles、system speedup、waterfall、effective GOP/s 与 paper headline 仍须保持 false，直到 M527 三个独立 gate 全部有真实 evidence。

本 review 只运行 CPU `py_compile`、unittest 和临时 fixture 攻击；未生成 production manifests，未运行 GPU/VCS/DC/PT/Formality/remote，未修改被审 builder/test/M527 合同或 `docs/359`。
