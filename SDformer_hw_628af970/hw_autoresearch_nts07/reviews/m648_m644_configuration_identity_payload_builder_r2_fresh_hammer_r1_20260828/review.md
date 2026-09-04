# M648 fresh independent hammer：M644/r2 M527 configuration-identity payload builder

## Verdict

**NEEDS REVISION，89/100，P0=0 / P1=1 / P2=0。**

M644/r2 已关闭 M638 的重封 overclaim、live-source 回绑、发布残留和输出路径四组缺口。作者 12/12 测试与本评审独立 A1--A9 重演均通过；恰好五档 B0/B1/B2/B3/Ours 共享同一 28 nm / 3 ns / 96 lanes / 240 KiB / 64 GB/s / Acc24 tuple、同一八项全收费 policy，B2 保留物理 K1x8 service width=8，只把 execution cap 限为 K1。该正向 builder 骨架可以保留。

但当前 `upstream_semantic_verification_receipt` 只被当作调用方提供的普通 JSON 做 schema/status/value/SHA 一致性检查；没有绑定受信 upstream analyzer source、analyzer contract、结果 bundle 双封印，也没有现场调用受信 semantic verifier。fresh F1 用**错 schema、零 decoder row** 的 trace，加一份自行声称六项 proof=true 的 receipt，再同步更新 measurement/config SHA，即可得到：

`PASS_M644_CONFIGURATION_IDENTITY_PAYLOAD__ALL_M527_ADMISSION_GATES_BLOCKED`

且 payload receipt 写入 `upstream_decoder_complete_semantics_verified=true`。因此 M644 能证明“调用方输入自洽”，尚不能证明“decoder-complete 语义已独立验证”。这不开放 M527 headline gate，故不是 P0；但会让 superseding M624 R6 把未验证语义包装成已验证语义，故是 P1 集成阻断。

## Intended grain and trust boundary

- 粒度：恰好五个 executable configuration identity manifest，加一个 common-resource manifest、registry、verification receipt。
- 当前可证明：配置身份、资源/收费相等、operator partition、路径/SHA、双封印、非准入 claim boundary。
- 当前不可证明：trace schema/population/aggregation/operator universe 的真实语义是否由受信 analyzer 执行验证。
- 明确不涉及：fixed numerator、unified cycle、waterfall、system speedup、effective GOP/s、paper headline。

## Independent checks

### Identity and author regression

- builder SHA：`435baacb13da5da1c30ca649353b0947476e4bec7a4164d4421c3cdd615abea7`。
- tests SHA：`5be169c9b50c3b19a3e5240df3126d21d748fa82cc5d0846fa623e608edb0b23`。
- contract SHA：`82a4b62a3a3b256328a010189a2fa71fcd46225ac1c84aa373780358d6d621c5`。
- frozen M634 base SHA：`b53429d9444e44f33cb9a240f696a3d847323da1af7929ed43e473e87fa564fa`。
- frozen M527 r3 SHA：`83ea25e43b53d12800ac64e971069a682e3077411ff10851a7861636ef77355b`。
- `docs/359` SHA：`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`，未修改。
- 作者 handoff member seal 与 outer seal 重验 PASS。
- Python 3.6.8 `py_compile` PASS；作者 unittest 12/12 PASS。

### Fresh attack matrix

| Check | Outcome | Evidence / implication |
|---|---:|---|
| 正向五配置、共同 resource/charge | PASS | 5 个 resource fingerprint=1；charge fingerprint=1；八项 charge 全 true |
| B2 physical charge | PASS | physical service width=8，execution limit=1，未删除七个 service |
| A1 operator partition 重封 | REJECT | live-source reconstruction mismatch |
| A2 waterfall overclaim 重封 | REJECT | exact config reconstruction mismatch |
| A3 registry identity/gate 重封 | REJECT | exact registry reconstruction mismatch |
| A4 receipt status/proof/system-speedup 重封 | REJECT | exact non-admission receipt mismatch |
| A5 config/simulator/trace path 重封 | REJECT | exact config reconstruction mismatch |
| A6 embedded measurement 漂移重封 | REJECT | exact common reconstruction mismatch |
| A7 receipt 单字段突变并更新 ref | REJECT | upstream identity cross-check mismatch |
| A8 output symlink ancestor | REJECT | symlink path component fail-close |
| repo 外 output / dangling leaf | REJECT | repo confinement / leaf symlink fail-close |
| staging verify 注入失败 | PASS cleanup | staging 与 canonical 均不残留 |
| A9 post-publish verify 注入失败 | PASS quarantine | canonical 清除；唯一 quarantine 含 failure receipt |
| 发布后 live config source 漂移 | REJECT | live reconstruction mismatch |
| **F1 自行伪造完整 upstream receipt + 错 schema/零 row trace** | **ACCEPTED** | 返回 PASS 且写 `upstream_decoder_complete_semantics_verified=true` |

所有 fresh fixture 均位于 repo 内的 `TemporaryDirectory`，退出时清理；未生成 production payload。

## Finding

### M648-P1-01：upstream receipt 的内容被检查，但 provenance / semantic execution 未绑定

**证据。** `verify_upstream_receipt()` 精确检查 schema、status、六个 verification boolean、non-admission boundary，并与 measurement 的 checkpoint/manifest SHA/population/frame/density/operator IDs 比较。这可以拒绝对既有 receipt 的局部 mutation。但 upstream receipt 本身没有以下任一受信根：

1. pinned superseding-M624 analyzer source path/SHA；
2. pinned analyzer contract path/SHA；
3. 受信 analyzer result bundle 的 exact member set、member seal 与 outer seal；
4. 对该 analyzer `validate`/semantic verifier 的现场调用。

所以调用方可自行生成 receipt 及全部被引用文件。F1 的 trace schema 为 `fabricated_not_decoder_complete_trace_v0`、rows=0、decoder_rows=0；只要 receipt 自报 proof=true 且 SHA 自洽，M644 即接受。

**影响。** standalone `validate` 的 `upstream_decoder_complete_semantics_verified=true` 不具备独立可复核含义；直接把这个 PASS 接入 superseding M624 R6，会错误关闭 M625 对 decoder trace schema/population/aggregation/operator universe 的语义要求。当前 false 的 M527 headline/system gates 没有被打开，因此严重度为 P1 而非 P0。

**最小修复。** 新一代 builder 只能二选一：

1. 在 runtime 以 pinned SHA 加载 superseding-M624 semantic analyzer，并对 trace/population/aggregation/operator universe/checkpoint bundle 现场执行其 read-only `validate`，然后从 validator 返回对象重建 upstream receipt；或
2. 接受一个 pinned analyzer/contract 产生的 sealed upstream result bundle，同时仍现场调用该 pinned analyzer 的 validate mode 重验 exact member set、member seal、outer seal 与语义。仅新增 producer path/SHA 或仅新增双封印不够，因为调用方仍能手工伪造同形 receipt。

增加 F1 为负向回归：任何 wrong-schema / zero-decoder-row / population-manifest mismatch，即使 receipt 所有 boolean 为 true 且所有 SHA 自洽，也必须 fail-close。修复前将 receipt 字段降名为 `upstream_semantic_receipt_identity_bound=true`，不得写 `...semantics_verified=true`。

## Superseding M624 R6 integration decision

- `READY_TO_INTEGRATE_AS_M624_R6_SEMANTIC_PROOF=false`
- `READY_TO_REUSE_POSITIVE_CONFIGURATION_BUILDER_SKELETON=true`
- `PRODUCTION_PAYLOAD_ALLOWED=false`
- `M527_CONFIGURATION_REGISTRY_READY=false`
- `WATERFALL_ADMITTED=false`
- `SYSTEM_SPEEDUP_ADMITTED=false`
- `PAPER_HEADLINE_ADMITTED=false`

下一版关闭 M648-P1-01 并经 fresh hammer 后，M644 的五配置 payload 才可作为 superseding M624 R6 的**配置身份子证据**；它仍不会单独开启 M527 registry gate，更不会产生性能数字。

本评审只运行 CPU `sha256sum`、Python `py_compile`、unittest 与 repo-local temporary fixture attacks；未运行/修改 production payload、GPU、VCS/DC/PT/Formality、remote、M511 或 `docs/359`。
