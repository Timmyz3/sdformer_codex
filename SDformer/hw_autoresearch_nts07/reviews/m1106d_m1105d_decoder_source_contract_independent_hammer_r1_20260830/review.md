# M1106D independent hammer of M1105D decoder source/contract

结论：**STOP；不授权 production runner。** canonical M1105D receipt 本身完整且数值正确，但 source/contract 组合没有形成 fail-closed 的执行信任根。

## 通过项

- receipt `receipt.json` SHA 为 `95e80734...`，manifest SHA 为 `949d2454...`，真正 outer seal-file SHA 为 `1cab0368...`；三者及 manifest 全覆盖均通过。
- canonical population 为 3 条序列、30 sample、120 次调用、261,090,000 packed bytes，每个 sample 严格 D0→D1→D2→D3。
- 30 个 D1 均保持 little-endian word `1065353139`，raw-content miter 0 mismatch；无 weight folding、无强制 1。
- 独立 projection validator 对 call reorder、漏 D1、重复 ordinal、payload path/SHA、D1 endian/word/folding/force-one、地址重叠、dependency/timestamp、96 lane、Acc24、240 KiB、192 B/cycle、3 ns、M700 和 final-checkpoint 误标共 19/19 个变异全部拒绝。
- source 的 wrong-theta helper 会拒绝；M700 字段注入会拒绝；source 不读取环境变量。

## P0：caller contract 可重写语义后仍获 PASS

M1105D 暴露 caller-selectable `repo_root / contract / output`。`build()` 没有要求 contract 等于 canonical SHA，也没有验证 contract double seal；资源和 transaction schema 又被原样回传。

一次临时 forged contract 同时把 96 lane→95、Acc24→23、3 ns→4 ns、192 B/cycle→191、240 KiB→256 KiB、psum 地址与 input 重叠、删除 dependency 字段、改成 caller timestamp、改写 D1 folding/coercion/theta 以及 checkpoint/rebind policy。M1105D 仍返回：

`PASS_SOURCE_AND_FULL_IDENTITY_PREFLIGHT__PRODUCTION_NOT_RELEASED`

并在 receipt 中原样回传被篡改的 resource 与 dependency schema。canonical receipt 没被修改，但未来 production runner 若调用同一接口，就可能由 caller 改写性能分母，因此本 hammer 不能释放 runner。

## P1：manifest/outer 术语

交接中把 `949d2454...` 称为 outer；它实际是 `SHA256SUMS` manifest SHA。outer seal-file SHA 是 `1cab0368...`。当前 contract 没有误用该值，所以这是术语 P1；本评审已分别固定两者。

第一次 discovery 发生在 receipt seal 落盘之前，因此结果为空。hammer 等到 receipt、manifest、outer 三件套出现并核对精确身份后才继续，没有降低信任门。

## 最小修复

冻结 M1105D 和本 STOP。新命名空间必须让 source 在任何 payload access 前，从自身固定路径导出 repo root，并硬编码 canonical contract path/SHA/sidecar/outer；所有资源、D1、地址、dependency/timestamp、checkpoint/rebind 和 claim-boundary 字段均逐项验证。为避免 source↔contract SHA 循环，contract 不再内含 successor source SHA，改由 sealed author receipt 和异作者 hammer 同时绑定二者。

生产授权路径不得接受 caller repo/contract/output 或 authority env。修复后必须重跑同一个 combined forged-contract attack，并证明在打开或 stat 任意 M699 payload 前即拒绝。

没有生成 runner、attempt、production transaction、周期、traffic 或 speedup；没有启动 EDA/RTL/GPU/remote。`docs/359` SHA 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
