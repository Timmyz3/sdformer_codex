# M731｜M714-r2 terminal identity revalidation fresh static hammer

## 裁决

**PASS，100/100；P0/P1/P2 = 0/0/0。** exact runner SHA 为 `350d4bb063f469ecea7729f51c1f23b9a7aaca5f5198abcab95669618df09c28`。M728 的 terminal exact-SHA TOCTOU 已闭合；在严格条件下授权这一个 one-shot capture。

本轮没有运行 runner，没有 import/执行作者 capture，没有查询 GPU/`nvidia-smi`，没有调用 EDA 或远端，也没有修改作者文件、创建 attempt/result 或读取未来 result receipt。

## M728 P0 闭合

runner 在 capture 返回后执行三次 `terminal_revalidate_identity`：

1. capture 返回后、payload 校验前；
2. payload 校验后、写 terminal receipt 前；
3. receipt 写完后、`seal_tree` 前。

每次都重新计算 canonical runner、contract、capture SHA，并同时要求等于冻结 expected SHA、启动时 observed SHA；还要求 consumed-attempt 的 `IDENTITY` 是非符号链接普通文件且三项 SHA 完全相等。任一漂移在 `set -Eeuo pipefail` 下进入 EXIT trap，staging 被写 `FAILED_DO_NOT_CITE`、双封后移入 `failed_or_incomplete`，不会发布成结果。

payload validator 另外要求：当前三个文件等于 expected；contract 的 `identity.m714_script.sha256` 等于冻结 capture SHA；payload 的 `m714_contract_sha256`、`m714_script_sha256` 等于这些冻结对象。因此不再只是“终态当前文件内部自洽”。

## M716/M720/M724 回归

| 门 | 静态结果 |
|---|---|
| exact runner/contract/capture + sealed review outer SHA | PASS |
| 四次 fresh GPU utilization/memory、compute-app 查询 | PASS，查询失败不被 `|| true` 吞掉 |
| `profile100/valid825/validate/trainer/trainonly/evaluation/training` 与项目别名 | PASS |
| 一次性 attempt、staging、失败隔离、原子发布、member manifest、outer seal | PASS |
| contract/capture milestone `M714-r2` 与五项 identity | PASS |
| M366 人口、数值零门、pattern 守恒 | PASS（source static） |
| randomized smoke 不再冒充 exhaustive | PASS |
| build/direct/resident 资源模式与 ideal-resource 口径 | PASS（仍仅 diagnostic） |
| real-output miter / executable cycle / RTL / PPA / energy / system | **FALSE，未豁免** |

## 唯一授权方式

只有同时提供：

- `M714_R2_EXPECTED_RUNNER_SHA256=350d4bb063f469ecea7729f51c1f23b9a7aaca5f5198abcab95669618df09c28`；
- 本目录 `SHA256SUMS.seal.sha256` 文件本身的独立 SHA，作为 `M714_R2_EXPECTED_STATIC_REVIEW_OUTER_SHA256`；
- launch 当下四次 fresh GPU/process idle 全部通过；
- canonical result 与 attempt 都尚不存在；

才授权一次 exact runner。合同本身不授权；第二次 attempt 不授权。

## Claim boundary

这个 PASS 只授权一次采集动作，不预先认可任何结果数字。未来即使产出并双封，也只能先作为 PCTDA pattern statistics 与 ideal-resource issue lower-bound 使用；不能称 executable cycle、real-output equivalence、RTL/VCS、Synopsys PPA/energy、accuracy、system speedup 或 paper headline。

`docs/359` 未修改，SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
