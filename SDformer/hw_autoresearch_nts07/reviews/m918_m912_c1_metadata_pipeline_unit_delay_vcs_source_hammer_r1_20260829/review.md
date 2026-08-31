# M918｜M912 C1 metadata-pipeline UNIT_DELAY VCS source hammer

Verdict: **FAIL / REPAIR_REQUIRED（91/100；P0=1，P1=3）**。本次仅做静态、只读审计；没有运行 VCS、simv、DC、PT、Formality、ICC2 或许可证查询，也没有创建 M912 attempt/result。

M912 的 RTL/TB/SVA 主体值得保留：冻结 r2 SHA 与冻结 M863 TB SHA 均正确；M912 与 r2 的 59 项顶层 port tuple 完全相同；新增边界按显式字段为 active 55 bit、next 53 bit、PF 13 bit、debug 9 bit，共 130 bit，低于 512-FF 门；没有新增 1824-bit psum 寄存槽，1152-bit 注册 payload 仍只有原有两项 response slots。平衡 tournament 的 key 为 `{invalid,popcount,row_id}`，六级 pairwise `min` 在 100,000 组 64-row 随机向量上与全局 `min` 一致；row-id 进入 key，equal-pop tie 因而严格保持低 row-id。active 首行填充后经一次 priming 才暴露 request，cleanroom oracle 显式收费两拍；有 next context 时完成边缘同拍 promote 并保持零 inter-row bubble。

功能门也没有被弱化：TB 保留 14 项 normal minima、P2 `consecutive_distinct_reads>=1 / response_identity_checks>=2`、held-final 一次、六类 attack 各定义/调用一次，以及唯一 coverage/held/PASS token。held-final 与 parent-only attack 检查真实内部 accept/fault 事件，不误用延迟 debug；九个 debug observer 在 TB 与 SVA 中均按一拍 `$past` 检查。PF accept 仍严格等于 forward 或 macro-read，父边计数由 `count_parent_edges == count_macro_reads + count_forwards` 双边约束。

## P0

runner 的 docs/359 exact-SHA 边指向不存在的旧路径：

`docs/359_硬件论文口径与冻结数字_20260812.md`

实际冻结文件、合同所需 SHA 对应的是：

`docs/359_DATE终局冻结_20260813.md`

后者 SHA 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`，前者不存在。当前 exact runner 因此会在 attempt marker 之前必然退出，不能授权 VCS。runner 已被合同 SHA 固定，禁止原地修补；必须做 additive successor runner + successor contract + 新双封，然后重新独立打铁。

## P1

1. 合同写明 `fresh_independent_hammer_required_before_launch=true`，但 runner 没有 pin/校验任何 hammer/release review；即使修复 docs 路径，程序本身仍可绕过独立授权直接消费 one-shot。successor 应 pin 最终 PASS hammer 的 review SHA 与双封。
2. collision guard 只看 `/proc/*/comm` 的精确集合，不识别 Synopsys 常见的 `common_shell_exec -shell dc_shell`（`comm` 还可能截断为 `common_shell_exe`）。这不满足合同的 same-UID DC/PT/FM/ICC2 collision 必须为零。successor 应在避免自匹配的前提下同时检查 `comm` 和 NUL 分隔 `cmdline` 的工具语义。
3. SVA 的 `ap_pf_candidate_is_later` 目前只证明 candidate consumer 与 held token 不同，没有证明 `{pop,row}` 严格更晚。TB cleanroom 会比较真实 PF 调度，故不是当前功能 P0；但 successor source checker/SVA 应增加严格 key-order obligation，防止以后 selector 改动只过“不同”而破坏稳定顺序。

## 允许的最小修复

- 只新增 successor runner/contract/checker identity；RTL、TB、SVA 当前 SHA 可保持不变。
- runner 将 docs/359 路径改为现存冻结文件，并 pin 新独立 hammer/release 双封。
- collision 检测补 `common_shell_exec` 语义。
- 静态 checker 增加 runner 所有 external exact-SHA 边的存在性检查、docs path 检查、hammer binding 检查及 PF strict-later 检查。
- 修复后必须重新发起 fresh source hammer；本 M918 失败审计不能作为发射授权。

Claim boundary：`functional_vcs_verified=false`、`timing_verified=false`、`cycles_measured=false`、`speedup=false`、`ppa=false`、`energy=false`、`paper_citable=false`。
