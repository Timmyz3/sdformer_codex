# M582｜M518 r4 单点 DC final-release correction fresh hammer

日期：2026-08-28  
模式：fresh independent、read-only、correction-overlay only  
结论：**PASS_STATIC_CORRECTION__ACTUAL_DC_BLOCKED_BY_FOREIGN_SIMV**  
评分：**100/100；P0/P1/P2 = 0/0/0**

本评审没有运行 DC、VCS、runner、远端任务或大 CPU 任务，没有创建 Fixed/rank3
result、attempt 或 paired-comparison admission，没有修改 M581 overlay、两个原
release、runner、RTL、Tcl、SDC、filelist、工具/DB、已有结果或 `docs/359`。

## 1. 裁决

M581 是有效的不可变、单向 provenance correction overlay。它逐字绑定 M577
FAIL 报告及双封、两个原 release 的冻结 SHA、两处错误旧值、M572/M575
candidate-hammer review JSON 的冻结 SHA 与真实 status；自身授权预算全部为零。

该 overlay 只修复 runner **不消费**的冗余
`release_chain.candidate_hammer_status` 解释。两个原 release 的字节、SHA、
runner 可消费的 status/point/auth/identity、result/attempt、资源门和 paired
边界全部保持不变。因此 M577 的唯一 P1 已由透明 successor evidence 闭合，
无需覆盖或静默改写原 release。

本 PASS 只是静态纠错链准入，不是运行回执。当前 UID 1909、PID 580855 的
foreign `simv` 仍在运行，实际 Fixed/rank3 DC 均 **BLOCKED**。

## 2. M581 身份与双封

审阅对象：
`contracts/m581_m518_r4_per_point_dc_release_status_correction_overlay_r1_20260828.json`

- payload SHA256：
  `62741dbb712384e800e0ea5544677684fa228da78a85dd3d551679b2e4e6168c`
- member sidecar 文件 SHA256：
  `e84352785f674f5f2596b73c18aa92b40a2a3b52ccedd4117b278b00806c8815`
- outer-seal 文件 SHA256：
  `6da74ecdfa5e9c4dd4843608c785b0d0a21e4a0322e8cfbf1e0c03167b3f67dd`

payload/member/outer seal 全部逐字节通过；payload 是普通文件而非 symlink；严格
JSON 解析没有 duplicate key 或非标准数值。overlay status 为
`READY_FOR_FRESH_FINAL_RELEASE_CORRECTION_HAMMER__NO_DC_EXECUTED`，自身
`run_dc/run_vcs/run_formality/run_pt/run_ptpx/run_remote=false`、
`max_attempts=0`，没有冒充一次新的运行权限。

## 3. M577 失败链与唯一 finding 闭合

M581 精确绑定：

- M577 `review.md` SHA256：
  `d7a8257a9516d8674d103b7660a6f4ad586c2713728e8cabe5a6e755ffee802e`
- M577 member manifest 文件 SHA256：
  `2e58611a16cd158297579a8d58d3b7de29a9143bb886a36b070c3c2f261f9bb3`
- M577 outer-seal 文件 SHA256：
  `57a19cf9acb41235575a21ddd11df33ca41444b9bf4751fda56cccdd7b28cd82`
- M577 verdict：96/100，P0/P1/P2=0/1/0。

M577 目录的 member/outer seal 重新验证通过。其唯一 P1 是两个 release 写入的
candidate-hammer status 不等于被绑定 review JSON 的真实 status；M581 没有
掩盖或重述成别的问题。

## 4. 两个原 release 与真实 successor status

| point | 冻结 release SHA256 | overlay 绑定的错误旧值 | SHA/旧值核验 |
|---|---|---|---|
| Fixed | `72e08fc809c149608f1b0701facc1dd41b433547dd6f36fe7e0f35ce1159bcb9` | `PASS_M572_M518_R4_PER_POINT_DC_LAUNCH_ADMISSION_CANDIDATE_HAMMER__NO_DC_AUTHORIZED` | PASS |
| rank3 | `64b191789d4fc908b1c269d215f8bf905b08eaf61da9ff40f49d9c93f85550bd` | 同上 | PASS |

两个原 release 的 payload/member/outer seal 仍闭合，live SHA 与 M577 审阅时一致，
证明 overlay 没有改写原字节。

权威 successor source 是：
`reviews/m572_m518_r4_per_point_dc_launch_admission_candidate_hammer_r1_20260828/review.json`，
live SHA256 为
`df459336391ead6372999de1e68b78439fdd5e225662646b64761dc10c389e3b`。
其真实 `/status` 逐字节为：

`PASS_M572_M518_R4_TWO_POINT_LAUNCH_ADMISSION_CANDIDATE_HAMMER`

该值与 M581 `authoritative_successor_value.exact_value` 完全一致；review 的 score
仍为 100，P0/P1/P2 仍为 0/0/0，review member/outer seal 重新验证通过。M581
只把这个真实值解释为两点 release 中错误冗余字段的 successor，不改变其他字段。

## 5. 为什么 overlay 不改变运行语义

冻结 runner SHA256 仍为：
`5240712aeaf5dd3b50d68fb29389b1be5d27ba0611c7c50b9d744185c63a00c8`。

对两个原 release 重新执行 runner 所用的静态 `jq -e` 表达式，Fixed 与 rank3
都通过：

- exact authorized status 与 point 匹配；
- `max_attempts=1`、`run_dc=true`；
- VCS/Formality/PT/PTPX/remote/paired 全为 false；
- identity runner SHA 与 live runner 一致；
- identity contract SHA 与 live source contract 一致；
- identity point/top 与 point 的硬映射一致。

runner 从不读取 `release_chain.candidate_hammer_status`，所以 M581 修复的是审计
provenance，不是运行行为。M581 也明确冻结
`runner_expected_release_paths_and_sha_remain_unchanged=true`，没有引入第二份
release path、第二种 result identity 或额外 attempt。

runner/RTL/Tcl/SDC/filelist/tool/DB 没有出现在 overlay 的可变替代集合中；两个
原 release SHA 未变又已绑定这些身份，因此它们与 M577 时逐字节相同。既有
setup/area-only、公平性、50 source tuple、1175 DC bit-level port、3 ns、一次
`compile_ultra`、0 incremental/hold-fix、slow/fast DB 边界继续成立。

## 6. result、attempt 与 paired 边界

评审时以下路径仍全部缺席：

- `dc_handoff/runs/m518_r4_fixed_setup_area_logic_only_dc_3p000ns_r1_20260828`
- `dc_handoff/runs/.m518_r4_fixed_setup_area_attempt_consumed`
- `dc_handoff/runs/m518_r4_rank3_setup_area_logic_only_dc_3p000ns_r1_20260828`
- `dc_handoff/runs/.m518_r4_rank3_setup_area_attempt_consumed`
- `contracts/m518_r4_fixed_rank3_paired_comparison_admission_r1_20260828.json`

Fixed/rank3 的 canonical result 和 attempt 身份互异；一点失败不消费另一点。
原 release 与 M581 都保持 paired comparison 未授权。只有两点各自 raw result
双封并经过独立 receipt hammer 后，才可另行审 paired admission。

## 7. 资源门与当前 live-host BLOCK

原 runner 继续保留：

- preflight：commit headroom ≥64 GiB，三次、间隔 10 秒；
- runtime：48 GiB 连续三次 soft gate、40 GiB immediate hard gate；
- MemAvailable ≥128 GiB、SwapFree ≥32 GiB；
- cgroup OOM/failcnt、同 UID EDA collision、exact child identity；
- runtime-final 更新连续计数、执行第三样本决策，且 ACK/monitor rc 为 PASS 门。

M581 明确这些 live runner gates 仍 mandatory，没有删除、放宽或绕过。评审时的
只读资源快照为 commit headroom `81,660,484 KiB`、MemAvailable
`414,540,136 KiB`、SwapFree `57,212,156 KiB`；这只是瞬时审计，不替代未来
runner 的 fresh 三样本 preflight。

全主机 EDA 观察仍有：UID `1909`、PID `580855`、`simv`，启动时间
2026-08-24 22:25:28 +08:00。runner 内建 collision 门只覆盖同 UID，因此 root
必须额外执行 full shared-host collision check。该 foreign `simv` 消失前，
**不得调用 Fixed 或 rank3 runner**。

## 8. 准入边界

M582 PASS 后，M581 只作为两个原 release 的透明 provenance successor 使用；
不得删除 M577 FAIL 或把原错误字段说成“从未存在”。实际运行仍须满足：

1. foreign `simv` 已消失；
2. 新的 full shared-host collision/resource preflight 全部通过；
3. M581、M582、原 release、M572/M575 review 双封仍 live；
4. 对应 point 的 result/attempt 仍唯一缺席；
5. 每点最多考虑一次 immutable runner 调用。

任何 raw result 仍不是 paired PPA、power、energy、system speedup 或 paper headline；
独立 point receipt hammer 仍 mandatory。

`docs/359_DATE终局冻结_20260813.md` SHA256 保持：
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
