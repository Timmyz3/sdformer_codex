# M719/M533 r9 final-release fresh hammer

## Verdict

**PASS，100/100，P0/P1/P2 = 0/0/0。** 精确 M719 r9 runner 现在被授权执行至多一次 VCS compile 和同一 attempt 内至多一次 simv；其他所有 EDA、CPU/GPU 实验及远端任务均不授权。

这是 receipt-blind、只读 final-release hammer。审阅过程没有执行 runner、VCS、simv、EDA、GPU 或远端任务，没有修改 release/candidate/runner/source/RTL/TB/SVA/macro，也没有创建 result 或 attempt marker。

## Runner 实际字段闭合

已逐项按 runner 第 572–623 行检查：

- release schema 精确，`launch_now=true`；
- authorization 和 resource policy 逐键相等；
- release 精确绑定 runner、source contract、source-static review、candidate 与 candidate-hammer SHA；
- result path 精确为 `results/m719_m533_m528_dead_write_only_1rw_vcs_r9_20260828`；
- 本 review schema、PASS、100、0/0/0 均符合 runner；
- `identity.final_release_sha256=0e559745...cf63ad`；
- runner/candidate SHA 绑定正确；
- `decision.exactly_one_vcs_attempt_authorized_now=true`；
- `decision.all_other_runs_authorized=false`。

## 完整链复核

- release、candidate、source contract 三组 JSON sidecar 双封通过；
- candidate hammer、source-static、M717 和 consumed r8 failure 四组 package 双封通过；
- source-static 与 candidate hammer 均为 100/100、P0/P1/P2=0/0/0，且此前均没有提前授权 launch；
- r8 仍 permanently consumed，functional VCS 保持 no conclusion；
- top r2、TB r4、SVA r2、macro adapter、binding plan 实时 SHA 匹配；
- old partial 恰好八个 regular file，名字和 SHA 全匹配；
- foundry manifest 全成员、slow Verilog、slow DB、VCS binary 身份匹配；
- `docs/359` 仍为冻结 SHA。

## 唯一 attempt 约束

审阅和双封前，r9 result 与 `.attempt` marker 均不存在。授权只有在以下条件同时保持时有效：

1. 使用 SHA `27f2d7c0...964604` 的精确 runner；
2. runner 立即执行的 same-UID collision、cgroup-v1、memory/swap/commit-headroom 门全部通过；
3. atomic result `mkdir` 前 result 仍不存在；
4. 一旦 atomic `mkdir` 成功，无论 VCS/simv 成败，该唯一 attempt 都永久消费，并必须形成双封 terminal receipt。

## Claim boundary

本 review 只授权一次受控执行，不预先准入 functional VCS、RTL correctness、trace recurrence、cycle、speedup、PPA、energy、系统指标或论文 headline。所有这些结论必须等待最终 result fresh hammer。
