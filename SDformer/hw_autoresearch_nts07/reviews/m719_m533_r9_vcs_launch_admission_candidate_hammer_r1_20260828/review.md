# M719/M533 r9 launch-candidate fresh hammer

## Verdict

**PASS，100/100，P0/P1/P2 = 0/0/0。** 本评审只准入 `launch_now=false` candidate 的身份、闭链和未来 release authoring；当前仍不授权 VCS、simv 或任何 EDA 执行。

审阅为 receipt-blind、只读。没有执行作者 runner，没有运行 VCS/simv/CPU 或 GPU 实验，没有访问远端，没有修改 candidate、runner、RTL/TB/SVA/macro/source contract，也没有创建 result 或 attempt marker。

## Runner 实际消费字段

已按 runner 第 572–623 行反向核对，而非只按作者说明核对：

- candidate schema 精确，`launch_now=false`；
- authorization 与 resource policy 逐键精确相等；
- candidate 的 runner/source-contract SHA 与实时文件一致；
- 本 `review.json` 使用 runner 要求的 schema、PASS、100、0/0/0；
- 本 review 精确绑定 candidate、runner、source-static review 三个 SHA；
- `decision.vcs_launch_authorized_now=false`。

candidate 的 `vcs_runs=1/simv_runs=1` 只是未来 release 链使用的封闭预算，不是当前权限；`launch_now=false`、candidate scope、claim boundary 和本 review decision 四处一致地禁止现在启动。

## 身份和 seal

- candidate SHA：`f1daf609...1015b7`，member sidecar 与 outer sidecar 均通过；
- runner SHA：`27f2d7c0...964604`；
- source-static review SHA：`7662ffb1...3500`，100/100、0/0/0、双封通过；
- source contract SHA：`fca6edc1...76185`，双封通过；
- r8 failure 与 M717 均双封通过，r8 保持 permanently consumed、functional VCS no conclusion；
- top r2、TB r4、SVA r2、macro adapter、binding plan 均与 candidate/source-static 身份一致；
- foundry manifest 全成员、slow Verilog、slow DB 与 VCS binary 实时 SHA 均匹配；
- `docs/359` 仍为 `dedde7ce...bdfc4`。

## Future-chain absence

在发布本 review 前，以下精确路径均不存在：

- r9 result 与 `.attempt` marker；
- candidate hammer 目录；
- launch release JSON、member sidecar、outer sidecar；
- final-release hammer 目录。

发布本 review 后只关闭 candidate-hammer 这一门。下一步可以单独编写、双封 `launch_now=true` release；之后还必须有 fresh final-release hammer。只有 final hammer 才可能授权一次 VCS+simv attempt。

## Claim boundary

本评审不准入或声称 functional VCS、RTL correctness、trace recurrence、cycle、speedup、PPA、energy、系统指标或论文 headline；`vcs_launch_authorized_now=false`。
