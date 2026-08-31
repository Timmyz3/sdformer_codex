# M729/M533 r10 VCS launch-candidate fresh hammer

## Verdict

**PASS，100/100，P0/P1/P2 = 0/0/0。** 本评审只准入 `launch_now=false` candidate 的身份和闭链，允许下一阶段编写 launch release；当前仍不授权 VCS、simv 或任何 EDA 执行。

本次为 receipt-blind、只读审阅。没有执行作者 runner、VCS、simv 或其他 EDA，没有创建 result/attempt，没有修改 candidate、runner、RTL、TB、SVA、macro、source contract 或 `docs/359`。

## 复算结果

- candidate SHA 为 `fb567cbe...19c28`，JSON 严格解析、member sidecar 和 outer sidecar 均通过。
- runner SHA 为 `dd601184...646d`；source contract SHA 为 `07e708c3...b92d`；source-static review SHA 为 `a061c75c...52e6`，后两者双封通过。
- r9 失败包与 M726 双封通过；r9 保持永久消费、compile failure、functional no conclusion，M726 只允许一个新的最小 r10 身份。
- top r2、TB r4 static-force repair、SVA r2、macro adapter 和 macro binding plan 均与冻结 SHA 一致。
- foundry manifest 全成员通过，slow Verilog、slow DB、VCS binary 与 runner 硬绑定一致；`docs/359` 仍为 `dedde7ce...bdfc4`。
- r10 result、launch release 及其两个 sidecar、final-release hammer 在发布本评审前均不存在。

## 权限边界

candidate 中 `vcs_runs=1/simv_runs=1` 是供未来完整 release 链使用的休眠预算，不是当前执行权限。candidate 的 `launch_now=false`、scope、claim boundary，以及本评审的 `decision.vcs_launch_authorized_now=false` 一致禁止当前启动。

下一步仍需独立编写并双封 `launch_now=true` release，再由 fresh final-release hammer 绑定 candidate、candidate hammer 和 release。只有 final hammer 才可授权唯一一次 VCS+simv attempt。

本评审不准入 functional VCS、RTL correctness、cycle、speedup、PPA、energy、系统指标或论文 headline。
