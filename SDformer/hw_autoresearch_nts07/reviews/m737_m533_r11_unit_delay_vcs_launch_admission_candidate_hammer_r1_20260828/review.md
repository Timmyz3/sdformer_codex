# M737/M533 r11 UNIT_DELAY VCS launch-candidate hammer

## Verdict

**PASS，100/100，P0/P1/P2 = 0/0/0。** 本评审只准入 `launch_now=false` candidate 的身份与闭授权链，允许下一阶段编写 launch release；当前仍不授权 VCS、simv、EDA、CPU/GPU 实验或远程任务。

本次是 fresh、独立、只读机械复算。没有运行作者 runner、VCS、simv 或任何 EDA，没有创建 result/attempt，没有修改 candidate、runner、RTL、TB、SVA、macro、source contract 或 `docs/359`。

## 身份与双封复算

- candidate SHA 是 `a3753a22...9786`，严格 JSON 解析、member sidecar 与 outer seal 全部通过。
- runner / source contract / source-static review SHA 分别为 `f658be40...4e70`、`160a94a6...a4e4`、`de3f15eb...6750`，与 candidate 精确绑定。
- 特别复算 source contract sidecar 文件 SHA 为 `7b1d4a5b...d49`，outer-seal 文件 SHA 为 `d15f825b...0552`；candidate 中两项均真实，不存在把 manifest 内容哈希与 sidecar 文件哈希混用的问题。
- 已消费的 r10 failure result 也逐成员和 outer seal 复算通过；receipt 仍为 `FAILED_DO_NOT_CITE`，functional/timing 均无结论，不能被 r11 重标为 PASS。
- M736 失败评审及其双封仍成立：r10 functional VCS 为 `NO_CONCLUSION`、物理 timing 仍 open，只允许一个未来 UNIT_DELAY functional candidate identity，且当前 launch=false。
- top r2、TB r4、SVA r2、macro adapter、binding plan、foundry manifest 全成员、foundry Verilog 和 VCS binary 均与冻结 SHA 一致；`docs/359` 仍为 `dedde7ce...bdfc4`。

## 模式、资源与权限边界

- runner 仅在 VCS compile 命令中出现一次 `+define+UNIT_DELAY`；没有 `+notimingcheck` 或 `+no_notifier`，没有修改 foundry model。
- `macro_model_mode=foundry_UNIT_DELAY_functional`，只可能闭合 functional VCS；当前 `functional_vcs_verified=false`、`timing_verified=false`，slow-DB DC/PT 仍是独立义务。
- candidate 的 `vcs_runs=1/simv_runs=1` 是未来 release 链的休眠预算，不是当前权限。三次资源采样、cgroup-v1 failcnt/OOM、same-UID Synopsys collision、atomic mkdir、terminal double seal 均由 runner fail-closed 消费；本评审没有执行 live gate。
- 在本评审发布前，r11 result、launch release 及其 sidecars、final-release hammer 均不存在。candidate hammer 自身也不存在，符合 fresh publication。

## 下一门

必须先生成 fresh、双封的 `launch_now=true` release，再由另一个 fresh final-release hammer 精确绑定 candidate、本文评审和 release。只有 final hammer 可以授权唯一一次 VCS+simv attempt。

本评审不准入 functional correctness、RTL verified、timing、cycle、speedup、PPA、energy、系统指标或论文 headline。
