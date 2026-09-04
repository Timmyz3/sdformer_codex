# M789｜M519 R15 atomic-artifact three-axis DC final release hammer

## Verdict

**PASS 100/100；P0/P1/P2 = 0/0/0。**

本评审只授权一次 M519 R15 三轴 logic-only DC attempt。它不构成 DC 完成、PPA、能量、FC2 完整性或系统加速证据。runner、DC、其他 EDA、远端任务与许可证查询均未由本评审执行。

## Frozen identity

- Release：`278eb851af42474e08887258006ef71137e28d0271f2e2e38ee77d783cd4238c`，双封 live。
- Runner：`9ad15627c89bb078c5453333e979a1b98c2309b66fc71cce6b4f7aa4f89863b4`，双封 live。
- Recovery contract：`cdb74d549386a2e9b952329bc055549735e3152fd3a6ed2a767dcaad098b6429`，双封 live。
- M783 candidate：`500cb67f57db36b8b991cc19da137691cf1af7c1fac5f35e5a90df8788f36786`，双封 live。
- M786：精确状态 PASS、100/100、P0/P1/P2=0/0/0，双封 live。
- M780：精确绑定两个 P1 失败及 R15 additive repair lineage，未被重解释。
- docs/359：`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`，未修改。

## Mechanical result

独立静态审计 204/204 通过，22 个引用证据目录双封重算通过，runner `bash -n` 通过。release 的 15 个 candidate preserved semantic sections 逐对象相等；authorization keyset 闭合且只准 `max_attempts=1, run_dc=true`。

以下 gate 均保持 fail-closed：

- K1/K8/K1x8 在同一 R15 attempt 中各执行一次，不复用跨 attempt 输出；
- HOME 必须不存在，许可证变量与本地文件/lmutil 身份精确冻结；
- 只删除 M769 冻结的单一 16 行 bootstrap block，其他 Error/Fatal/TIM-209/OPT-150 全部致命；
- mapped Verilog、mapped SDC、DDC 均为每轴必需的非空非 symlink 叶；
- 两份 artifact receipt 通过一次 staging-directory rename 原子发布；
- receipt 发布后、RUN_COMPLETE 后、封根 manifest 后均回查 live bytes；
- `compile_ultra` 静态仅一次，禁止 incremental/hold optimization；hold 明确未在本 DC 阶段闭合；
- runtime monitor、campaign identity、emergency latch 与 synchronous final ack 均保留。

## Live observation

2026-08-28 22:43:09–22:43:11（Asia/Shanghai）三次观察：

- 最小 commit headroom：106,485,836 KiB（门槛 67,108,864 KiB）；
- 最小 MemAvailable：416,279,372 KiB（门槛 134,217,728 KiB）；
- 最小 SwapFree：56,632,060 KiB（门槛 33,554,432 KiB）；
- cgroup failcnt/under_oom/oom_kill：0/0/0；
- reviewer UID 1913 同 UID EDA collision：0；
- 全主机另见 UID 1909 的一个长期 `simv`，它不是冻结的同 UID collision，已公开而未隐去；
- canonical result、attempt sentinel 与本 hammer 输出在创建前均不存在。

runner 仍须在启动后重复三采样资源/碰撞门，并在 attempt 消耗前自行执行许可证 status-only gate。本评审未替代这些 live gates。

## Exact one-shot command

```bash
env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  SNPSLMD_LICENSE_FILE=27030@ic.ismd-nemo \
  LM_LICENSE_FILE=/opt/synopsys/Synopsys.dat \
  M519_R15_EXPECTED_DC_RUNNER_SHA256=9ad15627c89bb078c5453333e979a1b98c2309b66fc71cce6b4f7aa4f89863b4 \
  M519_R15_EXPECTED_DC_LAUNCH_ADMISSION_SHA256=278eb851af42474e08887258006ef71137e28d0271f2e2e38ee77d783cd4238c \
  /home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/dc_handoff/scripts/run_dc_m519_r15_atomic_artifact_gate_three_axis_exact_sha_r1.sh
```

该命令最多执行一次。任何 preflight、license、axis、artifact、log、resource 或 final-ack 失败都必须保持 quarantine/noncitable。完成后还需 fresh production-result hammer，方可引用面积、setup、等带宽吞吐/mm² 等数字。
