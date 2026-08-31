# M622｜M621/M620/M617 r5 true-launch fresh hammer

日期：2026-08-28  
模式：`FRESH_INDEPENDENT_READ_ONLY_TRUE_LAUNCH_AND_ONE_SHOT_HAMMER__NO_FORMAL_ANALYZER`  
裁决：`PASS_M622_M621_M620_M617_R5_TRUE_LAUNCH_HAMMER`  
评分：**98/100**；`P0/P1/P2 = 0/0/0`

## 裁决

M621 production admission 与 true release 的 schema、SHA、member sidecar、outer seal 全部成立；admission 精确绑定 M617 r5 shell/Python、source contract/candidate、M615 true release、M616 FAIL evidence，以及 M620 PASS98 的 review/manifest/outer seal。只读 authorization predicate 返回 admission SHA `c3772a97597d9573a857591cf916a774897b315de87fc61177ab4e447d6c1f74`。

本 PASS 只允许 root 在**调用紧前重新完成 live resource/cgroup/UID collision、全链 SHA/seal 与坐标缺席检查后**，唯一调用一次 component analyzer。release 本身不是执行；raw result 仍需 fresh independent result hammer。

## Fail-closed 攻击

- 旧 M614 admission 由 M617 以 `authorization full path drift` 拒绝。
- 对 M615 release、M620 evidence、M621 authorization 分别注入删除和 SHA/内容篡改，6/6 全部在 analyzer 前拒绝；故障后真实链条 baseline 再次通过。
- 临时 results parent 上测试 10 类 blocker：result、attempt、consumed、普通/dot result staging、adapter internal staging、runtime、qraw、qstage、qfinal；每类 regular entry 与 dangling symlink 各一，共 **20/20** 由 `lexists/lstat` 谓词拒绝。
- validate-only 前后，production result/attempt/consumed 及所有 staging/quarantine 坐标均不存在。

## Fresh resource/cgroup/collision

重新采集 3 次、间隔 2 秒。最小 Commit headroom / MemAvailable / SwapFree 分别为 **83,900,324 / 414,579,836 / 57,216,252 KiB**，均超过门限；session 与 user cgroup 的 `failcnt/under_oom/oom_kill` 三次均为 0，UID-local M617/M612 analyzer 或 EDA collision 三次均为 0。

这些快照只证明 M622 时刻准入，不替代 root 在唯一 invocation 紧前的再次采样。

## 执行与 claim 边界

本评审仅运行一次 lineage preflight、只读 authorization validate-only、函数级身份故障注入和临时坐标故障注入。runner `--execute`、formal analyzer、正式 attempt/consumed/result、GPU、EDA、remote 均为 **0**。

输出仍是 component-only、per-frozen-sampled-inference；不是 camera-frame、paper data、system/full-network energy、system speedup 或 headline。`docs/359_DATE终局冻结_20260813.md` SHA 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
