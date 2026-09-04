# M516 H67 rank-3 isolated runner exact-SHA 静态打铁 r8

日期：2026-08-27  
审阅对象 SHA：`f973ebe8ec25944010334d2589e412161177781f186e5ec4f69cecc4fd38540b`  
结论：`STATIC_GO__REMOTE_CLEAN_EVALUATOR_IDENTITY_CLOSED__R7_NO_REGRESSION`  
评分：**97/100**  
P0：**0**  
P1：**8**  
远程 / GPU / 训练 / VCS / DC / PT / DSE 实际执行：**否**

## 终审结论

exact `f973ebe8...` 关闭了 r7 的 launch-side fail-closed：远端 clean pinned worktree 中
`run_h9_standard_valid825_eval.py` 的真实 SHA 是
`900cf8faa47cb4a2604ee0b500861b5d847cb074fe9c0ac6c09f010ebf955f3c`，不是本机工作副本的
`a9207ff...`。r8 只把 frozen-input 清单这一项改为远端 clean identity；其余脚本字节级保持
r7。

本审阅没有重新登录远端。远端 clean status、HEAD=`494593...`、真实文件 SHA 与语义 diff
来自 launch-side fail-closed 后的已核观察。本审阅独立证明的是：

1. 新 SHA 在 runner 中只出现一次，且只位于 `verify_frozen_inputs`；旧 SHA 不再出现；
2. 把该唯一新 SHA 逆替换成旧 SHA，整份 runner 的 SHA 精确恢复 r7
   `0722bd0c8cb244d333bbed37ad24433eb7b013b8e712efd94729cb42c4b5ddc3`；
3. r7 review 的双层 seal 仍通过，outer-seal file SHA 是
   `da1ca643f743609524c908b5b6d2c2ffac1ad4f961de2246db990f12df1a745e`；
4. 当前 runner `bash -n` 通过，docs/359 未变。

因此 r7 已闭合的 nested worktree/project root、M511 root-level M513 例外、数据/checkpoint
symlink、config receipt 重建、M513 sealed consumer、训练/valid receipt、双重复核和 final-only
same-parent atomic publish 均无退化。P0 为 0，允许 exact `f973ebe8...` 在全部运行时门通过后
启动五轮 rank-3 + valid825 **测量**。

## evaluator identity 修复是否合理

runner 目标明确是远端 pinned clean worktree，而非本机含其他研究修改的工作副本。因此
frozen SHA 必须绑定远端 clean 文件。运行开始及 prelaunch/end 前的 `verify_frozen_inputs` 都会
在内层 remote `REPO_ROOT` 执行 `sha256sum --strict -c`：

- 若 launch-side 观察的 `900cf8...` 不正确，runner 会在单元测试、配置、锁与 GPU 前退出；
- 若远端工作树随后改变，start/prelaunch/final 三阶段至少一个 exact hash gate 会退出；
- Git top-level、pinned HEAD、tracked/index cleanliness 另由外层 `WORKTREE_ROOT` gate 复核。

已核语义 diff 表明远端 clean evaluator 只是把 ATLIF/attention enabled/disabled module-count
审计拆开，检查更严格，没有放宽 sample=825、load missing/unexpected、module census、finite
metric 或 claim boundary。即使不依赖该语义判断，exact hash + downstream final receipt gates
也保持 fail-closed。

## r7 全门回归

- outer Git worktree / inner project root 分工：**PASS_NO_CHANGE**；
- Git pinned HEAD、tracked/index diff、关键 untracked-shadow：**PASS_NO_CHANGE**；
- inner data/checkpoint exact symlink target + SHA：**PASS_NO_CHANGE**；
- config receipt 在 inner root 重建与反复 path/SHA 验证：**PASS_NO_CHANGE**；
- M511 root-level M513 path 与 exact population/member/outer/identity consumer：
  **PASS_NO_CHANGE**；
- exact evaluator 远端 clean identity：**CLOSED**；
- ep40 45 factor pairs、105/45/60 census、same-chain 45/45 delta：**PASS_NO_CHANGE**；
- valid825 samples/load/module/finite metrics：**PASS_NO_CHANGE**；
- 四锁、GPU idle 双查询、runner 三阶段 SHA：**PASS_WITH_EXISTING_P1**；
- hidden same-parent staging 双重复核与最后唯一 `mv -T`：**PASS_NO_CHANGE**；
- publication 后命令数：**0**；
- hardware/DATE claim boundary：**PASS_NO_CHANGE**。

## P1（与 r7 相同）

1. M513 wait 只覆盖已存在 exact watcher tag，无 timeout/周期状态 receipt。
2. GPU 仍是 cooperative-only；未强制 exact-one A800/UUID/model/driver/index0。
3. untracked 扫描未覆盖 repo-root import shadow；DSEC 无 sample-content manifest。
4. final publish 前 attempt/result 失败无 early `FAILED_DO_NOT_CITE` 或 partial quarantine。
5. final receipt 未逐项重比 factor receipt 的所有 artifact SHA；传播的 M513 identity 最终仅
   type-check。
6. 第二轮复核不重新解析四个 JSON 的 schema/status/claim，也不重验 completion 文本；它
   依赖第一轮已验证且 member seal 未变。
7. persisted runner END SHA 早于 staging/seal/publication，无 true-end 第四次采样；不过
   exact reviewed source 的 publication 后无命令。
8. `mv -T` 非 kernel no-replace，且 verifier 对 staging root 使用 `.resolve()`；主动外部
   destination/staging race 仍需 `RENAME_NOREPLACE` + root `lstat`/dirfd 才能关闭。

## 准入边界

`STATIC_GO_FOR_EXACT_f973ebe8_SHA_ONLY`。启动时 runner SHA 必须仍为
`f973ebe8ec25944010334d2589e412161177781f186e5ec4f69cecc4fd38540b`；远端 evaluator 必须
精确为 `900cf8...f955f3c`，且 nested root、M511/M513、pinned commit、全部 source/input、
GPU/lock 门均通过。任何脚本变化使本准入失效。

本结论只允许 M516 五轮训练与 valid825 测量；不能据此准入 accuracy、INT8/QAT、cycle
speedup、energy、PPA 或 DATE headline。

`docs/359` 未修改，SHA 仍为
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
