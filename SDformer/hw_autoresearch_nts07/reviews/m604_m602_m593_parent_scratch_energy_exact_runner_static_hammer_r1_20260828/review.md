# M604｜M602/M593 parent-scratch energy exact-runner fresh static hammer

日期：2026-08-28  
模式：`FRESH_INDEPENDENT_READ_ONLY_RUNNER_STATIC_HAMMER__NO_FORMAL_EXECUTION_EDA_GPU_REMOTE`  
裁决：`FAIL_RUNNER_STATIC__TRUE_LAUNCH_ADMISSION_FORBIDDEN__R3_REPAIR_REQUIRED`  
评分：**60/100**；`P0/P1/P2 = 2/2/1`

## 1. 裁决与范围

本次只读审查 M602 runner、`launch_now=false/release=false` candidate、M602 handoff、M597 analyzer/
contract/handoff、M599 PASS 与全部 frozen business identities。执行了 `bash -n`、合法
`--preflight-only`、无 authorization 的 `--execute` fail-close，以及仅位于临时目录的 coordinate、
`RENAME_NOREPLACE`、原始 embedded verifier fault tests。没有正式运行 analyzer，没有创建 canonical
result/attempt/launch，没有 EDA/GPU/remote，也没有修改被审文件。

当前 candidate 的确不能执行：M604 true admission 不存在；无 authorization 的 `--execute` 在 attempt 前以
64 退出；preflight 输出规定 token 且 result/attempt/consumed 均 absent。runner 的 frozen identity、lexists/
lstat、trap-before-attempt、same-parent、普通 no-replace 和 pre-publish rehash 也有大量有效加固。

但 success verifier 可接受伪造的不完整 PASS tree，且 canonical result 发布后的任何失败都不会把它收入
quarantine。这两项都是可形成 false PASS 的 P0。因此本 review **不允许** root 另立 true launch admission。

## 2. P0 findings

### M604-P0-01｜embedded output verifier 可接受最小伪结果与空 terminal receipt

直接抽取 runner 内原始 `verify_analyzer_result_tree` 与 `verify_final_result_tree` heredoc，在临时目录构造并
双封以下 tree：result 只有 schema/status、两个仅含 design 的 row、一个布尔 conservation、两个 frozen
ablation 数和三个 claim flag；CSV 内容任意；`RUN_COMPLETE.txt` 内容为 `FAIL_NOT_COMPLETE`。第一个 verifier
仍返回 `PASS_M602_ANALYZER_RESULT_TREE`。

随后加入任意 stdout/stderr 和一个**无 schema、无 runner/analyzer/contract/auth identity、
`output_members_preseal={}`** 的 terminal receipt，按 runner 期望 member set 重封；第二个 verifier 仍返回
`PASS_M602_FINAL_RESULT_TREE`。

原因包括：未检查 result 精确顶层/row/conservation/claim key set；不核 CSV 与 JSON row；不核
`RUN_COMPLETE.txt` token；terminal receipt 不核 schema、identity、claim，也不要求
`output_members_preseal` 精确等于五个 preseal member。正式 analyzer 虽为冻结 source，但 runner 的职责是
对 runtime output fail-closed；当前 verifier 允许运行时替换/错误树发布。

影响：错误或被替换的 analyzer output 可通过 pre-publish、post-publish 两级检查并形成 sealed canonical
PASS result。

### M604-P0-02｜post-publish/attempt-seal/consume failure 留下 canonical PASS tree

`cleanup()` 只移动 `ATTEMPT_DIR`、runner staging 与 analyzer internal staging；源码中不存在
`rename_noreplace "$RESULT_DIR" <quarantine>`。第 630 行发布 result 后，若第 632--644 行的 canonical
rehash、static rehash、attempt completion/seal 或 consume 失败/收到 signal，cleanup 最多隔离 attempt，
**canonical result 原地保留**。failure receipt 仅记录 `canonical_result_lexists=true`，不会使该 tree
不可引用。

更窄的 signal 窗口位于 attempt 已移到 consumed 之后、`SUCCESS=1` 之前：cleanup 条件看不到 attempt、staging
或 internal staging，会完全不动作并以失败退出，result 与 consumed attempt 都仍存在。

影响：runner 失败退出后仍可留下 member-set、terminal receipt 和双封均看似完整的 canonical PASS tree，
直接违反“任一 post-attempt failure 不可能形成 false PASS”。

## 3. P1 findings

### M604-P1-01｜post-publish 与 consume 后的 terminal identity 不完整

pre-publish 会直接比较 runner/auth SHA并执行 `verify_future_authorization`、`verify_static_identity`。但
post-publish 只有 `verify_final_result_tree` 与 `verify_static_identity`：没有重新比较 auth SHA，也没有再次
执行 `verify_future_authorization`。attempt `seal_tree_exact` 后未立即验证 seal，移动为 consumed 后也没有对
consumed canonical 做 exact member/manifest/outer rehash。authorization 或 consumed attempt 在最后窗口漂移
仍可输出 success。

### M604-P1-02｜analyzer→runner-staging 的第一次 publish 仍是可覆盖 `os.rename`

runner 对最终 `STAGING_DIR→RESULT_DIR`、attempt consume 和 quarantine move 使用了真实
`renameat2(RENAME_NOREPLACE)`，collision 临时测试保持 source/target 不变。但冻结 M597 analyzer 在
`publish_result()` 中仍用 `Path.exists()` 检查 output/staging，并在第 563 行用 `os.rename(internal_staging,
STAGING_DIR)` 发布。runner 只在 attempt 前检查 STAGING absent，没有 fd/dir lock 或 no-replace 包装阻止
检查后到 analyzer rename 前的 entry 注入。M599 要求的 no-clobber hardening 因此未覆盖第一次 publish。

## 4. P2 finding

### M604-P2-01｜sealed-input directory verifier 不拒绝未列入 manifest 的额外 member

`verify_sealed_dir()` 验证 manifest 中每个 member，但不枚举 directory 的 actual regular-file set 与 manifest
set 完全相等。固定业务 JSON 本身仍有 exact SHA，故这不是当前数值 P0；建议下一版采用 exact member set，
与 result tree verifier 的标准一致。

## 5. 通过项

- runner SHA `6a54d938f598835114c2e463e56eb03f4e0754947dbbeb0b33f03fd04e569b2c`；candidate
  SHA `4261d4a4409e37e580b930afd239a3d4d8d65a851cdd4c78ebe3d86e568c0574`，双封均通过。
- M597 analyzer/contract/handoff、M599 review、M504/M528/macro-map/M595/docs359 frozen identity 的当前
  path/SHA/manifest/outer preflight 通过。
- `bash -n`、preflight token PASS；`launch_now=false/release=false`；无 authorization execute=64，且未创建
  result/attempt/consumed。
- coordinate policy 在临时目录接受四个 absent same-parent entry，并拒绝 dangling symlink；
  `RENAME_NOREPLACE` collision 返回失败且 source=`A`、target=`B` 均未变化。
- trap 安装行 565 早于 attempt mkdir 行 571；普通 pre-publish failure 可收集 attempt、runner staging 和
  analyzer internal staging并递归双封 quarantine。
- `docs/359_DATE终局冻结_20260813.md` SHA 保持
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## 6. 下一步

不得另立 M604 true launch admission。只允许新建 runner r3 identity：

1. output verifier 冻结 result/row/conservation/claim/CSV/RUN_COMPLETE/terminal receipt 的 exact schema、key
   set、值与交叉 hash；`output_members_preseal` 必须精确五项；
2. failure FSM 必须在任何 post-publish failure 时把 canonical result、attempt/consumed、所有 staging 一起移入
   唯一双封 quarantine，并最终断言所有 canonical success coordinate `lexists=false`；
3. post-publish 与 post-consume 再验 authorization、runner、全部 static identity、result 和 consumed seal；
4. analyzer 第一次 publish 也必须通过真实 no-replace 边界；
5. 新 identity 再做 fresh hammer，P0/P1=0 后 root 才能另立 true admission。正式 result 之后仍需独立 result
   hammer；`38.2283079189%`/`1.2622562287 mJ` 继续只作诊断。
