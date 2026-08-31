# M516 H67 rank-3 isolated runner exact-SHA 静态打铁 r7

日期：2026-08-27  
审阅对象 SHA：`0722bd0c8cb244d333bbed37ad24433eb7b013b8e712efd94729cb42c4b5ddc3`  
结论：`STATIC_GO__NESTED_WORKTREE_PROJECT_ROOT_CLOSED__R6_ATOMIC_PUBLISH_PRESERVED`  
评分：**97/100**  
P0：**0**  
P1：**8**  
远程 / GPU / 训练 / VCS / DC / PT / DSE 实际执行：**否**

## 终审结论

exact `0722bd0...` 正确区分远端嵌套布局的两个根：Git worktree 是
`/root/private_data/work/m516_rank3_iso_20260827/SDformer`，项目根是其下的
`SDformer/`。Git commit、tracked/index diff 与关键 untracked-shadow 扫描全部以外层
`WORKTREE_ROOT` 为基准；算法、配置、结果、`data` 与 checkpoint 则全部以内层
`REPO_ROOT` 为基准。路径职责没有混用。

脚本在启动端再次 fail-closed 地要求两个目录存在、`git --show-toplevel` 精确等于外层
根、HEAD 精确等于 pinned commit；因此本次按要求不登录远端也不会把未经核实的本地目录
假设升级为事实。若远端布局、符号链接或 Git 根与合同不符，runner 会在任何训练前退出。

r6 已准入的 hidden same-parent staging、两轮封存复核与最后唯一 `mv -T` 原子发布链完整
保留。未发现因新增根拆分导致的退化。P0 为 0，允许 exact `0722bd0...` 在所有运行时
gate 通过后启动 M516 五轮 rank-3 + valid825 **测量**。这仍不是 accuracy、INT8、cycle、
energy、PPA 或 DATE headline 的硬件准入。

## r6 路径 P0 的闭合

### Git 身份属于外层 worktree

`verify_repo_identity` 在 `WORKTREE_ROOT` 上执行：

1. `rev-parse --show-toplevel` 必须精确返回 `WORKTREE_ROOT`；
2. HEAD 必须等于 `494593afa0ea81332ca21fcd68fdc9d6b72bbf1a`；
3. tracked worktree 与 index diff 都必须为空；
4. untracked executable/import shadow 路径带 `SDformer/` 前缀，正确从 Git 根定位内层项目。

因此不会再把内层项目目录误当作 Git 根，也不会因 scope 少一层而漏掉三个已冻结的重要
import/executable 目录。

### 算法、配置、数据与 checkpoint 属于内层 project root

`verify_frozen_inputs`、`verify_generated_config`、单元测试、配置生成、训练、valid825、attempt
receipt、result 与 canonical evidence package 均在或显式锚定 `REPO_ROOT`。`data` 必须是
`REPO_ROOT/data` 的 symlink，且解析后精确等于 original project 的 `data`；checkpoint 也
必须是内层项目相对路径上的 symlink，解析后精确等于 original project 中的冻结 checkpoint。
随后冻结输入 SHA 仍会逐项复核，符号链接只提供共享载荷，不放松身份合同。

配置生成在首次 `verify_generated_config` 之前运行，并带 `--force`，因此 config receipt 中
base/checkpoint/output 的绝对路径会按新内层 `REPO_ROOT` 重建；后续三次校验再逐项比较
resolved path 与 SHA。这里不存在把 r6 旧绝对路径回执误消费的问题。

### M511 根路径例外是有意且 fail-closed

M513 是独立 M511 capture worktree 的 root-level output，继续从
`/root/private_data/work/m511_capture_20260827/SDformer/hw_autoresearch_nts07/results/...`
消费，不应机械追加第二个 `SDformer/`。consumer 要求 exact-four top-level population、所有
成员 regular/non-symlink、member seal、outer seal、completion 文本、schema/status、四个
冻结实现 SHA 与 claim boundary。若该独立根合同不成立，会在 M516 配置、锁与 GPU 启动前
fail closed。

## r6 原子发布链回归

- canonical result/evidence 预存在检查仍锚定内层 `REPO_ROOT`；
- `FINAL_PARENT`、`FINAL_PATH` 与 `mktemp -d` staging 在同一 parent；
- 第一轮验证 exact-five 后生成 member/outer seal，并复核 exact-seven、类型、摘要与
  completion；
- 第二个独立 Python 进程再次复核 exact-seven、regular/no-symlink、member 与 outer seal；
- 两次 destination gate 都同时拒绝 existing object 与 dangling symlink；
- 脚本最后唯一命令仍为 `mv -T "$FINAL_STAGING" "$FINAL_PATH"`，之后没有命令。

所以非对抗性 SIGKILL 仍只能留下 hidden partial staging、hidden fully verified staging，或
已完整发布的 canonical package；runner 内部没有半发布 PASS canonical 或 post-publish
failure window。

## 全量关键回归

- exact runner SHA 三阶段自检：**PASS**；
- shell `bash -n`：**PASS**；
- outer Git root / inner project root 分工：**PASS**；
- inner `data` 与 checkpoint symlink exact target + frozen SHA：**PASS**；
- config receipt 在 inner root 重建并逐次 path/SHA 校验：**PASS**；
- M511/M513 独立 root-level consumer 合同：**PASS**；
- M513 exact population、双 seal、身份与 claim boundary：**PASS**；
- ep40 45 对 rank-3 factor、105/45/60 census、finite/load-source：**PASS**；
- same-chain epoch36→40 45/45 factor pair bitwise delta：**PASS**；
- valid825 sample/module/load/finite metrics gate：**PASS**；
- 四把 cooperative lock 与 GPU idle 双查询：**PASS_WITH_P1_BOUNDARY**；
- hidden same-parent staging 双重复核与 final-only atomic publish：**PASS**；
- 硬件/论文 claim boundary：**PASS**。

## P1

1. M513 wait 只覆盖已存在 exact watcher tag，无 timeout/周期状态 receipt。
2. GPU 仍是 cooperative-only；未强制 exact-one A800/UUID/model/driver/index0。
3. untracked 扫描未覆盖 repo-root import shadow；DSEC 无 sample-content manifest。
4. final publish 前 attempt/result 失败无 early `FAILED_DO_NOT_CITE` 或 partial quarantine。
5. final receipt 未逐项重比 factor receipt 的所有 artifact SHA；传播的 M513 identity 最终仅
   type-check。
6. 第二轮复核不重新解析四个 JSON 的 schema/status/claim，也不重验 completion 文本；它
   依赖第一轮已验证且 member seal 未变。
7. persisted runner END SHA 早于 staging/seal/publication，无 true-end 第四次采样；不过
   exact reviewed source 的 publication 后确实无命令。
8. `mv -T` 非 kernel no-replace，且 verifier 对 staging root 使用 `.resolve()`；主动外部
   destination/staging race 仍需 `RENAME_NOREPLACE` + root `lstat`/dirfd 才能关闭。

## 准入边界

`STATIC_GO_FOR_EXACT_0722bd0_SHA_ONLY`。启动时 runner SHA 必须仍为
`0722bd0c8cb244d333bbed37ad24433eb7b013b8e712efd94729cb42c4b5ddc3`，且外层 worktree、
内层 project、M511/M513、pinned commit、source/input、GPU/lock 等运行时门必须全部通过。
任何脚本变化使本静态准入失效并要求重新 exact-SHA 审阅。

本结论只允许 M516 五轮训练与 valid825 测量；结果尚未产生，不能据此准入 accuracy、
INT8/QAT、cycle speedup、energy、PPA 或 DATE headline。

`docs/359` 未修改，SHA 仍为
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
