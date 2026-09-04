# M516 H67 rank-3 isolated runner exact-SHA 静态打铁 r4

日期：2026-08-27  
审阅对象 SHA：`159dbadd5e6f62e0118f918bf7afeba46d7f3804d38f8519cbc7dfd32c0ca335`  
结论：`STATIC_NO_GO__QUARANTINE_OPERATION_IS_NOT_FAIL_CLOSED`  
评分：**87/100**  
P0：**1**  
P1：**8**  
远程 / GPU / 训练 / VCS / DC / PT / DSE 实际执行：**否**

## 结论

exact `159d...` 已关闭 r3 的两个 P0 主体：

- M513 lexical root 在 `resolve()` 前要求 absolute/directory/non-symlink，top-level
  检查 every entry，再对 exact four members 要求 regular non-symlink；
- final staging 在写 seals 后重读 every entry、全部 member digests、outer seal 和
  completion，然后才执行 same-parent atomic `mv -T`。

因此 SIGKILL 在 rename 前只会留下非 canonical staging，在 rename 中/后则留下
已完整 preverify 的 canonical directory；这个 atomic-publish 论证成立。

但 ordinary postverify failure 的 quarantine 操作本身不 fail closed：目标未预先
要求 absent，`mv` 失败被 `|| true` 吞掉，且无 postcondition。一旦 quarantine
rename 因目标冲突、权限或其他原因失败，带 `RUN_COMPLETE` 的 canonical
会原地保留，trap 却打印“quarantined”后退出。这是 P0，所以本 SHA
仍不准启动。

## r3 两个 P0 关闭情况

### M513 lexical root/member/extra-directory：关闭

M513 input 来自 shell 固定的 absolute path。Python consumer 现在保留词法
`Path`，不先 resolve，要求：

1. absolute；
2. `is_dir()` 且 root leaf `not is_symlink()`；
3. `iterdir()` 的 every top-level entry 名集精确等于四个 expected names；
4. 四个 member 全部 `is_file()` 且 non-symlink。

所以 root symlink、expected-member symlink、额外 regular file、额外空目录和指向
directory 的 symlink 都会 fail closed。随后 member seal exact-two set、digest、
outer seal、completion、strict JSON、主要上游 SHA 和硬件 claim boundary 检查
均保持。

唯一 P1 是代码没有再显式比较传入 path 字符串与一个 Python 内嵌
canonical constant；但 shell 参数本身是固定常量、已引用传入，不构成可达 P0。

### staging tamper 与 SIGKILL atomic publish：主体关闭

staging 中的五个 evidence members 写入后，脚本生成 member seal 和 outer
seal，然后在 rename 前执行：

- every-entry exact-seven population；
- seven entries 全部 regular non-symlink；
- seal 每行 format、64-hex、name allowlist、no duplicate、内容 SHA；
- exact-five sealed set；
- outer format/digest；
- exact completion text。

因此 seal 写入后、完整 preverify 前的 member tamper 会被 digest 检查捕获；
extra directory/symlink 会被 every-entry population 捕获。

`FINAL_PATH` 发布前要求不存在，staging 与 canonical 在同一 parent
filesystem，`mv -T` 是 directory rename：

- SIGKILL before rename：canonical absent，最多留 hidden staging；
- SIGKILL during rename：原子语义下只会观察到 staging 或 canonical 之一；
- SIGKILL after rename/before postverify：canonical 是同一个已 preverified directory 对象。

所以“SIGKILL 来不及 postverify”本身不是 P0。剩下的 P0 是已经进入
ordinary/catchable-failure trap 后，quarantine rename 失败的处置。

## P0：quarantine rename 失败被吞掉

trap 中的关键逻辑是：

```bash
local quarantine_path="${FINAL_PATH}.quarantine.${ATTEMPT_TAG}"
mv -T "$FINAL_PATH" "$quarantine_path" || true
echo "... quarantined ..."
```

问题有三个：

1. 安装 trap 前没有构造/封存 unique quarantine path，也没有要求
   `[[ ! -e "$quarantine_path" && ! -L "$quarantine_path" ]]`；
2. `mv` 失败被 `|| true` 吞掉；
3. 没有检查 `FINAL_PATH` 已 absent 且 quarantine 是 expected non-symlink directory。

一个 pre-existing quarantine directory 就足以使 GNU `mv -T` 失败。更一般地，任何
rename failure 都会留下 canonical PASS。原始 postverify 失败码会被传递，但
下游不一定看 runner exit code，它可能只看 canonical `RUN_COMPLETE`/seals。

**最小修复：**

1. 在安装 trap 之前一次性生成 quarantine path，带 PID/timestamp/random UUID，并
   要求 `! -e && ! -L`；
2. trap 中不得无条件吞掉 `mv` failure；
3. rename 后明确检查 `! -e FINAL_PATH && ! -L FINAL_PATH`、quarantine
   `-d && ! -L`；
4. 只在 postcondition 成立后打印“quarantined”；若连 quarantine 都失败，
   必须显式打印“CANONICAL MAY REMAIN”的高优先级 failure，不能冒充成功。

## trap / EXIT / signal 语义

正向部分：

- `FINAL_PUBLISHED` 只在 atomic `mv` 返回 0 后设为 1；
- `FINAL_VERIFIED` 只在 postverify 返回 0 后设为 1；
- postverify 被 `if ! python ...; then die; fi` 包装，失败会经 `die` 带 90
  进入 EXIT trap；
- trap 开头先 `trap - EXIT HUP INT TERM`，其内 `exit` 不会递归重入；
- 成功路径设 `FINAL_VERIFIED=1` 后清除 traps。

剩余两个 P1：

- HUP/INT/TERM 进入同一函数时用 `local rc=$?`，不保证传递标准
  `128+signal` 码；如果信号恰好在上一命令成功后到达，可能 exit 0。
  应给 EXIT/HUP/INT/TERM 分别包装显式码。
- trap 只在 final publish 窗口安装；更早的 preflight/train/factor/valid825
  failure/signal 仍只留半成品，无 `FAILED_DO_NOT_CITE`/partial quarantine。

## 其他既有链复核

### source checkpoint 与 factor

- outer repo check 要求 source checkpoint 是指向 exact original ep35 path 的 symlink，并
  锁定 SHA `4f33...`；
- factor verifier 对 source 单独要求 symlink + resolved regular，对 config/epoch36/
  epoch40/state 要求 regular non-symlink；
- ep40 的 45 对 factor key/shape/prefix/finite，105/45/60 census，
  `checkpoint_factors` load source 和 missing/unexpected=0 均检查；
- epoch36 的同 45 prefixes/shapes/finite 均检查；45/45 每对至少一个 tensor
  在 epoch36→40 变化；
- ep40 training state 要求 internal epoch=4 且 optimizer state nonempty；
- claim 仍只是 epoch36→40 changed，没有退回 CPU/GPU ep35 SVD 对比。

### command substitution、locks 和 GPU

- M513/GPU identity command substitution 的 exit status 传递和引号正确；
- `while ! flock ... || ...` 的短路、失败全解锁、成功 fd 持有语义正确；
- watcher + algorithm/factorial/local5/A800 cooperative locks 穿过 train/valid/final publish；
- compute PID 等待为空，并在 prelaunch identity 复核后二次查空；
- 仍未消除非合作 GPU job 竞态，也未强制 exact-one A800/UUID/model/driver/
  index0，它们保持 P1。

### finite valid825、runner SHA 和 claim boundary

- valid825 samples=825，missing/unexpected=0，ATLIF=105，attention=12；
- AEE/AAE/AAE_Benchmark/DSEC_Fl 全部 finite 且 nonnegative；
- runner start/prelaunch/end 实测 SHA 一致且已写 receipt，不过 end 仍早于
  package publication，保留 P1；
- final claim 明确禁止 accuracy hardware admission、cycle speedup、energy、
  paper PPA 和 DATE headline。

## P1

1. M513 watcher 只等待已存在的 exact tag，无 timeout/周期状态 receipt。
2. GPU 仍为 cooperative-only；raw identity 没有强制 exact A800/UUID/model/driver/index。
3. untracked 扫描只覆盖三个主要 import roots，repo-root shadow 与 DSEC
   sample-content manifest 未完整锁定。
4. final publish 前的 attempt/result 失败无 EXIT marker 或 partial quarantine。
5. final verifier 没有把 factor receipt 里的 config/source/epoch36/epoch40/state SHA
   与当前实物逐项交叉比对；M513 propagated receipt 在 final 阶段仅 type-check。
6. published postverify 仍用 `if p.is_file()` 构造 population，不拒绝 postpublish
   新增的 empty/symlink directory，也不重验 completion JSON 语义。
7. HUP/INT/TERM 的 trap exit code 未显式固定为 129/130/143，可能以 0 退出。
8. persisted runner END SHA 采样早于 package staging/seal/publish/postverify，无 true-end
   第四次 SHA。

## 准入条件

`159d...` 不得启动。下一 SHA 至少必须：

1. quarantine path 在 trap 安装前生成且验证 absent/non-symlink；
2. quarantine `mv` 不得被静默吞掉；
3. 检查 canonical absent + quarantine present/non-symlink directory 后才报成功；
4. 新 exact SHA 再做静态打铁，P0=0。

`docs/359` 未修改，SHA 仍为
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
