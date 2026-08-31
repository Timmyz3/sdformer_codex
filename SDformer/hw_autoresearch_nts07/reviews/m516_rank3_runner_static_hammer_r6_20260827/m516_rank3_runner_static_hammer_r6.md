# M516 H67 rank-3 isolated runner exact-SHA 静态打铁 r6

日期：2026-08-27  
审阅对象 SHA：`6cae25eb84609cb7b8097a893c90c4160322bf20bc07a3a0231ac016ef1f108f`  
结论：`STATIC_GO__FULLY_PREVERIFIED_SAME_PARENT_ATOMIC_PUBLISH`  
评分：**96/100**  
P0：**0**  
P1：**8**  
远程 / GPU / 训练 / VCS / DC / PT / DSE 实际执行：**否**

## 终审结论

exact `6cae25...` 已关闭 r5 的最后一个 P0。新设计不再先 publish 再依赖
trap/quarantine 中和 PASS marker，而是把完整 canonical package 在同一 parent 下的
hidden staging 中生成、封存和验证；随后独立进行第二轮完整 population/member/outer
验证。canonical 路径在两个阶段均通过 `! -e && ! -L` gate，脚本最后唯一命令是：

```bash
mv -T "$FINAL_STAGING" "$FINAL_PATH"
```

该命令之后没有注释之外的 shell 命令、trap、postverify 或 receipt 写入。由于 source
与 destination 同 parent，正常文件系统上最终 publication 是同文件系统原子 rename。
所以任意非对抗性 SIGKILL 时窗只会留下两种可观察状态之一：canonical 不存在；或
canonical 已是完整、双层 seal 验证过的 package。不存在“canonical PASS 已出现、后续
命令失败却来不及撤销”的窗口。

P0 为 0，允许 exact `6cae25...` 在所有运行时 gate 通过后启动 A800 上的 M516
五轮 rank-3 + valid825 **测量**。这不是精度、INT8、cycle、energy、PPA 或 DATE
headline 的硬件准入。

## r5 P0 的闭合

r5 的根因是 canonical 先出现，随后 failure handler 在 PASS marker neutralization 前仍有
可失败分支。r6 删除了整套 postpublish repair 依赖：

1. `mktemp -d` 在 canonical 的同一 parent 创建 hidden staging；
2. 四份 receipt 与 exact completion marker 先复制/写入 staging；
3. 第一轮验证 initial exact-five population，拒绝 member symlink，生成 member seal 与
   outer seal；
4. 第一轮随后再次按 every top-level entry 验 exact-seven population、所有成员均为
   regular non-symlink、member digest/set、outer digest 和 exact completion text；
5. canonical absent/dangling-symlink gate；
6. 第二个独立 Python 进程重验 exact-seven、regular/no-symlink、member digest/set 与
   outer seal；
7. 第二次 canonical absent/dangling-symlink gate；
8. 以脚本最后命令完成 same-parent `mv -T`。

canonical 在第 8 步之前从未由 runner 创建，因此没有 PASS neutralization 顺序问题。
第 8 步成功时 package 已完整；失败时 `mv` 是最后命令，shell 自然返回非零。

## SIGKILL 全窗口攻击

### staging 构建或第一轮 seal/verify 期间

canonical 尚不存在。SIGKILL 最多遗留以 `.m516_rank3_valid825_evidence.staging.*`
命名的 hidden partial staging，不能被 exact canonical consumer 当作 PASS package。

### 第一轮与第二轮复核之间

canonical 仍不存在。第一轮已经验证 exact-seven、regular/no-symlink、member seal、outer
seal 与 exact completion text；第二轮还会独立重新读取目录和 seal。

### 第二轮复核后、rename 前

canonical 仍不存在，只有 hidden fully verified staging。在当前明确采用的静态、
非对抗 filesystem 模型下，这个对象不再变化。主动外部替换 staging 根或 member 的竞态
列为 P1，不伪装成 runner 内部可达 P0。

### rename 执行期间

source 与 destination 同 parent，因此 rename 不会退化为跨文件系统 copy/delete。
观察者只能看到原 hidden staging 或完整 canonical 之一，不会看到半复制目录。

### rename 之后

脚本已无任何后续命令。SIGKILL 只会终止即将自然退出的 shell；canonical 本身已经是
双层 seal 验证过的完整对象。

## same-parent / atomic / destination 语义

`FINAL_PARENT` 同时承载 `FINAL_PATH` 与 `mktemp -d` 生成的 `FINAL_STAGING`，满足
same-filesystem rename 的必要条件。两次 `[[ ! -e ... && ! -L ... ]]` 同时拦截普通
已存在对象和 dangling symlink。

GNU `mv -T` 不是 `renameat2(RENAME_NOREPLACE)`：在第二次 gate 和最终 rename 之间，
主动外部进程仍可创建 destination。若冲突对象导致 rename 失败，最后命令自然非零；若
底层允许 replacement，则被发布的仍是已验证 staging。唯一无法由 runner 证明的是外部
攻击者构造/切换 destination 或 staging root 并让 canonical 指向攻击对象。这要求主动
非合作 filesystem 修改，不是当前协作锁威胁模型中的内部执行路径，因此记 P1。

若要把该 P1 也关闭，应用小型 `renameat2(..., RENAME_NOREPLACE)` helper，并在第二轮
复核前后对 `FINAL_STAGING` 本身做 `lstat`，明确要求 root 是原始 regular directory、非
symlink，必要时持有目录 fd 后相对 fd 验证和 rename。

## stage symlink / extra entry

`mktemp -d` 初始创建的 staging 是真实目录。第一轮 seal 后的 population 使用 every
`iterdir()` entry，而不是仅 `is_file()` 过滤，因此额外目录、FIFO、socket 或 symlink
entry 都会造成 exact-seven mismatch；所有 expected member 还逐一要求 `is_file()` 且
非 symlink。第二轮在独立 Python 进程中重复 exact-seven 和 member 类型检查。

两个 Python verifier 对 staging root 调用了 `.resolve()`，因此“主动外部进程在复核
前用 symlink 替换 staging 根”未被 root `lstat` 直接拒绝；这是上一节所列的 P1
active-tamper 边界。脚本自身没有任何把 staging 根替换成 symlink 的路径。

## 错误与自然退出

- 任何 final rename 前的命令失败都由 `set -euo pipefail` 导致非零退出，canonical
  尚不存在；
- 第二次 destination gate 失败时调用 `die`，非零退出，runner 没有创建 canonical；
- final `mv -T` 失败时它本身就是脚本最后命令，进程自然返回 `mv` 的非零状态；
- final `mv -T` 成功时它也是最后命令，脚本自然返回 0；
- 没有 EXIT/signal trap 改写最后状态，也没有 postpublish 命令制造新的失败窗口。

## 全量旧 P0 回归

- Python `assert` fail-open：**CLOSED**，关键语义使用显式 `require`；
- M513 exact root/member/extra entry/member seal/outer seal/completion/identity：**CLOSED**；
- source checkpoint symlink 与 exact target/SHA：**CLOSED**；
- ep40 45 factor pairs、shape、finite、load-source、105/45/60 census：**CLOSED**；
- same-chain epoch36→40：45/45 pair 至少一个 tensor bitwise 改变：**CLOSED**；
- ep40 state internal epoch=4、optimizer state nonempty：**CLOSED**；
- valid825 samples=825、load audit=0/0、ATLIF=105、attention=12：**CLOSED**；
- AEE/AAE/AAE_Benchmark/DSEC_Fl finite 且 nonnegative：**CLOSED**；
- 四把 cooperative lock 的短路释放与成功持有：**CLOSED**；
- GPU idle 双查询和 identity receipt：**通过，非合作竞态保留 P1**；
- staging exact population/type/member/outer/completion：**CLOSED**；
- same-parent atomic canonical publication：**CLOSED**；
- postpublish failure window：**CLOSED**。

## P1

1. M513 wait 只覆盖已存在 exact watcher tag，无 timeout/周期状态 receipt。
2. GPU 仍是 cooperative-only；未强制 exact-one A800/UUID/model/driver/index0。
3. untracked 扫描未覆盖 repo-root import shadow；DSEC 无 sample-content manifest。
4. final publish 前 attempt/result 失败无 early `FAILED_DO_NOT_CITE` 或 partial quarantine。
5. final receipt 未逐项重比 factor receipt 的所有 artifact SHA；传播的 M513 identity
   最终仅 type-check。
6. 第二轮复核不重新解析四个 JSON 的 schema/status/claim，也不重验 completion 文本；
   它依赖第一轮已验证且 member seal 未变这一事实。
7. persisted runner END SHA 早于 staging/seal/publication，无 true-end 第四次采样；不过
   exact reviewed source 的 publication 后确实无命令。
8. `mv -T` 非 kernel no-replace，且 verifier 对 staging root 使用 `.resolve()`；主动
   外部 destination/staging race 仍需 `RENAME_NOREPLACE` + root `lstat`/dirfd 才能关闭。

## 准入边界

`STATIC_GO_FOR_EXACT_6cae25_SHA_ONLY`。启动时 runner SHA 必须仍为
`6cae25eb84609cb7b8097a893c90c4160322bf20bc07a3a0231ac016ef1f108f`，并且 exact
M513、pinned commit、source/input、GPU/lock 等运行时门必须全部通过。任何脚本变化都使
本静态准入失效并要求重新 exact-SHA 审阅。

本结论只允许进行 M516 五轮训练和 valid825 测量；结果尚未产生，不能据此准入 accuracy、
INT8/QAT、cycle speedup、energy、PPA 或 DATE headline。

`docs/359` 未修改，SHA 仍为
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
