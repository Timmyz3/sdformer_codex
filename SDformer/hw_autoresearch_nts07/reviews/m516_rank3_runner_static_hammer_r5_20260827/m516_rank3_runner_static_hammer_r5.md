# M516 H67 rank-3 isolated runner exact-SHA 静态打铁 r5

日期：2026-08-27  
审阅对象 SHA：`14145cada442abddcbd5108bf98150f279858325d10d8a1d91c016968f30931f`  
结论：`STATIC_NO_GO__PASS_MARKER_IS_NOT_NEUTRALIZED_FIRST`  
评分：**91/100**  
P0：**1**  
P1：**7**  
远程 / GPU / 训练 / VCS / DC / PT / DSE 实际执行：**否**

## 结论

exact `14145...` 几乎关闭 r4 唯一 P0：quarantine path 在 publish 前
要求 absent/non-symlink；HUP/INT/TERM 有显式 129/130/143；postverify 计入
every top-level entry；quarantine directory rename 不再吞错，且检查 canonical
absent、quarantine regular non-symlink directory 和 PASS marker absent。

但 failure handler 的顺序与修复描述相反：它不是先 neutralize canonical
`RUN_COMPLETE.txt`，而是先做两个可能 `die` 的检查：

1. quarantine target 必须仍 absent；
2. canonical completion 必须是 regular non-symlink file；
3. 然后才把 PASS marker 改名为 `POSTVERIFY_FAILED`。

如果外部进程在 publish 前 absent gate 与 failure handler 之间创建 quarantine
target，handler 会在 marker neutralization 之前 `die`。handler 开头已经清除
EXIT/HUP/INT/TERM traps，因此这个 `die` 不会再次尝试移除 canonical；
带 PASS marker 的 canonical 原地保留。marker 本身在 postverify failure 前被改成
symlink 时也有同样的提前退出路径。

因此 P0 仍为 1，`14145...` 不得启动。

## r4 P0 修复的正向部分

### collision pre-gate

`QUARANTINE_PATH` 基于本次 attempt tag（UTC second + PID）建立，在安装 trap
和 publish 之前要求 `! -e && ! -L`。如果旧 quarantine 已存在，runner
在 canonical publish 前失败，这条路径是 fail closed。

剩余问题是 gate 与 handler 使用之间的 TOCTOU。watcher lock 会排除另一个
cooperative M516，但不排除非合作进程或人工创建同名 target。正确顺序
应让这类 race 至多阻止 quarantine directory rename，不能阻止 PASS marker
neutralization。

### marker neutralization 与 directory rename

一旦实际执行到：

```bash
mv "$FINAL_PATH/RUN_COMPLETE.txt" \
   "$FINAL_PATH/RUN_COMPLETE.POSTVERIFY_FAILED.txt"
```

canonical 的 exact PASS marker 就消失，原有 member seal 也因 sealed member 缺失而必然
验证失败。此后即使 directory `mv -T` 失败或 SIGKILL，canonical 也不再
是可引用 PASS package。这是正确的故障优先级。

directory `mv -T` 不再有 `|| true`；若成功，后置检查要求：

- canonical `! -e && ! -L`；
- quarantine `-d && ! -L`；
- failed marker 存在；
- PASS marker 不存在。

这些 postconditions 正确。唯一 P0 是 marker neutralization 前仍有提前 `die`。

## P0：PASS marker 前面仍有可失败分支

handler 开头是：

```bash
trap - EXIT HUP INT TERM
if published && ! verified && canonical-is-dir; then
    quarantine-still-absent || die
    canonical-PASS-is-regular-nonsymlink || die
    mv PASS POSTVERIFY_FAILED
    mv -T canonical quarantine
    postconditions || die
fi
```

两个 `die` 都位于 neutralization 之前，且因 traps 已被清除而不可重试。

可达失败场景：

1. absent pre-gate 通过；
2. staging atomic publish 成功，`FINAL_PUBLISHED=1`；
3. postverify 返回非零；
4. 在 handler 前创建 `QUARANTINE_PATH`；
5. collision check 执行 `die`；
6. canonical `RUN_COMPLETE.txt` 仍存在。

这不需要不可捕获的 SIGKILL，是普通 postverify failure + path race。

**最小修复：**

1. handler 进入 published/unverified 分支后，在任何 quarantine collision 检查之前
   先中和 lexical canonical PASS marker；
2. neutralization 应使用不会跟随 source/destination symlink 的同目录原子操作，
   并在紧接的下一步确认 PASS marker absent；
3. marker 原本已 missing 时不应作为“留 PASS”故障；若为 symlink，应中和
   symlink 本身而不是在它前面 `die`；
4. 中和成功后再检 quarantine target 是否冲突；后续失败可以留下
   `POSTVERIFY_FAILED` canonical，但不能留 PASS canonical。

更干净的备选是：预验证 staging 中不放最终 admission marker，发布后
postverify 成功再以 exclusive/atomic 方式写入一个独立 admission marker。但若保留
“整目录原子发布”模型，则必须保证 failure handler 的第一个物理操作就是
PASS neutralization。

## trap / EXIT / signal / normal success

### 递归

handler 进入后立即 `trap - EXIT HUP INT TERM`，内部 `die`/`exit` 不会再次
触发 handler，无递归 trap。这部分通过。

### 退出码

- EXIT 使用 `quarantine_unverified_publish $?`，传递当前命令退出码；
- HUP/INT/TERM 分别明确传 129/130/143；
- handler 最后 `exit "$rc"`。

所以 r4 的“可捕获信号可能 exit 0” P1 已关闭。

### 正常成功

postverify 返回 0 后先设 `FINAL_VERIFIED=1`，再清除 EXIT/HUP/INT/TERM traps。
脚本以最后一个成功 `trap - ...` 的 0 状态结束。正常成功不会误
quarantine，也不会被 EXIT trap 改写退出码。这部分通过。

## SIGKILL 时窗

- prepublish preverify 前：canonical absent；
- preverify 后/rename 前：只有 hidden staging；
- atomic rename 中：只观察 staging 或 canonical 之一；
- rename 后/postverify 前：canonical 是已完整 preverified 对象；
- failure handler 在 marker neutralization 后：不论 directory rename 是否完成，都不再有
  exact PASS marker。

因此对静态、非对抗性 filesystem，原子发布本身的 SIGKILL 论证仍成立。
r5 P0 发生在 neutralization **之前**的普通提前退出，不是 SIGKILL 无法捕获
的局限。

## postverify 与 staging tamper

postverify 的 population 现在使用 every `iterdir()` entry，并对 exact seven 逐个
`is_file() and not is_symlink()`，因此 r4 的额外空目录/symlink-directory 缺口已
关闭。member hashes、sealed set 与 outer seal 重验也保持。

仍有 P1：postverify 没有重验 exact completion text 和四个 JSON 的 schema/status/
claim boundary；但该些内容在同一个已 preverified/sealed staging 中，非对抗性
模型下不升 P0。

seal 写入后、preverify 前的 staging member tamper 会被内容 SHA 捕获；extra
entry 会被 exact-seven 捕获。preverify 后到 atomic rename 前的外部主动竞态不在
当前非对抗性 seal 模型的保证范围，保持 P1 threat-model 说明即可。

## 全量旧 P0 回归

- Python `assert` fail-open：**CLOSED**；
- M513 seals/static identity/root/member/extra entry：**CLOSED**；
- source checkpoint symlink 语义：**CLOSED**；
- ep40 45 factor pairs/shapes/finite/load-source/105-45-60 census：**CLOSED**；
- CPU/GPU SVD 假证明：**CLOSED**，改为 same-chain epoch36→40；
- finite valid825：**CLOSED**；
- staging every-entry/member/outer/completion preverify：**CLOSED**；
- postverify every-entry/member/outer：**CLOSED**；
- quarantine collision/rename/postcondition：**PARTIAL，仍有 neutralization-order P0**。

## 其他通过项

- exact SHA 在审阅开始时为 `14145...`，`bash -n` 通过；
- command substitution 返回码、引号和 JSON/CSV 传递正确；
- 四重 cooperative lock 的短路/全解锁/成功持有正确；
- source/config/checkpoint/split/主要 entrypoint SHA 和 HEAD `494593...` 门保持；
- epoch36→40 每对 factor 至少一个 tensor bitwise 变化，ep40 optimizer state
  nonempty/internal epoch=4；
- valid825 samples=825、load audit=0/0、ATLIF=105、attention=12，四项 metric
  finite/nonnegative；
- runner start/prelaunch/end actual SHA 一致并进 receipt；
- accuracy hardware admission/cycle/energy/PPA/DATE headline 均 false。

## P1

1. M513 wait 只覆盖已存在 exact tag，无 timeout/周期状态 receipt。
2. GPU 仍 cooperative-only，未强制 exact-one A800/UUID/model/driver/index0。
3. untracked 扫描未覆盖 repo-root import shadow，DSEC 无 sample-content manifest。
4. final publish 前的 attempt/result 失败无 early failure marker/partial quarantine。
5. final receipt 未逐项重比 factor receipt 的 config/source/epoch36/epoch40/state SHA；
   M513 receipt 最终仅 type-check `upstream_identity`。
6. postverify 不重验 completion text 与 JSON schema/status/claim boundary，只依赖 preverified
   sealed object 不变。
7. persisted runner END SHA 仍早于 package staging/seal/publish/postverify，无 true-end 第四次
   采样。

## 准入条件

`14145...` 不得启动。下一 SHA 必须使 published/unverified handler 的
第一个物理效果成为 PASS marker neutralization；在此前不得有 quarantine collision/
marker-type 检查引发的提前 `die`。随后再检 collision、移 directory、检
postconditions。新 exact SHA 须再审，P0=0 才 GO。

`docs/359` 未修改，SHA 仍为
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
