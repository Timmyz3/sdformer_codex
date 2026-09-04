# M516 H67 rank-3 isolated runner 独立静态打铁 r2

日期：2026-08-27  
审阅对象 SHA：`76ba7cd4dde0e494c9109d9e3b1b58d9a9acc2034c5b7665bc2a0e566d947c80`  
结论：`STATIC_NO_GO__FOUR_FAIL_CLOSED_GAPS_REMAIN`  
评分：**67/100**  
P0：**4**  
P1：**6**  
远程 / GPU / 训练 / VCS / DC / PT / DSE 实际执行：**否**

## 结论

r2 相比 r1 有明显进步：关键 Python `assert` 已改为显式 `require`；
M513 member/outer seal 和主要上游 SHA 已检查；ep40 已直接检查 45 对
factor key/shape、45 个 T10 rank-3、60 个 T2 fallback 和
`checkpoint_factors` load source；四项 valid825 metric 已要求 finite/
nonnegative；runner start/prelaunch/end 实测 SHA 已写 receipt；最终证据已有
staging、member seal、outer seal 和 rename publication。

但 exact `76ba...` 仍不能进入 A800 队列。其中一个 P0 会使 canonical
run **必然失败**，另一个 P0 会把 CPU/GPU SVD 差异误当作训练更新，
其余两个 P0 使 M513 path 与最终 canonical PASS package 仍未完整
fail closed。

审阅开始时已实测对象为 `76ba...`并完整读取；审阅期间同路径
被其他进程改成了 `5f4c9242...`。本 r2 **只封存 `76ba...` 的裁定**，
不声称审过任何后续 SHA。

## r1 问题关闭矩阵

| r1 项 | r2 裁定 | 说明 |
|---|---|---|
| Python `assert` fail-open | **CLOSED** | 三组关键检查全部使用显式 `require`，并 unset `PYTHONOPTIMIZE/PYTHONPATH/LD_PRELOAD` |
| M513 seals/identity | **PARTIAL / P0** | member/outer seals 和四个主要 SHA 已锁，但 directory symlink 拒绝被 `resolve()` 消除 |
| ep40 factor 身份 | **PARTIAL / P0×2** | key/shape/census/load-source 闭合，但 source symlink 自相矛盾，且更新证明用了不同设备的 SVD |
| GPU race/identity | **PARTIAL / P1** | 加了第二次 busy query 和身份文本，但未锁 A800 UUID/model，竞态窗仍在 |
| untracked/env/data | **PARTIAL / P1** | 三个主要 import root 和三个 env 已处理，未覆盖全 import/data 身份 |
| 失败半成品 | **OPEN / P1** | attempt/result 仍 `mkdir -p`，无 EXIT marker/quarantine |
| finite metrics | **CLOSED** | AEE/AAE/AAE_Benchmark/DSEC_Fl 要求 finite 且非负 |
| atomic sealed publication | **PARTIAL / P0** | 已 staging+seal+rename，但 preverify 不完整且 postverify 失败不 quarantine |
| actual runner SHA | **CLOSED_WITH_P1_WINDOW** | 三个实测值已写入，但“end”在 package publish 之前 |

## P0-01：source checkpoint symlink 条件自相矛盾

`verify_repo_identity` 明确要求
`$REPO_ROOT/$CHECKPOINT_REL` 是 symlink，且 resolved target 精确等于
original repo 中的 ep35 checkpoint。

ep40 factor verifier 随后把同一词法路径作为
`source_checkpoint_path` 传入，并对 config/source/ep40/state 统一执行：

```python
require(path.is_file() and not path.is_symlink(), ...)
```

shell 没有在传参前对 source path 执行 `readlink -f`，`map(pathlib.Path, ...)`
也不会自动 resolve，因此 source 的 `is_symlink()` 必然为 true。这不是竞态或
边界情形，而是 canonical 路径的确定性死路：5 epochs 即使成功，runner 也会
在 valid825 前失败并留下半成品。

**最小修复：**对 source checkpoint 单独实施“词法路径必须是 symlink +
resolved target 必须是指定 regular file + SHA 精确”；或在已完成 shell 的 symlink/
target/SHA 检查后，只把 `source_checkpoint_path.resolve()` 传入 factor verifier。
ep40 checkpoint/state 仍应要求 regular non-symlink。

## P0-02：CPU 重建 SVD 不是 GPU 训练初值证明

训练 entrypoint 先将 base model `model.to(device)`，再安装 ATLIF 并从 ep35
迁移 factors。`_initialize_temporal_factors_from_dense()` 内部的
`torch.linalg.svd` 因而在 GPU 上执行。

r2 verifier 却显式用 `CUDA_VISIBLE_DEVICES=''` 在 CPU 重建 source model，然后
用 `torch.equal(ep40_factor, cpu_rebuilt_factor)` 判断每对 factor 是否更新。

这个证明不成立：SVD 的符号/子空间基本来就不唯一，CPU 与 CUDA
backend 的浮点结果也不保证 bit-exact。即使 optimizer 一步未更新，45 对
GPU initialization 与 CPU reconstruction 也可能全部 `torch.equal == false`，从而
写出虚假的 `factor_values_proven_changed_from_initialization=true`。

**最小修复：**不要事后重做 SVD。由实际训练进程在加载 ep35 后、
第一个 optimizer step 前导出 45 对 factor 的 canonical tensor snapshot/hash 与 exact
prefix/shape census，将该初值 receipt 封存后再与 ep40 逐对比较。若不愿
修改训练 entrypoint，就删除“proven changed”，只宣称 factor checkpoint emitted。

## P0-03：M513 leaf-directory symlink 拒绝实际失效

M513 consumer 先执行：

```python
directory = pathlib.Path(sys.argv[1]).resolve()
require(directory.is_dir() and not directory.is_symlink(), ...)
```

`resolve()` 已经跟随并消除了 leaf symlink，所以后续
`not directory.is_symlink()` 无法拒绝固定 M513 路径本身被替换为指向另一个
sealed directory 的链接。它还会将 resolved 后的非 canonical 路径写入传递
receipt。

member exact set、member SHA、outer seal、strict JSON 以及 analyzer/contract/verifier/
runner SHA 检查本身都是正确增量，但这个 lexical path 缺口意味着 r1
的 M513 P0 尚未完全关闭。

**最小修复：**先保留 `lexical = Path(sys.argv[1]).absolute()`，在任何
`resolve()` 之前要求 lexical 精确等于固定 canonical path、存在、且
`not lexical.is_symlink()`；然后再 resolve 并检查 members/seals。

## P0-04：canonical evidence publication 仍可留下未验证 PASS

最终 package 已改用 same-parent staging，这是对的。但发布前的 Python 只：

- 用 `p.is_file()` 建立 member set，因此额外空目录/指向目录的 symlink 不进集合；
- 写 member digests 和 outer digest；
- 复核 outer digest 和 completion 文本；
- **没有**在 rename 前重读全部 member seal 并验证每个 member hash/
  exact all-entry population。

`mv -T` 后才执行较完整的 postverify；如果 postverify 失败，`set -e`
直接退出，但 canonical 目录中的 `RUN_COMPLETE.txt` 和 seals 仍留在原位。
SIGKILL 发生在 rename 与 postverify 之间也同样。下游仅消费 package 会看到
可引用的 PASS canonical，这不是 fail closed publication。

**最小修复：**在 staging 中先用与 consumer 完全同一的 verifier 执行
all-directory-entry exact population、non-symlink regular members、member seal、outer seal、
completion 及 JSON 语义检查；只发布该已预验证目录对象。普通 postverify
异常必须先原子把 canonical 移到 unique quarantine 再退出，并检查
canonical absent/quarantine present。

## shell command substitution 与 lock 语义

本轮未发现 P0：

- `M513_IDENTITY_JSON="$(python ...)" || die` 中 assignment 的 exit status 传递自
  command substitution，Python 失败会进入 `die`；输出为单行 JSON，之后使用
  `printf '%s\n'` 不损坏 JSON。
- `GPU_IDENTITY_JSON="$(nvidia-smi ...)" || die` 同理，引用传参保留 CSV
  中的空格与换行。
- `while ! flock -n 210 || ...` 的短路语义正确：任一 lock 失败即
  解除本轮已获取的全部 lock；全部成功时跳出 while 且 fd 继续持有。
- watcher lock 在任何工作前获取，四个队列/GPU locks 穿过训练与
  valid825 保持。

## P1

1. **M513 等待可用性。** 只有 exact watcher tag 已存在时才等待，watcher 尚未
   启动或 tag 变更会立即失败；无 timeout/周期状态 receipt。
2. **GPU 仍是 cooperative-only。** 第二次 busy query 缩小但没有消除非合作
   job 在 query 与 CUDA context 创建间插入的窗口。记录了 raw GPU CSV，但
   没有要求精确一张 A800、指定 UUID/model/driver/index0。
3. **untracked/env/data 只部分关闭。** 检查了 entrypoints/overlay/baseline 三个目录，
   但非白名单的 repo-root import shadow 仍可存在；DSEC 只锁 data symlink 与
   split CSV，没有样本内容 manifest。
4. **attempt/result 失败路径未关闭。** `mkdir -p` 后无 `EXIT` trap、
   `FAILED_DO_NOT_CITE` 或 partial-result quarantine；任意 preflight/train/factor/valid 失败
   都留下阻塞重跑的半成品。
5. **receipt 交叉身份可再加固。** final verifier 检 factor receipt 的语义字段，
   但没有将其 config/checkpoint/state/source SHA 逐项与当前实物交叉比对；
   M513 propagated receipt 在 final 阶段仅要求 `upstream_identity` 是 dict。
6. **runner 的 end 仍有小窗口。** `RUNNER_SHA_END` 在 final receipt 之前采样，
   随后还有 repo/input 复核、staging/seal/mv/postverify；发布完成后没有第四次
   实测 runner SHA 或失败 quarantine。

## 已通过的工程检查

- target SHA 在审阅开始时精确为 `76ba...`，`bash -n` 通过；
- config receipt 的 schema/status/path/SHA/rank/module/headline 使用显式检查；
- HEAD 固定 `494593...`，tracked worktree/index 多次检查；
- source checkpoint/base config/split CSV/主要脚本 SHA 已锁；
- generated config 固定 SHA 且 launch 前复核；
- train nonzero exit 传递，epoch36--40 checkpoint/state 存在性检查；
- ep40 checkpoint 的 factor key pair、shape、model prefix、module census、load audit 都是
  实质增量，修复 P0-01/P0-02 后应保留；
- valid825 绑 epoch40、samples=825、ATLIF=105、attention=12，四项 metric
  finite/nonnegative；
- 最终 claim boundary 仍明确为 floating-factor measurement，accuracy/INT8/
  cycle/energy/PPA/DATE headline 均未准入。

## 准入条件

下一 SHA 只有同时满足以下条件才能 GO：

1. 修正 source symlink 确定性死路；
2. 使用真实训练进程的 pre-step factor snapshot，或降级“changed”宣称；
3. 在 `resolve()` 前关闭 M513 lexical canonical/non-symlink path；
4. final staging 完整 preverify，postverify 失败先 quarantine；
5. 新 SHA 稳定后重做静态复审，P0=0。

`docs/359` 未修改，本轮复核 SHA 仍为
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
