# M516 H67 rank-3 isolated runner exact-SHA 静态打铁 r3

日期：2026-08-27  
审阅对象 SHA：`5f4c924225f464a30074792dc117b784887b9fe12c4a499e4652c1a6c0c318cd`  
结论：`STATIC_NO_GO__R2_P0_03_AND_P0_04_REMAIN_OPEN`  
评分：**78/100**  
P0：**2**  
P1：**6**  
远程 / GPU / 训练 / VCS / DC / PT / DSE 实际执行：**否**

## 结论

exact `5f4c...` 关闭了 r2 四个 P0 中的前两个：

1. source ep35 checkpoint 现在单独要求为 symlink，epoch36/ep40/state/
   config 仍要求 regular non-symlink，不再自相矛盾；
2. factor 更新证明不再使用 CPU 重建 ep35 SVD，而是比较同一训练
   进程产生的 epoch36 与 epoch40 checkpoint，并把 claim 精确收窄为
   `factor_values_proven_changed_epoch36_to_epoch40`。

但后两个 P0 **一行未关**：M513 root directory 仍在 `resolve()` 之后
才查 symlink；final package 仍是不完整 staging preverify + 无 canonical
quarantine 的 publication。因此本 SHA 不准启动 M516/A800。

## r2 四个 P0 逐项复核

| r2 P0 | r3 裁定 | 证据 |
|---|---|---|
| source checkpoint symlink 自相矛盾 | **CLOSED** | source 单独要求 `is_symlink()` + resolved regular；epoch36/40/state/config 拒绝 symlink |
| CPU/GPU SVD 假更新证明 | **CLOSED** | 只比同一训练链的 epoch36→epoch40，45/45 每对至少一个 factor tensor bitwise 变化 |
| M513 root/leaf symlink | **OPEN / P0** | root 仍 `Path(arg).resolve()` 后查 `is_symlink()`；四个 expected internal leaf 的检查有效 |
| final preverify/postverify/quarantine | **OPEN / P0** | staging 仍忽略 directory entries、不重验 member hashes；publish 后失败不 quarantine |

## P0-01：M513 root symlink 拒绝仍在 `resolve()` 之后

target SHA 中的 M513 consumer 仍是：

```python
directory = pathlib.Path(sys.argv[1]).resolve()
require(directory.is_dir() and not directory.is_symlink(), "M513 directory")
```

`resolve()` 会先跟随 leaf directory symlink，所以后续
`not directory.is_symlink()` 不可能发现 canonical M513 path 本身是链接。

内部四个 expected members 的 `p.is_file() and not p.is_symlink()` 是在未对 `p`
resolve 的情况下执行，这部分能正确拒绝 JSON/RUN_COMPLETE/member seal/
outer seal 作为 leaf symlink。但 `rglob("*") if p.is_file()` 仍不把额外空目录
或指向目录的 symlink 纳入 population。

**最小修复：**保留 lexical path，在 resolve 前要求它精确等于固定
`$M511_ROOT/$M513_REL`、存在、是 directory 且 `not lexical.is_symlink()`；再
resolve。population 应检查 every immediate entry，而不是只收集 `is_file()`。

## P0-02：final package 仍会留下未完整验证的 canonical PASS

target SHA 中的发布代码与 r2 裁定的版本相同：

1. staging 的 `actual` 仍用 `{p.name for p in directory.iterdir() if p.is_file()}`，
   额外空目录/指向目录的 symlink 不会破坏 exact population；
2. 写入 `SHA256SUMS` 后只验 outer digest 与 completion，没有在 rename 前
   重读 seal 并逐 member 复核 digest、duplicate/name set 和 JSON 语义；
3. `mv -T` 之后的 postverify 若失败，`set -e` 直接退出，但带
   `RUN_COMPLETE.txt` 的 canonical 仍保留；
4. 没有 `trap`、unique quarantine 和“先移除 canonical 再报错”的异常路径。

因此普通 postverify failure 与 rename/postverify 间的 SIGKILL 都可以留下下游
可见的 PASS package。

**最小修复：**把完整 consumer verifier 封装成同一段固定代码，rename
前在 staging 上执行一次，rename 后执行一次；检查 every entry、regular/
non-symlink、exact-seven population、所有 member hashes、outer seal、completion 和
JSON schema/status/claim boundary。任何普通 postpublish 失败必须先用原子 rename
把 canonical 移入事先检查 absent 的 unique quarantine，并确认 canonical absent/
quarantine present。

## 已关闭：source checkpoint 与 epoch36→epoch40 factor 证明

source 的新语义与外层 `verify_repo_identity` 一致：词法路径必须是
symlink，resolved target 必须是 regular file；外层还精确检查 target 路径和
SHA。config/epoch36/epoch40/state 必须是 regular non-symlink。该 r2 P0 关闭。

factor verifier 现在：

- ep40 精确 45 对 `(10,3)` / `(3,10)` factor keys，左右 prefix set 一致；
- ep36 精确同一组 45 prefixes/shapes；
- ep36/ep40 因素全部 finite；
- ep40 model 加载后 105 ATLIF，T10 rank3=45，T2 fallback=60，所有 T10
  `temporal_factor_load_source=checkpoint_factors`，missing/unexpected=0；
- 45/45 每对至少 left/right 一个 tensor 在 epoch36→epoch40 间 bitwise 改变；
- ep40 state 内部 epoch=4，optimizer state 非空；
- receipt 与 final claim 都只说 epoch36→epoch40 changed，不再声称相对 ep35
  pre-step initialization。

这个证明不覆盖第一个 epoch 的更新，但它没有声称覆盖；postflight
另外要求 five-epoch process 返回 0 并产生 epoch36--40 的 checkpoints/states。
当前措辞是对的。

## command substitution、trap、lock 与 GPU

- `M513_IDENTITY_JSON="$(python ...)" || die` 和 GPU identity assignment 都正确继承
  command-substitution 的 exit status；所有参数使用引号，多行 CSV/JSON 不会被
  shell word-splitting。
- 四重 `while ! flock ... || ...` 的短路与失败全解锁语义正确；全部成功
  后 fd 210--213 持续保持，watcher fd209 也穿过全流程。
- runner 没有 trap。对 final postpublication 这是 P0-02；对早期 attempt/result
  半成品则是 P1。
- GPU 在获得四锁后等待 compute PID 为空，并在全部 prelaunch 身份复核
  后再查一次。这对 cooperative jobs 是正确的；非合作 job 的最后窗口仍是 P1。

## M513 seals/identity、valid825 与 claim boundary

除 root lexical symlink P0 外，M513 consumer 已正确检查：

- four expected regular/non-symlink files；
- JSON/RUN_COMPLETE exact sealed set，member SHA 和 outer seal；
- strict JSON（duplicate key/non-standard constant 拒绝）；
- analyzer `9790...`、contract `e556...`、payload verifier `222d...`、M511 runner
  `788d...`；
- dynamic upstream seal 字段为 64-hex，M513 决策与所有硬件 claim 均为 false；
- consumed identity 写入 attempt，最终 receipt 引用它的 SHA。

valid825 最终要求 samples=825、missing/unexpected=0、ATLIF=105、attention=12，
AEE/AAE/AAE_Benchmark/DSEC_Fl 全部 finite/nonnegative。final receipt 仍把 accuracy/
cycle/energy/PPA/DATE headline 设为 false，没有将浮点 factor 训练冒充硬件准入。

## P1

1. **M513 wait 可用性：**只等待已存在的 exact process tag，无 timeout/周期状态
   receipt；watcher 未启动时会立即失败。
2. **GPU cooperative-only：**非合作作业仍可在第二次 query 后插入；只记录
   raw GPU CSV，没有强制 exact-one A800/UUID/model/driver/index0。
3. **untracked/env/data 只部分关闭：**三个主要 import root 与 PYTHONOPTIMIZE/
   PYTHONPATH/LD_PRELOAD 已处理，但 repo-root shadow 与 DSEC sample-content manifest
   未锁。
4. **attempt/result 半成品：**仍使用 `mkdir -p`，无 EXIT failure marker、
   `FAILED_DO_NOT_CITE` 或 partial quarantine；preflight/train/factor/valid 失败会阻塞
   安全重跑。
5. **final receipt 交叉身份：**最终只查 factor receipt 语义字段，没有把
   receipt 中 config/source/epoch36/epoch40/state SHA 与当前实物逐项重比；
   M513 receipt 在 final 阶段只要求 `upstream_identity` 是 dict。
6. **runner true-end 窗口：**start/prelaunch/end 实测 SHA 已写 receipt，但 end 采样
   仍早于 final package staging/seal/publish/postverify，发布后无第四次实测。

## 准入条件

`5f4c...` 不得启动。下一 SHA 须同时：

1. 在 resolve 前锁定并拒绝 M513 root lexical symlink，且 population 检查 every entry；
2. final staging 使用完整 consumer preverify；
3. postpublish 任何异常都先将 canonical 原子移入 unique quarantine；
4. 新 exact SHA 静态评审 P0=0。

`docs/359` 未修改，SHA 仍为
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
