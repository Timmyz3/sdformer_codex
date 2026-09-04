# M516 H67 rank-3 isolated runner 静态打铁 r1

日期：2026-08-27  
结论：`STATIC_NO_GO__THREE_FAIL_CLOSED_GAPS_BEFORE_A800_LAUNCH`  
评分：**61/100**  
P0：**3**  
P1：**6**  
远程 / GPU / 训练 / VCS / DC / PT / DSE 实际执行：**否**

## 结论

当前 runner SHA 与委托身份一致：

```text
ddb20a53a13ead509b3f5bae5acb175434a49edf713dd51792a055e57d6e52f1
```

shell 语法、四重 cooperative lock、结果目录拒绝覆盖、训练返回码传递、
epoch36--40 文件存在性检查和硬件 claim boundary 基本方向正确。但这个
SHA **不得启动 A800 canonical M516**：三处 P0 使它无法 fail closed，也无法
证明最终 ep40 确实是完整 rank-3 factor 训练结果。

本评审没有打开远程、没有占 GPU，没有执行训练或 EDA，也没有修改
`docs/359`。

## P0-01：所有关键 Python 门都用 `assert`

runner 的三个 inline Python 分别用 `assert` 审计：

- 动态 config receipt 的 schema/status/path/SHA/rank/module count/headline；
- M513 schema/status/`new_performance_rtl_authorized=false`；
- postflight status、815 samples、checkpoint load audit 和 module count。

Python 在 `PYTHONOPTIMIZE=1` 或 `-O` 下会删除 `assert`。runner 没有清理继承的
`PYTHONOPTIMIZE`。本轮本地纯 CPU 最小复现：

```text
PYTHONOPTIMIZE=1: ASSERT_SKIPPED_UNDER_PYTHONOPTIMIZE, rc=0
normal python3: AssertionError, rc=1
```

因此当前 SHA 可在关键字段错误时继续到 PASS 发布。

**最小修复：**把三个 heredoc 里的全部 `assert` 改成明确的
`require(condition, message)` / `if not condition: raise RuntimeError(...)`。可同时
`unset PYTHONOPTIMIZE` 并拒绝非空值，但这不能代替显式检查。

## P0-02：M513 等待后没有消费 M513 的封存身份

M513 analyzer `9790f62d...` 已将 JSON 和 `RUN_COMPLETE.txt` 作为 exact-two
members，写 `SHA256SUMS` 与 outer seal，在 staging 里预验证后原子发布。
M516 却只读固定路径的 PASS 文本和 JSON 三个字段，没有：

- 拒绝 leaf symlink 和额外 member；
- 验证 `SHA256SUMS` 以及 `SHA256SUMS.seal.sha256`；
- 锁定 M513 analyzer/contract/capture/payload-verifier/M511 runner 身份；
- 把实际 M513 seal 身份写入 M516 launch/final receipt。

`pgrep` 仅在恰好存在 `^m513_fastkill_watcher_tag ` 时才等待。如果 watcher
尚未启动或 tag 漂移，runner 会立即读固定目录；现有检查无法区分 canonical
M513 与 stale/tampered 同名结果。

**最小修复：**在进入 GPU lock 前增加 exact M513 consumer：词法路径与
leaf symlink 检查，exact-four population（JSON、RUN_COMPLETE、member seal、outer seal），
双层 seal 复核，JSON 中所有已封存上游身份与 claim boundary 精确比对；
将实际 member/outer-seal SHA 传入 launch 与 final receipt。

## P0-03：`missing_count=0` 不证明 ep40 含 rank-3 factors

ATLIF 的 `_load_from_state_dict` 对缺少 factor key 的旧 dense checkpoint 支持
migration：从 dense weight 现场生成 factors，随后主动从 `missing_keys` 中移除
left/right key。只有 checkpoint 真含 factor pair 时，
`temporal_factor_load_source` 才变成 `checkpoint_factors`。

M516 最终只要求 `missing_count=0`/`unexpected_count=0` 和总 ATLIF=105，便写出
`rank3_training_complete=true`。这不能排除 ep40 缺 factor key 而在 valid825 时被
dense migration 补齐的情形，也没有证明应有的 45 个 T10 rank-3 modules 从
checkpoint factors 加载、60 个 T2 modules 保持 dense fallback。

**最小修复：**在 valid825 前对 ep40 加一个 pinned final-checkpoint verifier：

1. 直接检查 `model_state_dict` 的 45 对 left/right factor key，无缺失、无单边、shape 精确；
2. 加载后统计 T10 rank-3=45、T2 dense fallback=60；
3. 要求所有 T10 的 `temporal_factor_load_source == "checkpoint_factors"`；
4. 将 checkpoint key census 和 verifier SHA/receipt 封入 final receipt。

若要把“training complete”解释为 factors 确实受到优化，还应检查 optimizer
step/对应参数状态，或与 source balanced-SVD initialization 作数值差分；否则只能
宣称“5-epoch run completed and factor checkpoint was emitted”。

## 等待、锁与 GPU 竞态审阅

四个 `flock` 的短路求值与失败后全解锁是成立的，且 fd 会被后续
Python 子进程继承，所以 cooperative job 之间能保持互斥。但仍有两个 P1：

- M513 watcher 未启动时不等待、无 timeout；这是可用性问题，不应用降低身份检查来修。
- A800 lock 只约束合作进程。`nvidia-smi` 观察为空到 Python 创建 CUDA context
  之间，非合作作业可以插入；runner 也没有锁定 host/GPU UUID/model/driver/
  Python environment 身份。

## worktree、config、checkpoint 与路径

已通过的部分：

- HEAD 精确要求 `494593afa0ea81332ca21fcd68fdc9d6b72bbf1a`；
- tracked worktree/index 在开始、launch 前、结束后检查；
- data 和 source checkpoint 必须是指向 original root 固定路径的 symlink；
- source checkpoint、base config、split CSV 和主要 entrypoint 的 SHA 被锁定；
- generated config 文件本身的 SHA 固定，并在 launch 前重新检查。

剩余 P1：`git diff` 不发现任意 untracked shadow/import 文件；数据只锁定
symlink target 和两个 split CSV，没有训练/valid 样本内容 manifest；环境中
`PYTHONPATH`/`PYTHONOPTIMIZE`/`LD_PRELOAD` 等未清理。至少应拒绝代码目录下
非允许 untracked 集合，并锁定 Python/env/GPU 和 DSEC data manifest。

## 训练、postflight、valid825 与发布

epoch offset 36 + 5 epochs 与 checkpoint36--40 约定一致；train nonzero rc 被记入
postflight 后传递，valid825 绑 epoch40，这些都是正向设计。但还有四个
证据工程 P1：

1. 固定 result 和 attempt 使用 `mkdir -p`，没有 canonical one-shot atomic
   election、`EXIT` failure marker 或 quarantine；失败会留下不可重跑且容易被误读的半成品。
2. final receipt 与 completion 未作 member seal + outer seal，也没有 staging
   preverify + atomic publish。
3. `profile["metrics"]` 被直接复制，没有要求必需 key、有限值和排名方向；
   825 条样本不等于一个可用的 AEE/AAE 测量。
4. runner 的实际 SHA 在 start/prelaunch/end 被比较，但 final receipt 只写入
   调用者提供的 expected 字符串，没有封存三个实测值。

建议用 unique same-parent staging 生成 attempt，所有成员通过后写 exact-population
seal 并原子发布 canonical completion；普通失败写非 canonical
`FAILED_DO_NOT_CITE`，已发布验证失败先 quarantine。

## 宣传边界

这部分通过。当前 final receipt 明确将
`accuracy_hardware_admitted` / `cycle_speedup` / `energy` /
`paper_ppa_ready` / `date_headline` 全部设为 `false`，completion 也写
`NOT_HARDWARE_ADMISSION`。它不会把 floating-factor AMP 训练冒充 INT8/QAT、RTL、
周期、能量或 PPA 准入。

修复 P0 并封存 P1 证据链后，合法句子仍只能是：

> Isolated five-epoch rank-3 fine-tuning completed and produced an ep40
> factor checkpoint; standard valid825 metrics were measured. Accuracy and
> all hardware claims remain unadmitted pending the metric gate and the
> separate quantized/hardware chain.

## 准入条件

修复版 runner 只有在下列全部完成后才可进入 A800 队列：

1. 三组 inline `assert` 全部改为显式检查；
2. M513 exact sealed consumer 与上游身份传递闭合；
3. ep40 factor key/shape/load-source/module-population verifier 闭合；
4. 结果全部 finite metric 与 sealed atomic publication 完成；
5. 新 SHA 再做一次静态打铁，P0=0。

`docs/359` SHA 保持
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
