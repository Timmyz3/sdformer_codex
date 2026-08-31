# M652｜M649 ConvTranspose typed numeric audit fresh static hammer

## 裁决

`NEEDS_REVISION__GPU_NOT_AUTHORIZED`，91/100；P0=0、P1=2、P2=2。

本裁决只覆盖 launcher `32e347ff...`、contract `a846d84a...`、tests `10084664...`。这三个旧身份不得运行合同中的 GPU 命令。数值归约、first-two channel 语义和只读 M511 边界本身没有发现 P0；阻塞项在路径事务的 fail-closed 实现。

## 已独立通过的部分

- 23/23 个 M649 输入 SHA 全部重算一致；冻结 M511 producer/contract 分别仍是 `e16a454d...` / `e556743d...`，`docs/359` 仍是 `dedde7ce...`。
- M650 双 seal、M649 作者交接双 seal、M511 consumed attempt 双 seal均独立复核通过。M511 canonical 和 M649 canonical 均不存在；失败 staging 仍只有 `FAILED.json` 与 d0 partial bitpack 两个冻结成员，SHA/size 无漂移。
- `/opt/anaconda3/envs/pytorch310/bin/python3.10` 的 11/11 CPU tests 通过；`/usr/bin/python3.6 -m py_compile` 对 launcher/tests 通过。没有运行 GPU、模型 forward、EDA 或远端任务。
- `reduce_counts` 的 reduction dims `(0,1,3,4)` 对 `T_B_C_H_W` 是正确的；定向的 `T=2,B=1,C=3,H=1,W=2` 尾块测试逐 channel 得到正确 elements/zero/one 计数。
- d0 全 binary、d1-d3 first2 finite analog + suffix binary、last2 仅 diagnostic、suffix nonbinary、nonfinite、dtype 和 population 的正负测试均符合合同。JSON 采用 `allow_nan=False`，strict parser 拒绝 NaN/Inf token 和 duplicate key。
- launcher 没有调用 `m511.main`、`stream_binary_input`、`torch.save` 或 `np.save`；只输出逐 channel 计数、4 个候选 analog channel 的有限值统计、result/receipt/seal/failure metadata，不输出 activation payload。
- staging 采用同父目录 `mkdtemp`，完整 result 后写双 seal、原子 `os.replace`，post-publish 自验失败会尝试移入唯一 quarantine；双 seal 能拒绝新增 suffix 文件。

## P1 阻塞项

### P1-1｜canonical output symlink 可绕过

`main()` 先执行 `output = args.output_dir.resolve()`，再调用 `reject_symlink_chain(output, allow_missing_leaf=True)`。因此原始 canonical leaf 若是指向缺失 target 的 dangling symlink，`resolve()` 会先抹掉 symlink 证据；检查看到的只是允许缺失的 target。定向攻击已复现 `reject_symlink_chain(link.resolve(), allow_missing_leaf=True)` 成功返回。

后果是 `os.replace(staging, output)` 发布到 symlink target，而不是合同指定的 canonical 目录，canonical 本身仍是 symlink。测试 `test_symlink_output_path_is_rejected` 只直接调用 helper，没有覆盖 `main()` 的 resolve-before-reject 路径，因而产生假阴性。

修复门：对未经 `resolve()` 的原始 `args.output_dir` 和合同 canonical 路径先逐 component 拒绝 symlink/dangling symlink，再 resolve 并做路径相等；补一个复刻 main 顺序的 dangling-leaf 回归。

### P1-2｜输入与失败 M511 staging 的 exact-path 边界可被 symlink alias 绕过

`verify_contract_inputs()` 同样先 `(ROOT / entry["path"]).resolve()`，随后才检查 `not path.is_symlink()`。定向攻击把 launcher contract path 改成指向 exact launcher 的 symlink alias 后，23-input 验证仍通过。`verify_failed_m511_state()` 对 consumed attempt 会经 `verify_double_seal()` 拒绝路径 symlink，但对 failed staging 本身没有 `reject_symlink_chain()`，其两个成员也可经 symlink/alias 被跟随。

当前盘上 23 个输入和 failed staging 都是 regular file/directory，因此这不是已发生污染；但代码声称的 exact path / preserved failure scene 并未 fail-closed，fresh review 不能授权该旧身份。

修复门：所有 contract input 原始路径在 resolve 前逐 component 拒绝 symlink；failed attempt、failed staging、forbidden M511 canonical 的原始路径链分别检查；failed staging 的目录和两个 leaf 都必须是非 symlink regular objects；补 input alias、staging alias、member alias 三类回归。

## P2 边界

1. 结果没有记录 Python executable SHA、PyTorch/NumPy/SpikingJelly/CUDA/GPU identity。它不改变本次 exact `{0,1}` 诊断的静态语义，但 post-result receipt 应补环境身份，避免把换环境的 model forward 当同一测量。
2. `typed_split_decision()` 自身只检查总数和 per-module 数量，不独立检查 `(sample_id,module_index)` 40 格唯一且有序。当前 main 的 hook state machine 保证该性质，所以不阻塞本次执行；独立结果 verifier 必须显式重验 10x4 lattice，不能只信 decision Boolean。

## 唯一放行条件

作者必须生成新的 launcher/contract/tests SHA，并让 fresh hammer 复现：原始 output/input/failed-state symlink 攻击全部 fail closed，11/11 或更多定向测试通过，23 inputs 和失败 M511 状态未变，P0/P1=0。达到之前，不授权本合同唯一候选 GPU 命令；不授权 M511 重跑、payload、cycle、speedup、RTL、EDA、PPA 或 DATE headline。

