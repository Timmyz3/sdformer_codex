# H9 默认 AMP 裁剪、跳步修复与包装兼容审阅

2026-09-20，独立只读审阅；未运行 GPU，未编辑共享入口或 third_party。审阅期间根代理已落实以下最小补丁，本文记录实际文件状态。**默认入口现已修复 scaled-gradient clipping、逐 microbatch clipping，以及 AMP 跳步时仍执行／夹带手工阈值更新的问题；当前 candidate 包装不会重复 unscale。** 这项训练循环修复不等同于证明某个神经元代理已经改善 AEE。

## 精确落点与顺序

上游 `SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py` 原本在每次 backward 后、累积更新边界之前裁剪；AMP 时裁剪的是 scaled gradients。只在 `SCALER_STEP_PATCH` 加 unscale 仍然太晚，不能修复已经发生的裁剪。

现有最小实现位于 `SDformer/neuron_experiments/H9_bipolar_self_attention/entrypoints/train.py:578` 的 `_patch_amp_clipping`，并由 `_patch_source` 在全部原有 source 替换完成后调用（第 619 行）：删除精确匹配的旧外层 clip 块，在原 `((sample + 1) % num_acc_steps == 0) or ... == len(train_dataloader)` 边界内插入：

```python
# [H9 AMP] Unscale before clipping at the optimizer boundary.
if scaler is not None:
    scaler.unscale_(optimizer)
if config["loss"]["clip_grad"] is not None:
    torch.nn.utils.clip_grad_norm_(model.parameters(), config["loss"]["clip_grad"])
```

因此每个成功或溢出更新尝试只有一次 `unscale → clip → scaler.step → scaler.update`；未到累积边界时既不 unscale，也不 clip。非 AMP 同样在边界裁剪一次；`clip_grad=None` 时允许显式 unscale 后直接 step。保留原尾批更新条件，不改变尾批 loss 的既有归一化定义。

`SCALER_STEP_PATCH` 现先保存 scale，执行 `scaler.step → scaler.update`，仅在新 scale 大于或等于旧 scale 时调用原 `threshold_update`。使用 `>=` 正确保留了 scale 增长的成功更新。溢出时遍历全部 ATLIF 模块，将 `update_value` 与 `act_value` 设为 0.0，记录 `amp_skipped=1`，随后沿用边界末尾的 `optimizer.zero_grad()`。`update_value` 是跨 forward 累加的待用手工增量；清零阻止失败累积组夹带至下一成功组。`act_value` 是覆盖式正则项，清零释放失败组的旧图。未改 forward、架构、非 AMP 更新分支，也未用 `threshold_update(lr=0)` 代替清理，避免其额外 clamp／反馈副作用。

旧 `SCALER_STEP_ANCHOR`、`OPTIMIZER_STEP_ANCHOR` 和 epoch-loop anchor 文本保持不变，既有 warmup、threshold hook、诊断注入点继续匹配。整个 `_patch_source` 本来就接受原始上游源码，不应重复调用于已 patch 的源码；无需将整个 source patch 系统改为幂等。

## 包装兼容

| 入口／包装 | 实际影响与兼容要求 |
|---|---|
| `neuron_repair_20260919/train_binary_candidate.py:12` | 旧版本必须找到一次旧 clip，默认入口修复后会 assert 失败。当前第 18 行已经识别 H9 marker，确认旧 clip 不在且 unscale 恰一次后跳过自己的 clip 改写，继续注入 θ 规则／ε／FP32。该同步修改是必要部分，不能只放宽旧 assert 后继续插入，否则 GradScaler 会报重复 unscale。 |
| 同一包装的旧 H9 fallback | 无 marker 时仍要求旧 clip 和 boundary 各一次，按旧方式修复一次。当前 main 只调用一次，兼容成立；无需为了不存在的多次套用增加新机制。 |
| `diagnose_threshold_step.py:247–252` 及基于它的 recovery／pure trial | `_patch_source` 后在原 epoch loop 之前插入 probe 并 return；默认循环中的新 unscale 不会执行。这些实验自有循环中的 unscale 不会因此重复。 |
| `short_training_replay.py` | 自有 old/fixed 两种更新序列。历史 old 分支应保留为错误序列对照，不能因默认修复而改成 fixed。 |
| `claude_fusion_trials_20260914/t57_gradfix_train.py`、`t57_frozenbn_train.py` | 最终调用 H9 main，自动继承新默认裁剪，无自己的旧 clip 文本重写。今后重跑会使用新训练循环，历史结果不能标为相同训练实现。 |

## 已核验证据

本次在不导入 torch 的 CPU 静态检查中提取实际 H9 patch 函数和常量，patch 完整上游 source 并编译；AST 确认唯一累积边界内恰一次 unscale、一次 clip，且都先于 step。再组合 candidate 的 `official/margin/normalized` 三规则，开启 θ ε 和 FP32 注入；三者均编译通过，仍各只有一次 unscale／clip。

根代理的真实 CPU GradScaler 数值测试与结果分别在 [test_default_amp_clip.py](test_default_amp_clip.py) 和 [DEFAULT_AMP_CLIP_TESTS.json](DEFAULT_AMP_CLIP_TESTS.json)：AMP 累积 1／2／3、非 AMP 累积 2、clip=None 共五例，修复版均与标准参考零误差；触发旧问题的四例旧序列误差为 0.02717–0.10725。该测试包含五个 microbatch，因此累积 2／3 均覆盖不足一组的尾批。无需重复占用 GPU 验证这个循环顺序。

补充复审已核新增 `check_manual_update_skip()` 确实直接执行当前 `h9.SCALER_STEP_PATCH`，使用真实 CPU GradScaler／optimizer，仅替换外部 warmup、冻结和手工更新回调。成功／溢出／成功三次中，scale 为 128→256、256→128、128→256；手工回调仅收到 pending 2 和 5，失败组 pending 3 被丢弃。测试断言溢出不改权重、不增加 θ、清空 `update_value/act_value`；最终 θ 从 1→3→3→8，结果文件全部通过。该证据验证实际入口的门控和清理控制流；未将 stub 手工回调误称为全部可选阈值策略的数值验证。

## 当前隔离恢复与边界

已核 `run_recovery.py`：神经元 forward 在关闭 autocast 与 matmul TF32 的上下文执行（41–53 行）；backward 期间关闭 matmul TF32（122–127 行）；同 batch 重试恢复 buffers、reset 网络和 torch RNG（102–117 行）；只有成功才推进 batch，256 次成功更新而非固定尝试数；中途 evaluate 后恢复 `model.train()`（154–157 行）。这关闭了 DESIGN_REVIEW 实施前提出的两项接口风险。

非神经元 forward 的 matmul TF32 与 cudnn TF32 仍按共同配置保留并记录；“FP32 神经元”成立，“整网所有乘法都禁 TF32”不是本轮合同。四模型完整验证共用同一输入和环境，源 ep34 也在该环境重评。

此前“默认 threshold_update 未按 AMP skip 门控”的问题现已关闭。默认入口在跳步后丢弃该累积组并继续读取后续数据，仍不具备隔离恢复入口的同 batch 重试；可选 quantile／importance EMA 属历史观测，不在本次 pending 清理范围，也不应直接归零。因此可以明确称为“默认裁剪顺序与手工阈值跳步修复”，不能称为失败尝试的全部状态回滚。当前隔离恢复 η=0 继续沿用其独立成功更新／重试合同。
