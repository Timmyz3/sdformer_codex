# 纯 AT-LIF 隔离实验代码审阅

**执行后补记（root，2026-09-20）：** 下文审阅发现的 PSN bias decay 已在恢复启动前修正为 `psn_no_decay`，最终 [RECOVERY64.json](RECOVERY64.json) 的 optimizer 组确认 neuron_lr=1.25e−5、wd=0。原PSN64次成功，纯递推60次成功（64尝试，skip索引0/1/3/9），结果未称同64成功预算。下文保留初次审阅意见，便于理解改动原因。

2026-09-19。只读 `pure_atlif.py`、`run_trial.py`、`test_recurrence.py` 及相关 precision／optimizer／历史官方副本；未运行 GPU、未修改实验源码。**神经元实现是真正的无 PSN 递推，官方 hard-reset 与 BPTT 路径匹配；恢复前应修正控制臂 PSN bias 的 weight decay 分组。**

**须修：手写 optimizer 意外改变控制臂 PSN bias 的更新规则。** 审阅时 `run_trial.py` 把全部旧 PSN 非 θ 参数划入 `psn`，统一设置 `wd=config.optimizer.wd=0.001`。原 H9 `h28_optimizer.py` 对 `.bias` 使用 `atlif_neuron_no_decay`，其学习率为 neuron_lr、weight decay 为 norm_wd（本配置为 0）。PSN 的时间 bias 虽是 `[T,1]` 二维参数，名字仍以 `.bias` 结尾，应走免衰减组。纯递推臂已删此 bias，而 baseline／T10-only 的保留 PSN 部分被额外改变，形成可避免的控制臂偏置。建议仅增加 `psn_no_decay` 桶：先识别 θ；其余 PSN 参数按 `.bias`／norm 判断分到 neuron_lr 的 decay 或 no-decay 桶；不要直接并入使用 norm_lr 的普通 norm_bias。改动应在 baseline 与 all_tau1 恢复开始前完成，并将最终组别写入结果。此项不影响已开始的零样本评测。

**预算应按已有日志验收，不冒称 64 次成功更新。** 循环按 64 个预选 batch 各尝试一次，AMP skip 后直接推进；现已单列 `successful_updates`，比旧探针元数据清楚。若两臂都无 skip，则成功预算和有效 batch 对齐；若不同，报告同 64 次尝试及具体跳过的 batch。确需固定 64 次成功更新时，在同一个 batch／种子上降低 scale 后重试，并另记尝试数。无需为了修文案重跑已完成的零样本结果。

**其余核心检查通过。**

- **确实去 PSN：** `RecurrentATLIF` 只有 `thresh` 参数，没有 T×T weight、时间 bias 或隐藏线性变换；新模块替换原 `.spiking_neuron`，原网络 Conv／Linear／BN／attention 保留。`all` 去掉所有被发现的 ATLIFTernaryPSN；`t10` 只替换 T=10，因此该臂不能称全网无 PSN。保留旧 Python 对象用于各臂恢复不等于它们仍进入模型 forward 或 optimizer。
- **reset／梯度准确：** 每次 forward 以 mem=0 起步，逐 t 做 `tau·mem+x[t]`、官方 θ 幅值发放，再做 `(1−spike/θ.detach())·mem`；只 detach 分母，没有 detach spike／mem。候选采用 H9 OfficialATLIFSurrogate；其实现与测试所用 E2 官方副本一致。已落盘四组 T∈{2,10}、tau∈{0.9,1} 的输出、input-gradient、θ-gradient 均逐位一致；重复调用无状态残留，未来输入不影响已产生输出，均只针对所提供的张量首轴。
- **共同初态与精度合同：** 各臂重新挂回原模块并加载同一 CPU state 快照；BN buffers 也包含在快照内。baseline 的旧神经元 forward 被显式包成 FP32，新递推同样禁用 neuron 内 autocast，Conv／Linear 仍在共同 AMP 环境中。这是新的明确 FP32-neuron 对照，不能接到旧原生 AMP 控制臂假装只改 PSN。当前零样本 evaluator 本来关闭 AMP，因此该包装没有额外改变 baseline 推理算术。
- **恢复的共同干预已显式化：** 两臂共享 backbone／norm 学习率、θ eps=1e−12、相同样本和种子、正确 unscale→clip 顺序、η=0、阈值范围与 evaluator。θ eps 改动不应被隐去。官方递推每个 tick 调一次 surrogate，其 θ 的逐调用 mean 反传会累加；旧 PSN 一次处理全 T 再 mean，二者 θ 归约尺度天然不同。这来自官方递推定义，不能为了“对齐”静默除以 T，也不能把相同 θ LR 解释为相同有效梯度尺度。

**legacy T2 的界限保持正确。** 代码没有改变 `window_partition_v2`，文案也注明 T2 不保证 chronology。已有 CPU 索引复现显示该布局甚至可让首轴跨 batch 样本；它是旧网络行为，不由本次替换引入。all_tau1／all_tau09 应称“在既有张量布局上换官方递推”，不能据其结果单独否定真实时间上的 AT-LIF。T10-only 更便于隔离外部 T10 神经元替换，但仍保留原 T2 PSN／布局。此次不应同时修改分窗或逆变换。

**建议本轮只做两个修补／验收动作：** 恢复前修 PSN bias no-decay 分组；恢复后并列报告成功更新数及跳步索引。其余实现足以支持当前四臂零样本和 baseline 对 all_tau1 的有界恢复，不需要为本次代码审查额外发起长训。
