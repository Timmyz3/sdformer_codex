# AT-LIF＋PSN 神经元恢复：独立设计审阅

实施后补记：已核新 `run_recovery.py`，中途 `model.train()`、神经元 forward／backward matmul 禁 TF32、同输入／buffers／RNG 重试均已落实；下文保留实施前审阅。默认 H9 裁剪修复与包装兼容的实际核验见 [DEFAULT_TRAINING_PATCH_REVIEW.md](DEFAULT_TRAINING_PATCH_REVIEW.md)。

2026-09-20。审阅三臂方案及已有 `neuron_repair_20260919` 模块；新训练入口尚未提供，本文件是实施前审阅，不宣称新入口已通过验证。未运行 GPU、未修改生产代码。**保留图、冻结骨干、统一 256 次成功更新和完整 valid825 配对，足以回答本轮“神经元修复是否改善结果”。已有 helper 存在两处必须明确处理的接口：中途评测会留下 eval 模式，FP32 wrapper 不会禁用 TF32。**

**两处现成代码风险。**

1. `recovery_probe.evaluate()` 开头调用 `model.eval()`，返回前只恢复 BN buffers／track 标志，**不恢复 training 标志**。十帧中途诊断后若直接续训，就会关闭 dropout，并可能使用恢复后的 running BN，而非计划的动态 batch 统计。新入口应在诊断结束立即 `model.train()`，在下一训练 forward 前核 BN.training=true；也可在入口自己的评测包装中保存／恢复原模式。不需要改既有生产 helper。
2. `neuron_precision.install()` 只禁 autocast 并将张量转 float。当前配置 `runtime.allow_tf32=true`，上游训练入口据此启用 `torch.backends.cuda.matmul.allow_tf32` 和 `float32_matmul_precision("high")`；因此 PSN 的 addmm **仍被允许使用 TF32**。这不证明每种小矩阵实际选用了 TF32 kernel，但足以否定“禁 AMP 就必然是完整 FP32 乘法”的口径。若本轮要去掉神经元乘法精度障碍，应在新进程固定完整 FP32 matmul 策略，Conv 的 AMP 可继续保留；三臂及四模型最终评测使用同一策略，并记录实际 backend flags。原 ep34 必须在该共同环境重评，不能直接搬旧 AEE。

**256 次成功更新要实现为同有效数据，而非只对齐数字。** 预选 256 个 batch，以成功步编号生成 augmentation／dropout RNG。每个 batch 先生成一次增强后的输入；重试时保留相同输入、GT、mask、随机种子，清梯度、reset_net 后重新 forward/backward，GradScaler 退 scale 后继续同 batch，成功才进入下一条。不能失败后消费新 batch，也不能让重试消耗新的数据增强。记录成功步、尝试号、batch IDs、scale、skip、未裁剪范数／clip 系数；checkpoint 同时保存成功数和尝试数。三臂初始 scaler／optimizer 状态一致，但因各自真实溢出产生的后续 scale 差异应如实记录。若做阈值 clamp，放在成功更新之后。

动态 BN 的 running buffers 在失败 forward 中仍可能变化；batch-stat 训练与 no_running 评测通常不读取这些值，因此不应将此变化误判为骨干参数被训练。但为重试状态清楚，可在重试时恢复该 batch 第一次尝试前的 BN buffers，或统一采用明确的不累计 running 统计合同。η=0 时仍要清掉神经元 `update_value/act_value`，避免实验侧统计混入重试。中途评测会改 RNG，下一成功步仍应按预定种子重置。

**训练范围须按参数身份精确锁定。** 先获得 81 个实际活跃 ATLIFTernaryPSN 名单；当前 dense PSN 合同下，可训练集合恰为每个模块的 `weight/bias/thresh`，其余参数均 `requires_grad=False`，新建 optimizer 只收此集合。冻结骨干时不得用整段 `no_grad` 或 detach 中间特征：梯度仍须穿过固定 Conv／BN／attention，到达更早的 PSN。也不要为冻结 BN 仿射而调用 BN.eval()；γ/β 冻结与使用 batch 统计是两件事。以训练前后逐位比较验证非目标参数未动，BN running buffers 单列。

分组沿用已修复的规则：PSN weight 用 neuron_lr／既定 weight decay，时间 bias（虽为 T×1）用同 neuron_lr、wd=0，θ 用共同 threshold_lr、wd=0、ε=1e−12；其余 PSN Adam ε 保持共同原值。只看 θ 变化不足以判断修复：同时记录 PSN weight/bias 更新及 12 个 sn_q 的梯度。81 个前向活模块不等于每臂 81 个非零梯度模块；无 bridge 的 Q 支路仍可能断连。参数漂移还须区分任务梯度与 weight decay，不能从“有更新”反推 Q 支路已接通。

**三臂可解释的差异。**

| 对照 | 允许解释 | 不应宣称 |
|---|---|---|
| official_eps → normalized | 相同模型／精度／optimizer 合同下，将官方 θ 规则替换成 normalized θ 代理的总效果 | 唯一数学正确导数；仅补一个幅值项的净效果。该替换同时改变核形式和 mean→sum。 |
| normalized → normalized_q | 在相同 normalized θ 规则上增加 Q support 梯度通路的效果 | 只改变 12 个 θ；新增 Q 梯度也会沿固定算子流向上游可训练 PSN。 |

现有 `binary_surrogate` 保存原 gate、保留 legacy h 梯度；共用 FP32 神经元后，该候选无需再偷偷改输入梯度。Q bridge 当前只给 Q 增加 support 多项式代理，原 score／K 的局部 VJP 保留；它适用于当前非负 binary、head_dim 归一化、penalty=0、K-mag=0 的合同。其系数已改成 where，前次低精度 `(1+α)−α` 抵消问题已关闭。已有两个真实 batch 的记录显示输出四个 head 差异为 0、12 次 score 调用、Q 非零 θ 梯度由 0→12、全部有限；这支持连接性，不能代替本轮训练效果。

各臂须重新加载同一 ep34 参数及 buffers，并恢复原 `.act` 和原 score 函数；`load_state_dict` 不会撤销 monkeypatch。FP32 wrapper 安装一次，Q bridge 只在 normalized_q 生效。共同第一 batch 的初始输出／loss 应一致；若不一致，先解决实现差异，不进入 256 步比较。最终推理可对四个模型统一移除训练用 bridge，因为它不改变 forward。

**batch／layout 与完整验证边界。** 三臂固定同一 batch size；若采用 batch=1，可避开旧 T2 reshape 已确认的跨样本组合，但仍不修正其特殊时间／空间排列。不在本轮改变分窗／逆变换，不把 Q bridge 称为真实时间因果性的修复。

最终固定使用成功第 256 步终点，不根据十帧中途 AEE 换终点或改各臂预算。valid825 应逐一枚举相同 825 个唯一文件，四模型共用 batch=1、no_running BN、AMP／TF32 策略、flow head／缩放／mask／输入归一化；每帧保存原 ep34 与三个终点的 AEE、有效像素数，再算配对差值。主指标固定 frame-mean AEE，pixel-weighted 可作为另列指标，不能混用。当前 helper 标注 native `flow[-1]`；内部四模型配对成立，不自动保证与另一历史评测脚本的绝对数值等价。十帧结果仅诊断，不作为 full825 改善的替代证据。

在以上合同落实后，无需为本轮设计另加实验臂或长训。结论直接以同环境 ep34／official_eps／normalized／normalized_q 的完整配对结果和实际更新记录作判断。
