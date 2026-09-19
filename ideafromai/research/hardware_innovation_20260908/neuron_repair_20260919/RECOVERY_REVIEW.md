# 二值恢复探针独立审阅（2026-09-19）

只读审阅 `recovery_probe.py`、`diagnose_threshold_step.py`、`short_training_replay.py`、`binary_surrogate.py`，并核对本目录配置、SHORT_REPLAY、算子测试及 `RECOVERY_WHOLE_NETWORK.json`。未运行 GPU、未改实验源码。**未发现起点与恢复后三臂之间的 BN 评测口径切换，也未发现候选修改 input-gradient；发现“64 步”实际是尝试次数，三臂成功更新预算不同。**

**1. 直接影响结论的预算问题：跳步仍消耗 batch，并被计入 checkpoint 的 training_steps。**

`recovery_probe.py:94–110` 每个 batch 仅尝试一次，`skipped=True` 后继续下一 batch；123–124 行仍将 `training_steps=len(starts)` 写成 64。已核落盘结果：

| 规则 | 尝试 / 成功更新 | 跳过的 step（0 起算） | 十帧 AEE | 成功步最大裁剪前范数 |
|---|---:|---:|---:|---:|
| 共同 ep34 起点 | 0 / 0 | — | 1.045044308 | — |
| official | 64 / 63 | 46 | 1.106234916 | 8.8861 |
| margin | 64 / 64 | 无 | 1.146072671 | 9.0371 |
| normalized | 64 / 63 | 46 | 1.134433070 | 8.0702 |

跳步前 scale 均为 65536。因此可以说“同 64 个 batch 尝试预算”，不能说全三臂“同 64 次 optimizer update”。margin 还实际更新了另两臂跳过的 batch；official 与 normalized 这一对的成功预算及有效 batch 仍匹配。单次十帧探针不足以推广 surrogate 排名，但三臂都劣于共同起点，足以说明本轮没有得到精度恢复收益。应修正 checkpoint 元数据／报告口径，不必丢弃原数据。

`short_training_replay.py:57–98` 只在成功时推进 step，会用相同 batch／随机种子重试，属于另一种预算。已有 SHORT_REPLAY 四臂各 4 次成功更新、0 次跳步，此问题不影响其结果。下一轮若严格要求同成功步数／同有效 batch，应重试当前 batch 至成功后再前进，同时单列尝试数；仅降低初始 scale 只能避免已见溢出条件，最终仍须以实际 skip 日志验收预算。

**2. 内部 AEE 配对成立；不等于复刻历史 valid825。**

恢复开始前保存 ep34 全 `state_dict`；baseline 立即评测；每臂重新装回该状态并重建 optimizer/scaler。`evaluate` 对 baseline 和各臂使用同一批预选验证帧、batch=1、无 autocast、相同输入处理、`flow[-1]`／flow scaling／有效像素掩码。它先 `model.eval()`，再同时清除 BN running buffers 并设置 `track_running_stats=False`，实际使用当前 batch 统计，符合本目录配置的 `test.bn_policy=no_running`。评测后恢复原 buffers，各臂开始时又调用 `model.train()`。未发现 baseline 污染下一臂初态的证据；短回放四臂首个训练输出也全部逐元素相同。

因此上述 AEE 可作本探针内配对比较。当前恶化不能归因于“baseline 用另一种 BN”。不能把十帧结果拼接成历史完整 valid825 的比较；本次未重新核对历史评测器的 head／插值／mask，故不认证跨脚本绝对数值等价。

**3. input-gradient 保持 legacy；AMP 下是有意混合的数值规则。**

`binary_surrogate.py:31–34` 用原 saved h、θ dtype 计算 `native_p`，返回 `grad_output * native_p`，与官方表达式一致；显式转回 h dtype 与 autograd 对该输入最终传递的 dtype 一致。候选仅保存原 forward gate，并以 fp32（float64 测试除外）计算 θ 核与 sum，没有将 h 梯度路径提升到 fp32。因此“候选偷偷改变输入梯度”不能解释最初一步的 backbone 差异。

但须精确称为“normalized θ 规则 + legacy h 梯度”：AMP 下 h 路径的低精度核与 θ 路径的 fp32 核可能不同，不能要求逐位满足单个全 fp32 STE 图的尺度恒等式。float64 解析测试验证公式，低精度 VJP 对比验证保留 legacy 路径，两者是不同验收项。后续 θ 改变前向与 backbone 梯度是全网联合恢复的正常因果路径，因此全网 AEE 仍不能单独归因于固定 backbone 下 θ 的净效果。

**4. 未发现其他足以否定本轮内部对照的实现问题。**

三臂新建 AdamW／GradScaler，同一步使用相同训练样本和随机种子，均采用 `unscale_ → clip → step`。配置为 num_acc=1、η=0，64 次范围也未到 threshold freeze 点；未调用原训练循环的累积／手动 η 更新，不构成本轮臂间差异。诊断脚本仅做共享真实 backward 下的 fresh-optimizer 首步反事实，不证明连续训练结果。修正 clip 顺序会影响所有参数，官方新回路较起点变差不能单独归咎于 θ 更新。

候选 θ 梯度理论上也能通过全局裁剪改变 backbone 的更新，但本轮所有成功步 preclip norm 均低于 10，而 clip=100，故已排除此裁剪耦合解释。短回放 fixed 三臂的范数也仅为 0.94–3.49，无需以假设解释已有观察。

**下一轮 θ-only 最小判据。** 从同一 ep34 出发，冻结所有非 θ 参数并重建 optimizer，保持共同 train 模式、BN batch 统计和随机种子，不为某一臂单独切到 eval。逐臂核非 θ 参数逐位不变、实际成功更新／有效 batch 对齐，并复用同一个 no_running evaluator。若采用 `official_eps12`，它同时改变官方优化器 ε，应明确为“官方规则 + ε=1e−12”的实验臂，不能直接冒充本轮默认 ε 的官方延续。这样可分开解释固定 backbone 下的 θ 更新与全网协同恢复；验收继续看 AEE，不以 θ 变化数量代替精度。

**补充：Q-only support bridge 审阅。** 已读 `q_support_proxy.py`、`q_gradient_probe.py`、`test_q_support_proxy.py`，并复核原 `_ternary_alpha_xnor_token_scores`、Q token 重排、score 归一化与当前 h60 消费路径。以下是源码结论，尚未用真实 batch 结果替代验证。

| 检查点 | 当前代码结论 |
|---|---|
| 是否重复计算 Q 梯度 | 当前 TX 的匹配／共静默／异号项全为布尔比较，motion 项只在数值上依赖 K；single-active penalty 和 K-magnitude 都为 0，因此原 score 没有可微 Q 数值路径。桥接器向原 score 返回原上游梯度，另给 Q 一次 `a × [k_bit − α(1−k_bit)] × score_scale/head_dim`，此配置下没有双计。 |
| 是否改 K 局部梯度 | `backward` 对 original_score 返回原 `grad_score`，对额外 k_token 参数返回 None；原 K／motion 支路的局部 VJP 保持。新增 Q 梯度仍可能通过共享前驱或后续层改变其他 K 参数的全网梯度，不能扩展为“全网所有 K 梯度完全不变”。 |
| 是否前向原样返回 | `forward` 直接 `return original_score`，没有 `score + surrogate − detach` 的浮点抵消误差。原 score 只计算一次；Q 的 permute/reshape 仅连接新增梯度。当前消费者创建新 scores 后再中心化，没有发现对此 alias 做原地写入的冲突。 |
| 适用范围 | 正 θ 的非负二电平 Q/K、head_dim 或 dim 固定归一化、single-active penalty=0、K-magnitude=0；归一化因子与原函数一致。该函数不是完整 QK 多项式的双侧梯度：只加入 Q 的 unit-slope support STE，明确保留原 K 规则。 |

安装器目前检查归一化／penalty／K-mag，但自身未强制 binary、正 θ 或 h60 模式；当前网络探针会检查 binary，且使用正 θ 的 ep34，故不是本轮阻断项。若以后换成负极性／三元，`k>0` 会把负脉冲当静默；若开启单边 penalty、K-mag、非固定 active 归一化或替换原 TX backward，当前代理解释及是否双计须重新审定，不能直接复用此结论。其他独立 Q 分支（例如非零 SC 混合）本可合法贡献梯度，但那时不能再宣称全 Q 原先断连。

合成测试支持四种 dtype 下 forward 原值、原 K 支路 VJP 和实现所定义的 Q 公式；它没有覆盖完整 wrapper 的 token 重排、真实网络 score 调用或 12 个 Q 神经元。且 expected-Q 使用同一浮点表达式：fp16 的 `(1+α)−α` 在 α=.02 时可能为 0.9995117，因此“unit slope”是代理的实数定义，不是所有低精度活动位都逐位等于 1 的承诺；这不影响前向，当前实际 score dtype 是否命中该情况应按记录判断。

真实 batch 探针两臂均用 normalized θ 规则，复制 BN buffers、同数据及随机种子，只切换 bridge，没有 optimizer step，能验证桥接的前向与连接性。验收应同时要求所有输出 head 差异为 0、score_calls 命中预期、`all_finite=true`，再报告 12 个 sn_q 的梯度；不能先数非零值而忽略 NaN/Inf。即使全部通过，也只说明新的训练规则建立了 Q 支持位的代理梯度，既不是硬阶跃真导数，也不是 AEE 已改善或训练稳定性的证据。

**主代理后续处理与收据（审阅之后）：** 已将 Q 系数改为活动 K 直接取1、静默 K 取−α，消除上述低精度 `(1+α)−α` 问题；测试改用独立 FP64 分支参考及 dtype 容差，另要求活动分支精确。`Q_NETWORK.json` 两个真实 batch 均满足四个 head 差异0、12次调用、全部梯度有限，Q θ 非零梯度0→12，全图69→81。`RECOVERY_THETA_ONLY.json` 三臂均完成64次成功更新，无skip；仅ε修改臂的非θ参数和非BN buffers全等将在完整配对验证入口直接检查。未因此认定训练后的AEE改善。
