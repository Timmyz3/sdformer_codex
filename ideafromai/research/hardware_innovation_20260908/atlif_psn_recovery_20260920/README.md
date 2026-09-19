# 修复现有 AT-LIF＋PSN：神经元参数恢复

用户于2026-09-20明确先修现有路线。去PSN／递推AT-LIF试验暂停，回到原始 Motion C12 ep34；不改架构、T2布局、动态BN、二电平输出或硬件树。

这次回答的是：在训练回路和数值精度明确之后，让 PSN 时间矩阵、偏置与 θ 一起学习，修订 θ／Q 梯度是否有实际精度收益？不是再以阈值变化个数代替精度，也不沿用上轮纯AT-LIF的权重。

## 共同设置

- 起点：只读 incoming 的原ep34。各臂恢复同一参数和buffer快照。
- 可训练：81个活 AT-LIF＋PSN 的 `weight/bias/thresh`，共5247个标量。骨干及BN仿射冻结，梯度仍穿过这些固定算子传给前面神经元；没有用`no_grad`截断骨干。
- 保留原T10/T2、Motion-XOR α=0.125、θ二电平和现有数据布局。当前动态BN训练batch2，评测batch1/no_running；没有冻结BN统计。
- 共同256次**成功**更新，同一256批、增强、dropout种子。溢出后退loss scale，恢复该批buffer与RNG，重试原batch；成功才消费下一批。
- 训练次序：`unscale → clip → step`。PSN weight学习率1.25e−5、wd0.001；time bias相同学习率、wd0；θ学习率1.25e−6、wd0、Adam ε1e−12；η=0。
- 卷积／线性保留AMP。PSN forward使用FP32且临时关闭matmul TF32；backward阶段的FP32 matmul也关闭TF32。非神经元forward及cuDNN设置另行记录，不把“FP32张量”混同于所有运算均禁TF32。
- 每64步用固定十帧观察，不选中途checkpoint、不据此修改各臂预算。最终统一第256次成功更新终点。

## 三臂

| 臂 | θ规则 | Q support反向 |
|---|---|---|
| `official_eps` | 原官方 `−mean(a p²)`，共同θ ε1e−12 | 原路径 |
| `normalized` | `sum[a(g−(h/θ)p)]`，保留原h梯度 | 原路径 |
| `normalized_q` | 同normalized | 原score前向保持不变，补Q-only代理梯度 |

其中`g=1[h≥θ]`、`p=max(0,1−|h/θ−1|)`、`a=∂L/∂y`。normalized是明确的候选代理规则；它相对官方同时改变形式和mean→sum，不能只归因为“补了幅值项”。Q桥也会经固定骨干影响更早的可训练PSN，不能说只改12个θ。

复用[已验证的梯度实现](../neuron_repair_20260919/README.md)。[独立设计审阅](DESIGN_REVIEW.md)指出的中途评测train模式恢复、TF32精度、PSN bias免衰减、同batch重试和逐臂撤销monkeypatch均已在新入口落实。

## 完整验证

训练后固定比较原ep34及三个终点的全部valid825。每帧只解码一次，四模型都用batch1、同输入／mask／flow[-1]、共同数值策略；不混批以免改变动态BN归约。切换5247个神经元参数，验证其他参数／非BN buffer逐项不变，BN running buffers在评测中不用。

主指标是帧均值AEE，另列有效像素加权AEE；保留逐帧配对值。原ep34会在本次数值策略下重评，不直接拿历史1.199514或前轮1.198610当对照。

同时记录81个活神经元的全张量元素加权发放和逐层非零帧数，检查阈值／Q修订是否改变真实发放，而不把参数变化数量当作神经元恢复的唯一证据。

## 默认训练入口也已修复

本地及A800的 `H9_bipolar_self_attention/entrypoints/train.py` 已通过最小源码补丁修复两处更新顺序问题，未修改third_party：

1. 累积结束才 `unscale → clip → step`，不在scaled gradient或累积尚未完成时裁剪。
2. AMP overflow跳步时不执行手工阈值更新，清除该累积组的`update_value/act_value`，避免下次成功时夹带；loss scale增长仍算成功。

旧 `train_binary_candidate.py` 包装同步识别默认修复，避免二次unscale。[数值测试](DEFAULT_AMP_CLIP_TESTS.json)使用真实CPU GradScaler，AMP累积1/2/3、非AMP累积2、clip关闭五例都与标准参考零差；旧clip顺序在启用裁剪的四例误差0.027–0.107。另实际执行H9 scaler-step补丁，验证成功→overflow→成功时手工更新不夹带。完整H9源码和候选包装编译检查通过，本地与远端均已核实。

审阅见[默认入口复审](DEFAULT_TRAINING_PATCH_REVIEW.md)。默认训练仍采用常规AMP跳过该批的行为；“同批重试到成功”是本轮匹配实验的额外设计，不声称默认入口已经重试。候选normalized/Q桥和严格神经元精度设置仍为隔离选项，尚未因代码修复而自动改成默认算法。

## 文件与运行

- [run_recovery.py](run_recovery.py)：正常H9加载入口上的隔离训练循环。
- [validate_candidates.py](validate_candidates.py)：原模型＋三个终点的完整配对评测。
- 远端：`/root/private_data/work/hardware_innovation_20260908/atlif_psn_recovery_20260920/`。
- 训练输出：远端`neuron256/RECOVERY.json`及各臂新checkpoint。
- 最终输出：远端`neuron256/PAIRED_VALID825.json`，完成后同步本目录。

本轮结果待实验完成后补入；当前不宣称精度改善或修复已全部完成。
