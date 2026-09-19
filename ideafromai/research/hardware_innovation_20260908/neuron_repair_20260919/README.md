# 神经元训练修复：真实回路诊断与隔离恢复（2026-09-19）

用户本轮优先级是先解决神经元问题。冻结身份仍是 **binary AT-LIF `{0, θ}`**；θ 是可学习阈值及发放幅值，推理时在线性消费者处可以折入权重。没有切换三元、没有删除 Q、没有重训覆盖 ep34。

**已确认并修正隔离入口的 AMP 裁剪顺序；已把 θ 不动拆成裁剪、额外平均、Adam ε 和 Q 路径断梯度四个问题。还没有得到可晋级的新精度模型。** 全网短训反而退化，不能用“θ 更新个数增加”作为完成标准。

## 1. 已证实的训练回路问题

H9 继承的训练循环先对 scaled gradients 裁剪，再由 `scaler.step` 内部 unscale。这使实际裁剪上限随 loss scale 缩小。隔离入口 [train_binary_candidate.py](train_binary_candidate.py) 改为：**只在累积完成、即将更新时 `unscale → clip → step`**。上游库与生产 overlay 未改。

两个完整训练 batch（batch=2、480×640、T10、ep34）实际 backward：

| 量 | 首批 | 末批 |
|---|---:|---:|
| loss scale | 65536 | 65536 |
| 未缩放全局梯度范数 | 3.481575 | 1.839466 |
| clip 上限 | 100 | 100 |
| 旧顺序额外压小倍数 | 2281.685× | 1205.513× |
| 正确顺序是否需要裁剪 | 否 | 否 |
| 原 θ 规则首步可见更新数：旧 → 正确 | 0 → 22/81 | 0 → 18/81 |

这是本轮环境下的真实干预证据，不直接外推为旧 T57 所有 1200 步的唯一根因。收据：[amp_probe/DIAGNOSIS.json](amp_probe/DIAGNOSIS.json)。环境 Python 3.12.3 / torch 2.2.2+cu121；不同于先前部分 Python 3.11 产物。

## 2. θ 梯度：公式、归约和优化器不能混成一个改动

令 `g=1[h≥θ]`、`p=max(0,1−|h/θ−1|)`、`a=∂L/∂y`。

| 臂 | θ backward | 本轮角色 |
|---|---|---|
| inherited official | `−mean(a p²)` | 原规则；不是 abs 梯度 |
| single-kernel mean | `−mean(a p)` | 只消去一个核，诊断对照 |
| margin mean | `mean[a(g−p)]` | 改局部规则，保留额外 mean |
| margin sum | `sum[a(g−p)]` | 明确的 margin 代理 |
| normalized sum | `sum[a(g−(h/θ)p)]` | 明确的归一化坐标代理 |

真实单个共享 θ 的使用元素数为 **4,608,000–294,912,000**。因此 mean→sum 本身带来巨大尺度变化；不能把 T57 整体梯度倍率都归给补上幅值项。mean 可以是人工设计的归一化，但不等于广播标量通常的 autograd 求和。

两种 sum 候选在两批首步都使 69/81 个 θ 更新。另一个竞争解释同样成立：**原梯度公式不改，只在正确裁剪下把 θ 参数组 Adam ε 从 1e−8 改为 1e−12，两个 batch 首步也都变成 69/81。** [EPSILON_COUNTERFACTUAL.json](EPSILON_COUNTERFACTUAL.json) 是真实标量 AdamW 首步重放，尚不是该方案全网精度证据。

候选 [binary_surrogate.py](binary_surrogate.py) 保存原 forward 的 bool gate，避免 AMP 标量类型提升后重算比较发生变化。其输入梯度刻意保留 legacy 的 native 精度核，**仅改变 θ 路径**。108 个 PyTorch 解析 VJP 测试通过，最大误差 2.22e−16；四臂真实网络初始输出逐元素一致。测试见 [SURROGATE_TESTS.json](SURROGATE_TESTS.json)。硬阶跃没有唯一“数学正确的代理导数”。

## 3. 实际训练结果：能更新并不等于精度变好

四个真实优化器步、同模型/数据/RNG 的诊断：

| 配置 | 四步后 θ 可见变化数 |
|---|---:|
| 原梯度 + 旧裁剪 | 0/81 |
| 原梯度 + 正确裁剪 | 22/81 |
| margin sum + 正确裁剪 | 69/81 |
| normalized sum + 正确裁剪 | 69/81 |

四臂均无 overflow，见 [amp_probe/SHORT_REPLAY.json](amp_probe/SHORT_REPLAY.json)。这些不是 AEE 结果。

随后三臂各消费 **64 个不同训练 batch**（128 帧），全部正确裁剪，η=0。验证用事先等间距选定的 valid825 中十帧，FP32、batch1、no_running BN、最终 `flow[-1]`。起点 AEE 帧均值 **1.0450443**。

| 全网训练臂 | 成功 optimizer 更新 | 十帧 AEE | 相对起点 |
|---|---:|---:|---:|
| official | 63 | 1.1062349 | +0.0611906 |
| margin sum | 64 | 1.1460727 | +0.1010284 |
| normalized sum | 63 | 1.1344331 | +0.0893888 |

official / normalized 都在 batch46 溢出并跳步；margin 没有。这是**同数据尝试预算**，不是三臂同 64 次成功更新。原 checkpoint metadata 的 `training_steps=64` 表示尝试数；实际成功数以上表及 rows 为准。成功步最大未裁剪范数均 <10，因此也不能把退化解释为候选 θ 范数挤占 100 的裁剪预算。

**裁决：本轮全网短恢复没有精度收益，不晋级任何候选。** 十帧不能代替 valid825，也不作梯度家族的最终排名。收据：[RECOVERY_WHOLE_NETWORK.json](RECOVERY_WHOLE_NETWORK.json)。

## 4. 两个紧接诊断

1. **只训练 θ**：冻结其他参数，用相同 64 批训练数据、三种 θ 更新路线（official+θ专用ε1e−12 / margin / normalized），区分前述退化是否来自全网更新。使用起始 loss scale32768。
2. **Q 代理梯度**：12 个 `.sn_q` 是前向活模块，分数里的布尔比较截断了任务梯度；不是历史 `sn2_q`/`attn_sn` 死结果。新 [q_support_proxy.py](q_support_proxy.py) 保留原 score 值及原 K backward，仅为 Q 加入二值多项式 `qk+α(1−q)(1−k)` 的 unit-slope support STE。四种 dtype 的前向、原 K 局部梯度、Q 解析梯度检查通过，见 [Q_PROXY_TESTS.json](Q_PROXY_TESTS.json)。低精度活动 K 分支直接使用系数1，避免 `(1+α)−α` 的额外舍入。

Q bridge 已完成两个真实训练 batch 的配对验证（θ backward 两边同为 normalized sum）：**四个输出头差异元素均为0；每批命中12个 score 调用；Q 非零 θ 梯度从0/12变成12/12，全图由69/81变成81/81；全部有限。** 两个 batch 的12个 Q 均有非零发放。收据：[Q_NETWORK.json](Q_NETWORK.json)。没有 optimizer 更新，不能把“81个非零梯度”写成“81个 θ 已变化”，更不是 AEE 改善。局部 K backward 不变也不意味着通过共享前驱传播的全网 K 参数梯度一定不变。

这里多项式中的 α 是配置 `alpha0=0.02` 的共静默系数；**不是 Motion-XOR 的 α=0.125**。后者原分数与梯度均保留。不得将候选搬到三元或开启其他 Q 可微分支的配置而沿用本轮结论。

θ-only 已完成，三臂都是 **64 次成功更新、没有溢出**，最终各 69 个 θ 变化：

| θ-only 臂 | 十帧 AEE | ΔAEE | 最大阈值绝对变化 |
|---|---:|---:|---:|
| official + θ ε1e−12 | 1.0384630 | −0.0065813 | 2.4915e−5 |
| margin sum | 1.0468220 | +0.0017777 | 2.9683e−5 |
| normalized sum | 1.0517127 | +0.0066683 | 2.8551e−5 |

收据：[RECOVERY_THETA_ONLY.json](RECOVERY_THETA_ONLY.json)。这说明此前全网退化不能仅归因于“θ 变得可学习”；同时仅十帧、小改变量，不证明 ε1e−12 普遍最优。以改动最少的 official+ε 路线进入**同环境、同输入完整 valid825 配对验证**。所有 θ-only 臂本轮手动 η=0；没有借发放下降代理冒充精度。

**完整配对已完成：825帧 / 18序列 / 48,152,523有效像素。** 非θ参数、非BN buffers直接逐项相等；BN运行统计在两边均不用。结果：

| 指标 | 原 ep34 同环境重评 | official+ε 的 θ-only64 | 差值 |
|---|---:|---:|---:|
| 帧均值 AEE（主口径） | 1.198609554 | 1.201340636 | +0.002731082 |
| 有效像素加权 AEE | 1.147454600 | 1.148699435 | +0.001244835 |

18序列里8改善、10恶化；`zurich_city_09_a` 增加0.035722，是最大恶化项。见 [PAIRED_VALID825.json](PAIRED_VALID825.json)、[序列表](PAIRED_VALID825_SEQUENCES.json)。**十帧改善没有推广到全量，不替换冻结模型；不据此杀掉阈值学习或ε调节家族。** 原模型此次重评1.198610与历史1.199514约差−0.0754%，没有改写历史锚点，候选只与本轮同环境原模型作差。

### 4.1 新发现：FP32 参数移动，FP16 直接比较阈值未移动

两个真实训练 batch 的81个 neuron backward 都观测到 `h=FP16, θ=0-d FP32`。本轮三个 θ-only checkpoint 各69个 FP32 阈值变化，但转换到实际低精度比较值后：**81个全部仍为1.0，变化数0**，见 [THETA_COMPARISON_PRECISION.json](THETA_COMPARISON_PRECISION.json)。因此对固定同一个 FP16 h，这些阈值移动没有改变直接 gate 比较；**不等于整网 gate 不变**，因为 θ 幅值和传播后的 h 仍可变化。

也不能把“所有 θ≡1”写回 checkpoint 身份：本轮原始模型为71个精确1.0，其余10个约0.99988–1.0。这里只是 AMP 比较量化后同为1。

[neuron_precision.py](neuron_precision.py) 提供单独的 FP32 PSN 时间矩阵运算/比较选项，其他卷积保持 AMP。它**有意改变训练 AMP 前向**，与前向不变的 backward 候选分开试；不能只把已经舍入的 h cast 为 FP32，就声称恢复了 PSN 精度。小型时间矩阵/阈值反例里，FP16/BF16 AMP 各有3个输出与 FP32不同，该选项消除反例差异；见 [NEURON_PRECISION_TESTS.json](NEURON_PRECISION_TESTS.json)。不声称它消除前驱卷积的所有 AMP 误差。

**FP32 PSN 的独立短恢复已完成：** 与 θ-only official+ε 臂相同输入、64次成功更新，唯一额外变化为PSN运算保留FP32。初始十帧 AEE仍逐数为1.0450443078784823；终点 **1.0515971729（+0.0065528651）**，69个θ变化，没有溢出。见 [RECOVERY_THETA_FP32_PSN.json](RECOVERY_THETA_FP32_PSN.json)。这证明选项可以在真实网络训练中运行，**未证明其精度收益，也没有对该终点另跑825帧**；不能说PSN数值精度单独解释或解决了全部退化。

## 5. 复现与剩余边界

实验在 A800 独立目录 `/root/private_data/work/hardware_innovation_20260908/neuron_repair_20260919/`。脚本 `diagnose_threshold_step.py` 复用正常 H9 安装、训练损失和优化器，只在进程内替换训练循环。原始 ep34 来自只读 incoming 副本。新 checkpoint 为独立目录下的新普通文件，未覆盖源权重。依赖安装在该目录 `deps/`，未改共享 Python 环境。

- `--batches 2 --pilot-steps 4`：两个同 backward 诊断 + 四步真实更新。
- `--recovery-steps 64 --valid-samples 10`：上述全网三臂。
- 加 `--theta-only --recovery-arms official_eps12,margin,normalized --initial-scale 32768`：θ-only。
- `--q-probe --batches 2 --initial-scale 32768`：Q bridge 两批前向/反向验证，无更新。
- `train_binary_candidate.py --rule official --theta-epsilon 1e-12`：保留原梯度公式，仅采用正确裁剪和 θ 专用 ε 的进程内训练入口。不是已选定的长训最优超参数。
- `--neuron-fp32`：另开的 PSN 运算精度因素，不与 Q bridge 同时改变。

独立评审：[SURROGATE_REVIEW.md](SURROGATE_REVIEW.md)、[RECOVERY_REVIEW.md](RECOVERY_REVIEW.md)。没有生产硬件或论文数字变更；没有为神经元修复附带启动 BN 冻结训练、三元 warm start、长训或新 EDA。下一精度阶段必须保留修正回路后的匹配对照，再做完整验证。

## 6. 阶段裁决与接续顺序

- **确定性训练回路修复保留**：unscale-before-clip；不为负的短训结果恢复错误顺序。该修复已在隔离入口和真实回放中验证，尚未覆盖所有历史训练入口。
- **新的 backward 规则保留为候选**：margin、normalized、Q-only support桥都有明确的数学/数值边界；不包装成唯一真导数。Q桥目前只有两批连接性验证，没有其训练后精度结果。
- **ε与PSN精度因素保留，未选择新模型**：ε-only 全量微退化；FP32 PSN短恢复无精度改善。没有因一次布局/短训结果否决整个神经元家族，也没有以θ更新数量代替AEE。
- **下一组有界实验**：从同一ep34只解冻PSN时间矩阵、偏置和θ，固定正确裁剪和共同数值精度；对照“官方规则+θ专用ε”“明确的θ代理”“相同θ代理+Q桥”。先共同256个成功更新、相同训练数据和新建优化器，逐项区分θ规则与Q路径，不同时加入η剪枝/固定BN。全网骨干保持冻结，避免再次把大规模骨干更新混入神经元归因。候选若继续进入精度判断，用匹配valid825，而非按十帧选最终模型。此组**尚未启动**。

本目录所有远端实验已结束；没有留下后台长训。冻结ep34和生产overlay仍保留，后续可直接使用本目录脚本继续。
