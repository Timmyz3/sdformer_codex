# T55–T57 阈值与 BN 独立审阅

2026-09-19。范围：Claude T39–T57 交接中的算法结论，兼核 CodeBuddy H12 backward。没有训练、没有占 GPU、没有修改任何 overlay。

**结论：η 曲线、两臂 θ 终态差异和 BN 三次评测是真实可保留的结果；“官方阈值因 fp32 天生冻结”“12 个 sn_q 无消费者可删”“固定 BN 必亏 8.48%”“四种 surrogate 已数学修复”均超出证据，部分可直接反驳。** θ 吸收不需要冻结 BN，但“全图无条件精确且零硬件代价”还没有完成导出验证。

## 1. 证据与独立复核

本地审阅根 `../claude_fusion_trials_20260914/`：

- `results/CODEX_HANDOFF_T39_T57_20260919.md`，全部算法节；`results/T55_ADAPTIVE_HW_AND_ETASWEEP.md`、`T56_THRESHOLD_GRADIENT_AUDIT.md`、`T57_BN_AND_FOLD.md`。
- `t56_channel_activity_surrogate.py`、`t57_gradfix_train.py`、`t57_gradfix_run.sh`、`t57_bn_recal.py`、`t57_bn_chain.sh`、`t57_frozenbn_train.py`，以及 `results/t56/t56_audit.json`/CSV。
- 当前 H9 `overlay/models/STSwinNet_SNN/atlif_ternary_psn/atlif_ternary_psn.py`、`h28_optimizer.py`、`bsa_attention.py`、训练入口与上游训练脚本。

远端只读根 `R=/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention`：

- `R/results/t57_grad_{ctrl,fix}/warmstart/checkpoint_epoch34.pth` 与各自 `checkpoint_epoch35.pth`；**使用远端 `/usr/bin/python3.12` 的 zip/pickle 元数据读取器，只读标量存储，无 torch/GPU**。
- `R/results/t52_eta_{off,on}/standard_valid825/epoch35/spike_profile.json`、`t55_eta004`、`t55_eta016` 对应文件。
- `R/results/t57_bn_probe/{ep34_running_raw,ep34_running_recal_train,ep34_running_recal_valid}/spike_profile.json`。
- `/root/t57_gradfix.log`、`/root/t57_bn_chain.log`；远端当前 H9/H12 源码。

新产物：

- [THETA_CHECKPOINT_REVIEW.json](THETA_CHECKPOINT_REVIEW.json)：两臂全部 81 个 θ 值、变化及未动名单。
- [read_theta_checkpoint_cpu.py](read_theta_checkpoint_cpu.py)：可重复的只读提取器。
- [theta_bn_math_checks.py](theta_bn_math_checks.py) / [JSON](theta_bn_math_checks.json)：float32、BN 方差、Q 前向依赖、H12 负支的纯 CPU 数学核对。
- `theta_bn_sources/`：远端 H9/H12 源码、实际 ctrl 配置、两份日志的只读快照；路径在 `PATHS.json`。

## 2. 可采纳的实测结果及正确口径

### 2.1 两臂 θ 终态差异独立成立

| 项目 | 官方 ctrl | T57 修订 backward |
|---|---:|---:|
| warmstart θ 键 | 105 | 105 |
| 最终 checkpoint θ 键 | 81 | 81 |
| 两端共有键 | 81 | 81 |
| 最终值与初值不同 | **0/81** | **69/81** |
| 增加 / 减少 / 不变 | 0 / 0 / 81 | **5 / 64 / 12** |
| max abs(Δθ) | 0 | **0.00033921003341674805** |
| 全 81 键 Δθ 均值 | 0 | **−4.6855137671953364e−5** |

两臂 warmstart θ 完全一致。均值与交接的 −4.67256e−5 稍不同，原因是交接用 step20→step1200，本表用 checkpoint 起点→终点。

准确表述是“1200 步训练后，ctrl 81 个 θ 的终态与起点逐位相同；每 20 步日志均值亦未变”。没有逐步、逐键轨迹，不能将终态等同于“每一步都绝对没动”。checkpoint 仅保存 `model_state_dict`，没有该次 optimizer/scaler 状态。

### 2.2 η 曲线可以继续作为算法候选

远端 valid825 原始 JSON 核得：

| η | AEE | 脉冲总数 | JSON 的 activity-energy proxy / µJ |
|---:|---:|---:|---:|
| 0，同预算训练 | 1.1956454382520734 | 73,247,758,507 | 64,183.090193100004 |
| .004 | 1.2062192952271664 | 69,181,551,810 | 60,583.5671578 |
| .008 | 1.217246972214092 | 65,252,583,635 | 57,104.92234830002 |
| .016 | 1.2898980408003835 | 58,613,007,780 | 51,224.25174120001 |

全部 825 帧、480×640、窗 2×15×15、`no_running`、eval batch=1。η=.008 的活动量和代理约降 11%、AEE 约升 1.81%，数值可以保留。

**必须改名：这不是 28 nm 芯片能量。** 每份 JSON 自带 `energy_scope=spike_activity_proxy_only_excludes_overlay_attention_control_reduction_and_memory`，常数为 spike AC 0.9 pJ、attention logic 0.1 pJ。`eval_DSEC_flow_SNN.py:838–851` 按每层脉冲计数乘常数，没有逐脉冲 fanout、实际访存、BN 归约、PSN 连续算术、时钟/漏电。该数还是整次 825 帧统计的总量，不是单帧硅实测。

“BN running 只省 1.72% 能量，所以不值得”尤其不能这样判断：这个 proxy **恰好没有计入它要省的 BN 服务**。只能说它改变了脉冲代理 1.72%。是否划算应计入同一硬件时间线/状态/归约成本，再看 AEE。

η 是同预算干预，但权重也参与了训练，所以 −10.9% 是“启用 η 及其引发的协同训练”的结果，不是固定权重上只改 θ 的净因果量。没有种子重复，不能由 4 点断言普遍强凸或唯一最优拐点。用户当前精度门为优于匹配 SDformerFlow baseline，不能未经同协议对照就用 local5 1.2819 或旧 NB0 宣判 η=.016 必死。

### 2.3 BN 三次评测数值可保留

| 部署口径 | AEE，远端 JSON |
|---|---:|
| 训练 EMA，直接 running | 1.3007879134380456 |
| train 前 300 batches 重校准 | 1.3035984522284882 |
| valid 前 300 batches 重校准，仅诊断 | 1.3012877043810758 |

这是三种**已试的零恢复方案**，不是整个固定统计家族的下界或终局。

## 3. θ 吸收：核心更正正确，覆盖结论过强

对推理常数 θ，线性消费者满足 `W(θg)+b=(Wθ)g+b`；BN 若在该线性结果之后，无论使用运行统计还是样本统计，都不改变这个恒等式。**T55b 所称动态 BN 阻挡 θ-fold 的判断确实错误。**

但 T57 用两段 MLP/attention 源码推出“所有 ATLIF 直接接权重、全图无条件精确”不成立：

| 接口 | 正确处理 |
|---|---|
| MLP sn1/sn2 → FC、patch/residual/pred 等脉冲 → Conv | 按所有消费者的输入列/通道吸收 θ；bias 不乘 θ；保留神经元比较阈值 |
| proj_sn → linear_q 与 linear_k | 两份下游权重都要吸收；不是只改一条分支 |
| sn_q → TX 分数 | 没有直接可吸收的 W；当前正 θ 经过 sign 后只剩支持位，幅值在这里被消去，**但 Q 支持位仍参与分数** |
| sn_k → 分数与 gate⊙K | 当前 h60、μ=0、K-magnitude=0 等条件下，分数用支持/符号，载荷可写 θk·gate·gk，再吸收进后续 proj；不是“神经元后立即 Linear” |
| 拼接、多消费者、skip、量化/舍入边界 | 需逐边列出吸收点，不能给一个标量乘法示例就宣称全图编译完成 |

`θ=1` 不等于“普通 IF 神经元”：当前前端仍有学习到的非因果 T×T PSN 连续变换。θ 训练不动不说明整个神经元没有学习。

“精确”要区分实数代数与定点/FP32 实现。T57 自己的 max error 2.4e−7/4.8e−7 已说明不是逐位精确；后接阈值时，小误差也可能翻门。折叠后权重范围、dyadic 缩放和量化是否改变要另验。尚未看到全 81 活层的导出与整网折叠 A/B。可以说“二电平接口允许将静态幅值编译到线性系数中”，不能直接说“全网络在相同位宽下零增量、无条件逐位等价”。

## 4. 81/69/12 混账：sn_q 不是已删 attn_sn

独立 checkpoint 名单清楚显示，fix 未动的 12 个键全部是：

`sttmultires_unet.encoders.swin3d.layers.{0,1,2,3}.swin_blocks.*.attn.sn_q.spiking_neuron.thresh`

它们**不是**之前未调用的 12 个 `sn2_q`，也不是输出无消费者的 12 个 `attn_sn`。后两族已经从最终 81 键中移除了。不能用“另有 12 个注意力侧”含混地说与 105/93/81 自动吻合。

实际可保留的口径：105 历史安装；93 历史调用；81 推理功能图活层；此处 6 帧离线审计/1200 步训练里有 69 个 θ 获得可见任务梯度/变化。**81 功能活层 ≠ 81 非零梯度层。** 仅存在 optimizer `exp_avg` 也不证明梯度非零或 θ 更新。

为什么 sn_q 梯度为零而前向有用：

- `bsa_attention.py:_qk_shiftmax_gate_forward` 的 q 进入 `_tx_sc_fusion_score_pair`。
- `_ternary_alpha_xnor_token_scores` 的 `same_nonzero`、`same_zero`、`opposite` 使用 `==`、`!=`、布尔运算；尽管 sign 外壳有 STE，这些比较再次截断梯度。
- 当前配置 `bipolar_mu=0`、`k_magnitude_alpha=0`、`single_active_penalty=0`；Motion-XOR 的可微算术项只依赖 K。
- 因而观察到 Q 输出梯度为零符合代码，但 Q 改变仍改变 overlap/共静默分数和其他 token 的 gate。

纯数学反例见 `theta_bn_math_checks.json`：两个 token、相同非零 K、head_dim=32，从 Q 全零改一个 Q 位，gate 从 `[1,1]` 变为 `[1,0.978572...]`。不能删 Q。

**T56 “A 类 12 个 sn_q 可以直接删（无消费者）”必须撤回。** 是否需要为 Q 评分路径加合适的代理梯度，是一个训练问题；必须保留 forward 不变、独立验证梯度与 AEE，而不是把它当推理死模块。

## 5. 官方梯度、Adam、FP32：观测成立，因果尚未闭合

### 5.1 上游没有 `.abs()` 的那一类必须单独看

当前本地及 sd5ai H9 `OfficialATLIFSurrogate.backward` 为 `−mean(grad·tmp²)`，没有 `.abs()`；与[上游原始源码](https://raw.githubusercontent.com/putshua/Activity-Pruning-SNN/main/models/submodules/layers.py)的该函数相同。H9 的另外三种自定义 Binary/Ternary/Symmetric 变体有 `.abs()`。CodeBuddy 所称“4 种 backward 全都恒负，H67 主线也如此”不符合当前官方模式源码；不能迁移到 ep34。T57 的修订为 `sum(grad·(g−tmp))`。

### 5.2 这个 A/B 一次变了三件事

从 `−mean(grad·tmp²)` 到 `sum(grad·(g−tmp))` 同时：

1. 加幅值项 g；
2. tmp² 改 tmp；
3. mean 改 sum，对共享标量 θ 额外改变了元素数倍的尺度。

所以 1.75e7 的梯度倍率不能都归因于“补幅值项”。要判定主因，至少分开保持 mean 的改核、只改 sum、最终组合。它是训练规则变更，不是只改一个数学项的因果证明。

硬阶跃在通常意义下不可微；“唯一数学正确 surrogate”没有定义。若约定以 margin 的 δ 核 `tmp/θ` 近似，`g−tmp` 与输入梯度 tmp 相容；若从归一化坐标 `h/θ−1` 的代理函数链式求导，则 θ 项出现 `g−(h/θ)tmp`。必须明示代理规则，而非把自定义 backward 等同于硬函数的精确导数。

### 5.3 T56 离线梯度不是 6 帧平均

`t56_channel_activity_surrogate.py:run` 每个样本先把每层 `.thresh.grad=None`，循环结束后才读一次梯度；因此 5.31e−10 等是**第 6 帧的跨模块统计**。活性/窗口是累计的，loss 是样本均值，梯度却是最后一帧。还使用 eval/no_running、batch=1、自己的 masked L1 loss；T57 使用 train/batch=2/AMP/训练损失。直接外推每步更新缺少同口径基础。

### 5.4 ULP 论证原式有错

在 float32 的 **1.0 正上侧**间隔是 2⁻²³≈1.19209e−7，**下侧**是 2⁻²⁴≈5.96046e−8。方向不同，不能笼统说“1 附近 ULP=2⁻²⁴”。

代入报告的 g=5.31e−10、lr=1.25e−6、ε=1e−8，常梯度 Adam 首步经偏差修正后更新约 **6.30282e−8**；CPU 浮点核对：

- float32(1−Δ)=0.9999999403953552；
- float32(1+Δ)=1.0000001192092896。

**两者都会动。** 因而不能用该中位数推出“低于半 ULP，1200 步全被吃掉”。Adam 的 bias correction 不等于 warmup；实际 T57 `lr_warmup.enabled=false`。量化停滞可能存在，但需要真实 step 前的 m/v、lr、unscaled 梯度与 θ 写回前后的数值才能定因。

### 5.5 一个必须排除的竞争解释：AMP 中先 clip 后 unscale

远端实际 `third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py:360–375` 是：

```
scaler.scale(loss).backward()
clip_grad_norm_(..., 100)
scaler.step(optimizer)       # 这里才隐式 unscale
```

H9 entrypoints 的 `_patch_source` 修改 step 附近的阈值更新，没有补 `scaler.unscale_(optimizer)`；实际 T57 配置 `use_amp=true`、`num_acc=1`、clip=100。

如果该次 scaled 梯度范数触发裁剪，随后再除 scale，会把真正的 clip 门降为约 100/scale；小 θ 梯度可能进一步被压低。这是与 T56 无 AMP 不同的可检验机制。

**但未找到实际逐步 scale/preclip norm 记录，不能宣称已证明这就是此次不动的原因。** 最终 checkpoint 不保存 scaler/optimizer，不能事后从终态反演。这一项是竞争解释，不是替换旧说法的新“定案”。

### 5.6 方向与 700× 应降格

修订臂 64 个 θ 下降、5 个上升，不是“任务损失总想推低 θ”。平均方向可以报告，但不证明帕累托曲线凸性的原因。所谓 700× 比较的是一个训练段的平均漂移与另一训练的手动增量；模型/参数在协同变化、方向不同，不能视为两种优化器的固有速度比，也不能线性外推更多 epoch。

## 6. H12 “四种修复”还含不一致，不能直接替换 H9

sd5ai 当前 `H12_atlif_gradfix/.../atlif_ternary_psn.py` 四种 θ backward 都是 `mean(grad·code)−mean(grad·tmp)`，输入 backward 都是 `grad·max(pos_tmp,neg_tmp)`。它不是 T57 的 sum 版本。

以一种明示的 margin δ 代理约定为例：p 为正支三角核、n 为负支三角核、负阈值 rθ，δ+≈p/θ、δ−≈n/(rθ)。则：

| 输出函数 | 输入 h 的局部导数 | θ 的局部导数 |
|---|---|---|
| θH(h−θ) | p | g−p |
| θ[H(h−θ)−H(−h−rθ)] | p+n/r | code−p+n |
| θ[H(h−θ)+H(−h−θ)] | p−n | active−p−n |

H12 三元负支把应为 `+n` 的 θ 项写成 `−n`，输入项还漏 r；对称 binary 的输入负支应有负号。纯 CPU 例 θ=1,r=4,h=−5：p=0,n=.75,code=−1，该约定给 θ 导数 −.25、输入 .1875；H12 给 −1.75、.75。

共享标量 θ 的广播反传通常对全部元素 **sum**；继续 mean 是另行归一化的训练规则，应该显式定义学习率缩放。若选择其他代理规则也可以研究，但测试“按实现所写公式”通过，不等于数学链式导数验证。

因此保留 H12 为隔离候选，撤回“四类已修复定案”“三元训练崩溃完全排除 backward 嫌疑”。Forward 不变只保证相同参数下前向，不保证训练路径不变。其 local5/三元试验也不能与 ep34 binary 直接并账。

## 7. BN 的 N1c 不能证明固定统计的结构性极限

### 7.1 实际校准不等于“全数据精确统计”

实际 `/root/t57_bn_chain.sh` 传 `--batches 300`，脚本默认 batch=2、shuffle=False、drop_last=True；N1c 只看了 valid 前 **600/825** 帧。

`momentum=None` 是各批均值与各批无偏方差的累计平均。相同 batch 大小时均值可对应这部分样本，但**平均批内方差不等于总体方差**，少了批均值之间的项。CPU 反例两批 `[0,0]` 与 `[10,10]`：累计批内方差=0，总体 population variance=25。

此外所有 BN 同时处于 train，深层激活是在上游 batch-stat BN 下采集；部署时上游换 running 后，其输入分布再次变化。因此连“深层部署激活的精确统计”也没有得到。

### 7.2 即使获得全 valid 真矩，也不是“最优固定统计”

最大似然/矩匹配统计不等于最小化最终 AEE 的 μ/σ，更不等于固定统计家族的全局最优；γ/β、前后权重、校准次序和恢复训练也未优化。N1c 使用评测集，应明确为有泄漏风险的诊断，不能进入无泄漏部署结果；它也不是 oracle 上界。

合法结论：**当前 checkpoint 在这三种零恢复 running 方案下比 no_running 历史锚点差约 8.4–8.7%，简单重校准未恢复。** 不可写“拿掉动态 BN 必须付 8.48%”“N2 是唯一翻盘路径”。N2 是值得优先试的一个直接方案；分层校准、受限自适应统计、可训练仿射等接口尚未证伪。

### 7.3 N0 路径也必须先修

`t57_bn_chain.sh` 用 `sorted(glob('**/spike_profile.json'))[0]` 选锚点。实际 `/root/t57_bn_chain.log` 的自动 N0 为 **1.2072849134242896**，并不是报告引用的 1.19951。说明该 glob 选中了另一份评测，不能自动装订为控制臂。

报告的 1.19951 是历史锚点；T54 同环境重评为 1.1986095554。应显式选定同 checkpoint/config/overlay/环境的精确 N0 路径，必要时重评。±.08% 一次漂移不是统计置信带；不应在论文把它写成通用误差条。

### 7.4 N2 实现注意点

`t57_frozenbn_train.py` 临时将 `_BatchNorm.training=False`。只有实际存在 running buffers 的模块才保证固定统计；`track_running_stats=False` 或 buffers=None 时仍可能走 batch 统计。必须核到实际各 BN 模块，不是以 monkeypatch 调用计数代替语义确认。源脚本的 docstring 仍保留“动态 BN 挡 θ-fold”，也需在自己的新实验说明中纠正。

## 8. 下一步最小实验（建议，未启动）

1. **先做一组固定 batch 的训练回路回放，不开长训。** 从独立 ep34 副本、相同模型/optimizer 初态与 RNG 出发，保留官方 backward、η=0，比较旧 AMP 顺序与 `unscale_→clip→step`。固定少量 batch；每层记录 scale、unscale 前后 norm、实际 clip 系数、θ raw/unscaled 梯度、m/v、double 计算的拟议更新、float32 θ 写回值。另做无 AMP 对照用于解释，而不是用 T56 的另一个 loss 代替。若旧 clip 未触发，就排除该竞争解释；若触发，再判断它是否足以越过半 ULP。不要先跑一个 epoch 才找原因。
2. **Q 前向依赖与训练梯度分开确认。** 保持同 checkpoint，选一块真实非零 Q，做局部支持位扰动并比 score/gate，确认 hook 真命中。若研究 Q 代理梯度，forward 必须完全不变，且作为新训练规则单独评测；不删除 sn_q。
3. **固定 BN 的同预算成对恢复。** 只用训练集校准，明确初始 checkpoint 与每层 running buffers；固定 BN 臂与 batch-stat 控制臂使用相同数据顺序、预算、学习率、η/梯度规则以及一致的 AMP clip 顺序。记录 BN 运行统计在固定臂确实不变。先小预算配对探针，值得继续再 valid825。不能拿旧 t52_eta_off 代替变更训练管线后的新控制臂，也不能把 N1c 的 valid 校准带进部署臂。
4. **η=.004/.008 接真实硬件请求账。** 先从同层捕获比较完整源字/广播组/PSN 服务/BN 服务，而不是把 −11% 脉冲直接当 −11% 周期或能量。θfold 按所有消费者导出，单独保留量化后的 AEE。满足这一接口再讨论调度/稀疏电路贡献。

这四步不需要复扫几百篇文献，也不等于否掉 ATLIF/BN 家族。它们先把已测数字、可修问题和未证因果分开，让后续硬件试验不再依赖错误的“死模块”或“免费 BN”前提。
