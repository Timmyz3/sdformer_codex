# TTX 软件与硬件双轨研究路线（2026-07-11）

## 目标与共同约束

软件线目标是找到可替代当前 all-binary H60 TTX 的新 checkpoint；硬件线目标是在不损失
H60 数值结果的前提下降低 cycle、SRAM traffic、area 或 energy。两条线不混用结论。

共同部署约束：105 个 one-sided binary ATLIF；12 个 encoder attention 使用同一公式；
允许 `gate*K`，禁止 native `K*sn2_q(sumQ)` 后再叠第二 gate；DSEC 先行；最终必须
standard valid825，并报告 AEE、AAE、spikes、attention ops 和 memory traffic。

## 软件线：替代主线 checkpoint

### S0 已进入 full30

| ID | idea | 论文迁移 | 当前证据 | full30 状态 |
|---|---|---|---|---|
| H67 | Motion-XOR TTX | EDCFlow temporal difference -> binary XOR-popcount score prior | short360 valid40 1.5937；short checkpoint valid825 1.6537 | H67 先跑 |
| H68 | Castling-TTX | Castling-ViT training-rich/inference-cheap | short360 valid40 1.5650；deploy valid825 1.6544 | H67 后串行 |

短训 checkpoint 的 valid825 只能说明 360 steps 不够，不能替代 full30 结论。H67/H68
均从同一 TTX epoch2 独立启动标准 full30。H68 auxiliary 在 epoch20 前退火到 0，最后
10 epoch 只训练部署 H60。

### S1 下一批精度候选

1. **Dyadic-Temperature TTX**：Swin V2/cosine attention 的 logit-scale 思想迁移到 binary
   TX。当前 H60 profile 的 gate entropy `7.33985`、effective tokens `162/162`，说明
   Shiftmax 几乎均匀。用固定 power-of-two score scale 提高选择性，硬件只增加左移；先做
   3 个预注册 dyadic 点的 short/valid40，之后只允许一个 full30。
2. **Explicit Castling Distillation**：H68 是 output blend，不是显式 teacher loss。若 H68
   full30 仍不能超过 TTX，下一次只测试 `L_task + lambda*L(student,stopgrad(teacher))`，teacher
   为 full alpha-XNOR matrix，推理仍为 H60；不再扫 output blend。
3. **Polarity-Rail TTX**：从事件输入正负极性出发，用两个 binary rails 计算 same-polarity
   与 opposite-polarity popcount；12 块同式，不混 TX/SC。只有在现有 dual-rail 历史结果
   复核后才生成，避免重复实验。
4. **Progressive Temporal Curriculum**：借鉴视频 temporal progressive training，先保持
   H60，再逐步打开 motion/difference contribution；只作为 H67 full30 不稳定时的训练方法，
   不改变部署公式。

### S1 执行队列：H69 Dyadic-Temperature TTX

H69 已预注册 `score_scale={4,8,16}`。这三个值不是任意连续扫参，而是三档整数左移，分别
对应 H60 score accumulator 的 2、3、4 bit 左移。三项均从同一 TTX epoch2 做 360-step
训练和 valid40；只按预注册综合分数晋级一项 full30，最终执行 epoch
`0/4/9/14/19/24/28/29` 的 standard valid825。H69 不改变 ATLIF 数量、attention 数量、
Q/K 编码、`gate*K` 路径或 all12 一致性。

- config generator：`neuron_experiments/H9_bipolar_self_attention/entrypoints/make_h69_dyadic_temperature_configs.py`
- deferred queue：`neuron_experiments/H9_bipolar_self_attention/entrypoints/run_h69_after_h67_h68.py`
- manifest：`neuron_experiments/H9_bipolar_self_attention/configs/generated/h69_dyadic_temperature_ttx_manifest.json`

队列严格等待 H68 的 `profile_ranking_valid825.md`，不会和 H67/H68 抢 GPU。H69 的论文依据
来自 Swin Transformer V2 的 per-head scaled cosine attention：原论文以可学习温度控制
attention logit 并改善大模型稳定性。我们的迁移不是照搬 cosine/learnable float temperature，
而是针对已测得的 H60 均匀 gate，使用固定 dyadic temperature 恢复选择性。其有效性必须由
DSEC full30 证明，不能由 Swin V2 的分类结果代替。

### S2 执行队列：H70 Event-Selective TTX

NeurIPS 2024 Selective Attention 将 query/value temperature 与语义相似度解耦，用于控制
不同 token 的 attention spikiness；论文的 token-aware MLP 和 position-aware float scale
不适合本设计。H70 迁移为事件硬件形式：

```text
a_i     = popcount(Q_i OR K_i)
shift_i = min(ceil(log2(a_i + 1)), 3)
score_i = centered_TX_score_i << shift_i
gate    = Shiftmax(score)
out     = gate * K
```

它使用已有 binary Q/K、OR-popcount、leading-one detector 和 0--3 bit shift，不新增
attention 参数、第二套 Q/K、carrier 或混合 stage。动态温度放在 token 中心化之后，避免被均值
抵消。360-step 只检查 NaN、形状、加载和数值健康，不作为失败判据；随后固定执行 full30 与
standard valid825。

- generator：`neuron_experiments/H9_bipolar_self_attention/entrypoints/make_h70_event_selective_ttx_configs.py`
- deferred queue：`neuron_experiments/H9_bipolar_self_attention/entrypoints/run_h70_after_h69.py`
- manifest：`neuron_experiments/H9_bipolar_self_attention/configs/generated/h70_event_selective_ttx_manifest.json`

H70 相对 H69 的论文价值是“事件密度决定 attention selectivity”，不是单纯调一个全局超参；
相对 H67 的区别是 H70 不增加 motion score branch，只调制同一个 TX score。

### S3 执行队列：H71 Window-Context TTX

ICCV 2023 Context Broadcasting 的原式是 `X_i'=(X_i+mean_j(X_j))/2`，无新增参数，并在
标准 ViT 中显式承担 dense interaction。H60 的 `gate*K` 是逐 token 运算，gate 接近均匀不等于
已经发生 token mixing。H71 因此在每个 Swin window 的 H60 输出上应用原 CB 公式：

```text
Y_i = gate_i * K_i
Y_i' = (Y_i + mean_j(Y_j)) / 2
```

它保持 12 块同式、无 QK matrix、无 native carrier、无可训练参数。360-step 只做实现健康
检查，之后固定 full30。主要风险是广播 context 使后续激活变密，因此即便 AEE 改善，若
total spikes 不能保持 NB0 至少 20% 降幅，也不能替代主线。

- generator：`neuron_experiments/H9_bipolar_self_attention/entrypoints/make_h71_window_context_ttx_configs.py`
- deferred queue：`neuron_experiments/H9_bipolar_self_attention/entrypoints/run_h71_after_h70.py`
- manifest：`neuron_experiments/H9_bipolar_self_attention/configs/generated/h71_window_context_ttx_manifest.json`

### 2026-07-11 深读裁决：软件候选

| 来源 | 原机制 | 可迁移到 TTX 的最小形式 | 裁决 |
|---|---|---|---|
| Swin Transformer V2, CVPR 2022 | cosine Q/K 后除以每 head 可学习 `tau`，并加位置偏置 | 固定 power-of-two score temperature | **P0/H69**；不引入新乘法和参数 |
| Selective Attention, NeurIPS 2024 | token/position-aware query/value inverse-temperature | event-activity-conditioned dyadic temperature | **P0/H70**；固定公式 full30，不照搬 MLP |
| Context Broadcasting, ICCV 2023 | token 与全局均值各 1/2 相加，无参数 | window 内 `gate*K` 均值广播 | **P0/H71**；补 H60 缺失的 token mixing，需严查 spikes |
| Query-Key Normalization | Q/K L2 normalization，learnable scale 替代 `sqrt(d)` | 只迁移 scale calibration，不迁移 L2 datapath | 并入 H69 依据，不另立实验 |
| Differential Transformer, ICLR 2025 | 两套 Q/K attention map 相减以消除 common-mode noise | two-rail common-mode score cancellation | **P2**；双 QK、signed subtraction 和 GroupNorm 破坏当前简单数据流 |
| SimA, WACV 2024 | L1-normalized Q/K，重排为 `Q(K^T V)` 的 softmax-free attention | 固定维度统计量替代 token matrix | **停止**；需要矩阵乘与 `D x D` 状态，硬件不优于 H60 |
| BESTformer | reversible binary/full-precision coupling 与 information-enhancement distillation | 部署 TTX student 对训练期 matrix teacher 的显式蒸馏 | **P1**；仅在 H68 output blend full30 无效后做一次 |

#### H68 与 Castling-ViT 原论文的边界

CVPR 2023 Castling-ViT 的训练式包含 linear-angular 主分支、部署仍保留的 DWConv，以及与主分支
相加的 threshold-masked softmax auxiliary。其 `Mask_epsilon(x)=x if x>epsilon else 0`，论文
报告固定 `epsilon=0.02` 也可在后期自然全零。H68 则是 binary matrix output 与 H60 output 的
全局 `lerp`，并把 blend weight 确定性退火到 0；没有逐 attention-entry mask，也没有 DWConv。
因此 H68 的准确名称是 **Castling-inspired annealed matrix augmentation**，不能声称复现
Castling-ViT。H68 full30 配置保持冻结，避免启动前更换变量。

若 H68 full30 不超过 TTX，后续最多做一个 H72 faithful-mask 版本：masked matrix auxiliary
与 H60 相加、预注册 epsilon schedule、部署显式移除；不得同时加入 DWConv，否则会改变最终
硬件数据流。是否启动 H72 必须等待 H68 valid825，而不是根据 short360 决定。

Differential attention 不是当前首选：它解决的是两个 softmax map 的公共噪声，而 H60 当前
问题是 score 动态范围过小导致 gate 完全均匀。先校准温度比新增第二套 Q/K 更直接，也更符合
硬件边界。

Context Broadcasting 不能被用来断言 H60 的均匀 gate 已经合理，因为 H60 并没有把一个 token
广播到其他 token。H71 直接检验低成本 context mixing；它与 H69/H70 的 selectivity 校准是
互斥单变量实验，本轮不组合，避免同时改变 dense context 和 score temperature。

### 软件停止项

- TP/LR fixed-neighborhood attention：H66c valid825 AEE 1.6567，且 spikes 上升。
- full matrix 直接部署：运算和 SRAM 成本过大，且 valid40 泛化不稳定。
- TX/SC stage mixing、S2-only、partial replacement：不符合统一硬件数据流。
- SwiftFormer 原式：L2 normalization、learned global query、浮点 projections，不直接适合
  当前 multiplier-light TTX。

## 硬件线：不改变 checkpoint 数值

### H0 Exact Delta-TTX（当前第一优先级）

对 `alpha0=1/64` 使用 `S64=64*n11+n00`。t0 完整计算，t1 只更新 Q/K 翻转 lane。
100-sample element-weighted profile：Q toggle 0.7983%，K toggle 1.9946%，union 2.7832%；
t1 ideal skip 97.2168%，完整 T=2 compare reduction 上限 48.6084%。逐 lane 穷举等价已通过。

下一步：三档微架构候选（全并行、8-lane grouped、稀疏 index queue），核算 previous Q/K
state、S64 accumulator、scheduler、SRAM 和 clock gating 后的净 PPA。

### H1 Zero-Activity Folding

H60 profile 中多个 block 的 K-zero token ratio 达到 0.79-0.97。对 K=0 token，`gate*K`
输出严格为零，可跳过 late-scale 与 projection input read；这与 Bishop bundle sparsity 的
硬件目标一致，但不声称直接复现其 QK matrix ECP。需按 token bundle 统计 SRAM/cycle。

### H2 Error-Bounded Gate Bundling

对 binary K，省略 token 的投影前 L1 上界是 `abs(gate_i)*popcount(K_i)`。4/8-token bundle
总上界低于 epsilon 时跳过。它是近似硬件消融，必须报告 AEE-epsilon 曲线；不能与 exact
Delta-TTX 的无损结果混写。

### H3 Progressive TX Evaluation

迁移 SpAtten progressive refinement：每 8 channel 更新 score lower/upper bound，只有
低置信 token 继续读取下一组。由于 TTX channel 是 binary lane 而非 MSB/LSB，必须重新
推导 centered Shiftmax 下的 bound，并计控制发散。

### 2026-07-11 深读裁决：硬件候选

| 来源 | 原硬件机制 | 对 TTX 可用的部分 | 不可照搬部分 |
|---|---|---|---|
| SPRINT | ReRAM 近似筛分、数字精确重算、利用相邻 attention 的动态空间局部性 | changed-token index 缓存、稀疏读取后精确更新 | TTX 没有 `N x N` score matrix，不需要 ReRAM approximate pruning |
| Energon | mixed-precision 多轮 attention filtering | 8-lane grouped progressive refinement | TTX lane 已是 1 bit，不能宣称 bit-precision refinement |
| SpAtten | progressive quantization 与 token/head pruning | 分组 lower/upper bound、低置信组继续读 | 原 MSB-first 方法不能直接用于 binary channel |
| Bishop | token-time bundle、error-constrained pruning、density stratifier | bundle 上界、density scheduling | 其 QK/score/V ECP 不能直接等同于标量 TTX gate |
| MEET, CVPR 2025 | 线性层的 temporal Delta-Sigma execution，并联合优化 state memory | 强制把 previous-state memory/traffic 纳入净能耗 | 不能证明 nonlinear Shiftmax 或 TTX scheduler 节能；其 state-compression 网络重构不直接需要 |

由这些论文得到的硬件新包装是 **Delta-Locality TTX**：先用 exact Delta-TTX 产生 changed-lane
索引，再按 token/head bundle 压缩和调度索引，复用同一 popcount datapath。与 SPRINT 的关键
区别是本方案不做近似筛分，`S64` 更新逐 lane 数值完全等价。下一 profiler 必须新增：
zero-update token/head 比例、每 token 更新 lane 直方图、4/8-token bundle 全零比例和 changed
index run-length；只有扣除 previous-Q/K SRAM 和 index queue 后，才能报告净能耗收益。

上述 locality 指标已加入 profiler，使用 raw count 跨 block/head/sample 做 element-weighted
汇总。`run_delta_locality_after_h71.py` 会在 H71 full30/valid825 完成后对冻结 TTX ep2 自动跑
100 samples，不与软件训练抢 GPU，并把结果写回 redesign 与硬件文档。

MEET 对本项目的作用是风险约束而非算法来源：其核心观察是 temporal suppression 可能因 state
memory overflow 反而损失能效。Delta-TTX 的 previous Q/K 是 packed binary `2D=64 bit`，已经
远小于 MEET 所处理的 dense activation state，因此暂不引入额外 state compression；PPA 仍需
把该 64-bit state、S64 accumulator 和读写能耗完整扣除。

## 文献深读流水线

每篇论文按以下模板落盘，禁止只抄摘要：原公式/代码路径；训练与推理差异；状态与 memory
代价；可迁移机制；与 TTX 的语义差异；最小实验；硬件新增模块；可能的新 claim；停止条件。

检索分两组持续进行：

- 软件：CVPR/ICCV/ECCV/NeurIPS/ICLR/ICML 的 attention、video delta、optical flow、binary
  network、training-time auxiliary、logit calibration、token routing。
- 硬件：ISCA/HPCA/MICRO/ASPLOS/DATE/DAC/CICC/ISSCC 的 attention accelerator、temporal
  reuse、delta execution、structured sparsity、approximation bound、near-memory/event-driven。

## 当前主线判定规则

在 H67/H68 full30 完成前，当前主线仍是 TTX epoch2：AEE 1.5016、AAE 9.8431、spikes
23.2439G。软件候选只有 valid825 AEE 优于或统计上不差于 TTX、AAE 不恶化且 spikes 至少
保持 20% baseline 降幅，才可替代。Exact Delta-TTX 可作为同一 checkpoint 的硬件主线，
但 PPA 未扣除状态和调度成本前不得宣称 48.6% attention energy reduction。

## 能耗双口径（强制）

standard valid825 的 `energy_uj` 实际只按被 hook 的 spike layer 累加 `spikes x AC/logic`，不含
H67 motion XOR/popcount、H69 score shift、H70 OR-popcount/leading-one、H71 context
reduction/broadcast，也不含 SRAM/NoC。因此未来 ranking 显式改名为
`spike_energy_proxy_uj`。候选主线必须同时报告：

1. standard valid825 的 AEE/AAE/total_spikes/spike-energy proxy；
2. `attention_candidate_ops.json` 的增量 logic/add/fixed-MAC 与统一 45 nm proxy；
3. 最终 RTL/PPA 的 attention state/SRAM/NoC/控制能耗。

前两项只能用于软件筛选和操作审计，不能替代第三项 post-layout 或综合后结论。

## Float 训练与 dyadic 部署双评估

所有 full30 候选训练和第一轮 valid825 保持 `alpha0=0.02`，用于判断收敛；随后仅对各候选
float rank-1 epoch 做一次统一 `alpha0=1/64 + INT8 score/gate` valid825，并与 TTX ep2 在同一
脚本中重评。主线替换最终依据是该部署表，而不是把候选 float AEE 与 TTX dyadic AEE 直接
比较。Exact Delta-TTX 的 `S64=64*n11+n00` 也只对 dyadic 部署图逐 lane 精确，不对 float
`alpha0=0.02` 声称 bit-exact。

历史 `INT8` 配置的真实网格为 score `[-2,2]/2^-7`（513 levels，至少 10 code bits）和 gate
`[0,2]/2^-7`（257 levels，至少 9 bits）。软件表继续沿用该冻结网格，硬件线按 10/9-bit
核算；真正 8-bit 的 range/step 重新设计必须作为独立量化消融，不能偷换已有 AEE。
