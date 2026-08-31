# M1535｜Motion ep34 有损剪枝 / 稀疏跳过候选独立挖掘

日期：2026-08-31（Asia/Shanghai）  
性质：独立、只读、第一性原理候选审阅  
对象：Motion H67 ep34 checkpoint、M1458 40-sample capture 与既有 G/PAFT/Phi/Delta 负结果  
裁决：**只授权 S1、S2 的 CPU / forward 快杀；S3 等待 INT8 数值桥；S4、S5 只作训练侧低优先级筛选。未授权 GPU、EDA、SSH 或 RTL。**

本审阅没有修改旧结果、论文、`docs/359_DATE终局冻结_20260813.md` 或 `ucli.key`，没有运行 GPU、EDA、VCS、SSH 或训练。`docs/359` SHA256 复核为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## 1. 结论先行

现有 checkpoint 并不存在一片“免费等着硬件吃掉”的结构化权重零：本次只读 CPU 静态审计在主要 Conv、decoder、patch、FFN、downsample 与 attention projection 中观察到 **0 个 FP32 精确零权重**和 **0 个 16x16 FP32 全零 weight block**。把 FP32 临时按 per-output symmetric INT8 映射后，零权重也仅约 `1.0%--1.6%`；该映射只是诊断，绝不是 ep34 的量化权威。45 张 `10x10` ATLIF temporal matrix 同样没有精确零系数，行 / 列 L1 质量总体接近均匀。因此：

- 不能把“天然 weight sparsity”“天然空 temporal phase”预写成收益；
- 不能把既有 G7/G8/G11、Phi/PAFT 或 Delta 路线换名复活；
- 真正值得测的是 **非二值边界的小贡献**与 **能在 weight fetch 前整块拒绝的动态证书**。

本轮至多保留五条候选，优先级如下：

| 优先级 | 候选 | 类型 | 当前裁决 | 能跳过什么 |
|---:|---|---|---|---|
| **P0** | **S1 Analog-Boundary Contribution Gate（ABCG）** | 有损，`epsilon=0` exact | **GO 轻量 capture + CPU/forward 快杀** | dropped source 对应的 weight fetch、MAC / add、psum update；输入本身仍要读 |
| **P1** | **S2 Compact Certified Block Skip（CCBS）** | 有损，`epsilon=0` exact | **GO CPU 快杀，但先证明不退化为 G11** | 整个 `(source group, output tile)` 的 metadata 通过后，可同时跳 weight block fetch、compute、psum update |
| **P2** | **S3 Activity-Bounded Refinement-Plane Elision（ARPE）** | 有损，`b=0` exact | **BLOCKED by ep34 INT8 bridge** | 低位 refinement SRAM fetch、bit-plane compute、对应 psum add；最终 commit 不省 |
| **P3** | **S4 Phase-Structured T10 Pruning（PSTP）** | 重训有损，全 phase 为 baseline | **TRAIN-SCREEN ONLY** | 静态被删 temporal coefficient / phase 的 fetch、multiply、add；跨层 phase 消失须另证 |
| **P4** | **S5 Source-Owned Destination N:M（SD-N:M）** | 重训有损，`N=M` 为 baseline | **BASELINE / ABLATION ONLY** | compact weight fetch、被删 destination multiply / update；source read 与最终 commit 不省 |

当前最有 H67 对象差的是 **S1**：既有 G7 测的是 bottleneck 的 `{0, layer-constant}` ATLIF 源，因没有中间幅值而死亡；S1 打的是 ep34 中真实存在的非二值边界——raw event ingress 与 patch analog residual。S2 的潜在覆盖更大，但与原 G11 的动态累计预算相邻，只有“块 metadata 显著更小 + fetch-before-compute + executable 同资源收益”同时成立，才不算换名。

## 2. 已有负结果与禁止复活清单

### 2.1 明确死亡

| 旧路线 | 已有结论 | 本审阅约束 |
|---|---|---|
| G7 bottleneck amplitude gate | 真实源基本只有 `0` 与层常量，冻结阈值网格无额外稀疏 | 不在 bottleneck 上再造幅值比较器；S1 只允许打真实 non-binary boundary |
| G8 whole-token FFN bypass | exact whole-token 机会接近 0；较大 tau 是 postcompute oracle | 不把 residual 小写成可预知 skip；不以 post-BN2 oracle 估周期 |
| G11 static `beta` / top-m | 对 binary source 等价于固定 source-destination mask；M301 beta48 的 `Delta-AEE=+0.1105` 远超 `0.02` | S5 不得冒充新机制；S2 必须证明 dynamic block witness 与 metadata/cycle 优势 |
| G11 token-dynamic cumulative B | 数学上有界，但逐 source beta/order metadata 可达静态 mask 的 `24x`，扫描 / membership 可能比 compute 贵 | S2 只有块级 metadata 和整块 fetch gate 真正改变 max() 才能存活 |
| M70--M76 Phi / PAFT headline | nominal pattern opportunity 被 matcher、PWP、port/capacity、split/accuracy 身份削弱 | 不复活 pattern codebook / residual drop；只作 prior / baseline |
| RQTB / epsilon-RQTB | attention 在完整工作中份额太小 | 不作为系统剪枝主轴 |
| Delta / temporal XOR | decoder 相邻 XOR 比 full active source 更密；自然 dual-parent 机会小 | 不再用“跨帧 delta”换名；S1 是当前帧 source gate，不保存上一帧 feature |
| G10 empty tile | 空 output-site 约 `0.1117%` | 不把低幅 event gate 写成空 tile skip；两者判据、精度与硬件不同 |
| G12 ATLIF remaining-budget early stop | term skip 不对齐 32-lane issue，cycle 仅有极小变化 | S4 必须删整列 / 整行 phase 并打到服务周期，否则立即杀 |

### 2.2 与公开工作的合法关系

- [Bishop](https://arxiv.org/abs/2505.12281) 已覆盖 TTB、structured firing sparsity 与 error-constrained pruning；本项目不能把“有误差界的 pruning”本身称为新。
- [Phi](https://arxiv.org/abs/2505.10909) 已覆盖 pattern + residual hierarchy 与 PAFT；S5 不能包装成另一种 Phi。
- [DeltaCNN](https://arxiv.org/abs/2203.03996) 已覆盖帧差传播与小更新截断；本轮不复活跨帧 delta。
- [AccelTran / DynaTran](https://arxiv.org/abs/2302.14705) 已覆盖 runtime activation pruning；S1 的 claim 只能是 event-optical-flow 的具体对象、边界和 C2 fetch-before-compute protocol。
- [ProSparse](https://arxiv.org/abs/2402.13516) 已覆盖通过训练推动 activation threshold sparsity；S1 当前只测冻结 ep34，不宣称发明 threshold shifting。
- [SCA pruning](https://arxiv.org/abs/2406.01072) 和 [spatio-temporal SNN pruning](https://arxiv.org/abs/2104.12528) 已覆盖 SNN channel / temporal pruning；S4、S5 只能作 co-design / baseline。
- [N:M sparse training and accelerator co-design](https://arxiv.org/abs/2309.13015) 已覆盖 N:M 的硬件友好结构；S5 的 novelty ceiling 很低。
- [Bit Fusion](https://arxiv.org/abs/1712.01507)、[Pragmatic](https://arxiv.org/abs/1610.06920) 与 [Dynamic Stripes](https://arxiv.org/abs/1706.00504) 已覆盖可变精度 / bit-serial / zero-bit execution；S3 只能主张 H67 typed-source bound 与 refinement fetch gate，不能主张 bit-plane execution 是新发明。

## 3. ep34 checkpoint 与 M1458 的只读事实

### 3.1 权重 / temporal 静态审计

审计使用 `/opt/anaconda3/envs/pytorch310/bin/python` 在 CPU 上只读加载 checkpoint 的 `model_state_dict`。临时 INT8 诊断采用每个 output row 的 symmetric scale `max(abs(w))/127`；它没有量化、保存或修改 checkpoint，也不构成 M1526 所缺的 ep34 INT8 authority。

| scope | FP32 weight 数 | FP32 zero | 临时 INT8 zero | 16x16 FP32 全零块 | 2:4 保留 L2 能量（逐层均值） | 4:8 保留 L2 能量（逐层均值） |
|---|---:|---:|---:|---:|---:|---:|
| C1 bottleneck Conv | 21,233,664 | 0 | 1.496% | 0 / 82,944 | 84.81% | 88.66% |
| decoder deconv | 7,140,096 | 0 | 1.625% | 0 / 27,864 | 86.07% | 89.78% |
| patch family | 466,872 | 0 | 1.095% | 0 / 1,821 | 83.38% | 87.33% |
| FFN FC1 | 8,626,176 | 0 | 1.061% | 0 / 33,696 | 86.89% | 89.79% |
| FFN FC2 | 8,626,176 | 0 | 1.151% | 0 / 33,696 | 86.78% | 89.70% |
| downsample | 1,548,288 | 0 | 1.106% | 0 / 6,048 | 86.79% | 89.70% |
| attention projection | 6,469,872 | 0 | 1.027% | 0 / 25,272 | 95.03% | 96.12% |

解释：2:4 / 4:8 列只回答“未经重训直接保留最大幅值时，weight L2 留下多少”，不是 output error、AEE、speedup 或可综合结果。对高份额 patch / FFN / C1，直接删一半权重会丢掉约 `10%--17%` 的 weight L2，说明 S5 必须重训并做 AEE；不能靠自然零直接晋级。

另外：

- 45 张 `10x10` ATLIF matrix 共 4,500 个系数，精确零为 0；各 phase 的跨层平均 column-L1 share 为约 `9.09%--10.80%`，row-L1 share 为约 `9.62%--10.37%`，没有全局显著弱 phase。
- 12 个 MLP 的 BN1/BN2 gamma 大致落在 `0.75--1.19`，未观察到靠 gamma 近零即可删掉的 dead channel。
- 以 `||fc1_row||2 * ||fc2_column||2` 作静态 paired-channel 诊断时，最低 10% expanded channels 仍占各层约 `8.3%--9.2%` 的 score sum；分布并不呈现可免费删除的长尾。该事实与 M156/M159 的“必须训练 / valid”边界一致。

### 3.2 M1458 活动与非二值边界

`operator_runtime.json` 的 module sample 最多保留 1,024,000 个输入元素的统计，而不是完整 payload。以下数字只能用于候选优先级：

| 模块 / 类别 | 40-sample input activity | sampled binary-01 ratio | sampled range / mean-abs | 79-module activity-weighted-MAC proxy share |
|---|---:|---:|---|---:|
| `patch_embed.head.conv.0` raw event ingress | 26.506% | 65.420% | `[0,0.8426]`, mean-abs `0.00706` | 1.713% |
| `patch_embed.proj.conv_res` analog residual | 100.000% | 0% | `[-11.94,15.42]`, mean-abs `1.3801` | 4.309% |
| FFN FC1 aggregate | 12.310% | mostly binary / layer-code | ordered-statistics only | 23.39% within M1529 proxy taxonomy |
| FFN FC2 aggregate | 3.154% | mostly binary / layer-code | ordered-statistics only | 6.91% within M1529 proxy taxonomy |
| C1 bottleneck aggregate | 10.079% | `{0, layer-constant}` by retained payload audit | retained exact payload exists | 15.64% within M1529 proxy taxonomy |

关键判断：raw ingress 的 non-binary ratio 与很低 mean-abs 给 S1 一个真实快杀理由；但是该层本身的 proxy share 不是系统大头，所以 S1 必须实际证明后续 sparsity propagation 或显著 memory-energy，不能只报 source drop。`proj.conv_res` 的 analog 值也值得同一套 gate 统计，但其幅值分布当前未保留。

M1458 对 FC1/FC2/patch 的大多数记录只保留 ordered statistics，没有逐 token/channel value bitmap；因此 S2/S3 的完整高份额快杀需要轻量增量 capture。M1458 的 `activity-weighted-MAC proxy` 不是 cycle、latency、energy 或 full-network denominator。

## 4. S1｜Analog-Boundary Contribution Gate（ABCG）

### 4.1 机制

目标位置只限两个真实 non-binary 边界：

1. raw event tensor -> `patch_embed.head.conv.0`；
2. patch analog residual -> `patch_embed.proj.conv_res`。

对 output tile `O` 和一个输入 source / kernel offset `j`，离线存：

`beta(j,O) = max_{o in O} |w(o,j)|`。

运行时对准备丢弃的 source 集合 `D` 累积：

`E(O) = sum_{j in D} beta(j,O) * |x(j)|`。

由三角不等式：

`||Delta y(O)||_inf <= E(O)`。

只有 `E(O) + beta(j,O)|x(j)| <= epsilon(O)` 时才不发该 source。`epsilon=0` 时，非零权重下只会删除 `x=0`，严格退化为现有 exact zero-source path。若实现选择更便宜的固定 `theta` comparator 而不维护 runtime debt，则只能称“threshold gate + offline local bound”，不能称 runtime-certified budget。

### 4.2 硬件收费与可省对象

- 收费：每 ingress lane 的 magnitude compare、epsilon / debt state、`beta` metadata read、tile tag、tail / backpressure；若 beta 端口与 weight 端口争用，必须计 stall。
- 可省：被 gate 的 source 后续 weight-row / weight-block fetch、MAC / add 和 psum update。
- 不省：raw / residual source 本身至少要读一次；非空 output tile 的最终 commit 仍需执行。
- 若先读 full weight 才判断，候选失去意义，立即 NO-GO。

### 4.3 现有数据、缺失数据与 48 小时门

当前 M1458 只能证明“存在 non-binary 小量”，不能给 drop histogram。最小增量 capture 不保存完整 tensor，只输出每 sample / layer 的：

- `|x|` histogram / quantiles 与预提交 theta grid 的 source count、source mass；
- 按 output tile 的 `sum beta|x|` debt；
- drop 后第一层 output L1/Linf error、后续 source activity 变化；
- 同一 ep34 forward 的 per-sequence AEE、flow SHA 与 gate counters。

晋级门：

1. `epsilon=0` / `theta=0` bit-exact；local bound violation 为 0。
2. 同资源 cycle `>=1.15x`，或者 cycle 不回退超过 5%、weight bytes 减少 `>=30%`、计 metadata/controller 后 memory-energy proxy 减少 `>=20%`。
3. 全体 `Delta-AEE <=0.02`，任一 sequence `Delta-AEE <=0.03`；报告每序列与密度分层。
4. 若 raw head 单层 source drop 很大但 downstream activity、weight bytes 与周期没有对应变化，降为 energy / negative ablation。
5. metadata + beta 读占被省 weight bytes 的 `>=25%`，或 beta 端口把 compute 变慢超过 5%，一票否决。

论文位置：若过门，只写入 **C2 typed-source frontend 的 optional lossy mode**，不新增第四贡献。Novelty ceiling 约 `3.3/5`，对象差真实，但 threshold pruning prior 很强。

## 5. S2｜Compact Certified Block Skip（CCBS）

### 5.1 与 G11 的边界

S2 不是“G11 换成 block 名字”。它只有同时满足以下三项，才与旧 per-source cumulative B 不同：

1. 一个 metadata read 决定完整 `(source group G, output tile O)` 的 weight fetch；
2. metadata 相对逐 source beta/order 表至少减少 `8x`，且总 metadata 不超过 weight bytes 的 2%；
3. 真实 schedule 观察到整块 fetch / issue / psum-update 都被跳，而不是先扫描完才知道可跳。

若任一不成立，S2 直接归并回 G11 已知路线，不计新 candidate。

### 5.2 数学界

离线每块存：

`M(G,O) = max_{o in O, j in G} |w(o,j)|`。

C2 frontend 已有 source descriptor，可低成本得到：

`A(G) = sum_{j in G} |x(j)|`。

跳过块时：

`||Delta y(O)||_inf <= M(G,O) * A(G)`。

多个块的 debt 相加。`epsilon=0` 只允许 bound 为 0 的 block，退化到 exact path。对 binary FC/Conv，`A(G)` 是 active count，能区分 1 个 active 与 16 个 active；这比原 `||W||1 * ||x||inf` 非空即饱和的粗界更有用。

16x16 INT8 block 若用 16-bit `M` metadata，静态容量约为 weight bytes 的 `0.78%`；这只是容量比，不包含 bank、读能量、pointer 与 debt state。

### 5.3 48 小时门

- 先只用 retained C1 / decoder 做 block `{8x16,16x16,32x16}` local fast-kill；FC/patch 等增量 payload 后再扩。
- 强 baseline 是已有 zero-source skip + 同容量普通 row buffer + 同 K8 / K1x8 端口；禁止用 dense MAC 当唯一分母。
- 同时报告 block skip、weight bytes、issue、psum update、metadata bytes/read、bank conflict、bound debt 与 `Delta-AEE`。
- 晋级阈值沿用 S1；另加 **dynamic witness**：同一 `(G,O)` 在不同 token 因 `A(G)` / remaining budget 不同，至少一次 keep、一次 drop。没有 witness 即退化为 static mask。
- 任一 point 只有 MAC reduction、没有 fetch / psum reduction，或 scan + metadata 后不快于 exact K8，立即 KILL。

论文位置：若过门，作为 **C1/C2 的 block-fetch gate**；不是独立 pruning 理论。Novelty ceiling `3.0/5`，且碰撞风险高。

## 6. S3｜Activity-Bounded Refinement-Plane Elision（ARPE）

### 6.1 机制与界

将 signed integer weight 明确表示为 sign 与 magnitude：

`|w| = 2^b q + r, 0 <= r <= 2^b-1`。

base/high planes 常规读取；低 `b` 位作为 refinement plane 独立存放。若一个 source group `G` 不取 refinement，对 output `o`：

`|Delta y(o)| <= sum_{j in G} r(o,j)|x(j)| <= R(G,O) A(G)`，

其中 `R(G,O)=max r(o,j)`、`A(G)=sum |x(j)|`。当 `b=0` 时无 refinement 被省略，bit-exact。负数不能直接对 two's-complement 低位清零后套用该界；生产语义必须是 sign-magnitude 或明确的 toward-zero quotient / remainder，再与 RNE / saturation miter 对齐。

### 6.2 能省 / 不能省

- 省：refinement SRAM bytes、低位 AND / shift-add、对应 psum add。
- 不省：high/base plane、source descriptor、非空 output 的最终 commit。
- 收费：split SRAM、plane-valid directory、`R` metadata、debt compare、额外地址 / bank、base/refinement merge。

### 6.3 当前阻塞与门

ep34 当前 `hardware_quant_enabled=false`；M1526 已明确 decoder INT8 numeric bridge 未闭。因此本次 FP32 -> temporary INT8 的 `1.0%--1.6%` zero fraction不能用于 S3 结果。必须先有：

1. ep34 明确的 per-layer/per-output quant authority；
2. deterministic PTQ / QAT identity；
3. S40 integer miter、Acc24 range 与 paired AEE；
4. bit-sliced memory layout / port model。

桥闭合后 48 小时快杀报告 `{b=0,1,2,3}`、refinement bytes、active bit terms、metadata、cycle、energy 与 AEE。门与 S1 相同；若只省 multiplier toggles而不省 SRAM fetch，降为局部能量消融。该方向与 Bit Fusion / Pragmatic / Dynamic Stripes prior 强重叠，novelty ceiling `2.8/5`，只适合 C2 precision submode。

## 7. S4｜Phase-Structured T10 Pruning（PSTP）

### 7.1 为什么它不同于 G12

G12 在一个 32-lane issue group 内零散跳 term，结果不对齐 cycle。S4 只允许训练出 **整列 input phase**、**整行 output phase**或固定 aligned coefficient group mask，使 C3 scheduler 真正少发完整 phase group。

对 temporal matrix `H`，删除 coefficient set `D_t` 时：

`|Delta u_t| <= sum_{s in D_t} |H(t,s)| |x_s|`。

这只是 local preactivation bound，不是 AEE，也不能证明 membrane future state exact。全 mask 为 1 是 baseline；任何 phase deletion 都是新 checkpoint 的 lossy mode。

### 7.2 现有静态反证与训练门

ep34 的 45 张 10x10 matrix 无精确零，phase L1 share总体均匀。因此不先写 RTL，只允许小规模训练 / forward screen：

- mask granularity 固定为整 row / column / C3 aligned group；不得回到 rank-3；
- 每层 10-bit / 100-bit static mask、没有 runtime sorter；
- 必须报告 C3 exact service 的真实 cycle reduction，不只报删 coefficient 数；
- 只有 `Delta-AEE<=0.02` 且 C3 same-resource local cycle `>=1.25x`，才讨论把 phase bitmap 加入 C3；否则 KILL。
- 若删 phase 不能让 downstream tensor / source phase 一同消失，只省 ATLIF 内部乘加，按 Amdahl 支撑，不升 headline。

论文位置：C3 training co-design / ablation。prior 明确，novelty ceiling `2.6/5`。

## 8. S5｜Source-Owned Destination N:M（SD-N:M）

### 8.1 机制

传统 N:M 常沿 reduction 维组织。本候选若要对 C2 有对象差，只能沿 **每个 source 对连续 M 个 destination lane** 的权重组训练 / 压缩：例如每 8 个 destination 只存 N 个 weight 和 pattern id。C2 收到一个 source 后，只更新 pattern 指出的 N 个 destination。

`N=M` 是 dense/exact baseline；`N<M` 是重训后的 lossy checkpoint。静态 mask 的 local error满足：

`|Delta y_o| <= sum_{j: w(o,j) pruned} |w(o,j)x_j|`。

但该界通常需读 activation 后统计，不能自动变成 AEE bound。

### 8.2 为什么不能当新贡献

- 未重训时，destination top-N 本质上就是 M324 已指出的 G11 fixed top-m collision。
- N:M 的算法与硬件已有直接 prior；本项目最多主张“source-owned transpose mapping to typed K8”。
- ep34 没有自然 zero block；temporary INT8 zero 也仅约 1%。
- 直接 2:4 / 4:8 pruning 会移除高份额层约 `10%--17%` weight L2，训练不可省。

### 8.3 只允许的筛选

若算法服务器愿意开一个独立稀疏 checkpoint，只允许 `{N:M}={8:8,6:8,4:8}`，并保持训练 / validation / AEE 身份分列。硬件账必须包含 pattern bits、compact weight SRAM、gather / scatter、lane imbalance、final commit 与 dense fallback。门：`Delta-AEE<=0.02` 且 weight bytes `>=30%`、same-resource local cycle `>=1.20x`；否则只作为 N:M baseline，不写 RTL、不算 novelty。

## 9. 不再立项的相邻 idea

| idea | 否决理由 |
|---|---|
| paired FFN expanded-channel prune | M156/M159 已覆盖；ep34 paired norm 与 BN gamma 无 dead-channel 长尾，必须大幅重训，且 channel pruning prior 很强 |
| raw low-event tile reuse / drift | 与 DeltaCNN / 既有 drift 线相邻；N=0 空 tile 已死，当前没有低-event threshold 的多序列 AEE 与 state SRAM 账 |
| near-match product / pattern residual | Phi/PAFT、M306/M307/M579 已占位；不能因换 checkpoint 重开 matcher |
| attention token/head/epsilon prune | attention Amdahl 太小；SpAtten/Bishop prior 强 |
| output-sign early termination | H67 ATLIF future state依赖完整数值；仅证明当前 threshold 不翻转不足以保持下一时刻 state，数学前提不成立 |

## 10. 共用 AEE 与硬件验证协议

所有有损候选必须使用同一个 ep34 checkpoint family、同一 40-sample order 和后续正式多 sequence protocol：

1. baseline 与候选用同一软件、量化、rounding、输入和 output-flow writer；每点保存 config SHA、checkpoint SHA、prediction SHA。
2. 报 baseline AEE、candidate AEE、`Delta-AEE`，同时给每 sequence mean / p95 / max；不以总均值掩盖序列退化。
3. `epsilon=0` / all-mask / `N=M` / `b=0` 必须逐位回到 exact baseline；有损行与 exact C1/C2/C3 分表，禁止相乘。
4. cycle simulator 必须同时计 source / weight / metadata / psum / commit transactions、bank conflict、queue/backpressure 与 tail。
5. 任一“skip”若在读完被省 weight / activation 后才决定，不能记为 fetch saving。
6. 论文若只得到 energy 正结果，可写“weight traffic / energy reduction”，不能写 latency / system acceleration。

## 11. 最小增量 capture 与 48 小时排序

### 0--8 小时：不抢 GPU 的本地准备

1. 冻结 S1 theta / epsilon grid 与 S2 block sizes；写 source-only schema，不启动生产。
2. 用 M1458 retained C1 / decoder payload做 S2 metadata / bound / fetch fast-kill。
3. 量化桥未闭前，S3 只整理 memory layout，不产生数值。

### GPU 空闲后的最小一次增量 capture

- S1：raw ingress 与 `proj.conv_res` 的 magnitude hist、`sum beta|x|`、source / weight-byte counters，不落完整 tensor。
- S2：FC1/FC2/patch 的 per-token group support、nonzero fixed-point code / sign、weight block identity、bank key。
- accuracy：只对过本地 byte/cycle 门的 S1 或 S2 点做 paired forward / AEE。

### 8--48 小时决策

1. S1 先跑；若只动 raw head 且完整 local bytes / cycle 不够，立即降级。
2. S2 必须先过“非 G11 退化”三门，再跑 AEE。
3. S1/S2 最多升级一条 supporting RTL candidate，且不得阻塞 Table-A / power / ep34 rebind。
4. S3 等 M1526；S4/S5 若无算法端明确训练窗口，本轮 DATE 不再推进。

## 12. 最终评分与推荐

| 候选 | novelty /5 | 潜在 significance /5 | 48h 可判性 /5 | 推荐 |
|---|---:|---:|---:|---|
| S1 ABCG | 3.3 | 3.1 | 4.2 | **首测**；真实 non-binary 对象，但先防 Amdahl |
| S2 CCBS | 3.0 | 3.8 | 3.7 | **次测**；覆盖大，但 G11 collision / metadata 风险最高 |
| S3 ARPE | 2.8 | 3.2 | 1.8 | 等 INT8 bridge；不抢当前主线 |
| S4 PSTP | 2.6 | 2.8 | 2.2 | 训练侧低优先级，静态先验不乐观 |
| S5 SD-N:M | 2.4 | 3.4 | 2.0 | 只作 baseline / retraining option，不当新贡献 |

**独立最终判断：** 可以继续探索 sparse / prune / skip，但当前真正值得立刻花 48 小时的只有 S1 与 S2。S1 利用 ep34 新出现的 non-binary boundary，避免了 G7 的二值退化；S2 若能用极小 block metadata 在 fetch 前关掉高份额 FC/Conv weight block，可能给 C1/C2 补 energy / traffic 优势。其余三条要么被量化桥阻塞，要么需要新 checkpoint，要么 prior 太强，只适合作为备选和对照。没有过 cycle/byte/AEE 门之前，不应为任何候选开发 RTL。

## 13. 证据身份

| 对象 | SHA256 |
|---|---|
| ep34 checkpoint | `4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48` |
| M1458 `operator_runtime.json` | `eb0cd40e701361f8acc08d6003680de0ca35626e8e75dcf56827c978899e8a8e` |
| M1458 `execution_trace.json` | `55759fb2e723b4d1a5902a84b95682245b8fde70b21187f1fe1ad9fa08c4ffaa` |
| M1529 review | `8e90e886a5533f168fc497efce16f6995a43988c83dd0c107e0ccde41c22618e` |
| M324 G11 collision review | `f247106c90592d25c53bb41284f4799196a5eb8121dccc873754aa55d14e8cbb` |
| M286 non-attention lossy audit | `669b52dc75362972097949198df34576e536271b2c1930ba21f124ba9455311b` |
| M370 G7 fast-kill | `3c7f759026a548112346a21f6703863b13ac821e5c34cfded92764f2fc74fa0b` |
| protected docs359 | `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4` |

