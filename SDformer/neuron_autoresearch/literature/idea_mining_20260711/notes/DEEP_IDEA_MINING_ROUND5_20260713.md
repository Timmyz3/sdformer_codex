# Round5 深度文献与开源代码审计（2026-07-13）

## 0. 结论先行

本轮在先去重 Round1--Round4、再核对全文公式和官方实现后，只保留 **1 个暂不分配
H 编号的互斥候选**：`SM9-DE`（双证据 Sparsemax9 Match-Code）。它保持 H73 的
`n11/n00`、固定 `Omega9` 和静态 codebook，只把两路 Shiftmax9 替换为精确
Sparsemax9，目的是用输入自适应的精确零支持抑制歧义位移并减少无效 codebook 读。

**不立第二个候选。** Differential Attention 被 H78 G4 的分组分布与有符号静态
codebook 表达包含；SSSA、SEA、SeerAttention、BLADE/ASA、XFeat 和 VecKM normal flow 均至少违反
动态 carrier、固定 9 邻域、one-sided ATLIF 或 attention block 局部重做中的一项。为凑数
给这些路线分配 H 编号会制造不可解释的 full30 开销。

硬件侧形成两项不改变数学语义的建议，不分配 H 编号：

1. 将固定 `Omega9` 的边按 displacement-major/DIA 顺序存取，以 9 路固定偏移 FIFO
   取代通用 CSR/散址；H80 的 destination pass 可复用反向遍历。
2. 用 Stellar 作为独立的 dense/compressed SRAM、space-time transform 和 RTL 生成审计
   框架；它不是 PPA 结果，也不能代替本项目 RTL 综合。

## 1. 审计边界和去重

### 1.1 不可放宽的项目合同

- DSEC 先做；不把 MDR/MVSEC 结果作为候选淘汰依据。
- 12 个 encoder attention block 使用同一公式、同一候选集合和同一量化规则；不得按 stage
  混合 TX/SC 或不同 attention。
- 105 个神经元保持 one-sided binary ATLIF；本轮不改神经元动力学、时间矩阵或阈值定义。
- Q/K 是 one-sided binary events；输出由静态 codebook 重建；不得恢复 native `K`/`V`
  carrier，也不得引入 `gate*K`。
- attention block 允许局部重做，但不得改变 encoder、patch merge、decoder 或既定片上
  主数据流。
- 新算法必须与 H60/H67/H73--H80 在部署公式上非重复，而不是只换论文叙事。

### 1.2 Round1--Round4 已覆盖而本轮不重复展开

现有报告已经审计 EEMFlow/CDC、EMatch、EDCFlow、E-STMFlow、Flow1D、KPA-Flow、
DICL/VCN、LightGlue/Efficient-LoFTR、FlowFormer++、RCM、NAT、DIP、Bishop、LoAS、
RAWAtten、ITA 等。现有算法队列已覆盖：H67 temporal XOR，H73 DE9，H74 MC49，H75
AX17，H76 PC9，H77 LC4，H78 G4，H79 CF10/null 和 H80 DN9 双向归一化。

因此，本轮只保留改变 **9 邻域概率映射本身** 且不扩展 offset/stage/carrier 的新机制。

## 2. 证据与复现台账

下表中的代码均在 2026-07-13 只读拉取到 `/tmp` 后按 commit 审计；未向本仓库复制代码。

| 论文 | 会议 | 全文 | 官方实现（审计 commit / 关键路径） |
|---|---|---|---|
| AdaSplash | ICML 2025 | [PMLR PDF/页面](https://proceedings.mlr.press/v267/goncalves25a.html) | [deep-spin/adasplash](https://github.com/deep-spin/adasplash), `4b64c94fa4cd60d274dafff3178bb96d2f831edb`, `adasplash/adasplash_block_mask.py`, `tests/test_adasplash.py` |
| Differential Transformer | ICLR 2025 Oral | [OpenReview](https://openreview.net/forum?id=OvoCm1gGhN) | [microsoft/unilm/Diff-Transformer](https://github.com/microsoft/unilm/tree/master/Diff-Transformer), `833df7e7832e5064a281131ee64a481afa8e5b95`, `multihead_diffattn.py` |
| Spiking Vision Transformer with Saccadic Attention | ICLR 2025 | [OpenReview](https://openreview.net/forum?id=qzZsz6MuEq) | 截止审计日未找到作者公开官方代码；OpenReview supplementary 只有材料包，以下结论仅基于正文/附录公式 |
| SEA | ICLR 2024 | [ICLR 全文](https://proceedings.iclr.cc/paper_files/paper/2024/file/00d1f03b87a401b1c7957e0cc785d0bc-Paper-Conference.pdf) | [gmlwns2000/sea-attention](https://github.com/gmlwns2000/sea-attention), `5609413b14f480d741cf1edde49691576fc92b50`, `src/models/perlin_attention/`、`src/models/perlin_opt/` |
| SeerAttention | NeurIPS 2025 | [NeurIPS 论文页](https://proceedings.neurips.cc/paper_files/paper/2025/hash/50e9dbc4ab68d94f15261ddc26c8ca2b-Abstract-Conference.html) | [microsoft/SeerAttention](https://github.com/microsoft/SeerAttention), `aba03e3f2caefd0ccd21e576670aa830b748c84e`, `seer_attn/prefill_sparse/attn_gate.py`, `seer_attn/modules/attention_distill.py` |
| BLADE / Adaptive Block-Sparse Attention | ICLR 2026 | [OpenReview](https://openreview.net/forum?id=O9J20MsmRl)、[arXiv 全文](https://arxiv.org/abs/2508.10774) | [ziplab/VIDEO-BLADE](https://github.com/ziplab/VIDEO-BLADE), `c572b9b87b26b4dd60184ff2229c4c98ed8f5a29`, `wanx/train/special_attentions_local/TrainRelated/wanx_blocksparseattn.py` |
| XFeat | CVPR 2024 | [CVF 全文](https://openaccess.thecvf.com/content/CVPR2024/papers/Potje_XFeat_Accelerated_Features_for_Lightweight_Image_Matching_CVPR_2024_paper.pdf) | [verlab/accelerated_features](https://github.com/verlab/accelerated_features), `e92685f57f8318b18725c5c8c0bd28c7fe188d9a`, `modules/model.py`, `modules/xfeat.py` |
| Learning Normal Flow Directly From Events | ICCV 2025 | [CVF 全文](https://openaccess.thecvf.com/content/ICCV2025/html/Yuan_Learning_Normal_Flow_Directly_From_Events_ICCV_2025_paper.html) | [dhyuan99/VecKM_flow](https://github.com/dhyuan99/VecKM_flow), `c5daf836a8f0dfd5709a90a10514eb56d93b0895`, `train/s0_model.py` |
| ASADI | HPCA 2024 | [作者全文](https://www.comp.nus.edu.sg/~tulika/HPCA24.pdf) | 未找到作者公开 RTL/模拟器仓库；仅采用论文 Algorithm 1/2 的 DIA 语义，不移植其 ReRAM claim |
| Token-Picker | DAC 2024 Best Paper | [arXiv 全文](https://arxiv.org/abs/2407.15131) | 未找到作者公开官方实现；仅审计正文 Eq. (4)--(5) 和 Figure 5 scoreboard |
| Stellar | MICRO 2024 | [作者全文](https://people.eecs.berkeley.edu/~ysshao/assets/papers/stellar-micro2024.pdf) | [hngenc/stellar](https://github.com/hngenc/stellar), `59b99332359aa1c04acb2b0c02f3764f72bc6804`, `src/main/scala/stellar/FiberTreeAxis.scala`, `Examples.scala`, `rtl/` |

## 3. 软件线深读

### 3.1 AdaSplash：保留“精确零概率”，不移植长序列 GPU kernel

#### 原机制与代码证据

AdaSplash 的核心不是固定稀疏 mask，而是数据相关的 alpha-entmax：

```text
entmax_alpha(s) = [ (alpha - 1) s - tau * 1 ]_+^(1/(alpha - 1)),
sum_j entmax_alpha(s)_j = 1.                              (paper Eq. 2)
```

`alpha=2` 时退化为 sparsemax。论文 Algorithm 1 用 Halley 更新，越出二分区间时退回
bisection；Algorithm 2 以 block mask 跳过全部低于阈值的 score block。官方实现
`adasplash_block_mask.py:15-36` 逐行计算 `f/f'/f''` 和 Halley 更新，`170-221` 计算
`tau` 边界并只对 `qk > tau` 累加；`tests/test_adasplash.py:24-56` 用
`entmax_bisect(QK/sqrt(d)) @ V` 做 dense reference。

AdaSplash 的 Triton/HBM 优化针对数千到数十万 token；本项目每行只有最多 9 个固定局部
候选，不能照搬其多轮 Halley、block mask 或 FlashAttention kernel。可迁移的唯一机制是
**归一化后产生精确零**。

#### 可迁移公式：`SM9-DE`（唯一候选，暂不编号）

保持 H73 的 event、offset、边界和两路证据。对 token `i`、head `h`、有效位移
`delta in Omega9(i)`：

```text
n11[i,h,delta] = popcount(Q[i,h] & K[opp_t(i), h, i+delta])
n00[i,h,delta] = popcount((1-Q[i,h]) & (1-K[opp_t(i), h, i+delta]))
z11 = quant_score(n11 / D);  z00 = quant_score(n00 / D), D=32

Sparsemax9(z):
  sort z_(1) >= ... >= z_(m), m = |Omega9(i)|
  k*  = max{k: 1 + k*z_(k) > sum_{j<=k} z_(j)}
  tau = (sum_{j<=k*} z_(j) - 1) / k*
  p_delta = max(z_delta - tau, 0)

d[i,h] = concat(Sparsemax9(z11), Sparsemax9(z00))            # 18 lanes
Y[i,h,:] = sum_r d[i,h,r] * C[h,r,:]                        # static 18x32 codebook
```

无效边在排序前屏蔽，`m` 取真实有效边数，因此角点/边缘不会把概率分给 clamp 复制位置。
12 个 block 使用完全相同的 `Omega9`、sparsemax 和量化规则。输出只读静态 `C`，无 native
`K/V` carrier。

#### 额外状态、算子和硬件风险

- 相对 H73：popcount、18 路 score、codebook 容量完全不变；两套 Shiftmax9 替换为两套
  9 项排序/比较、prefix-sum、`k*` 检测、阈值减法和 clamp。
- 动态状态仅为每路 9 个 score、排序索引、9 个 prefix sum 和 `k*`；不增加跨 token、跨
  layer 或跨 timestep 状态。
- `1/k*` 中 `k*=1..9` 可用 9 项常数 reciprocal ROM，但 Q1.7 部署必须定义统一舍入和
  概率和残差归属；否则软件 sparsemax 与 RTL 会有边界分歧。
- 精确零只有在 codebook SRAM/累加器按 `p=0` clock-gate 时才转化为能耗收益。若仍固定
  18 拍全读，只有算法稀疏度而没有硬件收益。
- 9 项排序网比 Shiftmax 的 max/shift/add 控制复杂。该候选只有同时提升 AEE、产生稳定
  中等 support、并经综合证明读/累加节省覆盖排序开销时，才适合 DATE 主 claim。

#### 与 H60/H67/H73--H80 的非重复性

- H60/H67 的 Shiftmax selector 最终仍有 carrier；SM9-DE 只输出静态 codebook。
- H73 与 SM9-DE 的二值证据、`Omega9`、18-lane codebook 完全一致，唯一因变量是
  `Shiftmax9 -> Sparsemax9`，因此是干净的归一化消融。
- H74/H75 改 search support；H76 改 patch evidence；H77 改 contingency coefficients；
  H78 改 channel rank；这些都不产生自适应精确零 support。
- H79 追加 fixed-zero null 类但仍给有效位移分配稠密 Shiftmax 概率；SM9-DE 不增加 null，
  而是在有效位移内产生零。
- H80 乘 row/destination 两路 Shiftmax；SM9-DE 只做 source-row sparse projection，没有
  destination normalization。

#### 最小可证伪实验与 full30 协议

先做不训练的 frozen trace，但 **smoke 不得淘汰**：在同一批 DSEC valid window 上记录
H73 Shiftmax 与 SM9-DE 的 top-1 一致率、每路 active support、边界/内部 support、Q1.7
归一化误差和预计 codebook SRAM skip。trace 只用于发现实现退化。

随后只跑一个独立 full30，不扫 `alpha/temperature/support`：

- 从与 H67--H80 相同的冻结 TTX epoch-2 checkpoint 独立 warm-start；不得从 H73 或其他
  候选串接。
- all12 同一 SM9-DE；binary ATLIF 105、attention 12、candidate 12；原 overlay 210 键
  正确加载，missing 只允许 12 个静态 codebook，注册后 strict reload 为
  `missing=0/unexpected=0`。
- batch 8、workers 8、AMP、cupy、30 epochs、warmup 720、milestones 20/25；保存
  `0,4,9,14,19,24,28,29`；每个保存点标准 valid825。
- 主判据：best valid825 AEE 必须严格优于 H67 `1.4626`，并报告 AAE、total_spikes、
  energy、support histogram、zero codebook-read 比例和 boundary AEE。
- 证伪：若 best AEE 不胜 H67，或 10/12 block 的中位 support 长期为 9（没有稀疏），或
  大量塌缩为 1 且 boundary AEE 恶化，立即停止该线，不扫参数。

**DATE novelty 风险：高。** sparsemax 本身不是新算法，AdaSplash 的主要贡献又是长序列
GPU kernel。SM9-DE 只有在“binary local match + exact sparse projection + static-codebook
skip”的算法/PPA 联合结果成立时，才能作为本项目贡献；否则只应是归一化消融。

### 3.2 Differential Transformer：双分布噪声抵消，被 H78 表达包含

原公式为：

```text
A1 = softmax(Q1 K1^T / sqrt(d));  A2 = softmax(Q2 K2^T / sqrt(d))
lambda = exp(lambda_q1 . lambda_k1) - exp(lambda_q2 . lambda_k2) + lambda_init
Y = (A1 - lambda A2) V
lambda_init(layer) = 0.8 - 0.6 exp(-0.3*layer)
```

官方 `multihead_diffattn.py:52-67` 将 head_dim 减半并注册四个 lambda 向量和 RMSNorm；
`82-117` 生成两套 Q/K softmax、相减、乘动态 V、RMSNorm，并使用 layer-dependent
`lambda_init`。

可迁移形式最多是把 32 lanes 分两组，各做一套局部 binary score：

```text
p1 = Shiftmax9(score(Q_group1,K_group1))
p2 = Shiftmax9(score(Q_group2,K_group2))
Y  = sum_delta (p1_delta - lambda*p2_delta) C_delta.
```

但 H78 已计算 4 个固定 8-lane group 的 Shiftmax9，并用无约束有符号静态 codebook 映射
36 维描述子；取其中两组并令 codebook 权重成比例即可表达上述 readout。原版还需两套
Q/K、动态 V、RMSNorm 和层号相关 lambda，分别违反 carrier、最小局部重做和 all12 参数
同构。故 **不立项、不做 full30**。

### 3.3 SSSA：firing-rate relevance 最终仍是门控动态 V

正文 Eq. (5)--(11) 的关键链为：

```text
H(q,k) = -[p_q log p_k + (1-p_q) log(1-p_k)]
CroAtt(Q,K) = Q' K'^T,  Q'=sum_D Q, K'=sum_D K
SSSA = Theta(Mw (Q'K'^T)L - Vth) * V
SSSA-V2 inference: S[t] = Theta(Q'[t] - alpha^-1 Mw^-1 Vth[t]).
```

论文先删除 silent term，再用 `x` 替代 `log(x)`；V2 又把 `K'^T L` 视为 learned scale。
因此迁移到 `Omega9` 后，位移证据只剩 `(sum Q)(sum K_delta)`，丢失 lane-wise `n11/n00`
对应关系。保留原输出会恢复 `gate*V`；换静态 codebook 后只剩 activity gate，与既有
density/temperature 路线重叠。`Mw` 及其逆阈值还会修改 ATLIF 时间动力学。额外算子为
Q/K firing-count、整数乘法、learned temporal lower-triangular state 或逐时阈值表。故
**违反边界且表达力低于 H73，不立项**。

### 3.4 SEA：为估计全局 mask 付出的代价大于精确计算 9 条边

SEA 先用 Performer/FAVOR+ 编码：

```text
Cperf = FAVOR+(Q,K,Vcat), Vcat=[V_I;V]
Z = MLP([Cperf;V]); A_hat = CNN3(reshape(MLP(Z))) in R^(T x K)
M* = interpolate(grouped_topk(A_hat))
A* = s_prob * softmax((QK^T) masked by M*)
Csea = s_mix*C + (1-s_mix)*Cavg.
```

全文 Sec. 3.1 明确需要 Performer、两个 MLP、三层 2-D CNN、grouped top-k、FlatCSR、
dynamic `s_prob/s_mix` 和全局平均 V；官方仓库也保留 estimated attention、partial mask 和
teacher distillation 路径。对每行只有 9 条边的本项目，精确 9 次 binary popcount 比该
mask estimator 更小。它还需要动态 V、全局池化和稀疏索引控制。故 **不立项**。

### 3.5 SeerAttention：learned block gate 与 9 邻域/禁 gate 条件冲突

SeerAttention 将 Q/K 按 block size `B=64` 池化并预测 block mask：

```text
Qc = RoPE(Wq concat_m Pm(Q)); Kc = RoPE(Wk concat_m Pm(K))
o  = softmax(Qc Kc^T / sqrt(d))
gt = MaxPool2D(softmax(QK^T/sqrt(d)))
Lgate = KL(gt || o)
b_ij = 1[j in TopK(o_i,k)]  or  1[o_ij > threshold].
```

论文 Sec. 3.1--3.3 与官方 `modeling_qwen2_seerattn.py:195-301` 一致：额外 AttnGate、
pooling、linear projection、KL teacher 和 threshold/top-k mask；config 默认 block 64、
hidden 128。原收益来自长上下文 block-sparse FlashAttention。迁移到 9 条固定边会增加一套
动态 gate 网络，且仍需原 attention 生成 teacher；这与用户明确避免 gateK、固定数据流的
要求冲突，也部分重复 H79 的 matchability/null。故 **不立项**。

### 3.6 BLADE/ASA：2026 动态 block mask 在固定 9 边上没有可摊销空间

ICLR 2026 BLADE 的 Adaptive Block-Sparse Attention 先以 Gilbert curve 保局部性重排，
再以 block size `b=128`、每 block 随机采样 `k=16` 个 Q/K 估计重要度。正文
Algorithm 2/3 为：

```text
Qs, Ks = BlockSample(Q, K, b, k)
P_tilde = softmax(Qs Ks^T / sqrt(d))
P_imp = blockwise_max(P_tilde)
M_i = minimal top-m blocks whose cumulative P_imp[i] >= eta
Y = softmax((QK^T + mask(M))/sqrt(d)) V.
```

ASA G 还追加：

```text
K_aug = concat(K, MeanPool_n(K)); V_aug = concat(V, MeanPool_n(V)),
```

并给 global-token score 加固定 `ln(n)`。官方
`wanx_blocksparseattn.py:37-60` 对每 block 生成随机数并 `topk` 采样，`162-223` 排序、
累计 energy 并产生 mask，`344-372` 池化 K/V 并以 log-sum-exp 权重混合输出，与正文一致。

本项目每个 query 只有最多 9 条已知局部边，`b=128` 的估计/裁剪层级不存在。若硬改为
9 边，prober 仍需随机采样、完整 score 子集、sort/cumsum、动态 mask 和二次 attention；
其代价不低于完成 9 个 binary popcount。ASA G 更直接恢复动态 K/V carrier，并增加全局
pooling、token rearrangement 和第二输出支路。其动态 top-p support 表面上与 SM9-DE 都有
零边，但前者是近似预选择加 dense softmax，后者是 9 个精确 score 上定义的概率投影，数学
语义不同。BLADE 的可迁移部分仍违反 carrier/固定数据流，因此 **不立项**。

### 3.7 XFeat：可靠性与 8x8 refinement 不适合嵌入每个 all12 block

XFeat 产生 64-D descriptor `F`、reliability `R` 和 65 类 keypoint logits（64 个 cell
位置加 dustbin）；训练中用 dual-softmax confidence 监督：

```text
S = F1 F2^T
Rbar1=max_row softmax_row(S); Rbar2=max_row softmax_row(S^T)
Lrel=|sigmoid(R1)-Rbar1*Rbar2|+|sigmoid(R2)-Rbar1*Rbar2|.
```

官方 `modules/model.py:95-111` 的 fine matcher 是 `128->512->512->512->512->64` MLP；
`modules/xfeat.py:306-325` 对 64 logits softmax 后解 8x8 offset 并按 confidence 过滤。
移植会为每条局部 pair 引入大 MLP/高分辨率 offset head；reliability/dustbin 语义又与 H79
CF10 null 重叠。它适合前后处理 matcher，不是 12 个 attention block 的同构替代。故
**不立项**。

### 3.8 VecKM normal flow：事件点云分支会重做主数据流

ICCV 2025 论文 Eq. (4) 使用稀疏邻接和复数随机 Fourier features：

```text
J = adjacency_matrix(X)
A = exp(i X R), R fixed Gaussian
G = normalize((J A) ./ A)
normal_flow = MLP(G).
```

官方 `train/s0_model.py` 构造 sparse adjacency，使用 `torch.complex64` VecKM，并在 inference
中默认多角度 ensemble 估计 uncertainty。其额外状态/算子包括原始 event point cloud、
半径邻接、复数 exp/division、sparse matmul、MLP 和可选多次推理。即使 normal-flow 先验
可能帮助 DSEC，它也会绕过 voxel/encoder attention 主路径，而不是局部重做 attention
block。故 **本 DATE 边界内不立项**；不能把其 DSEC normal-flow 结果当作本模型光流
baseline。

## 4. 硬件线深读（只做语义保持优化）

### 4.1 ASADI 的 DIA 思路迁移为固定 offset stream

ASADI 观察一般 sparse attention 的非零具有 diagonal locality，并以 DIA 格式执行
`QK^T`（Algorithm 2）和 `S V`（Algorithm 1）。其论文硬件是 ReRAM in-situ；本项目不应
引用其 `18.6x/2.9x` 数字，也不应声称复现该电路。

本项目的 `2 x 9 x 9` window 更规则。令展平 token：

```text
i(t,y,x) = t*81 + y*9 + x
j = i(1-t, y+dy, x+dx)
j-i = (1-2t)*81 + 9*dy + dx,  (dy,dx) in Omega9.
```

因此每个时间方向是 9 条固定 diagonal；可以按 `delta` 主序流化 Q/K 和 codebook：

1. 9 路固定延迟/FIFO 对齐相反时间的 K，边界只携带 valid bit。
2. popcount/score/normalizer 输出沿相同 `delta` bank 写入，不存通用 row/column index。
3. H73/H76/H77/H78/H79/SM9-DE 的静态 codebook 地址由 `head,delta,lane` 直接生成。
4. H80 destination normalization 以相反方向重放 9 个 bank；无需构造 scatter CSR。

额外存储是 9 路短 FIFO、valid shift-register 和必要的 9-score row buffer；相对 CSR 删除
row pointer/column index。控制是固定 9 状态循环，角点/边缘由 valid mask 精确处理。

**可审计 claim 仅限：** 与 dense reference 逐位等价、索引 SRAM 字节数、实际 FIFO
深度、SRAM bank conflict 和测得 cycle。未综合前不得宣称 ASADI 论文的加速/能耗比例。
它与 H60--H80 均不重复，因为不改 score、normalization 或 codebook 数学语义。

### 4.2 Token-Picker：不迁移 predictor，只吸收“先证安全再跳读”的审计原则

Token-Picker Eq. (4) 将二补数 K 按 MSB bit-chunk 读取，并为未读 bit 构造 dot-product
上下界。Eq. (5) 用：

```text
p_i <= exp(s_i,max^b) / sum_(j in subset) exp(s_j,min^b) = p_i''.
```

仅当该保守上界低于 threshold 才停止读取后续 K chunks/V；Figure 5 的 scoreboard 支持
HBM on-demand/out-of-order 请求。收益来源是 autoregressive LLM 长 KV cache 的片外带宽。

在本项目中每行只有 9 个 K、每个 32 bit binary，且它们已由固定局部 FIFO 提供；计算
上界需要 Q 正负分类、partial popcount、阈值/归一化估计和 scoreboard，通常比完成剩余
popcount 更贵。若用近似阈值还会改变数学语义并重复 Bishop/ECP 类 early prune。因此
**不迁移 predictor，也不建立硬件候选**。

可保留的审计原则是：任何未来 clock-gating 只能在证明“被跳过项对已定义的定点输出必为
零”后触发。SM9-DE 的 `p=0` 满足该条件；仅凭低概率近似不满足。

### 4.3 Stellar：独立 RTL/数据流审计工具，不是 attention 算法

Stellar 将 accelerator 分为 functionality、space-time dataflow、sparse data structure、
load balancing 和 private memory 五个独立轴；论文 Sec. III/IV 描述从 Halide-like 公式和
仿射 space-time transform 到 Chisel/Verilog。官方代码 `FiberTreeAxis.scala` 明确支持
Dense、Compressed、LinkedList 和 Bitvector metadata，`Examples.scala` 含 SRAM、
compressed axes 和 load-balancer 实例，`rtl/` 负责硬件生成。

对本项目可做的独立审计：

- 用 dense `[token,delta]` 或 compressed zero-support 两种 SRAM 描述同一 codebook MAC；
- 用仿射 transform 比较 token-major 与 displacement-major 的 PE/FIFO/端口需求；
- 固定相同位宽、bank、吞吐目标后生成 RTL，比较面积、周期和 SRAM traffic。

限制也必须写清：Stellar 原生 space-time transform 仅支持 affine mapping，论文明确指出
tree reduction 等递归结构需手写；Shiftmax/Sparsemax 的 max、sort、prefix/reduction 不能
假设由框架自动生成最优 RTL。故 Stellar 结果只用于布局和存储的独立复核，最终 claim 仍需
本项目 RTL、定点 golden test、综合和功耗流程。

## 5. 最终立项矩阵

| 优先级 | 路线 | all12/ATLIF/carrier 合规 | 与 H73--H80 独立 | full30 | 结论 |
|---:|---|---|---|---|---|
| 1 | **SM9-DE**：H73 双证据 + exact Sparsemax9 + static codebook | 是 | 是，只改变概率投影 | **值得 1 次** | 暂不分配 H 编号；先完成公式/RTL定点定义再注册 |
| - | Differential local maps | 可强行改写 | 否，被 H78 codebook 包含 | 否 | 不立项 |
| - | SSSA/SSSA-V2 | 否，动态 V/阈值动力学 | 与 density gate 重叠 | 否 | 不立项 |
| - | SEA/SeerAttention | 否，动态 gate/teacher/carrier | 与 mask/null 路线部分重叠 | 否 | 不立项 |
| - | BLADE/ASA | 否，随机 block prober、动态 mask 和 K/V global branch | 与精确 9-edge 投影不等价 | 否 | 固定 9 边无可摊销收益 |
| - | XFeat reliability/refinement | 否，大 MLP/外部 matcher | 与 H79 null 重叠 | 否 | 不立项 |
| - | VecKM normal-flow branch | 否，重做输入与主数据流 | 不属于 attention 局部替换 | 否 | 本 DATE 边界外 |

本轮没有证据支持第二个互斥候选。若 SM9-DE 不胜 H67，正确结论是保留 H67 或等待
H73--H80 的直接 full30 结果，而不是组合 sparsemax+null、sparsemax+DN9 或继续扫 support。

## 6. 论文表述和禁止性结论

- 可以写：`We evaluated exact adaptive support projection as an isolated alternative to Shiftmax
  under identical binary evidence and static-codebook reconstruction.`
- 只有有 RTL 计数后才能写：zero probability suppresses codebook SRAM reads/MAC toggles。
- 不可写：复现 AdaSplash、ASADI、Token-Picker 或 Stellar 的论文性能；本轮只是有边界的
  公式/数据流迁移审计。
- 不可把 sparsemax 的一般数学机制声明为本论文原创。潜在 novelty 是其与 binary local
  matching、all12 static codebook 和可跳读 SRAM 的共同设计及实测结果。
- 不组合任何 Round5 机制与 H79 null 或 H80 destination gate，直到各自独立 full30；否则
  无法归因且会破坏“最多两个互斥候选”的实验纪律。
