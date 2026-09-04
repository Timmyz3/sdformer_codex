# M503｜threshold-carry / late-scale 断点审计（只读）

日期：2026-08-27  
范围：H67/Motion ep35；M32 → M35/M41 → M163/M167 → M234/M480 → C1/M473 与 C2/M496 证据链  
裁决：**GO_OFFLINE_EXACT_MITER_ONLY**  
禁止：本评审不准入新 RTL、VCS、DC/PT、系统倍速、能量、PPA 或 DATE headline；不复活 M32 的 2.77× sensitivity。

## 1. 结论先行

M32 中真正的大机会不是“省掉一个接近 1 的 threshold 乘法”，而是把 ATLIF 的 `{0, theta}` 输入改写为 binary support 上的 selected-weight/event accumulation。该主体已经被后续 C1/M473 和 C2/FC2 路线吸收或重新公平化：

- 四层 bottleneck Conv 的 binary/product 执行已进入 M473；它在 fused 假设下是 1.9436×，但 unfused 同步点只有 1.0147×，最终 CPU DSE 为 NO-GO，不能回填 M32 的 2.77×。
- FC2 是另一组边界。M231 证明 H67 ep35 的 12 个 FFN `sn2` threshold 都是 bit-exact 1.0，因此 C2/M496 的 binary FC2 不需要运行时 amplitude multiplier；但这 12 个 threshold 不是 M32 的十个非单位 threshold。
- 显式 late-scale 本身已有 M35：10-descriptor、8 result/cycle 的 exact raw-Q24 standalone block。M35 的 1.7821× 只是相对 M33 的 peak checkpoint-specialization result-rate/area density，不是 operator 或 system speedup。

剩余未闭合的是“完整数值边界”：M35 的 raw Q24 product 之后如何做 RNE、overflow/saturation、bias，并与 current-batch BN 的 moment/epsilon/affine 顺序一致。这个缝真实存在，但证据尚未说明需要一块新的 RTL；很可能被现有 M234 coefficient descriptor 或静态 scale descriptor 吸收。因此当前只准许做离线、硬件顺序、零失配 miter。

严格裁决：

| 候选裁决 | 本轮结果 | 原因 |
|---|---:|---|
| `NO_GO` | 否 | 七个 BN-followed consumer 仍有一个可验证的 exact epsilon/RNE 收口问题，不能未经 miter 直接宣告已吸收。 |
| `GO_OFFLINE_EXACT_MITER_ONLY` | **是** | 数学上可 hoist，但 finite-Q/RNE/current-batch-BN 顺序未闭合；收益上限小，先证明是否为 descriptor-only。 |
| `GO_RTL` | 否 | M35 已实现通用 exact raw scale；没有 matched numeric-boundary 证据证明新 sidecar 比 M35 更省，且系统上限不足以支撑新 contribution。 |

## 2. M32 的十个 consumer 到底去了哪里

M32 冻结了 10 个 immediate-ATLIF consumer、每样本 30,456,000 个输出、105,888,197 个旧 activity-cycle 机会。拓扑审计将它们分为两类：

| 类别 | 算子数 | 输出/样本 | 后继 | 正确处理方向 | 当前状态 |
|---|---:|---:|---|---|---|
| patch merging reduction | 1 | 2,304,000 | current-batch norm | binary Acc 后 hoist 正尺度，并以 `eps/theta^2` 修正 BN；需硬件顺序 miter | 未闭合 |
| stage-3 FC1 | 2 | 18,432,000 | BN1 → ATLIF | 同上；需同时覆盖 BN 输出与下游 ATLIF 决策 | 未闭合 |
| bottleneck Conv3x3 | 4 | 9,216,000 | BN1/BN2；Conv2 后有 residual | binary product 部分被 C1/M473 吸收；scale/RNE/BN/residual 边界仍在外 | 部分吸收 |
| prediction Conv | 3 | 504,000 | 无 BN，带 bias | 将静态正尺度折入 weight/dequant descriptor；必须固定 bias/RNE 顺序 | 未做三头 exact miter，但不构成新乘法器 |

代码拓扑证据：

- `MS_Spiking_Mlp` 是 `sn1 → fc1 → bn1 → sn2 → fc2 → bn2`。
- `MS_SpikingPatchMerging` 是 `sn → reduction → norm`。
- `MS_ResBlock` 是 `sn1 → conv1 → norm1 → sn2 → conv2 → norm2 → residual ADD`。
- `MS_SpikingPredLayer` 是 `sn → conv`，没有 BN；其 Conv 使用 bias。
- DSEC `no_running` 协议把全部 `_BatchNorm` 设置为 `track_running_stats=False` 并清空 running mean/variance，因此不能用静态 BN fold 冒充生产语义。

七个 BN-followed consumer 占 M32 输出的 29,952,000 / 30,456,000，即 98.35%。三头 prediction 只占 1.65%，即使静态 fold 成功也只能作为数值收口。

## 3. 后续证据链：吸收了什么，没吸收什么

| 里程碑 | 已证明 | 未证明 | 对 M503 的裁定 |
|---|---|---|---|
| M32 r3 | frozen H67 ep35 S10 的十组 runtime tensor identity；实数域 `W(theta*b)+bias=theta*(W*b)+bias`；Q24 constructive identity | Q-format、RNE、saturation、bias、SRAM、可执行 schedule | 只保留机会来源，不保留 2.77× |
| M35 r7 | 10 个 frozen descriptor 的 exact `Acc*(2^24-delta)`；8 result/cycle；3 ns logic-only DC/STA/Formality | RNE、bias、BN、SRAM、full schedule/energy | 已有显式 scale RTL；新 RTL 必须严格优于它 |
| M39/M42 | 把 M35 late-scale 放进条件 cycle sensitivity | executable/full numeric schedule | 只能用于 Amdahl 上界 |
| M41 r2 | 四个 Conv 的 92.16M INT8×binary accumulator 全量 0 mismatch；RNE correction 全量复核 | 其他六个 M32 consumer；BN/残差；general sidecar RTL | 证明“theta 近 1”不等于“theta 可直接删” |
| M163r2/M167 | 共享 product kernel 与 ATLIF amplitude transport；M167 提出 `eps/theta^2` 选项 | front-to-back 数值桥、全 BN population、requant、FC2、accuracy | 未吸收 M32 exact gap；属于 rank3/PAFT 旁线 |
| M234 | current-batch BN coefficient DSE，variance+epsilon 入口为 UQ6.16；64-LUT+1 Newton 为 selected screen | moment finalizer、完整 affine/ATLIF、exact fixed-point production order | 可能 descriptor-only 吸收 epsilon 修正，须 miter |
| M480 | 强公平 BN baseline：保留 global barrier/raw replay，只消除 normalized intermediate；Q24@64B/cyc local 1.4999× | fixed-point accuracy、runtime affine RTL、macro/power | M503 必须接在 fused replay baseline 上，不能另建弱基线 |
| M473 r3 | 四 Conv integer product/reconstruction exact；fused 1.9436× opportunity | physical scratch/CAM、numeric scale/RNE/BN；unfused 仅 1.0147× | 吸收 M32 的大 compute 机制，但不吸收 late numeric boundary |
| M231/C2/M496 | 12 个 FFN `sn2` threshold 全为 exact 1.0；FC2 K8/K1 matched-top 合同 | BN2/SN2/requant/full FFN；M32 非单位 threshold | FC2 无 amplitude gap，但不可外推到 M32 十算子 |

## 4. current-batch BN 的 exact hoist 条件

令二值支持上的整数卷积/线性输出为 `A`，ATLIF 的正幅值为 `theta>0`，基线 BN 输入为 `z=theta*A`。实数域：

```text
mu_z  = theta * mu_A
var_z = theta^2 * var_A

BN(z; eps)
= gamma * (z - mu_z) / sqrt(var_z + eps) + beta
= gamma * (A - mu_A) / sqrt(var_A + eps/theta^2) + beta
```

所以只有同时满足以下条件，才可 exact 删除显式 `theta`：

1. `theta` 为全 population 共用的正标量；M32 十个值均为正，满足。
2. current-batch BN 使用 `eps' = eps/theta^2`；把 `eps` 原样保留不是严格等价。
3. moment 的 population、biased/unbiased variance 定义、归约顺序和 affine 顺序一致。
4. finite-Q 路径必须匹配。若基线先做 `q=RNE(theta*A)` 再统计 moments，则直接对未量化 `A` 做上述实数变换通常不再 bit-exact。
5. bias、saturation、残差加法与 ATLIF threshold comparison 的位置不得交换。

M32 的最大 `|1-theta|` 是 3.5047531e-5。若 `eps=1e-5`，最大 `eps/theta^2-eps` 仅 7.0099e-10；在 M234 的 UQ6.16 入口上，十个 `eps'` 与原 `eps` 都量化为 raw 1。这个结果说明“epsilon descriptor 可能零成本”，但**不能单独证明端到端等价**：RNE 发生在 moment 之前还是之后仍决定结果。

M41 给出了最直接的反例证据。在四层 Conv、92,160,000 个值上，前三层 `RNE(theta*A)=A`；最后一层有 1,348 个 ±1 correction，虽然 bypass 率为 99.994149%，但 exact 路径不能删除它们。冻结 trace 的 correction 只有 ±1；然而按 analytic `sum(abs(weight))` 上界与 delta=588，`|A*delta|/2^24` 可到 6.6854，所以 general exact sidecar 必须覆盖至少 `[-7,+7]`，不能把冻结 trace 的 ±1 当作硬件范围证明。

## 5. 性能上限：为什么不能复活 2.77×

M32 的 motion `balanced_radix20` 2.7693× 是一个把 105.9M 旧 consumer activity 全部替换成 event accumulation、再借用 frontend/control 的 optimistic sensitivity。它不是 late-scale removal，也不是 executable schedule。

按同一 620,868,243-cycle envelope：

| 删除对象 | late-scale cycles | envelope 占比 | 理想完全删除上限 |
|---|---:|---:|---:|
| M32 十算子，M35 8 result/cycle | 3,807,000 | 0.6132% | **1.00617×** |
| 四个 bottleneck Conv，M35 8 result/cycle | 1,152,000 | 0.1855% | **1.00186×** |
| M32 十算子，旧 M33/M32 4 result/cycle | 7,614,000 | 1.2263% | **1.01242×** |

因此：

- threshold hoist 即使零面积、零周期，也不能成为 DATE headline speedup。
- 它的合理价值是让 C1/C2 的 exact 数值收口更干净，减少一个 13,701.6 um² logic-only scale block或其动态能耗。
- 任何新 sidecar 若不能在 matched numeric boundary 上替换 M35 并显著降低面积/能量，就应 NO-GO。

## 6. 与 Prosperity / Phi / FireFly-T / LoAS 的差异

| 工作 | 核心机制 | 与 threshold-hoist 的关系 |
|---|---|---|
| [Prosperity](https://arxiv.org/abs/2503.03379) | binary spike 的 product sparsity 与组合相似复用 | C1/M473 是更接近的对标；BN epsilon hoist 不是 product sparsity |
| [Phi](https://arxiv.org/abs/2505.10909) | predefined weight patterns、residual sparsity、PAFT | M163/M167/rank3 更接近；M503 不依赖 PAFT，也不是 pattern pruning |
| [FireFly-T](https://arxiv.org/abs/2505.12771) | 多非零提取、bank dispatch、乱序/load balance 与 binary attention | C2 source decode 更接近；M503 是数值顺序收口 |
| [LoAS](https://arxiv.org/abs/2407.14073) | fully temporal-parallel dual-sparse SpMSpM、压缩与 inner join | 与 C1/C2 sparse execution 同族；不覆盖 current-batch BN scale algebra |

M503 的差异是 H67 analog ATLIF amplitude 与 current-batch BN 之间的 exact scalar-hoist；这是有用的 H67-native glue，但不是足以单列贡献的全新 sparse engine。论文里最多作为“binary execution 仍保持 analog threshold exactness”的子机制/数值 lemma，不能与上述工作按独立加速器 headline 对打。

## 7. 唯一准许的下一步：offline exact miter 合同

不得先写 RTL。离线 miter 必须同时包含如下硬门，任一失败即保持 `NO_GO_RTL`：

### G0｜身份与语义

- 固定 H67/Motion ep35 checkpoint SHA、DSEC S10 sample manifest、`no_running` BN policy 和十个 M32 state-dict key。
- 固定七个 BN-followed consumer 与三个 no-BN prediction consumer，禁止把 12 个 exact-one FC2 threshold 混入。
- 固定生产 PyTorch 顺序和候选 fixed-point 顺序；每一级记录 dtype、Q format、signedness、RNE tie-even、saturation/overflow、bias 与 residual 位置。

### G1｜七个 BN-followed consumer

- 基线：binary Acc → explicit M35 Q24 scale → authoritative RNE/saturation → current-batch moment → affine → 后继 ATLIF/residual。
- 候选：binary Acc → hoisted scale/修正 → `eps/theta^2` 或等价 coefficient descriptor → 同一后继顺序。
- 对 S10 全量值在以下边界全部 0 mismatch：post-scale code、mean/variance、BN affine code、后继 ATLIF event/amplitude；Conv2 还必须在 residual commit 后 0 mismatch。
- 必须补齐 downsample 与两层 FC1 的 accumulator trace；当前 M41 只覆盖四个 Conv。

### G2｜三个 prediction consumer

- 比较 explicit theta path 与静态 weight/dequant-scale fold；三头全部覆盖，不得用 M61 单头代替。
- bias 必须保持未缩放/缩放的正确代数位置；最终 INT output 或浮点 dequant output按冻结生产边界 0 mismatch。

### G3｜解析范围与攻击向量

- 对每层证明 accumulator、`A*delta`、correction、moment sum/sumsq、affine 和 residual 不溢出。
- mutation/rail 至少覆盖：`theta=1`、最小/最大 frozen theta、正负 accumulator extrema、half-way tie-even、zero/near-zero variance、epsilon quantization boundary、saturation rails、bias extrema、空 population 协议攻击。
- correction sidecar 若被提出，范围必须来自解析上界；冻结 S10 的 ±1 不能替代 general `[-7,+7]` 覆盖。

### G4｜报告边界

- miter 的零失配只准入“候选 fixed-point 相对 explicit-scale fixed-point 的变换等价”；它不自动准入相对原始 FP32 模型的 valid825 accuracy。
- 结果必须分别报告七个 BN consumer、三个 prediction consumer和四层 Conv 子集，不得只报 aggregate。
- 若 UQ6.16 下 `eps'` 与 `eps` raw 相同且全部 miter 通过，应裁为 **descriptor/schedule absorption**，不开发新 RTL。

## 8. 只有出现以下结果，才允许从 miter 升格 GO_RTL

升格对象只能是一个小型 exact correction/descriptor sidecar，不得再造一个通用 multiplier。全部硬门：

1. 现有 M35 + M234 descriptor 无法在同一 hardware-order 上表达 exact path，且 miter 明确定位到需要在线 correction。
2. sidecar 对七个 BN consumer与三个 prediction consumer的解析输入域均 exact；不能只对 S10 exception bitmap exact。
3. 96-lane/目标吞吐下零 backpressure，C1/C2 accepted-source throughput 不下降。
4. 与包含相同 RNE、bias、BN descriptor 接口的 M35 baseline 做 matched Synopsys 比较：新增 sidecar cell area ≤ 3,425.4 um²（M35 13,701.6 um² 的 25%），并真正移除 M35 而不是外置它。
5. 3 ns logic-only setup ≥ 0.05 ns、hold ≥ 0.01 ns；随后才允许 Formality与 matched SAIF/PTPX。
6. matched selected-boundary 动态能量至少降低 20%，面积至少降低 50%，无吞吐损失；否则 `NO_GO_RTL`。
7. 即使全部通过，论文中只作为 C1/C2 exactness/energy 子机制；不得声称独立系统倍速。其 8-lane 理想系统上限仍只有 1.00617×。

## 9. 论文收口建议

可写的一句话：

> H67 的非零 ATLIF 输出携带 checkpoint-bound 正幅值而非纯 bit。我们在 binary selected-weight accumulation 后保持显式 exact-scale reference，并研究将公共正尺度穿过 current-batch BN；该变换仅在 epsilon 重参数化与固定点舍入顺序闭合时成立。

若 miter 通过且 descriptor-only：把它写成 C1/C2 的 exactness lemma 和消融（“删除显式 scale block，0 mismatch”），不列第四个 contribution。

若 miter 需要 sidecar且过全部物理门：写成“rare exact amplitude correction + epsilon-aware BN descriptor”，报告局部面积/能量；仍不报告 2.77×。

若 miter 失败：保留 M35 作为 explicit exact fallback，论文中如实说明 binary product engine 后仍有 checkpoint-scale numeric tail。

## 10. 数据质量与不可外推项

- 本审计只读；未运行 VCS/DC/PT/GPU，未修改生产 RTL、合同、结果或 docs/359。
- docs/359 SHA 复核仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
- M41 的 92.16M population 只覆盖四个 bottleneck Conv，不能替代七个 BN consumer 的全量 miter。
- M234 是 coefficient-only DSE，M480 是 exact schedule DSE；两者都不是完整 fixed-point BN RTL。
- M496 在本审计截点仍是 locked-before-execution contract；只使用其 scope/公平基线定义，不把合同目标当结果。
- Prosperity/Phi/FireFly-T/LoAS 的公开倍率与本项目 M503 不同 workload/模块/资源口径，只用于机制定位，不进入性能表横比。

## 11. 复核源（SHA256）

完整清单见同目录 `source_inventory.sha256`。核心文件包括 M32、M35 r3/r7、M41r2、M163r2、M167、M231、M234、M473、M480、M496 contract，三个生产源码文件以及 docs/359。
