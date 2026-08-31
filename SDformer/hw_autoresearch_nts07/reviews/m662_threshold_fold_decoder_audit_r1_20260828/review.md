# M662｜H67 threshold-folded decoder / scale-coded binary 独立审计与实测计划

## 裁决

`CONDITIONAL_GO_D1_THRESHOLD_FOLD_MEASUREMENT__NO_PERFORMANCE_OR_RTL_ADMISSION`。

M649/M658 对“D1 不是 exact `{0,1}`”的 NO-GO 仍然正确；但把所有 finite-nonbinary 项直接归类为任意 analog 值并据此终止 decoder 稀疏线，存在一个尚未排除的假阴性。冻结源码与 checkpoint 的静态证据支持更窄的假设：D1 的 770 通道在进入 `ConvTranspose2d` 前已经共同通过一个 scalar official-ATLIF，因而候选形式是 `x = theta * mask`，其中 `mask` 为 `{0,1}`，`theta` 是 checkpoint 静态 FP32 标量。当前证据足以授权一次新的、无 payload 的数值等价审计；不足以授权 capture、cycle、speedup、RTL、VCS/DC/PT、energy/PPA、system 或 DATE headline。

## 1. 冻结执行链：为什么 D1 值得复活

冻结 H67 配置使用 `MS_SpikingformerFlowNet_en4`、`use_upsample_conv: false`，因此 decoder 是 `MS_SpikingTransposeDecoderLayer`。其真实顺序是：

1. decoder forward 先执行 `x = self.sn(x)`；
2. 再执行 `x = self.deconv(x)`；
3. D1 的 hook 名为 `sttmultires_unet.decoders.1.deconv.0`，所以 M649 看到的是 **ATLIF 之后** 的张量，而不是 `skip_concat` 的原始 flow/feature 拼接张量。

配置的 `all_non_qk_binary_atlif` 通过 `path_selection: all_non_qk` 覆盖 decoder 的 `Spiking_neuron` wrapper。`OfficialATLIFSurrogate.forward` 的定义是：

```text
mask = float(input >= theta)
output = mask * theta
```

冻结 checkpoint 的 CPU 精确构建（同一 config/checkpoint，load audit `missing=0/unexpected=0`）观察到：

| 项 | 静态观察 |
|---|---|
| D1 wrapper | `sttmultires_unet.decoders.1.sn` |
| D1 neuron | `ATLIFTernaryPSN`, `output_mode=binary`, `threshold_mode=official_atlif`, `center_mode=zero`, `T=10` |
| D1 threshold | shape `[]`, numel `1`, FP32，值 `0.9999954104423523`，little-endian bytes `b3ff7f3f` |
| D1 consumer | `ConvTranspose2d`, weight layout `[Cin=770, Cout=192, 3, 3]`，bias `null` |
| decoder thresholds | D0=`1.0`，D1=`0.9999954104423523`，D2=`1.0`，D3=`1.0` |
| 全模型 ATLIF | 105/105 scalar；105/105 official binary；94 个 threshold exact `1.0`，11 个 non-unit；分布为 attention 60、encoder-other 29、resblock 8、decoder 4、prediction 4 |

`model.eval()` 并不会把 `thresh.requires_grad` 改为 false；“inference 静态”成立的原因是 forward 只读 `thresh`，阈值更新位于外部训练流程，冻结推理没有 optimizer step 或 `threshold_update` 调用。实测门仍必须逐 sample 证明 threshold bit pattern 没有变化，不能仅凭源码推理宣布通过。

## 2. M649 数据如何重新解释

M658 独立重算确认 D1 的 10 个样本共 `92,400,000` 项：

| 类别 | 项数 | 比例 |
|---|---:|---:|
| numerical zero | 75,314,174 | 81.508846% |
| exact one | 0 | 0% |
| finite nonbinary | 17,085,826 | 18.491154% |
| nonfinite | 0 | 0% |

M649 的安全统计已在抽查通道上观察到 nonzero 最大值恒为 `0.9999954104423523`，与 D1 checkpoint threshold 相同；而 770/770 通道都出现 nonbinary，恰好符合“整个 concat 先过同一 ATLIF”的执行顺序。可是 M649 没有对 **全张量** 做 `value_bits in {+0, theta_bits}` 的比较，也没有保存全体 nonzero unique-bit 计数。因此目前只能称为 `SCB_CANDIDATE`，不能称为 exact scale-coded binary 已通过。

M649/M658 的原结论无须撤销：它们测试的是 `{0,1}` 和 first2/suffix typed split；本机制提出的是一个新的、严格更窄的 `{0,theta}` 表示问题。

## 3. 无损 threshold fold 的代数与 ConvTranspose 数值顺序

PyTorch `ConvTranspose2d` 的 weight shape 为 `[Cin, Cout/groups, kH, kW]`。对 D1 的 scalar threshold，实数域中：

```text
x[c,q] = theta * mask[c,q]
y[o,p] = sum_(c,k) x[c,q(p,k)] * W[c,o,k]
       = sum_(c,k) mask[c,q(p,k)] * (theta * W[c,o,k])
```

因此可离线计算 FP32 `W_fold = fl32(theta * W)`，运行时只传/读 mask 并累加 `W_fold`。D1 bias 为 null；即使有 bias，也应只折 weight、原样保留 bias。

在同一 FP32 kernel、同一 reduction order 下，bitwise equality 是合理预期：`mask=1` 时两边的单项乘积都落到 `fl32(theta*W)`，`mask=0` 时都为 signed zero 候选。但 backend algorithm、signed-zero、TF32 或重排仍可能破坏输出 byte hash，所以**必须实测，不能用实数代数替代数值准入**。

## 4. 下一次实测的 fail-closed 门（先审计，后 simulator）

新合同只审计 D1，固定 ep35 checkpoint、同一 10 个样本、同一 model/config/source identity；不保存 raw activation。

### G0｜身份和执行顺序

- exact checkpoint load：`missing=0/unexpected=0/overlay_missing=0/overlay_unexpected=0`；
- 固定 D1 wrapper/neuron/deconv 路径、类、weight shape/bytes/hash、bias null、stride/padding/output-padding；
- 固定 Python/PyTorch/CUDA/cuDNN、host、GPU UUID、driver 和 deterministic/TF32 配置；
- pre-hook 必须绑定 `decoders.1.deconv.0`，同时从 `decoders.1.sn.spiking_neuron` 读取 threshold。

### G1｜全元素 exact `{0,theta}`

- threshold 必须是 finite positive FP32 scalar，shape `[]`、numel `1`；
- sample 前、每次 forward 后、最终的 threshold bytes 必须全部相同；候选 frozen bits 为 little-endian `b3ff7f3f`；
- 对 10 个样本全部 `92,400,000` 项做 bit-level 分类：`+0`、`-0`、`theta_bits`、other、nonfinite；主门要求 `other=0`、`nonfinite=0`，并单列 signed-zero；
- 从 `theta_bits` 生成 mask；每样本及总 mask popcount 必须与独立 M649 nonzero 账一致，总数候选为 `17,085,826`；
- 保存的只能是 count/hash/receipt，不得保存 activation payload。

### G2｜FP32 folded-weight 与输出等价

- `W_fold = (W.float() * theta.float()).contiguous()`，记录 raw/folded weight byte hash；folded hash 跨样本不变；
- 原路径：`conv_transpose2d(theta*mask, W, None, stride=2, padding=1, output_padding=1)`；
- 折叠路径：`conv_transpose2d(mask.float(), W_fold, None, same geometry)`；
- 关闭 TF32、启用 deterministic algorithms，并显式同步；逐样本记录两路 output byte hash、bit-difference count、max-abs、max-ULP；
- **准入要求是 10/10 `torch.equal`、byte hash 相同、bit-difference=0、max-abs=0。** 如果只满足 tolerance，则 FP32 exact fold NO-GO；误差可以作为后续有损/量化点，不能写 lossless。
- 可选强化：把原/折叠 deconv 输出分别送过同一 BN/后续链，验证 prediction hash；它不能替代 deconv 主门。

### G2a｜必须拆开的两个 admission bit

`G1 exact_SCB_representation` 与 `G2 exact_FP32_weight_fold` 不是同一个结论：

- G1 通过只证明可用 `mask + one finite-positive theta sidecar` 无损表示 D1 input；
- G2 通过才证明当前 PyTorch FP32 ConvTranspose 可以把 theta bake 进 weight 且 output byte-identical；
- 若 G1 通过、G2 失败，仍可保留 **exact representation + sidecar**：硬件传 mask，并在 consumer 端重构原 FP32 amplitude，语义无损但不保证加速；
- 另一个候选是先计算 `ConvTranspose(mask,W)` 再做一次 output-scale。它在实数域等价，但 FP32 reduction/rounding 仍可能不同，必须另做 miter；未通过不得命名为 admitted exact fold；
- folded-weight、output-scale、per-event sidecar 三条 route 必须分别标注，不能用 G1 的成功替 G2 背书。

finite-positive 是 G1 的硬门，不只是健康检查：`theta=0` 时 active 与 zero 的数值表示坍缩、mask 无法从 input 唯一恢复；nonfinite 不能进入标准算术；负 theta 虽可另建 signed-scale 协议，却不符合本次 official positive-threshold 合同。

信息量上限可用于解释、不能当流量结果：S10 D1 raw FP32 input 是 `92.4M * 32 = 2.9568 Gbit = 369.6 MB`，完整 bitmap 加一个 FP32 theta 是 `92.4 Mbit + 32 bit = 11.550004 MB`，纯表示约 `32x` 缩小。若某 amplitude-aware event baseline 为每个 active event 携带 32-bit amplitude，则静态 sidecar 对 `17,085,826` 个 active event 最多消掉 `68,343,304 B` amplitude 字段。两者都忽略 address、descriptor、bitmap padding、weight 和 SRAM transaction，故只能标 `representation_information_bound`，不能写 DRAM saving 或 speedup。

### G3｜固定点部署边界

FP32 fold 通过只证明软件数值桥。若 paper hardware 使用 INT8/fixed point，必须另立 oracle：明确是 `Q(theta*W)` 还是运行时 `Q(theta)*Q(W)`，比较 folded/unfolded accumulator bitstream，并重跑有效集精度。不能把 FP32 fold 自动写成整数 RTL exact。

### G4｜进入周期模拟器的门

只有 G0-G3 通过后，才允许生成 D1 mask descriptor，并在同一 96-lane、同一 SRAM/DRAM/metadata 账本中比较：

- B0：dense amplitude MAC；
- B1：exact sparse amplitude-aware（每 event 显式携带/施加 theta）；
- SCB：mask event + offline folded weight；
- C1/C2 coexist：相同 K8 descriptor、相同 completion/commit、相同 SRAM latency。

SCB 对 B1 的收益才是“threshold fold”自身收益；SCB 对 B0 的总体收益还包含 81.5% exact-zero sparsity，二者必须分列。

## 5. 当前只允许使用的 analytical opportunity

由 M649 D1 count 可得：activity `18.491154%`、zero `81.508846%`、dense/source-active `5.407991x`。由 M510 的边界修正 product bounds 可得 D1 dense/active-product opportunity `5.324275x–5.949361x`。

这些都是 **exact-zero work opportunity / analytical**，不是 cycle speedup。M510 给出的 D1 ideal-96-lane 周期为 `27.415M–30.634M/frame`，只占 corrected envelope 的 `3.411%–3.873%`；即使 D1 免费，系统 Amdahl 上限也只有 `1.0353x–1.0403x`。因此 D1 threshold fold 是 decoder 完整性的关键使能模块，但不能单独当性能 headline。

四层 decoder 的 M510 analytical bounds 仍是：corrected network share `21.572%–22.826%`，dense/active product opportunity `4.4767x–4.8139x`，decoder-free ceiling `1.2751x–1.2958x`。只有在 D0/D2/D3 exact `{0,1}` 与 D1 exact `{0,theta}` 都通过 payload、cycle 和 memory 账后，才可把它们合为 decoder 结果。

## 6. 跨 Conv/FC/decoder 的 scale-coded binary（SCB）统一模型

条件定义：每个 producer 输出 `x_g = theta_g * mask_g`，其中 `theta_g` 是 checkpoint 静态 scalar，`mask_g` 为 exact binary。`theta=1` 是普通 binary 的硬件子集。

| 图边界 | 是否可直接无损折叠 | 条件/实现 |
|---|---|---|
| Conv2d / Linear / ConvTranspose2d | 是 | 把 producer theta 折进对应 input-channel weight slice；保留 bias |
| concat | 是，按 group | 每个 concat 源保留 channel range + theta，分别折对应 weight slice；不能假设全 concat 同 theta |
| 一个 producer 多 consumer | 是 | 每个 consumer 离线折自己的 weight；静态图不需 event scale tag |
| BN after folded linear op | 是 | 先保证 folded linear 输出相同，BN 参数/状态不改 |
| BN 直接消费 SCB | 不能仅“折 weight” | 需要两值 LUT 或完整 affine 推导，running mean/bias 不能丢 |
| residual/add | 不能直接当 binary | `theta_a*m_a + theta_b*m_b` 是多级值；只有同尺度且保留整数 multiplicity，或把线性 consumer 分配到两支，才能 exact |
| attention Q/K | 不能直接折 consumer weight | score 含动态点积，尺度变为 `theta_q*theta_k`；Shiftmax/量化/阈值是非线性边界，需单独 score-scale 合同 |
| attention V / dynamic gate | 条件式 | 静态 V scale 可在后续线性边界吸收；动态 gate 不能伪装成 static weight |
| normalization、comparison、clamp、softmax/Shiftmax | 否 | 必须在非线性前显式保留/转换 scale semantics |

硬件成本的正确口径：

- offline 每个 weight 一次 FP32 scale multiply；部署后 weight 数量和 SRAM 容量不增；
- 若完全 bake 入 weights，runtime threshold metadata 可为 0；若要求 checkpoint 可编程，105 个 FP32 scalar 上限为 420 B，另加 group-range manifest；
- event payload 从 amplitude+address 降为 mask/address，运行时不需要 theta multiplier；
- concat 需要静态 channel-group 映射，不需要每-event scale tag；
- checkpoint/weight identity 必须更新，阈值改变必须重新生成 folded weights；
- 105 个 ATLIF 里 94 个 theta 已 exact 1，真正发生 weight rewrite 的候选仅 11 个。这个事实让硬件代价低，但也意味着 SCB 本身不能预写夸张倍率。

网络级 SCB 当前只允许写为 **conditional generalization**；主实证对象仍是 D1。任何全网周期/能量数字都必须来自逐边界准入后的统一 simulator。

## 7. 相对 Prosperity、Phi、FireFly-T 的合法包装

- [Prosperity](https://arxiv.org/abs/2503.03379) 利用 binary spike GEMM 中的 product similarity/reuse；SCB 不应声称发明 product sparsity。合法差异是：H67 learned ATLIF 输出带 checkpoint-static amplitude，先用 exact scale absorption 把它变成无 per-event magnitude 的 mask，再让既有 bit/product-sparse engine 消费。必须把官方 Prosperity 作为 prior，并将其 2.46x iso-workload 继续标为 external opportunity，不是本 RTL。
- [Phi](https://arxiv.org/abs/2505.10909) 用 binary activation 的 L1 pattern/PWP 与 L2 residual/PAFT。SCB 不产生新 pattern，也不等价于 PAFT；它只可作为“把 scale-coded spike 无损映射到 pattern/PWP weight bank”的输入规范。若后续复用 PWP，theta 必须折入对应 PWP，且 PAFT 有损列继续分开。
- [FireFly-T](https://arxiv.org/abs/2505.12771) 的 sparse decoder 并行提取多个 nonzero spike，并用 load balancing/weight dispatch 提高吞吐；SCB 可把其 binary event 协议迁移到 H67 的 learned-threshold spike，避免每-event amplitude，但不能声称发明 multi-nonzero decoder、out-of-order 或 bank-conflict elimination。
- H67 可主张的对象差：**learned scalar-threshold ATLIF + dense optical-flow decoder + ConvTranspose weight layout + concat group scales + K8 signed-source/240 KiB 资源边界**。这是一种引用充分的 mechanism transfer，不是换名冒充 first。

Novelty 预判：D1 threshold fold 单独最多是中等强度 supporting mechanism（代数本身简单）；若它闭合此前遗漏的 22–23% decoder，并与 K8 descriptor/同资源周期链共同证明，才可成为 C2/decoder contribution 的关键协议点。论文推荐名称是 `checkpoint-static scale-coded binary folding`，避免使用 `first` 或把它包装成新的 sparsity 定义。

更细的 novelty 边界是：G1 单独通过只有“learned-threshold spike 的 exact mask+sidecar representation”，新颖度偏低，必须靠实测 payload/datapath 节省才能成为硬件贡献；G2 bit-exact 通过后，才可主张 `compile-time scale absorption for ConvTranspose groups`；若 G2 失败，论文名称应退回 `scale-coded binary sidecar protocol`，不得继续使用 `exact folded-weight execution`。

## 8. 停止线与推荐论文句

任何一个条件触发即停止 exact 线：

- D1 存在非 `{0,theta}` bit pattern 或 threshold 随 sample/forward 变化；
- folded/unfolded FP32 deconv 不是 bitwise equal；
- fixed-point oracle 不能闭合，或有效集精度身份漂移；
- 同资源 SCB 相对 B1 没有减少 payload/scale datapath，且 decoder unified cycle 没有实质收益。

若 G0-G4 全通过，可写：

> We represent checkpoint-static ATLIF outputs as a one-bit event mask plus a producer scale and absorb each scale into the corresponding Conv/FC/ConvTranspose input-channel weight group. This preserves the measured H67 decoder computation while exposing exact zero work to the same signed-source engine; concatenated groups retain independent compile-time scales.

在通过前只能写：

> Source and checkpoint inspection identify D1 as a candidate scale-coded-binary boundary; a bitwise folded-weight equivalence test is required before admission.

## 9. 本审计边界

- 本次未启动 GPU、VCS、DC、PT、Formality 或任何 simulator；
- 未修改 checkpoint、config、source、M510、M649、M658 或 `docs/359`；
- 未生成 activation payload、cycle、speedup、energy、PPA 或 accuracy 结果；
- `docs/359` SHA256 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
