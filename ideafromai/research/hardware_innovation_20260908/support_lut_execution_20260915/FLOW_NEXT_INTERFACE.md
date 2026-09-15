# 现存 preds.1 → decoder2 → preds.2：细节准入的可执行边界

2026-09-15。本轮只读实际模型、加载器、checkpoint、已有运行 shape 和 primary 方法，并完成 CPU 拓扑闭包探针；没有执行网络推理、训练、GPU、RTL 或 EDA。新写文件仅本文、[探针](probe_flow_dependencies.py)及[结果](probe_flow_dependencies.json)。探针 mask 全为合成，不能作为真实光流质量或速度结果。

**可提前决策的位置确实存在：`preds.1` 完成后、仍在运行的 `decoder2` 之前。实际失配是 decoder2 使用当前输入全域 BN：若保留某个细节像素的原数值，其 BN 统计依赖仍覆盖全部 decoder2 输出。** 因而当前图不能把局部输出 mask 直接变成等价的局部转置卷积省算。可继续的有损适配是**仅冻结或重训 decoder2 这一处 BN，再比较同固定 BN 的全细节、普通块 mask、任务驱动 mask，以及更早 `preds.1` 直接输出**。动态 BN 的失败不等于细节选择家族失效。

## 1. 源码定位与实际张量

图来自 [Spiking_STSwinNet.py:161](/home/zhumd/work/sdformer_codex/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_STSwinNet.py:161)。每轮先拼接 encoder skip，除首轮外再把前一 head 拼到最前，然后执行 decoder 和本轮 head。当前 [coarse_readout:22](../algorithm/evaluate_coarse_readout.py:22) 在 `preds.2` 的 hook 中保存 `output.sum(0)` 并抛出 `CoarseReady`，故 decoder3 不会运行；本接口不再计其删除收益。

| 节点 | 实际 `[T,B,C,H,W]` 或参数形状 | 数据依赖及处理 |
|---|---|---|
| decoder1 输出 `D1` | `[10,1,192,60,80]` | 已完成上一级 PSN、转置卷积和 BN |
| 更早 head `P1=preds.1(D1)` | `[10,1,2,60,80]`；W `[2,192,1,1]` | PSN 混合完整 T10，随后 1×1；有 bias，无 head BN |
| encoder `blocks[1]` skip `E1` | `[10,1,192,60,80]` | 早已由 encoder 生成，仍需保留供 decoder2 使用 |
| decoder2 输入 `X2` | `[10,1,386,60,80]` | 通道顺序严格为 **`[P1(2),D1(192),E1(192)]`** |
| decoder2 门 `G2` | `[10,1,386,60,80]` | 先对 `X2` 做 PSN；当前二值配置给出 `{0,θ}` |
| decoder2 转置卷积 `Y2` | `[10,1,96,120,160]`；W `[386,96,3,3]` | stride=2、padding=1、output_padding=1；无 bias |
| decoder2 BN `N2` | `[10,1,96,120,160]` | 当前输入统计，随后才进入最后 head |
| 当前出口 `P2=preds.2(N2)` | `[10,1,2,120,160]`；W `[2,96,1,1]` | 完整 T10 PSN → 1×1；有 bias，无 head BN |
| 当前 readout | `[1,2,480,640]` | `sum_T(P2)` 后直接双线性插值，`align_corners=False` |

转置卷积定义在 [Spiking_modules.py:398](/home/zhumd/work/sdformer_codex/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py:398)，其实际执行顺序在 [461 行](/home/zhumd/work/sdformer_codex/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py:461)，head 顺序在 [643 行](/home/zhumd/work/sdformer_codex/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py:643)。不能拿同文件另一种“插值后卷积”的 decoder 类替代本图的 ConvTranspose 支持。

尺寸有已有真实运行 [profile.json](../open_fusion_execution/major_operator_fusions_20260913/root_owned/profile.json) 交叉验证；probe 重新读取其中的 D2 行，并读取实际配置和 checkpoint。D2 的 333,504 个 W、两个相关 PSN 的各 100 个时间系数、P1/P2 的 384/192 个 head 系数均非零。这只证明结构依赖；没有假设某次实际门全发放。

**所有 head 都是完整 flow，不是残差。** `P2` 没有与 `P1` 相加；前级 flow 只是 D2 的两个输入通道，且同 feature/skip 一起先过 PSN。不能将这两个通道当作直接送入 deconv 的连续乘法源；也不能把 `sum_T(P1)` 替代 D2 需要的十个 `P1[t]`。当前 [AT-LIF PSN forward](/home/zhumd/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/atlif_ternary_psn/atlif_ternary_psn.py:344) 逐位置、逐通道做完整时间矩阵混合，没有空间混合；其 `P1[t]` 是 readout 的时间贡献，不能解释成十个运动轨迹采样点。

## 2. BN 模式已从“未知”落实为当前全域统计

这里只看 checkpoint 中有没有 running_mean 会得出错误结论。实际加载路径是：

1. [run_bn_probe.py:232](../algorithm/run_bn_probe.py:232) 建模、加载 checkpoint、设多步模式，再调用 `set_bn_mode(model)`。
2. [set_bn_mode:84](../algorithm/run_bn_probe.py:84) 将未选定 BN 的 `track_running_stats=False`，并把 running_mean/var 设为 None；单纯 `eval()` 不会恢复固定统计。
3. [load_system](../algorithm/nrv_cost_probe/run_probe.py:173) 只固定选中的 FFN first BN；[ParentNetwork](../open_fusion_execution/breadth_20260912/algorithm/parent_network.py:31) 再固定 patch residual BN。[当前 R8 模型访问层](../r8_consumer_fusion_20260914/data/model_access.py:11) 沿这条父模型路径，没有给 D2 另换 BN。
4. 本轮用 NumPy checkpoint reader 实读 [patch_train_calibration.pt](../algorithm/patch_probe/patch_train_calibration.pt)，其中仅 `resblocks.0/1.norm1/2.norm_layer` 四项，**没有 decoder 或 pred 项**。

模型的 [SpikingNormLayer:101](/home/zhumd/work/sdformer_codex/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py:101) 对当前 `BN` 直接调用 `layer.BatchNorm2d`。项目 [requirements](/home/zhumd/work/sdformer_codex/SDformer/requirements.txt:15) 固定 SpikingJelly 0.0.0.0.14；该版本 primary 源码的 [BatchNorm2d.forward](https://raw.githubusercontent.com/fangwei123456/spikingjelly/0.0.0.0.14/spikingjelly/activation_based/layer.py) 在多步模式调用 [seq_to_ann_forward](https://raw.githubusercontent.com/fangwei123456/spikingjelly/0.0.0.0.14/spikingjelly/activation_based/functional.py)，先合并 T、B，再做普通 BN。因此当前 B1 每个通道的统计域是 `10×120×160=192,000` 个 Y 值；96 通道共 18,432,000 值。

选定一个 P2 位置需要该位置全部 96 个 BN 输出；每个输出又连接本通道全域均值/方差。通过密集 W 和时间 PSN 反推，任何非空细节 mask 的**当前计算图依赖闭包**都包含所有 4,800 个 D2 源空间位置和全部 Y2 统计域。把未选 Y 置零、只用选中区域算 BN、拿上一帧或训练均值代替，都会改变函数。

这不是一个数学复杂度下界：理论上可能另推全域统计的代数算法，不一定逐项物化全部 Y。本轮没有实现或计时此类算法。准确结论是：**当前逐点 deconv→BN 的接口，不能凭局部 mask 撤销统计所需的主体任务。** 统计完成后的 P2 PSN/1×1、部分物化流量仍可能选择执行，但不能把它记成已取消整个 D2 的收益。

## 3. 已执行的真实几何、合成 mask 闭包

复现：

```bash
/opt/anaconda3/bin/python3.12 /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/support_lut_execution_20260915/probe_flow_dependencies.py
```

结果 **PASS，15 个合成 mask**。程序用源 scatter 与输出 gather 两种独立枚举核对转置卷积的有效边，读取配置中的 kernel/T 和源码构造器中的 stride/padding/output_padding，并核实际参数支持。

对于源 `(iy,ix)`，有效输出为 `(2iy+ky−1,2ix+kx−1)`，`ky,kx∈{0,1,2}`，裁切到 120×160。全图有 **42,781 条空间边**。4,800 个源中，1/138/4,661 个分别连接 4/6/9 个输出位置；19,200 个输出中，4,941/9,598/4,661 个分别依赖 1/2/4 个源位置。故“每个输出一律 3×3 个源”是错误的普通卷积闭包。

一条 post-PSN 源门最多供给 `9×96=864` 个输出标量；一个 pre-PSN 输入标量还可影响十个时间门，结构上对应最多 8,640 条后续系数贡献。它们是共享消费者上限，不是某帧实际非零乘加计数。全图有效 dense 系数贡献为 `42,781×10×386×96=15,852,927,360`，已裁掉越界 tap；不是 RTL 周期，也不能拿它替代二值 AAC 的强基线。

下表展示**固定 D2 BN 的反事实函数**所允许的局部闭包；当前动态 BN 列对每个非空 mask 都需要全域。每个选中 P2 空间位置均保留全部 T10、96 个输入通道及两个 flow 分量。

| 合成请求 | P2 所需位置 / 19,200 | 固定 BN 下源位置 / 4,800 | 选中空间边 | 同时服务未选输出的源位置 | 当前 BN 所需源位置 |
|---|---:|---:|---:|---:|---:|
| 空集，纯粗头 | 0 | 0 | 0 | 0 | 0 |
| 全细节 | 19,200 | 4,800 | 42,781 | 0 | 4,800 |
| 中心一个偶偶位置 | 1 | 1 | 1 | 1 | 4,800 |
| 中心一个奇奇位置 | 1 | 4 | 4 | 4 | 4,800 |
| P2 网格中心 8×8 块 | 64 | 25 | 144 | 16 | 4,800 |
| 两个相邻 8×8 块的并集 | 128 | 45 | 288 | 24 | 4,800 |
| 仅所有奇奇位置，25% 输出 | 4,800 | 4,800 | 18,921 | 4,800 | 4,800 |
| P2 网格随机约 10% | 1,989 | 3,033 | 4,487 | 3,033 | 4,800 |
| 最终 480×640 中心 32×32 块 | 100 | 36 | 225 | 20 | 4,800 |
| 最终 480×640 中心 64×64 块 | 324 | 100 | 729 | 36 | 4,800 |
| 最终 480×640 随机约 10% | 19,167 | 4,800 | 42,714 | 61 | 4,800 |

最后三行先反推 **4×双线性插值**的 support，再反推 deconv。`align_corners=False` 的源坐标为 `(x+0.5)/4−0.5`，边界按现实现裁切。随机选中最终 30,668/307,200 个像素，只占 9.98%，却要 19,167/19,200=**99.83%** 的 P2 位置；随机像素百分比几乎不能代表可删任务比例。连续中心 32×32 请求也不是简单对应 P2 的 8×8，因为插值 halo 使其需要 10×10。

两个相邻 8×8 块分别执行各需 25 个源，并集只需 45，能共用 **5 个源位置的 PSN/供数**；其 288 条输出贡献边仍各有消费者，不能把源复用同时算成卷积算术消失。对保留的源盲发所有九 tap 还会产生额外未选输出：单个 8×8 块会触及块外 57 个位置。真正的取消单元至少要带输出 phase/tap 及消费者集合，而不是一位“这个源活跃”。

## 4. 决策提前量、输出定义与状态账

`P1` hook 的软件顺序足以让决策早于 D2 拼接/PSN/deconv；它不证明未来硬件中的首次预取尚未发生。应对每项真实请求记录 `decision_ready`、`source_accept`、`weight_accept` 和 `consumer_retire`，决策只取消尚未接受的服务。**目前没有这些时间戳的 RTL 或硬件测量，不给提前量/周期数字。**

到 P1 可用时，所有 encoder、decoder0、decoder1 已完成；这些工作不能取消。E1 已供后续 encoder 使用，也不能把其生成算成新节省。D1 同时供 P1 和 D2，P1 同时供早期 readout/决策和 D2，E1 的 D2 义务还没退休。只有对应消费者集合清空才可回收源；不能为早退先销毁 D1/E1/P1，再声称失败重放只需几拍。

定义 `f1=sum_T(P1)`、`f2=sum_T(P2)`，`U8/U4` 分别**直接**插值到 480×640，最终像素 mask 为 M：

```
Fout(x) = M(x) ? U4(f2_sparse)(x) : U8(f1)(x)
```

这是两个完整 flow 的选择，没有 `f1+f2`。`f2_sparse` 必须覆盖 M 所需的插值前驱；M 全零须逐值等于普通 P1 直接 readout，M 全一须等于相同 BN 函数的全 D2。不能把粗分支改成 60→120→480 两次插值后仍称相同 baseline。现 readout 不乘额外 flow 幅值缩放；若在 60×80 坐标上研究对应关系，`p+f1(p)/8` 中的 `/8` 只转换坐标单位，不改变最终输出的 flow 值。

| 需显式计费或保留 | 当前 FP32 全张量账，仅用来界定义务 |
|---|---:|
| P1 十个时间面，D2 输入仍需要 | 96,000 值，384,000 B |
| 新提前 readout `f1` | 9,600 值，38,400 B；朴素 sum 额外 86,400 次加法 |
| D1 / E1，各自仍被 D2 需要 | 每份 36,864,000 B |
| X2 全拼接 | 74,112,000 B；可按通道视图供数，不能把视图免费改成更多读口 |
| Y2 全统计域 | 73,728,000 B；可流式统计，但保存/重算/再读必须选一种并付费 |
| D2 原 W / P2 head W+bias | 1,334,016 B / 776 B |
| 最终像素 / P2 / 源位置 bit mask | 38,400 / 2,400 / 600 B；它们不是可以互换的一张 mask |

这些字节不是必需的片上 SRAM 下界，也不是新的硬件资源承诺。实际实现还需给 mask 向前传播、union/引用计数、源/skip 重读、决策器算术、统计/验证、失败补算和输出 holding 单独记账。固定 BN 后若某非空 mask 仍涉及所有 tap，仍可能需要全部 W；W 已驻留时也不能把参数总字节当逐块省下的外存流量。

## 5. A → 具体失配 → 有损恢复与一个待证 X

**A 的来源。** [BiLD 原文](https://papers.nips.cc/paper_files/paper/2023/file/7b97adeafa1c51cf65263459ca9d0d7c-Paper-Conference.pdf)以小模型已经产生的置信度推迟大模型调用，并在大模型真正运行后取得 logits、检查并回滚相应后缀。可借的是决策时间和状态退休边界。这里没有现成 token softmax，完整运行 P2 来“免费验证”会付掉待省的 D2 成本；T10 非因果 PSN 也没有同一种时间后缀恢复。详见已读的 [C-Transformer/BiLD 链](../paper_mechanism_transfer_20260915/literature/compass_ctransformer.md)。

**最近邻比 BiLD 更近。** [WaveletVFI 原文](https://arxiv.org/abs/2309.03508) 的 Algorithm 1 已用粗层高频系数/阈值产生细层 mask，经过空间支持扩张执行 sparse decoder，随后 IDWT 重建。其输出是视频插帧的 wavelet 系数，不是本图的完整 flow；但“粗运动/粗表示指导后续细节计算、计 halo”已是成熟机制。[作者代码](https://github.com/ltkong218/WaveletVFI)也是公开的。普通块 mask、动态阈值或粗头早退不能作为新意标题。

本方 [9 月 13 日报告](../deep_target_research_20260913/flow/decoder2_addendum.md)已经提出过 P1→D2 边、halo 和事件可观测性；**本轮增量是准确判定动态 BN、实测真实几何闭包和固定 BN 恢复接口，不是再次发现 coarse-to-fine。**

有损恢复只改这一处 `sttmultires_unet.decoders.2.norm_layer.norm_layer`：从训练集独立校准固定均值/方差；若质量不够，可仅围绕此 BN 做受限重训，再冻结。固定 BN 的全 D2 首先单独评估，之后才引入选择。不能挪用 patch BN 校准项、不能把验证帧当前统计免费赠给决策器，也不能继承原 P2 或当前 R8 的 AEE。本轮未执行校准/重训。

**一个任务驱动的待证 X：依据粗 flow 的对应关系风险选择细节，并按新增依赖而非输出面积付预算。** 在已经得到的 60×80 完整 `f1` 上，将 `p+f1(p)/8` 投到有界目标格，识别多对一、越界及邻域运动不一致；若利用已有事件输入支持，还须防止把“无事件/孔径问题”误当高置信。优先给这些对应关系风险区域分配细节请求，并在真实 ConvTranspose phase/源闭包并集上计算新增任务数，合并共用源后再准入。该候选直接使用光流的对应关系，区别于只看 flow 梯度或事件数量的普通 mask。

但这仍是**启发式，不是遮挡或误差证书**：独立运动、遮挡和低纹理会使粗 flow 的映射失真。不能使用 GT、未执行的 P2、额外后向 flow 当免费决策输入；归一化 voxel 值也不等于原始事件计数或真实时间。目标格计数/碰撞表、坐标运算、读取和闭包合并都要付费。若要补算，仅能在输出提交前追加闭包并保留源，不能借 BiLD 名字宣称精确恢复。

独立新颖性评价：**暂约 3/10，尚未成立。** 对应风险、事件可观测性、预算感知剪枝和消费者合并分别都有强邻近方法；本轮没有证明其组合超出普通强控制。它的价值是给失配后留下具体可测接口。只有在同固定 BN、同闭包实现、同决策预算下，优于下列普通控制，且质量合格后完整成本为正，才值得讨论进一步创新。

| 必须保留的臂 | 要隔离的原因 |
|---|---|
| 当前动态 BN、完整 D2→P2 | 原函数及原成本锚点 |
| 仅固定 D2 BN、完整 D2→P2 | 先量 BN 改函数本身的质量与成本；给稀疏臂同函数全细节分母 |
| P1 直接早退：同 sumT + nearest/bilinear 到 480×640 | 最简单的更早粗头；目前没有本轮 AEE，不能拿已有 P2 早退成绩代替 |
| 同固定 BN 的普通连续块选择、梯度/密度/不确定性选择 | 给它们相同的真实 halo、源并集和退休规则，避免用重复加载的弱实现放大 X |
| 同固定 BN 的对应风险选择 | 唯一候选 X；同时计粗流汇总、决策、mask/状态、detail、补算及最终 readout |

质量应按当前同环境 NB0 门槛和正式验证口径逐帧评估，不沿用旧的“+0.005 就杀家族”。即使 X 不胜普通 mask，也保留失败定位；若 P1 直接早退已胜所有复杂选择，应如实采用这个普通强控制。

## 6. 本轮闭合与保留缺口

已闭合：实际 P1→D2→P2 边和通道顺序、heads 全 flow、完整 T10 依赖、当前 D2 动态 BN、真实转置卷积的 phase/边界/共享源计数、最终插值反向闭包，15 个合成 mask 的 CPU 双重枚举全部通过。没有改算法或生产文件。

仍未闭合的两个先后接口：**先**测普通 P1 早退和仅固定 D2 BN 的全细节质量，确定新函数是否值得；**再**用真实决策 mask 测提前量与完整执行成本，比较普通 mask 和上述对应风险选择。目前没有这两项 AEE、真实 mask 分布、RTL 净收益或 PPA。当前证据支持一次有界恢复，不支持宣称已获得任务原生加速。
