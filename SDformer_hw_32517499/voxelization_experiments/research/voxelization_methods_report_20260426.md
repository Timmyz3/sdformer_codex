# 可移植到 SDformerFlow 的事件体素化方法调研

日期：2026-04-26

范围：近几年顶会或接近顶会级别的事件相机工作中，和 SDformerFlow 这类事件光流任务相关、并且有机会迁移到当前项目的体素化或事件表示方法。当前 baseline 的路径是：从 `event_tensors/10bins/left` 加载 DSEC 预处理体素，在 `train_flow_parallel_supervised_SNN.py` 中把 signed voxel 按正负拆成 `[B, T, 2, H, W]`。原始体素生成逻辑主要在 `DSEC_dataloader/event_representations.py`，重点是 `VoxelGrid.convert_CHW`、`VoxelGrid.convert_CHW_polarities` 和 `events_to_voxel_grid_v2`。

## 2025-2026 新增结论

这次补充搜索重点看了 CVPR/ICCV/ICLR/AAAI/IJCAI 在 2025-2026 年已经公开的论文。结论是：真正直接作用在“体素化/事件表示”层面的新工作不算多，但有几篇比旧版报告更值得放到第一优先级。

| 新优先级 | 方法 | 年份/会议 | 对本项目最有用的点 | 建议实验编号 |
| --- | --- | --- | --- | --- |
| 1 | EventPillars | AAAI 2025 | 用 Temporal Event Range、Event Polarities、Normalized Event Density 构造更高效 dense event representation，声称可 plug-and-play 到下游任务；很适合替换当前普通 voxel。 | V1 |
| 2 | EDCFlow temporally dense difference maps | CVPR 2025 | 仍用标准 voxel grid，但把事件流拆成更密集的时间片，在特征层显式计算相邻时间差分；非常贴合光流任务。 | V2 |
| 3 | Unbiased Gradient Estimation for Event Binning | ICLR 2026 | 专门解决 event binning 不连续导致的梯度截断/偏差问题；适合做可学习体素化或端到端 binning。 | V3 |
| 4 | OmniEvent | AAAI 2026 | 将事件表示做空间/时间解耦、增强、融合，输出 grid-shaped tensor，可直接接标准视觉模型；比普通 voxel 更强但改动更大。 | V4 |
| 5 | EventFlash | ICLR 2026 | 自适应 temporal window aggregation + sparse density-guided attention，可处理长事件序列并减少空区域冗余；适合做输入前 token/window 压缩。 | V5 |
| 6 | UniCT Depth 的 voxel + cross-modal fusion block | IJCAI 2025 | 仍用双线性 voxel，但它的局部卷积补偿和模态 self-attention 可借鉴为体素增强 adapter。 | V6 |
| 7 | Fully Sparse Event-Camera Voxel Grids | 2026 arXiv | 不是顶会正式论文，但方向很重要：不再转 dense tensor，而是原生 sparse voxel 计算。适合长期方案。 | V7 |

截至 2026-04-26，CVPR 2026 主会论文页面能访问入口，但开放论文内容还不稳定；因此本版不把 CVPR 2026 作为可靠主来源。AAAI 2026 和 ICLR 2026 已能查到正式或 OpenReview 页面，优先采用。

## 推荐优先级

| 优先级 | 方法 | 来源 | 可复用核心思想 | 预计改动量 | 风险 |
| --- | --- | --- | --- | --- | --- |
| 1 | EventPillars / pillar-based event representation | AAAI 2025 | 用时间范围、极性激活和归一化事件密度构造高效 dense 表示，比普通 voxel 更强调完整时间分布和目标区域。 | 中 | 中 |
| 2 | EDCFlow 时间密集差分体素 | CVPR 2025 | 以标准 voxel 为基础，把事件时间维拆得更细，在特征层计算多尺度时间差分，直接服务光流。 | 中 | 中 |
| 3 | 可学习/无偏 event binning | ICLR 2026 | 针对 binning 不连续导致的梯度问题，适合把当前固定体素化改成可学习或半可学习体素化。 | 中到高 | 中 |
| 4 | OmniEvent 统一事件表示 | AAAI 2026 | 空间和时间先解耦增强，再融合成 grid-shaped tensor；可接标准视觉模型。 | 高 | 中到高 |
| 5 | EventFlash 自适应时间窗口聚合 | ICLR 2026 | 用自适应时间窗口和 density-guided 稀疏注意力减少空体素/长序列冗余。 | 高 | 中 |
| 6 | 离散计数体素 / RVT 风格事件张量 | CVPR 2023 RVT | 不做时间维双线性插值，直接 hard-bin 计数；作为低风险 baseline 替换仍然值得做。 | 低 | 低 |
| 7 | 自适应事件密度体素 / ADM | ICCV 2023 MDR event-flow | 生成多种事件密度版本，或对体素做空间密度归一化，再按密度选择/融合。 | 中 | 中 |
| 8 | 多窗口时间体素堆叠 | ECCV 2024 / ICCV 2025 Temporal Event Stereo | 复用过去几个窗口，做级联时间聚合；和 baseline 的 `num_chunks` 思路接近。 | 中 | 中 |
| 9 | V2V 风格离散体素模拟与增强 | NeurIPS 2025 V2V | 直接从普通视频生成 event voxel，或者把其中的阈值/噪声随机化思想用于现有 DSEC 体素增强。 | 中到高 | 中 |
| 10 | 原生 sparse voxel | 2026 arXiv 趋势 | 不再转 dense tensor，保持事件 voxel 稀疏结构；和后续稀疏剪枝方向一致。 | 高 | 高 |

## 逐项分析

### 0. EventPillars，柱状高效事件表示

来源：EventPillars: Pillar-based Efficient Representations for Event Data，AAAI 2025。

可复用思想：

- 普通 voxel 只是按时间 bin 累积事件，容易丢失完整时间分布或造成冗余。
- EventPillars 从 pillar-based sparse data representation 借鉴思想，显式编码：
  - Temporal Event Range：描述一个空间柱内的时间分布范围；
  - Event Polarities：显式记录正负极性动态；
  - Normalized Event Density：作为空间注意力先验，让模型更关注有信息的区域。
- 论文声称该表示可 plug-and-play 到不同下游任务，并在计算和存储上明显低于常规 dense 表示。

迁移方案：

- 新建 `voxelization_experiments/V1_eventpillars/`。
- 先不要改 SDformerFlow 主干，只在 dataset 输出前做表示替换。
- 输出仍保持 `[T, 2, H, W]` 或 `[C, H, W]`，让训练入口最小改动。
- 可以先做三通道/多通道版本：
  - 正事件计数；
  - 负事件计数；
  - normalized event density；
  - temporal range 或 timestamp mean/std。

风险：

- 原论文主要在识别/检测任务上验证，光流需要重新验证。
- 如果通道数改变，需要同步调整模型输入 stem 或增加 adapter。

### 0.1 EDCFlow，时间密集差分体素

来源：EDCFlow: Exploring Temporally Dense Difference Maps for Event-based Optical Flow Estimation，CVPR 2025。

可复用思想：

- 它仍使用标准 voxel grid：每个 event 根据时间位置贡献到两个最近的 temporal bins。
- 关键不是换掉 voxel，而是不要把一个事件窗口当成单帧特征；应在更密集的时间片上提取相邻差分。
- 对光流尤其重要，因为光流本质上依赖时序变化。

迁移方案：

- 新建 `voxelization_experiments/V2_temporal_difference_voxel/`。
- 在当前 `[B, T, 2, H, W]` 输入后添加一个无参或轻量模块：
  - `diff_1 = voxel[:, 1:] - voxel[:, :-1]`
  - 多尺度差分：间隔 1、2、4 个 bin；
  - 把差分作为额外通道拼回输入，或者用 1x1/3D conv 压回原通道数。
- 这比完全替换 voxel 更稳，因为保留当前输入语义。

风险：

- 通道增加会涨显存。
- 建议先做“差分后压缩回 10 bins/2 polarity”的版本。

### 0.2 ICLR 2026：无偏 event binning 梯度估计

来源：Unbiased Gradient Estimation for Event Binning via Functional Backpropagation，ICLR 2026。

可复用思想：

- 普通 event binning/hard binning 是不连续操作，会截断梯度。
- 如果未来希望让 bin 边界、时间缩放、阈值或体素核函数可学习，就需要解决这个问题。

迁移方案：

- 第一阶段不直接上端到端 raw event learning。
- 先实现可学习 binning 的实验模块：
  - learnable temporal scale；
  - learnable bin offset；
  - learnable interpolation kernel width。
- 然后在反向传播中使用近似/无偏梯度估计。

风险：

- 当前 baseline 用的是预处理好的 `.npy` 体素，不保留 raw event list；要端到端学习 binning 必须改 dataset 或重新预处理。
- 工程量比 V1/V2 大，但论文方向非常贴合“体素化本身优化”。

### 0.3 OmniEvent，统一事件表示学习

来源：OmniEvent: Unified Event Representation Learning，AAAI 2026。

可复用思想：

- 事件数据空间和时间维度的尺度/密度不均衡，直接把它们当 3D 点或普通 voxel 容易不稳定。
- OmniEvent 做 decouple-enhance-fuse：先分别增强空间域和时间域，再通过 attention 融合。
- 输出是 grid-shaped tensor，可以接标准视觉模型。

迁移方案：

- 新建 `voxelization_experiments/V4_omnievent_adapter/`。
- 不直接复现完整 OmniEvent，先做轻量版：
  - spatial branch：2D depthwise conv 处理每个时间 bin；
  - temporal branch：1D temporal conv/SSM 处理每个像素的 bin 序列；
  - fusion：1x1 conv 或 attention gate；
  - 输出保持 `[B, T, 2, H, W]`。

风险：

- 已经接近模型 stem 改动，不是纯预处理。
- 需要和神经元实验分开评估，否则变量太多。

### 0.4 EventFlash，自适应时间窗口和稀疏密度注意力

来源：EventFlash: Towards Efficient MLLMs for Event-Based Vision，ICLR 2026。

可复用思想：

- EventFlash 面向事件 MLLM，但它提出的两个模块对体素化很有启发：
  - adaptive temporal window aggregation：根据事件流内容自适应压缩时间 token；
  - sparse density-guided attention：根据事件密度选择信息区域，减少空区域计算。
- 这和 SDformerFlow 当前固定 10 bins 的体素化形成对比：不是所有窗口都应该用一样的时间切分。

迁移方案：

- 新建 `voxelization_experiments/V5_adaptive_window_density/`。
- 用非学习版先试：
  - 根据每个样本的事件密度，把 10 bins 合并成 5/8/10 个有效时间段；
  - 对低密度区域降低权重；
  - 输出再插值/补齐回 10 bins，保持模型输入不变。

风险：

- 自适应时间窗口会改变事件时间语义，可能影响光流标注对齐。
- 需要先做小序列可视化和 AEE 对比。

### 1. 离散计数体素，RVT 风格

来源：Recurrent Vision Transformers for Object Detection with Event Cameras，CVPR 2023。代码：`https://github.com/uzh-rpg/RVT`。

RVT 使用非常直接的 dense tensor：2 个极性通道，`T` 个时间离散步，每个像素记录事件计数，之后把极性和时间维展平给 2D 卷积。它和当前 SDformerFlow 的主要区别是：baseline 对时间维做双线性插值，而 RVT 是 hard-bin 计数。

迁移方案：

- 新建体素化实验目录，例如 `voxelization_experiments/V1_voxel_discrete_count/`。
- 在实验 overlay 中新增 `DSEC_dataloader/event_representations.py` 或独立预处理脚本。
- 保持输出和当前训练入口兼容：
  - signed 形式：`[T, H, W]`
  - 极性分离形式：`[T, 2, H, W]`
- 配置建议：
  - `data.voxel_method: discrete_count`
  - `data.voxel_dtype: uint8|float16|float32`
  - `data.voxel_normalize: none|minmax|std`

优点：

- 改动量最低，最适合作为第一组体素化替换实验。
- 预处理比时间插值更快。
- 缓存体积可以更小。
- 对 SNN 友好，因为每个时间 bin 更像干净的事件帧/脉冲计数。

风险：

- 丢掉了 bin 内精细时间信息，可能影响细粒度光流精度。
- 但对 SNN 来说，离散计数可能反而更稳定，需要实验验证。

### 2. 自适应事件密度体素，ADM

来源：Learning Optical Flow from Event Camera with Rendered Dataset，ICCV 2023。论文提出 MDR 数据集和 ADM，其中 ADM 是面向不同事件密度的 plug-in 模块。

可复用思想：

- 事件光流对事件密度非常敏感。
- 不只使用一个固定阈值或固定事件窗口，而是构造多密度体素，或根据空间局部密度做归一化/选择。
- 用一个小模块根据密度图选择或融合不同密度版本。

迁移方案：

- 预处理版：
  - 生成 `dense`、`normal`、`sparse` 三种体素版本。
  - 通过事件 drop、计数缩放或不同时间窗口构造密度差异。
  - 根据 density map 做融合，输出仍然是 `[T, 2, H, W]`。
- 模型 adapter 版：
  - 在 encoder 前加 `AdaptiveDensityVoxelAdapter(nn.Module)`。
  - 输入 `[B, T, 2, H, W]`，计算局部密度图，输出同形状体素。

建议文件：

- `voxelization_experiments/V3_adaptive_density/voxelization/adaptive_density.py`
- 实验入口 patch：把 adapter 插到模型输入前
- 配置：`data.voxel_method: adaptive_density`

风险：

- 额外计算开销。
- 如果 selector/fusion 太早学习，训练可能不稳定。
- 建议先做非学习型密度归一化，再做可学习 ADM。

### 3. 多窗口时间体素堆叠

来源：Temporal Event Stereo via Joint Learning with Stereoscopic Flow，ECCV 2024。代码：`https://github.com/mickeykang16/TemporalEventStereo`。

可复用思想：

- 不把每个事件窗口当作完全独立样本。
- 利用过去窗口的体素或特征，做时间聚合。
- 原论文用于 stereo，但它强调的“连续事件流、历史信息复用”对光流也有意义。

迁移方案：

- 先利用 baseline 已经有的 `num_chunks=2` 逻辑做最小实验。
- 再在实验局部 dataset wrapper 中支持 `num_chunks=3/4`。
- 为了保持模型输入稳定，有两种路线：
  - 直接把多个 chunk 拼到时间维；
  - 在 encoder 前加小型时间压缩模块，把多窗口压回 10 bins。

优点：

- 和 SDformerFlow 现有结构比较贴合。
- 低事件量窗口可能受益明显。

风险：

- 直接拼接会显著增加显存。
- 时间压缩模块会引入额外变量，需要和纯体素替换实验分开做。

### 4. V2V 风格离散体素模拟与增强

来源：V2V: Scaling Event-Based Vision through Efficient Video-to-Voxel Simulation，NeurIPS 2025。代码：`https://github.com/HYLZ-2019/V2V`。

可复用思想：

- 不一定要先生成 raw event stream，再转 voxel。
- 可以直接从普通视频生成离散 event voxel。
- 训练时随机化事件相机参数，例如正负阈值、背景噪声、hot/dead pixel，使模型对不同事件相机更鲁棒。

迁移到当前项目的两级方案：

- 短期可做：只把 V2V 的随机化思想用于 DSEC 已有体素增强。
m  - 正负阈值缩放
  - hot/dead pixel 模拟
  - 事件计数噪声
  - 正负极性不平衡
- 长期可做：用普通视频 + pseudo flow 预训练 SDformerFlow，再回到 DSEC fine-tune。

风险：

- 完整 V2V 预训练需要外部视频和 pseudo-flow pipeline，工程量较大。
- 但“体素增强子集”很适合作为中期实验，改动量可控。

### 5. 多事件表示一致性

来源：EventDance，CVPR 2024。

可复用思想：

- 单一 voxel grid 会丢失部分信息。
- 可以同时使用 stack image、voxel grid、EST 等不同事件表示，并对不同分支输出加一致性约束。

迁移方案：

- 做双分支实验：
  - 分支 A：当前 voxel grid。
  - 分支 B：event stack image 或 count frame。
- 两个分支可以共享后端 head，或者做 teacher-student 蒸馏。
- loss 中加入 prediction consistency。

风险：

- 显存上升。
- 训练变量变多。
- 建议在 V1/V2 证明单一体素改动有效后再做。

### 6. 事件激活先验 / EventBlend

来源：EventFly，CVPR 2025。

可复用思想：

- 事件激活区域本身携带强先验。
- 可以用 `activation_map = sum(abs(voxel), time, polarity)` 找出高事件活动区域。
- 在训练中对高激活区域加权，或者用事件激活图指导样本混合。

迁移方案：

- 辅助通道版：
  - 把 activation map 作为额外输入或输入 adapter 的 gating。
- loss 加权版：
  - 在光流 loss 中对事件高激活区域加权。
- 数据增强版：
  - 用 activation mask 混合两份 DSEC voxel，而不是普通 MixUp。

风险：

- 原方法主要面向跨平台 dense perception 和语义分割，直接迁移到光流不一定稳定。
- 建议先作为轻量 loss/augment 做 ablation。

### 7. STP / 自适应 Prompt 融合

来源：Efficient Event Camera Data Pretraining with Adaptive Prompt Fusion，ICCV 2025。

可复用思想：

- 在事件体素和视觉 backbone 之间加轻量时空融合模块。
- 通过局部时空感受野和时间维全局交互，缓解事件数据稀疏区域，同时保留时序结构。

迁移方案：

- 在 SDformerFlow encoder 前加一个 pre-encoder adapter：
  - 对 `[T, P, H, W]` 做局部 3D/depthwise temporal conv；
  - 对时间 bins 做全局混合；
  - residual 输出同形状体素。

风险：

- 这已经不只是体素预处理，而是改模型 input stem。
- 不建议第一阶段直接做，适合 V1/V3 有结果后再扩展。

### 8. Superevent 分组 / OpenESS

来源：OpenESS，CVPR 2024 Highlight。代码：`https://github.com/ldkong1205/OpenESS`。

可复用思想：

- 用图像超像素、teacher 特征或语义区域把事件特征分成 superevent。
- 对事件表示做更高层次的区域级约束。

迁移方案：

- 如果 DSEC 对应 frame 可用，先生成 superpixel/boundary mask。
- 用这些 mask 做：
  - 事件密度重加权；
  - 边界区域增强；
  - teacher 蒸馏辅助 loss。

风险：

- 需要 frame 对齐和 teacher 模型。
- 工程量较大，不建议作为第一轮体素化实验。

## 具体实验队列

| 实验编号 | 名称 | 配置键 | 需要新增的模块 | 是否破坏 baseline |
| --- | --- | --- | --- | --- |
| V1 | EventPillars 轻量版 | `data.voxel_method=eventpillars_lite` | `voxelization/eventpillars_lite.py`；dataset wrapper；可选预处理脚本 | 否 |
| V2 | EDCFlow 时间差分体素 | `data.voxel_method=temporal_diff` | `voxelization/temporal_difference.py`；输入 adapter | 否 |
| V3 | 可学习/无偏 binning | `data.voxel_method=learnable_binning` | raw event dataset wrapper；learnable binning module | 否 |
| V4 | OmniEvent 轻量 adapter | `data.voxel_method=omnievent_lite` | spatial branch、temporal branch、fusion adapter | 否 |
| V5 | 自适应时间窗口体素 | `data.voxel_method=adaptive_window_density` | density-guided window aggregation module | 否 |
| V6 | 离散计数体素 | `data.voxel_method=discrete_count` | 实验局部 `DSEC_dataloader/event_representations.py`；可选预处理脚本 | 否 |
| V7 | 极性分离 uint8 计数体素 | `data.voxel_method=polarity_count`, `data.voxel_dtype=uint8` | dataset wrapper，读取 uint8 后上 GPU 转 float | 否 |
| V8 | 自适应密度体素 | `data.voxel_method=adaptive_density` | `voxelization/adaptive_density.py`；训练入口 patch | 否 |
| V9 | 双窗口时间堆叠 | `data.num_chunks=2`, `data.temporal_aggregate=concat` | 实验局部 dataset wrapper/config | 否 |
| V10 | 阈值/噪声体素增强 | `data.voxel_aug.threshold_random=true` | voxel augmentation transform | 否 |

## 第一阶段建议

建议第一阶段做 V1、V2、V6 三组，因为它们能在不大改主干的情况下覆盖“新表示”和“低风险对照”：

1. V6：先做离散计数体素，作为最小改动对照。
2. V1：做 EventPillars 轻量版，验证 density/range/polarity 特征是否比普通 voxel 更适合 SNN。
3. V2：做 EDCFlow 风格时间差分体素，验证光流任务是否吃到更密集的时序差分。
4. 用相同模型和相同训练配置跑 smoke。
5. 对比：
   - train step time；
   - dataloader time；
   - validation AEE；
   - spike rate / total SOPs；
   - voxel sparsity / nonzero density；
   - 输入体素缓存大小。

如果 V1 或 V2 有明确收益，再做 V3/V4/V5。V3 是研究价值最高但工程量大的方向，V4/V5 则更像“体素化 + 输入 stem”联合优化。

## 2025-2026 来源补充

- AAAI 2025 EventPillars：`https://ojs.aaai.org/index.php/AAAI/article/view/32292`
- CVPR 2025 EDCFlow：`https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_EDCFlow_Exploring_Temporally_Dense_Difference_Maps_for_Event-based_Optical_Flow_CVPR_2025_paper.pdf`
- ICLR 2026 EventFlash：`https://openreview.net/forum?id=QuvGqzLwf6`
- ICLR 2026 Unbiased Gradient Estimation for Event Binning：`https://openreview.net/forum?id=BRj3HvQnSZ`
- ICLR 2026 EVA / Maximizing Asynchronicity：`https://openreview.net/forum?id=nGbhxxdhqz`
- AAAI 2026 OmniEvent：`https://dblp.dagstuhl.de/rec/conf/aaai/YanLWCLSLZ26.html`
- IJCAI 2025 UniCT Depth：`https://www.ijcai.org/proceedings/2025/0144.pdf`
- 2026 sparse voxel trend reference：`https://arxiv.org/abs/2603.21638`

旧版第一阶段建议保留为低风险对照：

1. 把一小部分 DSEC 样本转换成离散计数体素。
2. 用相同模型和相同训练配置跑 smoke。
3. 和当前插值体素 baseline 对比：
   - train step time
   - dataloader time
   - validation AEE
   - spike rate / total SOPs
   - voxel sparsity / nonzero density

如果 V1 在速度或稀疏性上有提升，并且精度没有明显崩，再做 V3 自适应密度体素。

## 参考来源

- CVPR 2023 RVT：`https://openaccess.thecvf.com/content/CVPR2023/html/Gehrig_Recurrent_Vision_Transformers_for_Object_Detection_With_Event_Cameras_CVPR_2023_paper.html`
- RVT 代码：`https://github.com/uzh-rpg/RVT`
- ICCV 2023 MDR/ADM：`https://openaccess.thecvf.com/content/ICCV2023/html/Luo_Learning_Optical_Flow_from_Event_Camera_with_Rendered_Dataset_ICCV_2023_paper.html`
- ECCV 2024 Temporal Event Stereo：`https://eccv.ecva.net/virtual/2024/poster/1741`
- Temporal Event Stereo 代码：`https://github.com/mickeykang16/TemporalEventStereo`
- CVPR 2024 EventDance：`https://openaccess.thecvf.com/content/CVPR2024/papers/Zheng_EventDance_Unsupervised_Source-free_Cross-modal_Adaptation_for_Event-based_Object_Recognition_CVPR_2024_paper.pdf`
- CVPR 2024 OpenESS：`https://openaccess.thecvf.com/content/CVPR2024/papers/Kong_OpenESS_Event-based_Semantic_Scene_Understanding_with_Open_Vocabularies_CVPR_2024_paper.pdf`
- OpenESS 代码：`https://github.com/ldkong1205/OpenESS`
- CVPR 2025 EventFly：`https://openaccess.thecvf.com/content/CVPR2025/papers/Kong_EventFly_Event_Camera_Perception_from_Ground_to_the_Sky_CVPR_2025_paper.pdf`
- ICCV 2025 Adaptive Prompt Fusion：`https://openaccess.thecvf.com/content/ICCV2025/html/Liang_Efficient_Event_Camera_Data_Pretraining_with_Adaptive_Prompt_Fusion_ICCV_2025_paper.html`
- NeurIPS 2025 V2V：`https://arxiv.org/abs/2505.16797`
- V2V 代码：`https://github.com/HYLZ-2019/V2V`
