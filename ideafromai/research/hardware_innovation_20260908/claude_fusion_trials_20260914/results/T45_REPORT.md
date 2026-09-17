# T45：**整个硬件映射的实测成本分布**（2026-09-17）

> **用户指示（2026-09-17）**：*"还有得注意整个硬件映射最贵的部分是什么，利用从其他论文工作
> 学到的方法去尝试加速这些最贵的，而不是加速你所学的论文中加速的部分（如果可以迁移的话），
> 相当于迁移改进得到新idea，针对这些贵的部分着重探索调研挖掘实验筛选"*
>
> 这句话直接点破了本项目的一个结构性风险：**T1–T44 的 C1 线（组级 BFP + 位平面证书 +
> 逐 t 退休）与"抄全 BitFair"这条线，都只作用在 `encoders.swin3d.layers.*.swin_blocks.*.mlp`
> 的 fc1→sn2 门路链接上。**在优化之前必须先量：这到底占整张网络映射的多少。
> 本文是这次测量的结果。

## 1. 方法（实测，不是估计）

| 项 | 值 |
|---|---|
| 模型 | `MS_SpikingformerFlowNet_en4`，部署配置 `dsec_c12_alpha0125_ep29_resume5_20260830.yml`（= T19/T44 的捕获模型） |
| 输入 | (1, 10 bins, 2 pol, **480×640**)，T=10；patch-embed 后工作分辨率 240×320 |
| MAC | **forward hook 在真实张量上数**（不是解析估计），逐算子 `out.numel() × in_features`（Conv 为 `ic·k_h·k_w`），含 `Conv2d` / `ConvTranspose2d` / `Linear` |
| 脚本 | `t45_mapping_cost.py` → `results/t45_mapping_cost.json`（含逐算子明细与活动率） |

两个口径**必须分开报**：

1. **dense 等效 MAC**：所有神经元全发放时的乘加数。回答"做成稠密阵列要花多少"。
2. **活动加权 SOP**：dense MAC × 该算子**输入**的发放率。SNN 加速器的标准口径，
   回答"spike-driven 阵列实际算多少"。

⚠ **活动率来源的诚实边界**：SOP 列用的逐 neuron 发放率来自
`neuron_experiments/_profiles/sops_20260511_120258/layer_firing_rates.csv`，
**那是另一个 checkpoint（tokenmix_pool）的 profile，不是 ep34 部署权重**。
架构相同（105 个层名逐一对上、要素数逐层核过），但**活动率必须用我们的权重复测才算数**。
**dense MAC 与参数两列不受此影响**（纯结构量，对我们的配置精确）。

## 2. 结果

### 2.1 超级分组（480×640, T=10）

| 超级组 | dense MAC | MAC% | 活动加权 SOP | **SOP%** | 参数 | 参数% |
|---|---:|---:|---:|---:|---:|---:|
| **decoder（4 个 ConvTranspose）** | 320.33 G | 33.85% | 65.75 G | **55.58%** | 7.14 M | 13.2% |
| **frontend（patch_embed）** | 307.00 G | 32.44% | 16.35 G | 13.82% | 0.47 M | 0.9% |
| enc.swin stage2（6 block） | 124.31 G | 13.13% | 14.40 G | 12.18% | 10.91 M | 20.2% |
| **bottleneck resblocks（2×768ch）** | 63.70 G | 6.73% | 9.36 G | 7.91% | **21.23 M** | **39.2%** |
| enc.swin stage0（2 block） | 42.80 G | 4.52% | 5.34 G | 4.52% | 0.28 M | 0.5% |
| enc.swin stage3（2 block） | 44.24 G | 4.67% | 4.35 G | 3.68% | 12.98 M | 24.0% |
| enc.swin stage1（2 block） | 43.79 G | 4.63% | 2.72 G | 2.30% | 1.11 M | 2.1% |
| pred heads | 0.21 G | 0.02% | 0.02 G | 0.02% | 0.002 M | 0.0% |
| **合计** | **946.4 G** | 100% | **118.3 G** | 100% | **54.12 M** | 100% |

**enc.swin3d 全部 = 26.82 G SOP（22.7%）**，其中 MLP 支 = 19.20 G（16.2%）。
而 C1 动的是 MLP 里 **fc1 输出 → sn2 门 → fc2 输入** 这条**传输**链路——
**比 MLP 的 16.2% 还小**。

### 2.2 三个轴上的"最贵"不是同一个东西

| 轴 | 第一名 | 第二名 | 合计 |
|---|---|---|---|
| **dense 等效 MAC** | frontend 32.4% | decoder 33.9% | **66.3%**（encoder 仅 27.0%） |
| **活动加权 SOP** | **decoder 55.6%** | frontend 13.8% | **69.4%**（encoder 仅 22.7%） |
| **权重存储（参数）** | **bottleneck 39.2%** | enc.stage3 24.0% + stage2 20.2% | **83.4%**（frontend+decoder 仅 14.1%） |

⇒ **没有单一"最贵的部分"**：**算力（尤其 spike-driven 口径）贵在 decoder + frontend；
权重存储贵在 encoder stage2/3 + bottleneck。** 一句话概括：
**encoder 是"权重大户"，decoder/frontend 是"运算大户"。**

### 2.3 逐条读数

- **decoder.3 单条就是全网络 SOP 第一名（29.20 G，24.68%）**：`ConvTranspose2d(194→96, k=3)`
  在 **240×320 全分辨率 × T=10** 上跑，输入发放率 **0.2268**。
- **decoder 四个层的发放率 0.154 / 0.190 / 0.227 / 0.227，是全网络均值（0.069）的 2–3×**。
  ⇒ **decoder 是"穿着 SNN 外衣的稠密 CNN"**：分辨率最高、活动率最高、单层 channel 数少但
  空间量大。任何针对"SNN 稀疏性"的机制在这里都拿到最少收益。
- **frontend.patch_embed.residual_encoding 是全网络 dense MAC 第一名（254.80 G，26.92%）**：
  4 个 `Conv2d(96→96, k=3)` 在 240×320 上 × T=10 —— **同一个卷积被时间步重复了 10 次**。
  但它的输入发放率只有 0.033–0.068 ⇒ SOP 只有 11.79 G（9.97%）。
  **⇒ frontend 是"稠密代价被稀疏活动掩盖"的典型**：pipeline 延迟/面积上贵，但 SOP 账上看不见。
- **bottleneck resblocks 参数 21.23 M = 39.2%**（2 个 resblock × 4 个 `Conv2d(768→768,3×3)`），
  但 SOP 只 9.36 G（7.91%）——**纯权重存储成本**。
- **encoder stage2/3 MLP 参数 16.5 M = 30.5%**——正是 C1 所在的层，**它们的价值主要不在 SOP
  而在权重存储**。
- **注意力支的 SOP 几乎为 0**（`attn.proj` 四层全 0.000 G）：profile 里 `attn.attn_sn` 的发放率
  **在全部 12 个 block 上都是 0.0000**。⚠ 这条**必须在我们的权重上复测**——若成立，说明
  注意力输出被门完全压住，注意力支在 SOP 账上是死的；若不成立，是本 profile 的伪影。

## 3. 结论：C1/BitFair 覆盖了多大一块

| 口径 | C1 相关部分 | 占全网络 |
|---|---|---|
| 12 个 MLP 的 fc1 计算 | 14.06 G SOP | **11.9%** |
| 整个 encoder MLP（fc1+fc2） | 19.20 G SOP | **16.2%** |
| 整个 encoder.swin3d | 26.82 G SOP | **22.7%** |
| encoder.swin3d dense MAC | 255.1 G | **27.0%** |

**而 C1 实际优化的是 fc1→sn2 这条传输链路——比上表任一行都更小。**
⇒ T1–T44 的全部工作 + "抄全 BitFair"这条线，**作用面 ≤ 全网络的 ~23%**，
且是**这 23% 里的一个子项**。这不是说 C1 是错的（它判决零差、有精确性定理），
而是说：**它不可能是论文的"乘性加速"主叙事**，除非能证明那 23% 就是整机瓶颈
（当前证据相反：decoder 与 frontend 合计 69.4% SOP / 66.3% MAC）。

> ⚠ **与既有文档的冲突必须记录**：`SDformerFlow_MODULE_DATAFLOW_MAP_ZH.md` §12 写
> "encoder.swin3d SOPs share = **76.43%**"，与本文实测 **22.7%** 相差 3.4×。
> 该数来自另一个 checkpoint 与另一套（未公开的）归组口径，**本文以实测为准**，
> 但差异本身要标出来，不能悄悄覆盖。

## 4. 诚实边界

- **活动率是借来的**（另一 checkpoint）。SOP% 只能当**量级判断**，
  不能写进论文。dense MAC% 与参数% 对本配置精确。
- 未计：神经元内部膜更新/阈值比较（逐元素 O(神经元数)，与 MAC 不同量纲）、
  BN/layernorm、以及**所有数据搬运**。真正的硬件成本还需要一个显式的带宽/存储模型
  （GustavSNN 的 Dr/Dw 计数式），本文只给了算力与权重两个下界。
- 分辨率 480×640 T=10 单样本。变分辨率不改**占比**结论（各层都按分辨率缩放），
  但 decoder/stage3 的比例会随输入尺寸变（decoder 在全分辨率上，encoder 在最粗尺度上）。

## 5. 下一步

1. **T45b（便宜、必做）**：用**我们自己的 ep34 权重**在 sd5ai 复测逐 neuron 发放率，
   把 SOP% 从"借来的量级"变成"我们的数字"。（复用 `tools/profile_sops.py` 的 hook 口径，
   但要把它的 `dense_ops × global_firing_rate` 粗模型换成逐算子输入率。）
2. **T45c（本报告的真正目的）**：针对 **decoder（55.6% SOP）** 与 **frontend（32.4% MAC）**
   做跨论文机制筛选——**不再**把预算花在 encoder 的传输链路上。
   筛选纪律沿用既有两条：① 照抄→根因→改进→实测（T38 范式）；② **任何优化都必须带精度约束**
   （T40f 教训：裸的供数/稀疏目标一定收敛到退化解）。

## 6. 文件

- `t45_mapping_cost.py` → `results/t45_mapping_cost.json`（逐算子 MAC/参数/活动率明细）
- 本文件：`results/T45_REPORT.md`
