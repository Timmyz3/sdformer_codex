# Efficient Transformer Survey for SDFormer-based SNN Optical Flow

## Scope and screening

This note focuses on efficient attention, efficient transformer redesign, and token/window/network compression papers from 2023-2026, with older work only used when a recent line explicitly builds on it. Priority was given to official CVF, OpenReview, ECVA, NeurIPS, IEEE, and DOI/PubMed pages. Each entry is screened on three axes:

1. Academic value
   - Top-tier venue or strong official record.
   - Explicit gains in throughput, FLOPs, latency, memory, or KV/cache traffic.
   - Mechanism is sufficiently clear to re-implement in SDFormer.
2. SNN transferability
   - Penalize methods that depend on dense softmax attention, global sorting, fine-grained gather/scatter, or high-precision continuous similarity matching.
   - Reward methods that preserve event sparsity, admit structured masks, and can reuse decisions over timestep `T`.
3. Hardware friendliness
   - Reward reduction of MACs and SRAM/DRAM traffic through structured sparsity, window/block scheduling, or reuse.
   - Penalize irregular token reordering, all-to-all matching, and unstructured per-token dynamic indexing.

## Main table

| 论文标题 | 年份 | 会议/期刊 | 任务领域 | 方法类别 | 核心思想 | 降复杂度对象 | 是否需要重训练 | 是否训练无关 / 零样本可插拔 | 是否适合视觉任务 | 是否适合事件流 / optical flow | 是否适合 SNN 化 | 是否适合硬件实现 | 可迁移到 SDFormer 的具体位置 | 代码可获得性 | 推荐优先级 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [EfficientViT: Memory Efficient Vision Transformer With Cascaded Group Attention](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_EfficientViT_Memory_Efficient_Vision_Transformer_With_Cascaded_Group_Attention_CVPR_2023_paper.html) | 2023 | CVPR | 分类/检测/分割/通用视觉 | attention 改进, architecture redesign | 指出不少 ViT 延迟瓶颈来自 reshape、elementwise 和 memory-bound MHSA，而不只是 FLOPs。用 sandwich layout 和 cascaded group attention 让注意力头分组、减少冗余 head 计算，同时改善 memory access。 | attention FLOPs, memory access, FFN FLOPs | 是 | 否 | 强 | 中 | 中 | 强 | 可替换浅层 token mixer 或 stage 内 attention block；也可借其 “single MHSA + efficient FFN” 重构高分辨率 stage | 是，官方页写明 code/models available soon，现已有官方 repo | High |
| [SparseViT: Revisiting Activation Sparsity for Efficient High-Resolution Vision Transformer](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_SparseViT_Revisiting_Activation_Sparsity_for_Efficient_High-Resolution_Vision_Transformer_CVPR_2023_paper.html) | 2023 | CVPR | 检测/分割/高分辨率视觉 | window/locality 优化, structured pruning / sparse FFN | 关键不是“找到稀疏”，而是把稀疏约束到窗口级，从而把不规则稀疏转成可批处理的 block/window 级跳算。论文直接强调窗口 attention 的块规则性可以带来真实速度收益。 | attention FLOPs, token length, memory access | 是 | 否 | 强 | 强 | 强 | 强 | 最适合迁移到 `window_scheduler`，在高分辨率事件 token 上先做窗口级裁剪，再进入后续 attention / FFN | 未在官方页直接给 repo 链接 | High |
| [FastViT: A Fast Hybrid Vision Transformer Using Structural Reparameterization](https://openaccess.thecvf.com/content/ICCV2023/papers/Vasu_FastViT_A_Fast_Hybrid_Vision_Transformer_Using_Structural_Reparameterization_ICCV_2023_paper.pdf) | 2023 | ICCV | 分类/检测/分割 | architecture redesign | 用 RepMixer 把 token mixing 改写为可重参数化结构，核心目标是降低 skip connection 带来的 memory access 成本，而不是单纯减 FLOPs。大卷积和 train-time overparameterization 只留在训练期，推理期保持低延迟。 | memory access, FFN FLOPs | 是 | 否 | 强 | 中 | 中偏强 | 强 | 可替换 SDFormer 浅层 FFN/mixer，尤其适合 patch embedding 后的低层局部 mixing | 是，官方页给出 [apple/ml-fastvit](https://github.com/apple/ml-fastvit) | Medium |
| [Token Merging: Your ViT But Faster](https://openreview.net/forum?id=JroZRaRw7Eu) | 2023 | ICLR | 分类/视频/音频/通用 | token merging/fusion | 运行时把相似 token 逐层合并，用轻量 matching 代替纯删除。优点是 off-the-shelf 可用且精度损失小；缺点是 matching 仍然是动态、全局且不够硬件友好。 | token length, attention FLOPs, memory access | 否/可选 | 是 | 强 | 中 | 中 | 中偏弱 | 可作为 `token_merger` 的算法上界；对 SDFormer 建议局部窗口化后再 merge，避免全局匹配 | 是，OpenReview 给出社区实现，已有官方实现流传 | Medium |
| [Zero-TPrune: Zero-Shot Token Pruning through Leveraging of the Attention Graph in Pre-Trained Transformers](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Zero-TPrune_Zero-Shot_Token_Pruning_through_Leveraging_of_the_Attention_Graph_CVPR_2024_paper.html) | 2024 | CVPR | 分类/通用视觉 | token pruning | 用 attention graph 上的 weighted PageRank 同时考虑 token importance 和 similarity，实现零样本 token pruning。优势是切换 pruning 配置无需重训，适合边缘部署；弱点是重要性估计仍基于 dense attention graph。 | token length, attention FLOPs | 否 | 是 | 强 | 中 | 中 | 中偏弱 | 最适合改写成 training-free `token_pruner`，但要把全局 attention graph 近似成局部时空能量图，才更适配 SNN/硬件 | 是，官方项目页 [Zero-TPrune](https://zerotprune.github.io/) | High |
| [Multi-criteria Token Fusion with One-step-ahead Attention for Efficient Vision Transformers](https://openaccess.thecvf.com/content/CVPR2024/html/Lee_Multi-criteria_Token_Fusion_with_One-step-ahead_Attention_for_Efficient_Vision_Transformers_CVPR_2024_paper.html) | 2024 | CVPR | 分类/通用视觉 | token merging/fusion | 不只按相似度 merge，而是联合 similarity、informativeness 和 fused-token size；再用 one-step-ahead attention 预测下一层重要性。比纯 pruning 更保信息，适合背景多、局部运动连续的任务。 | token length, attention FLOPs | 可选，最佳结果需训练 | 部分 | 强 | 强 | 中 | 中 | 适合 `token_merger`，用于静态背景和低纹理区域；在 optical flow 中可做“背景合并、边缘保留” | 是，官方页给出 [mlvlab/MCTF](https://github.com/mlvlab/MCTF) | High |
| [MADTP: Multimodal Alignment-Guided Dynamic Token Pruning for Accelerating Vision-Language Transformer](https://openaccess.thecvf.com/content/CVPR2024/html/Cao_MADTP_Multimodal_Alignment-Guided_Dynamic_Token_Pruning_for_Accelerating_Vision-Language_Transformer_CVPR_2024_paper.html) | 2024 | CVPR | 多模态 | token pruning | 用跨模态 alignment 约束动态 token pruning，避免某一模态中语义关键 token 被误删。思想上最值得借鉴的是“任务相关 token 保留”与“每层、每样本自适应 budget”。 | token length | 是 | 否 | 中 | 中 | 低 | 低 | 适合转化为 motion/task guided selector，例如用事件活动度 + 估计光流残差联合决定 token 保留；原始跨模态模块不建议直接移植 | 未在官方页直接给出 repo | Medium |
| [You Only Need Less Attention at Each Stage in Vision Transformers](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_You_Only_Need_Less_Attention_at_Each_Stage_in_Vision_CVPR_2024_paper.html) | 2024 | CVPR | 分类/检测/分割 | architecture redesign, attention 改进 | 每个 stage 只少量层显式计算 attention，其他层通过 attention transformation 复用已有 attention。它直接针对“不是每层都必须重新算 attention”这一点，非常适合时间连续任务。 | attention FLOPs, memory access | 是 | 否 | 强 | 强 | 强 | 强 | 最适合迁移成 `attention_reuse_unit`，在 event bins 或相邻帧间复用 attention / feature alignment | 未在官方页直接给出 repo | High |
| [Efficient Vision Transformers with Partial Attention](https://eccv.ecva.net/virtual/2024/poster/1877) | 2024 | ECCV | 分类/密集预测 | attention 改进, window/locality 优化 | 认为大量 attention weight 相似，因而只让 query 和一小部分 relevant tokens 交互。部分 attention 保留了长程建模，但比全量 window attention 更省。 | attention FLOPs | 是 | 否 | 强 | 中偏强 | 中 | 中偏强 | 适合转成 `sparse_attention_masker`，在低分辨率 stage 引入固定块局部或局部+少量全局 token 连接 | 未在官方页直接给出 repo | High |
| [Token Compensator: Altering Inference Cost of Vision Transformer without Re-Tuning](https://eccv.ecva.net/virtual/2024/poster/1918) | 2024 | ECCV | 分类/20+ 下游视觉任务 | token pruning, token merging/fusion | 关注“训练时压缩率”和“推理时压缩率”不一致导致性能掉点的问题，用小型 plugin 做 self-distillation 弥补 compression mismatch。价值在于它很适合把 token reduction 做成可插拔补偿件。 | token length | 只需小型 plugin 重训练 | 部分 | 强 | 中 | 中 | 中 | 可在 SDFormer 上作为 `structured_pruning_controller` 的补偿头，帮助不同硬件 budget 下共享主干权重 | 官方页未直接给 repo | Medium |
| [Selective Attention Improves Transformer](https://openreview.net/forum?id=v0FzmPCd1e) | 2025 | ICLR | LLM/通用 Transformer | attention 改进, selective attention | 参数无关地从 attention context 中压掉“无用元素”，降低 context buffer。它在长上下文语言建模中显著降 attention memory，但依赖标准 attention 语义。 | attention FLOPs, memory, KV/cache | 是 | 否 | 中 | 低 | 低 | 中 | 更适合作为 `sparse_attention_masker` 的概念来源，而不是直接搬到事件视觉；可借鉴 “context pruning” 而非 softmax 实现 | 未见官方代码链接 | Low |
| [Token Cropr: Faster ViTs for Quite a Few Tasks](https://openaccess.thecvf.com/content/CVPR2025/html/Bergner_Token_Cropr_Faster_ViTs_for_Quite_a_Few_Tasks_CVPR_2025_paper.html) | 2025 | CVPR | 分类/分割/检测/实例分割 | token pruning | 用辅助预测头学习任务相关 token selection，但推理时移除辅助头，使吞吐接近随机 pruner 的速度。最有价值的点是它同时报告多任务和真实 speedup，而不是只看 FLOPs。 | token length, attention FLOPs, latency | 是 | 否 | 强 | 强 | 中偏强 | 中偏强 | 适合 `token_selector + token_pruner`，特别适合 optical flow 中用 motion boundary / residual 训练一个轻量 task-aware selector | 官方页未直接给 repo | High |
| [DivPrune: Diversity-based Visual Token Pruning for Large Multimodal Models](https://openaccess.thecvf.com/content/CVPR2025/html/Alvar_DivPrune_Diversity-based_Visual_Token_Pruning_for_Large_Multimodal_Models_CVPR_2025_paper.html) | 2025 | CVPR | 多模态 | token pruning | 把 token pruning 写成 Max-Min Diversity 问题，目标不是只留“最重要”的 token，而是留“更具代表性”的子集。它更适合高压缩率场景，因为能抑制 retained token 间冗余。 | token length, latency, memory access | 否 | 是 | 中 | 中 | 中偏低 | 中偏低 | 可以作为 `token_selector` 的次级准则，在边缘/运动区域之外再加一个 diversity 惩罚，避免全部 token 聚到同一运动团块 | 是，官方页说明 code available | Medium |
| [BHViT: Binarized Hybrid Vision Transformer](https://openaccess.thecvf.com/content/CVPR2025/html/Gao_BHViT_Binarized_Hybrid_Vision_Transformer_CVPR_2025_paper.html) | 2025 | CVPR | 分类/边缘视觉 | architecture redesign, structured pruning / sparse FFN | BHViT 不是单纯 token reduction，而是把局部信息交互、shift 操作、层级聚合和 attention matrix binarization 一起改成适合二值化的形式。对 SNN/硬件最关键的是它证明了 attention/MLP 可以往低比特、shift-friendly 方向重构。 | FFN FLOPs, MACs, memory access | 是 | 否 | 强 | 中偏强 | 强 | 强 | 适合迁移到 FFN/mixer 和硬件量化路径；可把其 shift-style binary MLP 思想用于 SDFormer 的低层局部 mixing | 是，官方页给出 [IMRL/BHViT](https://github.com/IMRL/BHViT) | High |
| [Twilight: Adaptive Attention Sparsity with Hierarchical Top-p Pruning](https://openreview.net/forum?id=Ve693NkzcU) | 2025 | NeurIPS | LLM | attention 改进 | 用 hierarchical top-p 让 sparse attention 的预算随输入自适应变化，而不是静态 top-k。优点是可大幅减 KV token；缺点是自适应 top-p 仍较依赖 dense score 统计和不规则选择。 | attention FLOPs, KV cache, memory access | 否 | 是 | 低 | 低 | 低 | 中 | 不建议直接移植 attention 形式，但可把“adaptive budget decision”迁移到 `structured_pruning_controller` | 未见官方代码链接 | Low |
| [Spark Transformer: Reactivating Sparsity in Transformer FFN and Attention](https://openreview.net/forum?id=o4zN34ahEK) | 2025 | NeurIPS | LLM/通用 | structured pruning / sparse FFN, attention 改进 | 在 FFN 和 attention 两边同时重新引入 activation sparsity，亮点是用 statistical top-k 近似常规 top-k，避免排序带来的训练和硬件开销。它比大多数“稀疏但难算”的方法更接近硬件可落地。 | FFN FLOPs, attention FLOPs | 是 | 否 | 中 | 中 | 中偏强 | 强 | 适合转为 `structured_pruning_controller` 或后续 stage FFN gate；特别适合低分辨率、高通道瓶颈层 | 未见官方代码链接 | Medium |
| [Efficient Visual Transformer by Learnable Token Merging](https://doi.org/10.1109/TPAMI.2025.3588186) | 2025 | TPAMI | 分类/通用视觉 | token merging/fusion, architecture redesign | 把 token merging 做成可学习的 mask 模块，直接替换已有 Transformer block。相比启发式合并，它更强调把 merge 机制内生到 block 中。 | token length, inference time | 是 | 否 | 强 | 中偏强 | 中 | 中偏强 | 对 SDFormer 最有价值的是“merge 成为 block 内生机制”而不是外挂后处理；适合低分辨率 stage 的背景 token 融合 | 官方 DOI/PubMed 未给 repo | Medium |
| [Top-Theta Attention: Sparsifying Transformers by Compensated Thresholding](https://openreview.net/forum?id=YBgjDBYPzz) | 2026 | ICLR 2026 submitted | LLM | attention 改进 | 用 per-head 静态阈值代替 top-k，靠校准阈值保留每行近似常数个显著元素，并用补偿项弥补精度损失。阈值法在硬件上比排序更自然，但它依然建立在 attention score 显式可得上。 | attention FLOPs, KV cache, memory access | 否 | 是 | 低 | 低 | 低 | 中偏强 | 可为 `sparse_attention_masker` 提供 threshold 而非 sort 的硬件友好思路，但不建议直接用在当前 SNN optical flow 主线 | 未见官方代码链接 | Low |
| [PPT: Token Pruning and Pooling for Efficient Vision Transformers](https://arxiv.org/abs/2310.01812) | 2023 | arXiv preprint | 分类/通用视觉 | token pruning, token merging/fusion | 代表性地把 pruning 和 pooling 联合起来，而不是只删 token。它的价值在于提醒我们“删除”和“压缩/汇聚”应分层使用，尤其适合背景大于运动区的视觉场景。 | token length, attention FLOPs | 是 | 否 | 强 | 强 | 中偏强 | 中偏强 | 最适合做 `token_pruner + token_merger` 联合模块：先删极低活动窗口，再对保留背景做 pooling/merge | 是，可从 [Papers with Code](https://paperswithcode.com/paper/ppt-token-pruning-and-pooling-for-efficient) 追到公开实现；但非顶会/顶刊正式发表 | Medium |

## Three-layer shortlist

### C1. Academic value shortlist

优先级最高的一组是：

- EfficientViT, SparseViT, Zero-TPrune, MCTF, LaViT, PartialFormer, Token Cropr, BHViT, Spark Transformer
- 原因：机制清晰、改造点明确、官方页面能直接确认复杂度收益或速度收益

### C2. SNN transferability shortlist

最适合 SNN optical flow 的不是“最强稀疏 attention”，而是以下几类：

- SparseViT: 结构化窗口裁剪，不依赖 dense QK 排序，天然兼容事件空间稀疏
- LaViT: attention 复用与 optical flow 的时间连续性高度一致
- Token Cropr: task-aware token selection 可映射为运动边缘/残差引导
- MCTF / PPT / LTM: 背景 token 融合比纯删除更适合保持流场连续性
- BHViT: 对硬件和低比特友好，适合作为 SNN/脉冲近似的架构侧支撑

高风险项：

- Selective Attention, Twilight, Top-Theta
  - 依赖标准 attention score、softmax 语义或 KV cache 读写模型
  - 对 SNN 化帮助不直接，且动态索引不规则
- MADTP
  - 跨模态对齐模块本身与单模态事件流不匹配

### C3. Hardware friendliness shortlist

最值得优先落地到软硬件协同路径的方案：

- SparseViT: window/block 粒度规则，最利于 tile 化和 SRAM 重用
- LaViT: 直接减少 attention 计算频次，利于 controller 级跳算
- EfficientViT / FastViT: 强调 memory access 而非只看 FLOPs，适合 RTL 重构
- BHViT: 低比特和 shift-style 操作最接近 SNN/edge 硬件
- Spark Transformer: statistical top-k 比标准 top-k 更接近可综合实现

## Required comparison table

| paper | idea | complexity target | retraining needed | SNN-fit | HW-fit | SDFormer insertion point | priority |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [SparseViT](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_SparseViT_Revisiting_Activation_Sparsity_for_Efficient_High-Resolution_Vision_Transformer_CVPR_2023_paper.html) | 窗口级结构化稀疏 | attention, memory, latency | Yes | High | High | `window_scheduler` before stage attention | High |
| [LaViT](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_You_Only_Need_Less_Attention_at_Each_Stage_in_Vision_CVPR_2024_paper.html) | 减少 attention 计算频次并复用 | attention, memory | Yes | High | High | `attention_reuse_unit` across timesteps/stages | High |
| [Token Cropr](https://openaccess.thecvf.com/content/CVPR2025/html/Bergner_Token_Cropr_Faster_ViTs_for_Quite_a_Few_Tasks_CVPR_2025_paper.html) | 任务相关 token 选择 | token length, latency | Yes | High | Medium-High | `token_selector` + `token_pruner` after encoder / shallow stage | High |
| [MCTF](https://openaccess.thecvf.com/content/CVPR2024/html/Lee_Multi-criteria_Token_Fusion_with_One-step-ahead_Attention_for_Efficient_Vision_Transformers_CVPR_2024_paper.html) | 多准则 token fusion | token length, attention | Optional | Medium | Medium | `token_merger` on background / low-motion windows | High |
| [Zero-TPrune](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Zero-TPrune_Zero-Shot_Token_Pruning_through_Leveraging_of_the_Attention_Graph_CVPR_2024_paper.html) | 零样本 attention-graph pruning | token length, attention | No | Medium | Medium-Low | training-free `token_pruner` approximation | High |
| [EfficientViT](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_EfficientViT_Memory_Efficient_Vision_Transformer_With_Cascaded_Group_Attention_CVPR_2023_paper.html) | 分组注意力 + sandwich block | memory, attention | Yes | Medium | High | stage/block redesign in high-res encoder | High |
| [PartialFormer](https://eccv.ecva.net/virtual/2024/poster/1877) | partial attention | attention | Yes | Medium | Medium-High | `sparse_attention_masker` in low-res stages | High |
| [BHViT](https://openaccess.thecvf.com/content/CVPR2025/html/Gao_BHViT_Binarized_Hybrid_Vision_Transformer_CVPR_2025_paper.html) | 二值化友好混合 ViT | MACs, memory | Yes | High | High | low-bit FFN/mixer redesign | High |
| [Spark Transformer](https://openreview.net/forum?id=o4zN34ahEK) | sparse FFN + sparse attention | FFN, attention | Yes | Medium | High | `structured_pruning_controller` for bottleneck layers | Medium |
| [Token Compensator](https://eccv.ecva.net/virtual/2024/poster/1918) | compression mismatch compensation | token length | Small plugin only | Medium | Medium | budget-conditioned plug-in compensation head | Medium |
| [DivPrune](https://openaccess.thecvf.com/content/CVPR2025/html/Alvar_DivPrune_Diversity-based_Visual_Token_Pruning_for_Large_Multimodal_Models_CVPR_2025_paper.html) | diversity-aware pruning | token length, latency | No | Medium | Medium-Low | diversity-aware selector for retained motion tokens | Medium |
| [Selective Attention](https://openreview.net/forum?id=v0FzmPCd1e) | context pruning in attention | memory, attention | Yes | Low | Medium | concept only, not direct backbone transplant | Low |
| [Twilight](https://openreview.net/forum?id=Ve693NkzcU) | adaptive top-p sparse attention | KV cache, attention | No | Low | Medium | controller inspiration only | Low |
| [Top-Theta](https://openreview.net/forum?id=YBgjDBYPzz) | thresholded sparse attention | attention, V-cache | No | Low | Medium-High | threshold-style mask calibration only | Low |

## Transfer conclusions for SDFormer

### Best direct-transfer families

- Structured window scheduling and structured token pruning
  - Best representatives: SparseViT, Token Cropr, Zero-TPrune
  - Reason: they attack the real cost driver in high-resolution event flow, namely active token/window count
- Attention reuse and fewer-attention stage design
  - Best representative: LaViT
  - Reason: event streams are temporally correlated, so reusing masks or feature alignments across `T` is natural
- Token fusion instead of token deletion
  - Best representatives: MCTF, PPT, LTM
  - Reason: optical flow is sensitive to edge continuity and background-motion contrast; fusion is less destructive than hard pruning
- Hardware/low-bit friendly block redesign
  - Best representatives: EfficientViT, FastViT, BHViT
  - Reason: these lines reduce memory traffic and use regular operators, which matter more than nominal FLOPs on real accelerators

### Lower-priority families

- LLM sparse attention lines based on KV cache management
  - Selective Attention, Twilight, Top-Theta
  - Useful as controller inspiration, but not as direct SNN optical-flow blocks
- Multimodal alignment-guided pruning
  - MADTP and DivPrune
  - Useful for “relevance-aware” and “diversity-aware” thinking, but direct migration requires simplifying away multimodal and combinatorial machinery

## Mapping to implemented module pool

The repository now contains `src/models/modules/external_inspirations/` with the following abstracted transfer targets:

- `token_selector.py`
  - `motion_guided_selector`
  - Main inspirations: Token Cropr, DivPrune, MADTP
- `token_pruner.py`
  - `graph_token_pruner`
  - Main inspirations: Zero-TPrune, SparseViT
- `token_merger.py`
  - `similarity_token_merger`
  - Main inspirations: MCTF, ToMe, PPT, LTM
- `window_scheduler.py`
  - `activity_window_scheduler`
  - Main inspirations: SparseViT, structured window pruning lines
- `sparse_attention_masker.py`
  - `block_sparse_attention_masker`
  - Main inspirations: PartialFormer, Selective Attention, Top-Theta
- `attention_reuse_unit.py`
  - `temporal_attention_reuse`
  - Main inspirations: LaViT
- `structured_pruning_controller.py`
  - `structured_latency_controller`
  - Main inspirations: Spark Transformer, BHViT, real-latency-aware pruning lines

## Notes on source reliability

- All rows except `PPT` use official CVF, OpenReview, ECVA, DOI, or PubMed records.
- `Top-Theta` is included because the user explicitly requested it, but it is only an ICLR 2026 submission as of March 19, 2026.
- `PPT` is included because the user explicitly requested pruning+pooling coverage, but it is an arXiv preprint rather than a formal top-tier publication.
