# 统一公开工作 / 开源实现 / 开放 idea 入口

这是可直接用于融合试做的导航汇总，不是等待全部文献核完才准实验的门槛。先从适配接口挑组合，在隔离原型中试，再把真实结果写回执行计划；历史负结果只约束当时的布局和身份。

当前聚合得到 **763 个保守归并的工作实体、88 个仓库 URL、305 条结构化 idea 视图、720 份 Markdown 入口**。另有 **15 组同名候选尚未消歧**，所以实体数不是已最终核净的论文数。实体包含论文、代码、内部别名等；idea 视图有重叠，不能称 305 个独立新机制。全树索引是机器导航，不是本人全文阅读计数。

原表中“先等某门、暂不训练/RTL、停标题”等为当时的状态与建议，不构成本轮继续试做的限制。用户当前已授权先把适配组合做成隔离原型、测性能再决定保留；借入来源和本地新增部分仍需分别列明。

2026-09-12用户取消相对父学生+0.005的精度淘汰门，改以同评价设置的原SDformerFlow为质量对照。历史来源中的旧门判断不再作为当前裁决；见[现行精度规则](../ACCURACY_POLICY.md)、[同帧基线与重评](../accuracy_baseline/README.md)。新颖性、同资源净服务和完整评价范围仍分别判断。

## 从哪里用

- [WORKS.md](WORKS.md) / [works.csv](works.csv)：全部归并实体、来源、阅读和尝试状态、A/B/X、未试接口、代码身份。
- [REPOSITORIES.md](REPOSITORIES.md) / [repositories.csv](repositories.csv)：全树仓库入口。`author_code_reported` 为原调研明确报告的作者代码；`third_party_reported` 为第三方声明；`repository_link_owner_unverified` 表示有链接但作者关系未核。有论文无代码使用 `code_not_located_in_aggregated_sources`，不等于不存在代码。
- [VENUES.md](VENUES.md)：与本表同步生成的会刊论文实体计数；补表阅读层级按原来源保留，不将实体数充作全文数。
- [venue_coverage.csv](venue_coverage.csv)：当前比较审计明确记录的会刊/年份检索边界、未取得正文与未试接口；不是全球覆盖证明。
- [idea_views.csv](idea_views.csv)：逐篇融合提取表、32 卡筛查、科研流程记录和 Pro 候选段落集中查询；状态带来源/时间，不假装统一裁决。
- [document_index.csv](document_index.csv)：全树 Markdown 的标题、章节入口；包括 Grok 新增目录、Pro、原始 Card 和具体试验结果。
- [source_records.csv](source_records.csv)、[source_links.csv](source_links.csv)、[identity_variants.csv](identity_variants.csv)：追溯归并与链接。[possible_aliases.csv](possible_aliases.csv) 保留未消歧的同名记录；代码实体不与论文实体合并。
- [counts.json](counts.json)：覆盖计数，含实体类型与 idea 视图类型。

## 归并方法与边界

自动归并依据同一源 UID、相同完整标题、论文唯一相同 arXiv 编号、或代码唯一相同仓库 URL。另对本轮实际核对了名称、刊会/年份、机制和原记录的跨审计别名，在脚本中逐个列明 MAIN/MUSHA ID；Pro 的精确同名同年记录可以挂回唯一库存项。CICC补表的Zhang完整题名与旧MAIN-R270占位明确归并，C-DNN短名线索与补表完整题名明确归并；旧来源和阅读冲突仍保留。没有模糊题名相似度归并。共享 arXiv 等别名保留在 `identity_variants.csv`，刊会冲突/多代作品族等未消歧项保留为不同实体。正文引用了很多论文的内部 Card 不参与论文 arXiv 归并。公开工作与开源实现是两个维度；GustavSNN-public-mirror 的旧 `open_source` 标签只是论文镜像/本地迁移入口，未被提升为作者完整工件。

旧材料中的身份冲突保留为历史来源陈述；本轮统一身份是 **AT-LIF={0,θ}，推理 θ 可吸收进下一层 W**，连续 PSN/PED/残差义务另算。旧 ep35 统计和旧 AEE 不能直接变成 ep34 或当前学生结果。这里不改写源材料，不生成新的性能/新颖性分数。

## 直接聚合的库存

| 输入 | 本轮读取条数 | 作用 |
|---|---:|---|
| [research/hardware_innovation_20260908/survey_ab_fusion_20260910/literature_merged_300plus.json](/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/survey_ab_fusion_20260910/literature_merged_300plus.json) | 732 | 原记录/注释；非新增全文阅读 |
| [research/literature_audit_20260909/literature_inventory.json](/home/zhumd/work/sdformer_codex/ideafromai/research/literature_audit_20260909/literature_inventory.json) | 358 | 原记录/注释；非新增全文阅读 |
| [research/literature_audit_mushaolong_20260909/literature_inventory.json](/home/zhumd/work/sdformer_codex/ideafromai/research/literature_audit_mushaolong_20260909/literature_inventory.json) | 127 | 原记录/注释；非新增全文阅读 |
| [research/hardware_innovation_20260908/literature/direct_priors_20260909/method_inventory.json](/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/literature/direct_priors_20260909/method_inventory.json) | 2 | 原记录/注释；非新增全文阅读 |
| [research/audit_comparison_20260909/CICC_SUPPLEMENT.md](/home/zhumd/work/sdformer_codex/ideafromai/research/audit_comparison_20260909/CICC_SUPPLEMENT.md) | 18 | 原记录/注释；非新增全文阅读 |
| [research/audit_comparison_20260909/venue_supplement_other.json](/home/zhumd/work/sdformer_codex/ideafromai/research/audit_comparison_20260909/venue_supplement_other.json) | 15 | 原记录/注释；非新增全文阅读 |
| [research/literature_audit_20260909/LITERATURE_INVENTORY.md](/home/zhumd/work/sdformer_codex/ideafromai/research/literature_audit_20260909/LITERATURE_INVENTORY.md) | 6 | 原记录/注释；非新增全文阅读 |
| [gptpro/ChatGPTpro-#硬件idea深挖0911.md](/home/zhumd/work/sdformer_codex/ideafromai/gptpro/ChatGPTpro-#硬件idea深挖0911.md) | 12 | 原记录/注释；非新增全文阅读 |
| [gptpro/ChatGPTpro-#第二轮跨领域调研：.md](/home/zhumd/work/sdformer_codex/ideafromai/gptpro/ChatGPTpro-#第二轮跨领域调研：.md) | 27 | 原记录/注释；非新增全文阅读 |
| [research/hardware_innovation_20260908/open_fusion_execution/literature_followup.md](/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/open_fusion_execution/literature_followup.md) | 2 | 原记录/注释；非新增全文阅读 |
| [research/hardware_innovation_20260908/open_fusion_execution/catalog/local_execution_priors.json](/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/open_fusion_execution/catalog/local_execution_priors.json) | 13 | 原记录/注释；非新增全文阅读 |
| [research/hardware_innovation_20260908/survey_ab_fusion_20260910/idea_extract_per_paper.csv](/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/survey_ab_fusion_20260910/idea_extract_per_paper.csv) | 252 | 原记录/注释；非新增全文阅读 |

## 本轮执行先验补录

下列补录来自已使用的实际先验与强对照；原库存能识别的条目归回原ID，其余续排。底座/接口近邻与本地新增机制分开；阅读层级不自动升级。

| ID | 原工作与主来源 | 实际阅读范围 | 执行角色 / 未试部分 |
|---|---|---|---|
| W0755 | [Finch: Finch: Sparse and Structured Tensor Programming with Control Flow](https://commit.csail.mit.edu/papers/2025/Finch-OOPSLA-2025.pdf) | 主来源题名、摘要与格式说明已核；此前仅部分阅读库/格式说明，未全文深读。 | 纳入非零默认值和压缩执行的公共底座；完整编译器未迁入，不单独构成X；完整Finch编译器及原作者工件；动态BN的真实FP32舍入顺序、默认向量更新和双消费者服务的公平移植。 |
| W0136 | [BNFF / BN restructuring: Restructuring Batch Normalization to Accelerate CNN Training](https://proceedings.mlsys.org/paper_files/paper/2019/file/9db64c20dee0011899dfdf200e61ef35-Paper.pdf) | 本轮核官方摘要；已有MAIN-R164记录原文方法已读和有限本地BN费用迁移，保留来源差别，不借本次补录宣称新增全文阅读。 | 已补完整域单遍统计与真实BN+PED同Engine强对照；X=0，实际diverse10另列；原作者整个训练系统未迁；当前native投影到统计的全域生产仍未重放；新FP32公式的成对AEE核验单列。 |
| W0155 | [FlexAcc: FlexAcc: Accelerating Batch Normalization through GPU-FPGA Integration](https://jiemingyin.github.io/docs/ISCAS2025_FlexAcc.pdf) | 作者PDF前向/数据流章节已读；本轮重点核§III-B与Fig4，既有记录还报告III-A/B/C。完整异构工件未迁入，未称全文系统复现。 | 完整192000×96单遍sum/sumsq、同Engine normalize+PED已经执行；普通common baseline，不是X；原GPU-FPGA平台未迁；本地完整原生producer费用仍未闭；AEE成对检查见onepass/aee_check。 |
| W0756 | [Vecim: Vecim: A 289.13GOPS/W RISC-V Vector Co-Processor with Compute-in-Memory Vector Register File for Efficient High-Performance Computing](https://doi.org/10.1109/ISSCC49657.2024.10454387) | 已读原始ISSCC digest全文及Fig30.6.1–7；原论文PDF为第三方镜像承载，会议官方节目核身份。 | 纳入普通源驻留MAC、分队列发射及依赖转发的强底座；不是新标题，未迁原CIM宏；原Vecim实现未取得；Ara仅在论文中被识别，未checkout。未迁其定制全数字计算SRAM，尚须本地I24×INT16/Acc48/RNE宽度及端口/队列适配，原8位周期/PPA不可移用。 |
| W0757 | [SPIDER: Efficiently Detecting Inclusion Dependencies](https://hpi.de/oldsite/fileadmin/user_upload/fachgebiete/naumann/publications/PDFs/2007_bauckmann_efficiently.pdf) | 已读作者原文§2.2 / Algorithm1；本轮再次核首页正式题名。算法相关章节阅读，不称全篇数据库实验复现。 | 列流包含关系RTL叶已验；完整K16原作子集语义也已迁入覆盖全部K864，对允许重叠TCAM仅有约0.1%增量；原DB排序、去重、游标和复合/部分IND扩展未迁；本地输入已有列集合，仅适配包含关系更新。完整同资源消费者服务仍应由主线程模型评价，叶子RTL通过不等于整链加速。 |
| W0758 | [Maestro low-rank LoD: Maestro: Uncovering Low-Rank Structures via Trainable Decomposition](https://proceedings.mlr.press/v235/horvath24a.html) | 官方PMLR题名/摘要及LoD相关方法由执行agent核读；未全文/原工件复现。 | 纳入有序/嵌套秩强对照；原坐标无训渐进PED已实际试过，LoD训练未实施；作者完整LoD/group-lasso训练；固定rank同预算恢复与可部署连续组预测。 |
| W0759 | [FLRC: FLRC: Fine-grained Low-Rank Compressor for Efficient LLM Inference](https://aclanthology.org/2025.emnlp-main.755/) | ACL出版身份与摘要核实；agent读取progressive decoding相关段落，未迁完整LLM工件。 | 动态低秩的直接先验；当前连续PED渐进原坐标无训布局已测无可用早停机会；作者完整分层分配/LLM解码策略，本网训练有序表示与付费判定。 |
| W0760 | [SVD-LLM: SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model Compression](https://proceedings.iclr.cc/paper_files/paper/2025/file/3104e1ab39875cf54fe1eb4473e7c5a1-Paper-Conference.pdf) | agent读取原作§3.1白化构造，根代理核官方出版题名；§3.2逐层恢复未迁，不称完整复现。 | 已重构真实PED潜空间，原RNE/sat/宽度执行与同端口R24服务已测；真实AEE另列；完整逐层模型更新/恢复；嵌套训练和共同连续消费者的动态完成。 |
| W0289 | [MotionDeltaCNN: MotionDeltaCNN: Sparse CNN Inference of Frame Differences in Moving Camera Videos with Spherical Buffers and Padded Convolutions](https://openaccess.thecvf.com/content/ICCV2023/html/Parger_MotionDeltaCNN_Sparse_CNN_Inference_of_Frame_Differences_in_Moving_Camera_ICCV_2023_paper.html) | 原文§3.1/3.2和buffer/边界章节导读、官方ICCV出版页核实；未复现作者CUDA/全部实验。 | 纳入运动残差强对照并实际尝试同推理T10本块参考；当前该接口源项机会不足2%；真实视频帧间、跨tile/球面buffer/新表示或训练；不得把可控输入摘要位移冒充光流预测器。 |
| W0358 | [LoopTree: LoopTree: Exploring the Fused-layer Dataflow Accelerator Design Space](https://arxiv.org/abs/2409.13625) | §II-C/III/VI-F及arXiv作者刊物注记/DOI核对。 | 纳入R24/R32驻潜变量强对照和新颖性边界，未迁原作者完整工件；完整作者工件与原论文平台；本地只实现当前有限RF/端口的具体驻留布局。 |
| W0761 | [TwinQuant: TwinQuant: Learnable Subspace Decomposition for 4-Bit LLM Quantization](https://arxiv.org/abs/2606.01556) | 原文§4.3硬件相关kernel被审阅agent核读，根代理核题名/预印本身份。 | 纳入R24/R32驻潜变量强对照和新颖性边界，未迁原作者完整工件；完整作者工件与原论文平台；本地只实现当前有限RF/端口的具体驻留布局。 |
| W0762 | [LowRank-SSM: LowRank-SSM: Hardware-Software Co-Design for Rank-Reduced Mamba Acceleration on FPGA](https://arxiv.org/abs/2608.02954) | 审阅agent核§2.2/4.1/4.2，根代理核题名/预印本身份。 | 纳入R24/R32驻潜变量强对照和新颖性边界，未迁原作者完整工件；完整作者工件与原论文平台；本地只实现当前有限RF/端口的具体驻留布局。 |
| W0763 | [On-FPGA tensor-compressed training: Ultra Memory-Efficient On-FPGA Training of Transformers via Tensor-Compressed Optimization](https://web.ece.ucsb.edu/~zhengzhang/journals/2025-TCAD-on-FPGA-tensor-transformer.pdf) | 审阅agent读§V-B.2/V-C；作者出版页标录用，本轮未核最终卷页。 | 纳入R24/R32驻潜变量强对照和新颖性边界，未迁原作者完整工件；完整作者工件与原论文平台；本地只实现当前有限RF/端口的具体驻留布局。 |

另外加载同 UID 的 `literature_precision_732.csv` 作为精度分层注释，关联 **245** 篇 `idea_cards/*.md`。全部 Markdown 还扫描了文中 URL，使未进入历史 CSV 的新 Grok/Pro 内容仍可被找到。

## 主要目录入口

- [research/literature_audit_20260909](/home/zhumd/work/sdformer_codex/ideafromai/research/literature_audit_20260909/README.md)：Codex 去留审计及领域子清单。
- [research/literature_audit_mushaolong_20260909](/home/zhumd/work/sdformer_codex/ideafromai/research/literature_audit_mushaolong_20260909/README.md)：Grok 独立审计；未覆盖或替换。
- [audit_comparison_20260909](/home/zhumd/work/sdformer_codex/ideafromai/research/audit_comparison_20260909/README.md)：CICC16条与2条伴随先验、其他电路会刊8条主项及7条明确保留缺读线索全部作为来源记录纳入；阅读层级是原补表声明。
- [research/hardware_innovation_20260908/survey_ab_fusion_20260910](/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/survey_ab_fusion_20260910)：732 库存、精度分层、逐篇 idea 提取、P0 补读、box RTL 和融合建议。
- [gptpro](/home/zhumd/work/sdformer_codex/ideafromai/gptpro)：两轮跨领域调研；文献矩阵和候选段落分别进入目录。
- [idea_screen_20260907](/home/zhumd/work/sdformer_codex/ideafromai/research/idea_screen_20260907/README.md)：32 卡、ep35 普查、Orchestra/K-Dense 原记录。
- [exploration_tcasii_20260911](/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/exploration_tcasii_20260911/CODEX_HANDOFF.md)：可吸收身份后的 R2/R3/R4；文档与 URL 纳入导航，未凭 prose 自动宣称新论文身份。
- [codex_cards](/home/zhumd/work/sdformer_codex/ideafromai/codex_cards)、[Grok 4.6](/home/zhumd/work/sdformer_codex/ideafromai/research/grok46_20260905/00_READ_THIS_FIRST.md)、[早期独立包](/home/zhumd/work/sdformer_codex/ideafromai/codex_independent_20260905/README.md)：原始机制家族仍可查，不用当前单布局失败抹掉。

## 尚有缺口

1. 这是现有目录的全树入口和结构化记录归并，不是全球开源工作的穷尽搜索。新 prose 中未给出稳定题名/编号的条目只进入文档/链接索引，尚未逐条提升为作品实体。
2. 仓库 URL 不保证有效、完整、许可允许复用或和论文作者对应；除明确来源声明外均保留未核状态。借入时应读该仓库 README/代码和许可，不能继承其 PPA。
3. 历史阅读状态可能冲突（例如已有全文记录与后来某轮未取得全文）。保留每个来源，未自动取“最乐观”状态。
4. 论文族、内部组件别名、工具框架与单篇作品分开，尚不能把它们当同粒度的创新候选排名。
5. 本脚本未打开 PDF 全文、未执行任何候选、不取代主线程的最新性能结果。最新试验请从主执行目录进入。

复跑：

```bash
/usr/bin/python3.12 /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/open_fusion_execution/catalog/build_catalog.py
```
