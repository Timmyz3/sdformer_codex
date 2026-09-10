# 文献库存总表（幕僚长执行线 · 独立审计）

条目数：**127**。来源：seed notes `00`–`14` + `hardware_innovation_20260908/literature` + 同负载/BN/PSN 判决页。**未** verbatim 抄 Codex `literature_audit_20260909`。

字段：name / venue / year / domain / depth / migration / disposition / 理由 / 未尝试 / 证据路径

## domain = `sparsity`（29）

### SP001 · GustavSNN
- **题名**：GustavSNN: Gustavson SpMM + CPTB + NRV for spiking networks (HPCA 2026 public mirror)
- **会议/期刊**：HPCA · 2026
- **阅读深度**：author_pdf · **迁移**：partial · **处置**：`include_as_open`
- **杀/纳入理由**：默认主底座A1。已部分迁GP/NRV/局部S与2×4 mount-only RTL；非因果T10消费者、θg、广播组并集、有损共同完成仍是洞B。完整Gustav服务模型尚未作充分优化对照闭环。
- **未尝试**：完整64PE/源SRAM/PSN/FC2；与Prosperity同资源净收益≥10-15%
- **证据**：hardware_innovation_20260908/literature/AB_STACK_RESEARCH_20260908.md, hardware_innovation_20260908/psn/gustavsnn_hardware_direction.md, hardware_innovation_20260908/psn/gustavsnn_implementation_status_20260908.md, hardware_innovation_20260908/literature/GustavSNN_HPCA2026_public_mirror.txt

### SP002 · Prosperity
- **题名**：Prosperity / ProSparsity pattern-reuse partial sums
- **会议/期刊**：待核 · 待核
- **阅读深度**：secondhand · **迁移**：attempted_variant_failed · **处置**：`stop_this_variant`
- **杀/纳入理由**：旧C1 Prosperity融合同资源无正增量（G4融合仍多2.964%；G2多81.52%）。作强对照保留；该融合布局停止。非否定ProSparsity整类。venue待核。
- **未尝试**：改变共同构造/散播/残差共享后再测；非当前布局复活
- **证据**：same_workload_c1c2_20260907/net_benefit.md, same_workload_c1c2_20260907/c1_independent_review.md, hardware_innovation_20260908/literature/AB_STACK_RESEARCH_20260908.md

### SP003 · HiNM
- **题名**：HiNM: Hierarchical N:M structured sparsity / output clustering
- **会议/期刊**：arXiv · 2024
- **阅读深度**：html_arxiv · **迁移**：attempted_variant_failed · **处置**：`stop_this_variant`
- **杀/纳入理由**：广播域剪枝五轴未胜hidden50/强二阶对照（gyro联合服务仍慢约19%）。停止当前H8/C16/Gyro迁移变体；未否定完整HiNM输入重排+内层2:4+区域搜索+Hard Concrete。arXiv:2407.20496
- **未尝试**：完整HiNM输入重排/OBS补偿/Hard Concrete；迁到昂贵patch挂点
- **证据**：hardware_innovation_20260908/algorithm/group_pruning_probe/README.md, hardware_innovation_20260908/literature/sparsity_candidate_review_20260908.md, hardware_innovation_20260908/literature/AB_STACK_RESEARCH_20260908.md

### SP004 · FlexHiNM-GP_Gyro
- **题名**：FlexHiNM-GP Gyro grouping pipeline
- **会议/期刊**：ICLR · 2026
- **阅读深度**：html_arxiv · **迁移**：attempted_variant_failed · **处置**：`stop_this_variant`
- **杀/纳入理由**：已移植Gyro分组流程到s2b3；联合项未超强对照。停止该明确迁移版本。
- **未尝试**：完整FlexHiNM Hard Concrete渐进训练
- **证据**：hardware_innovation_20260908/algorithm/group_pruning_probe/README.md

### SP005 · VENOM
- **题名**：VENOM two-level N:M structured sparsity
- **会议/期刊**：待核 · 待核
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`include_as_open`
- **杀/纳入理由**：AB_STACK A4底座；未按本网T10+FC2消费损失与物理64b字优化。当前局部失败≠整法失败。
- **未尝试**：完整OBS保留权重补偿；T10消费损失目标
- **证据**：hardware_innovation_20260908/literature/AB_STACK_RESEARCH_20260908.md, hardware_innovation_20260908/literature/sparsity_candidate_review_20260908.md

### SP006 · CRISP
- **题名**：CRISP block navigation structured sparsity
- **会议/期刊**：待核 · 待核
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`include_as_open`
- **杀/纳入理由**：块导航结构稀疏底座；未完整迁。
- **未尝试**：完整迁移+本网消费损失
- **证据**：hardware_innovation_20260908/literature/AB_STACK_RESEARCH_20260908.md

### SP007 · Avalanche
- **题名**：Avalanche SpMM completion recycling and input reuse
- **会议/期刊**：ASPLOS · 2025
- **阅读深度**：full_text · **迁移**：partial · **处置**：`include_as_open`
- **杀/纳入理由**：完成回收/输入复用已读出版社方法文。本网S在非因果T10消费前不能按SpMM last-use释放。依赖生存期训练仍开放；静态探针偏负≠训练思想否证。DOI:10.1145/3695053.3730990
- **未尝试**：训练掩码约束广播组生存期；乐观上界先于训练
- **证据**：hardware_innovation_20260908/literature/sparsity_candidate_review_20260908.md, hardware_innovation_20260908/literature/AB_STACK_RESEARCH_20260908.md

### SP008 · SparseInfer
- **题名**：SparseInfer: predict-then-skip weights
- **会议/期刊**：arXiv · 2024
- **阅读深度**：html_arxiv · **迁移**：partial · **处置**：`include_as_open`
- **杀/纳入理由**：预测后跳权是有损共同完成强先验。当前C192前缀探针仅边界；完整联合完成训练未闭。arXiv:2411.12692
- **未尝试**：同检查点独立预测 vs 请求并集损失双轴；完整AEE
- **证据**：hardware_innovation_20260908/literature/sparsity_candidate_review_20260908.md, hardware_innovation_20260908/psn/group_completion_prefix_probe.json

### SP009 · BitFair
- **题名**：BitFair: learned early termination with shared-input PE termination
- **会议/期刊**：arXiv · 2026
- **阅读深度**：html_arxiv · **迁移**：not_tried · **处置**：`include_as_open`
- **杀/纳入理由**：学习提前终止+共享输入终止控制；有损共同完成A。不能把全完成归约门单独记增量。arXiv:2607.05445
- **未尝试**：完整迁移到T10门字预测
- **证据**：hardware_innovation_20260908/literature/sparsity_candidate_review_20260908.md

### SP010 · SCNN
- **题名**：SCNN: compressed-sparse Cartesian-product dataflow
- **会议/期刊**：ISCA · 2017
- **阅读深度**：author_pdf · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：经典稀疏卷积数据流对照；已有author PDF。
- **未尝试**：-
- **证据**：13_isscc_vlsi_hotchips_edge_npus.md, hardware_innovation_20260908/literature/SCNN_ISCA2017_author.txt

### SP011 · HighLight
- **题名**：HighLight: Efficient and Flexible DNN Acceleration with Hierarchical Structured Sparsity
- **会议/期刊**：MICRO · 2023
- **阅读深度**：html_arxiv · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：层次结构稀疏加速；N:M HW对照族。
- **未尝试**：-
- **证据**：01_ann_sparsity_mechanisms.md

### SP012 · BBS_BitVert
- **题名**：BBS: Bi-directional Bit-level Sparsity for Deep Learning Acceleration
- **会议/期刊**：MICRO · 2024
- **阅读深度**：html_arxiv · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：双向比特级稀疏；与θg幅值路径不同，作对照。
- **未尝试**：-
- **证据**：01_ann_sparsity_mechanisms.md

### SP013 · Sparseloop
- **题名**：Sparseloop taxonomy: format / gating / skipping
- **会议/期刊**：methodology · 2022
- **阅读深度**：secondhand · **迁移**：used_as_control · **处置**：`include_as_control`
- **杀/纳入理由**：方法论分类学；非PE。用于声明边界。
- **未尝试**：-
- **证据**：07_transformer_attention_accelerators.md

### SP015 · RISCSparse
- **题名**：RISCSparse ICCAD 2024 sparse acceleration
- **会议/期刊**：ICCAD · 2024
- **阅读深度**：author_pdf · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：稀疏加速对照；已存author PDF。
- **未尝试**：-
- **证据**：hardware_innovation_20260908/literature/RISCSparse_ICCAD2024_author.txt

### SP017 · WINS
- **题名**：WINS: Winograd Structured Pruning for Fast Winograd Convolution
- **会议/期刊**：ICCV · 2025
- **阅读深度**：html_arxiv · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：变换域结构剪枝近邻；BN零响应一页：不能把换域剪枝本身当X。
- **未尝试**：-
- **证据**：hardware_innovation_20260908/bn_state/response_zero_one_page.md

### SP018 · S3Net
- **题名**：S3Net / submanifold sparse front-end
- **会议/期刊**：待核 · 待核
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`include_as_open`
- **杀/纳入理由**：子流形稀疏前端对准patch大头(AB_STACK A8)；未做端到端稀疏stem训练。局部patch探针≠S3Net失败。
- **未尝试**：端到端稀疏stem重训
- **证据**：hardware_innovation_20260908/literature/AB_STACK_RESEARCH_20260908.md, hardware_innovation_20260908/README.md

### SP019 · Graham_SBNet_SSCN
- **题名**：Submanifold sparse conv lineage (Graham/SBNet/SSCN)
- **会议/期刊**：multiple · 2018
- **阅读深度**：secondhand · **迁移**：used_as_control · **处置**：`include_as_control`
- **杀/纳入理由**：patch背景：普通压缩Y+零掩码+BN补偿+PSN零列跳过已可得，作强底座非独立创新。
- **未尝试**：-
- **证据**：hardware_innovation_20260908/literature/patch_background_followup_20260908.md, hardware_innovation_20260908/algorithm/patch_probe/README.md

### SP020 · DeltaCNN
- **题名**：DeltaCNN / MotionDeltaCNN temporal delta convolution
- **会议/期刊**：待核 · 待核
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：帧差/运动扭曲delta；视频时间稀疏对照。
- **未尝试**：ASIC候选映射
- **证据**：08_video_temporal_sparsity_accelerators.md, hardware_innovation_20260908/literature/patch_background_followup_20260908.md

### SP022 · row2of4_N_M
- **题名**：N:M / row 2:4 structured sparsity (control identity)
- **会议/期刊**：methodology · 2021
- **阅读深度**：full_text · **迁移**：used_as_control · **处置**：`include_as_control`
- **杀/纳入理由**：强对照：valid825 AEE 1.163570；十帧目标层服务少17.12%。任何结构剪枝须先胜过它与hidden50。
- **未尝试**：-
- **证据**：hardware_innovation_20260908/README.md, hardware_innovation_20260908/algorithm/group_pruning_probe/README.md

### SP023 · hidden50_channel_prune
- **题名**：Ordinary hidden-channel 50% prune (control)
- **会议/期刊**：methodology · 2026
- **阅读深度**：full_text · **迁移**：used_as_control · **处置**：`include_as_control`
- **杀/纳入理由**：最强简单控制之一：valid825 AEE 1.164732；目标层服务少49.90%。同时删PSN/门/FC2列。
- **未尝试**：-
- **证据**：hardware_innovation_20260908/README.md, hardware_innovation_20260908/algorithm/group_pruning_probe/README.md

### SP024 · MoE-OPU
- **题名**：MoE-OPU expert-aware N:M
- **会议/期刊**：ICCAD · 2025
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`deferred`
- **杀/纳入理由**：专家感知N:M；MoE故事与本网不同。
- **未尝试**：-
- **证据**：01_ann_sparsity_mechanisms.md

### SP025 · Pre-gated_MoE
- **题名**：Pre-gated MoE algorithm-system co-design
- **会议/期刊**：ISCA · 2024
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`deferred`
- **杀/纳入理由**：预门控MoE；任务域不同。
- **未尝试**：-
- **证据**：01_ann_sparsity_mechanisms.md

### SP026 · UbiMoE
- **题名**：UbiMoE MoE-ViT FPGA
- **会议/期刊**：ISCAS · 2025
- **阅读深度**：html_arxiv · **迁移**：not_tried · **处置**：`deferred`
- **杀/纳入理由**：MoE-ViT FPGA；衍生风险低相关。
- **未尝试**：-
- **证据**：01_ann_sparsity_mechanisms.md

### IS008 · QNAP_ISSCC21
- **题名**：Dual-mode CNN with effective-weight convolution and error-compensation prediction
- **会议/期刊**：ISSCC · 2021
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：有效权卷积+误差补偿预测；条件执行硅片对照。
- **未尝试**：-
- **证据**：13_isscc_vlsi_hotchips_edge_npus.md

### LX001 · C1_Prosperity_fusion_G2G4
- **题名**：C1 input G2/G4 APEC + Prosperity residual fusion (same-workload)
- **会议/期刊**：local_experiment · 2026
- **阅读深度**：full_text · **迁移**：attempted_variant_failed · **处置**：`stop_this_variant`
- **杀/纳入理由**：同负载：相对Prosperity多81.52%/16.44%；大缓存后仍多2.964%。停止该融合布局。失败布局≠APEC/Prosperity整类失败。
- **未尝试**：改变共同构造后再测
- **证据**：same_workload_c1c2_20260907/net_benefit.md, same_workload_c1c2_20260907/c1_attempt_r1/result.json

### LX005 · patch_continuous_check_vs_one_permit
- **题名**：Patch continuous completion check vs one-shot permit
- **会议/期刊**：local_experiment · 2026
- **阅读深度**：full_text · **迁移**：attempted_variant_failed · **处置**：`stop_this_variant`
- **杀/纳入理由**：持续检查比一次许可慢0.6274%；不能只报相对完整执行少6.89%。停止该控制版本创新扩展。
- **未尝试**：参考选择/缓存链/完整训练未实施，未冒写成失败
- **证据**：hardware_innovation_20260908/README.md, hardware_innovation_20260908/algorithm/patch_probe/partial_completion/README.md

### LX007 · NR4_cost_training_variant
- **题名**：NR4 execution-cost training with shuffled-group control
- **会议/期刊**：local_experiment · 2026
- **阅读深度**：full_text · **迁移**：attempted_variant_failed · **处置**：`stop_this_variant`
- **杀/纳入理由**：匹配类别边际后真实组≤打乱组；该NR4费用训练版停止。
- **未尝试**：-
- **证据**：hardware_innovation_20260908/README.md, hardware_innovation_20260908/algorithm/nrv_cost_probe/README.md

### LX012 · TermiNETor_DynConv_priors
- **题名**：TermiNETor joint termination training / DynConv precursor production mask
- **会议/期刊**：待核 · 待核
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：条件生产原文支持；不再作为新增原则包装。
- **未尝试**：venue核实
- **证据**：hardware_innovation_20260908/README.md, hardware_innovation_20260908/literature/patch_conditional_fusion_followup_20260908.md

### LX015 · Flextron_nested_width
- **题名**：Flextron nested dynamic width routing
- **会议/期刊**：待核 · 待核
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`deferred`
- **杀/纳入理由**：AB_STACK A7；共享源并集可能≈满宽。
- **未尝试**：-
- **证据**：hardware_innovation_20260908/literature/AB_STACK_RESEARCH_20260908.md

## domain = `snn_flow`（22）

### OF001 · ERAFT
- **题名**：An FPGA-based Real-Time Optical Flow Accelerator for Recurrent All-Pairs Field Transforms
- **会议/期刊**：ISCAS · 2025
- **阅读深度**：html_arxiv · **迁移**：not_tried · **处置**：`include_as_open`
- **杀/纳入理由**：帧RAFT预测早退可迁到token/tile唤醒；未接入SNN-Transformer调度。纳入开放对照，非已否证。
- **未尝试**：事件体素输入；与Motion-XOR/ATLIF协同；同资源周期模型
- **证据**：00_seed_notes.md, 01_ann_sparsity_mechanisms.md, 03_opticalflow_data_hw_algo.md, 11_event_camera_stack_accelerators.md, hardware_innovation_20260908/literature/AB_STACK_RESEARCH_20260908.md

### OF002 · FlowAcc
- **题名**：FlowAcc optical-flow accelerator with BNN pyramid and Hamming matching
- **会议/期刊**：DATE · 2022
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：强二进制描述子OF HW对照；与连续θg路径不同。
- **未尝试**：Hamming辅助接到Motion-XOR分数叶的费用
- **证据**：00_seed_notes.md, 03_opticalflow_data_hw_algo.md, 11_event_camera_stack_accelerators.md

### OF003 · TCAS-I_2025_adaptive_OF
- **题名**：Adaptive optical flow via dynamic direction prediction; reconfigurable pyramid
- **会议/期刊**：IEEE TCAS-I · 2025
- **阅读深度**：name_only · **迁移**：not_tried · **处置**：`deferred`
- **杀/纳入理由**：方向预测可迁移；全文深度不足，待核。
- **未尝试**：全文；FPS核对；SDformer映射
- **证据**：00_seed_notes.md, 03_opticalflow_data_hw_algo.md

### OF004 · TCAS-I_2023_OF_tracking
- **题名**：Real-time optical-flow tracking FPGA
- **会议/期刊**：IEEE TCAS-I · 2023
- **阅读深度**：name_only · **迁移**：not_tried · **处置**：`deferred`
- **杀/纳入理由**：背景实时OF FPGA；未尝试迁移。
- **未尝试**：全文
- **证据**：00_seed_notes.md

### OF005 · VD56G3_onsensor_OF
- **题名**：On-sensor VD56G3 ASIC optical flow (arXiv:2305.13087)
- **会议/期刊**：arXiv · 2023
- **阅读深度**：name_only · **迁移**：not_tried · **处置**：`deferred`
- **杀/纳入理由**：片上传感器OF；工艺/接口超出当前数字RTL范围。
- **未尝试**：与事件前端融合是否值得探针
- **证据**：00_seed_notes.md, 03_opticalflow_data_hw_algo.md

### OF008 · TDE-3
- **题名**：TDE-3: improved prior for optical flow in SNNs
- **会议/期刊**：Frontiers in Neuroscience · 2025
- **阅读深度**：html_arxiv · **迁移**：not_tried · **处置**：`include_as_open`
- **杀/纳入理由**：SNN光流先验；与spikeformer残差共设计仍开放；非当前主线。
- **未尝试**：TDE-3+spikeformer残差datapath
- **证据**：11_event_camera_stack_accelerators.md, 18_POST_TDE3_TMA_HARD_KNIVES.md

### OF009 · SDformerFlow
- **题名**：SDformerFlow: Spiking Transformer for Event Optical Flow
- **会议/期刊**：ICPR · 2024
- **阅读深度**：secondhand · **迁移**：used_as_control · **处置**：`include_as_control`
- **杀/纳入理由**：任务/网络身份；冻结Motion C12 ep34。非外部新颖性主张。
- **未尝试**：-
- **证据**：11_event_camera_stack_accelerators.md, hardware_innovation_20260908/README.md

### OF011 · Spike-FlowNet
- **题名**：Spike-FlowNet: Event-based Optical Flow Estimation with SNNs
- **会议/期刊**：ECCV · 2020
- **阅读深度**：name_only · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：SNN光流算法祖先；非HW加速器。
- **未尝试**：-
- **证据**：11_event_camera_stack_accelerators.md

### OF014 · Ultra-Flow
- **题名**：Ultra-Flow FPGA optical flow
- **会议/期刊**：FPL · 2022
- **阅读深度**：name_only · **迁移**：not_tried · **处置**：`deferred`
- **杀/纳入理由**：帧OF FPGA；早退可参考；输入非事件。
- **未尝试**：全文
- **证据**：11_event_camera_stack_accelerators.md

### SN001 · AT-LIF_Bu
- **题名**：Activity Pruning for Efficient Spiking Neural Networks (AT-LIF)
- **会议/期刊**：NeurIPS · 2025
- **阅读深度**：html_arxiv · **迁移**：base_A_candidate · **处置**：`include_as_control`
- **杀/纳入理由**：θg=θ·H(m-θ)身份定义来源；冻结τ≠θ。神经元身份控制，非HW创新。
- **未尝试**：-
- **证据**：02_snn_atlif_realvalued_mechanisms.md, 16_ATLIF_IDENTITY_VERDICT_GROKBOT.md, hardware_innovation_20260908/README.md

### SN002 · Quantized_SpikeDriven_Transformer
- **题名**：Quantized Spike-driven Transformer (IE-LIF multi-bit train / binary infer)
- **会议/期刊**：arXiv · 2025
- **阅读深度**：html_arxiv · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：多比特训练/二值推理对照；本网保留连续θg推理。arXiv:2501.13492
- **未尝试**：IE-LIF训练配方作有损PSN对照
- **证据**：00_seed_notes.md, 02_snn_atlif_realvalued_mechanisms.md

### SN003 · Spike_Firing_Approx_V3
- **题名**：Spike Firing Approximation / Spike-driven Transformer V3
- **会议/期刊**：arXiv · 2024
- **阅读深度**：html_arxiv · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：整数训练脉冲推理；与θg连续幅值不同。arXiv:2411.16061
- **未尝试**：-
- **证据**：00_seed_notes.md

### SN005 · SpinalFlow
- **题名**：SpinalFlow: An Architecture and Dataflow Targeting Spiking Neural Networks
- **会议/期刊**：ISCA · 2020
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：时序排序压缩尖峰+输出驻留；强假设二值稀疏，对θg幅值部分失效。
- **未尝试**：幅值AER扩展
- **证据**：02_snn_atlif_realvalued_mechanisms.md, 01_ann_sparsity_mechanisms.md

### SN006 · SATO
- **题名**：SATO: temporal-parallel SNN accumulate with binary adder-search tree
- **会议/期刊**：DAC · 2022
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：时间并行+发放搜索树；幅值路径打破二值搜索树假设。
- **未尝试**：-
- **证据**：02_snn_atlif_realvalued_mechanisms.md

### SN007 · STELLAR
- **题名**：STELLAR: Few-Spikes neuron with spatiotemporal Row-Stationary dataflow
- **会议/期刊**：HPCA · 2024
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：stRS/窗口并行可部分迁移；FS稀疏在幅值密时变弱。
- **未尝试**：stRS tiling到T10非因果
- **证据**：02_snn_atlif_realvalued_mechanisms.md

### SN008 · FireFly
- **题名**：FireFly: A High-Throughput Hardware Accelerator for Spiking Neural Networks
- **会议/期刊**：IEEE TVLSI · 2023
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：FPGA脉冲×权；需改为gated MAC(amp×W)。
- **未尝试**：幅值门控PE
- **证据**：02_snn_atlif_realvalued_mechanisms.md

### SN010 · ESSA
- **题名**：ESSA: Efficient Sparse Spiking Architecture with adaptive spike compression
- **会议/期刊**：IEEE TVLSI · 2022
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：二值AER压缩；可扩(addr,amp) AER。
- **未尝试**：幅值压缩格式
- **证据**：02_snn_atlif_realvalued_mechanisms.md

### SN017 · Parallel_Time_Batching
- **题名**：Parallel Time Batching for Spiking Neural Network Accelerators
- **会议/期刊**：HPCA · 2022
- **阅读深度**：secondhand · **迁移**：partial · **处置**：`include_as_control`
- **杀/纳入理由**：tick-batch祖先；GustavSNN外壳已部分继承。非独立X。
- **未尝试**：-
- **证据**：01_ann_sparsity_mechanisms.md, hardware_innovation_20260908/literature/AB_STACK_RESEARCH_20260908.md

### SN018 · MFPSN
- **题名**：Multiplication-free channel-wise Parallel Spiking Neuron
- **会议/期刊**：NeurIPS · 2025
- **阅读深度**：html_arxiv · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：可训练mul-free PSN强对照；PSN一页要求有损版给足完整STE训练。相关arXiv:2501.14490
- **未尝试**：完整训练到本网valid825
- **证据**：hardware_innovation_20260908/psn/psn_decision_one_page.md, hardware_innovation_20260908/README.md

### LX006 · patch_row34_temporal_structure
- **题名**：Patch public 3-col + 7-private-tail temporal structure (row34)
- **会议/期刊**：local_experiment · 2026
- **阅读深度**：full_text · **迁移**：partial · **处置**：`stop_this_variant`
- **杀/纳入理由**：结构可取消更多前驱Conv，但新增电路贡献薄(~5.5/10)。连续分组增量仅2.03%逻辑取权。停止作标题；精度底座保留。
- **未尝试**：改变空间P/混合Y-U/热权重用的完整链同预算
- **证据**：hardware_innovation_20260908/README.md, hardware_innovation_20260908/algorithm/patch_probe/dependency/

### LX008 · class_vs_packed_time_path
- **题名**：Integer class path vs packed time path under strong NR4 controls
- **会议/期刊**：local_experiment · 2026
- **阅读深度**：full_text · **迁移**：attempted_variant_failed · **处置**：`stop_this_variant`
- **杀/纳入理由**：弱对照14-20%优势在强分块+共同NR4后仅0.58%-1.70%；三原行下类别更慢。撤回贡献。F_live>1公平重测仍开放(P1-4)。
- **未尝试**：F_live>1同资源搜索
- **证据**：hardware_innovation_20260908/README.md, hardware_innovation_20260908/literature/AB_STACK_RESEARCH_20260908.md

### LX011 · fixed_BN_cal32_integer_S0
- **题名**：Fixed BN (cal32) + INT8 S0 integer deployment base
- **会议/期刊**：local_experiment · 2026
- **阅读深度**：full_text · **迁移**：used_as_control · **处置**：`include_as_control`
- **杀/纳入理由**：通过AEE预算的数值底座（S0 INT8 AEE 1.201807）；量化/BN折叠/阈值编译不作标题创新。
- **未尝试**：联合固定全部BN范围
- **证据**：hardware_innovation_20260908/README.md, hardware_innovation_20260908/algorithm/valid825_cal32/summary.json, hardware_innovation_20260908/algorithm/integer_s0_valid825/summary.json

## domain = `circuit`（32）

### SN004 · SpiDR
- **题名**：SpiDR: Reconfigurable digital CIM SNN with zero-skip
- **会议/期刊**：arXiv · 2024
- **阅读深度**：html_arxiv · **迁移**：not_tried · **处置**：`deferred`
- **杀/纳入理由**：CIM SNN多比特W/Vmem零跳过；当前主线数字RTL非CIM。arXiv:2411.02854
- **未尝试**：零跳过语义迁到数字稀疏执行
- **证据**：00_seed_notes.md, 10_cim_spike_opticalflow_accelerators.md

### SN011 · COMPASS
- **题名**：COMPASS: SRAM-CIM with adaptive spike speculation
- **会议/期刊**：MICRO · 2024
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`deferred`
- **杀/纳入理由**：CIM推测发放；工艺线不同。
- **未尝试**：幅值感知speculate-or-MAC数字版
- **证据**：02_snn_atlif_realvalued_mechanisms.md, 10_cim_spike_opticalflow_accelerators.md

### SN019 · ISSCC22_UED_SNN
- **题名**：Ultimate Event-Driven SNN wake-up SoC
- **会议/期刊**：ISSCC · 2022
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`deferred`
- **杀/纳入理由**：超低功耗唤醒；模拟/CIM边界，非数字主线。
- **未尝试**：-
- **证据**：02_snn_atlif_realvalued_mechanisms.md

### SP014 · FlexSpIM
- **题名**：FlexSpIM sparse in-memory computing (ISCAS 2025 author PDF)
- **会议/期刊**：ISCAS · 2025
- **阅读深度**：author_pdf · **迁移**：not_tried · **处置**：`deferred`
- **杀/纳入理由**：稀疏存算；CIM路径，当前数字主线外。已存author PDF。
- **未尝试**：数字侧可迁移的跳过语义
- **证据**：hardware_innovation_20260908/literature/FlexSpIM_ISCAS2025_author.txt

### CR001 · LUT-DLA
- **题名**：LUT-DLA LUT-based deep learning accelerator (CCM/IMM/LUTBoost)
- **会议/期刊**：arXiv · 2025
- **阅读深度**：html_arxiv · **迁移**：partial · **处置**：`include_as_open`
- **杀/纳入理由**：支持码LUT底座A。已部分供数+完整PSN消费；表INT8 QAT与全部编码/LS组织未闭。不能宣称完整A不足。零响应X停在创新门。arXiv:2501.10658
- **未尝试**：表INT8 QAT；CCM距离/argmin完整；IMM查表/局部累加；LS双缓冲；LUTBoost分阶段训练
- **证据**：hardware_innovation_20260908/bn_state/response_zero_one_page.md, hardware_innovation_20260908/bn_state/support_service_notes.md

### CR002 · LUT-NN
- **题名**：LUT-NN lookup-table neural network with INT8 QAT
- **会议/期刊**：arXiv · 2023
- **阅读深度**：html_arxiv · **迁移**：partial · **处置**：`include_as_open`
- **杀/纳入理由**：表INT8 QAT/分级累加/预取补齐项；尚未完整迁。arXiv:2302.03213
- **未尝试**：完整表INT8 QAT到本学生
- **证据**：hardware_innovation_20260908/bn_state/response_zero_one_page.md

### CR003 · T-MAC
- **题名**：T-MAC table/lookup MAC related prior (seed followup)
- **会议/期刊**：待核 · 待核
- **阅读深度**：secondhand · **迁移**：partial · **处置**：`include_as_control`
- **杀/纳入理由**：与LUT精读一并；普通无损INT10已将表载荷降至23.232MB/帧，不能作新标题。
- **未尝试**：venue核实
- **证据**：hardware_innovation_20260908/README.md, hardware_innovation_20260908/bn_state/support_service_notes.md

### CR004 · da4ml
- **题名**：da4ml: depth-limited matrix factorization / CSE for constant matrix multiply
- **会议/期刊**：arXiv · 2025
- **阅读深度**：html_arxiv · **迁移**：not_tried · **处置**：`include_as_open`
- **杀/纳入理由**：完整常矩阵编译是PSN最强对照A之一；尚未作为共同控制补齐。保留PSN问题。arXiv:2507.04535
- **未尝试**：同学生完整常矩阵编译+跨p流水；位域费用模型
- **证据**：hardware_innovation_20260908/psn/psn_decision_one_page.md

### CR005 · CompRRAE
- **题名**：CompRRAE MSB-first bound early-stop for approximate compute
- **会议/期刊**：arXiv · 2019
- **阅读深度**：html_arxiv · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：高位先算+严格界早停对照；仅迁判决算法不迁RRAM。当前X未超此A。arXiv:1906.03180
- **未尝试**：严格MSB界+共享图按位需求门控同学生对照
- **证据**：hardware_innovation_20260908/psn/psn_decision_one_page.md

### CR006 · SPARK_CSE
- **题名**：SPARK conditional CSE / dynamic scheduling (TODAES lineage)
- **会议/期刊**：IEEE TODAES · 2004
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：条件复制/动态CSE已有；按完成次序撤销公共子式须获同等权限对照。
- **未尝试**：-
- **证据**：hardware_innovation_20260908/psn/psn_decision_one_page.md

### CR007 · USEFUSE
- **题名**：USEFUSE unfinished intermediate MSB consumption
- **会议/期刊**：journal (待核) · 2025
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：未完成中间值高位消费先验；PSN改写须对照。
- **未尝试**：-
- **证据**：hardware_innovation_20260908/psn/psn_decision_one_page.md

### CR008 · ISSCC25_ConvFormer
- **题名**：ISSCC 2025 23.2 ConvFormer author PDF
- **会议/期刊**：ISSCC · 2025
- **阅读深度**：author_pdf · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：边缘视觉Transformer硅片对照；已存author PDF。
- **未尝试**：与OF-SNN-T映射
- **证据**：hardware_innovation_20260908/literature/ISSCC2025_23_2_ConvFormer_author.txt, 13_isscc_vlsi_hotchips_edge_npus.md

### CM001 · IMPULSE
- **题名**：IMPULSE fused weight + membrane-potential digital CIM
- **会议/期刊**：待核 · 待核
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`deferred`
- **杀/纳入理由**：数字CIM膜电位融合；当前非CIM主线。
- **未尝试**：venue
- **证据**：10_cim_spike_opticalflow_accelerators.md

### CM002 · Neuro-CIM
- **题名**：Neuro-CIM MSB word skipping + early stopping ADC-less
- **会议/期刊**：待核 · 待核
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`deferred`
- **杀/纳入理由**：CIM早停；工艺外。
- **未尝试**：-
- **证据**：10_cim_spike_opticalflow_accelerators.md

### CM003 · Spike-CIM
- **题名**：Spike-CIM sparsity-adaptive charge-domain IF
- **会议/期刊**：待核 · 待核
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`deferred`
- **杀/纳入理由**：电荷域IF CIM；工艺外。
- **未尝试**：-
- **证据**：10_cim_spike_opticalflow_accelerators.md

### CM004 · TFSRAM
- **题名**：TFSRAM twin-column TTFS SRAM CIM
- **会议/期刊**：待核 · 待核
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`deferred`
- **杀/纳入理由**：TTFS SRAM CIM；工艺外。
- **未尝试**：-
- **证据**：10_cim_spike_opticalflow_accelerators.md

### CM005 · Park_TCAS-I_TD_SNN_CIM
- **题名**：Time-domain async SNN CIM (Park TCAS-I)
- **会议/期刊**：IEEE TCAS-I · 待核
- **阅读深度**：name_only · **迁移**：not_tried · **处置**：`deferred`
- **杀/纳入理由**：时域异步SNN CIM；年份/全文待核。
- **未尝试**：年份核实
- **证据**：10_cim_spike_opticalflow_accelerators.md

### CM006 · X-Former_CIM
- **题名**：X-Former hybrid NVM projection + SRAM attention
- **会议/期刊**：待核 · 待核
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`deferred`
- **杀/纳入理由**：混合NVM/SRAM Transformer CIM。
- **未尝试**：-
- **证据**：10_cim_spike_opticalflow_accelerators.md

### CM007 · Chih_ISSCC21_DigiCIM
- **题名**：Full-precision all-digital SRAM CIM (Chih)
- **会议/期刊**：ISSCC · 2021
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`deferred`
- **杀/纳入理由**：全数字SRAM CIM payload lane参考。
- **未尝试**：-
- **证据**：10_cim_spike_opticalflow_accelerators.md

### IS001 · Samsung_6KMAC_NPU_ISSCC21
- **题名**：6K-MAC Feature-Map-Sparsity-Aware NPU in 5nm
- **会议/期刊**：ISSCC · 2021
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：特征图稀疏感知旗舰NPU；边缘稀疏执行对照。
- **未尝试**：-
- **证据**：13_isscc_vlsi_hotchips_edge_npus.md

### IS004 · Tu_ISSCC22_BitlineTranspose_CIM
- **题名**：Bitline-transpose CIM-based sparse Transformer accelerator
- **会议/期刊**：ISSCC · 2022
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`deferred`
- **杀/纳入理由**：CIM稀疏Transformer硅片；工艺线外。
- **未尝试**：-
- **证据**：13_isscc_vlsi_hotchips_edge_npus.md

### IS006 · DIANA
- **题名**：DIANA digital+analog hybrid NN SoC
- **会议/期刊**：ISSCC · 2022
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`deferred`
- **杀/纳入理由**：数模混合NN SoC；非本数字主线。
- **未尝试**：-
- **证据**：13_isscc_vlsi_hotchips_edge_npus.md

### IS007 · TinyVers
- **题名**：TinyVers extreme-edge ML SoC with eMRAM
- **会议/期刊**：Symp. on VLSI · 2022
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`deferred`
- **杀/纳入理由**：极端边缘SoC；背景。
- **未尝试**：-
- **证据**：13_isscc_vlsi_hotchips_edge_npus.md

### IS010 · CogniVision
- **题名**：CogniVision always-on smart vision SoC
- **会议/期刊**：Symp. on VLSI · 2024
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`deferred`
- **杀/纳入理由**：always-on视觉；背景。
- **未尝试**：-
- **证据**：13_isscc_vlsi_hotchips_edge_npus.md

### IS011 · Alpha-Vision
- **题名**：Alpha-Vision always-on vision processor
- **会议/期刊**：ISSCC · 2026
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`deferred`
- **杀/纳入理由**：always-on视觉子系统；背景。
- **未尝试**：-
- **证据**：13_isscc_vlsi_hotchips_edge_npus.md

### IS012 · Eyeriss
- **题名**：Eyeriss row-stationary CNN accelerator
- **会议/期刊**：ISCA · 2016
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：经典边缘数据流控制祖先。
- **未尝试**：-
- **证据**：13_isscc_vlsi_hotchips_edge_npus.md

### LX002 · C2_static_common_mode_share
- **题名**：C2 static common-mode share on time-parallel baseline
- **会议/期刊**：local_experiment · 2026
- **阅读深度**：full_text · **迁移**：attempted_variant_failed · **处置**：`stop_this_variant`
- **杀/纳入理由**：加法少8.96%但周期多0.23454%。停止该静态共享布局；不证明PSN完整A失败。
- **未尝试**：完整源打包/LoAS/BN2域
- **证据**：same_workload_c1c2_20260907/net_benefit.md, same_workload_c1c2_20260907/c2/functional_r4c/result.json

### LX003 · support_code_zero_response_X
- **题名**：Train W for input-code-correlated whole-word cancellation (zero-response X)
- **会议/期刊**：local_proposal · 2026
- **阅读深度**：full_text · **迁移**：attempted_variant_failed · **处置**：`stop_as_title_claim`
- **杀/纳入理由**：创新门停止：码展开one-hot后普通块稀疏L可跳过同一字；无区别于约束块剪枝的新求解。概念约4.5/10。完整LUT-DLA未迁完。
- **未尝试**：若提出区别于块剪枝的新训练结构可重开审查
- **证据**：hardware_innovation_20260908/bn_state/response_zero_one_page.md, hardware_innovation_20260908/README.md

### LX004 · PSN_stop_when_theta_g_determined_X
- **题名**：Stop computing U once theta_g issuance is determined
- **会议/期刊**：local_proposal · 2026
- **阅读深度**：full_text · **迁移**：attempted_variant_failed · **处置**：`stop_as_title_claim`
- **杀/纳入理由**：可由严格上下界迁移；按完成次序CSE未超SPARK/USEFUSE+按位按需。保留PSN费用问题，停当前两种提法。
- **未尝试**：完整da4ml A + 指出可省的具体位宽/扇出/RF
- **证据**：hardware_innovation_20260908/psn/psn_decision_one_page.md

### LX009 · support_set_proj_share_T10_bound
- **题名**：Support-set output projection share / full-T10 strict suffix bound
- **会议/期刊**：local_experiment · 2026
- **阅读深度**：full_text · **迁移**：attempted_variant_failed · **处置**：`stop_this_variant`
- **杀/纳入理由**：额外机会仅~1%-1.7%；两种具体形态停止。非整类条件执行否证。
- **未尝试**：-
- **证据**：hardware_innovation_20260908/README.md, hardware_innovation_20260908/psn/gustavsnn_hardware_direction.md

### LX010 · SVD_lowrank_PSN_gate
- **题名**：SVD prediction + PSN exact with norm-gate control
- **会议/期刊**：local_experiment · 2026
- **阅读深度**：full_text · **迁移**：attempted_variant_failed · **处置**：`stop_this_variant`
- **杀/纳入理由**：补齐静态常量折叠后普通范数门控更快；停止该低秩版本。
- **未尝试**：-
- **证据**：hardware_innovation_20260908/README.md, hardware_innovation_20260908/psn/independent_review.md

### LX014 · HYTE_Buffets
- **题名**：HYTE + Buffets tiling/capacity/credit multicast
- **会议/期刊**：待核 · 待核
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：AB_STACK A3；分块/容量/信用多播底座，不改变稀疏结构本身。
- **未尝试**：-
- **证据**：hardware_innovation_20260908/literature/AB_STACK_RESEARCH_20260908.md

## domain = `attention`（24）

### SN009 · FireFly-T
- **题名**：FireFly-T: sparse engine + binary AND-PopCount attention for Spike Transformer
- **会议/期刊**：arXiv · 2025
- **阅读深度**：html_arxiv · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：Spike attention稀疏引擎；本网注意力份额小(AB_STACK A9旁路)。arXiv:2505.12771
- **未尝试**：AND-PopCount vs Motion-XOR分数叶费用
- **证据**：11_event_camera_stack_accelerators.md, hardware_innovation_20260908/literature/AB_STACK_RESEARCH_20260908.md

### SN012 · Bishop
- **题名**：Bishop: Sparsified Bundling Spiking Transformers with Error-Constrained Pruning
- **会议/期刊**：ISCA · 2025
- **阅读深度**：html_arxiv · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：捆绑尖峰Transformer+误差约束剪枝；强同族对照。禁止声称first spiking transformer accelerator。
- **未尝试**：TTB/BSA/ECP迁到OF任务+θg
- **证据**：01_ann_sparsity_mechanisms.md, 07_transformer_attention_accelerators.md

### SN013 · ISCAS25_SpikingTransformer_HW
- **题名**：Hardware Efficient Accelerator for Spiking Transformer With Reconfigurable Parallel Time Step Computing
- **会议/期刊**：ISCAS · 2025
- **阅读深度**：html_arxiv · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：并行时间步尖峰Transformer；衍生风险高。须用OF+多比特ATLIF区分。arXiv:2503.19643
- **未尝试**：IAND残差 vs 本网shortcut
- **证据**：01_ann_sparsity_mechanisms.md

### SN014 · Xpikeformer
- **题名**：Xpikeformer: hybrid AIMC + stochastic spiking attention
- **会议/期刊**：IEEE (journal 待核) · 2024
- **阅读深度**：html_arxiv · **迁移**：not_tried · **处置**：`deferred`
- **杀/纳入理由**：AIMC+随机尖峰注意力；CIM路径。arXiv:2408.08794
- **未尝试**：-
- **证据**：10_cim_spike_opticalflow_accelerators.md, 11_event_camera_stack_accelerators.md

### SN015 · SPARTA
- **题名**：SPARTA Spiking Transformer accelerator
- **会议/期刊**：ICCAD · 2025
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：尖峰Transformer加速同族；分类/注意力为主。
- **未尝试**：-
- **证据**：11_event_camera_stack_accelerators.md

### SN016 · ASTER
- **题名**：ASTER layer-skip / spike transformer related
- **会议/期刊**：arXiv · 2025
- **阅读深度**：name_only · **迁移**：not_tried · **处置**：`deferred`
- **杀/纳入理由**：层跳过相关；注意力份额小。arXiv:2511.06770 待核。
- **未尝试**：全文
- **证据**：01_ann_sparsity_mechanisms.md, hardware_innovation_20260908/literature/AB_STACK_RESEARCH_20260908.md

### SN020 · SMAM
- **题名**：SMAM dual-spike Mask-Add for Spike-driven Transformer
- **会议/期刊**：待核 · 待核
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`deferred`
- **杀/纳入理由**：双尖峰Mask-Add注意力；venue待核。本网Motion-XOR路径不同。
- **未尝试**：全文与venue
- **证据**：07_transformer_attention_accelerators.md

### AT001 · SpAtten
- **题名**：SpAtten: Efficient Sparse Attention Architecture with Cascade Token and Head Pruning
- **会议/期刊**：HPCA · 2021
- **阅读深度**：html_arxiv · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：级联token/head剪枝+渐进量化；强注意力稀疏对照。注意力份额小，不作主X。
- **未尝试**：MSB-first渐进量化迁到θg门控
- **证据**：01_ann_sparsity_mechanisms.md, 07_transformer_attention_accelerators.md

### AT002 · HeatViT
- **题名**：HeatViT: Hardware-Efficient Adaptive Token Pruning for Vision Transformers
- **会议/期刊**：HPCA · 2023
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：自适应token剪枝FPGA；ViT对照。
- **未尝试**：-
- **证据**：01_ann_sparsity_mechanisms.md, 07_transformer_attention_accelerators.md

### AT003 · DOTA
- **题名**：DOTA: detect and omit weak attentions
- **会议/期刊**：ASPLOS · 2022
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：弱注意力检测省略；级联稀疏对照族。
- **未尝试**：-
- **证据**：01_ann_sparsity_mechanisms.md, 07_transformer_attention_accelerators.md

### AT004 · Sanger
- **题名**：Sanger: sparse attention accelerator
- **会议/期刊**：MICRO · 2021
- **阅读深度**：name_only · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：稀疏注意力加速同族对照。
- **未尝试**：全文深度
- **证据**：01_ann_sparsity_mechanisms.md

### AT005 · FACT
- **题名**：FACT: sparse/approx attention accelerator
- **会议/期刊**：ISCA · 2023
- **阅读深度**：name_only · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：注意力加速同族。
- **未尝试**：全文
- **证据**：01_ann_sparsity_mechanisms.md

### AT006 · FLAT
- **题名**：FLAT: An Optimized Dataflow for Mitigating Attention Bottlenecks
- **会议/期刊**：ASPLOS · 2023
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：注意力数据流优化；融合基线对照。
- **未尝试**：-
- **证据**：01_ann_sparsity_mechanisms.md, 07_transformer_attention_accelerators.md

### AT007 · FuseMax
- **题名**：FuseMax: Leveraging Extended Einsums to Optimize Attention Accelerator Design
- **会议/期刊**：MICRO · 2024
- **阅读深度**：html_arxiv · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：Einsum融合注意力；FlashAttention族教训。
- **未尝试**：-
- **证据**：01_ann_sparsity_mechanisms.md

### AT008 · FlashAttention
- **题名**：FlashAttention / FA-2 / FA-3 GPU kernels
- **会议/期刊**：NeurIPS · 2022
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：GPU内核；HW教训是融合/分块，非ASIC声明。
- **未尝试**：-
- **证据**：07_transformer_attention_accelerators.md

### AT009 · ViTCoD
- **题名**：ViTCoD: prune-and-polarize with denser/sparser engines
- **会议/期刊**：待核 · 待核
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`deferred`
- **杀/纳入理由**：ViT剪枝+双引擎；venue待核。
- **未尝试**：venue核实
- **证据**：07_transformer_attention_accelerators.md

### AT010 · A3_approx_attention
- **题名**：A3 approximate candidate selection for attention
- **会议/期刊**：待核 · 待核
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`deferred`
- **杀/纳入理由**：近似候选选择；venue待核。
- **未尝试**：venue
- **证据**：07_transformer_attention_accelerators.md

### AT011 · AccelTran
- **题名**：AccelTran / DynaTran runtime activation prune + tiling
- **会议/期刊**：待核 · 待核
- **阅读深度**：name_only · **迁移**：not_tried · **处置**：`deferred`
- **杀/纳入理由**：运行时激活剪枝；venue待核。
- **未尝试**：全文
- **证据**：07_transformer_attention_accelerators.md

### AT012 · Motion-XOR_identity
- **题名**：Motion-XOR attention (project frozen identity)
- **会议/期刊**：project · 2026
- **阅读深度**：full_text · **迁移**：used_as_control · **处置**：`include_as_control`
- **杀/纳入理由**：冻结身份：Motion-XOR attention。K=0输出跳过/行内复用边界已记录；本轮未新增注意力RTL。不作系统主加速。
- **未尝试**：完整打分/归一化/gated-K服务费用补测
- **证据**：hardware_innovation_20260908/README.md, hardware_innovation_execution_plan_20260908.md, idea_reassessment_four_questions_20260907.md

### SP021 · SparseVideoGen
- **题名**：Sparse VideoGen spatial-temporal head dual sparse attention
- **会议/期刊**：待核 · 待核
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`deferred`
- **杀/纳入理由**：视频稀疏注意力算法；venue待核。
- **未尝试**：全文
- **证据**：08_video_temporal_sparsity_accelerators.md

### IS002 · Tambe_ISSCC23_SparseTransformer
- **题名**：12nm Sparse Transformer with entropy early-exit and mixed-precision predication
- **会议/期刊**：ISSCC · 2023
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：熵早退+混合精度谓词；注意力硅片对照。
- **未尝试**：-
- **证据**：13_isscc_vlsi_hotchips_edge_npus.md, 01_ann_sparsity_mechanisms.md

### IS003 · Wang_ISSCC22_ApproxTransformer
- **题名**：Approximate-Computing Transformer with asymptotic sparsity speculation
- **会议/期刊**：ISSCC · 2022
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：渐近稀疏推测+乱序；稀疏执行硅片对照。
- **未尝试**：-
- **证据**：13_isscc_vlsi_hotchips_edge_npus.md

### IS005 · C-Transformer_ISSCC24
- **题名**：C-Transformer homogeneous DNN/Spiking-Transformer processor
- **会议/期刊**：ISSCC · 2024
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：DNN/Spiking-Transformer同构处理器；同族硅片对照。
- **未尝试**：-
- **证据**：13_isscc_vlsi_hotchips_edge_npus.md

### IS009 · Keller_VLSI22_INT4
- **题名**：Per-vector scaled INT4 Transformer engine in 5nm
- **会议/期刊**：Symp. on VLSI · 2022
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：INT4 Transformer引擎；量化对照。
- **未尝试**：-
- **证据**：13_isscc_vlsi_hotchips_edge_npus.md

## domain = `event`（14）

### OF006 · E-RAFT
- **题名**：E-RAFT: Dense Optical Flow from Event Cameras
- **会议/期刊**：3DV · 2021
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：事件体素RAFT算法基线；任务对照非HW A。
- **未尝试**：硬件迁移
- **证据**：03_opticalflow_data_hw_algo.md, 11_event_camera_stack_accelerators.md

### OF007 · TMA
- **题名**：TMA: Temporal Motion Aggregation for Event-based Optical Flow
- **会议/期刊**：ICCV · 2023
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`include_as_open`
- **杀/纳入理由**：时间运动聚合；AB_STACK A10类唤醒候选。未训事件/TDE小预测器，保持开放。
- **未尝试**：可训练运动唤醒；禁止用最终flow作oracle
- **证据**：11_event_camera_stack_accelerators.md, 18_CARD_G_TDE3_TMA_GROKBOT.md, hardware_innovation_20260908/README.md

### OF010 · EV-FlowNet
- **题名**：EV-FlowNet: Self-Supervised Optical Flow for Event Cameras
- **会议/期刊**：RSS · 2018
- **阅读深度**：name_only · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：经典事件光流算法基线。
- **未尝试**：-
- **证据**：11_event_camera_stack_accelerators.md

### OF012 · ASNA-Flow
- **题名**：ASNA-Flow: Asynchronous Neuromorphic Accelerator for Real-Time Event-Based Optical Flow
- **会议/期刊**：IEEE TVLSI · 2025
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：空间局部稀疏神经形态OF datapath；事件OF HW对照。
- **未尝试**：空间局部PE map迁到patch token
- **证据**：11_event_camera_stack_accelerators.md, 03_opticalflow_data_hw_algo.md

### OF013 · EDFLOW_ABMOF
- **题名**：EDFLOW: Event Driven Optical Flow Camera With Keypoint Detection and Adaptive Block Matching
- **会议/期刊**：IEEE TCSVT · 2022
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`deferred`
- **杀/纳入理由**：自适应块匹配事件流；传感器侧，当前数字岛外。
- **未尝试**：自适应切片曝光→token wake
- **证据**：11_event_camera_stack_accelerators.md

### OF015 · hARMS
- **题名**：hARMS: event-based motion sensing platform
- **会议/期刊**：IEEE Access · 2022
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`deferred`
- **杀/纳入理由**：事件历史流近亲；平台不同。
- **未尝试**：-
- **证据**：11_event_camera_stack_accelerators.md, 03_opticalflow_data_hw_algo.md

### OF016 · plane_fit_OF_FPGA
- **题名**：Event-based Plane-fitting Optical Flow for FPGA
- **会议/期刊**：ISCAS · 2018
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：经典事件OF FPGA对照（Aung/Teo/Orchard）。
- **未尝试**：-
- **证据**：11_event_camera_stack_accelerators.md

### OF017 · BMOF
- **题名**：Block Matching Optical Flow for event cameras
- **会议/期刊**：ISCAS · 2017
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：EDFLOW前驱；仅对照。
- **未尝试**：-
- **证据**：11_event_camera_stack_accelerators.md

### OF018 · Lele_fuse_frame_event_OF
- **题名**：Fuse frame+event optical flow (Lele & Raychowdhury)
- **会议/期刊**：ISCAS · 2022
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`deferred`
- **杀/纳入理由**：帧-事件融合OF；非当前主挂点。
- **未尝试**：-
- **证据**：11_event_camera_stack_accelerators.md

### SP027 · ESDA
- **题名**：ESDA: submanifold sparse event dataflow on FPGA
- **会议/期刊**：FPGA · 2024
- **阅读深度**：html_arxiv · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：子流形稀疏事件数据流；事件栈对照。arXiv:2401.05626
- **未尝试**：-
- **证据**：11_event_camera_stack_accelerators.md

### SP028 · EvGNN
- **题名**：EvGNN event-queue neighbor search accelerator
- **会议/期刊**：IEEE TCAS-AI · 2024
- **阅读深度**：html_arxiv · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：事件队列邻居搜索；边缘事件GNN对照。arXiv:2404.19489
- **未尝试**：-
- **证据**：11_event_camera_stack_accelerators.md

### IS013 · Guo_stacked_CIS_EVS
- **题名**：Three-wafer-stacked hybrid 15MP CIS + 1MP EVS
- **会议/期刊**：ISSCC · 2023
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`deferred`
- **杀/纳入理由**：堆叠CIS+EVS传感器；禁止声称first hybrid DVS-CIS。
- **未尝试**：-
- **证据**：11_event_camera_stack_accelerators.md

### IS014 · Prophesee_Sony_EVS_ISSCC20
- **题名**：Prophesee/Sony 1280x720 event vision sensor
- **会议/期刊**：ISSCC · 2020
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：事件传感器硅片基线。
- **未尝试**：-
- **证据**：11_event_camera_stack_accelerators.md

### IS015 · Cha_ISCAS25_DVS_CIS_NPU
- **题名**：Energy-Efficient Daily Surveillance with Event-based NPU Triggering
- **会议/期刊**：ISCAS · 2025
- **阅读深度**：secondhand · **迁移**：not_tried · **处置**：`deferred`
- **杀/纳入理由**：DVS-CIS NPU触发；系统级融合背景。
- **未尝试**：-
- **证据**：11_event_camera_stack_accelerators.md

## domain = `other`（6）

### SP016 · LoopTree
- **题名**：LoopTree TCAS-AI 2024 loop/nest mapping
- **会议/期刊**：IEEE TCAS-AI · 2024
- **阅读深度**：author_pdf · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：循环/嵌套映射方法论；已存author PDF。
- **未尝试**：-
- **证据**：hardware_innovation_20260908/literature/LoopTree_TCASAI2024_author.txt

### QT001 · GOBO
- **题名**：GOBO outlier-aware 3-4bit weight quantization
- **会议/期刊**：MICRO · 2020
- **阅读深度**：html_arxiv · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：异常值感知量化对照。
- **未尝试**：-
- **证据**：01_ann_sparsity_mechanisms.md

### QT002 · OliVe
- **题名**：OliVe outlier-victim pair quantization
- **会议/期刊**：ISCA · 2023
- **阅读深度**：html_arxiv · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：异常值-牺牲对量化对照。
- **未尝试**：-
- **证据**：01_ann_sparsity_mechanisms.md

### QT003 · ANT
- **题名**：ANT adaptive numerical type
- **会议/期刊**：MICRO · 2022
- **阅读深度**：name_only · **迁移**：not_tried · **处置**：`deferred`
- **杀/纳入理由**：自适应数值类型；次要。
- **未尝试**：全文
- **证据**：01_ann_sparsity_mechanisms.md

### QT004 · ControlVariate_DAC21
- **题名**：Control Variate Approximation for DNN Accelerators
- **会议/期刊**：DAC · 2021
- **阅读深度**：html_arxiv · **迁移**：not_tried · **处置**：`include_as_control`
- **杀/纳入理由**：控制变量近似；条件执行/近似计算对照。
- **未尝试**：-
- **证据**：01_ann_sparsity_mechanisms.md

### LX013 · BNFF_AVP_DavisArel
- **题名**：BNFF stats+consumer fusion / AVP moment propagation / Davis-Arel low-rank conditional compute
- **会议/期刊**：multiple · 2019
- **阅读深度**：secondhand · **迁移**：used_as_control · **处置**：`include_as_control`
- **杀/纳入理由**：BN/低秩条件计算先验；独立评审引用。不作新标题。
- **未尝试**：-
- **证据**：hardware_innovation_20260908/README.md, hardware_innovation_20260908/psn/independent_review.md
