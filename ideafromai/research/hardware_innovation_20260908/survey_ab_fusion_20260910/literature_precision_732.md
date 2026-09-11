# 732 篇精度报告

日期：2026-09-10

## 本轮精度操作

1. 全量相关度打分（主战场加权关键词）
2. venue 归一化
3. 抽取 arXiv id，并对 P0–P2 有 id 的条目拉取官方摘要/分类/题名核验
4. 回写 precision_tier / precision_status / f_candidates
5. 用 literature_audit_20260909 的全文证据升格 depth_bucket

**诚实边界**：本轮不是 732 篇逐章全文精读；是全量精度分层 + 高相关摘要核验 + 既有深读证据对齐。

## 计数

| 量 | 数 |
|---|---:|
| 总条目 | 732 |
| arXiv 摘要核验成功 | 301 |
| 审计深读证据升格 | 13 |
| arXiv API 错误批次数 | 0 |

### precision_tier

- P0_high_rel: 246
- P1_mid_rel: 198
- P3_peripheral: 158
- P2_low_mid: 130

### precision_status

- arxiv_abstract_verified: 297
- metadata_only: 260
- pending_arxiv_abstract: 61
- opensource_inventory: 60
- prior_method_ok_needs_spotcheck: 41
- metadata_only|audit_deep_evidence: 5
- arxiv_abstract_verified|audit_deep_evidence: 4
- prior_method_ok_needs_spotcheck|audit_deep_evidence: 3
- pending_arxiv_abstract|audit_deep_evidence: 1

## P0 高相关（246）

- **[100]** Prosperity｜HPCA 2025｜arxiv_abstract_verified｜F=F1,F3,F7｜Prosperity: Accelerating Spiking Neural Networks via Product Sparsity
- **[100]** LoAS｜MICRO 2024｜arxiv_abstract_verified|audit_deep_evidence｜F=F2,F7｜LoAS: Fully Temporal-Parallel Dataflow for Dual-Sparse Spiking Neural Networks
- **[100]** Phi｜ISCA 2025｜arxiv_abstract_verified|audit_deep_evidence｜F=F2,F3,F4,F7｜Phi: Leveraging Pattern-based Hierarchical Sparsity for High-Efficiency Spiking Neural Networks
- **[98]** FireFly-S｜IEEE Transactions on Circuits and Systems I 2025｜arxiv_abstract_verified｜F=F1,F7｜FireFly-S: Exploiting Dual-Side Sparsity for Spiking Neural Networks Acceleration with Reconfigurabl
- **[98]** ASTER｜arXiv 2025｜arxiv_abstract_verified｜F=F2,F4,F7｜ASTER: Attention-based Spiking Transformer Engine for Event-driven Reasoning
- **[96]** ExSpike / APEC｜FPL 2026｜arxiv_abstract_verified｜F=F3,F7｜ExSpike: A General Full-Event Neuromorphic Architecture for Exploiting Irregular Sparsity with Event
- **[94]** Hardware Efficient Reconfigurable Time-Step Spiking Transformer｜ISCA 2025｜arxiv_abstract_verified｜F=｜Hardware Efficient Accelerator for Spiking Transformer With Reconfigurable Parallel Time Step Comput
- **[92]** ELSA (SNN 2026)｜ISCA 2026｜arxiv_abstract_verified｜F=F5,F7｜ELSA: An ELastic SNN Inference Architecture for Efficient Neuromorphic Computing
- **[88]** MFPSN｜NeurIPS 2025｜arxiv_abstract_verified｜F=F1｜Multiplication-Free Parallelizable Spiking Neurons with Efficient Spatio-Temporal Dynamics
- **[88]** BAT｜AAAI 2026｜arxiv_abstract_verified｜F=F2｜BAT: Learning Event-based Optical Flow with Bidirectional Adaptive Temporal Correlation
- **[88]** SNE｜DATE 2022｜arxiv_abstract_verified｜F=F7｜SNE: an Energy-Proportional Digital Accelerator for Sparse Event-Based Convolutions
- **[84]** PrimeSVT: An Automated Memory-aware Pruning Framework with Prioritized Compressi｜arXiv 2026｜arxiv_abstract_verified｜F=F1｜PrimeSVT: An Automated Memory-aware Pruning Framework with Prioritized Compression Policy for Spikin
- **[82]** OliVe｜ISCA 2023｜arxiv_abstract_verified｜F=｜OliVe: Accelerating Large Language Models via Hardware-friendly Outlier-Victim Pair Quantization
- **[82]** APEX｜arXiv 2026｜arxiv_abstract_verified｜F=F6｜APEX: A Dual-Sparsity Accelerator for Precise and Efficient SNN Inference
- **[82]** Seneca synaptic-delay hardware-aware training｜arXiv 2024｜arxiv_abstract_verified｜F=F1｜Hardware-aware training of models with synaptic delays for digital event-driven neuromorphic process
- **[82]** Motion-aware Event Suppression｜RSS 2026｜arxiv_abstract_verified｜F=F1,F2,F7｜Motion-aware Event Suppression for Event Cameras
- **[82]** Platinum: Path-Adaptable LUT-Based Accelerator Tailored for Low-Bit Weight Matri｜arXiv 2025｜arxiv_abstract_verified｜F=F3｜Platinum: Path-Adaptable LUT-Based Accelerator Tailored for Low-Bit Weight Matrix Multiplication
- **[80]** GustavSNN｜HPCA 2026｜prior_method_ok_needs_spotcheck|audit_deep_evidence｜F=F5｜GustavSNN: Unleashing the Power of Gustavson's Algorithm on SNN Acceleration with Column-Parallel Ti
- **[80]** SpinalFlow｜ISCA 2020｜metadata_only｜F=F5｜SpinalFlow: An Architecture and Dataflow Tailored for Spiking Neural Networks
- **[80]** Parallel_Time_Batching｜HPCA 2022｜metadata_only｜F=F5｜Parallel Time Batching for Spiking Neural Network Accelerators
- **[76]** Bishop｜ISCA 2025｜arxiv_abstract_verified｜F=F1,F2,F7｜Bishop: Sparsified Bundling Spiking Transformers on Heterogeneous Cores with Error-Constrained Pruni
- **[76]** Adaptive-SpikeNet｜ICRA 2023｜arxiv_abstract_verified｜F=｜Adaptive-SpikeNet: Event-based Optical Flow Estimation using Spiking Neural Networks with Learnable 
- **[76]** Bishop｜ISCA 2025｜metadata_only｜F=F1,F7｜Bishop: Sparsified Bundling Spiking Transformers with Error-Constrained Pruning
- **[76]** GustavSNN｜HPCA 2026｜metadata_only｜F=F3,F5｜GustavSNN: Gustavson SpMM + CPTB + NRV for spiking networks (HPCA 2026 public mirror)
- **[76]** MOSAIC: A Workload-Driven Simulation and Design-Space Exploration Framework for ｜arXiv 2026｜arxiv_abstract_verified｜F=｜MOSAIC: A Workload-Driven Simulation and Design-Space Exploration Framework for Heterogeneous NPUs
- **[74]** VENOM｜SC 2023｜arxiv_abstract_verified|audit_deep_evidence｜F=F1｜VENOM: A Vectorized N:M Format for Unleashing the Power of Sparse Tensor Cores
- **[74]** ESDA｜ACM FPGA 2024｜arxiv_abstract_verified｜F=F7｜A Composable Dynamic Sparse Dataflow Architecture for Efficient Event-based Vision Processing on FPG
- **[74]** HOMI｜arXiv 2025｜arxiv_abstract_verified｜F=｜HOMI: Ultra-Fast EdgeAI platform for Event Cameras
- **[74]** Kraken｜arXiv 2022｜arxiv_abstract_verified｜F=F1｜Kraken: A Direct Event/Frame-Based Multi-sensor Fusion SoC for Ultra-Efficient Visual Processing in 
- **[74]** GustavSNN-public-mirror｜open-source｜opensource_inventory｜F=F5｜GustavSNN public paper mirror + local CPTB/NRV migration artifacts
- **[72]** SOFA｜arXiv 2024｜arxiv_abstract_verified｜F=F2,F7｜SOFA: A Compute-Memory Optimized Sparsity Accelerator via Cross-Stage Coordinated Tiling
- **[72]** FLARE / BitSift｜arXiv 2024｜arxiv_abstract_verified｜F=F7｜FLARE: FP-Less PTQ and Low-ENOB ADC Based AMS-PiM for Error-Resilient, Fast, and Efficient Transform
- **[72]** SDformerFlow｜ICPR 2024｜arxiv_abstract_verified｜F=F7｜SDformerFlow: Spatiotemporal swin spikeformer for event-based optical flow estimation
- **[72]** Spiking Patches｜IROS 2026｜arxiv_abstract_verified｜F=F7｜Spiking Patches: Asynchronous, Sparse, and Efficient Tokens for Event Cameras
- **[72]** SLIACF｜arXiv 2026｜arxiv_abstract_verified｜F=F1,F7｜Spiking Local Interaction and Adaptive Complementary Fusion for Spiking Transformer
- **[72]** ERAFT FPGA｜ISCA 2025｜prior_method_ok_needs_spotcheck｜F=｜An FPGA-based Real-Time Optical Flow Accelerator for Recurrent All-Pairs Field Transforms
- **[72]** Liu optical-flow FPGA TCAS-I 2025｜IEEE Transactions on Circuits and Systems I: Regular Papers 2025｜prior_method_ok_needs_spotcheck｜F=｜An Ultra-High Performance and Scalable Optical Flow Hardware Accelerator Based on FPGA for Autonomou
- **[72]** FlexSpIM｜ISCA 2025｜arxiv_abstract_verified｜F=｜An Event-Based Digital Compute-In-Memory Accelerator with Flexible Operand Resolution and Layer-Wise
- **[72]** At-the-Roofline Sparse Tensor Contractions on Vector Processors for Transformer ｜arXiv 2026｜arxiv_abstract_verified｜F=F1,F5｜At-the-Roofline Sparse Tensor Contractions on Vector Processors for Transformer Inference
- **[70]** SATO｜DAC 2022｜metadata_only｜F=｜SATO: Spiking Neural Network Acceleration via Temporal-Oriented Dataflow and Architecture
- **[70]** DPES｜IEICE Electronics Express 2024｜prior_method_ok_needs_spotcheck｜F=F2｜Advancing energy efficiency of spiking neural network accelerator via dynamic predictive early stopp
- **[70]** TFSRAM｜IEEE Transactions on Circuits and Systems for Artificial Intelligence 2024｜prior_method_ok_needs_spotcheck｜F=｜TFSRAM: A 249.8TOPS/W Timing-to-First-Spike Compute-in-Memory Neuromorphic Processing Engine With Tw
- **[70]** Inference-Time Gaze Refinement for Micro-Expression Recognition: Enhancing Event｜arXiv 2025｜arxiv_abstract_verified｜F=F2,F7｜Inference-Time Gaze Refinement for Micro-Expression Recognition: Enhancing Event-Based Eye Tracking 
- **[70]** Realizable N:M Sparse Transformer Inference via Search-Kernel Co-Design｜arXiv 2026｜arxiv_abstract_verified｜F=F1｜Realizable N:M Sparse Transformer Inference via Search-Kernel Co-Design
- **[68]** SCNN｜ISCA 2017｜arxiv_abstract_verified｜F=F1｜SCNN: An Accelerator for Compressed-sparse Convolutional Neural Networks
- **[68]** BBS / BitVert｜MICRO 2024｜arxiv_abstract_verified｜F=F1,F2,F4｜BBS: Bi-directional Bit-level Sparsity for Deep Learning Acceleration
- **[68]** SENECA ANN vs SNN optical flow｜Neural Networks 2025｜arxiv_abstract_verified｜F=｜Event-based Optical Flow on Neuromorphic Processor: ANN vs. SNN Comparison based on Activation Spars
- **[68]** SSER｜SPIE Real-time Processing workshop 投稿/展示 2025｜arxiv_abstract_verified｜F=F2｜Self-Supervised Event Representations: Towards Accurate, Real-Time Perception on SoC FPGAs
- **[68]** Keller Per-Vector INT4｜JSSC 2022｜metadata_only｜F=｜A 95.6-TOPS/W Deep Learning Inference Accelerator With Per-Vector Scaled 4-bit Quantization in 5 nm
- **[68]** Quantized_SpikeDriven_Transformer｜arXiv 2025｜metadata_only｜F=｜Quantized Spike-driven Transformer (IE-LIF multi-bit train / binary infer)
- **[68]** PSViT: A Methodology for Structurally Pruning Spiking Vision Transformers｜arXiv 2026｜arxiv_abstract_verified｜F=F1｜PSViT: A Methodology for Structurally Pruning Spiking Vision Transformers
- **[66]** Gustavson algorithm｜ACM Transactions on Mathematical Software 1978｜metadata_only｜F=F5｜Two Fast Algorithms for Sparse Matrices: Multiplication and Permuted Transposition
- **[66]** GOBO｜MICRO 2020｜arxiv_abstract_verified｜F=F7｜GOBO: Quantizing Attention-Based NLP Models for Low Latency and Energy Efficient Inference
- **[66]** FireFly-T｜IEEE Transactions on Computers 2026｜arxiv_abstract_verified｜F=F7｜FireFly-T: High-Throughput Sparsity Exploitation for Spiking Transformer Acceleration with Dual-Engi
- **[66]** SpikeX｜arXiv 2025｜arxiv_abstract_verified｜F=F1,F7｜SpikeX: Exploring Accelerator Architecture and Network-Hardware Co-Optimization for Sparse Spiking N
- **[66]** SMAM / Sparse Spike-Driven Transformer HW｜arXiv 2025｜arxiv_abstract_verified｜F=F2,F4,F7｜An Efficient Sparse Hardware Accelerator for Spike-Driven Transformer
- **[66]** Event-triggered Implicit Perturbation for Zeroth-Order Fine-Tuning of Spiking Tr｜arXiv 2026｜arxiv_abstract_verified｜F=｜Event-triggered Implicit Perturbation for Zeroth-Order Fine-Tuning of Spiking Transformers
- **[66]** Focus Session: Hardware and Software Techniques for Accelerating Multimodal Foun｜arXiv 2026｜arxiv_abstract_verified｜F=F1,F2,F7｜Focus Session: Hardware and Software Techniques for Accelerating Multimodal Foundation Models
- **[64]** STELLAR｜HPCA 2024｜metadata_only｜F=｜STELLAR: Energy-Efficient and Low-Latency SNN Algorithm and Hardware Co-Design with Spatiotemporal C
- **[64]** Xpikeformer｜TVLSI 2025｜arxiv_abstract_verified｜F=F7｜Xpikeformer: Hybrid Analog-Digital Hardware Acceleration for Spiking Transformers

… 其余 186 条见 CSV。

## 与 F1–F7 映射计数

| F | 命中条目 |
|---|---:|
| F1 | 147 |
| F2 | 128 |
| F3 | 7 |
| F4 | 28 |
| F5 | 17 |
| F6 | 5 |
| F7 | 130 |

## 错误

- none

## 文件

- literature_precision_732.json
- literature_precision_732.csv
- 本报告 literature_precision_732.md

- literature_precision_P0_method_notes.md（P0 Top80 方法级要点）
