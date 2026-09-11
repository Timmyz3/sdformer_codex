# P0 方法级要点（Top80｜精度第二刀）

从 P0=246 中按 F1/F2 加权选出 Top80；每条给线索词 + 摘要首句/导语。**仍非逐章全文**。

## 1. LoAS  [rel=100 | MICRO 2024 | F=F2,F7]

- 题名：LoAS: Fully Temporal-Parallel Dataflow for Dual-Sparse Spiking Neural Networks
- 状态：arxiv_abstract_verified|audit_deep_evidence｜depth=deep_or_method
- 线索：sparsity, spiking, accelerator, dataflow, attention
- 要点：Spiking Neural Networks (SNNs) have gained significant research attention in the last decade due to their potential to drive resource-constrained edge devices.
- arXiv：2407.14073

## 2. Phi  [rel=100 | ISCA 2025 | F=F2,F3,F4,F7]

- 题名：Phi: Leveraging Pattern-based Hierarchical Sparsity for High-Efficiency Spiking Neural Networks
- 状态：arxiv_abstract_verified|audit_deep_evidence｜depth=deep_or_method
- 线索：sparsity, spiking, accelerator, attention
- 要点：Spiking Neural Networks (SNNs) are gaining attention for their energy efficiency and biological plausibility, utilizing 0-1 activation sparsity through spike-driven computation.
- arXiv：2505.10909

## 3. Prosperity  [rel=100 | HPCA 2025 | F=F1,F3,F7]

- 题名：Prosperity: Accelerating Spiking Neural Networks via Product Sparsity
- 状态：arxiv_abstract_verified｜depth=unclear
- 线索：sparsity, spiking, accelerator, product
- 要点：Spiking Neural Networks (SNNs) are highly efficient due to their spike-based activation, which inherently produces bit-sparse computation patterns.
- arXiv：2503.03379

## 4. FireFly-S  [rel=98 | IEEE Transactions on Circuits and Systems I 2025 | F=F1,F7]

- 题名：FireFly-S: Exploiting Dual-Side Sparsity for Spiking Neural Networks Acceleration with Reconfigurable Spatial Architecture
- 状态：arxiv_abstract_verified｜depth=metadata_or_list
- 线索：sparsity, pruning, spiking, accelerator, dataflow, attention
- 要点：Spiking Neural Networks (SNNs), with brain-inspired structure using discrete spikes instead of continuous activations, are gaining attention for their efficient processing on neuromorphic chips.
- arXiv：2408.15578

## 5. ASTER  [rel=98 | arXiv 2025 | F=F2,F4,F7]

- 题名：ASTER: Attention-based Spiking Transformer Engine for Event-driven Reasoning
- 状态：arxiv_abstract_verified｜depth=unclear
- 线索：sparsity, spiking, accelerator, dataflow, attention, transformer
- 要点：The integration of spiking neural networks (SNNs) with transformer-based architectures has opened new opportunities for bio-inspired low-power, event-driven visual reasoning on edge devices.
- arXiv：2511.06770

## 6. MFPSN  [rel=88 | NeurIPS 2025 | F=F1]

- 题名：Multiplication-Free Parallelizable Spiking Neurons with Efficient Spatio-Temporal Dynamics
- 状态：arxiv_abstract_verified｜depth=unclear
- 线索：spiking
- 要点：Spiking Neural Networks (SNNs) are distinguished from Artificial Neural Networks (ANNs) for their complex neuronal dynamics and sparse binary activations (spikes) inspired by the biological neural system.
- arXiv：2501.14490

## 7. BAT  [rel=88 | AAAI 2026 | F=F2]

- 题名：BAT: Learning Event-based Optical Flow with Bidirectional Adaptive Temporal Correlation
- 状态：arxiv_abstract_verified｜depth=unclear
- 线索：sparsity, optical flow
- 要点：Event cameras deliver visual information characterized by a high dynamic range and high temporal resolution, offering significant advantages in estimating optical flow for complex lighting conditions and fast-moving obje
- arXiv：2503.03256

## 8. VENOM  [rel=74 | SC 2023 | F=F1]

- 题名：VENOM: A Vectorized N:M Format for Unleashing the Power of Sparse Tensor Cores
- 状态：arxiv_abstract_verified|audit_deep_evidence｜depth=deep_or_method
- 线索：sparsity, pruning, transformer
- 要点：The increasing success and scaling of Deep Learning models demands higher computational efficiency and power.
- arXiv：2310.02065

## 9. PrimeSVT: An Automated Memory-aware Pruning Framework with Prioritized Compressi  [rel=84 | arXiv 2026 | F=F1]

- 题名：PrimeSVT: An Automated Memory-aware Pruning Framework with Prioritized Compression Policy for Spiking Vision Transformers
- 状态：arxiv_abstract_verified｜depth=metadata_or_list
- 线索：sparsity, pruning, spiking, accelerator, transformer
- 要点：The large sizes of Spiking Vision Transformers (SViTs) still hinder their embedded implementation, highlighting the need for model compression.
- arXiv：2606.03428

## 10. Seneca synaptic-delay hardware-aware training  [rel=82 | arXiv 2024 | F=F1]

- 题名：Hardware-aware training of models with synaptic delays for digital event-driven neuromorphic processors
- 状态：arxiv_abstract_verified｜depth=unclear
- 线索：pruning, spiking, accelerator
- 要点：Configurable synaptic delays are a basic feature in many neuromorphic neural network hardware accelerators.
- arXiv：2404.10597

## 11. Motion-aware Event Suppression  [rel=82 | RSS 2026 | F=F1,F2,F7]

- 题名：Motion-aware Event Suppression for Event Cameras
- 状态：arxiv_abstract_verified｜depth=unclear
- 线索：pruning, transformer
- 要点：Event cameras report asynchronously per-pixel brightness changes with microsecond latency, encoding dynamic visual information as a sparse stream of events.
- arXiv：2602.23204

## 12. ExSpike / APEC  [rel=96 | FPL 2026 | F=F3,F7]

- 题名：ExSpike: A General Full-Event Neuromorphic Architecture for Exploiting Irregular Sparsity with Event Compression
- 状态：arxiv_abstract_verified｜depth=metadata_or_list
- 线索：sparsity, spiking, accelerator, dataflow, attention
- 要点：Spiking neural networks (SNNs) promise energy-efficient computing due to their sparse spatio-temporal activity.
- arXiv：2606.20414

## 13. Bishop  [rel=76 | ISCA 2025 | F=F1,F2,F7]

- 题名：Bishop: Sparsified Bundling Spiking Transformers on Heterogeneous Cores with Error-Constrained Pruning
- 状态：arxiv_abstract_verified｜depth=metadata_or_list
- 线索：sparsity, pruning, spiking, accelerator, attention, transformer
- 要点：We present Bishop, the first dedicated hardware accelerator architecture and HW/SW co-design framework for spiking transformers that optimally represents, manages, and processes spike-based workloads while exploring spat
- arXiv：2505.12281

## 14. Hardware Efficient Reconfigurable Time-Step Spiking Transformer  [rel=94 | ISCA 2025 | F=-]

- 题名：Hardware Efficient Accelerator for Spiking Transformer With Reconfigurable Parallel Time Step Computing
- 状态：arxiv_abstract_verified｜depth=metadata_or_list
- 线索：spiking, accelerator, dataflow, transformer
- 要点：This paper introduces the first low-power hardware accelerator for Spiking Transformers, an emerging alternative to traditional artificial neural networks.
- arXiv：2503.19643

## 15. Kraken  [rel=74 | arXiv 2022 | F=F1]

- 题名：Kraken: A Direct Event/Frame-Based Multi-sensor Fusion SoC for Ultra-Efficient Visual Processing in Nano-UAVs
- 状态：arxiv_abstract_verified｜depth=unclear
- 线索：accelerator
- 要点：Small-size unmanned aerial vehicles (UAV) have the potential to dramatically increase safety and reduce cost in applications like critical infrastructure maintenance and post-disaster search and rescue.
- arXiv：2209.01065

## 16. ELSA (SNN 2026)  [rel=92 | ISCA 2026 | F=F5,F7]

- 题名：ELSA: An ELastic SNN Inference Architecture for Efficient Neuromorphic Computing
- 状态：arxiv_abstract_verified｜depth=metadata_or_list
- 线索：sparsity, spiking, accelerator, dataflow, gustavson, product
- 要点：Spiking neural networks (SNNs) exploit event-driven and addition-only computation to substantially improve efficiency for intelligent computation.
- arXiv：2605.20802

## 17. SOFA  [rel=72 | arXiv 2024 | F=F2,F7]

- 题名：SOFA: A Compute-Memory Optimized Sparsity Accelerator via Cross-Stage Coordinated Tiling
- 状态：arxiv_abstract_verified｜depth=metadata_or_list
- 线索：sparsity, accelerator, attention, transformer
- 要点：Benefiting from the self-attention mechanism, Transformer models have attained impressive contextual comprehension capabilities for lengthy texts.
- arXiv：2407.10416

## 18. SLIACF  [rel=72 | arXiv 2026 | F=F1,F7]

- 题名：Spiking Local Interaction and Adaptive Complementary Fusion for Spiking Transformer
- 状态：arxiv_abstract_verified｜depth=unclear
- 线索：spiking, attention, transformer
- 要点：Spiking Transformers model token interactions primarily through spiking self-attention (SSA).
- arXiv：2608.19238

## 19. At-the-Roofline Sparse Tensor Contractions on Vector Processors for Transformer   [rel=72 | arXiv 2026 | F=F1,F5]

- 题名：At-the-Roofline Sparse Tensor Contractions on Vector Processors for Transformer Inference
- 状态：arxiv_abstract_verified｜depth=metadata_or_list
- 线索：sparsity, pruning, dataflow, gustavson, transformer
- 要点：Fine-grained weight pruning and activation sparsification have emerged as effective approaches for reducing the compute and memory cost of inference for Transformer models.
- arXiv：2607.25504

## 20. Inference-Time Gaze Refinement for Micro-Expression Recognition: Enhancing Event  [rel=70 | arXiv 2025 | F=F2,F7]

- 题名：Inference-Time Gaze Refinement for Micro-Expression Recognition: Enhancing Event-Based Eye Tracking with Motion-Aware Post-Processing
- 状态：arxiv_abstract_verified｜depth=metadata_or_list
- 线索：optical flow, attention
- 要点：Event-based eye tracking holds significant promise for fine-grained cognitive state inference, offering high temporal resolution and robustness to motion artifacts, critical features for decoding subtle mental states suc
- arXiv：2506.12524

## 21. Realizable N:M Sparse Transformer Inference via Search-Kernel Co-Design  [rel=70 | arXiv 2026 | F=F1]

- 题名：Realizable N:M Sparse Transformer Inference via Search-Kernel Co-Design
- 状态：arxiv_abstract_verified｜depth=metadata_or_list
- 线索：sparsity, dataflow, transformer
- 要点：Vision Transformers (ViTs) achieve strong accuracy but incur high inference latency.
- arXiv：2607.12505

## 22. TASD  [rel=58 | MLSys 2025 | F=F1]

- 题名：Enabling Unstructured Sparse Acceleration on Structured Sparse Accelerators
- 状态：arxiv_abstract_verified|audit_deep_evidence｜depth=deep_or_method
- 线索：sparsity, accelerator, product
- 要点：Exploiting sparsity in deep neural networks (DNNs) has been a promising area for meeting the growing computation requirements.
- arXiv：2403.07953

## 23. SCNN  [rel=68 | ISCA 2017 | F=F1]

- 题名：SCNN: An Accelerator for Compressed-sparse Convolutional Neural Networks
- 状态：arxiv_abstract_verified｜depth=deep_or_method
- 线索：pruning, accelerator, dataflow, product
- 要点：Convolutional Neural Networks (CNNs) have emerged as a fundamental technology for machine learning.
- arXiv：1708.04485

## 24. BBS / BitVert  [rel=68 | MICRO 2024 | F=F1,F2,F4]

- 题名：BBS: Bi-directional Bit-level Sparsity for Deep Learning Acceleration
- 状态：arxiv_abstract_verified｜depth=metadata_or_list
- 线索：sparsity, pruning, accelerator, quantization
- 要点：Bit-level sparsity methods skip ineffectual zero-bit operations and are typically applicable within bit-serial deep learning accelerators.
- arXiv：2409.05227

## 25. SSER  [rel=68 | SPIE Real-time Processing workshop 投稿/展示 2025 | F=F2]

- 题名：Self-Supervised Event Representations: Towards Accurate, Real-Time Perception on SoC FPGAs
- 状态：arxiv_abstract_verified｜depth=unclear
- 线索：spiking
- 要点：Event cameras offer significant advantages over traditional frame-based sensors.
- arXiv：2505.07556

## 26. SNE  [rel=88 | DATE 2022 | F=F7]

- 题名：SNE: an Energy-Proportional Digital Accelerator for Sparse Event-Based Convolutions
- 状态：arxiv_abstract_verified｜depth=unclear
- 线索：accelerator, attention
- 要点：Event-based sensors are drawing increasing attention due to their high temporal resolution, low power consumption, and low bandwidth.
- arXiv：2204.10687

## 27. PSViT: A Methodology for Structurally Pruning Spiking Vision Transformers  [rel=68 | arXiv 2026 | F=F1]

- 题名：PSViT: A Methodology for Structurally Pruning Spiking Vision Transformers
- 状态：arxiv_abstract_verified｜depth=metadata_or_list
- 线索：sparsity, pruning, spiking, transformer
- 要点：Spiking Vision Transformer (SViT) models are promising low-power ViT models for solving vision-based tasks with state-of-the-art performance.
- arXiv：2606.03257

## 28. SpikeX  [rel=66 | arXiv 2025 | F=F1,F7]

- 题名：SpikeX: Exploring Accelerator Architecture and Network-Hardware Co-Optimization for Sparse Spiking Neural Networks
- 状态：arxiv_abstract_verified｜depth=metadata_or_list
- 线索：sparsity, spiking, accelerator, dataflow, attention, product
- 要点：Spiking Neural Networks (SNNs) are promising biologically plausible models of computation which utilize a spiking binary activation function similar to that of biological neurons.
- arXiv：2505.12292

## 29. SMAM / Sparse Spike-Driven Transformer HW  [rel=66 | arXiv 2025 | F=F2,F4,F7]

- 题名：An Efficient Sparse Hardware Accelerator for Spike-Driven Transformer
- 状态：arxiv_abstract_verified｜depth=metadata_or_list
- 线索：sparsity, spiking, accelerator, attention, transformer
- 要点：Recently, large models, such as Vision Transformer and BERT, have garnered significant attention due to their exceptional performance.
- arXiv：2501.07825

## 30. Bishop  [rel=76 | ISCA 2025 | F=F1,F7]

- 题名：Bishop: Sparsified Bundling Spiking Transformers with Error-Constrained Pruning
- 状态：metadata_only｜depth=unclear
- 线索：pruning, spiking, transformer
- 要点：(无摘要；依赖既有审计/题名)

## 31. Focus Session: Hardware and Software Techniques for Accelerating Multimodal Foun  [rel=66 | arXiv 2026 | F=F1,F2,F7]

- 题名：Focus Session: Hardware and Software Techniques for Accelerating Multimodal Foundation Models
- 状态：arxiv_abstract_verified｜depth=metadata_or_list
- 线索：pruning, spiking, accelerator, dataflow, attention, quantization
- 要点：This work presents a multi-layered methodology for efficiently accelerating multimodal foundation models (MFMs).
- arXiv：2604.21952

## 32. OliVe  [rel=82 | ISCA 2023 | F=-]

- 题名：OliVe: Accelerating Large Language Models via Hardware-friendly Outlier-Victim Pair Quantization
- 状态：arxiv_abstract_verified｜depth=metadata_or_list
- 线索：sparsity, accelerator, quantization, transformer
- 要点：Transformer-based large language models (LLMs) have achieved great success with the growing model size.
- arXiv：2304.07493

## 33. FireFly  [rel=62 | TVLSI 2023 | F=F1]

- 题名：FireFly: A High-Throughput Hardware Accelerator for Spiking Neural Networks With Efficient DSP and Memory Optimization
- 状态：arxiv_abstract_verified｜depth=metadata_or_list
- 线索：pruning, spiking, accelerator, dataflow, product
- 要点：Convolutional Neural Networks (CNNs) have emerged as a fundamental technology for machine learning.
- arXiv：1708.04485

## 34. APEX  [rel=82 | arXiv 2026 | F=F6]

- 题名：APEX: A Dual-Sparsity Accelerator for Precise and Efficient SNN Inference
- 状态：arxiv_abstract_verified｜depth=metadata_or_list
- 线索：sparsity, spiking, accelerator, dataflow, quantization
- 要点：Spiking Neural Networks (SNNs) have emerged as an energy-efficient alternative to Artificial Neural Networks (ANNs), leveraging sparse accumulate operations in the place of power-hungry multiply-and-accumulate operations
- arXiv：2608.19046

## 35. EvGNN  [rel=62 | TCAS 2024 | F=F1]

- 题名：EvGNN: An Event-driven Graph Neural Network Accelerator for Edge Vision
- 状态：arxiv_abstract_verified｜depth=unclear
- 线索：accelerator
- 要点：Edge vision systems combining sensing and embedded processing promise low-latency, decentralized, and energy-efficient solutions that forgo reliance on the cloud.
- arXiv：2404.19489

## 36. Platinum: Path-Adaptable LUT-Based Accelerator Tailored for Low-Bit Weight Matri  [rel=82 | arXiv 2025 | F=F3]

- 题名：Platinum: Path-Adaptable LUT-Based Accelerator Tailored for Low-Bit Weight Matrix Multiplication
- 状态：arxiv_abstract_verified｜depth=metadata_or_list
- 线索：spiking, accelerator, quantization
- 要点：The rapid scaling of large language models demands more efficient hardware.
- arXiv：2511.21910

## 37. GustavSNN  [rel=80 | HPCA 2026 | F=F5]

- 题名：GustavSNN: Unleashing the Power of Gustavson's Algorithm on SNN Acceleration with Column-Parallel Tick-Batch Dataflow
- 状态：prior_method_ok_needs_spotcheck|audit_deep_evidence｜depth=deep_or_method
- 线索：dataflow, gustavson
- 要点：(无摘要；依赖既有审计/题名)

## 38. BitFair  [rel=60 | TCAS 2026 | F=F2]

- 题名：BitFair: A 12-nm Bit-Serial CNN Accelerator with Learnable Early Termination and Adaptive Bit Ordering for Ultra-Low-Power XR Vision
- 状态：arxiv_abstract_verified｜depth=method_or_partial
- 线索：sparsity, accelerator
- 要点：Extended Reality (XR) wearables require always-on perception within tight power envelopes of a few watts and motion-to-photon latency budgets below 20 ms, leaving only a few milliseconds for neural-network inference.
- arXiv：2607.05445

## 39. DPES  [rel=70 | IEICE Electronics Express 2024 | F=F2]

- 题名：Advancing energy efficiency of spiking neural network accelerator via dynamic predictive early stopping
- 状态：prior_method_ok_needs_spotcheck｜depth=method_or_partial
- 线索：spiking, accelerator
- 要点：(无摘要；依赖既有审计/题名)

## 40. CATFormer  [rel=60 | AAAI 2026 | F=F2]

- 题名：CATFormer: When Continual Learning Meets Spiking Transformers With Dynamic Thresholds
- 状态：arxiv_abstract_verified｜depth=unclear
- 线索：spiking, transformer
- 要点：Although deep neural networks perform extremely well in controlled environments, they fail in real-world scenarios where data isn't available all at once, and the model must adapt to a new data distribution that may or m
- arXiv：2603.15184

## 41. SpiLiFormer  [rel=60 | ICCV 2025 | F=F2,F7]

- 题名：SpiLiFormer: Enhancing Spiking Transformers with Lateral Inhibition
- 状态：arxiv_abstract_verified｜depth=unclear
- 线索：spiking, attention, transformer
- 要点：Spiking Neural Networks (SNNs) based on Transformers have garnered significant attention due to their superior performance and high energy efficiency.
- arXiv：2503.15986

## 42. GMFlow  [rel=60 | CVPR 2022 | F=F2,F7]

- 题名：GMFlow: Learning Optical Flow via Global Matching
- 状态：arxiv_abstract_verified｜depth=unclear
- 线索：optical flow, attention, transformer
- 要点：Learning-based optical flow estimation has been dominated with the pipeline of cost volume with convolutions for flow regression, which is inherently limited to local correlations and thus is hard to address the long-sta
- arXiv：2111.13680

## 43. Chasing Day and Night: Towards Robust and Efficient All-Day Object Detection Gui  [rel=60 | arXiv 2023 | F=F1,F7]

- 题名：Chasing Day and Night: Towards Robust and Efficient All-Day Object Detection Guided by an Event Camera
- 状态：arxiv_abstract_verified｜depth=metadata_or_list
- 线索：spiking, optical flow, attention
- 要点：The ability to detect objects in all lighting (i.e., normal-, over-, and under-exposed) conditions is crucial for real-world applications, such as self-driving.Traditional RGB-based detectors often fail under such varyin
- arXiv：2309.09297

## 44. Unsupervised Optical Flow Estimation with Dynamic Timing Representation for Spik  [rel=60 | arXiv 2023 | F=F2,F7]

- 题名：Unsupervised Optical Flow Estimation with Dynamic Timing Representation for Spike Camera
- 状态：arxiv_abstract_verified｜depth=metadata_or_list
- 线索：optical flow, attention
- 要点：Efficiently selecting an appropriate spike stream data length to extract precise information is the key to the spike vision tasks.
- arXiv：2307.06003

## 45. SCENIC: Semantic-Conditioned Edge-Aware Neural Framework for Structured IoT Comm  [rel=60 | arXiv 2026 | F=F1]

- 题名：SCENIC: Semantic-Conditioned Edge-Aware Neural Framework for Structured IoT Command Generation
- 状态：arxiv_abstract_verified｜depth=metadata_or_list
- 线索：sparsity, pruning, quantization, transformer
- 要点：Edge Internet of Things (IoT) agents are often constrained by memory capacity, privacy requirements, communication latency, and recurring inference cost.
- arXiv：2606.22296

## 46. ThUnderVolt  [rel=58 | DAC 2018 | F=F1,F2,F4]

- 题名：ThUnderVolt: Enabling Aggressive Voltage Underscaling and Timing Error Resilience for Energy Efficient Deep Learning Accelerators
- 状态：arxiv_abstract_verified｜depth=method_or_partial
- 线索：pruning, accelerator
- 要点：Hardware accelerators are being increasingly deployed to boost the performance and energy efficiency of deep neural network (DNN) inference.
- arXiv：1802.03806

## 47. NeuraLUT  [rel=58 | FPL 2024 | F=F1,F2,F4]

- 题名：NeuraLUT: Hiding Neural Network Density in Boolean Synthesizable Functions
- 状态：arxiv_abstract_verified｜depth=method_or_partial
- 线索：sparsity, accelerator, product, quantization
- 要点：Field-Programmable Gate Array (FPGA) accelerators have proven successful in handling latency- and resource-critical deep neural network (DNN) inference tasks.
- arXiv：2403.00849

## 48. Best of Both Worlds  [rel=58 | arXiv 2023 | F=F1]

- 题名：Best of Both Worlds: Hybrid SNN-ANN Architecture for Event-based Optical Flow Estimation
- 状态：arxiv_abstract_verified｜depth=unclear
- 线索：spiking, optical flow
- 要点：In the field of robotics, event-based cameras are emerging as a promising low-power alternative to traditional frame-based cameras for capturing high-speed motion and high dynamic range scenes.
- arXiv：2306.02960

## 49. Spike-FlowNet  [rel=58 | ECCV 2020 | F=F2]

- 题名：Spike-FlowNet: Event-based Optical Flow Estimation with Energy-Efficient Hybrid Neural Networks
- 状态：arxiv_abstract_verified｜depth=unclear
- 线索：spiking, optical flow
- 要点：Event-based cameras display great potential for a variety of tasks such as high-speed motion detection and navigation in low-light environments where conventional frame-based cameras suffer critically.
- arXiv：2003.06696

## 50. Systolic Array Acceleration of Diagonal-Optimized Sparse-Sparse Matrix Multiplic  [rel=58 | arXiv 2025 | F=F1,F5]

- 题名：Systolic Array Acceleration of Diagonal-Optimized Sparse-Sparse Matrix Multiplication for Efficient Quantum Simulation
- 状态：arxiv_abstract_verified｜depth=metadata_or_list
- 线索：sparsity, accelerator, dataflow, gustavson, product
- 要点：Hamiltonian simulation is a key workload in quantum computing, enabling the study of complex quantum systems and serving as a critical tool for classical verification of quantum devices.
- arXiv：2510.14172

## 51. RAMAN: A Re-configurable and Sparse tinyML Accelerator for Inference on Edge  [rel=58 | arXiv 2023 | F=F1,F5]

- 题名：RAMAN: A Re-configurable and Sparse tinyML Accelerator for Inference on Edge
- 状态：arxiv_abstract_verified｜depth=metadata_or_list
- 线索：sparsity, accelerator, dataflow, gustavson
- 要点：Deep Neural Network (DNN) based inference at the edge is challenging as these compute and data-intensive algorithms need to be implemented at low cost and low power while meeting the latency constraints of the target app
- arXiv：2306.06493

## 52. Adaptive-SpikeNet  [rel=76 | ICRA 2023 | F=-]

- 题名：Adaptive-SpikeNet: Event-based Optical Flow Estimation using Spiking Neural Networks with Learnable Neuronal Dynamics
- 状态：arxiv_abstract_verified｜depth=unclear
- 线索：spiking, optical flow
- 要点：Event-based cameras have recently shown great potential for high-speed motion estimation owing to their ability to capture temporally rich information asynchronously.
- arXiv：2209.11741

## 53. MOSAIC: A Workload-Driven Simulation and Design-Space Exploration Framework for   [rel=76 | arXiv 2026 | F=-]

- 题名：MOSAIC: A Workload-Driven Simulation and Design-Space Exploration Framework for Heterogeneous NPUs
- 状态：arxiv_abstract_verified｜depth=metadata_or_list
- 线索：sparsity, spiking, accelerator, dataflow, transformer
- 要点：AI model architectures are diversifying rapidly.
- arXiv：2606.05362

## 54. A Flexible Sparsity-Aware FPGA Accelerator with Column-Wise Compression for Effi  [rel=56 | arXiv 2026 | F=F1]

- 题名：A Flexible Sparsity-Aware FPGA Accelerator with Column-Wise Compression for Efficient CNN Inference
- 状态：arxiv_abstract_verified｜depth=metadata_or_list
- 线索：sparsity, pruning, accelerator
- 要点：Efficient acceleration of convolutional neural networks (CNNs) on resource-constrained platforms remains challenging due to the irregularity of sparsity patterns and the associated hardware overhead.
- arXiv：2607.19248

## 55. MatFormer  [rel=54 | arXiv 2023 | F=F1]

- 题名：MatFormer: Nested Transformer for Elastic Inference
- 状态：arxiv_abstract_verified｜depth=method_or_partial
- 线索：accelerator, transformer
- 要点：Foundation models are applied in a broad spectrum of settings with different inference constraints, from massive multi-accelerator clusters to resource-constrained standalone mobile devices.
- arXiv：2310.07707

## 56. ESDA  [rel=74 | ACM FPGA 2024 | F=F7]

- 题名：A Composable Dynamic Sparse Dataflow Architecture for Efficient Event-based Vision Processing on FPGA
- 状态：arxiv_abstract_verified｜depth=unclear
- 线索：sparsity, accelerator, dataflow
- 要点：Event-based vision represents a paradigm shift in how vision information is captured and processed.
- arXiv：2401.05626

## 57. HOMI  [rel=74 | arXiv 2025 | F=-]

- 题名：HOMI: Ultra-Fast EdgeAI platform for Event Cameras
- 状态：arxiv_abstract_verified｜depth=unclear
- 线索：sparsity, accelerator
- 要点：Event cameras offer significant advantages for edge robotics applications due to their asynchronous operation and sparse, event-driven output, making them well-suited for tasks requiring fast and efficient closed-loop co
- arXiv：2508.12637

## 58. Tambe Sparse Transformer Processor  [rel=54 | ISSCC 2023 | F=F2,F4]

- 题名：Sparse Transformer Processor with Entropy-Based Early Exit, Mixed-Precision Predication and Fine-Grained Power Management
- 状态：prior_method_ok_needs_spotcheck|audit_deep_evidence｜depth=deep_or_method
- 线索：transformer
- 要点：(无摘要；依赖既有审计/题名)

## 59. 2:4-sparse-stack  [rel=64 | open-source | F=F1]

- 题名：NVIDIA 2:4 sparse training examples / apex related
- 状态：opensource_inventory｜depth=open_source_inventory
- 线索：-
- 要点：(无摘要；依赖既有审计/题名)

## 60. Skydiver  [rel=52 | IEEE TCAD（旧表称） 2022 / IEEE TCAD 2022 | F=F1,F2]

- 题名：Skydiver: A Spiking Neural Network Accelerator Exploiting Spatio-Temporal Workload Balance
- 状态：arxiv_abstract_verified｜depth=metadata_or_list
- 线索：sparsity, spiking, accelerator
- 要点：Spiking Neural Networks (SNNs) are developed as a promising alternative to Artificial Neural networks (ANNs) due to their more realistic brain-inspired computing models.
- arXiv：2203.07516

## 61. SpAtten  [rel=52 | HPCA 2021 | F=F1,F2,F7]

- 题名：SpAtten: Efficient Sparse Attention Architecture with Cascade Token and Head Pruning
- 状态：arxiv_abstract_verified｜depth=metadata_or_list
- 线索：sparsity, pruning, accelerator, attention, quantization
- 要点：The attention mechanism is becoming increasingly popular in Natural Language Processing (NLP) applications, showing superior performance than convolutional and recurrent architectures.
- arXiv：2012.09852

## 62. HeatViT  [rel=52 | HPCA 2023 | F=F1,F7]

- 题名：HeatViT: Hardware-Efficient Adaptive Token Pruning for Vision Transformers
- 状态：arxiv_abstract_verified｜depth=metadata_or_list
- 线索：pruning, accelerator, attention, quantization, transformer
- 要点：While vision transformers (ViTs) have continuously achieved new milestones in the field of computer vision, their sophisticated network architectures with high computation and memory costs have impeded their deployment o
- arXiv：2211.08110

## 63. FLARE / BitSift  [rel=72 | arXiv 2024 | F=F7]

- 题名：FLARE: FP-Less PTQ and Low-ENOB ADC Based AMS-PiM for Error-Resilient, Fast, and Efficient Transformer Acceleration
- 状态：arxiv_abstract_verified｜depth=method_or_partial
- 线索：attention, quantization, transformer
- 要点：Encoder-based transformers, powered by self-attention layers, have revolutionized machine learning with their context-aware representations.
- arXiv：2411.14733

## 64. SDformerFlow  [rel=72 | ICPR 2024 | F=F7]

- 题名：SDformerFlow: Spatiotemporal swin spikeformer for event-based optical flow estimation
- 状态：arxiv_abstract_verified｜depth=unclear
- 线索：spiking, optical flow, attention, transformer
- 要点：Event cameras generate asynchronous and sparse event streams capturing changes in light intensity.
- arXiv：2409.04082

## 65. Efficient synaptic-delay implementation  [rel=52 | arXiv 2025 | F=F1]

- 题名：Efficient Synaptic Delay Implementation in Digital Event-Driven AI Accelerators
- 状态：arxiv_abstract_verified｜depth=unclear
- 线索：sparsity, accelerator
- 要点：Synaptic delay parameterization of neural network models have remained largely unexplored but recent literature has been showing promising results, suggesting the delay parameterized models are simpler, smaller, sparser,
- arXiv：2501.13610

## 66. Spiking Patches  [rel=72 | IROS 2026 | F=F7]

- 题名：Spiking Patches: Asynchronous, Sparse, and Efficient Tokens for Event Cameras
- 状态：arxiv_abstract_verified｜depth=unclear
- 线索：sparsity, spiking, transformer
- 要点：We propose tokenization of events and present a tokenizer, Spiking Patches, specifically designed for event cameras.
- arXiv：2510.26614

## 67. FlexSpIM  [rel=72 | ISCA 2025 | F=-]

- 题名：An Event-Based Digital Compute-In-Memory Accelerator with Flexible Operand Resolution and Layer-Wise Weight/Output Stationarity
- 状态：arxiv_abstract_verified｜depth=unclear
- 线索：spiking, accelerator, dataflow
- 要点：Compute-in-memory (CIM) accelerators for spiking neural networks (SNNs) are promising solutions to enable $μ$s-level inference latency and ultra-low energy in edge vision applications.
- arXiv：2410.23082

## 68. CFMP / ConvFormer chip  [rel=52 | ISSCC 2025 | F=F1,F2,F7]

- 题名：A 28nm 0.22μJ/Token Memory-Compute-Intensity-Aware CNN-Transformer Accelerator with Hybrid-Attention-Based Layer-Fusion and Cascaded Pruning for Semantic-Segmentation
- 状态：arxiv_abstract_verified｜depth=unclear
- 线索：pruning, accelerator, attention, transformer
- 要点：This work presents a 28nm 13.93mm2 CNN-Transformer accelerator for semantic segmentation, achieving 3.86-to-10.91x energy reduction over previous designs.
- arXiv：2512.17555

## 69. Sparsity-Aware Streaming SNN Accelerator with Output-Channel Dataflow for Automa  [rel=52 | arXiv 2026 | F=F1,F2]

- 题名：Sparsity-Aware Streaming SNN Accelerator with Output-Channel Dataflow for Automatic Modulation Classification
- 状态：arxiv_abstract_verified｜depth=metadata_or_list
- 线索：sparsity, spiking, accelerator, dataflow, product
- 要点：The rapid advancement of wireless communication technologies, including 5G, emerging 6G networks, and the large-scale deployment of the Internet of Things (IoT), has intensified the need for efficient spectrum utilizatio
- arXiv：2601.02613

## 70. Sparse Compressed Spiking Neural Network Accelerator for Object Detection  [rel=52 | arXiv 2022 | F=F2]

- 题名：Sparse Compressed Spiking Neural Network Accelerator for Object Detection
- 状态：arxiv_abstract_verified｜depth=metadata_or_list
- 线索：sparsity, spiking, accelerator, product
- 要点：Spiking neural networks (SNNs), which are inspired by the human brain, have recently gained popularity due to their relatively simple and low-power hardware for transmitting binary spikes and highly sparse activation map
- arXiv：2205.00778

## 71. Training for temporal sparsity in deep neural networks, application in video pro  [rel=52 | arXiv 2021 | F=F2,F4]

- 题名：Training for temporal sparsity in deep neural networks, application in video processing
- 状态：arxiv_abstract_verified｜depth=metadata_or_list
- 线索：sparsity, spiking, accelerator, product
- 要点：Activation sparsity improves compute efficiency and resource utilization in sparsity-aware neural network accelerators.
- arXiv：2107.07305

## 72. Sol-Attn: Accelerating Video Generation Inference via On-the-Fly Attention Spars  [rel=52 | arXiv 2026 | F=F2,F7]

- 题名：Sol-Attn: Accelerating Video Generation Inference via On-the-Fly Attention Sparsification
- 状态：arxiv_abstract_verified｜depth=metadata_or_list
- 线索：sparsity, attention, transformer
- 要点：Diffusion transformers are essential for high-fidelity video generation, but long token sequences make attention a dominant inference bottleneck.
- arXiv：2607.24027

## 73. The Sparsity Tax: Weight Sparsity Trade-offs in Event-Driven SIMD and SIMT Neuro  [rel=52 | arXiv 2026 | F=F1,F2]

- 题名：The Sparsity Tax: Weight Sparsity Trade-offs in Event-Driven SIMD and SIMT Neuromorphic Cores
- 状态：arxiv_abstract_verified｜depth=metadata_or_list
- 线索：sparsity, pruning, accelerator
- 要点：Event-driven neuromorphic inference exploits activation sparsity by updating neuron state only on spikes.
- arXiv：2607.22790

## 74. SpinalFlow  [rel=80 | ISCA 2020 | F=F5]

- 题名：SpinalFlow: An Architecture and Dataflow Tailored for Spiking Neural Networks
- 状态：metadata_only｜depth=metadata_or_list
- 线索：spiking, dataflow
- 要点：(无摘要；依赖既有审计/题名)

## 75. Spikformer  [rel=50 | ICLR 2023 | F=F1,F7]

- 题名：Spikformer: When Spiking Neural Network Meets Transformer
- 状态：arxiv_abstract_verified｜depth=method_or_partial
- 线索：spiking, attention, transformer
- 要点：We consider two biologically plausible structures, the Spiking Neural Network (SNN) and the self-attention mechanism.
- arXiv：2209.15425

## 76. Spike-driven Transformer  [rel=50 | NeurIPS 2023 | F=F1,F7]

- 题名：Spike-driven Transformer
- 状态：arxiv_abstract_verified｜depth=method_or_partial
- 线索：spiking, attention, transformer
- 要点：Spiking Neural Networks (SNNs) provide an energy-efficient deep learning option due to their unique spike-based event-driven (i.e., spike-driven) paradigm.
- arXiv：2307.01694

## 77. Spike-driven Transformer V2 / Meta-SpikeFormer  [rel=50 | ICLR 2024 | F=F1,F2,F4,F7]

- 题名：Spike-driven Transformer V2: Meta Spiking Neural Network Architecture Inspiring the Design of Next-generation Neuromorphic Chips
- 状态：arxiv_abstract_verified｜depth=method_or_partial
- 线索：spiking, attention, transformer
- 要点：Neuromorphic computing, which exploits Spiking Neural Networks (SNNs) on neuromorphic chips, is a promising energy-efficient alternative to traditional AI.
- arXiv：2404.03663

## 78. Parallel_Time_Batching  [rel=80 | HPCA 2022 | F=F5]

- 题名：Parallel Time Batching for Spiking Neural Network Accelerators
- 状态：metadata_only｜depth=unclear
- 线索：spiking, accelerator
- 要点：(无摘要；依赖既有审计/题名)

## 79. Anti-Gravity Walking by a Flying Humanoid Robot via Thrust-Rate Input Whole-Body  [rel=50 | arXiv 2026 | F=F1,F2]

- 题名：Anti-Gravity Walking by a Flying Humanoid Robot via Thrust-Rate Input Whole-Body Model Predictive Control
- 状态：arxiv_abstract_verified｜depth=metadata_or_list
- 线索：-
- 要点：Flying humanoids are expected to perform tasks in diverse environments, while their existing locomotion is mainly limited to aerial flight and ground walking.
- arXiv：2609.07544

## 80. The Sparsity Ceiling: Where Spiking Networks Can and Cannot Trade Activity for E  [rel=50 | arXiv 2026 | F=F2,F7]

- 题名：The Sparsity Ceiling: Where Spiking Networks Can and Cannot Trade Activity for Energy
- 状态：arxiv_abstract_verified｜depth=metadata_or_list
- 线索：sparsity, spiking, attention, transformer
- 要点：Spiking neural networks (SNNs) are promoted as an energy-efficient substrate because sparse, event-driven activity replaces dense multiply-accumulates with cheap accumulates.
- arXiv：2607.26648

---

Top80 中含 F1/F2 标记：62
