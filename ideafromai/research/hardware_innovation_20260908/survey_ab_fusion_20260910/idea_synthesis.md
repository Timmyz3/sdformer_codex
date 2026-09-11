# Idea 汇总（过夜中期稿）

基于已提取 **86** 条 idea 行（P0 未完成，诚实中期）。

## Idea 簇（按 F / Stage B 粗分）

### f1（26）
- Phi
- VENOM
- SCNN
- sparseml
- CUTLASS
- SparseDNN-accel-survey-repos
- Vitis-AI
- SPARSEBACKBONE-OF
- SparseGPT
- 2:4-sparse-stack
- FireFly→正文SCNN
- SpAtten
- …另有 14

### f2（13）
- LoAS
- TASD
- APEX
- ThUnderVolt
- SpAtten
- Skydiver
- SOFA
- Control Variate approx-multiplier
- SMAM / Spike-Driven Transformer HW
- SpiLiFormer
- Bishop
- ASTER
- …另有 1

### f3（2）
- ExSpike / APEC
- Prosperity

### f4（7）
- APEX
- ThUnderVolt
- NeuraLUT
- BBS / BitVert
- Control Variate approx-multiplier
- SMAM / Spike-Driven Transformer HW
- ASTER

### f5（7）
- GustavSNN
- SCNN
- FireFly→正文SCNN
- MASR
- Skydiver
- Hardware Efficient Reconfigurable Time-Step Spiking Transformer
- ELSA (SNN 2026)

### f6（2）
- BBS / BitVert
- SpikeX

### f7（26）
- Tambe Sparse Transformer Processor
- vLLM
- FlashAttention
- MASR
- GOBO
- SpAtten
- FABNet
- Adaptive-SpikeNet
- HeatViT
- TDE-3
- SOFA
- Xpikeformer
- …另有 14

### stage b（50）
- GustavSNN
- LoAS
- ExSpike / APEC
- Phi
- VENOM
- LUT-DLA
- Tambe Sparse Transformer Processor
- verilator
- FireFly→正文SCNN
- ThUnderVolt
- MASR
- GOBO
- …另有 38

### other（24）
- GustavSNN
- FlexSpIM
- SCNN
- LoopTree
- RISCSparse
- ISSCC25-ConvFormer
- GustavSNN-public-mirror
- spike-flow-net
- MemFlow
- SEA-RAFT
- Taming-Event-Cameras
- v2e
- …另有 12

## 对 F1–F7 的修订建议（中期）

- **F1**：SCNN/结构稀疏/Bit* 族强化「删字后索引/对齐税」分项（RISCSparse/BitVert 线索）；开源 Wanda/2:4 仍作对照。
- **F2**：时间并行/半步检查点对照更硬（LoAS/TASD/Spike*）；MP2 已证明组接受控制可 RTL 粗验。
- **F3**：Prosperity 仍第二队列；产品稀疏≠联合图过门。
- **F5**：Gustav/FlexSpIM 供数与驻留面作为底座；LoopTree 提供占用语言。
- **F7**：Transformer/注意力硬件（ConvFormer/Xpikeformer等）明确旁路主岛。
- **Stage B**：MP1 支持 same-port 信用计数表达；仍不代替净服务实验。

## 待定位（need_locate）

P0 中 best_source=need_locate：**62** 条。处理策略：用题名/venue 回查 arXiv 或审计证据；仍找不到则标 `unresolved_no_fulltext`，**不虚报精读**。

- MAIN-R012｜Gustavson algorithm｜Two Fast Algorithms for Sparse Matrices: Multiplication and Permuted Transposition
- MAIN-R032｜Eyeriss (row stationary)｜Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow for Convolutional Neural Networks
- MAIN-R053｜ANT｜ANT: Exploiting Adaptive Numerical Data Type for Low-Bit Deep Neural Network Quantization
- MAIN-R054｜SpinalFlow｜SpinalFlow: An Architecture and Dataflow Tailored for Spiking Neural Networks
- MAIN-R055｜SATO｜SATO: Spiking Neural Network Acceleration via Temporal-Oriented Dataflow and Architecture
- MAIN-R056｜STELLAR｜STELLAR: Energy-Efficient and Low-Latency SNN Algorithm and Hardware Co-Design with Spatiotemporal Computation
- MAIN-R065｜Configurable CSA for Spiking Transformers｜An Area-Efficient and Bit-Width Configurable Carry-Save Adder Tree for Spiking Transformers
- MAIN-R066｜ESSA｜ESSA: Design of a Programmable Efficient Sparse Spiking Neural Network Accelerator
- MAIN-R068｜COMPASS｜COMPASS: SRAM-Based Computing-in-Memory SNN Accelerator with Adaptive Spike Speculation
- MAIN-R076｜ELSA (attention 2021)｜ELSA: Hardware-Software Co-design for Efficient, Lightweight Self-Attention
- MAIN-R124｜QP-SNN｜QP-SNNs: Quantized and Pruned Spiking Neural Networks
- MAIN-R136｜CompRRAE｜CompRRAE: RRAM-based Convolutional Neural Network Accelerator with Reduced Computations through a Runtime Activation Estimation
- MAIN-R139｜DPES｜Advancing energy efficiency of spiking neural network accelerator via dynamic predictive early stopping
- MAIN-R179｜MACISH｜MACISH: Designing Approximate MAC Accelerators With Internal-Self-Healing
- MAIN-R260｜EEMFlow｜Efficient Meshflow and Optical Flow Estimation from Event Cameras
- MAIN-R267｜SNNIM｜SNNIM: A 10T-SRAM based Spiking-Neural-Network-In-Memory Architecture with Capacitance Computation
- MAIN-R279｜Cha DVS/CIS NPU trigger｜完整题名待核；Cha DVS/CIS NPU trigger
- MAIN-R281｜NullHop｜完整题名待核；NullHop
- MAIN-R282｜DVS + NullHop demonstration｜完整题名待核；DVS + NullHop demonstration
- MAIN-R283｜Lele fused frame-event optical flow｜完整题名待核；Lele fused frame-event optical flow
- MAIN-R285｜Spike-CIM｜Spike-CIM: A 290TOPS/W Spike-Encoding Sparsity-Adaptive Computing-in-Memory Macro with Differential Charge-Domain Integrate-and-Fire
- MAIN-R286｜TFSRAM｜TFSRAM: A 249.8TOPS/W Timing-to-First-Spike Compute-in-Memory Neuromorphic Processing Engine With Twin-Column SRAM Synapses
- MAIN-R290｜ASNA-Flow｜ASNA-Flow: An Efficient Asynchronous Neuromorphic Accelerator for Real-Time Event-Based Optical Flow
- MAIN-R291｜ERAFT FPGA｜An FPGA-based Real-Time Optical Flow Accelerator for Recurrent All-Pairs Field Transforms
- MAIN-R292｜FlowAcc｜FlowAcc: Real-Time High-Accuracy DNN-based Optical Flow Accelerator in FPGA
- MAIN-R293｜Liu optical-flow FPGA TCAS-I 2025｜An Ultra-High Performance and Scalable Optical Flow Hardware Accelerator Based on FPGA for Autonomous Driving
- MAIN-R294｜Optical flow tracking FPGA｜A Real-Time and Efficient Optical Flow Tracking Accelerator on FPGA Platform
- MAIN-R295｜FPGA plane-fitting event optical flow｜Event-based Plane-fitting Optical Flow for Dynamic Vision Sensors in FPGA
- MAIN-R297｜Neuro-CIM JSSC extension｜Neuro-CIM: ADC-Less Neuromorphic Computing-in-Memory Processor With Operation Gating/Stopping and Digital–Analog Networks
- MAIN-R298｜Clock-free multilevel-event SNN｜An 82-nW 0.53-pJ/SOP Clock-Free Spiking Neural Network With 40-µs Latency for AIoT Wake-Up Functions Using a Multilevel-Event-Driven Bionic Architecture and Computing-in-Memory Technique
- …另有 32

## 下一步
1. 收口 batch_04
2. 处理 need_locate
3. 若 p0_txts 增长则重打包增量 arXiv
4. 定稿 idea_synthesis + 更新 ab_fusion_*


## 过夜收口（计数）
- idea 卡 155；CSV 161；P0 非 unresolved 覆盖约 100/246；unresolved 56；仍缺口 91。
- 优先仍 F1→F2；RTL 微探针 MP1/MP2 已由管理 PASS。
