# Seed notes (parent, 2026-09-05)

## Optical-flow HW
- ERAFT ISCAS 2025: RAFT-like + prediction mechanism FPGA
- FlowAcc DATE 2022 / JSA: BNN pyramid + hamming for OF
- TCAS-I 2025: adaptive OF via dynamic direction prediction; reconfigurable pyramid pipeline; 405 FPS
- TCAS-I 2023: real-time OF tracking FPGA
- On-sensor VD56G3 ASIC OF (arxiv 2305.13087)

## SNN / multi-bit
- SpiDR arxiv 2411.02854: CIM SNN, multi-bit W/Vmem, zero-skip at sparsity
- L-SPINE: 2/4/8-bit SIMD SNN, multiplier-less
- Mega 22nm 0.375 pJ/SOP: spike map LZC streaming
- Quantized Spike-driven Transformer (IE-LIF multi-bit train / binary infer) arxiv 2501.13492
- Spike Firing Approximation / Spike-driven Transformer V3 arxiv 2411.16061 — integer train, spike infer

## Opportunity gap for SDformer
- Most OF HW = dense CNN/RAFT on frames, NOT SNN-Transformer
- Most SNN HW = binary events; real-valued ATLIF is under-served (can claim datapath that keeps amplitude)
- Direction-prediction / temporal coherence from OF HW can be RETARGETED to token/spike scheduling in C1/C2
