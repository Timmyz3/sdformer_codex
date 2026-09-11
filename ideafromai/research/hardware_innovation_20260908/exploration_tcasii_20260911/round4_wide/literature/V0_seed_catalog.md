# V0 — facilitator seed catalog (independent of venue agents)

Date: 2026-09-11. Identity: AT-LIF absorb. Not a title. Local dump has **150** p0_txts; keyword triage of 2024–2026 files not already in round-2/3 map: **74 hits**. Agents V1–V10 + deep-research-5/6 are filling venue tables in parallel. This file is the facilitator’s own first pass so the wide round is not “wait for children.”

Class: **A** copy as prior / **B** strong negative / **stop** as our title / **n/a** off-object / **unresolved**.

---

## Silicon / FPGA / ASIC (newly opened or upgraded this pass)

| ID | Venue | Depth | Mechanism | vs absorb-cut dual last-use |
|---|---|---|---|---|
| Li et al. 2501.07825 Spike-driven Transformer FPGA | FPGA, Virtex US, CIFAR-10 | **full HTML** | Encode spike *positions*; skip zeros in linear / maxpool / **dual-spike** SDSA Mask-Add; ResBuffer+Adder for residual; 307.2 GSOP/s (their number) | **A** for post-SN binary skip + dual-spike attention. Residual is SDT-class ResBuffer, **not** PED/I24. Same ESTU/FireFly skip family. |
| Xpikeformer 2408.08794 | TVLSI-class analog-digital hybrid | local txt Abstract+intro | AIMC for FFN/FC + stochastic spiking attention engine | **A then stop** (CIM/AIMC). Not digital PED join. |
| ASTER 2511.06770 | arXiv cs.AR, analog-digital PIM spiking transformer | local Abstract | Hybrid analog-digital PIM, input sparsity, Bayesian layer-skip / T-reduce at infer | **A then stop** (PIM). Layer-skip is infer-time, not r1 dual last-use. |
| ELSA 2605.20802 | arXiv 2026 SNN elastic inference | local Abstract+intro | Token/spine-wise pipeline; forward as soon as produced; bundled AER; **mini-batch spiking Gustavson-product** | **A** for token-wise *binary* last-use / Gustavson. Elastic early-response ≠ PED last-use. Stronger cousin of Gustav Skip/Fetch/Exec. |
| L-SPINE 2604.03626 | RISC-V SIMD SNN engine | local Abstract | 2/4-bit SIMD neuron core | **n/a** (generic SNN SIMD). |
| Mega 2606.30039 | 22 nm conv SNN | local exists | conv SNN silicon | pending V1. |
| FireFly-T IEEE TC 2026 | 10.1109/TC.2026.3672901 | round3 full | dual-engine overlay | **A** overlay. |
| ESTU TCAS-II 2025 | same journal | AAM | binary SSA skip + mW | **desk-reject script**. |
| ConvFormer ISSCC 2025 23.2 / 2512.17555 | ISSCC | local | LFS = layer fusion; **CFMP = cascaded fmap pruning, not fusion** | **A** fusion/prune. Not G1. |
| DATE hybrid 2411.15409 | DATE 2025 | round3 full | dense **input layer** / sparse rest | **A** layer-split. |
| Fang CICC 2024 | abstract | spiking-only 3D AC | **A** binary AC. |
| SPARTA ICCAD 2025 | news + ACM | RL token skip + hetero ReRAM-CIM + token-spike digital engine | **A then stop** CIM; token skip = ESTU/FireFly class. |
| ICCAD 2025 Xu 3D MoE/MHA spiking + dynamic head prune | program | 3D + head prune | **A** prune. |
| ICCAD 2024 3D spiking transformer | ACM 3676536.3676826 | 3D stacking | **A** packaging, not mechanism X. |
| COBRA ICCAD 2025 | program | binary transformer FPGA | **A** binary GeMM (ANN binary, not SNN). |
| FlexSpIM ISCAS 2025 / 2609.08446 | local | CIM | stop title. |
| Spike-IAND ISCAS 2025 | round3 full | IAND deletes residual | **B** if we keep PED. |
| ERAFT FPGA ISCAS 2025 | abstract | **frame** RAFT | n/a event-SNN. |
| TCAS-I 2025 adaptive OF FPGA | IEEE 11030859 | 405 FPS ZCU104, AEE 1.02, **image pyramid LK-class** | **A** as **frame** OF FPGA. Not event, not SNN. |
| ASNA-Flow TVLSI 2025 | abstract only | event OF 28 nm | spatial locality **taken**; body unresolved. |
| EventShiftFlow 2605.28312 | full | occupancy FPGA | **A** event OF FPGA. |
| SENECA 2407.20421 | full | FireNet neuromorphic OF | **A** already-event-OF HW. |

## Arch (already A; keep on the table)

Prosperity HPCA’25, GustavSNN HPCA’26, LoAS MICRO’24, Phi/Bishop ISCA’25, ExSpike, COMPASS MICRO’24 (CIM speculate, unresolved PDF), Avalanche ASPLOS’25.

ELSA’s Gustavson-product is the same sparse-product family as GustavSNN — copy, don’t stack as X.

## Event OF algorithms (GPU unless noted)

| ID | Venue | HW? | Note |
|---|---|---|---|
| ST-FlowNet 2503.10195 | journal-style SNN OF | **GPU energy model only** (300× theoretical vs ANN). ConvGRU + ANN-to-SNN / BISNN | **A** as SNN-OF algorithm. Not silicon. Not transformer. |
| EDCFlow 2506.03512 | CVPR 2025 | GPU | temporal difference maps + cost volume; plug-in for RAFT-like. **Oracle-adjacent** if used as skip. |
| BAT 2503.03256 | local | GPU | backward-only past events. Causal. |
| SciFlow 2404.08135 | local | GPU | SCI = warp by **current flow** — **banned oracle**. |
| P-SSE / STSSM CVPRW 2025 | HTML | GPU | SSM event OF. |
| Dampfhoffer GNN CVPR 2025 | HTML | claims tens-of-µs on **async HW** (not a taped-out OF chip) | don’t write first-OF-HW. |
| Learning Normal Flow ICCV 2025 / 2412.11284 | GPU | point-based | |
| SDformerFlow 2409.04082 | GPU | our family; live BN; deformed PED | algorithm A. |
| TDE-3 2402.11662 | Frontiers | algorithm | |
| SpikePack 2501.14484 | SNN info flow | algorithm | |

## SNN Transformer algorithms (GPU)

Bipolar SSA NeurIPS 2025; RPE spiking transformers NeurIPS 2025; Otters ICLR 2026 TTFS **optoelectronic** (stop analog); LRF-Dyn ICLR 2026; Plug-and-Play Spiking Operators ICML 2026; SLI+ACF 2608.19238 (local-interaction + complementary fusion — **algorithm A**, residual still spike/MS); SpikePool GPU max-pool; QKFormer/SDT MS residual **A**.

## Compiler / fusion

ConvFormer LFS = layer fusion **A**; CFMP ≠ fusion. LoopTree, RISCSparse, da4ml CSE of T10 mix **A**. No located IR with typed last-use of {binary GeMM, continuous PED} from one producer (pending V10).

---

## What this pass already changes vs round-3

1. **Spiking-transformer FPGA/ASIC zoo is large:** ESTU, FireFly-T, Spike-IAND, Li 2501.07825, Xpikeformer, ASTER, SPARTA, 3D ICCAD, Acta Phys. FPGA Spikformer. A letter that is “another spikeformer skip FPGA” dies.
2. **ELSA (2026)** is a new complete **A** for token-wise binary streaming + Gustavson product — closer to H1’s *timing* than FireFly-T, still **one tensor type**.
3. **ST-FlowNet / EDCFlow** expand SNN-OF algorithm A; neither is silicon; EDCFlow must not become a flow-oracle skip.
4. **TCAS-I 2025 405 FPS OF FPGA** is frame OF, same-journal-family collision on “OF FPGA” if we don’t pin **event-SNN-Transformer dual path**.
5. Dual last-use of **binary GeMM + continuous PED after absorb** still **not located** in this seed set.

Search limit: seed, not exhaustive. Venue agents + workflows still running.
