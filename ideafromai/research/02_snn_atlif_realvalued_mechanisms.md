# Hardware Mechanisms for Optical-Flow SNN-Transformer (SDformer) with Real-Valued ATLIF-like Outputs

**Target venue:** ISCAS (circuits/systems co-design angle)  
**Scope date:** 2019–2026 literature (ISSCC / ISCAS / DATE / DAC / TCAS / TBioCAS / Frontiers / NeurIPS neuromorphic / named accelerators)  
**Key differentiator:** neuron outputs are **real-valued / multi-bit / continuous-amplitude** (ATLIF-like), not binary `{0,1}` events.

---

## 0. Clarifying ATLIF vs. “real-valued” (important for HW claims)

| Model | Output semantics | Hardware implication |
|-------|------------------|----------------------|
| Canonical **AT-LIF** (Bu, Shi, Yu, NeurIPS 2025, *Activity Pruning for Efficient Spiking Neural Networks*) | \(o_i[t]=\theta_i\cdot H(m_i[t]-\theta_i)\in\{0,\theta_i\}\) — **binary event × real amplitude**; authors note it is *not* graded spikes; at inference \(\theta\) can be absorbed into next-layer weights | Hybrid **binary gate + scalar payload**; if \(\theta\) folded into W, collapses to binary SNN HW |
| **User / SDformer-side ATLIF-like** (target of this note) | **True real / multi-bit / continuous** membrane or spike amplitudes that are *not* absorbable into W (per-token, per-timestep, or learnable graded) | Needs **amplitude-aware datapath** — many classic SNN sparsity tricks partially break |
| **MBLIF / multi-bit spikes** (Xiao et al., arXiv:2407.05739) | Fixed-point \(m\) integer + \(n\) fractional bits; still sparse zeros | Bit-decompose + shift-add (FireFly-v2 style) |
| **Ternary / MLF** (Guo AAAI’24 Ternary Spike; Feng IJCAI’22 MLF) | \(\{-\alpha,0,\alpha\}\) or summed multi-threshold | Sign bit + magnitude; AC still viable |
| **Graded spikes / temporal aggregation** (e.g. CLANE-style on Loihi 2) | Up to ~24-bit “graded spike” vectors after temporal pool | Closer to ANN MAC; event skip only on near-zero |

**HW claim framing for ISCAS:** treat user ATLIF as **hybrid event**:  
`gate = (amp ≠ 0)` (or `|amp| > ε`) + `payload = quantized amp`.  
Preserve event-driven clocking via the gate; use payload only when gate=1. This is the cleanest differentiator vs. FireFly-T / SpikeTA / SpinalFlow (binary-centric).

---

## 1. Survey: mechanisms beyond plain LIF accumulate-and-fire

### 1.1 Temporal coding & chronological decoupling

| Work | Venue | Mechanism | Binary-only? | Real-valued survivability |
|------|-------|-----------|--------------|---------------------------|
| **SpinalFlow** (Narayanan et al.) | ISCA 2020 | Chronologically sorted, compressed spike trains; output-stationary; merge-sort front-end; skip zero spikes; eliminate Vmem storage by consecutive neuron compute | Strongly assumes sparse binary temporal code | **Breaks / heavy rewrite**: sort key must become `(time, |amp|)` or rate bins; merge-sort of real payloads costly; weight-buffer reuse still useful |
| **SATO** (Liu et al.) | DAC 2022; also TPDS 2024 temporal-unroll | Temporal-parallel accumulate all timesteps; **binary adder-search tree** fires spike train; bucket-sort dispatcher | Binary search-tree for fire times | **Breaks**: search-tree is for binary fire/no-fire; replace with **threshold-scan / sorted amplitude tree** or keep parallel Vmem MAC + compare-to-θ |
| **STELLAR** (Mao et al.) | HPCA 2024 | Few-Spikes neuron + FSBP training; spatiotemporal Row-Stationary (stRS); window-parallel | Few binary spikes | **Partial**: stRS / window tiling OK; FS sparsity weaker if amps are dense |

### 1.2 Spatiotemporal dataflow, membrane reuse, on-the-fly spike gen

| Work | Venue | Mechanism | Real-valued survivability |
|------|-------|-----------|---------------------------|
| **FireFly** (Li et al.) | TVLSI 2023 | DSP48E2 multiplex-accumulate for spike×weight; on-the-fly spike processing | Binary AND/mux → must become **gated MAC** (amp×W) |
| **FireFly v2** (Li et al.) | TCAD 2024 | 4D parallelism (Cin/Cout/pixel/time); **eliminate long-term Vmem storage**; on-the-fly spike gen; **first SNN accel with non-spike ops** via bit-decompose + shift-add for SEW / avg-pool / direct encode | **Works well**: already handles multi-bit / fractional “spikes”; natural base for ATLIF payloads |
| **FireFly-S** (Li et al.) | TCAS-I 2024 | Dual-side sparsity (activation + weight); Bitmap decoder; spatial overlay | Gate=nonzero still works; payload bits add decode cost |
| **FireFly-T** (Li et al.) | arXiv 2505.12771 (2025) | Dual-engine: sparse conv + **binary AND-PopCount attention**; multi-lane sparse decoder; latency-hiding Q/K/V vs QKᵀV | Attention path **breaks** if Q/K/V are real — need hybrid attention engine (see §3) |
| **Lee & Li** | ICCD 2020 | Spatiotemporal systolic; Psum-friendly | Parallel time OK if Vmem update serializable or PSN-style |

### 1.3 Spike compression, sparsity decode, event queues

| Work | Venue | Mechanism | Real-valued survivability |
|------|-------|-----------|---------------------------|
| **ESSA** (Wang et al.) | TVLSI 2022 | Temporal + weight sparsity; self-adaptive spike compression; combinable dendrites | Compression of binary AER; extend to **(addr, amp)** AER |
| **Skydiver** (Chen/Gao et al.) | TCAD 2022 | Spatio-temporal workload balance (APRC + CBWS) | Balance still useful; workload ∝ nonzero amps |
| **COMPASS** (Wang et al.) | MICRO 2024 | SRAM-CIM + **adaptive spike speculation** | Speculation on binary fire; need amp-aware speculate-or-MAC |
| **Seneca SCDQ / synaptic delay queues** (arXiv 2404.10597, 2501.13610) | — | Shared Circular Delay Queue vs Loihi ring buffer; sparsity-scaled event storage | **Works**: store `(neuron_id, delay, amp)` tuples |
| **ISSCC 2022** UED SNN wake-up (82 nW, 0.53 pJ/SOP) | ISSCC | Clock-free ultimate-event-driven + CIM | Event presence still gates; payload ADC/DAC cost if analog amp |

### 1.4 Multi-bit / non-spike / transformer-specific HW

| Work | Venue | Mechanism | Real-valued survivability |
|------|-------|-----------|---------------------------|
| **SpikeSim / SpikeFlow** (TCAD 2023) | TCAD | IMC crossbar eval; digital LIF; DIFF for signed MAC; shows LIF module >11% area | Multi-bit activations raise ADC / digital MAC cost — useful for area budgeting |
| **SpikeTA** (Gao et al.) | TCAD 2025 | First FPGA TSNN accelerator; DSP-efficient add trees; depth-aware buffers; streaming + split-engine | Binary spike×weight; **attention breaks** for real QKV |
| **Xpikeformer** (arXiv 2408.08794) | — | Hybrid AIMC + stochastic spiking attention (AND-like) | Stochastic binary; weak for continuous amp |
| **Sparse Spike-driven Transformer accel** (arXiv 2501.07825) | — | Encoded spike positions; sparse ADD | Binary-centric |
| **L-SPINE** (arXiv 2604.03626) | — | Unified 2/4/8-bit SIMD SNN engine; multiplier-less shift-add | **Strong fit** for quantized ATLIF payloads |
| **SNNIM** (ISCAS 2022) | ISCAS | 10T-SRAM CIM; capacitance compute; membrane+threshold in SRAM | Analog-ish amp natural; digital ATLIF needs careful mapping |
| **DYNAP-SE2** | neuromorphic platform | Conductance dendrites, multi-compartment, delays | Analog graded dynamics — inspiration, not digital ISCAS PE |

### 1.5 Mechanisms that **break** under real-valued outputs (call-out list)

1. **AND / mask / PopCount attention** (Spike-driven Transformer SDSA; FireFly-T binary engine; Xpikeformer SSA) — assumes `{0,1}` Q,K,V.  
2. **SpinalFlow chronological binary merge-sort** without payload.  
3. **SATO binary adder-search tree** for fire-time recovery.  
4. **TrueNorth / Loihi-style 1-bit spike NoC packets** (unless extended AER).  
5. **Spike-as-mux-select** DSP tricks (FireFly binary mux) without multiply.  
6. **θ-absorption into W** (canonical AT-LIF inference trick) — **invalid** if user keeps per-event or per-token real amplitudes.

### 1.6 Mechanisms that **still work** (or adapt cheaply)

1. FireFly-v2 **bit-decompose + shift-add** multi-bit path.  
2. **Output-stationary / stRS / 4D spatiotemporal tiling**.  
3. **On-the-fly residual Vmem cache** of size \(M\times N\) (FireFly v2), not full feature×T.  
4. **Event queues / SCDQ** with widened payload.  
5. **Dual-side sparsity** if zeros remain frequent (ATLIF sparsity + weight prune).  
6. **Workload balancing** (Skydiver / SATO buckets / FireFly-T workers) keyed on nonzero gates.  
7. **PSN-style parallel temporal GEMM** (Fang et al., NeurIPS 2023) — membrane is already a real matrix; HW = temporal MAC array (great for optical flow).

---

## 2. Ranked hardware ideas (novelty 1–10) for SDformer + real ATLIF

Ranking = **novelty for ISCAS co-design given real-valued ATLIF** × relevance to **optical-flow spikeformer** (swin / SDSA / temporal tokens).  
`*` = invented / adapted specifically for this note (not a direct copy of prior HW).

### Rank A — Lead ISCAS claims (novelty 8–10)

#### A1. Hybrid Binary-Gate + Real-Payload Datapath (HBG-RP)* — **novelty 9.5**
- **Idea:** Every ATLIF output is a packet `{g, p}` with `g=1{|amp|>ε}`, `p=quant(amp)` (4–8b). PE clock / SRAM read / NoC flit gated by `g`; MAC uses `p×W` only if `g`.  
- **Why new:** Prior accel are either binary-event (FireFly-T) or dense multi-bit ANN-like; few co-design the **gate for control-path sparsity** with **payload for value**. Canonical AT-LIF is `{0,θ}` — generalize to non-absorbable payloads.  
- **OF / Transformer fit:** Swin windows often sparse in event cameras; attention can keep binary mask from `g` and real V from `p`.  
- **Estimate:** If gate sparsity 70–90%, dynamic power ≈ binary SNN; accuracy keeps real ATLIF.  
- **Risk:** Quantization of `p`; need ablation ε.

#### A2. Value-Gated Temporal Skip (VGTS)* — **novelty 9**
- **Idea:** Per neuron/channel, skip entire future timestep tiles if `|amp_t| < ε` **and** predicted membrane won’t cross θ (cheap linear predictor from leak). Unlike binary spike-skip, skip is **amplitude- and dynamics-aware**.  
- **Builds on:** FireFly-v2 temporal tiling; COMPASS speculation — but speculation uses real residual membrane.  
- **OF fit:** Static background regions in DVS → long skips.  
- **Breaks if:** Dense high-amp spikes (rare in pruned ATLIF).

#### A3. Mantissa-Shared Temporal MAC (MST-MAC)* — **novelty 8.5**
- **Idea:** Across T timesteps of one neuron, share exponent / scale (from ATLIF θ or BN) and only stream mantissas; temporal PE does shared-scale accumulate: `Y = scale · Σ_t mant_t · W`.  
- **Builds on:** Bit-serial FireFly-v2 merge; L-SPINE multi-precision; block floating-point DNN accel.  
- **ATLIF hook:** θ_i is natural per-channel scale (even if not absorbed into W).  
- **OF fit:** Short T (4–16) in SDformerFlow → small exponent cache.

#### A4. Residual Membrane Caching Across Transformer Tokens (ReMem-Tok)* — **novelty 8.5**
- **Idea:** In swin / spikeformer blocks, cache **compressed residual membrane** (not spikes) across overlapping windows / shifted windows / recurrent OF iterations — reuse Vmem between spatially adjacent tokens and temporal OF updates.  
- **Builds on:** FireFly-v2 tiny residual Vmem; Spikingformer **pre-neuron membrane residual** (Zhou et al.); SDformerFlow MS shortcuts.  
- **Why HW-new:** Most SNN HW cache spikes or full Vmem maps; few do **token-overlap membrane reuse** for transformers.  
- **ISCAS angle:** address generator + bank conflict analysis for shifted-window Vmem.

#### A5. Spike-Amplitude Gated PE Clock / DVFS (SAG-CLK)* — **novelty 8**
- **Idea:** Fine-grain clock-gate PE columns by OR-reduce of gates in a tile; optional DVFS from average `|p|` (small amps → lower precision / voltage).  
- **Builds on:** ISSCC event-driven clock-free; adaptive clock SNN processors (Li et al., TCAS-I 2021).  
- **Differentiator:** Gate from **amplitude**, not only binary spike presence.

### Rank B — Strong adaptations (novelty 6–7.5)

#### B1. Amplitude-Aware Sparse Attention Engine* — **novelty 7.5**
- Replace FireFly-T AND-PopCount with:  
  - **Path G:** binary Q_g ∧ K_g → sparse index (keep PopCount structure)  
  - **Path P:** gather real V_p (or Q_p) and MAC only on surviving indices  
- Maps SDSA (Yao NeurIPS’23) → **gated real attention** for ATLIF-SDformer.  
- Closest prior: SpikeTA / FireFly-T (binary only).

#### B2. Temporal XOR / Motion-XOR Friendly Datapath (TTX-HW)* — **novelty 7**
- **Idea:** For OF front-end, compute **temporal difference / XOR-like polarity change** on event polarity or on ATLIF gate streams: `m_t = g_t XOR g_{t-1}` (or signed amp delta) to gate heavier MACs; motion-salient tiles get full real MAC.  
- **Note:** No established “Motion-XOR” SNN accelerator paper found (2026-09 search); treat as **algorithm+HW co-design invention** inspired by event polarity and classical motion energy. Cite OF SNN algos (Spike-FlowNet, Adaptive-SpikeNet) as motivation, not as prior HW.  
- **ISCAS story:** tiny XOR/delta front-end + gated PE array.

#### B3. PSN Temporal GEMM Array for Parallel Neurons — **novelty 7**
- Map Fang et al. PSN (`H = W_temp X`) to a **T×T temporal weight stationary** array; spikes/amps after compare. SDformerFlow-v2 already uses PSN — **first HW claim** for OF spikeformer PSN.  
- Survives real-valued X perfectly (X can be multi-bit events).

#### B4. Widened AER / SCDQ with Amplitude Payload — **novelty 6.5**
- Extend Seneca-style shared delay queues to `(id, t, amp)`; OF pipelines with synaptic delays for multi-scale motion.  
- Low novelty alone; high **system** value with A1.

#### B5. FireFly-v2-style Non-Spike Unification for ATLIF — **novelty 6**
- Treat ATLIF payload as another “non-spike” mode in FireFly-v2 partial-sum merge FSM (1b / 2b / 4b / ATLIF-fp).  
- Safe engineering path; novelty = **co-design with ATLIF training**, not microarch.

### Rank C — Supporting / lower novelty (3–5.5)

| ID | Idea | Novelty | Comment |
|----|------|---------|---------|
| C1 | Dendritic fan-in combine (ESSA) for multi-bit | 4 | Useful for large OF kernels |
| C2 | Dual-side Bitmap sparsity (FireFly-S) | 5 | Keep; extend bitmap to run-length of amps |
| C3 | IMC crossbar for W, digital ATLIF periphery (SpikeSim lesson) | 4 | Area: don’t ignore neuron module |
| C4 | STDP on-chip | **2 (deprioritize)** | Irrelevant to supervised OF / transformer BPTT unless continual-learn side story |
| C5 | Rate-vs-temporal mode switch | 5 | Optical flow may mix rate (texture) + temporal (edges) |

---

## 3. ATLIF-specific mechanism section (design recipes)

### 3.1 Recommended top-level architecture (ISCAS figure story)

```
Event cam → voxel/ATLIF encode → HBG-RP packets {g,p}
        → VGTS skip controller
        → Spatiotemporal sparse engine (FireFly-T-like workers + MST-MAC)
        → Amplitude-aware attention (binary mask + real V)
        → ReMem-Tok cache (swin / recurrent OF)
        → Flow head (multi-bit / ANN-lite OK)
```

### 3.2 Microarchitecture recipes

1. **Packet format (NoC / SRAM):** `[valid g | amp_exp | amp_mant | axon_id | t]`  
2. **PE:** `if (g) acc += p * w; else nop` + optional shared exp.  
3. **Attention:** reuse FireFly-T LUT6 AND-PopCount on **gates only**; second-stage gather-MAC on payloads (small systolic).  
4. **Membrane:** keep FireFly-v2 two-phase neurodynamics but compare **real** partial sums to learnable θ; emit `{g,p}` with `p=θ` (canonical) or `p=f(m)` (user graded).  
5. **OF TTX front-end:** 1-bit history buffer per pixel/channel for XOR gate; amp delta optional.

### 3.3 What *not* to claim as novel
- Plain LIF integrate-and-fire.  
- Binary-only SDSA AND-PopCount (already Spike-driven Transformer + FireFly-T).  
- Absorbing θ into weights (NeurIPS’25 AT-LIF already says this for binary HW).

### 3.4 Ablations ISCAS reviewers will expect
- Binary spikes vs `{0,θ}` vs full real amp.  
- Gate sparsity vs EPE / AEE on DSEC & MVSEC.  
- Energy breakdown: gate path vs payload MAC vs Vmem.  
- Comparison baselines: FireFly v2 (multi-bit), FireFly-T / SpikeTA (binary transformer), ANN OF (E-RAFT class) energy model.

---

## 4. Algorithm-only papers that justify a **first HW co-design** claim

These lack dedicated accelerator papers (or only GPU energy models). Strong “we are first to map X to silicon/FPGA” narrative:

### 4.1 Optical-flow / event SNN (highest priority)

| Paper | Venue | Why HW gap |
|-------|-------|------------|
| **SDformerFlow** (Tian & Andrade-Cetto) | arXiv:2409.04082; ICPR’24 lineage | First spikeformer dense OF; LIF + **PSN** variants; only SW energy estimates |
| **Adaptive-SpikeNet** (Kosta & Roy) | ICRA 2023 / arXiv:2209.11741 | Learnable threshold & leak OF SNN — neuron params need HW configurability |
| **Spike-FlowNet** (Lee et al.) | ECCV 2020 | Hybrid SNN-ANN OF — hybrid datapath HW still scarce |
| **Best of Both Worlds hybrid SNN-ANN OF** (arXiv:2306.02960) | — | Layer-wise SNN vs ANN assignment → heterogeneous PE modes |
| **Schnider et al. Neuromorphic OF + real-time** | CVPRW 2023 EventVision | Notes **analog-valued spikes** improve OF; Loihi deploy but not custom amp-aware PE |
| STE-FlowNet / EVA-Flow / E-RAFT | AAAI’22 / Sensors’25 / … | ANN OF — contrast baselines, not SNN HW |

### 4.2 Spike transformers / neurons (map to attention + ATLIF)

| Paper | Venue | Why HW gap |
|-------|-------|------------|
| **Spike-driven Transformer** (Yao et al.) | NeurIPS 2023 | SDSA mask+ADD — HW exists (SpikeTA/FireFly-T) but **not** for real ATLIF QKV |
| **Spike-driven Transformer V2 / Meta-SpikeFormer** (Yao et al.) | ICLR 2024 | Meta architecture for next neuromorphic chips — explicit HW invitation |
| **Spikformer** (Zhou et al.) | ICLR 2023 | SSA with float attention remnants — HW usually forces binary |
| **Spikingformer** (Zhou et al.) | arXiv:2304.11954 | Pre-neuron membrane residual — motivates ReMem-Tok |
| **PSN** (Fang et al.) | NeurIPS 2023 | Parallel neuron dynamics — **no** dedicated OF/transformer PE array paper |
| **SEW-ResNet** (Fang et al.) | NeurIPS 2021 | Non-spike residual — FireFly v2 supports; still room for ATLIF residual |
| **AT-LIF + Activity Pruning** (Bu et al.) | NeurIPS 2025 | Algorithm-only; HW compatibility claim is “absorb θ” — **your non-absorbable real outputs invert that** → co-design hook |
| **Multi-Bit Mechanism / MBLIF** (Xiao et al.) | arXiv:2407.05739 | Algorithm multi-bit spikes — FireFly-v2-ish HW possible; OF+transformer unused |
| **Ternary Spike** (Guo et al.) | AAAI 2024 | Algorithm; limited OF use |
| **QKFormer** (Zhou et al.) | NeurIPS 2024 | Hierarchical QK attention SNN — HW open |
| **CATFormer DTLIF** | arXiv 2603.15184 | Dynamic thresholds — related to ATLIF; HW open |

### 4.3 Gap statement (use in intro)
> Prior FPGA/ASIC SNN accelerators (SpinalFlow, SATO, FireFly series, SpikeTA, STELLAR, COMPASS) optimize **binary** spikes or limited multi-bit SEW residuals. Algorithmic OF spikeformers (SDformerFlow) and ATLIF-like **real-valued** neurons have **no** published amplitude-aware transformer+flow accelerator. This work closes that gap with HBG-RP + VGTS + ReMem-Tok …

---

## 5. Suggested ISCAS contribution packaging

**Title sketch:** *Amplitude-Aware Acceleration of Spiking Transformers for Event Optical Flow with Real-Valued ATLIF Neurons*

**3 bullets:**
1. HBG-RP datapath preserving event-driven sparsity with real ATLIF payloads.  
2. VGTS + ReMem-Tok for spatiotemporal OF / swin reuse.  
3. Amplitude-aware attention adapting SDSA/FireFly-T to non-binary QKV; FPGA/ASIC results on DSEC/MVSEC energy vs AEE.

**Baselines to implement / model:** FireFly v2, FireFly-T or SpikeTA (binary), dense INT8 ANN transformer OF, GPU SDformerFlow energy table.

---

## 6. Core bibliography (real citations)

1. S. Narayanan et al., “SpinalFlow: An Architecture and Dataflow Tailored for Spiking Neural Networks,” **ISCA**, 2020.  
2. F. Liu et al., “SATO: Spiking Neural Network Acceleration via Temporal-Oriented Dataflow and Architecture,” **DAC**, 2022.  
3. J. Li et al., “FireFly: A High-Throughput Hardware Accelerator for Spiking Neural Networks…,” **IEEE TVLSI**, 2023.  
4. J. Li et al., “FireFly v2: Advancing Hardware Support… Spatiotemporal FPGA Accelerator,” **IEEE TCAD**, 2024. (arXiv:2309.16158)  
5. T. Li et al., “FireFly-S: Exploiting Dual-Side Sparsity…,” **IEEE TCAS-I**, 2024.  
6. T. Li et al., “FireFly-T: High-Throughput Sparsity Exploitation for Spiking Transformer…,” arXiv:2505.12771, 2025.  
7. R. Mao et al., “STELLAR: Energy-Efficient and Low-Latency SNN Algorithm and Hardware Co-Design…,” **HPCA**, 2024.  
8. A. Moitra et al., “SpikeSim: An End-to-End Compute-in-Memory Hardware Evaluation Tool…,” **IEEE TCAD**, 2023.  
9. Y. Gao et al., “SpikeTA / Advancing Neuromorphic Architecture towards Emerging Spiking Neural Network on FPGA,” **IEEE TCAD**, 2025.  
10. Z. Wang et al., “COMPASS: SRAM-Based Computing-in-Memory SNN Accelerator with Adaptive Spike Speculation,” **MICRO**, 2024.  
11. Q. Chen et al., “Skydiver: A Spiking Neural Network Accelerator Exploiting Spatio-Temporal Workload Balance,” **IEEE TCAD**, 2022.  
12. M. Yao et al., “Spike-driven Transformer,” **NeurIPS**, 2023.  
13. M. Yao et al., “Spike-driven Transformer V2: Meta Spiking Neural Network Architecture…,” **ICLR**, 2024.  
14. Z. Zhou et al., “Spikformer: When Spiking Neural Network Meets Transformer,” **ICLR**, 2023.  
15. W. Fang et al., “Parallel Spiking Neurons with High Efficiency…,” **NeurIPS**, 2023.  
16. W. Fang et al., “Deep Residual Learning in Spiking Neural Networks (SEW-ResNet),” **NeurIPS**, 2021.  
17. Y. Tian & J. Andrade-Cetto, “SDformerFlow: Spatiotemporal Swin Spikeformer for Event-based Optical Flow Estimation,” arXiv:2409.04082, 2024.  
18. A. K. Kosta & K. Roy, “Adaptive-SpikeNet…,” **ICRA**, 2023.  
19. C. Lee et al., “Spike-FlowNet…,” **ECCV**, 2020.  
20. T. Bu, X. Shi, Z. Yu, “Activity Pruning for Efficient Spiking Neural Networks (AT-LIF),” **NeurIPS**, 2025.  
21. Y. Xiao et al., “Multi-Bit Mechanism: A Novel Information Transmission Paradigm for SNNs,” arXiv:2407.05739, 2024.  
22. Y. Guo et al., “Ternary Spike…,” **AAAI**, 2024.  
23. ISSCC 2022: “An 82nW 0.53pJ/SOP Clock-Free Spiking Neural Network…,” **ISSCC**, 2022.  
24. Synaptic delay / SCDQ: arXiv:2404.10597; arXiv:2501.13610.  
25. Xpikeformer: arXiv:2408.08794.  
26. L-SPINE: arXiv:2604.03626.  
27. SNNIM: **ISCAS**, 2022.  
28. ESSA: **IEEE TVLSI**, 2022.  
29. Hybrid OF SNN-ANN: arXiv:2306.02960.  
30. Schnider et al., “Neuromorphic Optical Flow…,” **CVPRW**, 2023.

---

## 7. Bottom line for the parent agent / authors

- **Do not** sell binary FireFly-T attention as ATLIF-ready.  
- **Do** sell **HBG-RP + VGTS + ReMem-Tok + amplitude-aware attention** as the ISCAS novelty stack.  
- Canonical NeurIPS’25 AT-LIF is `{0,θ}` absorbable — if the user’s outputs are truly continuous / non-absorbable, that is the **wedge** against “just fold into weights.”  
- Strongest algorithm anchors: **SDformerFlow + PSN + AT-LIF/MBLIF + Spike-driven Transformer**.  
- Strongest HW anchors to beat/extend: **FireFly v2 (multi-bit)** and **FireFly-T / SpikeTA (binary transformer)**.
