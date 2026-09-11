# ADV — ESTU, same journal: desk-reject of “another spikeformer FPGA with skip”

**Freeze:** 2026-09-11  
**Role:** what a TCAS-II reviewer who just handled ESTU does to a letter that still looks like ESTU.  
**Contract:** `PROBLEM.md` / `SCOPE.md`. Identity is ATLIF **continuous θg**, dual consumers, DSEC valid825 AEE, same-port service — not binary SSA overlay.

---

## Source status (do not treat as invented)

| Page | Result |
|---|---|
| Author accepted manuscript (CC-BY), 5 pages | **Opened in full.** PDF header prints *Citation information: DOI 10.1109/TCSII.2025.3626209*. Title, authors, equations, tables, and SOTA comparison below are from this PDF. URL: `https://cdn.prod.website-files.com/66ea88c9d1833f340063dd7b/6908b1f223438480c621533f__7ao3FYSCYjS1X-vxTLubmyaFxfScXHqWYukL3y3AU4.pdf` |
| IEEE Xplore `11218914` | Landing HTML **not retrieved** (empty fetch). Record metadata from IEEE/W1: *IEEE Trans. Circuits Syst. II, Exp. Briefs*, vol. 72, no. 12, pp. 2027–2031, Dec. 2025; date of publication 27 Oct. 2025. |
| GitHub `EOLAB-2025/ESTU` | Direct fetch **failed**. README recovered only as search snippets (iCE40UP5K SoC, SERV, encode/decode, integer SpikeTransformer notebook). Treat code layout as **search-incomplete**. |
| UNICA IRIS `11584/460645` | Abstract only; “no files attached.” |
| EdgeAI project page | Secondary summary of the same letter (3.76 mW / 0.23 mW / 4.28 μJ / NinaPro 87.21%). Not a substitute for the PDF. |

**Paper body: opened. IEEE HTML + GitHub tree: search-incomplete.** DOI below is the string printed on the opened AAM, also listed on the IEEE record and in `web/W1_2025_2026_venues.md` P12 — not minted here.

- **Title:** ESTU: Enabling Spiking Transformers on Ultra-Low-Power FPGAs  
- **Authors:** Leone, Busia, Orrù, Raffo, Meloni (Univ. of Cagliari)  
- **DOI (on AAM / IEEE record):** 10.1109/TCSII.2025.3626209  

W1 already marked ESTU **CONTROL (same journal, binary spiking transformer)**. This sheet is the reviewer-voice of that control.

---

## What ESTU actually is (from the opened letter)

ESTU is a **5-page TCAS-II Express Brief** whose object is: *binary* spiking transformer **on a 5k-LUT FPGA**, by **reusing one microcode datapath** and **skipping inactive spike groups**.

Mechanisms the reviewer now treats as “already in this journal”:

1. **Binary SSA.** Q/K/V are binarized; softmax is gone; attention is \(QK^\top V / \text{scale}\) on spikes (eq. 1, Spikformer [8] in their refs).
2. **LIF that *emits* a binary spike.** Eq. (2): \(s(t)=v(t)>\theta\), then reset. Integer ops exist, then **LIF binarizes** the result into spike memory. Continuous membrane is not a second consumer.
3. **Operator overlay, not two live consumers.** Table I: `Dense(spike)`, `Dense(int)`, `Sum(spike,*)`, `Mul(spike,spike)` (16 AND + 16-input **popcount**), `Mul(spike,int)`. A result is written to **spike mem or integer mem or through LIF**. That is typed storage, not a fork of one source vector into gate **and** residual/PED.
4. **Skip = activity stack on binary groups.** Spikes packed in groups of 4; stack stores only groups with ≥1 active bit. Table II: software sparsity 0.95 → exploited 0.82 at group-4. Idle when no events. This is the journal’s current meaning of “sparsity skip” for spikeformers.
5. **Task = classification, not dense flow.** NinaPro DB-5 sEMG 87.21% (86.97% after 8-bit W); +2% vs vanilla SNN; also prosthetic-hand and EEG models (their Table V). Metric is **accuracy %** and **real-time ms**.
6. **Figures of merit:** iCE40UP5K, 4301 LC / 30 BRAM / 6 DSP / 21 MHz; shunt-resistor **3.76 mW** inference, **0.23 mW** standby, **4.28 μJ**/inf @ 12 MHz; 29% time from sparsity; 0.192 GOPS, 51.06 GOPS/J (Table VI). Eq. (3) sums `sparsity×data/throughput` over operators.

Headline they already used in *this* journal: first low-power spiking-transformer **edge FPGA**; overlay + skip + mW + class-%.

---

## Desk-reject script (reviewer who just accepted / just saw ESTU)

> “We published ESTU in TCAS-II 72(12). It is a microcode overlay that maps binary SSA onto a tiny FPGA and skips inactive spike groups. This submission is another spiking-transformer accelerator with skip. The increment is a larger FPGA, a different classifier, or a different skip encoding. That is not a new circuit object. Reject.”

Triggers that fire the script (any two is enough):

| If the letter says… | Reviewer maps it to ESTU |
|---|---|
| “spiking transformer on FPGA” as the title | ESTU abstract / contribution 1 |
| overlay / microcode / reused datapath for all layers | ESTU §III.A–C, Fig. 3 |
| skip zeros / skip inactive neurons / activity stack / NRV-like row skip | ESTU §III.B Table II (group-of-4) |
| AND + accumulate / popcount SSA | ESTU 16 AND + 16-input popcount |
| LIF then binary spike as the *output* of the block | ESTU eq. (2) |
| LUT / DSP / mW / μJ/inf as the story | ESTU Tables III–VI, Fig. 4 |
| CIFAR / ImageNet / DVS-Gesture / sEMG / EEG **accuracy** | ESTU Table V (sEMG/EEG) |
| “first edge spikeformer hardware” | ESTU already claimed this, same journal |
| integer path that is then **binarized** | ESTU `Dense(int)` → LIF → spike mem |

A bigger Xilinx part, FireFly-T-style dual-**engine** (sparse vs binary attention), or “we also skip” does **not** escape. Those are ESTU’s cited SOTA (their [11]–[13]) plus ESTU’s own skip. The AE will say: *cite ESTU, state the increment in one sentence, or this is a reskin.*

---

## Collision is the **object**, not the board

ESTU’s object, compressed to one line:

**binary spike tensor → one accumulate consumer → skip inactive groups → classification accuracy + mW.**

A letter that measures that object — even with a cleverer skip, a compiled adder tree, or tick-batch — **collides**. Same journal, same five-page slot, same reviewer pool. Incremental LUT/mW on that object is ESTU Table VI with a new row.

What does **not** save us:

- Calling skip “product sparsity,” “NRV,” “dual-side,” or “dual-engine.” ESTU already skips; FireFly-T already split engines. Reviewer hears “skip.”
- Keeping a residual **add** that is then LIF-binarized. ESTU already has integer add then LIF. ISCAS 2025 IAND (W1 P34) even *deletes* residual to stay binary — the anti-prior.
- Reporting source-only always-ready slots. ESTU already reports operator ops/cycle × sparsity. Neither is dual-consumer same-port service.

---

## Different object the letter **must** measure (or it is ESTU)

Four quantities ESTU does **not** measure. If any one is missing, the letter still looks like ESTU-plus-skip.

### 1. Continuous θg (not \(s\in\{0,1\}\))

ATLIF **threshold amplitude θg** is the source value. ESTU’s neuron legally throws that amplitude away in eq. (2). AND/popcount and group-of-4 skip are defined on bits. A skip that drops a “silent” group **deletes θg** that PED still needs.

**Measure:** integer θg (and I24) 0-diff vs the frozen student on the captured windows. If the datapath can be described as “spike present / absent,” it is ESTU.

### 2. Dual consumers after one source (not typed memories)

One compiled T10 / θg word must be **live for two consumers at once**:

- (i) spike/gate path (integer gate),  
- (ii) continuous residual / PED (I24 / q24).

ESTU’s spike-mem vs int-mem is a **muxed destination**, not a dual-ready fork. Slot retirement = `gate ∧ PED` (and BN-stat if that path is live). Occupancy is `support(θg) ∪ support(PED)`, not binary NRV.

**Measure:** handshake / service trace that splits `wait_gate` vs `wait_PED` vs joint stall. A single “stall while sparse” bit is ESTU Table II.

### 3. DSEC valid825 AEE (not class-%)

Task identity is event-camera **2D optical flow**, frozen Motion C12 / H67 / ep34.

| Gate | Number |
|---|---|
| Ordinary dense-source AEE | 1.219801338 |
| Absolute | ≤ 1.259 |
| Relative vs ordinary | ≤ +0.005 ⇒ AEE ≤ 1.224801338 |
| Lifting T10 as currently measured | 1.232979368 — **fails** the strong relative gate |

NinaPro / CIFAR / ImageNet accuracy is ESTU’s object. Reporting it as the headline **is** the collision.

### 4. Same-port / same-state / same-backpressure **net service** (not ops/cycle × sparsity)

ESTU never had two consumers fighting two ports. This net already showed the trap: always-ready 6938 → 5354 (−22.83%) while **long backpressure stayed 8088 = 8088**. Source-only skip is absorbed at the join.

**Measure:** complete-chain net service **≥ 15%** on the **same** two-stage SIMD resource; 8088 must **move**. Do not add the SIMD-source table to the separate integer-consumer −5.78%. Do not convert that % into FPS.

---

## One-sentence non-collision claim (what the five pages are allowed to be)

A shared compiled T10 source of **continuous ATLIF θg** that **simultaneously serves** an integer gate and a q24 PED under **the same two ports**, with **DSEC valid825 AEE** inside the gates above and **net service ≥ 15%** after dual completion — not a binary-spike overlay that skips inactive groups.

If the abstract can be rewritten as “FPGA spiking transformer with sparsity skip” without mentioning θg, two consumers, DSEC AEE, and same-port service, a TCAS-II reviewer who just handled ESTU will desk-reject it.

---

## Kill-if-collides (author checklist)

- [ ] Title/abstract still readable as ESTU + skip / overlay / mW.  
- [ ] Neuron output is a binary spike; θg is only a comparator.  
- [ ] Skip unit is a spike group / NRV / popcount-zero.  
- [ ] Second “consumer” is a dense engine **or** an integer memory, not PED/I24.  
- [ ] Headline metric is classification % or GOPS/W.  
- [ ] Service is operator throughput × sparsity, not dual-completion same-port cycles.  
- [ ] ESTU is missing from the relative-prior paragraph.

Any checked box: the letter collides with TCAS-II 72(12). Uncheck all four objects in the section above, or do not send.
