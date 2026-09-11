# R3L4 — Codex full_chain observations only (2026-09-11)

**Label: observation.** Not a claim, not net service, not a title. No RTL. Identity: AT-LIF absorb; continuous work is PSN-before-threshold, residual, PED — not analog AT-LIF amplitude.

Sources: `schedule_compare_same_port/full_chain/README.md`, `consumer_lifetime_result.json`, `preview_v_schedule_result.json`, parent `source_service.md`, `two_stage_writeback/result.json`, `consumer_service_result.json`.

---

## 1. Frozen numeric observations (verified in JSON/MD this round)

### AEE (students, not this capture’s new valid825)

- ordinary dense/raw: **1.219801338**
- lifting-raw: **1.232979368**
- Δ **+0.013178** — absolute ≤1.259 passes; relative ≤+0.005 **fails**.

### Source CSE (source_service.md, P2 grey zone)

- arithmetic mix per T10 vector: ordinary **260** add/sub; lifting **159 add + 35 RNE + 35 sat + 10 gate**
- always-ready: **6914 → 6170 (−10.7608%)**
- long back-pressure: **both 8088**
- FIFO-full wait under long-BP: ordinary 1159, lifting 1903 (lifting waits **more**)

### Two-stage writeback (different construction; do not add to the table above)

`two_stage_writeback/result.json` comparisons.ready / blocked:

- ordinary unfused/fused ready: **6938**
- lifting fused ready: **5354** → fused_structure_reduction **22.8308%**
- lifting unfused ready: 6194 (−10.72% vs ordinary)
- **blocked: all four 8088**, fused reduction **0%**

Round-2 “6938→5354 (−22.83%)” is this fused writeback ready column, **not** `source_service.md` 6914→6170. Tables are **not additive**.

### Integer consumer (parent `consumer_service_result.json`)

- 758777 → 714889 slots (−5.78%). Separate integer-consumer construction; **not** full-chain net service.

### Full-chain serialized frame (`full_chain/README.md`) — **NOT closed same-port net service**

| Stage | ordinary slots | lifting slots | share of ordinary |
|---|---:|---:|---:|
| full T10 source | 352,972,800 | 292,147,200 | 17.18% |
| FP preview + noncausal sn2 | 1,005,355,863 | 947,579,079 | 48.93% |
| integer residual + two consumers | 465,782,418 | 454,704,592 | 22.67% |
| native proj before BN | 175,737,993 | 173,963,838 | 8.55% |
| full-domain BN + reread + add | 54,723,072 | 54,723,072 | 2.66% |
| **this construction total** | **2,054,572,146** | **1,923,117,781** | 100% |

Lifting **−6.3981%** on this serialized appointment model. README: not a closed cycle table; BN reduction not closed; FP/integer consumers not fully address-timed; not full Gustav Skip/Fetch/Exec. **Cannot pass or kill the family.**

### Delayed-V / native BN (`consumer_lifetime_result.json`)

- `arithmetic_saving`: **0**
- `native_BN_reduction_hardware_closed`: **false**
- `early_V96_bytes`: **55,296,000**
- `late_U32_bytes`: **18,432,000**
- integer 0-diff on four blocks vs continuous exit (README)
- DMA-protocol bus occupancy released ≠ net service (README)

### Preview-V microkernel (`preview_v_schedule_result.json`)

- ordinary always-ready **10396** slots (direct prefetch, 8 FP32 FMA lanes)
- transpose layout 11356; long-BP 13362 vs 14145
- `native_BN_statistics_closed`: **false**

### Native proj BN identity

eval, `track_running_stats=False`, empty running mean/var; must wait **192,000** values per channel; input `[10,96,120,160]` = 18,432,000 FP32 = 73,728,000 B raw. Aligns with SDformerFlow “disable tracking of running states” (software).

Integer gates / I24 / PED q24: 0-diff vs model on two captured windows. Final FP32 max diffs ~1.6e-4 (cuDNN TF32 vs NumPy). Not whole-chain equivalence.

---

## 2. Map to round-2 islands (still not titles)

| Observation | Maps to | Why |
|---|---|---|
| long-BP **8088 = 8088**, lifting FIFO wait **larger** (1903 vs 1159) | **G1 question, not G1 win** | Always-ready can drop; the contracted long-BP does not. Rival explanations: output FIFO (already in the source kernel), residual/PED still live, BN barrier. **Wait-class histogram of 8088 is still missing.** |
| fused writeback −22.83% ready / 0% blocked | generic round→sat fusion **A** + same FIFO | Part of ready gain is generic fusion, not lifting X. Blocked still 8088. |
| integer consumer −5.78% | not 15% | Separate table. |
| serialized −6.40% | **stop as performance claim** | Construction gap; FP preview is ~49% of this bill. |
| delayed V, arithmetic_saving=0 | **G3 hygiene / lifetime A** | Both axes get it. Bus ≠ service. |
| live full-domain BN | **G3 hygiene** | Software identity confirmed; hardware reduction **not closed**. |
| preview 10396 | layout A | Scratch transpose **stopped** as X. |

---

## 3. What Codex still owes before G1 can be a mechanism

A **wait-class dump** of the 8088 long-BP slots, same resource point, both students:

1. output FIFO full (already instrumented as 1159 / 1903)
2. residual / PED / raw I still live (dual last-use)
3. BN-stat barrier (should be **zero** on the **source** kernel if BN is downstream — if 8088 is the source kernel, BN cannot be the cause **of that 8088**)
4. instruction / RF / ROM structural

**Prediction (candidate, not finding):** if 8088 is the source kernel with a fixed blocked receiver waveform (896/1024 unready), then G1 (PED last-use) is **not** the cause of *that* 8088; the cause is the **test’s output contract**. G1 could still be the cause of a **later** dual-consumer kernel’s back-pressure. Discriminating test: rerun source kernel with infinite sink; if both drop below 8088 and lifting keeps an advantage, 8088 was FIFO. If both stay 8088, the stall is inside the source. If only the dual-consumer kernel shows 8088, look at PED last-use.

Do not draw new RTL until that dump exists.

HEAD / uncommitted status: not re-checked this file; do not treat as git fact.
