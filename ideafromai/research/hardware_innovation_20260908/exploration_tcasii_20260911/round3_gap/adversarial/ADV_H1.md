# ADV — H1 typed last-use after round-3 priors (2026-09-11)

Reviewer who has ESTU (TCAS-II 2025), FireFly-T, DATE 2025 hybrid, Spike-IAND, Fang CICC, and ASNA-Flow on the desk.

---

## Hostile mapping

| Letter says | Reviewer maps to |
|---|---|
| dual-engine / hybrid overlay | FireFly-T (sparse conv + binary attention) or DATE (dense input + sparse rest) |
| skip inactive spikes / groups | ESTU group-4 stack; FireFly-S bitmap; Fang NZ fetch |
| residual port | FireFly-T pre-neuron membrane I/O; SDT MS |
| keep binary residual | Spike-IAND **deleted** ADD to stay binary — why is PED still here? |
| spatial skip for OF | ASNA-Flow abstract, SENECA grouping, EventShiftFlow occupancy |
| mW / LUT / FPS | ESTU same journal |
| continuous AT-LIF amplitude shared by two MACs | **identity error**; NeurIPS AT-LIF absorbs θ |

To survive, the letter must say in **one sentence** an object that is **none** of those rows.

H1’s sentence: *one post-absorb producer is not free until both the binary-GeMM use and the continuous PED/I24 use have retired.*

That sentence is **not located** in the opened FireFly-T, DATE, ESTU, Spike-IAND, Fang-abstract, FlexSpIM, SpikePool, SENECA, EventShiftFlow texts.

---

## Why it still fails as a title today

1. **No wait-class.** Source long-BP is 8088 on **both** students, with lifting FIFO-full wait **larger** (1903 vs 1159). Discriminating prediction: that 8088 is the **test’s blocked sink**, not PED last-use. If true, H1 does not explain the only stubborn number we have.
2. **Union density.** If PED is dense, last-use of the producer is PED’s last-use; the extra bit does no work. Kill-gate: report `s_done` vs `r_done` lag histogram. If lag ≈ 0, stop.
3. **Same-port 15%.** Integer two-consumer −5.78%; serialized full chain −6.4% not closed; delayed-V arithmetic_saving=0. Nothing yet shows typed last-use pays 15%.
4. **Five pages.** Scoreboard + overlay + OF + BN is a zoo. One mechanism only.

---

## What would make H1 informative (not yet done)

On a **dual-consumer** kernel (not the source-only tile):

- infinite sink vs contracted sink;
- wait tags `{fifo, spike_BP, PED_BP, BN, other}` summing to the stall;
- `s_done` vs `r_done` cycle lag;
- same-port service of typed retire vs fused writeback vs single refcount.

If PED_BP and lag are both real and service ≥15% with AEE held, **then** revise H1 into a one-page mechanism. Until then: **instrument, do not draw RTL.**
