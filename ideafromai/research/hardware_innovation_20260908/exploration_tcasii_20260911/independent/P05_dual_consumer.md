# P05 — Dual-consumer residual datapath (independent ideation)

**Participant:** P05  
**Perspective:** one producer (T10 source / continuous ATLIF θg) fans out to (i) a spike/gate consumer, (ii) a continuous residual/PED consumer, and (iii) a later native-projection path whose BN legally needs **full-domain** actual batch statistics over `10×96×120×160`. Residual I24 is reread. Last-use of Z vs PED U is not an architectural signal. Source-only arithmetic already lost its gain to long backpressure (always-ready 6938→5354, −22.83%; long backpressure **both 8088**).  
**Rule:** each item is **one** mechanism, not a module zoo. At most one island can be the TCAS-II paper. Do not stack I1–I8.  
**Identity (hard):** ATLIF θg stays continuous; no binary-spike retitle; no analog CIM; no Yosys/OpenROAD as foundry PPA; do not add the SIMD-source table to the integer-consumer table; do not multiply component speedups into FPS.

**Frozen numbers used as controls (not claims):**

| Quantity | Value |
|---|---|
| Ordinary dense-source/raw valid825 AEE | 1.219801338 |
| Learnable 40-coeff lifting T10 raw AEE | 1.232979368 (Δ +0.013178; abs 1.259 pass, rel +0.005 **fail**) |
| Ordinary T10 graph after official CSE | 260 add/sub per vector |
| Lifting T10 graph | 159 add/sub + 35 intermediate RNE/sat |
| Same two-stage SIMD source, always-ready slots | 6938 → 5354 (−22.83%); part of this is generic round→sat fusion |
| Same resource, long backpressure | **8088 = 8088** (source-only gain absorbed) |
| Separate integer consumer model | 758777 → 714889 (−5.78%); **do not add to the SIMD table** |
| Integer gates / I24 / PED q24 vs model (two captured windows) | 0 difference |
| Native-projection BN | actual batch stats, full `10×96×120×160`; local-window replay with **free** mean/var **undercharges** wait/storage |
| Time at this PSN | noncausal T10 |

**Global kill-gates (every idea inherits these; an idea may add a tighter local gate):**

1. valid825 AEE **absolute ≤ 1.259** and **relative ≤ +0.005** vs ordinary 1.219801338 ⇒ AEE ≤ 1.224801338. Lifting 1.232979368 is **not** a passing strong control.
2. **Complete-chain** same-port / same-state / same-backpressure net service **≥ 15%**. Source-only always-ready slots are **not** service.
3. Long-backpressure cycle count on the same two-stage SIMD source resource must **move** from 8088. If 8088 stays, the idea is dead even if always-ready looks better.
4. Integer gates / I24 / PED q24 remain **0-diff** on the two captured windows.
5. BN uses **actual full-domain batch statistics**. Frozen running stats, Ghost-BN, or “mean/var given free” are identity/accounting cheats → kill.
6. Do not binarize ATLIF. Do not quote OpenROAD/Yosys as ASIC PPA. Do not add the two resource tables.

**Causal hole this perspective treats as the paper island (observation, not novelty):**

```
T10 / θg produce
    ├─ gate / spike-law consumer          (thin or wide? unknown until measured)
    ├─ continuous residual / PED consumer (I24 / q24)
    └─ (later) native conv → BN(full 10×96×120×160 actual stats) → residual ADD
                                                              ↑
                                                    I24 skip must still be alive
                                                    unless last-use of Z vs U
                                                    legally retires it earlier
```

Source CSE/lifting shortens the producer. The join at the right-hand side does not care. That is why 8088 did not move.

---

## P05-I1 — Completion-vector hybrid fork (eager PED, early-eval gate, join on last sample)

**One sentence.** Replace always-ready source occupancy with a 3-bit completion vector `{gate_sampled, PED_sampled, BNstat_sampled}` implemented as a Cortadella/Carloni elastic fork plus early-evaluation anti-token, so a producer slot retires only when the last dual-consumer (including the BN-stat tap) has actually sampled.

**A (complete prior to copy, not a slogan).** Copy in full:

- Carloni, McMillan, Sangiovanni-Vincentelli, *Theory of latency-insensitive design*, IEEE TCAD 2001: patient processes, relay stations, latency equivalence.
- Cortadella, Kishinevsky, Grundmann, *SELF: Specification and design of a synchronous elastic architecture*, plus *Synthesis of synchronous elastic architectures* (DAC 2006): valid/stop, elastic buffers, **lazy fork** (wait all ready) vs **eager fork** (fire per ready consumer), join, and the documented combinational-cycle hazard of lazy-fork+join.
- Cortadella, Kishinevsky, *Synchronous elastic circuits with early evaluation and token counterflow* (DAC 2007): forward tokens + backward **anti-tokens** that cancel a copy a consumer will never use.
- ARM AMBA AXI4-Stream (IHI0051) TVALID/TREADY/TLAST register-slice, as the synthesizable bundled-data instance of the same handshake.
- CDC 6600 scoreboard / Tomasulo CDB valid bits, as the 3-bit completion vector encoding.

Do **not** invent a new handshake algebra. Instantiate SELF fork+join+early-eval on this producer.

**B (hole in THIS net).** Same two-stage SIMD source: always-ready 6938→5354 (−22.83%) while long backpressure stayed **8088=8088**. The producer is not the bottleneck; the unmodeled join of gate, continuous PED, and later BN-stat is. Always-ready counts vacancies on the source side of a stall that lives on the consumer side. Part of −22.83% is generic round→sat fusion and is not a paper.

**X (why not a reskin).** Textbook eager fork **duplicates** the full I24 to every consumer; textbook lazy fork **barriers** all consumers and recreates 8088. THIS net needs a **hybrid**: PED/I24 may take an eager wide copy; gate may early-evaluate from θg (continuous amplitude, not a binary spike) and send an anti-token for any I24 replica it does not use; BN-stat (see I3 if chosen instead; here only a 1-bit “sampled” in the vector) samples without taking an I24 replica. Slot retirement = last of the three sample bits, which is exactly the measured 8088 region. That hybrid is not “add a FIFO.”

**Strongest controls.**

- Ordinary dense T10 (AEE 1.219801338) as the numeric control; lifting T10 only if it also meets rel ≤+0.005.
- Same two-stage SIMD source resource, **long-backpressure** cycles (8088), not always-ready slots.
- Integer 0-diff on gates/I24/PED q24 captured windows.
- Handshake occupancy trace: cycles spent in `{wait_gate, wait_PED, wait_BNstat, joint}` must be published; a single stall bit is illegal.

**Kill-gate (number).** On the same two-stage SIMD resource: long-backpressure cycles **≤ 6875** (≥15% below 8088) **and** complete-chain same-resource net service ≥15% **and** AEE ≤ 1.224801338 **and** 0-diff integer captured windows. If always-ready moves and 8088 does not, **kill I1**.

**Two-sentence TCAS-II pitch.** Source-graph CSE and lifting reduced arithmetic and even always-ready slots, but the dual-consumer join plus BN-stat sampling never saw that reduction, so long backpressure stayed 8088. A 3-bit elastic completion vector with hybrid eager/early-eval fork is the circuit that makes producer issue equal consumer completion, which is the quantity TCAS-II can measure in five pages.

**Biggest objection.** Elastic fork/join is 2001–2007 textbook; reviewers will say “you added valid/ready.” The answer has to be the **measured 8088 invariance** plus the hybrid (not eager, not lazy) forced by continuous θg + wide PED + BN-stat; if the trace does not split those three waits, the objection wins.

**Assumptions.**

- Gate, PED, and BN-stat do **not** become ready on the same cycle for most T10 vectors (otherwise a lazy fork is optimal and I1 is empty).
- A 3-bit scoreboard plus two anti-token wires fits in the same-port budget (no extra SRAM port).
- Early-eval of the gate from θg does not legally require the full I24 replica (must be checked against the integer gate law).

**Predictions.**

- P1. Occupancy histogram will show `wait_PED` or `wait_BNstat`, not `wait_source_ALU`, dominates 8088.
- P2. Hybrid fork reduces long-backpressure ≥15% without changing AEE at 1e-9 relative on ordinary T10.
- P3. Pure eager fork (duplicate I24) will **increase** SRAM-port pressure and may fail the same-port gate even if 8088 drops.
- P4. Pure lazy fork will reproduce 8088 within 2%.

**Disconfirmers.**

- If a cycle-accurate trace shows gate, PED, and BN-stat ready on the same cycle for ≥95% of vectors, hybrid fork has nothing to fire → kill.
- If anti-tokens require a fourth SRAM port, same-port fails → kill.
- If gate integer law reads the full I24, early-eval is illegal → I1 collapses to lazy fork → kill.

---

## P05-I2 — Last-use recolor of Z versus PED U (in-place I24 bank, kill the silent reread)

**One sentence.** Compile a hardware last-use tag for Z vs PED U onto one I24 SRAM bank so the residual is **recolored in place** when the earlier tensor dies, eliminating the I24 write-then-reread that currently acts as a third, uncounted consumer.

**A (complete prior to copy).**

- Chaitin–Briggs graph coloring + Cooper/Simpson live-range splitting, as the **allocation algorithm** whose output becomes a 1-bit last-use tag per vector, not as a software pass run once offline and forgotten.
- Sweldens lifting in-place DWT (predict overwrites odd, update overwrites even): the **storage discipline** of last-use overwrite. Copy the discipline; do **not** retitle the paper as lifting (lifting T10 already fails rel +0.005).
- NVIDIA SASS operand `.reuse` / architected last-use (IBM z/Architecture last-use operand patents, e.g. US9483267 / US20140095848): the **instruction-bit** that frees a physical register at the consuming op.
- DSP overlay / IDMA bank recolor (TI C6x, Hexagon L2 overlay): the SRAM-bank interpretation of the same tag.

Copy the full live-range algorithm, the in-place overwrite rule, and the architected last-use bit. Do not “share a buffer” by hope.

**B (hole in THIS net).** Dual consumers keep Z and U logically distinct; residual I24 is written then reread for the later native add. If last-use(Z) precedes first-use(U), or they are complementary in the noncausal T10 window, the reread is a scheduling artifact that occupies the same limited ports that already absorbed source-only gain. BN-full-domain wait extends that live range unless last-use is allowed to fire **before** affine (only legal if BN-stat has already sampled Z/U or does not need them).

**X (why not a reskin).** Compiler overlay without a **BN-domain last-use** is a silent resource cheat: the tensor looks dead to the gate/PED pair but is still legally live until residual add after BN. The tag is generated by the **same** completion event that retires T10/θg for that vector, and it is forbidden to recolor while BN-stat still owes a sample of that location. That coupling is the circuit, not “memory reuse.”

**Strongest controls.**

- Live-interval dump on the two captured windows: must prove `live(Z) ∩ live(U) = ∅` **or** a documented split point, **including** the BN-stat sample and the residual-add sample.
- Port-cycle counters: I24 residual reread cycles → 0.
- AEE 0-diff vs integer residual (reread elimination must be bit-true, not approximate).
- Same-port: overlay must not add a third bank-tag SRAM.

**Kill-gate (number).** On captured windows, residual I24 **reread cycles = 0** and `max_live(Z,U)` in that bank ≤ 1 vector-width, **and** AEE ≤ 1.224801338, **and** complete-chain service ≥15%. If live intervals overlap anywhere in the two windows, overlay is illegal → **kill I2** (do not “approximate” the overlap). If overlay needs a domain-sized shadow (`10×96×120×160`), it is not overlay → kill.

**Two-sentence TCAS-II pitch.** The residual SRAM currently stores Z and then rereads I24 for PED U and for the post-BN add, so a dead tensor still backpressures the producer. An architected last-use bit that recolors one I24 bank at the earlier of {gate-done, PED-done} **subject to BN-stat having sampled** removes that phantom consumer without changing the integer residual law.

**Biggest objection.** “You just did register allocation.” If the live-interval dump does not include BN-stat and residual-add as first-class uses, the objection is correct.

**Assumptions.**

- Z and U are the same storage width (I24/q24-class) or one is a legal truncation of the other.
- Last-use order is **static per layer** (or a 1-bit dynamic tag from the completion vector), not data-dependent in a way that requires a second bank.
- Residual add after BN is the last use of the skip; nothing downstream rereads Z.

**Predictions.**

- P1. On captured windows, last-use(Z) occurs at gate or at PED-form, whichever is later, **before** native conv finishes; U then occupies the same bank until add.
- P2. Eliminating the reread moves long-backpressure (8088) by ≥ the fraction of 8088 that is SRAM-port stall, predicted ≥15% if port stall dominates.
- P3. If BN-stat is ignored, a “successful” overlay will still fail complete-chain service because affine+add re-introduces the reread.

**Disconfirmers.**

- Any captured window with overlapping Z/U live ranges → kill.
- If residual add needs **both** Z and U, recolor is illegal → kill.
- If q24 U is not a bit-identical overlay of I24 Z (scale/offset differ), recolor changes the integer law → 0-diff fails → kill.

---

## P05-I3 — BN-stat sidecar: full-domain Welford as a third peer consumer that does not pin I24

**One sentence.** Give native-projection BN its own narrow statistic consumer of the producer/activation stream so actual batch mean/var over `10×96×120×160` are **paid**, while residual I24 is **not** held for the whole domain.

**A (complete prior to copy).**

- Welford, *Note on a method for calculating corrected sums of squares and products*, Technometrics 1962; Knuth TAOCP vol. 2 popularization; Chan–Golub–LeVeque pairwise / parallel updates (1983) for tree reduction.
- Ioffe & Szegedy, *Batch Normalization*, 2015: **two-pass training BN** (reduce then affine), not inference FrozenBN.
- Streaming/online BN hardware units that keep `(count, mean, M2)` in a reduction engine **beside** the datapath (TPU-style separate reduction; cuDNN two-pass LN/BN).
- **Do not copy** FrozenBN, running-stat inference, Ghost-BN, or Sync-BN-over-devices as a substitute for actual batch stats — those change this student’s identity.

Copy Welford/Chan **and** the two-pass BN split (pass-1 reduction || conv, pass-2 affine+add). Copy a sidecar reduction engine, not a second full tensor.

**B (hole in THIS net).** Captured students use **actual** batch statistics over `10×96×120×160`, not frozen running stats. Local-window replay given **free** mean/var undercharges wait/storage. That undercharge is the missing third consumer: BN-stat’s live range is domain-sized, so I24 skip cannot retire, so producer backpressure stays 8088 even after source CSE.

**X (why not a reskin).** Not “online BN” as a numerical trick. The sidecar is a **peer of gate and PED** on the completion vector: datatype is `(Σ, n, M2)` (narrow), ready is per-tile, and I24 last-use may fire when `{gate, PED, BNstat}` have sampled **that tile**, while affine waits on **domain** ready. Affine+residual-add is a **later** consumer of (stats, conv-out, skip), not a reason to pin skip at full I24 for `10×96×120×160`. Splitting BN into {stat now} vs {affine later} is the mechanism; collapsing them is the current undercharge.

**Strongest controls.**

- Bit-bounded match of sidecar mean/var to full-domain actual batch stats of the student (not a local window). Publish max |Δμ|, |Δσ²| vs the captured BN.
- Storage for residual I24 during pass-1 must **not** scale as `10×96×120×160`.
- AEE rel ≤+0.005 vs 1.219801338 (stats rounding is the likely AEE killer).
- Complete-chain service with **paid** stat cycles, never with free mean/var.

**Kill-gate (number).** Sidecar extra storage ≤ `O(C)` (per-channel `(mean, M2, n)` for 96 channels, plus one tile buffer), **not** `O(10×96×120×160)`. Domain mean/var must match the student’s BN to the integer/RNE contract that keeps AEE ≤ 1.224801338. Complete-chain service ≥15% **with stat cycles included**. If the sidecar materializes a second full-domain tensor, **kill**. If running stats are frozen, **kill** (identity). If AEE is saved by feeding free μ/σ², **kill** (accounting).

**Two-sentence TCAS-II pitch.** Full-domain actual BN is a third consumer whose wait was invisible in local-window replay with free mean/var, and that wait pins the residual skip. A Welford sidecar pays the legal statistics on a narrow accumulator so I24 can turn over per tile, which is the first accounting that can honestly claim complete-chain service.

**Biggest objection.** Two-pass BN rereads activations; you may have moved the I24 reread onto the conv-out. If pass-2 reread ≥ the skip reread you saved, net service fails.

**Assumptions.**

- BN axes are such that `(mean, M2)` state is `O(C=96)` or `O(C×T)` with T≪120×160, not `O(H×W)`.
- Native conv-out can be regenerated or kept in a **tile** buffer for affine; a second domain-sized store is forbidden.
- Integer/RNE of Welford vs the student’s BN implementation is reconcilable inside the AEE budget (FP32 CUDA rounding diffs already exist; integer path must still 0-diff on gates/I24/PED).

**Predictions.**

- P1. Paying true domain stats will **add** wait vs the free-μ/σ² replay; the paper must still net ≥15% after that tax.
- P2. I24 skip live-range in time-cycles drops from domain-length to tile-length.
- P3. If Welford is run in the same I24 ALU as PED, you will recreate 8088 (sidecar must be a **peer**, not a muxed extra op on the PED ALU).

**Disconfirmers.**

- If BN is per-location (no reduction across the 10×96×120×160 cube), sidecar state is not small → kill.
- If affine legally needs the **producer I24** rather than conv-out + skip, split fails → kill.
- If Welford vs student BN exceeds AEE rel +0.005, do not “fix” by FrozenBN → kill I3.

---

## P05-I4 — Skip forwarding: residual ADD as late last-use of the still-live producer, not an SRAM reread

**One sentence.** Implement the native-projection residual add as an EX-stage forwarding mux from the operand that the dual-consumer fork already holds (producer register or PED U), so I24 is never written back solely to be reread.

**A (complete prior to copy).**

- Patterson/Hennessy 5-stage forwarding + hazard-detect unit (EX/EX, MEM/EX bypass). Copy the mux and the stall-when-not-forwardable rule, not the CPU pipeline.
- TVM / TensorRT Conv+BN+Add fusion **as a warning**, plus the **correct** residual-add placement of ResNet (add **after** BN of the projection). Copy fused-add microarchitecture only where it does not move add before BN.
- Clustered-VLIW operand bypass (TI C6x) for vector-width forwarding.
- GPU write-through operand collector (Lindholm Tesla 2008; BOW-style collector bypass patents): a value written once may be read by add without a register-file reread.

**B (hole in THIS net).** Residual I24 reread is a write-then-read of a value the producer fork already materialized. Naive Conv-BN-Add fusion **extends** I24 live range across full-domain BN and re-pins the skip — the opposite of service. The hole is a bypass that **skips over** the BN-stat interval without a domain-sized bypass buffer.

**X (why not a reskin).** Standard Conv-BN-Add fusion holds activations across BN and is exactly the live-range bug. Legal forwarding holds **one T10 vector or one spatial tile** and is coordinated with BN-stat completion: add issues when `{BNaffine_ready AND skip_in_forwarding_file}`. If the forwarding file grows to `10×96×120×160`, it **is** the reread SRAM under another name.

**Strongest controls.**

- Bypass-buffer capacity in vectors, not in “it is a mux.”
- Bit-true integer residual add vs model (0-diff).
- Same-port: forwarding mux must not steal the PED consumer’s only read port on the conflicted cycle.
- AEE rel ≤+0.005; **add stays after BN**.

**Kill-gate (number).** Extra storage for the forwarded addend **≤ 1 T10 vector** (or 1 spatial tile of the skip), complete-chain service ≥15%, AEE ≤ 1.224801338, 0-diff add. If bypass capacity ≥ current residual SRAM, **kill I4**. If add is moved before BN to make forwarding easy, **kill** (different network).

**Two-sentence TCAS-II pitch.** The skip is already in the producer’s live file when dual consumers sample it; writing I24 out and rereading it after BN is a ports tax that ate source-only gain. A tile-bounded forwarding mux that issues residual add only when BN affine is ready removes that tax without changing the add-after-BN law.

**Biggest objection.** Noncausal T10 plus domain BN means the skip needed at add time is **not** the skip that is live now; forwarding will silently use the wrong time index.

**Assumptions.**

- Skip identity at add is the same tensor the dual-consumer fork saw, delayed only by pipeline depth and BN wait, not by a different T10 tap.
- A tile-sized forwarding file can be kept coherent until affine of **that tile** (requires I3-style stats-complete-per-domain **broadcast**, with add per-tile after broadcast).
- PED U, not Z, is the addend (must be checked; if addend is a third tensor, I4 is empty).

**Predictions.**

- P1. Wrong-time-index forwarding will move AEE by ≫0.005; a time-tag on the forwarding file is mandatory.
- P2. Correct forwarding zeroes I24 reread cycles for residual add (distinct from PED’s own read).
- P3. Without a domain-stat broadcast barrier, per-tile add-before-domain-stats will desynchronize BN affine and fail 0-diff.

**Disconfirmers.**

- If the addend is a **post-projection** tensor rather than producer/PED U, there is nothing to forward from this fork → kill.
- If tile-bounded file cannot cover BN wait (stats need a second pass over the skip), I4 collapses to I3’s second store → kill.
- If forwarding mux + PED read conflict every cycle, same-port fails → kill.

---

## P05-I5 — Asymmetric dual-width decoupled access/execute: thin θg-gate vs wide PED/I24

**One sentence.** Split the two consumers onto unequal widths and independent queues so the gate path cannot impose I24 backpressure and the PED path cannot wait on gate-side bubbles, **without** binarizing ATLIF.

**A (complete prior to copy).**

- Smith, *Decoupled access/execute computer architectures*, ISCA 1982 / ACM TOCS 1984: two streams, two architectural queues, deadlock discussion. Copy A-queue / X-queue **and** the deadlock rule.
- Sutherland micropipelines (1989): bundled control rail vs data rail.
- Eyeriss v2 / SCNN independent sparse and dense meshes, as the **width-asymmetric** instance — copy the split, not the sparsity ideology.
- ARM/Itanium predicate datapath: a thin predicate file beside a wide GPR file.
- AXI-Stream **sideband** (TUSER/TSTRB) for the thin rail next to TDATA.

**B (hole in THIS net).** One producer feeds a spike/gate consumer **and** a continuous residual/PED consumer. If they share an I24 bus, the thin consumer pays wide-bus ready and the wide consumer pays thin-consumer stalls. Joint long-backpressure 8088 is the signature of a **common** ready. Continuous θg forbids pretending the gate is a 1-bit SNN spike (out of scope / identity kill).

**X (why not a reskin).** Not Loihi/TrueNorth spike-vs-membrane dual rail — that is a binary-ATLIF paper and is forbidden. The thin rail carries whatever the **integer gate law** actually reads (packed θg fragment and/or predicate), width measured from the 0-diff gate path, not chosen as 1. Independent skid buffers implement Smith queues; deadlock (both waiting on the other’s credit) is a first-class kill, copied from Smith §deadlock, not ignored.

**Strongest controls.**

- Bit-width of the gate consumer as **measured** from the integer gate law (publish W_gate and W_PED).
- Two stall traces, never one: `stall_gate`, `stall_PED`.
- 0-diff gates **and** 0-diff PED q24.
- Deadlock checker on the two queues (bounded credits).

**Kill-gate (number).** W_gate < W_PED/4 (otherwise it is not asymmetric — kill as reskin). Long-backpressure 8088 **splits**: each trace published, and complete-chain service ≥15%. AEE ≤ 1.224801338. If θg is quantized to 1-bit, **kill** (identity). If both rails remain I24-wide, **kill**. If the two queues deadlock on any captured window, **kill**.

**Two-sentence TCAS-II pitch.** Joint ready on an I24 bus makes a thin continuous-θg gate and a wide PED share stalls, which is why source-only arithmetic never moved 8088. A Smith-style dual queue with measured unequal widths decouples those stalls without converting ATLIF into a binary spike chip.

**Biggest objection.** The gate law actually consumes full I24/θg at vector width; the “thin” rail is a fiction.

**Assumptions.**

- Integer gate path 0-diff can be preserved with W_gate ≪ 24·V (V = SIMD vector).
- PED never needs a gate-side value in the same cycle in a way that re-joins the queues (if it does, I5 becomes I1’s join).
- Credits are bounded (Smith deadlock: unbounded queues are not a circuit).

**Predictions.**

- P1. `stall_PED` ≫ `stall_gate` on ordinary T10; decoupling helps only if they are **anti-correlated**. If they are correlated, decoupling does not move 8088.
- P2. Packing θg to W_gate bits without AEE hit is possible because gates already 0-diff as integers on captured windows.
- P3. A 1-bit spike version will pass service and fail identity/AEE relative.

**Disconfirmers.**

- Gate integer law reads the full I24 vector → W_gate ≮ W_PED/4 → kill.
- Stall traces correlated ≥0.9 → decoupling cannot provide ≥15% → kill.
- Any deadlock on valid825 streaming → kill.

---

## P05-I6 — Typed credit-matched issue: source rate = min(credit_gate, credit_PED, credit_BNstat)

**One sentence.** Slave T10/θg issue to an explicit **typed** credit minimum of the three consumers so arithmetic CSE/lifting cannot over-issue into the already-measured 8088-cycle backpressure wall.

**A (complete prior to copy).**

- Dally & Towles, *Principles and Practices of Interconnection Networks*: forward flits + reverse **credits**, credit round-trip, buffer occupancy = credits_in_flight.
- Lee & Parks SDF / Bilsen CSDF: static token counts compiled from the graph; issue only when tokens exist on all required edges.
- GPU producer–consumer counts / Cell MFC DMA credits.
- Token-bucket as an **issue throttle** (Turner/ATM), not as QoS marketing.

Copy credit accounting **with types**. Untyped FIFO depth is not this prior.

**B (hole in THIS net).** Lifting cut add/sub 260→159 and always-ready −22.83%, yet long backpressure stayed 8088. The source is faster than the join of consumers. Same-port/same-state/same-backpressure is the venue test; source-only graphs fail it. Part of −22.83% is generic round→sat fusion, so even the always-ready win is partly not-the-paper.

**X (why not a reskin).** Not “slow the source.” Credits are **typed**: a BN-stat credit is one reduction sample, a gate credit is one predicate/θg bundle, a PED credit is one q24 vector. Typed credits let the source **skip issuing a replica a consumer has declined** (gate may not need every intermediate RNE/sat of lifting; BN-stat may sample a subset of tiles if — and only if — that subset is still full-domain legal, which it is not; so BN-stat credit is 1-per-element of the domain). Untyped “stall when FIFO almost full” is a reskin of the current 8088.

**Strongest controls.**

- Credit counters visible in the same SIMD resource model that produced 8088.
- Complete-chain service; **never** add 758777-table to 6938-table.
- Ordinary T10 AEE control; do not use lifting AEE 1.232979368 as a passing prior.
- Issue-idle breakdown: `idle_no_gate_credit`, `idle_no_PED_credit`, `idle_no_BNstat_credit`, `idle_ALU`.

**Kill-gate (number).** Complete-chain net service ≥15% at same ports/state **after** credits. If the mechanism only throttles (8088 drops because the source does less, and wall-time service **does not** rise ≥15%), **kill I6**. AEE ≤ 1.224801338. BN-stat credits must equal the full domain cardinality (no “credit 1 per window”).

**Two-sentence TCAS-II pitch.** Always-ready slots improved 22.83% while long backpressure stayed 8088 because the source was allowed to issue into a join that had no credits. Typed credits compiled from {gate, PED, full-domain BN-stat} make issue equal legal consumption, which is the only source-side knob that can still move complete-chain service.

**Biggest objection.** Credits without extra consumer parallelism cannot raise service; you are decorating a stall. That objection is the kill-gate: service, not occupancy.

**Assumptions.**

- At least one consumer can run ahead when another is credited (otherwise min() is lazy-fork I1).
- BN-stat can absorb samples whenever the producer issues (sidecar ALU, not muxed on PED).
- Credit round-trip fits in the existing ready wire budget (1–3 bits).

**Predictions.**

- P1. Untyped credits reproduce 8088 occupancy shape; typed credits change the **mix** of idle reasons.
- P2. Service ≥15% appears only if BN-stat sidecar (narrow) can take samples on cycles PED is not using the port — I6 **alone** may fail and that is acceptable (do not silently stack I3).
- P3. Applying credits only to gate+PED and leaving BN free-μ/σ² will fake a pass.

**Disconfirmers.**

- If `min(c_g, c_p, c_bn)` = `c_p` always, typed credits collapse to “stall on PED FIFO” → reskin → kill.
- If service gain <15% after credits → kill (throttle without parallelism).
- If BN-stat credits < domain cardinality → accounting cheat → kill.

---

## P05-I7 — Alias the noncausal T10 delay line as the BN-domain buffer (one temporal store, two last-uses)

**One sentence.** Walk the already-required noncausal T10 lookahead buffer as the BN full-domain statistic window so BN does not allocate a second `10×96×120×160` live residual.

**A (complete prior to copy).**

- Oppenheim/Schafer overlap-save / noncausal FIR delay-line streaming: the same delay line is the legal operand window.
- HEVC deblock+SAO hardware **buffer sharing** (same reconstructed samples, two last-uses).
- Audio DSP RMS/envelope sharing a biquad delay line as the statistics window.
- Sweldens in-place delay-line addressing (even/odd taps), as an addressing prior only.
- Two-pass BN (Ioffe) whose **pass-1 inputs** are those delay-line taps.

Copy delay-line sharing **and** the second address generator. Do not copy FrozenBN.

**B (hole in THIS net).** Time at this PSN is **noncausal T10**. BN domain is `10×96×120×160`. If the leading 10 **is** the T10 window (see assumption/kill), then T10 lookahead and BN-domain store **double-count time**. Local-window replay with free mean/var hid the double allocation and undercharged wait. Residual I24 reread is partly a **temporal** reread of a tap that is still in the delay line.

**X (why not a reskin).** Not “reuse a buffer.” The circuit is a **second address generator** over the existing T10 delay line: AGU-S (source operands for gate/PED) and AGU-B (BN-stat samples), with **independent last-use** of a tap for source vs for BN-stat. Stats remain actual full-domain. If AGU-B needs a shadow copy, aliasing failed.

**Strongest controls.**

- Proof (shape dump) that BN’s reduction domain’s size-10 axis **is** the T10 noncausal window, not batch/N.
- SRAM bytes: BN gather extra beyond T10 delay line = 0.
- Mean/var match full-domain actual.
- AEE ≤ 1.224801338.

**Kill-gate (number).** **Immediate kill** if the `10` in `10×96×120×160` is **not** the T10 temporal axis (e.g. it is batch 10, or 96 is time, etc.). Extra SRAM for BN gather = **0** beyond the existing T10 delay line. Domain μ/σ² match the student. Complete-chain service ≥15%. AEE rel ≤+0.005.

**Two-sentence TCAS-II pitch.** Noncausal T10 already pays a 10-tap store; native-projection BN legally reduces over a 10-sized domain that may be that same store, yet the residual path currently allocates it twice and then rereads I24. Dual last-use address generators over one delay line pay full-domain stats without a second cube, which is a five-page circuits fact if and only if the axes match.

**Biggest objection.** The 10 in BN domain is batch, not time; the alias is a category error.

**Assumptions.**

- Axis identity: `10` ↔ T10 window. **This is the first experiment, not a belief.**
- Spatial 120×160 and channel 96 already sit in the same packing as T10 operands (or a transpose AGU is still cheaper than a second cube).
- Gate/PED last-use of a tap does not occur before BN-stat has sampled that tap, **or** a 1-bit “BN sampled” is kept per tap (O(10) bits, legal).

**Predictions.**

- P1. Axis check is a one-line shape print; if it fails, I7 is deleted the same day.
- P2. If axes match, BN-stat traffic is an extra read port on the T10 delay line, not extra bytes; the port, not the bytes, becomes the 8088 term.
- P3. A second read port on the delay line will show up as same-port conflict with PED; then I7 needs a **time-multiplexed** AGU, not more SRAM.

**Disconfirmers.**

- BN domain axis ≠ T10 → kill immediately.
- Extra read port cannot be time-multiplexed without losing ≥15% service → kill.
- Tap last-use for PED happens before BN-stat sample and per-tap bits are not kept → stats miss elements → AEE/identity fail → kill.

---

## P05-I8 — Single-write operand collector: T10/θg written once, three independent reads, last-use frees the slot

**One sentence.** Write T10/θg once into a banked operand collector; gate, PED, and BN-stat read independently; a last-use bit (Z vs U vs skip-addend) frees the slot so residual SRAM is no longer a producer-side copy.

**A (complete prior to copy).**

- Lindholm et al., *NVIDIA Tesla: A unified graphics and computing architecture*, IEEE Micro 2008: **operand collector** in front of a banked RF.
- Bank-conflict stall of GPU shared memory / RF (XOR bank hash, collector gathers until all operands present).
- CDC 6600 scoreboard last-use / result-bus valid.
- Architected last-use operand bits (IBM; NVIDIA `.reuse`).
- FPGA registered fanout of a single-producer multi-consumer net, as the small-N instance.

Copy collector + banked RF + last-use free **as one published unit**. Do not “add a register file.”

**B (hole in THIS net).** The producer currently appears to materialize a residual I24 that is later reread, while Z and PED U have no architectural last-use. Dual consumers plus BN-stat are **three reads after one write**. Long backpressure 8088 is consistent with write-back / bank-conflict stalls rather than T10 ALU depth (CSE already cut the ALU; 8088 did not move).

**X (why not a reskin).** Not a bigger SRAM. Collector **depth** is bounded by in-flight noncausal T10 vectors (O(10) or O(pipeline)), not by `96×120×160`. Bank-conflict between gate and PED **on the same cycle** is the circuit question that must be measured on captured windows; that measurement is the contribution, not the name “collector.” Recolor of Z vs U (the I2 story) happens **inside** this file via last-use; I8 is the microarchitecture **only if chosen instead of I2**, not stacked.

**Strongest controls.**

- Collector depth (entries) and bank count published.
- Conflict counters: `bank_conflict(gate, PED)`, `bank_conflict(*, BNstat)`, `writeback_stall`.
- Complete-chain service ≥15% at **same** SRAM ports as the baseline residual file (collector **replaces** producer-side I24 copy, does not add to it).
- 0-diff integer reads of Z/U/skip.

**Kill-gate (number).** Collector extra entries ≤ **2× T10 noncausal taps** (≤20 if T=10), not domain-sized. Same-port: residual producer-side SRAM bytes **decrease**. Complete-chain service ≥15%. AEE ≤ 1.224801338. If gate and PED conflict on **>50%** of issue cycles and the collector is single-banked, **kill** (need two-read-port collector; if that breaks same-port, kill). If collector depth grows to `10×96×120×160`, **kill**.

**Two-sentence TCAS-II pitch.** One producer and three consumers is an operand-collector topology, not a residual-SRAM topology; the I24 copy is a leftover that backpressures T10. A last-use-retired collector bounded by the noncausal window removes that copy and makes 8088 a measurable bank-conflict number instead of an anonymous stall.

**Biggest objection.** Three read ports is an extra SRAM, i.e. not same-port. If the collector is not cheaper than the I24 file it replaces, there is no paper.

**Assumptions.**

- In-flight producer values needed by {gate, PED, BNstat, later add} fit in O(T10) collector entries **if** BN-stat is a running reduction (does not need random later reread of all taps). This assumption **fails** if I3/I7 are false and BN needs a second full pass over stored I24 — then I8 is illegal alone.
- Bank hash can separate gate (thin) and PED (wide) most cycles (ties to I5 only as a **measurement**, not a stacked module).
- Last-use of skip-addend can wait on BN affine **inside** the collector only for **tile** lifetime.

**Predictions.**

- P1. Write-back stalls, not T10 add/sub count, track 8088; collector removes write-backs of unused I24 copies.
- P2. Dual-read same-cycle of gate+PED is rare if gate is thin; if it is frequent, two-port collector is mandatory.
- P3. If BN-stat needs a second full pass, collector depth explodes and I8 dies in favor of I3/I7 or nothing.

**Disconfirmers.**

- BN-stat second pass over domain-sized I24 → collector depth explodes → kill I8.
- `bank_conflict(gate, PED) > 50%` and two-port RF exceeds baseline ports → kill.
- Any 0-diff miss on Z vs U vs skip reads → kill.

---

## Cross-idea discipline (not a menu)

These eight are **alternatives**. TCAS-II gets one mechanism.

| ID | Island (one mechanism) | First experiment (cheap, CPU + captured windows) | Likely death |
|---|---|---|---|
| I1 | Hybrid elastic completion vector | Split 8088 into wait_gate / wait_PED / wait_BNstat | Waits are simultaneous → lazy fork, empty paper |
| I2 | Last-use recolor Z vs U | Live-interval dump including BN-stat and residual add | Overlap → illegal overlay |
| I3 | Welford BN sidecar as peer consumer | Pay domain μ/σ²; measure extra cycles vs free-μ/σ² replay | Second full tensor, or AEE from stats rounding |
| I4 | Tile-bounded skip forwarding | Bypass capacity; add-after-BN 0-diff | Bypass grows to domain; wrong T10 tap |
| I5 | Dual-width Smith queues | Measure W_gate vs W_PED; stall correlation | Gate is actually wide; stalls correlated |
| I6 | Typed credits | Issue-idle breakdown; service not occupancy | Throttle without parallelism |
| I7 | T10 delay line = BN domain | **Print BN axes vs T10 axis** | `10` is not time |
| I8 | Operand collector, last-use free | Conflict %; depth vs domain | BN second pass explodes depth |

**Forbidden stacks.** Do not write “I1+I3+I8.” If a kill-gate of the chosen island **logically requires** a fact from another island (e.g. I8 requires BN-stat not to need a second I24 pass), that fact must be **true of the net today** or the island is dead — it is not a license to add the other island.

**Lifting is not the title.** Ordinary AEE 1.219801338 is the strong control. Lifting 1.232979368 already fails rel +0.005. A dual-consumer paper may keep a lifting **source graph** only if AEE is repaired inside the same kill-gates; the novelty claim is the dual-consumer completion/last-use/BN-domain circuit, not 260→159 add/sub.

**Minimum causal figure for whichever island survives.** One producer, two (three) consumers, one I24 skip, last-use marks on Z and U, BN-stat domain vs tile live ranges, and the 8088 bar that must move. If the figure needs more than those boxes, the idea has become a zoo.
