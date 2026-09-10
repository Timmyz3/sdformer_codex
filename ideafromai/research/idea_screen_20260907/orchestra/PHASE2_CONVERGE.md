# Orchestra Phase 2 — Converge

**Goal (skill text):** narrow to 3–5 strongest ideas.

Kill criteria (verbatim meaning): cannot state in two sentences → drop; nobody suffers → drop; a simpler approach already works → simplify or drop; no beneficiary → drop; clearly infeasible with available resources → park.

The 32 cards and R01–R20 stay on disk. This table is a filter, not a deletion.

Weights and K-Dense scores are **not** used here. Phase 2 is the Orchestra qualitative filters. Numeric matrix is a later decision *aid* in the other skill.

---

## Two-sentence pitches (F10) for every R that could survive

Template: (1) domain struggles with X because consequence. (2) we approach by mechanism because reason.

| ID | F10 two sentences | Pass? |
|---|---|---|
| R01/I001 | Event-OF spike Transformers score Q-K with a temporal XOR that FireFly-T/QKFormer/SDformerFlow do not implement, so those ALUs do not match. We build one 32-lane triple-popcount leaf with a t0 `K_peer` shadow because that is the formula the net already uses. | Yes |
| R02/I002 | Recomputing a silent token wastes the only cheap energy in the leaf, but a 15×15 Shiftmax row is almost never fully clean. We clock-gate the leaf per token when K is zero both T or Q/K equal t0, not as a row memo. | Yes (merge into R01) |
| R03/I003 | It is unknown whether OF actually needs the XOR term, which matters because a simpler linear Q-K would shrink the circuit. We swap **one** stage-2 block to SDSA or QKFormer and read subset AEE. | Yes |
| R04/I004 | Conv/FC dominate the historical envelope, so a leaf brief may have no system PPA. We run mixed T=2/T=10 FC join against LoAS FTP on the same capture because that split is the only local difference. | Yes |
| R05/I005 | Weight rows are reread by many consumers. We broadcast B8 on frozen ep34 traces because the CPU premodel already exists. | Yes as comparator; weak problem-first |
| R06/I006 | Bottleneck conv repeats products. We copy Prosperity. | F1 weak (pain already owned by Prosperity); F7 says copy is the simpler approach that *is* the prior. **Kill from winner pool.** |
| R07/I007 | BN is late relative to the membrane. We keep ATLIF inside the island and commit θ at the boundary because A8 RTL already exists. | Borderline; keep as fifth |
| R08/I008 | Decoder ConvTranspose is a large share. We need a complete Table-A before an island. | Feasibility fail (no Table-A). **Park.** |
| R09/I011 | Sparsity training was killed by a freeze that no longer exists. We would retrain PAFT on ep34. | Feasibility: A800 + prior 1.47. **Park until I003/M0 done.** |
| R10 | Dual-port might help C1. | F7: stall≈0. **Drop.** |
| R11/I009 | T×15×15 layout is awkward on 1RW. Byte permute from FireFly-T. | Feasibility unknown on foundry macros. **Park.** |
| R12/I010 | Co-silence may over-attend. Lateral-inhibition training. | No circuit face for a 5-page hardware brief. **Park.** |
| R13 | Fold BN. | Same as R09 risk. **Park.** |
| R14/I012 | Overlap popcount is almost always empty. Skip terms independently. | Yes as a measurement hanging on I001, not a fifth title |
| R15/I013 | Temporal skip on encoder maps is the CICC analogy. | Distinct from I001; not this brief’s object. **Park as ablation.** |
| R16 | Projection island. | Infeasible until T0. **Park.** |
| R17 | M2 grain experiment. | This is a measurement, not an idea. Fold into I001/I002. |
| R18 | Change window/T. | Burns A800 before T0. **Park.** |
| R19 | SLI bypass. | Extra path; not 5-page. **Park.** |
| R20 | +1.58% parent promotion. | Already measured, too small. **Drop.** |

---

## Filter matrix (shortlist candidates only)

| ID | F10 | F1 genuine problem | F7 simplicity | F8 beneficiary | Feasibility (1 A800, existing RTL/traces, 2-week pilot) | Shortlist |
|---|---|---|---|---|---|---|
| I001 | pass | pass | pass (SV exists) | circuits author + reviewer who wants a native ALU | pass: T0/M1 first | **Yes** |
| I002 | pass | pass (silent tokens) | pass if enable-only | same as I001 | window skip infeasible; token skip feasible | **Yes, merge into I001** |
| I003 | pass | pass (do we need XOR?) | kernel swap is not simple; *probe* is simple | algo+circuit | 10-frame overlay | **Yes** |
| I004 | pass | pass if T0 kills leaf | mixed-T is the one extra idea | system-PPA reader | A6 RTL + FTP baseline | **Yes** |
| I005 | pass | weak | simple comparator | operator (bytes) | CPU done; RTL later | **Yes, comparator** |
| I007 | pass | medium | existing RTL | neuron-island reader | A8 functional | **Yes, fifth / contingent** |
| I012 | pass | pass as measurement | simpler than new ALU | I001 authors | same census | hang on I001, not a 6th shortlist slot |
| I006 | pass | fail F1 | copy *is* the prior | Prosperity authors already | done | No |
| I008 | pass | share is real | missing table | system table | fail | Park |
| I009–I011, I013, R10, R16–R20 | — | — | — | — | park/drop as above | No |

**Shortlist (5, skill maximum):**

1. **I001** (includes I002 token enable and I012 term-skip as measurements on the same leaf)
2. **I003** algorithm-face probe
3. **I004** contingent circuit face if T0 < ~2%
4. **I005** performance comparator, not a title default
5. **I007** neuron-boundary island, fifth

I006/I008/I009–I011/I013 remain in the K-Dense register as deferred/candidate so they can recombine.

---

## Skeptic’s first objection (F10 calibration) per shortlist item

| ID | First objection | Response stored for Phase 3 |
|---|---|---|
| I001 | Leaf, no FPS | T0 first; if <2% switch face to I004 |
| I003 | AEE returns to 1.58 | 10-frame, one block, stop if direction is up |
| I004 | “LoAS with theta” | Write the mixed-T packing difference; FTP is baseline |
| I005 | Unoriginal broadcast | Do not title it; keep the number |
| I007 | Mixed-T ASIC already exists | Do not claim first; measure real BN recovery |
