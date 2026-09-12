# Post-check candidates and fixed experiment

2026-09-13. These are proposals, not established contributions. The original five independent ideas are preserved in `INDEPENDENT_IDEAS.md`.

## Read scope and corrections

Read both supplied GPT Pro documents in full, F1/F2 and F3–F7 paper designs, Grok `CODEX_FOLLOWTHROUGH.md`, coverage README and remaining batches, then the B4 representation/packed/decoder results and actual source code. Prior precision gates in older idea pages are obsolete: the current admissibility rule is better than NB0, without a +0.005 extra gate. The newer source students and their 825 results are distinct from the old R24 ordinary/lifting_raw quantizer parents used here.

Initial I01 overlaps F2/F4/F6; I02 overlaps Phi/LUT-DLA and the already executed 5+5 decoder; I03 overlaps delta/LoAS style controls; I05 overlaps existing temporal-default/bitplane work. Do not relabel those as new interfaces. I04's hoped-for common projection rowspace is not assumed: projection 3×3 gate weights and PED 1×1 U are different operators. Its useful surviving question is narrower: can a known encoded PED representation reach the actual U consumer without materializing its continuous decoded input?

## C1 — Consume Q8 residual and prediction inside PED's original U accumulator

- **B:** For the existing deployed quantizer `xhat=sat24(Dg+c+s_t q)`, accumulate `s_t Uq + D(Ug) + c_t sum_h U_h + U e_sat` before the one original U RNE/sat. Keep V/RNE/bias exactly as deployed. `e_sat=sat24(raw)-raw` is explicit unless a universal range proof removes it. This is the B4 residual-code/step consumer interface; previous runs reconstructed I24 before U or timed a decoder-only output.
- **A borrowed concretely:** Jacob et al.'s affine integer arithmetic and constant zero-point terms; Phi's pattern-plus-correction separation; ordinary tensor-product distributivity; existing common source-resident MAC, CSE, PoT, and P1 residency. Full D requires a newly charged `Ug`, not credit for the unrelated projection convolution. LUT-DLA supplies lookup-cost controls, not a free table.
- **X to test:** whether the *same producer's already-needed gate* reduces the paid continuous representation/consumer cost beyond ordinary affine with identical factor extraction, constant caching, packing, saturation correction, and read permissions. Affine extraction itself is common baseline.
- **Strongest controls:** (1) same-function ready reconstructed I24 → original resident U, with reconstruction granted free at this boundary; (2) expanded24 residual-code → factored U; (3) signed8 residual-code → the same factored U; (4) ordinary fixed/center/affine with identical privileges; (5) input-side cached conditional-add Dg and the already measured best decoder, if full-D timing is later pursued. All share 96×8×48 RF, SR64/SW64/CR256, 128 KiB state/coefficient, and one issue.
- **Fixed one experiment:** Existing train-only old ordinary/lifting_raw quantizers; original corner and interior 4×4 anchor windows. No refit, no rank/width/step search. Twenty numerical cases (2 parents×2 windows×5 representations) verify pre-U accumulator and original U/V/RNE/bias. Exact universal range/48-bit checks cover all gate/code inputs. For the three no-gate modes, execute the three U-entry controls above in the existing finite payload machine over all 16 P1 positions; retain U output boundary and charge real fills/reads/gathers/scales/base addition/stores. This is 36 fixed local U executions, not full-chain timing. The gate prediction extension gets exact arithmetic and explicit necessary-operation counts, not fabricated cycles.
- **Stop this layout on a negative result:** if factored Q8 loses to expanded reconstructed I24 at the ready U boundary, stop this fixed input layout as a speed claim. If full-D extra gate projection and 19-bit coefficient processing erase its benefit over ordinary affine, stop latent-side full-D decode placement. Do not stop Q8, prediction, or low-bit hardware families. Producer encoding, native projection, dynamic BN, and final join remain outside timing, so no end-to-end speed or inherited new-student AEE is authorized by this probe.

## C2 — Decode packed U coefficients in the shared coefficient response

- **B:** Place one signed8-to-signed16 expansion in the existing 32 B coefficient response/staging path, delivering H8 weights to the original 16×24 MAC without the RF84 decode→staging round trip of the tested W8 layout. Keep row-scale continuation and original U/V RNE boundaries.
- **A borrowed concretely:** MiLo's zero-bit-waste packing, explicit dequantization, and pipeline accounting; ordinary packed-weight decoders. This is not a replication of its GPU kernel.
- **X:** only a measured reduction in shared RF/issue contention from one response-local representation, relative to the same-function expanded16 and tested RF-decoded packed8. No claim that packing or dequantization is new.
- **Strongest control:** ordinary and lifting receive the identical decoder path, existing coefficient cache, row-scale optimization and total staging/port budget; compare old packed helper and expanded16 using the exact same W8 parameters.
- **Fixed once:** use ordinary/interior ready and the existing fixed pressure trace, W8 only; one response register allocation within the existing 64 B staging budget. Price sign extension/selection and response occupancy, do not invent extra banks or a second issue. No AEE rerun is needed only if exact same-function equality is proved.
- **Negative stop:** stop the response-local decoder placement if row-scale work still dominates or it increases response stalls. Earlier W8 negative results stop RF materialization, not this placement; this placement is still unexecuted in this subtask.

## C3 — Fixed rank compensation learned around deployment rounding

- **B:** one fixed low-bit PED U approximation plus a fixed rank correction, with both contributions merged before the original U RNE; choose the correction around the real rounded consumer, not only real-matrix error.
- **A borrowed concretely:** MiLo's alternating quantized-weight/residual low-rank compensation and explicit compensator precision; ordinary low-rank/quantization recovery. No mixture-of-experts mechanism is asserted for this dense optical-flow layer.
- **X:** deployment-boundary-aware compensation reduces actual charged operands versus same-budget narrow dense and W8/low-rank controls while preserving the current NB0 quality criterion. Low-rank error correction by itself is already occupied.
- **Strongest control:** same-bit/same-rank weight-only MiLo-style alternating approximation, ordinary affine activation quantization, existing three matched source students and unchanged raw head; equal recovery examples/steps/seed when training is eventually run.
- **Fixed once:** if C2 supplies a credible low-bit execution endpoint, choose one preregistered bit/rank pair and one training budget; evaluate the resulting function on its own validation set. No training in this subtask.
- **Negative stop:** stop that bit/rank and merge boundary if correction's real products/metadata exceed the saved cost or its own AEE is not better than NB0. This is lower priority than closing C1/C2 because it needs a new student and quality evaluation.

Decision owner: root/user. This subtask advances C1 first and leaves C2 as the next directly codeable alternative. C3 is a preserved minority direction, not an automatic winner. Source verification is recorded in `PRIMARY_SOURCES.md`.
