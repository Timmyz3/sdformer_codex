# M732 fresh hammer: M722R2 decoder LB-FUSE CPU fast-kill

## Verdict

**ADMIT_KILL_NO_RTL, 99/100, P0/P1/P2 = 0/0/1.**

The negative result is sound and stronger than a threshold-only rejection. A fair A1-OSG already keeps the three-row psum lifetime on chip under the same 240 KiB budget, so LB-FUSE has no spill traffic left to remove. The candidate then loses destination packing and becomes slower on every selected sequence.

No new CPU model, GPU, RTL, VCS, EDA, or remote job was run. This hammer only reverified sealed inputs and independently recomputed integer ledger facts from the existing `rows.jsonl`.

## Independently recomputed result

| Quantity | A1-OSG | LB-FUSE | Candidate / baseline consequence |
|---|---:|---:|---:|
| Headline cycles, D0+D2+D3 | 21,590,945,350 | 23,377,337,337 | A1/LB = 0.923584; LB is 8.273802% slower |
| K8 issue groups | 827,946,728 | 1,170,190,821 | LB/A1 = 1.413365x |
| On-chip psum RMW bytes | 476,897,315,328 | 549,335,071,872 | LB/A1 = 1.151894x; +72,437,756,544 B |
| Dense commit bytes | 11,612,160,000 | 11,612,160,000 | identical |
| Off-chip psum spill bytes | 0 | 0 | traffic gate is impossible |

The selected population contains 120 sealed records and 1,200 timestep planes. Each decoder module contributes 300 planes. The headline uses the 900 exact-binary D0/D2/D3 planes; scaled-binary D1 remains diagnostic.

Per-sequence A1/LB is 0.926111 on `interlaken_01_a`, 0.923851 on `thun_01_b`, and 0.920822 on `zurich_city_12_a`. Thus the candidate loses on all three selected sequences, not only in the aggregate.

## Fairness and first-principles audit

The A1 baseline uses the same three-output-row lifetime, 96 product lanes, K8 groups, 240 KiB budget, common dense commit, and serialized single-1RW psum service. D3 remains Acc24 by using legal width stripes `[0,256)` and `[256,320)` with one repeated input column. Its storage is 243,200 B, leaving 2,560 B inside 245,760 B, and its off-chip psum spill is zero.

The model charges 12 serialized psum port operations inside each 15-cycle 96-lane group for both paths. All 1,200 rows satisfy the port schedule and all common commit charges match. LB-FUSE cannot combine contributors from up to four source positions at a shared destination, so it creates 41.34% more groups. Reintroducing destination-keyed packing would collapse the mechanism back toward A1-OSG/PIDP.

The arithmetic miter has zero Acc24 mismatches. On the complete selected local-INT8 trace, D3's order-independent absolute-prefix bound is 7,288, so full-width Acc16 is trace-safe. D0 final values also fit Acc16, but its prefix bound is 62,696, so D0 is not safe for arbitrary accumulation order in this trace. These are local-INT8 scheduling/width observations, not checkpoint accuracy admission or all-input proofs.

## P2 wording constraint

Do not call the full comparison “matched Acc24.” The fair A1 baseline is Acc24 throughout, but the selected D3 LB candidate receives trace-bounded Acc16 to fit full width. This asymmetry favors LB-FUSE and therefore makes the kill conservative; it still must be disclosed.

## Paper boundary

The paper may cite this only as a model-labeled negative ablation: on the complete selected S3x10 local-INT8 population, source-order LB-FUSE was 8.27% slower because group count rose 1.413x and on-chip psum RMW rose 1.152x, while both designs had zero off-chip psum spill. The three sequence ratios may support robustness of this negative result.

It must not be presented as a headline/system speedup, full decoder cycle simulation, accuracy result, checkpoint numeric admission, RTL/VCS/EDA/PPA/energy evidence, or a claim that line buffers, polyphase mapping, or Acc16 are novel. No generalization beyond the three selected sequences is admitted.
