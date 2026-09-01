# M1575 ep34 live93 S2 CCBS16 activity-relative fast-kill

Status: **PASS_ACTIVITY_RELATIVE_FASTKILL__CONDITIONAL_RETAIN_FOR_PAIRED_AEE_AND_ADDRESS_TIMED_REPLAY__NO_RTL_OR_PERFORMANCE**.

All 30 decoder samples from the sealed ep34 live93 capture were consumed (three DSEC sequences, ten samples each, four ConvTranspose layers). The debt owner is one destination/output-tile and the reference is that owner's observed active bound mass. This repairs the dense global-capacity denominator that made the prior 99.2% drop number scientifically unusable.

## Global 16x16 screen

| epsilon | keep | drop | drop fraction | weight-byte eligibility | aggregate bound debt |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 4,662,192 | 1,328,208 | 22.172% | 21.999% | 0.000% |
| 0.01 | 4,537,088 | 1,453,312 | 24.261% | 23.971% | 0.868% |
| 0.02 | 4,482,903 | 1,507,497 | 25.165% | 24.889% | 1.853% |
| 0.05 | 4,362,151 | 1,628,249 | 27.181% | 26.925% | 4.861% |
| 0.10 | 4,183,204 | 1,807,196 | 30.168% | 29.930% | 9.857% |

Static uint16 directory: **6,240 B**, or **0.0874%** of the hypothetical packed INT8 decoder weights; **15.89x** smaller than the old per-source G11 metadata.

## Per-sequence/layer at epsilon=0.10

| sequence | layer | keep | drop | drop fraction | weight-byte eligibility | debt / active bound |
|---|---:|---:|---:|---:|---:|---:|
| interlaken_01_a | D0 | 1,023,840 | 450,720 | 30.566% | 30.566% | 9.933% |
| interlaken_01_a | D1 | 242,321 | 133,999 | 35.608% | 35.208% | 9.793% |
| interlaken_01_a | D2 | 68,254 | 27,746 | 28.902% | 27.012% | 9.774% |
| interlaken_01_a | D3 | 38,390 | 11,530 | 23.097% | 20.368% | 9.140% |
| thun_01_b | D0 | 1,040,263 | 434,297 | 29.453% | 29.453% | 9.936% |
| thun_01_b | D1 | 246,098 | 130,222 | 34.604% | 34.166% | 9.801% |
| thun_01_b | D2 | 68,895 | 27,105 | 28.234% | 26.208% | 9.766% |
| thun_01_b | D3 | 38,943 | 10,977 | 21.989% | 19.184% | 9.145% |
| zurich_city_12_a | D0 | 1,056,169 | 418,391 | 28.374% | 28.374% | 9.939% |
| zurich_city_12_a | D1 | 249,858 | 126,462 | 33.605% | 33.160% | 9.794% |
| zurich_city_12_a | D2 | 70,304 | 25,696 | 26.767% | 24.778% | 9.787% |
| zurich_city_12_a | D3 | 39,869 | 10,051 | 20.134% | 16.825% | 9.160% |

## Verdict and red lines

The candidate is **retained** for only the next two measurements: paired same-checkpoint AEE and address-timed bank/burst suppression with metadata charged. Exact sampled local errors stayed inside the certified debt; the primary proxy median/p90 is 1.839%/4.054% of activity-bound mass. This is not optical-flow accuracy.

`paired_aee=false`, `cycles=false`, `traffic=false`, `energy=false`, `speedup=false`, and `rtl=false`. The weight-byte number is eligibility, not measured DRAM/SRAM traffic and not a system acceleration.
