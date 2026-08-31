# M191 H67 FC2 multi-context bank co-issue DSE

Status: `PASS_EXACT_PAYLOAD_REPLAY_KERNEL_DSE_RTL_OPEN`.

M191 rechecked all 120 frozen H67 ep35 FC2 payloads (143,894,510 source
events, 6,523,707 bounded windows) and batches adjacent windows from the same
FC2 call without reordering.  In a batch of `C` independent accumulation
contexts, each of the eight weight banks serves at most one source per cycle
and emits a context tag.  The exact batch service time is therefore
`max_b(sum_c(bank_population[c,b]))` per output block.

| contexts | exact replay cycles | speed vs C1/K8 | bank utilization | serial-K1 replay speed |
|---:|---:|---:|---:|---:|
| 1 | 79,397,844 | 1.0000x | 65.00% | 5.2004x |
| 2 | 71,233,088 | 1.1146x | 72.46% | 5.7965x |
| 4 | 67,218,210 | 1.1812x | 76.78% | 6.1427x |
| 8 | 64,622,733 | 1.2286x | 79.87% | 6.3894x |
| 16 | 62,671,956 | 1.2669x | 82.35% | 6.5883x |

C2 is the first RTL screen because it is the minimum point that exposes the
bank-to-context tag and dual partial-sum machinery.  C4 remains the stronger
performance DSE point.  None of these numbers include context storage/ports,
descriptor fill, SRAM response latency, RTL timing, BN2/residual, complete-FC2
or system cycles.  The `optimistic_wall` fields merely add the measured M187
overhead unchanged and are not admitted speedups.

The analyzer pins every input SHA, refuses overwrite, and cross-checks C1
against M187's exact K8 replay total (`79,397,844`).
