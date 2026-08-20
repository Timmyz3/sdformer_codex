# DATE appendix loading table

Status: `PASS_FROM_SPIKE_PROFILE`. Counts come from rank-1 `spike_profile.json` load audits.

| ID | Role | Epoch | ATLIF | Shiftmax | Overlay keys | Missing | Unexpected | Overlay missing | Overlay unexpected | Remap | Checkpoint SHA | Config SHA | RTL claim scope |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| NB0 | baseline | 29 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | v1 | `7e8d524e0784` | `55aeb36c71dd` | none |
| H81 | no-motion control | 29 | 105 | 12 | 210 | 0 | 0 | 0 | 0 | v1 | `8825c933e491` | `c11600fe23a7` | none; recipe-level Motion control only |
| H67 | DATE mainline | 35 | 105 | 12 | 210 | 0 | 0 | 0 | 0 | v1 | `4f33e086070b` | `86db3960c7d1` | checkpoint-bound component RTL exact for score/SCS/Shiftmax, ATLIF temporal matrix, and real-weight projection |
| Local5_rank1 | accuracy extension rank-1 | 44 | 105 | 12 | 210 | 0 | 0 | 0 | 0 | v1 | `19820bec07cc` | `c5d7be623fd1` | none on ep44; existing Local5 RTL remains bound to ep29 |
| Local5_rtl_anchor | Local5 hardware anchor only | 29 | 105 | 12 | 210 | 0 | 0 | 0 | 0 | v1 | `6e0e92a56229` | `cf8c3da8fd8a` | ep29 component RTL only; not the algorithm rank-1 |

H67 identity-contract SHA checks still bind. H81 has a complete overlay load but no paper RTL.
Local5 ep44 same-checkpoint component RTL was rebound 2026-08-15
(`local5_ep44_hardware_rebind_20260815_final_audit.json`, status `PASS`).
That receipt is component-level and was explicitly not allowed to replace the frozen H67 main table.
