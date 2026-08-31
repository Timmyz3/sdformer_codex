# M520 H67 paper metric registry v1

Status: `REGISTRY_READY__SYSTEM_TABLE_BLOCKED`

This is a provenance-checked evidence inventory, not a system performance table. No system speedup is generated.

| Row | Numeric cells | Blocked/null cells |
|---|---:|---:|
| Fixed dense | 3 | 12 |
| Exact bit-sparse | 4 | 11 |
| Prosperity official external iso-workload | 8 | 7 |
| Phi-like | 0 | 15 |
| Ours C1 | 1 | 14 |
| Ours C2 | 2 | 13 |
| Ours C3 | 1 | 14 |
| Ours A1 | 1 | 14 |

## Blocking gates

- Exact decoder ConvTranspose2d coordinates and executable cycles are missing; M510 proves the old 620M-class envelope omitted four decoder operators.
- Phi-like has no admitted same-workload adapter, cycle result, memory result, PPA result, or accuracy result.
- A common executable full-network schedule with compute, preprocessing, stalls, and completion semantics is missing.
- SRAM/DRAM timing and energy are not closed on one common memory hierarchy, and logic-only energy cannot be promoted to inference energy.
- Target SRAM macro area and macro energy are missing; zero instantiated macros in logic-only DC is not zero macro area.
- Multi-sequence DSEC coverage is missing; the current ten windows come from one Zurich sequence and are not ten sequences.
- C1, C2, C3, and A1 remain component or included-scope evidence and cannot be combined additively or multiplicatively.

## Hard boundary

The official Prosperity row is external support-tile iso-workload evidence only; its local ratio is intentionally absent. C1/C2/C3/A1 values retain component or included-scope labels. Null means blocked, never zero.
