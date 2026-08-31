# M1158D3 author CPU fast-kill receipt

The static D3-only branch survives locally but fails the preregistered all-four gate. D0-D2 were fixed to A1-OSG before reading the workload; no sample, sequence, density, miss-rate or runtime oracle was used.

The analyzer independently replayed the frozen D3 bitpack and counted 96,760,057 contributors and 17,288,869 modulo-8 bank-conflict groups. This is 1.052797x the sealed M712 conflict-free optimistic group count. Both width axes charge 15 cycles per actual group, source ingress, bitmap probes, 130 exact cold weight refills for 13 identities in 16 entries, dense commit, and owner plus terminal control.

| Width | D3 local ratio | All-four static mixed ratio |
|---|---:|---:|
| 128 bit | 1.354761x | 1.153212x |
| 96 bit | 1.351047x | 1.151846x |

The minimum D3 local ratio passes 1.20, but the minimum all-four ratio does not. Status is `NO_GO_RTL__ALL_FOUR_1P20_GATE_FAILED`. No RTL, VCS, DC or EDA is authorized. A fresh different-author result hammer is required only to validate the CPU decision and seals, not to reopen RTL.
