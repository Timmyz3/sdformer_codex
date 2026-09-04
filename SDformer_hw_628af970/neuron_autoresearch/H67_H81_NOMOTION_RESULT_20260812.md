# H67 Motion versus H81 no-motion control

Status: `PASS_PROTOCOL_AND_IDENTITY`; H81 convergence: `operationally_plateaued_or_overfit`.

| route | epoch | AEE | AAE-2D | AE-3D | Fl(%) | spikes(G) | energy proxy(uJ) |
|---|---:|---:|---:|---:|---:|---:|---:|
| H67 Motion | 35 | 1.329678 | 5.900353 | 5.650878 | 6.4279 | 82.1107 | 72508.06 |
| H81 no-motion | 29 | 1.330597 | 5.969235 | 5.672632 | 6.4310 | 80.9024 | 71571.17 |

H67 AEE change versus H81: `-0.069%`; negative is better.

This is a same-parent/seed/recipe control, not a bit-exact step-paired training trajectory. H81 has no inherited hardware provenance.
