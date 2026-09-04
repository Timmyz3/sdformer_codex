# H67 paper identity contract

Status: `PASS`; the unique paper checkpoint is full-resolution H67 `ep35`.

- H67 is a retrained all12 H60 Motion-XOR Shiftmax gated-K operator, not the public SDformerFlow SDSA.
- Symbols: `T_snn=10`, `T_w=2`, `H_w=W_w=15`, `N_tok=450`, `N_pair=225`.
- Deployment: Q7 score, Q8 LUT Shiftmax, Q1.7 gate, K reused as value.
- Valid825: AEE `1.329678`, AAE-2D `5.900353`, AE-3D `5.650878`, spikes `82.1107G`.
- Hardware evidence is read-only and component-level; it includes checkpoint-bound real-weight projection but not full-network RTL exactness.
