# Literature check + reopen (K-Dense step 8)

Search date: 2026-09-07. Reviewer: P02 (queries) / P03 (challenge pass).  
Limits: English; 2024-01 onward for *new* candidates; classics only as named controls; arXiv abs/HTML + IEEE names in `../SOURCE_LEDGER.md` + this repo. No Scopus/WoS. Full texts not all read. **Absence from this search ≠ novelty.** Supportive literature ≠ validated mechanism.

Evidence packet given to the post-check round: `../SOURCE_LEDGER.md`, `../T1_T4_ep35.json` verdict, grill-me contract.

---

## Per-idea checks

| idea | queries | sources screened (stable IDs) | support (located) | challenges (located) | status |
|---|---|---|---|---|---|
| I001 | FireFly-T AND-PopCount; alpha-XNOR SSA; Bishop AAC; Motion-XOR optical flow ASIC | arXiv:2505.12771; CVPR 2025 α-XNOR; arXiv:2505.12281 Bishop; `h67_motionxor_score_q7.sv` | No published **temporal-peer K XOR** popcount ASIC located *in this search* | AND-PopCount and co-silence exist separately; historical attn ~0.59% | challenge-located |
| I002 | DeltaCNN skip; CBinfer; event OF DLSS CICC 2026 | DeltaCNN literature; Zhang CICC 2026 (local note) | Token skip is a known idea | Those priors skip pixels/feature maps, not Motion-XOR tokens; ep35 window-clean 9.9% | challenge-located |
| I003 | QKFormer; SDformerFlow SDSA; Spike-driven Transformer | arXiv:2403.16552; arXiv:2407.15801; SDT V2 ICLR 2024 | Public linear / SDSA kernels exist to swap | They are not OF-native; AEE may regress | support-located |
| I004 | LoAS FTP MICRO 2024; RSR++; mixed-T SNN 28nm | arXiv:2407.14073; Chen/Chang arXiv:2503.19643 | Mixed T=2 vs T=10 is a local delta vs uniform-T LoAS | FTP inner-join is a strong prior; 28 nm mux T already exists | support-located |
| I005 | TSBG; Eyeriss; Gustavson | owner prior judgment; local CPU result | Broadcast works on these traces (CPU) | Family is old; CPU ≠ RTL | challenge-located |
| I006 | Prosperity HPCA 2025; SumMerge | Prosperity-HPCA-2025; ICS 2021 SumMerge | Product/bit 2.33× located locally | Title novelty challenged by the prior itself | challenge-located |
| I007 | Gist ISCA 2018; late BN SNN hardware | Gist 2018; A8 RTL notes | Packet-at-boundary has local RTL | Mixed-T neuron 28 nm already published | search-incomplete |
| I008 | sparse ConvTranspose SNN | FireFly-S arXiv:2408.15578; local m1681 shards | Decoder share historically large | No complete Table-A in this repo | search-incomplete |
| I009 | FireFly-T SRAM permute | arXiv:2505.12771 | Byte-write 3D attention layout described | Foundry 1RW/2RW map untested | support-located |
| I010 | SpiLiFormer lateral inhibition | arXiv:2503.15986 v2; ICCV 2025 pp. 24539–24548 | Training analog exists | Classification, not OF; not an ALU | search-incomplete (full method not re-implemented) |
| I011 | PAFT running-BN; NF-SpikingVTG | local m247; NF-SpikingVTG name | Running vs frozen BN split located locally | PAFT-ep4 AEE ~1.47 fails 1.259 | mixed |
| I012 | sparse popcount zero-term skip | T1_T4_ep35.json overlap mean 0.013 | Local census suggests overlap almost empty | Ordinary zero-skip ALU; may already be in SV | no-direct-evidence-located for *this net’s* term histogram on **ep34** |
| I013 | Zhang CICC 2026 DLSS; event OF 28nm | local SOURCE_LEDGER row | Feature-map temporal skip exists | Different object from I001 | support-located |

---

## Reopen round (after the packet)

Prompt (same to the single participant): *Given the evidence packet, add revisions, alternatives, or disconfirming studies. Do not overwrite I001–I008.*

New records:

- **I009** (B11): FireFly-T byte SRAM permute for T×15×15. Stage `post-check`. Parked (feasibility).
- **I010** (B12): SpiLiFormer-style inhibition on `same_zero`. Parked (no circuit face).
- **I011** (C5): foldable running-BN. Parked (1.47 identity).
- **I012**: term-wise skip. Kept as M1 add-on, not a title.
- **I013**: encoder feature-map temporal skip, explicitly split from I001 so the CICC analog cannot hide inside the score leaf. Parked as a different object.

ASTER analog CIM (arXiv:2511.06770) and SpikePool (arXiv:2510.12102) re-confirmed **out of scope**.
