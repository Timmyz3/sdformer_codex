# M669｜M660-r2 repaired author handoff

Status: `AUTHOR_HANDOFF_R2__NO_SELF_REVIEW__FRESH_HAMMER_REQUIRED`

This package is an author handoff, not an independent review and not a GO.
It repairs the four P1 and two P2 findings in the double-sealed M666 review
without modifying M660-r1, M649/M658/M659/M662/M666, or `docs/359`.

## Frozen r2 target

| object | SHA256 |
|---|---|
| producer | `53b91b9ec8be00e60a5e029c63c392f5fe5e4773de92b440c6d4561dc1ab0116` |
| runner | `c8549148eed848fc0b8c6e58a5003f4b2c99f5822dce1ea89c5b31368ca78bb9` |
| contract | `0c6c22532ffa1a1cb70fd5a55cf94a75a594a20244ed878e6dc85f5ff47452fd` |
| author tests | `aa76ff11f95be8faf7de2eca9b7fa54035be6238fb467afe284f043d3f258ddd` |
| CPU preflight receipt | `8dbab013ed5099b699eed0a1d7e085e6afdd9f873f73d53006c027338b37af3a` |
| CPU preflight outer-seal file | `adbc96005afa1126567f2c2ce70283b1db37d7cb6c81f6590d5cffff132b05ae` |
| M666 outer-seal file | `455447d9693f57fc5b1ddf5610009bdfbcb2af8b57f6473e3f546e3865cff82a` |

## Repairs implemented

- The exact CPU checkpoint load now proves the real
  `owner.sn` `Spiking_neuron` wrapper and its named
  `owner.sn.spiking_neuron` `ATLIFTernaryPSN` leaf before one-shot
  consumption. The actual parameter path is
  `sttmultires_unet.decoders.1.sn.spiking_neuron.thresh`.
- The threshold is cloned to independent storage. Live identity is re-read at
  leaf pre/post, D1 deconv pre/post, every sample pre/post, and finalization;
  the exact successful lattice is 62 checks including the initial identity.
- Folded-output comparison uses raw FP32 `uint32` patterns and canonical bytes,
  reports signed-zero mismatch and maximum ULP, and admits deployment only
  when mismatch/sign-zero/max-ULP are all zero and every hash pair is equal.
- Deterministic algorithms, cuDNN deterministic/benchmark, both TF32 controls,
  and `CUBLAS_WORKSPACE_CONFIG=:4096:8` are frozen and receipted.
- Folded weight and output-scale sidecar are not written before the complete
  S10 theta gate. Any exception first scrubs D1 candidate bitpacks, folded
  weight, sidecar, stale success markers and stale seals, then double-seals a
  clean failure package.
- M658 P2 closure remains
  `PENDING_POST_RESULT_INDEPENDENT_HAMMER`; the author and static contract do
  not claim to close it.
- The output manifest intentionally retains
  `m660_h67_ep35_layer_static_decoder_payload_v1`, little-bit-first
  `C_ORDER_FLAT`, and schema/route-driven records for M665 compatibility.

## Validation boundary

The author CPU suite plus M665 compatibility suite passed `39/39`. The exact
CPU checkpoint preflight passed with `missing=0`, `unexpected=0`, no model
forward, and no GPU. Shell syntax, Python compile, strict JSON/input hashes,
M658/M659/M662/M666 nested seals, and the frozen 40-record predecessor lattice
also passed.

The M660-r2 canonical S10 output and attempt-consumed directory are absent.
No GPU, one-shot, performance simulator, RTL, VCS, DC, Formality, PTPX,
speedup, energy, PPA, system-speedup, or DATE claim is admitted. A fresh
independent static hammer with P0=0/P1=0 and explicit GO is still mandatory.

