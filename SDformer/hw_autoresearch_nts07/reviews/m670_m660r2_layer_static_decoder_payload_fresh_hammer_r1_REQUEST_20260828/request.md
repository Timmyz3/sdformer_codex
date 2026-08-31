# M670 fresh independent static hammer request for M660-r2

This request freezes the repaired M660-r2 author candidate. It is not a GO,
does not authorize the runner, and must not consume the GPU one-shot.

## Frozen target

| object | SHA256 |
|---|---|
| producer | `53b91b9ec8be00e60a5e029c63c392f5fe5e4773de92b440c6d4561dc1ab0116` |
| runner | `c8549148eed848fc0b8c6e58a5003f4b2c99f5822dce1ea89c5b31368ca78bb9` |
| contract | `0c6c22532ffa1a1cb70fd5a55cf94a75a594a20244ed878e6dc85f5ff47452fd` |
| author tests | `aa76ff11f95be8faf7de2eca9b7fa54035be6238fb467afe284f043d3f258ddd` |
| author-handoff outer-seal file | `509f836c054647a854690d9accc762756aa4d5147bea6b44aac64dbd83303a35` |
| CPU preflight receipt | `8dbab013ed5099b699eed0a1d7e085e6afdd9f873f73d53006c027338b37af3a` |
| CPU preflight outer-seal file | `adbc96005afa1126567f2c2ce70283b1db37d7cb6c81f6590d5cffff132b05ae` |
| M666 outer-seal file | `455447d9693f57fc5b1ddf5610009bdfbcb2af8b57f6473e3f546e3865cff82a` |

## Mandatory attacks

- independently rehash every contract input, the author handoff, CPU preflight
  and M649/M658/M659/M662/M666 nested seals;
- reconstruct the real H67 wrapper/leaf boundary independently and verify that
  the CPU receipt has exact checkpoint load, no forward, the exact named leaf,
  and a contract identity equal to the frozen r2 contract;
- attack direct-leaf test doubles, wrapper aliases, symlink/parent traversal,
  bad scalar shape/dtype/mode, NaN/infinity/nonpositive theta and checkpoint
  topology drift;
- mutate the live theta before/inside/after leaf, deconv and sample boundaries;
  prove the cloned theta does not alias storage and that all 62 checks are
  sample/order bound;
- attack the miter with `+0.0/-0.0`, adjacent ULPs, NaNs/infinities, chunk
  boundaries and unequal bytes with equal numerical values; recompute raw
  uint32 mismatches, signed-zero counts, max ULP and both hashes independently;
- prove deployment admission is the conjunction of theta S10, mismatch zero,
  signed-zero zero, max-ULP zero and per-call hash equality for all ten calls;
- inject early, middle, late and post-finalization exceptions. Verify all D1
  candidate masks, promoted masks, folded weight, sidecar, stale manifest and
  stale seals are scrubbed before a clean failure package is double sealed;
- prove folded weight and sidecar cannot be serialized before the complete S10
  gate and that fallback `weights` metadata never precontains candidates;
- independently verify deterministic algorithms, cuDNN modes, both TF32
  controls, CUBLAS workspace environment and runtime receipt population;
- keep M658 P2 closure pending until a post-result independent hammer;
- verify M665 schema/packing/route compatibility and the exact 30-or-40 payload
  plus 40-hook lattice;
- ensure the runner completes/reuses the exact CPU preflight before creating
  the attempt directory and preserves the two external SHA roots;
- confirm the canonical S10 output and attempt directory are absent and
  `docs/359` remains unchanged.

Only P0=0 and P1=0 plus an explicit independent `GO` may publish one unique
candidate command. Otherwise the result is NO_GO with no command.

