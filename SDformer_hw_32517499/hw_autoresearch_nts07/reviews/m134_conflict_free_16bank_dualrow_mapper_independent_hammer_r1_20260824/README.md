# M134 conflict-free 16-bank dual-row mapper independent hammer

## Verdict

**92/100, conditional pass. P0=0, P1=1, P2=4.** The fixed 3,680-word modulo-16 mapping is correct for every 16-word window. The evidence closes logical bank conflict, row crossing, address bounds, and reorder for the combinational port cut. It does not close the physical sixteen-bank response path, bank macro cost, latency alignment, power, or physical speedup.

`docs/359_DATE终局冻结_20260813.md` remained at SHA-256 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

## Exhaustive functional result

| Check | Result |
|---|---:|
| Frozen-source production VCS rebuild | PASS, compile/sim RC 0 |
| Independently exhausted legal bases | 3,665 (`0..3664`) |
| Independently exhausted illegal 12-bit bases | 431 (`3665..4095`) |
| Legal logical-word checks | 58,640 |
| Physical address/bank-cardinality checks | 58,640 / 58,640 |
| Row-crossing windows | 3,435 |
| Incremented-bank row-address checks | 27,480 |
| Base offset 0 / each offset 1..15 | 230 / 229 |
| Physical bank row range | `0..229`, no overflow |
| Logical reorder mismatches | 0 |
| Valid-low payload independence checks | 64 |

For every legal base and logical offset `i`, the independent model checked `word=base+i`, `bank=word mod 16`, and `row=floor(word/16)`. The sixteen offsets formed exactly the set of banks `0..15`. For base bank `b`, exactly banks `0..b-1` used `base_row+1`; all others used `base_row`. Independently generated physical-word payloads then returned in exact logical order.

The result is stronger than the production test on invalid inputs: the independent test quarantined all 431 illegal 12-bit bases, not just three representatives.

## Parameter and port-cut attacks

Five nonproduction simulations (`WORDS=3679`, `BANKS=8`, `WORD_W=16`, `BASE_W=11`, `ROW_W=7`) reached the exact time-zero `M134 production geometry drift` fatal. VCS maps this `$fatal/$finish` to shell RC 0, so the fatal site/message—not nonzero process status—is the rejection evidence.

That guard is inside `` `ifndef SYNTHESIS ``. A `SYNTHESIS + BANKS=8` attack bypassed it and exposed an unknown logical word from the hardcoded modulo-16 selection. `BASE_W=11` also raised a VCS out-of-bounds-select warning. The parameters are therefore documentation for one frozen geometry, not a synthesis-safe generalized interface.

The independent port-cut attack deliberately supplied stale/skewed per-bank data unrelated to `bank_row_addresses`. M134 accepted and rotated it because the interface has no response valid, row/bank tags, latency alignment, or skew checks. That behavior is consistent with its declared combinational port cut, but it is the principal P1 blocker before real SRAM integration.

With `request_valid=1` and an X in the base, `request_legal` remains X while the other outputs stay quiet. This is a four-state verification boundary, not a demonstrated two-state silicon functional error.

## Exact-SHA DC result

The sealed Synopsys DC V-2023.12-SP3 run independently verifies its input and evidence manifests:

| 3 ns virtual-clock logic-only metric | Result |
|---|---:|
| TSMC28 corner | `ssg0p9v125c` |
| Cell area | 2,054.555977 um2 |
| Leaf / combinational / sequential cells | 3,808 / 3,808 / 0 |
| Logic levels | 14 |
| Worst setup / hold slack | +1.5947 / +0.5069 ns |
| Critical data arrival | 0.9553 ns (`logical_base_word[1]` to `logical_words[343]`) |
| Constraint violations | 0 |
| SRAM macros | 0 |

This number includes the legality check, sixteen address generators, rotation, mask, and conflict indicator under virtual ideal clock and ZeroWireload. It excludes the sixteen 230x32 SRAM macros, macro access, periphery, 512-bit bank wiring, alignment registers, clocking, and power. It is not complete frontend latency, Fmax, macro-inclusive PPA, or physical throughput.

DC emitted two `VER-318` signedness warnings. Precompile `check_design` reported one `LINT-1` and sixteen `LINT-31` warnings; postcompile `check_design` was warning-free after optimization.

## Findings

### P1 — physical response path remains open

The mapper assumes `bank_words` already correspond to its combinational row addresses. Implement or model sixteen realizable banks, register request identity across actual read latency, align responses, and verify stale/skew faults plus address-to-macro-to-rotation timing.

### P2 — claim boundaries

- Doubling eight banks to sixteen eliminates the second logical read port, but is not free. A matched macro/periphery/wire/energy comparison is absent.
- The geometry guard disappears under synthesis and internal constants remain fixed to modulo 16.
- Unknown request payload is not explicitly fail-closed in four-state simulation.
- The exact DC point is a warning-bearing zero-macro combinational port cut; no Fmax sweep, power, or physical integration exists.

## Verify

From the hardware root:

```bash
sha256sum -c reviews/m134_conflict_free_16bank_dualrow_mapper_independent_hammer_r1_20260824/input_manifest.sha256
reviews/m134_conflict_free_16bank_dualrow_mapper_independent_hammer_r1_20260824/audit_m134_independent.py
(cd reviews/m134_conflict_free_16bank_dualrow_mapper_independent_hammer_r1_20260824 && sha256sum -c manifest.sha256)
```

`run_m134_independent_hammer.sh` is a one-shot commercial VCS runner and intentionally refuses to overwrite captured evidence. Generated `csrc`, `simv.daidir`, and `simv.vdb` trees are reproducible scratch excluded from the durable review manifest.

