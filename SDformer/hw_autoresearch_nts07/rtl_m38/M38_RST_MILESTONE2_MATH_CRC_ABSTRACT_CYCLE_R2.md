# M38-RST milestone-2 math, CRC, and abstract-cycle contract, revision 2

Status: milestone-2 executable-reference evidence only. This revision admits
the complete RST arithmetic reference, a canonical context frame and strict
loader reference, recursive identity of the final M31/M37 evidence, and safety
and liveness of an abstract integrated cycle model. It does **not** admit
integrated RTL, integrated-RTL VCS, DC/STA/Formality, macro-aware area or power,
trained accuracy, memory traffic, Local/Motion full-system cycles, energy, or a
paper headline.

The machine-readable source of truth is
`contracts/m38_rst_math_input_contract_r2_20260822.json`; the frozen audit is
`results/m38_rst_math_crc_and_cycle_r2_20260822/m38_rst_math_crc_and_cycle.json`.
Revision-1 artifacts remain frozen and are not promoted by this revision.

## 1. Recursively closed evidence identity

The audit binds exact schemas, statuses, hashes, run manifests, source files,
logs, vectors, and parsed pass counters. Any absent, extra, stale, or changed
item fails closed.

| Dependency | Frozen final identity |
|---|---|
| M31 receipt | r3, `M31_UNIFIED_T10_T2_VCS_PASS_R3_LEAF96` |
| M31 run | `m31_unified_t10_t2_vcs_r3_leaf96_20260822` |
| M31 contract SHA-256 | `f98cdde7ad617ba0ceac14d9b145e3671403698ee40bae54c492010dc91997fd` |
| M31 receipt SHA-256 | `3785a36272845bb5ea240d9aa7eed5bdc934b6cf453ebf2a90f5a16131109577` |
| M37 receipt | r2, final VCS r7 |
| M37 run | `m37_csd_reconstruct_t10_vcs_r7_20260822` |
| M37 contract SHA-256 | `476537dfe5cc7ada88d161a73f7c0d7b50c7a05ba0608a4f90c42b6a5097be5a` |
| M37 receipt SHA-256 | `441531803e3f193bd1f348bacf16291bfab18db4903320549dd6f67d17b43344` |

The M31 source census still proves one shared multiplier pool, 96 parameter
lanes, and one data-multiplication leaf. The M37 receipt is used as a recursively
verified phase-decoupled control predecessor; it is not evidence that M38 RTL
exists.

## 2. Complete arithmetic semantics

M38 consumes 30 signed q8 right factors and 30 two-bit ternary left codes for
rank 3 and ten temporal rows. Legal codes are `00 -> 0`, `01 -> +1`, and
`10 -> -1`; `11` is illegal. Negation is performed after widening, so
`-(-128)=+128`. A ternary product is signed 9 bit and a three-term rank sum is
signed 10 bit over `[-384,384]`.

The rank sum is combined with the signed-Q24 bias in the explicit wide
reference domain, clamped to signed Q24, and compared using
`saturated_value >= threshold`. Equality emits an event. The executable audit
covers all 768 q8-by-legal-ternary scalar pairs, all legal rank triples over
the complete q8 range, extrema, saturation, and threshold boundaries.

## 3. Canonical context frame

All array indices are serialized in ascending order. Every fixed-width field
uses exact two's-complement or unsigned encoding as applicable. Bits within a
field are emitted least-significant bit first; the first serial bit becomes
bit 0 of byte 0.

| Field | Width (bit) | Running logical end |
|---|---:|---:|
| `right_factor_q8[0:29]` | 240 | 240 |
| `left_ternary_code[0:29]` | 60 | 300 |
| `bias_q24[0:9]` | 240 | 540 |
| `threshold_q24` | 24 | 564 |
| `stage1_requant_shift_u5` | 5 | 569 |
| `generation_u16` | 16 | 585 |
| zero pad before CRC | 7 | 592 |
| CRC-32C | 32 | 624 serialized |

Thus the arithmetic payload is 569 bits, the CRC-protected payload before pad
is 585 bits, the logical context excluding pad is 617 bits, and the physical
serialized frame including pad is 624 bits (78 bytes). Padding is explicit and
is not counted as logical state.

CRC-32C/Castagnoli uses the reflected recurrence polynomial `0x82F63B78`,
initial value `0xFFFFFFFF`, final XOR `0xFFFFFFFF`, no extra output reflection,
and least-significant-bit-first CRC serialization. The mandatory standard
check is `CRC32C("123456789") = 0xE3069283`.

The nontrivial frozen golden frame has generation `0xBEEF`, protected payload
length 74 bytes, CRC `0x4FBC4933`, and complete-frame SHA-256
`d77db6f549d0c851715b6353e1916670b36022bc19f05cc72b72a1dad6f97102`.
Its complete serialized hexadecimal value is:

```text
8081c0dfeff7fbfdfeff000102030507090b0d11171f2f3f4f5f6f787e7f2449
922449922409000008702f0cdce1ffffff0f000010000000241e0090d0e3fffff
7fffff744b3d8de7d013349bc4f
```

## 4. Strict fragment and generation protocol

The frame is loaded as ten 64-bit fragments: fragments 0 through 8 contain
64 valid bits; fragment 9 contains 48 valid low bits and its high 16 bits must
be zero. Indices must arrive exactly once in ascending order. Duplicate,
missing, out-of-order, wrong-valid-width, nonzero-unused-bit, bad-CRC, illegal
ternary, stale-generation, incomplete-frame, or undrained-activation cases
invalidate the entire shadow. Recovery starts at fragment 0.

Every failed or incomplete shadow leaves the active context byte-for-byte
unchanged. A verified shadow becomes active atomically only after the datapath
is fully drained. If an active generation exists, the modular delta
`(candidate - active) & 0xffff` must be in `[1, 0x7fff]`; this admits forward
wrap while rejecting equal, stale, and ambiguous half-range updates.

## 5. Executable abstract integrated cycle protocol

The abstract model contains five stage-1 phases, one 384-bit intermediate
slot, an explicit stage-1-complete pending holder, five reconstruction beats,
a shared 16-entry FIFO, and exactly one FIFO push port. It is a cycle-level
control reference, not RTL.

Let `pop` be a successful sink dequeue in the current cycle. The credit rules
use post-pop occupancy:

```text
0 <= occupancy + reserved <= 16

M38 launch allowed iff:
    occupancy_after_pop + reserved + 5 <= 16

other writer allowed iff:
    no M38 beat owns the push port
    && occupancy_after_pop + reserved + 1 <= 16
```

An M38 reconstruction launch atomically reserves five FIFO entries. Each
committed M38 beat converts one reservation into one occupied entry. M38 owns
the only push port on those cycles; the other writer is denied. `done` occurs
when beat 4 commits into the FIFO, not when the sink later consumes it.

On reconstruction phase 4, combinational reads observe the old slot. The same
edge may install a completed-pending tile or a same-cycle stage-1 completion.
Pending materialization depends on returned slot credit, not on replaying the
original one-cycle commit pulse. Consequently the old tag is retired and the
new tag is installed without aliasing.

FIFO-full simultaneous pop/push returns the old head to the sink and writes the
new tail with unchanged occupancy. A context release or T10/T2 change is
rejected until stage 1, pending, slot, reconstruction, reservations, and FIFO
are all drained.

## 6. Executable results and bug witnesses

The frozen Python-3.6 audit reports:

- 578 legal credit-state/input combinations checked, with maximum
  `occupancy + reserved = 16`;
- 32 no-stall tiles accepted at cycles `0,5,...,155` and completed at
  `9,14,...,164`, giving finite completion `5 + 5*N = 165` cycles;
- a blocked tile materialized from pending at cycle 14 while phase 4 read the
  old slot tag and the edge wrote the new tag;
- 40 tiles completed under 90 sink-stalled cycles, by cycle 290, with maximum
  FIFO occupancy 15;
- FIFO-full simultaneous pop/push, single-writer priority, and drained
  `T10 -> T2 -> T10` switching pass;
- two undrained context-switch attempts are rejected.

The model also preserves two concrete counterexamples that the protocol fixes:

1. without explicit reservation and shared-writer arbitration, occupancy can
   evolve `13,14,15,16,17`;
2. if pending materialization incorrectly depends on the old commit pulse, a
   completed tile remains deadlocked after slot credit returns.

Liveness is conditional on eventual sink readiness. Permanent downstream
backpressure is intentionally not claimed to complete.

## 7. Honest performance boundary

For resident-context, no-backpressure T10 reconstruction only, serialized M31
has steady II 10 and the abstract parallel schedule has steady II 5. The
conditional kernel throughput limit is therefore 2.0x. For finite `N`, the
cycle ratio is `10*N/(5+5*N)`, which approaches but never exceeds 2.0x.

This ratio excludes configuration loading, physical FIFO/SRAM timing, the
other shared writer, T2 fallback, attention, operator transitions, address
traffic, memory contention, and trained module coverage. It is neither a
Local/Motion system acceleration nor an energy/area result. CRC, control,
reservation, tags, selectors, rank adders, saturation, comparison, and context
storage are known nonzero costs and remain unmeasured.

## 8. Next admission gate

The next milestone must implement this exact protocol in integrated RTL and
verify it with Synopsys VCS, including arithmetic miters, golden-frame loading,
all negative loader cases, generation wrap/staleness, pending materialization,
single-push-port collisions, FIFO-full pop/push, long sink stalls, and drained
T10/T2/T10 switching. Only after that may identical-constraint DC/STA,
Formality, SAIF/PTPX, macro-aware memory, trained Local/Motion deployments, and
address-timed full-system cycles be used to consider PPA or headline claims.
