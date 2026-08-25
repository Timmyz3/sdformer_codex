# M38-RST math, strict protocol, and reachable-state reference, revision 3

Status: `PASS_M38_R3_MATH_PROTOCOL_COMPLETE_REACHABLE_STATE_ONLY`.

This revision admits an executable Python 3.6 reference for RST arithmetic,
canonical configuration loading, state-atomic offer rejection, a complete
finite abstract reachable-state graph, directed drain paths, and conditional
T10 kernel scheduling. It does not admit integrated RTL, integrated-RTL VCS,
DC/STA/Formality, PPA, power, energy, memory timing, trained Local/Motion
coverage, full-system speedup, or a paper headline.

The machine-readable sources of truth are:

- contract: `hw_autoresearch_nts07/contracts/m38_rst_math_input_contract_r3_20260822.json`, SHA-256 `96198cd2f40be1edcd750d1d8f7b35ca03a24e4cbc348c47b24ade596750315a`;
- analyzer: `hw_autoresearch_nts07/system_simulator/scripts/analyze_m38_rst_math_protocol_reachable_r3.py`, SHA-256 `1efaaad25e6dabfbe76870bad95fde371470c598f13438de018087c7a4b050c6`;
- regression: `hw_autoresearch_nts07/system_simulator/tests/test_m38_rst_math_protocol_reachable_r3.py`, SHA-256 `7b2bc0f52462e95727ca536b587b894ad8fd656710c7bc3a936a1042d081624e`;
- result: `hw_autoresearch_nts07/results/m38_rst_math_protocol_reachable_r3_20260822/m38_rst_math_protocol_reachable_state.json`, SHA-256 `c4158ef218c06263bb1976bb8c2a89dfd39d6c4b963fae5c1c062f91b807a2dc`.

## 1. Supersession and recursive evidence boundary

Revision 2 is frozen as `NO_GO_STALE_SUPERSEDED_DO_NOT_CITE`. Its contract SHA
is `6aedab8129034c490b9914592b1815878a2395a1b3c9964c7174009fbf28f5dc`
and result SHA is
`6065f9662bd864eb3162d080e36f9f4b83881f665b62b3193cdf8410e9bab095`.
It cannot be cited because its live dependency chain drifted, it described
rank-triple coverage too broadly, its 578-case credit projection was not a
reachable-state proof, and offer validation did not precede every mutation.

M38-r3 recursively admits only the exact current M31-r4 and historical M37-r8
VCS/source-intent identities through independent, hash-bound review artifacts:

| Anchor | Exact admission |
|---|---|
| M31-r4 receipt | `bae2f05e74ffa8863195bda9f222c22fc06364ade872e9cf83d3cd4106e5b77d` |
| M31-r4 independent admission | `e8bd1b6452280396a5c8fc83ce79f34d1ae08256f97b469613207418dcfd0ff6` |
| M31-r4 frozen-input ledger, 10/10 | `41009ec9ec86d4e19489bd49816634ca148340a0f19f784bd2d18bf2d3d0f22d` |
| M31-r4 snapshot admission | `f573cf946332e3ef3b35f307e6de4b873e0d06f341df17ad2f3e5e4ed1e97661` |
| M37-r8 receipt | `363fb61d2838b6379a065dd8eb23b6219441cfb8ed70164766f07d8469e95d97` |
| M37-r8 independent admission | `f133b96a458686e17f94ecf52c26db3c9b753ef7145f4b396a9f047acfda0fa2` |
| M37-r8 immutable RTL snapshot | `ab7d73a6a82f8547437919813d6cf9496d0672fc23f46cfaec0c3d9be46c8cbd` |
| M37-r8 snapshot ledger, 2/2 | `01dc86fcda8ba3627e2de27fbab26866ca794b0e3e8da05d6fbd563cf72364a3` |
| M37-r8 provenance | `f7b88ceafe4447ad7dc1abb11751bead49d3170293ffec1ea6f521aac0c99f99` |

The M31 snapshot directory and its ten members are read-only. The M37 snapshot
is mode `0444`; its validator substitutes the historical input-manifest RTL
target with that immutable r8 file. Live M37 has advanced to r9 and is never
used as r8 evidence. Both independent admissions explicitly keep DC, STA,
Formality, PPA, power, energy, system speedup, and headline claims false.

## 2. Exact arithmetic statement

The executable scalar audit checks every signed q8 value `[-128,127]` against
each legal ternary code `0`, `+1`, and `-1`: 256 x 3 = 768 exact products.
Negation occurs after widening, so `-(-128)=+128`.

The rank-3 audit proves every integer sum in `[-384,384]` by constructing and
machine-checking one legal three-term witness for each value: 769 sums. It does
not enumerate or claim to have exhaustively checked every legal rank triple.
Signed-Q24 saturation, threshold equality, just-below threshold, and both
saturation rails are checked under the frozen `saturated >= threshold` rule.

## 3. Canonical configuration protocol

The arithmetic payload remains 569 bits. Generation and CRC produce 617
logical context bits; explicit zero padding produces a 624-bit serialized
frame. CRC-32C/Castagnoli uses reflected polynomial `0x82F63B78`, initial and
final XOR `0xFFFFFFFF`, and the standard `123456789 -> 0xE3069283` check.

Every fragment must have exactly `data_u64`, `index`, and `valid_bits`, each an
integer but not a boolean, with exact ranges and index-dependent valid width.
Validation rejects wrong valid widths, extra keys, booleans, range errors,
out-of-order fragments, nonzero duplicates, nonzero unused bits, bad CRC,
CRC-correct nonzero padding, illegal ternary code, incomplete frames, and
undrained activation. Any failure discards the shadow without changing the
active context; recovery begins only at fragment 0.

Generation tests cover delta `0x7fff` (new), `0x8000` (ambiguous/rejected),
equal generation (rejected), stale `0x8001`, and forward wrap.

## 4. Typed offers and state-atomic rejection

T10, T2, and other-writer offers have exact key populations, integer-not-boolean
types, ranges, mode, generation, tag, beat, and context rules. All offered
objects are validated before FIFO pop, stage-1 acceptance, slot installation,
counter/history update, or any other state mutation. Eleven adversarial offers
are rejected, and the canonical before/after snapshots are byte-identical for
every case.

## 5. Complete finite abstract reachable-state graph

The BFS starts from reset and continues to a fixed point over mode, context
drain, stage-1 phase, elastic-slot validity, completed-pending state,
reconstruction phase, reservation count, FIFO occupancy, and single-writer
ownership. It reaches 669 states and checks 10,438 transitions.

All reachable reserved values `0..5`, stage-1 phases `idle,1..4`, and
reconstruction phases `idle,0..4` occur. The graph proves reservation
consistency, one writer, no overflow, and maximum `occupancy + reserved = 16`.
Every reachable state has a constructed drain path of at most 26 steps under
the directed drain environment. This is not general fairness or hardware
liveness; permanent sink refusal remains outside the claim.

## 6. Finite scheduling regressions

For the conditional resident-context, no-backpressure T10 kernel, serialized
cycles are `10*N` and the abstract overlapped schedule completes in `5+5*N`:

| N | Overlapped cycles | Exact ratio |
|---:|---:|---:|
| 1 | 10 | 1 |
| 2 | 15 | 4/3 |
| 3 | 20 | 3/2 |
| 32 | 165 | 64/33 |
| 100 | 505 | 200/101 |

Forty-tile eventual-sink tests complete in 206, 290, and 700 cycles after 0,
90, and 500 stalled cycles respectively. The regression also checks phase-4
old-slot read with same-edge new-slot write at cycle 15, FIFO-full simultaneous
pop/push, M38 versus other-writer arbitration, beat-4 commit/done timing, and
drained `T10 -> T2 -> T10` switching.

The asymptotic 2x number is only the conditional T10 kernel scheduling limit.
It excludes configuration load, backpressure assumptions outside the stated
tests, SRAM/DRAM, other operators, T2 fallback, attention, trained module
coverage, and Local/Motion full-network execution. It must not be reported as
system speedup.

## 7. Next hardware gate

The next admission requires integrated RTL implementing this exact arithmetic,
loader, offer-validation, reservation, slot, FIFO, and context-switch protocol;
Synopsys VCS must reproduce the positive and adversarial cases. Only afterward
may identical-constraint DC/STA, Formality, SAIF/PTPX, macro-aware memory, and
address-timed Local/Motion system experiments be used for PPA or acceleration
claims.
