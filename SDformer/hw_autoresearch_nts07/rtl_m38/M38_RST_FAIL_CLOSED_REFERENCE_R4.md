# M38-RST fail-closed executable reference, revision 4

Status: `PASS_M38_R4_FAIL_CLOSED_MATH_PROTOCOL_COMPLETE_REACHABLE_STATE_ONLY`.

This revision repairs the independent-review findings in r3. It admits only
the executable Python 3.6 arithmetic, configuration protocol, typed-offer
atomicity, finite abstract reachable-state safety, directed drain paths, and
conditional T10 kernel scheduling reference. Integrated RTL, integrated-RTL
VCS, DC/STA/Formality, PPA, power, energy, memory timing, trained coverage,
Local/Motion system cycles, system speedup, and headline claims remain false.

## 1. Exact machine artifacts

- contract: `hw_autoresearch_nts07/contracts/m38_rst_math_input_contract_r4_20260822.json`, SHA-256 `d32c6437fc2a70001da1ebeb8f3d52f0acba2f07a81398368abaf852d3f3590c`;
- analyzer: `hw_autoresearch_nts07/system_simulator/scripts/analyze_m38_rst_math_protocol_reachable_r4.py`, SHA-256 `169e5dc3085cdcb6a87d945b53f8bb9f5420242e81534ecbe6fd4ac98ceabf21`;
- regression: `hw_autoresearch_nts07/system_simulator/tests/test_m38_rst_math_protocol_reachable_r4.py`, SHA-256 `b19774fd6dde6bed0f42d7c74d11619c1ff45077af1f86629739ab7a2017f2d7`;
- result: `hw_autoresearch_nts07/results/m38_rst_math_protocol_reachable_r4_20260822/m38_rst_math_protocol_reachable_state.json`, SHA-256 `b2b79a148f738fedb9c67529a991b1549e29c25a3c6dd2b8300bc14aa9673075`.

The arithmetic, cycle, and BFS implementation is the exact frozen r3 analyzer
with SHA-256
`1efaaad25e6dabfbe76870bad95fde371470c598f13438de018087c7a4b050c6`.
R4 replaces only the reviewed loading and evidence gates and records that base
identity in every result.

## 2. R3 is superseded NO-GO evidence

R3 is `NO_GO_FAIL_CLOSED_REVIEW_SUPERSEDED_DO_NOT_CITE`. The independent r3
review is
`hw_autoresearch_nts07/results/m38_rst_math_protocol_reachable_r3_20260822/m38_r3_independent_hammer_nogo_review.json`,
SHA-256 `d93335610d5d01d02a33507014188e9348f8a33e3c16035c7b51c640747ff9d6`.
It scored r3 at 70/100 with three P1 findings and requires r4 before recursive
use of M38 evidence.

The superseded r3 contract, analyzer, regression, result, specification, and
NO-GO review are all rehashed by r4. R3 files are not overwritten.

## 3. Fail-closed repairs

### Complete semantic-contract equality

R4 compares the full key population and every value of
`frozen_architecture`, `canonical_configuration_frame`, `offer_schemas`,
`reachable_state_model`, and `theory_rules` with the exact SHA-bound canonical
r3 contract. This is full object equality, followed by the executable
invariant checks; it is not a selected-field audit.

The regression individually mutates all 14 reviewer counterexamples:

- eight reachable-state fields: state fields, context modes, both phase
  domains, reservation relation, writer rule, overflow rule, and liveness
  scope;
- T10 exact keys and tag range plus other-writer mode enum;
- frame field order, field bit order, and reflected CRC polynomial.

Every modified contract has a new SHA and is rejected before a PASS result.

### Complete loader failure reset

`datapath_drained` type validation now executes inside the same fail-closed
path as fragment validation. A non-boolean value on fragment 1 after fragment
0 sets `failed=true`, resets `next_index=0`, clears the shadow to zero bytes,
and preserves the active configuration. A subsequent fragment 1 is rejected;
only fragment 0 can restart, after which a complete valid frame activates.

### Full M31 independent-admission reconstruction

R4 rejects duplicate keys, checks the exact top-level population, runs the
hash-bound M31-r4 validator over the receipt and live evidence, and requires
the entire rebuilt object to equal the bound machine admission. It therefore
validates identity, manifests, logs, observed counters, source audit, r3
regression, current Formality-filter boundary, admission flags, and claim text.

The five reviewer forgeries—claim text, warning count, T10 II, dynamic phase
index count, and an extra headline key—are each rejected after rebinding their
new path and SHA in a temporary contract.

### Duplicate-key-safe JSON

Contracts, JSON inputs, and both independent review artifacts use a duplicate
detecting `object_pairs_hook`. Contracts containing an early forged `schema`
or `claim_boundary` followed by the legal value are rejected. Duplicate-schema
M31 and M37 review artifacts are also rejected.

## 4. Preserved executable evidence

The reviewed repair leaves the underlying results unchanged:

- 768 signed-q8 by legal-ternary scalar products;
- 769 constructive witnesses covering every integer rank sum `[-384,384]`,
  explicitly not every legal rank triple;
- exact Q24 saturation and threshold-boundary semantics;
- strict CRC32C frame and generation boundaries;
- 11 state-atomic illegal offer rejections;
- complete fixed-point abstract BFS: 669 states and 10,438 transitions, maximum
  directed drain path 26, and maximum `occupancy + reserved = 16`;
- finite T10 ratios for N=1/2/3/32/100 of `1`, `4/3`, `3/2`, `64/33`, and
  `200/101`;
- 40-tile completion under 0/90/500 initial sink stalls in 206/290/700 cycles.

General fairness and hardware liveness remain unadmitted. The asymptotic 2x is
only a resident-context conditional T10 kernel scheduling limit, not a system
speedup.

## 5. Regression and next gate

The Python 3.6 suite retains 17 test methods and embeds the complete adversarial
population within them. A second build must be byte-for-byte identical to the
frozen r4 result.

The next gate is a fresh independent r4 hammer. Only after a GO may downstream
work use r4 as model evidence. Integrated RTL and Synopsys VCS/DC/Formality/
PTPX remain separate future milestones.
