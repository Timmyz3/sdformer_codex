# M38-RST type-strict executable reference, revision 5

Status: `PASS_M38_R5_TYPE_STRICT_MATH_PROTOCOL_COMPLETE_REACHABLE_STATE_ONLY`.

This revision repairs the independent-review finding in r4: Python container
equality treated JSON booleans and integers as interchangeable. R5 admits only
the executable Python 3.6 arithmetic, configuration protocol, typed-offer
atomicity, finite abstract reachable-state safety, directed drain paths, and
conditional T10 kernel scheduling reference. Integrated RTL, integrated-RTL
VCS, DC/STA/Formality, PPA, power, energy, memory timing, trained coverage,
Local/Motion system cycles, system speedup, and headline claims remain false.

## 1. Exact machine artifacts

- contract: `hw_autoresearch_nts07/contracts/m38_rst_math_input_contract_r5_20260822.json`, SHA-256 `5ec623bba1023035dad68d695774168783efa45e1d7caafe63a16f7f16d32f6e`;
- analyzer: `hw_autoresearch_nts07/system_simulator/scripts/analyze_m38_rst_math_protocol_reachable_r5.py`, SHA-256 `e88a1016c9e258f26c45ea2ea11e86c20afaeb78d1c5b5fea27f6928cb6f2748`;
- regression: `hw_autoresearch_nts07/system_simulator/tests/test_m38_rst_math_protocol_reachable_r5.py`, SHA-256 `54b2afa852ead7952491e1454e8b0407d2bab3f31e33258c5955c11b5990aba9`;
- result: `hw_autoresearch_nts07/results/m38_rst_math_protocol_reachable_r5_20260822/m38_rst_math_protocol_reachable_state.json`, SHA-256 `fd4e4769fe39ce0eadb3b7f9c7df5cdae7088933b564d22e58a0fa03867570de`.

R5 imports the exact frozen r4 analyzer, SHA-256
`169e5dc3085cdcb6a87d945b53f8bb9f5420242e81534ecbe6fd4ac98ceabf21`,
and reuses its executable arithmetic, protocol, cycle, and BFS implementation.
It replaces only the JSON loader and semantic/review equality gates, then
records the frozen base identity in the result.

## 2. R3 and r4 remain superseded NO-GO evidence

R3 remains `NO_GO_FAIL_CLOSED_REVIEW_SUPERSEDED_DO_NOT_CITE`. Its independent
review is
`hw_autoresearch_nts07/results/m38_rst_math_protocol_reachable_r3_20260822/m38_r3_independent_hammer_nogo_review.json`,
SHA-256 `d93335610d5d01d02a33507014188e9348f8a33e3c16035c7b51c640747ff9d6`.

R4 is `NO_GO_TYPE_STRICT_REVIEW_SUPERSEDED_DO_NOT_CITE`. Its independent
review is
`hw_autoresearch_nts07/results/m38_rst_math_protocol_reachable_r4_20260822/m38_r4_independent_hammer_nogo_review.json`,
SHA-256 `d45406ce03b486d98a33e0a8fdf486dc3c1e1bde662392d38aec82865857f14a`.
That review scored r4 at 86/100 with one P1 finding: four boolean/integer
type-confusion contracts still passed. R5 rehashes the exact r4 artifacts and
both NO-GO reviews. No r3 or r4 file is overwritten.

## 3. Recursive type-strict repair

The r5 equality predicate walks the complete object recursively:

- every node must have exactly the same Python type;
- dictionaries must have exactly the same key population and recursively equal
  values;
- lists must have equal length and type-strictly equal elements in order;
- scalar values must compare equal only after their types match.

The gate covers all five frozen semantic sections:
`frozen_architecture`, `canonical_configuration_frame`, `offer_schemas`,
`reachable_state_model`, and `theory_rules`. It also covers the complete M31-r4
and M37-r8 independent-review payloads against their hash-bound canonical
evidence, and the entire rebuilt M31-r4 admission payload against the bound
admission. The prior r4 executable/subset checks still run after these gates.

The four exact r4 counterexamples are now separately rejected:

- `intermediate_elastic_slots_target: 1 -> true`;
- `configuration_load_cycles_included: false -> 0`;
- `t10_offer.ranges.tag[0]: 0 -> false`;
- `reachable_state_model.reserved_domain[0]: 0 -> false`.

The JSON loader also rejects duplicate keys and non-standard numeric constants
such as `NaN` and `Infinity` before semantic validation.

## 4. Preserved executable evidence and claim boundary

The underlying executable results remain unchanged:

- 768 signed-q8 by legal-ternary scalar products;
- 769 constructive witnesses covering every integer rank sum `[-384,384]`,
  explicitly not every legal rank triple;
- exact Q24 saturation and threshold-boundary semantics;
- strict CRC32C frame, fragment, type, reset, and generation boundaries;
- 11 state-atomic illegal-offer rejections;
- complete finite abstract BFS with 669 states and 10,438 transitions, maximum
  directed drain path 26, and maximum `occupancy + reserved = 16`;
- exact finite T10 ratios for N=1/2/3/32/100;
- completion under 0/90/500 initial sink stalls.

General fairness and hardware liveness remain unadmitted. The asymptotic 2x is
only a resident-context conditional T10 kernel scheduling limit and is not a
Local/Motion or full-system speedup.

## 5. Reproduction and independent-review gate

The r5 regression reports 17/17 passing tests. It retains all r4 tests and adds
the four boolean/integer substitutions plus a non-standard-number forgery.
Two clean r5 builds are byte-for-byte identical at SHA-256
`fd4e4769fe39ce0eadb3b7f9c7df5cdae7088933b564d22e58a0fa03867570de`.

An independent invocation of the hash-bound M31-r4 validator rebuilt a payload
byte-for-byte equal to the canonical admission. The M37-r8 source-intent
validator returned
`M37_R8_VCS_SOURCE_INTENT_ADMISSION_VALID=1` with the exact frozen review.

The next gate is a fresh independent r5 hammer. Downstream work must not cite
r5 as admitted model evidence before that review returns GO. Integrated RTL and
Synopsys VCS/DC/Formality/PTPX remain separate future milestones.
