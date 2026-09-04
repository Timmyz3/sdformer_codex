# M1191 R5 service-attack oracle source author receipt

Status: `PASS_SOURCE_ONLY__FRESH_DIFFERENT_AUTHOR_HAMMER_AND_RELEASE_REQUIRED__NO_VCS_NO_EDA`.

R4 compiled, elaborated and linked, then failed at 354 ns with the independent
weight checker high and `protocol_error=1`. The failed harness had presented
both joined responses while forcing the wrapper-side copy of frozen M935's
`core_issue_data_ready`. That can assert wrapper `core_issue_data_valid` outside
M935's real execution context; frozen M935 independently treats issue data
without its own request as a core protocol fault. Since wrapper
`protocol_error = core_protocol_error || boundary_fault_q`, R4 mixed a harness
artifact into an external-service assumption oracle.

M1162 itself is unchanged. Its external response-hold rule remains a service
obligation checked by the independent verification checker; the RTL boundary
sticky fault covers early/spurious responses and request cancellation/mutation,
not arbitrary held-response payload comparison.

R5 changes only the two service-negative-test stimuli/oracles. The weight test
presents weight response valid while psum is absent; the psum test presents psum
response valid while weight is absent. The atomic join naturally keeps the
attacked response unready. No hierarchical core-ready force is used. Each test
requires only its own independent sticky service fault, the peer service fault
low, and clean wrapper boundary, frozen-core and composed protocol faults.

The frozen R3 SVA remains byte-identical with 16 assertions and six covers. The
full regression remains 24 deterministic legal transactions, seven DUT protocol
attacks, two service attacks, 29 legal mask-clear checks, three reset states,
explicit completed-beat II=2 and one normal frozen-M935 row/task. Static author
checking passed 498 checks and rejected eight source mutations. No simulator or
EDA tool was invoked. Failed R4 is sealed, immutable and non-reusable; any VCS
attempt requires a fresh different-author source hammer, a new R5 launcher and a
separately hammered release.
