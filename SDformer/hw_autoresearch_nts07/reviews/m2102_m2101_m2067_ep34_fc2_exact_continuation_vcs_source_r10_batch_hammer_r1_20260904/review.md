# M2102 independent source hammer: M2101/R10 single-simv batch

Verdict: **PASS (99/100; P0/P1/P2 = 0/0/1)**. This review authorizes only creation of the separately sealed M2103 launch release. It does not authorize a license query, VCS compile, simulation, or paper claim.

The R10 source retains the R8 functional tasks and oracle byte-for-byte while changing only process granularity. Slots 0--959 execute in one compiled loop. Before every slot, both DUTs are reset and workload-local counters, scoreboards, sticky accepts, address ledgers, and expected-accumulator arrays are cleared. The parser requires 960 ordered reset markers, 960 ordered workload PASS records, one batch PASS, 13,440 row/chunk records, 1,843,200 integer checks per axis, and 115,200 commits per axis.

Mutation checks rejected a 959-PASS partial transcript, a 961-PASS duplicate transcript, and a transcript missing one reset marker. Cartesian row/chunk uniqueness and one-batch-PASS cardinality are enforced in the parser.

R9 is a sealed, non-citable failure: 163 slots completed before slot 163 stalled, 164 simv processes were attempted, automatic retry is false, the owner PID is dead, and no R9 success namespace exists. Its original work/stage evidence remains preserved. R10 ignores those predecessor-private namespaces while requiring the exact sealed R9 failure identity.

The future M2103 release must be a regular, non-symlink, double-sealed file. The runner checks its exact schema, status, source/review/predecessor identities, authorization budget, and claim boundary; changing only an environment hash cannot bypass those semantic checks. Authority is revalidated before each future compile/simulation tool call.

P2: one simv substantially reduces authorization startups but makes the whole batch the failure unit. An incomplete R10 run remains wholly non-citable and cannot be resumed or retried under this identity.
