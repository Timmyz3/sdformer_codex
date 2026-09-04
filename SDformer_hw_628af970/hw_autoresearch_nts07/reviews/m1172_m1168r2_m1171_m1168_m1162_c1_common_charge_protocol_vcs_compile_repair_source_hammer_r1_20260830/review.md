# M1172 independent hammer: M1168R2 compile repair

Verdict: **PASS (99/100)** for authoring a separate M1173 launch release. This
hammer does not itself authorize or execute VCS.

The old r1 attempt remains consumed. Its pre-elaboration failure quarantine is
recursively sealed and still contains exactly five `DTINPCIL` plus five
`IRFPCA-AUTOVAR` diagnostics. Neither its attempt identity nor its result
namespace is reusable. The r2 attempt, work, and result namespaces are fresh.

The repaired `force_request` has exactly five static module-scope staging
fields. Every field is assigned from its matching formal before the first
hierarchical force. The task contains exactly ten distinct hierarchical DUT
force targets and zero automatic task formals on a force RHS. Independent
attacks rejected a missing assignment, two field aliases, a direct automatic
RHS, two force-target corruptions, an automatic-lifetime staging declaration,
old-namespace reuse, and hammer/release bypasses.

The r2 TB is byte-for-byte derivable from r1 using only the documented staging
repair and r2 identity strings. The SVA differs only in module identity. Thus
the package retains 16 assertions, six covers, 18 directed cases, 24
deterministic random transactions, seven protocol attacks, two service-side
assumption attacks, three reset states, the II=2 check, and the frozen M935
one-row/one-task normal path.

The future runner remains exact-SHA, fresh-namespace, exactly-once, and gated by
both this recursively sealed hammer and a separately sealed M1173 release. No
runner, VCS, simv, EDA process, or license query ran during this hammer.

`docs/359_DATE终局冻结_20260813.md` remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

Claim boundary: source hammer only. Functional VCS, timing, cycles, speedup,
PPA, power, energy, system speedup, paper-citability, and headline status all
remain false pending the separate release and single permitted r2 run.
