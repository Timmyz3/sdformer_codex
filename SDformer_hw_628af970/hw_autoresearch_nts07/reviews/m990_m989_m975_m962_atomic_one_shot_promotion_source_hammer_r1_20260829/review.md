# M990 independent M989 source hammer

Verdict: **GO author M991 release only (96/100, P0/P1/P2 = 0/0/2).** The production promotion and all EDA remained unexecuted.

M989 repairs the M987 race at the state-machine level: a fixed atomic `mkdir` lock serializes cooperating runners; a canonical permanent ATTEMPT is consumed before copy; WORK is fixed rather than PID-specific; TARGET is rechecked after the complete WORK seal; and `mv -T` publishes to a literal destination without nesting. Independent 24-way waves produced exactly one first-wave copy winner and zero second-wave winners. A crash immediately after canonical ATTEMPT creation also permanently blocked retry.

Two P2 boundaries remain. The protocol is not intended to defend against a same-UID noncooperating mutator in the final check-to-rename window. Also, the outer root manifest excludes the nested `original_quarantine/SHA256SUMS` pair by basename; actual payloads remain root-covered and expected inner hashes remain in root-covered provenance, but the M993 result hammer must explicitly verify the nested quarantine.

M991 release authoring is permitted. M993 execution remains forbidden until M991 and an independent M992 release hammer are sealed.
