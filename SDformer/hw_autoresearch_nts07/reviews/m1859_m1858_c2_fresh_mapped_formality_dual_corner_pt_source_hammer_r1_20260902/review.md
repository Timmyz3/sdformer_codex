# M1859 independent M1858 Formality/PT source hammer

## Verdict

**PASS SOURCE ONLY — P0=0, P1=0, P2=0, score 99/100.** This review authorizes only a future exact, double-sealed M1860 release for one M1858 attempt: two Formality processes and two PrimeTime processes, one per K8/K1x8 axis, with no automatic retry.

M1858 makes one permitted Formality behavior change relative to the consumed M1850 source: exactly one `set_mismatch_message_filter -warn FMR_ELAB-147`, after `read_sverilog` and immediately before reference `set_top`. The eight frozen warning sites match the double-sealed M1850 failure log. PrimeTime is byte-identical after namespace substitution, and Formality is byte-identical after namespace substitution plus removal of the explanatory comment and this one filter.

The filter is not a proof waiver. The future runner still requires a valid reference/implementation pair, positive passing compare points, zero failing/aborted/unmatched/unverified/black-box evidence, consumed `verify_formality` and `verify_pt` results, semantic `check_timing`/coverage/constraint reports, and verbatim setup/hold WNS publication.

Both official checker runs pass and all 78 official tests pass on CPython 3.6 and 3.12. The independent hammer synchronizes the contract source SHA with every source mutation and rejects 28/28 attacks on both runtimes: the eight M1844 material escapes, two result-consumption bypasses, six filter/order/cardinality mutations, four other message-suppression attempts, six Formality valid-pair/result bypasses, and two M1857 authority bypasses.

All M1858 source/author seals, the consumed M1850 attempt and failure quarantine, the M1857 review/manifest/outer triplet, M1811/M1830 identities, 13 live RTL files, six distinct axis-specific mapped artifacts, and docs/359 are exact. No EDA, simulator, license query, attempt, result, release, timing result, power, energy, speedup, commit, push, predecessor write, frozen-document write, or `ucli.key` access occurred during M1859.
