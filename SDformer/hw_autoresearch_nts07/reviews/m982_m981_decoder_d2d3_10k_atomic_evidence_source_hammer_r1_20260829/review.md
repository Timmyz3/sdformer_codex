# M982 independent source hammer

Verdict: **STOP (82/100, P0/P1/P2 = 1/0/0).** No real 10K, EDA, GPU, or remote run was executed.

M981 correctly repairs the M981→M985 numbering, preserves frozen M946/M896/docs359 identities, passes its static checker and 7/7 source tests, and atomically exposes complete payload seal bundles. Its M981 source receipt also passes recursive exact-set verification.

The one-attempt lifecycle is nevertheless not fail-closed. `consume_attempt` first creates a randomized `.attempt.stage.*`, writes `attempt.json`, seals it, and only then renames it to the canonical ATTEMPT. If interrupted before that rename, cleanup merely retains the random stage. A later invocation checks only canonical RESULT/ATTEMPT and its newly randomized stage; it ignores the earlier retained stage. A temporary-directory crash/recovery test retained the first consumed-attempt receipt and then successfully consumed and published a second receipt.

This violates `max_future_attempts=1` and `retry=false`, so M983 release authoring and M985 execution are not authorized. The repair must atomically create the canonical ATTEMPT as the irreversible consumption point before writing/sealing it; an interrupted canonical attempt must permanently block retry while retaining forensic evidence. The additive successor chain is M994→M995→M996→M997→M998.
