# M1948 independent M1947 TSBG runner hammer

Verdict: **FAIL / DO NOT AUTHORIZE RELEASE OR ATTEMPT**. Static review only; no license query, attempt, VCS, simv, DC, or PT was run.

M1947 correctly preserves nearly all launch governance: clean environment, scoped `VCS_HOME`, exact frozen source identities and double seals, fresh namespace, same-UID EDA collision exclusion, attempt-before-license ordering, exactly one license query/compile/sim invocation, no automatic retry, signal-safe quarantine, no-replace publication, and future M1949/M1950 identity parsing.

## P0 finding

The final sim-log rejection is not compatible with this installed VCS release's concurrent-SVA diagnostic format. M1947 rejects:

```text
Assertion failed|Error-|$fatal|Fatal:
```

Actual VCS V-2023.12-SP1 failures already present in the repository use:

```text
<hierarchical property>: started at <time> failed at <time>
```

Those historical logs also demonstrate that simulation can continue and print a TB PASS token after the SVA failure. Therefore, the unique PASS token plus the current grep does not prove zero SVA failures; M1947 can publish a false `RAW_PASS`.

Required correction: use an additive fresh runner, make assertion failure terminal (for example, the supported global assertion max-fail control), and independently reject the installed-tool signature `: started at .* failed at` together with broader assertion/fatal signatures. Bind this sealed failed review, keep all other governance, use fresh namespaces, and obtain a new different-author review and launch audit before one VCS attempt.

Claim boundary remains source-only: no RTL execution, speedup, PPA, energy, system, or paper claim is admitted.
