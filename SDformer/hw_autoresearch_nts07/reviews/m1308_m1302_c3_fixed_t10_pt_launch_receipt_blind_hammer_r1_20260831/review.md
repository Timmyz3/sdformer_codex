# M1308 receipt-blind hammer — M1302 C3 Fixed-T10 PT launch

Verdict: **99/100, P0=0, P1=0, PASS. Root may run exactly one live PT attempt.**

The exact wrapper, admission, test, M1302/M1288 contracts, M1288 runner,
M1299 review, M917 netlist result, M928 result hammer, M1285 audit, mapped
netlist/SDC, tool/library identities, and `docs/359` are pinned and sealed.
Admission and source-contract authority objects use exact closed keysets.

The verified launch order is: exact identities and seals; fresh M1288/M1302
namespaces; same-UID EDA collision gate; memory, commit-headroom, and disk gates;
real PrimeTime license query requiring issued strictly greater than in-use;
repeat collision and freshness gates; consume the M1302 attempt; invoke the
exact M1288 one-shot runner. Attempt creation is atomic and reuse fails closed.

Independent mock attacks against paths, SHA values, keysets, claim promotion,
boolean-as-integer attempt count, exhausted or malformed licenses, negative
setup/hold, nonzero unconstrained paths, incomplete coverage, constraint
violations, and missing result files cannot produce an admitted PASS. Positive
results still keep power, energy, speedup, system, paper-PPA-ready, and headline
claims false and require a fresh result hammer. No ECO, automatic hold fix,
power, PTPX, or direct alternate PT launch exists in the wrapper.

Authorization is limited to root running the exact wrapper once. The wrapper's
own live checks must pass before either attempt is consumed. Any failure after
attempt consumption is terminal and must not be retried. The generated M1288
and M1302 result directories require a different-author result hammer before
their timing values become citable.

The run remains component-only, pre-layout, ideal-clock, ZeroWireload,
zero-macro and no-SPEF; a PASS is not paper-ready PPA or system evidence.

The M1302 author receipt was not read or trusted. No PT, license query, EDA,
GPU, or remote operation was performed by this hammer. `docs/359` remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
