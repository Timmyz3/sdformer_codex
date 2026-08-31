# M803/C2 R16 fresh source-hammer request

Perform a fresh, adversarial source-only review of M803. Do not reuse the
author's PASS conclusion. The repair is deliberately additive: frozen M490
same-cycle reuse remains, while only M499 R5 request/response channel opening
and response-before-request state mutation are migrated into the new namespace.

The hammer must inspect RTL semantics, wrapper identities, SVA/TB contracts,
filelists, exact-SHA runner closure, atomic result/failure behavior, and all
frozen SHA boundaries. It must rerun the Python 3.6 closure and pre-mkdir dry
run, including the wrong-runner-SHA negative. It must not invoke VCS, simv,
lmutil, DC, Formality, PT, or any EDA process and must not create an attempt,
result, candidate release, or launch admission.

A PASS may authorize only authorship of a separate VCS launch candidate. It
may not authorize execution. Any semantic ambiguity in same-slot reuse,
request-fault/legal-response retirement, ledger conservation, or exact cycle
hard gates is P0 and must remain fail-closed.
