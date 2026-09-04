# M1186 independent hammer of the M1183 M1168R3 VCS release

Verdict: **FAIL CLOSED, 78/100, P0=2, P1=0. Do not launch M1183.**

The exact 29-file R3 identity set, four recursive evidence seals, consumed R2
quarantine, fresh R3 namespaces, one-compile/one-simv cardinality, foundry
`UNIT_DELAY`, same-UID/memory gates, failure quarantine sealing, and normal
attack-mask counts all check.  No runner, VCS, simv, EDA executable, or license
client was invoked.

Two launch-boundary defects block release:

1. The exact runner's pre-attempt Python gate dereferences
   `release.identity.contract_sha256`.  The exact M1183 release instead contains
   `source_contract_sha256` and no `contract_sha256`.  Launch therefore raises a
   deterministic `KeyError` before creating the attempt namespace; VCS cannot
   run.
2. The runner binds M1182's **source-hammer** review and outer seal, but accepts
   and verifies no digest for the mandatory fresh **release hammer**.  The
   release-hammer gate exists in prose/control policy but is not cryptographically
   bound through the launch path.

Repair must be additive: create a successor runner and successor inert release,
use one canonical source-contract identity key, and bind the fresh release-hammer
review plus outer seal before attempt creation.  Do not overwrite or launch the
exact M1183/R3 artifacts.

All functional/timing/cycle/PPA/power/energy/system/paper claims remain false.
`docs/359` remains `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
