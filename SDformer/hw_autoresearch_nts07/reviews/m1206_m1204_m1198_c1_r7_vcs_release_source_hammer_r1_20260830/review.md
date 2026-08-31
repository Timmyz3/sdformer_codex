# M1206 independent C1/R7 release-source hammer

## Verdict

**STOP / FAIL.** M1204 is not authorized to launch VCS. One P0 issue makes the fresh release-hammer identity unconstructible.

## What passed

- Exact identities for the M1204 runner, launcher-source contract, release contract, R7 checker/filelist, clean R6 TB, frozen R3 SVA, and docs/359.
- Recursive seals for the M1198 author receipt, M1201 source hammer, and M1204 author receipt.
- Source/release sidecars, shell parsing, the M1204 author checker, and the R7 source checker.
- Fresh attempt/result/work/quarantine namespaces.
- Static presence and ordering of the fresh-hammer gate, attempt token, exactly one UNIT_DELAY VCS compile, and exactly one simv run.
- Same-UID collision, license, memory, timeout, failure-quarantine, PASS-token, four coverage-line, assertion, attack, and oracle gates.
- Sixteen in-memory identity/namespace/command/token/count/claim mutations were rejected.

No VCS, simv, EDA license, GPU, or network action was taken.

## P0: recursively sealed review self-reference

The runner verifies `RELEASE_HAMMER` as a complete recursive seal, so `review.json` must be a member of `SHA256SUMS`. It then requires fields inside that same `review.json` to equal the SHA-256 of `SHA256SUMS` and its outer seal. Updating either field changes `review.json`, which changes `SHA256SUMS`, which changes both hashes being embedded. M1204 supplies no reproducible fixed-point construction, and ordinary recursive sealing cannot satisfy the checks.

## Minimal repair

Use an acyclic authority chain: require exact SHA values for `review.json`, `SHA256SUMS`, and the outer seal as independent runtime environment inputs (or pin them in a later, separately sealed release object), verify all three before attempt creation, and remove the two self-referential fields from the recursively sealed review. Preserve every other M1204/R7 semantic and use a new attempt/result namespace.

## Claim boundary

This review authorizes only successor release-source repair. Functional VCS, timing, cycle, speedup, PPA, power, energy, system, paper-citable, and headline claims remain false.
