# M530 / M529 DW1RW source-repair author admission hammer

## Verdict

**PASS, 98/100, P0/P1/P2 = 0/0/2.** Root may spawn exactly one fresh source-only repair author. This review authorizes no HDL, EDA, CPU, GPU, remote job, result directory, VCS launch admission, trace recurrence or performance claim.

## Identity and seal result

- M530 repair admission JSON: `ee84cfb56607053c2d38d33f5f691773c40e17bc8794bd2df8e26ca94e8afeb2`.
- Its member sidecar and outer sidecar file hashes are respectively `bfa01dcce9d26f9a96b83095fe5a8c7276c5c1d44723ecb9ccfc2f4105c50edd` and `eb0db2b6a5568cc2764772569bfb442e3e3d79c9a5dff985277442a3b02a0ab7`; both seals pass.
- The admission exactly binds the failed M529 r1 source contract, failed handoff, and independent 70/100 static hammer with P0/P1/P2 = 1/4/2. The failed handoff and failed hammer are double-sealed and jointly prove that the original M529 author admission and its only source package were consumed.
- M528-r4 result and its independent 99/100 hammer resolve with the expected hashes and valid package seals. The macro adapter and binding plan hashes match. `docs/359` remains `dedde7ce...dfc4`.
- Identity mismatches: **0**.

## Authorization boundary

The contract permits one fresh repair author and one source-only package. It authorizes zero VCS, Icarus, Verilator, DC, Formality, PT, PTPX, CPU, GPU, remote jobs, or result directories. New `_r2` top/SVA/TB/runner/source-contract names preserve the failed `_r1` source identity; the new M530 handoff and request paths do not overwrite or reseal the M529 failure record.

The author must not create a VCS launch admission. A fresh independent static hammer of the exact r2 package is mandatory; only P0=0 and P1=0 may lead root to create a separate one-attempt functional VCS launch admission.

## Repair completeness

All blocking findings from the 70/100 review are represented:

1. The parent-only format check becomes an explicit combinational preaccept predicate and atomically gates ready/accept, psum write, completion, scratch write/elision, and every architectural event/counter. A malformed beat must cause zero architectural events before sticky fault.
2. Overflow may use only authoritative operands with a matching parent response. A held final payload cannot be poisoned by stale slot data.
3. The TB must independently reconstruct parent refcounts, the full live bitmap, and deterministic event/counter totals; DUT directory/live state cannot seed the oracle.
4. Every frozen normal corner needs an explicit reproducible hit/minimum, separate from attacks, and the runner must accept exactly one coverage-summary token with every minimum met.
5. Functional VCS, frozen-trace RTL/cycle recurrence, and the already sealed CPU DSE are three distinct identities. Only the later trace attempt may evaluate recurrence and the two 1.50x gates.

## Frozen mechanism and claim boundary

The admission freezes M504 matcher/tie/equal-later behavior, stable population/row-ID order, one lookahead and immediate-next deadline, queue+pending <=2 without consume credit, dead-write-only with mandatory live-parent storage, signed12/INT8/16-source/signed19 boundaries, nine 128x128 1RW slices and lower-64-row address binding. It forbids combined PVRF, concurrent 1R1W, a second lookahead/architecture, decoder, and full-network scheduler. M528-r4 CPU/traffic/capacity numbers and all non-headline boundaries remain unchanged.

## P2 observations

- Old-admission consumption and the M528 hammer seal are currently resolvable through exact-bound descendants plus this independent check, but the r2 source contract/handoff should repeat their direct paths/hashes to avoid later transitive reconstruction.
- The r2 contract/handoff and next static hammer should enumerate all eleven named normal-cover minima and explicitly retain exact-SHA foundry behavioral `.v`/private-manifest VCS binding. Any later DC/Formality admission must independently bind the `.db` and nine blackbox/cutpoints and reject the behavioral model or register-array fallback.

These are non-blocking for one source-only repair author; they are mandatory checks at the repaired-source static hammer.

## Next gate

Spawn exactly one fresh source-only repair author, confined to the listed r2/new-M530 outputs, with an all-zero tool-run receipt. Do not run VCS or any other tool until a fresh independent source static hammer seals P0=P1=0 and root creates a separate launch admission.
