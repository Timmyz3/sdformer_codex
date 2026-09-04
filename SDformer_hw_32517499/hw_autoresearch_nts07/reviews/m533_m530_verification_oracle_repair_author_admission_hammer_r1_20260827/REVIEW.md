# M533 / M530 verification-oracle repair author admission hammer

## Verdict

**PASS, 98/100. P0/P1/P2 = 0/0/2.** Root may spawn exactly one fresh verification-only repair author. This review does not authorize VCS or any other HDL/EDA/CPU/GPU/remote run.

## What was independently checked

- The M533 admission JSON and both seal levels pass. It exactly binds the sealed M530-r2 static failure: 83/100, P0/P1/P2 = 0/3/2.
- M528-r4 CPU result and result-hammer identities resolve without mismatch. `docs/359_DATE终局冻结_20260813.md` remains at `dedde7ce...`.
- Frozen top-r2, SVA-r2, macro adapter and macro-binding plan match their required SHA-256 identities.
- M529 and M530 author/source identities are consumed audit records. The new names permit only TB-r3, runner-r3, source-contract-r3, author handoff and a fresh static-hammer request; they do not permit overwrite or reseal of old identities.
- Authorization is source-only: all tool, CPU/GPU, remote/network and result-directory counts are zero.

## Closure of the three blocking M530-r2 findings

1. The cleanroom cycle oracle is sufficiently specified: it must reconstruct row/order/progress, written rows, earliest parent edge, queue/pending/no-credit state and accepted handshakes from stimulus and its own model. It must not read DUT ready, debug, directory, live, queue or internal state. It then predicts macro-read, forward, deadline-hold and stall pulses and compares both cycles and per-epoch totals.
2. Stalled RAW recovery becomes causal and bounded: epoch, consumer, parent and age are recorded; only a matching forward in cycles 1--8 earns credit; timeout is fatal and unrelated/cross-task forwards are forbidden.
3. The future launch schema is closed: exactly one VCS run, every other named run/job counter explicitly zero, unknown authorization keys rejected, and exact runner/contract/static-review/foundry manifest/slow-`v` identities bound.

The two original P2 reinforcements are also mandatory: the malformed parent-only attack must begin in a proven otherwise-accepting state and change only the payload; consecutive SRAM reads must use distinct addresses/data and check exact response identity.

## Non-blocking P2 notes

- Some old-admission/M528 identities are resolved transitively through the exact-bound M530-r2 contract. The repaired r3 contract and handoff should repeat direct paths and outer-seal identities.
- If the consecutive-read regression exposes a real frozen-top event-ordering defect, this verification-only author must not patch top-r2. The next gate must fail closed and require a new source-repair admission.

## Next gate

After one author produces only the permitted r3 verification package with an all-zero run receipt, a fresh independent source static hammer is mandatory. Only its P0=0 and P1=0 result may support a separate one-attempt functional VCS launch admission.
