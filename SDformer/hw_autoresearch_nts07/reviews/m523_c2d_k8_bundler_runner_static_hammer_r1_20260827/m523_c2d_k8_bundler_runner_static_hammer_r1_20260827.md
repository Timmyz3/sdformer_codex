# M523 D8 descriptor bundler independent static hammer r1

## Decision

**STATIC GO.** P0=0, P1=1, P2=0, score 97/100. Exactly one invocation of `run_vcs_m523_c2d_k8_polyphase_tap_bundler_exact_sha.sh` at SHA `b735b9ffd8be0795c1ff5d2bca7d30be57a9f9fb1c547ce94b4d7334091ff816` is authorized. No DC, Formality, PT/PTPX, performance, energy, PPA, direct-C2, system, or headline claim is authorized.

This was a static review only. No VCS, synthesis, formal, power, timing, or open-source RTL tool was run. No author input and no `docs/359` content was modified.

## Identity and mechanical evidence

All seven frozen identities match the request: runner, contract, RTL, SVA, TB, filelist, and `docs/359`. The runner is mode 0755 and passes `bash -n`; the request and contract pass strict finite-JSON parsing. The canonical result and one-shot attempt marker were absent at review time.

An independent integer oracle gives event fanouts `[4,6,6,9,9,9]`, 43 taps, semantic bundle sizes `[8,2,6,8,1,8,8,2]`, and phase totals `[6,10,10,17]`. An exhaustive arithmetic check covered 8,352 legal combinations of head, count, pop count, and atomic push fanout. Every transition preserves `tail'=(head'+count') mod 18`, keeps count in 0..18, and keeps all appended writes disjoint from unpopped entries. At count 18, tail equals head and accepted same-edge writes are a subset of the just-popped slots.

## RTL judgment

The partial-tail rule is safe: a non-stream-last event boundary with no successor remains invalid until an equal-context successor arrives, a mismatching successor makes the partial bundle flushable, or sticky fault makes it drainable. Cross-event selection checks tag and time after each per-lane event fence; stream-last stops selection immediately. Hence a bundle cannot cross tag, time, or stream-last.

Stalled output is stable. With no pop, a full FIFO admits no event; a nonfull FIFO writes only at its free tail, disjoint from selected head lanes. Scalar and per-lane repeated SVA properties cover the stalled payload. Sticky protocol fault disables every later event acceptance but leaves accepted descriptors drainable, including a partial nonterminal tail.

## Runner judgment

The caller self-SHA gate is line 26 and precedes result, attempt, review, resource, and Synopsys side effects. The authorizing review is double-sealed and binds the exact runner and contract. Resource checks precede the atomic attempt marker; the attempt precedes exact `vcs -full64 -ID` and the `-full64` compile. A wrong-SHA child exits 10 before attempt or VCS.

The run must match one exact PASS line and ten nonzero cover gates. Its finite JSON receipt, regular-file manifest, and topology are verified in staging and again after atomic canonical rename. The only permitted VCS symlinks are one PID archive link and the assertion coverage shape link; path, raw target, in-tree resolved regular target, and resolved-target SHA are sealed. Missing, additional, dangling, or external links fail closed.

## Claim boundary

This is an 8-lane descriptor transport, not an 8-bank weight issue engine. It has no flattened `(source_channel,kernel_index)` key, bank-conflict deferral, or stored-weight identity proof. A VCS pass therefore proves only the directed descriptor behavior. It cannot be used as evidence of direct M218/C2 compatibility, frontend or decoder speedup, energy, PPA, system speedup, or a DATE headline.

## Finding

P1 `M523-STATIC-P1-001`: contract field `author_static_repairs.tb_final_ledger_repair` retains stale prose `full8=3, tails1=3`. Every authoritative executable gate and the independent oracle require `full8=4, tails1=1`. The stale sentence does not make this one-shot VCS ambiguous, but it must not be cited and should be repaired under a future identity before paper-facing use.

## Authorized command identity

Only runner SHA `b735b9ffd8be0795c1ff5d2bca7d30be57a9f9fb1c547ce94b4d7334091ff816`, for one directed Synopsys VCS invocation, is GO. The required outer-seal-file SHA must be supplied by the caller. Any file change, second attempt, or scope expansion requires a new independent gate.
