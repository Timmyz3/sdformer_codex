# M241r2 elastic tagged checkpoint/no-forward milestone r1

## Outcome

M241r2 is the immutable repair of the two concrete negative behaviors found by
the independent M249 review of M241 r1.  The repair keeps the same standalone
four-bank, eight-lane representative and preserves the intended low-state
features: lazy accumulator valid bits, zero same-address forwarding payload and
no M149 instance.

Both external memory cuts now use explicit valid/ready request and response
channels.  The response identity includes the accepted transaction identity;
weight traffic carries sequence, operator, partition, window, checkpoint epoch,
payload ID, order, source and destination half, while accumulator traffic carries
sequence, window, checkpoint epoch, payload ID and order.  A mismatched response
is held unready, never consumed, and latches a fail-closed protocol fault.

Context open is additionally bound to a valid loader tuple of operator,
partition, checkpoint epoch and payload ID.  The weight cache key now includes
the window and loader payload identity, rather than relying on caller epoch
metadata alone.

Numeric overflow no longer appears as a successful commit.  It raises an
explicit abort transaction; `commit_valid=0`, `commit_accept=0`, and all four
write enables remain zero.  The directed attack places two already accepted
younger tokens in s0/s1, verifies both are discarded, stalls abort for two
cycles, acknowledges it, and then proves reset recovery with one clean commit.

## Exact Synopsys VCS evidence

The exact-SHA run passed four replays of the same H67/Motion ep35 checkpoint
subset:

- Fixed weight/accumulator response latency of 1, 2 and 3 cycles.
- Randomized 1-to-3-cycle latency with request and response-launch stalls.
- 126 real ordered descriptors per mode.
- 504 exact writes and 4,032 exact lane comparisons per mode.
- Zero integer mismatches and zero assertion failures.
- 56 weight read groups plus 448 cache hits per mode.

Directed traffic creates real downstream response backpressure rather than
depending on random coverage.  The sealed run records four weight-request stall
cycles, two weight-response stall cycles, two accumulator-request stall cycles,
two accumulator-response stall cycles and 19 commit-stall cycles.  SVA also
checks that the response bank masks and every data lane remain stable while the
receiver is not ready.

One stale weight response and one stale accumulator response were attacked; no
stale response was accepted.  The loader binding, RAW interlock, two-younger
overflow abort, abort backpressure, context-abort pulse and post-abort recovery
are all covered.

## Claim boundary

This closes the fixed-one-cycle protocol defect and the overflow-success defect
for the tested standalone representative.  It does not establish a selected
physical SRAM, full 96-lane closure, a complete M152 finite trace, DC/STA, energy
or a measured cycle speedup.  The 9.0x value remains weight-read work only, and
M238's 1.687017659x remains an unadmitted cycle-model target.

Therefore `physical_speedup=false`, `system_speedup=false`,
`paper_ppa_ready=false`, and `headline=false`.

## Evidence

- Exact VCS run: `results/m241r2_elastic_tagged_checkpoint_no_forward_directed_vcs_r1_exact_20260825/`
- Contract: `contracts/m241r2_elastic_tagged_checkpoint_no_forward_exact_vcs_contract_r1_20260825.json`
- Independent review that drove the repair: `results/m249_m241_checkpoint_no_forward_independent_hammer_r1_20260825/`
- RTL/SVA/TB: `rtl_m241r2/`, `verif_m241r2/`, `tb_m241r2/`

No DC was launched.  M241 r1 and `docs/359` were not modified.  This milestone
is ready for a different agent to perform an independent M241r2 hammer review.
