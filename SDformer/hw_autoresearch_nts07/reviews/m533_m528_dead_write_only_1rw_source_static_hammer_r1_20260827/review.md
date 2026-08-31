# M533/M528 r3 verification-repair source static hammer

Verdict: **PASS, 100/100, P0/P1/P2 = 0/0/0.** This is a read-only source-static verdict. It authorizes no VCS, HDL/EDA, CPU/GPU, or remote run.

## Blocking-gate result

The r3 verification-only package closes all three P1 findings from the sealed failed M530-r2 identity without modifying the frozen top r2, SVA r2, generated macro adapter, or macro binding plan.

1. **Independent cycle oracle — PASS.** The expected execution state is reconstructed from accepted prep handshakes, the independently rebuilt maximum-population exact-subset directory, the fixed 64-row matcher/launch schedule, and the testbench-driven sink readiness. The model independently maintains stable row progress, written/prefetched/completed sets, earliest unprefetched edge, ordered two-slot queue state, the one-cycle pending response, and reservation without consume credit. It derives read, forward, deadline-hold, issue-stall, completion, write, response, and enqueue pulses from that state. DUT ready/debug/directory/live/queue/internal signals appear only on the `got`/observation side after the expected state or pulse has been determined. Per-cycle pulses, per-epoch totals, architectural data, conservation, order, and counters are compared.
2. **Causal stalled-RAW token — PASS.** A pending token freezes epoch, consumer, parent, and age. Only a matching forward at age 1 through 8 earns recovery credit; an unrelated or cross-task forward is fatal, age eight without recovery is fatal, and recovery, reset, protocol abort, disabled-score task boundary, and task drain all clear or reject outstanding state. There is no sticky historical-credit path.
3. **Closed launch authorization schema — PASS.** The future runner accepts no arguments or path overrides, verifies exact source/model/contract/review hashes and both seals, and requires exactly the ten authorization keys. `vcs_runs` must equal one; Icarus, Verilator, DC, Formality, PT, PTPX, CPU, GPU, and network/remote counters must all equal zero. The set comparison rejects missing and unknown keys before launch.

## P2 strength checks

- The malformed parent-only test first forces both sinks ready and proves an identity-stable, synthetic-parent, zero-residual beat is otherwise accepting. It then changes only the residual payload and requires ready, both architectural valids, scratch read/write, forward/dual-enqueue/elision/deadline/overflow/RAW pulses, and all current-beat counters to remain sterile before the sticky fault is credited.
- The dedicated five-row pattern creates adjacent macro reads with distinct parent addresses and unequal 1152-bit data. The cleanroom pending/slot model checks response parent, consumer, validity, and all data bits after the foundry-model edge; coverage requires at least one adjacent distinct-read pair and at least two response-identity checks.
- Eleven normal cover classes and six protocol attacks are separate counters with exact runner minima and single summary/PASS tokens.

## Identity and seal audit

- top r2: `726039dbfc1fe611de7beee7d0854028f4163e36b814329251a2e77b7fa790e1`
- macro adapter: `8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783`
- macro binding plan: `db4075cb9d34323dcc8c9bb04e575104acb9cb97a819b7f0750ce4a2d3976983`
- SVA r2: `b9f66febb5578e3c5a792dee42d87edb0ec68a71845b096a4f47c8c7cdde2c7b`
- TB r3: `73b9c6c45f9cd4a8185e386b9a13d674e888af938d3dbcbc29567ad40a558c32`
- runner r3: `e90f7e12b31013503822e265f89f4e9eacc1da48a414368d8bac87fd83082425`
- source contract r3: `3e50884bdfd8ea1c6a206d93d13d30995d17438c5f6485ff2d67740fbaae6d9b`
- private macro manifest: `c070d542c4f54338713d4c0941fa29b8b08d829587f518740ed6ef2f6c92694f`
- foundry slow Verilog: `8343acf01604cf0c6ac4757fd268a8f409401e0b80964ff671b030281ebb444d`
- foundry slow DB: `cd8c20508a7ea374eab09563f526944843c3e302f50986dfda4e00fa1b6aecbf`
- frozen docs/359: `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`

The contract, static-hammer request, and author handoff member/outer seals verify. The source contract sidecars also match the request identity. The exact future result directory and the VCS launch admission were absent during review. Runner shell syntax parses cleanly. No VCS, Icarus, Verilator, DC, Formality, PT, PTPX, CPU/GPU experiment, or network/remote job was run.

## Claim boundary and next gate

This PASS admits only the exact r3 source package to the next authorization gate. It does not establish functional VCS correctness, trace recurrence, speedup, PPA, energy, full-network performance, or a paper headline. Root may now author one separate, double-sealed functional-VCS launch admission bound to this exact review; that admission requires its own independent review before the sole run.
