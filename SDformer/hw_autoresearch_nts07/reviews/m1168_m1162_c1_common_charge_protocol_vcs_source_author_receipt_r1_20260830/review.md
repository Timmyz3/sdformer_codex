# M1168 — M1162 common-charge protocol VCS package author receipt

Status: **source-only PASS; fresh different-author M1169 hammer required; no
VCS or EDA has run.**

M1168 consumes the exact authorization granted by the sealed M1166 review and
adds one verification package without modifying M1162, frozen M935, the nine
parent SRAM wrapper, or `docs/359`:

- executable SVA for request stability/suppression, atomic response joining,
  response prerequisites, non-first behavior, reset, sticky faults and the
  no-consecutive-completion half of the II bound;
- a directed/random TB covering both partial request orders, both response
  skew orders, five-cycle request stalls and response backpressure, a
  non-first beat, explicit attained II=2, three occupied reset states, seven
  DUT fault attacks, two external-service assumption attacks, 24 deterministic
  random legal transactions, and exact one-fire request scoring;
- one normal task through the public prep interface and frozen M935, with two
  source beats, one architectural row completion and one task completion;
- a six-source foundry UNIT_DELAY filelist and a fail-closed exact-SHA one-shot
  launcher.

The service mutation cases are intentionally not mislabeled.  M1162 relies on
the external service rule that a held response retains `valid` and payload.  A
separate TB monitor detects deliberate weight-payload mutation and psum-valid
drop; these attacks do not claim that `protocol_error` detects an environment
violation or that M1162 contains a wide response shadow FIFO.

The launcher cannot run from this receipt.  It requires exact SHA values for a
future recursively sealed M1169 review and M1170 release, checks a one-shot
attempt token before compilation, and authorizes only one VCS compile plus one
`simv` execution.  DC, PT, Formality, PTPX, replay, GPU and remote work remain
blocked.  UNIT_DELAY would establish only functional behavior, not timing.

No cycle, speedup, PPA, power, energy, full-storage, system or paper-ready
claim is created.  In particular, no prior M1114/M528 CPU speedup is inherited.

