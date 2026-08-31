# M275 port-feasible two-bank Hamming-tree PWP schedule

M275 closes the materialization-overlap condition raised by the independent
M271 review.  It reconstructs all 17,280 M251 phases and assigns distinct
current/next weight and PWP banks.  Current compute reads a 96-byte weight port
or 144-byte PWP port; the next bank is filled at 32 bytes/cycle and then read by
a dedicated 96-lane signed12 tree generator that writes the next PWP bank.
Fill and generation are serial, and neither touches the current banks.

Every one of the 17,270 in-sample partition transitions hides preparation with
zero exposed cycles.  The worst transition has 1,008 current-service cycles,
651 next-preparation cycles, and 357 cycles of slack.  The maximum preparation
remains 787 cycles.  Operator boundaries use the same recurrence; every sample
starts cold, and the final phase performs no speculative next preparation.

The first partition needs 675 rather than 960 cycles, so the exact wide-port
ten-sample total is 352,332,270 cycles, 2,850 fewer than stored fixed-PWP M251.
The isolated four-Conv speedup is 1.540570x versus the strong bit-sparse
reference and 18.833241x versus dense.  The 2,850-cycle improvement is tiny;
the important contribution is retaining M267's 31,850,496-byte fixed-PWP
payload elimination without exposing materialization stalls.

The result does not eliminate on-chip working capacity: two PWP banks consume
36,864 bytes and two weight banks consume 24,576 bytes.  Generator work and
energy are real and remain unmeasured.  This is an exact port-feasible module
cycle schedule, not SRAM-macro proof, generator RTL/VCS/DC, complete Conv,
system speedup, paper PPA, or headline evidence.
