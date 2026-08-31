# M270 FC1 malformed-header fail-closed correction

M270 closes the M268 P1 without changing the M262 clean trace mapping.  Invalid
mode, descriptor count above 3072, and factor-address wrap are rejected from
IDLE into a tagged sticky protocol abort.  Legal empty bypass now explicitly
holds under `done_ready=0`, and clean context-mask popcounts one through eight
are all checked against `6+3*popcount`.

Fresh exact-SHA Synopsys VCS passes with six tiles, 26 descriptors, 22 clean
cycle checks, 40 commits, seven attacks, all six stall types, and zero numeric,
transaction, or assertion mismatch.  The first development run timed out
because the malformed-header testbench left `abort_ready` high before sampling;
the sealed r2 run drives backpressure before fault injection and passes.

M262's frozen trace ratios remain `1.672240x` context-factorized versus
bit-sparse lifecycle and `2.580060x` weight-request reduction.  M270 adds no new
speedup.  The evidence remains an eight-lane small-width module without
address-timed SRAM, 96-lane RTL, full-trace RTL, DC, complete FC1/FFN, system, or
headline admission.
