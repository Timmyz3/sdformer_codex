# M279 independent hammer review of M275 Conv tree materializer

M275's raw arithmetic is reproducible: an independent standard-library audit
imports neither the M275 producer nor its M251/M43 support modules, rehashes all
40 M248 payloads, reconstructs 17,280 phases and 17,270 transitions, and exactly
matches the M251 work, M267 flip histograms, 352,332,270 cycles, 357-cycle
minimum slack, 1.540569872x bit-sparse ratio and 18.833240566x dense ratio.

The important correction is the next-PWP parent read.  M275 declares a 144-byte
next-PWP write but no parent-read port and emits no per-cycle port trace.  A tree
edge needs a parent PWP read, weight updates, and a child PWP write.  With a
144-byte 1R1W bank, or a two-accumulator single-port prefetch schedule, the
original total is feasible.  A synchronous single-port, one-accumulator stress
schedule instead has 707 cold-start cycles and 352,332,590 total cycles.

Both independent schedules still hide all 17,270 in-sample transitions: the
two-accumulator case keeps 357 cycles of minimum slack, while the one-accumulator
case keeps 333.  Thus zero transition stalls is robust, but the exact total is
conditional on a microarchitecture that M275 has not frozen.

Evidence quality scores 91/100 and hardware admission 72/100.  Findings are
P0=1, P1=3, P2=3.  M275r2 should freeze the parent-fetch implementation, emit
explicit bank/port events, count metadata plus accumulator state, and bind the
M251r2 signed-range correction.  The 1.54x and 18.83x values belong to the full
isolated PWP Conv model; tree materialization itself is only 1.00000809x versus
stored fixed PWP.

Clean exact replay is byte-identical, wrong-SHA preflight exits 10 before replay,
no open-source RTL tool was used, and `docs/359` remains unchanged.
