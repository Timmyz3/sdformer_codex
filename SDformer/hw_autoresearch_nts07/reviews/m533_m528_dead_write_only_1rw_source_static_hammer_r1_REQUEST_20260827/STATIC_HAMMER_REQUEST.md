# M533 M528 r3 verification-repair source static hammer request

Perform one fresh, independent, read-only static review of the exact source identity in `request.json`. Do not run VCS, Icarus, Verilator, DC, Formality, PT, PTPX, CPU/GPU experiments, or any remote job.

The blocking review target is the verification repair, not a redesign of frozen top r2. Confirm that the TB cleanroom model derives expected state and pulses only from accepted prep stimulus, the frozen matcher/order schedule, and externally driven sink readiness. DUT ready, debug, directory, live, queue, and internal signals may be observed only after an expected value has already been independently committed; they must never generate expected macro-read, forward, deadline-hold, stall, queue, pending, or counter values.

Audit the bounded stalled-RAW token causally: epoch, consumer, parent, and age; one-to-eight-cycle matching forward only; timeout/reset/abort/recovery cleanup; no historical, unrelated, or cross-task credit. Audit the runner's exact closed authorization key set and exact foundry manifest/slow-Verilog bindings. Also inspect the legal-zero-then-payload-only malformed attack and the directed adjacent distinct-address/data foundry-read identity test.

A PASS requires `p0_count=0`, `p1_count=0`, schema `m533_m528_dead_write_only_1rw_source_static_hammer_v1`, status `PASS_M533_M528_DW1RW_SOURCE_STATIC_HAMMER`, and a member manifest plus outer seal. Static PASS never authorizes VCS; root must author and independently review a separate double-sealed one-attempt launch admission.
