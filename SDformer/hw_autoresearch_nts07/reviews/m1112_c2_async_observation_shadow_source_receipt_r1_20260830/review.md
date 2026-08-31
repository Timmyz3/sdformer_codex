# M1112 C2 async-observation shadow source receipt

Status: `M1112_ASYNC_OBSERVATION_SHADOW_SOURCE_FROZEN__M1113_INDEPENDENT_HAMMER_REQUIRED__NO_EDA`

M1112 is an additive diagnostic repair. It does not overwrite M1090r3/M1091r3 and does not retry the consumed M1091r3 attempt. The frozen C2 implementation remains the functional cone; its thirteen synchronous-reset debug outputs are terminated locally. Thirteen new service/adapter observation values are maintained in a separate 337-bit asynchronous-reset shadow bank driven only by read-only `raw_accept`, bank request/response accept, and result-accept events.

The mapped testbench computes all 22 `$isunknown` predicates into one bitmap every sampled cycle. It records the first-X cycle and bitmap, continues sampling the remaining signals and cycles, accumulates a union bitmap, and fails only after the complete 128-cycle window. No initreg, delayed check, warm-up mask, false path, or case analysis is authorized.

The future engine requires each of the 337 mapped shadow bits to use a cell with an explicit async reset/set pin before mapped VCS may start. D/CP-only cells are rejected even if reset behavior was reconstructed in the D cone. Live inputs remain regular-file SHA pinned; the only symlink-followed-byte exception is the exact sealed M1091r3 historical quarantine.

Author static checks passed: 47 checks, zero EDA invocation, zero new attempt, zero launcher. This is not mapped-functionality, PPA, activity, performance, or paper evidence. The only allowed next step is independent M1113 hammer review.
