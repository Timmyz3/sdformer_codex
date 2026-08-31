# M1009 decoder stratified-window first-principles source plan

## Verdict

`PASS_M1009_STRATIFIED_WINDOW_FIRST_PRINCIPLES_SOURCE_PLAN__GO_M1010_SOURCE_ONLY`

M998 proved exactness for its D2/D3 10K prefixes, but both prefixes contained
zero commits. M1009 repairs the measurement design without running a full row
or changing the network algorithm.

The key first-principles correction is that arbitrary request windows are not
cycle-additive under M785: port calendars, outstanding responses and terminal
dependencies cross their boundaries. Scaling a standalone 10K window by full
request count would therefore be invalid.

M1009 instead defines an explicit block-reset tiled schedule. Every block
starts from a declared idle state, charges the same boundary-ready, fill and
drain transactions to candidate and baseline, and ends with zero live tokens
and outstanding returns. Serial block concatenation is then additive. This is
a real executable schedule, but it must be named as block-reset cycles rather
than continuous-M785 cycles.

## Workload and strata

D0 uses the frozen M896/M785 exact route; D2 and D3 use M946 plus M896/M785.
D1 is strictly common-charge and receives no window or cycle estimate.

Four mutually exclusive strata are frozen before observing cycles:

- SOURCE_INIT_CENSUS: complete source fetch, measured once per layer;
- COMPUTE_REGULAR: ordinary noncommit work blocks;
- DEPENDENCY_STRESS: blocks containing psum movement, external weight refill,
  or dependency fan-in at least three;
- COMMIT_TAIL: dependency-closed blocks with positive commit count.

Noncensus strata use deterministic simple random sampling without replacement,
eight pilot blocks per layer/stratum and at most 32 adaptively. Adaptation is by
variance contribution only. Each window is capped at 10K expanded requests and
may split only at a complete service-group boundary.

## Estimator and CI

For stratum `h`, the estimated total is `N_h * mean_h`. Layer cycles are the
exact source census plus all estimated stratum totals. Variance uses the finite
population correction:

`sum_h N_h^2 * (1 - n_h/N_h) * s_h^2 / n_h`.

The conservative two-sided 95% multiplier is 2.365 because every sampled
noncensus stratum has at least eight observations unless it is a census.
Candidate and baseline use identical block IDs; the log-speedup interval keeps
their paired covariance. The CI describes sampling uncertainty across the
frozen deterministic block population, not stochastic runtime variation.

At relative CI half-width at most 5%, a block-reset layer cycle estimate may be
admitted. Between 5% and 10% it remains diagnostic or triggers additional
variance-targeted sampling. Above 10%, no point cycle or speedup estimate is
allowed. A local acceleration sentence additionally requires point speedup at
least 1.10x and lower 95% bound above 1.0.

## Required exactness

Every sampled block must miter all 14 existing M946 fields, including cycles,
expanded and compressed schedules, addresses, commits, cycle classes,
readiness and port calendars. M1009 additionally requires exact block census,
transaction IDs, reset IDs, per-kind counts/bytes, dense commit addresses,
fill/body/drain cycles, and zero final live/outstanding state.

Any mismatch, population omission/duplication, or zero-commit COMMIT_TAIL window
stops the entire estimate. Expanded/compressed transaction ratio remains a
diagnostic and can never satisfy a speedup gate.

## Minimum implementation

One additive M1010 wrapper may now be authored. It must import the frozen
M785/M890/M896/M946 sources by exact SHA and may not edit them. It must:

1. select frozen D0/D2/D3 sample-0/t0 records;
2. build and seal a metadata-only block population before running cycles;
3. select deterministic windows from that population;
4. emit explicit reset/fill/body/drain block transactions;
5. exact-miter each block with fresh M896 and M890 schedulers;
6. seal each window independently;
7. apply the frozen finite-population paired estimator.

No window execution is authorized by M1009. The M1010 source requires another
independent hammer and release before any bounded CPU run.

The contract and static estimator validator both pass. No large CPU, full row,
EDA, GPU or remote job ran; `docs/359` is unchanged.
