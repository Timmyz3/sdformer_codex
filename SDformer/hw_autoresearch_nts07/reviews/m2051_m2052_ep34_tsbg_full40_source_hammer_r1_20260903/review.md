# M2051/M2052 ep34 TSBG full-cohort static source hammer

**FAIL / NO-GO, 88/100; P0/P1/P2 = 1/1/2. M2052 is not authorized.**

The fixture and campaign expansion are otherwise structurally sound, but the
current parser cannot publish this campaign correctly. No VCS, simulator, DC,
or GPU process was launched by this review.

## Blocking findings

### P0: inevitable empty-workload parser failure

The independently reconstructed cohort contains **286 empty workloads**. The
testbench emits:

`M2051_EMPTY_WORKLOAD_RETIRED_REPLAY_NOT_APPLICABLE`

but the parser requires:

`M2048_EMPTY_WORKLOAD_RETIRED_REPLAY_NOT_APPLICABLE`

Therefore even a functionally correct set of 1,920 simulations would fail on
the first empty log after the runner had consumed its one allowed attempt. This
is deterministic, not a hypothetical corner.

### P1: geometric-mean overflow

The parser computes `math.prod(speedup) ** (1/1920)`. That formulation is not
numerically stable. Repeating the admitted M2050 192-workload distribution ten
times makes the product `Infinity`, while the correct log-domain geometric mean
remains 1.8725796x. The successor must use
`exp(sum(log(speedup))/len(rows))`.

## What passes statically

- Samples are exactly 0--39: four DSEC sequences with ten samples each.
- The inventory is 16 layers: all 12 FC1 layers and four FC2 layers supported
  by G48. Each sample/layer contributes fixed first, middle, and last B4 token
  quartets, giving exactly 1,920 workloads. Selection never reads performance.
- All 1,920 selected quartets were redecoded from sealed M1707. The rebuilt
  368,640 fixture words, 1,920 stat words, and metadata rows are byte-exact.
  All 192 predecessor M2050 rows reproduce exactly.
- Independent LRU reconstruction gives ordinary misses/hits/evictions of
  91,399/857/85,225 and TSBG counts of 32,673/59,583/26,499.
- The source generator mechanically reproduces the checked-in builder, TB,
  parser, filelist, and runner byte-for-byte. Python and Bash syntax pass.
- The runner pins every runtime input, permits one license query, one compile,
  1,920 simulations at `-P 4`, no retry, and publishes only after parsing.
- The parser otherwise retains empty rows, enforces equal cycles for empties,
  reports the 0.998x-style minimum without incorrectly requiring every row to
  improve, and preserves the narrow component claim boundary.

## Successor gate

M2052 must not run. A successor may be authorized only after it:

1. checks the M2051 empty marker actually emitted by the TB;
2. uses a log-domain geometric mean;
3. receives new parser/runner identities and exact pins rather than modifying
   this reviewed source set; and
4. passes a fresh static source hammer.

The eventual result must remain scoped to all 40 captured samples, all 12 FC1
and four G48-supported FC2 layers, and three fixed B4 token regions. It is not
all-FC2, full-token-population, real-weight, same-area, energy, or system
evidence.
