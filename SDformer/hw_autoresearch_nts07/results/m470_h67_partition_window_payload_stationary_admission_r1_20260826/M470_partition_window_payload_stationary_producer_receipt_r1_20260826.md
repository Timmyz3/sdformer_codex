# M470 partition-window payload-stationary producer receipt

## Outcome

The exact M470 CPU DSE passed its frozen identity and accounting checks.  The
best point over the full preflight axis is `P=5`, four resident block banks,
`row_tile=64`, and 128 B/cycle: `892,869,158` cycles at 209,416 logical bytes
and 228,352 macro-rounded bytes.  Against a strong-zero baseline forced to use
the same partition-window/full-psum-spill schedule, it is `1.286498482x`.

This is a nomination for independent hammer only.  It is not a performance
admission and does not justify RTL yet.

## Requested landmarks

For the explicitly requested `P={1,2,4,8}` landmarks, the best point is `P=4`,
four banks, `row_tile=64`: `964,742,918` cycles versus `1,218,613,216` for its
same-resource/same-schedule strong-zero peer, or `1.263148133x`.  Stored PWP at
`P=8` is infeasible under the 240-KiB macro gate even at `row_tile=1`: the
four-bank point is 231,074 logical / 311,296 macro-rounded bytes, and the
eight-bank point is 444,978 / 592,384 bytes.

## The decisive negative comparison

The `P=5` point pays every one of the 3,440 operator-window boundaries in full:
18,823,680,000 B spill writes plus 18,823,680,000 B reload reads.  Its total
modeled DRAM traffic is 38,563,705,600 B.  Consequently:

- versus the frozen strongest-zero anchor `742,148,386`, its speed is only
  `0.831195007x` (candidate needs `1.203087111x` as many cycles);
- versus the sealed M468R3 240-KiB stored-PWP point `872,452,768`, its speed is
  `0.977133951x` (2.3401% more cycles);
- versus full-resident M430 `517,041,352`, the cross-resource diagnostic is
  `0.579078522x` and is not a fair headline comparison.

So M470 proves a useful scheduling fact—PWP wins materially if both paths are
forced through full-psum spills—but it does not improve the current hardware
frontier.  The producer score recommendation is 74/100: strong identity,
capacity, and fail-closed discipline; zero points for superiority over the
existing best.  Independent hammer should verify the nomination, after which
the axis should be held or killed unless a missing fair advantage is found.

## Evidence boundary

Scope is only the four frozen H67 ep35 bottleneck Conv3x3 operators.  Arithmetic
is exact and the checkpoint is unchanged.  Numbers are from a CPU cycle model,
not RTL, Synopsys, physical macros, energy, full-network speedup, or system
speedup.  The M40 one-shot payload was not read; the sealed compact sidecar
already contained every selected row tile, so M410R2 was not re-traversed.
