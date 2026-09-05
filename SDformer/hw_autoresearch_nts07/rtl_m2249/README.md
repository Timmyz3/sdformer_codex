# Consumer-scoped bank fill (M2249)

This is an optional extension of the existing M2018 C2/TSBG frontend, not a
replacement accelerator. It retains the real M803 eight-bank adapter, four
cached weight rows, signed arithmetic, independent Acc24 contexts, and commit
protocol. Ordinary and TSBG both support partial-bank refill.

On a row access, ordinary requests its current context's active banks. TSBG
uses the union of the four already-loaded context masks. Cached banks are
subtracted before issuing a request. A 16-bit valid mask per cache row is
published only after all six output slices of newly requested banks arrive.
No product or signed value is reused between contexts; only weight data is.

The cache is valid within one immutable weight identity. Reset before changing
layers or weights. Persistence across such changes is not implemented.

## What is being tested

`run_m2249_bank_selective_vcs.py` compiles two Synopsys VCS modes and runs the
same preselected low/median/high windows. Bank counts come from the independent
CPU model; cycle counts come from RTL; all 384 committed Acc24 values per window
are checked against the scalar signed reference. The memory model has independent
bank readiness, out-of-order responses and downstream backpressure.

The real windows do not activate partial cache hits or negative signs. A separate
four-bundle, no-reset directed test therefore checks:

- low-bank fill followed by high-bank/other-bank partial refill;
- use of both old and newly filled banks, including `-1 * -128 = +128`;
- a fully warm round with zero bank reads;
- a fifth-row insertion and ensuing LRU evictions.

Expected directed reads are ordinary `[72, 24, 0, 222]` and TSBG
`[72, 24, 0, 24]`. These are synthetic protocol cases, not workload speedups.

## Claim boundary

The completed VCS run is
`results/m2249_bank_selective_3xd7_3fp/result.json` (2026-09-05):

| Preselected window | Ordinary cycles | TSBG cycles | Ordinary / TSBG bank reads |
| --- | ---: | ---: | ---: |
| low | 2,044 | 2,044 | 312 / 312 |
| median | 2,904 | 2,189 | 384 / 300 |
| high | 12,844 | 5,848 | 2,442 / 1,098 |

Both four-round directed tests also pass. These runs use a negedge reset
deassertion, integer-picosecond measurement checks, and VCS `-no_save`.
The earlier compile-order failure and floating-point TB duration failure are
retained in their original run directories; neither is a passing experiment.

The six points calibrate the literal FSM model in
`system_simulator/scripts/m2252_masked_c2_cycle_model.py` with zero cycle/read
mismatch. Its 4,320 independently reset chunks yield 14,508,203 / 8,052,073
modeled cycles (1.8018x). This is CPU modeling, not 4,320 RTL replays or full FC.

The prior CPU experiment counts 4,320 cold G48 chunks once: union fill predicts
38.85% fewer bank activations than mask-aware ordinary LRU4. This is traffic,
not energy or throughput. The six-window RTL pilot must not be extrapolated to
that population, and neither experiment provides mapped area or timing for this
new variant. Conventional sector-valid caching and sparse broadcast scheduling
are antecedents; the implementation contribution is their concrete combination
with this C2 signed/banked interface and fair ordinary baseline.
