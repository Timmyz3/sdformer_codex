# M974 | C2 three-axis PT/SAIF/PTPX first-principles review

Verdict: `GO_SOURCE_PLAN__NO_GO_CURRENT_POWER_OR_ENERGY_CLAIM`, score 96/100,
P0=0/P1=3/P2=2.

## What can be reused

M872 and M903 are clean enough to reuse as the physical starting point. The
sealed K1/K8/K1×8 mapped netlists and SDCs come from one DC attempt, use the same
top-level pins, 3 ns clock, libraries and ZeroWireload assumptions, and all meet
setup. Their admitted areas are 124,620 / 131,086 / 585,479 µm². The fair
headline remains K8 versus equal-bandwidth K1×8: 1.0167× directed cycles,
4.541× throughput/mm² and 77.61% less logic area.

Existing activity cannot be reused. Neither canonical M859 nor M872 contains a
SAIF/VCD/FSDB. A compiled RTL simv survives only in an unsealed work directory,
contains no activity capture, omits the K1 axis, and is not the exact mapped
netlist identity. Power and energy therefore remain false today.

## Minimum executable campaign

Create only four source units:

1. A common mapped-gate replay TB compiled separately with each exact M872
   netlist. It drives the same five M867 cases and the same deterministic eight
   memory banks, checks numeric outputs plus request/response multisets, and
   uses K8/K1×8 cycles as hard regression gates. K1 remains diagnostic.
2. A common UCLI activity controller that captures only the DUT. Each axis run
   emits five per-case SAIFs, with reset/inter-case idle excluded and each window
   bounded by accepted header and accepted token-done.
3. One PT/PTPX Tcl engine that reads the exact netlist/SDC, checks setup, maps
   each SAIF, and only calls `update_power` after annotation and `check_power`
   gates pass.
4. One fail-closed sequential runner/contract/release. All three axes must be
   produced in one attempt; partial or cross-attempt tables are forbidden.

Direct mapped-gate annotation must cover 100% of nets and leaf cells, have
positive duration equal to measured cycles × 3 ns, and contain zero TX entries.
Nonzero-toggle percentage is reported rather than forced above 95%: in a sparse
accelerator, low real activity is the quantity being measured. Clock/control,
raw/header, memory request/response, accumulator/result and token-done cones must
all show nonzero activity.

PTPX should use TT 0.9 V/25 °C, 333.33 MHz ideal clock, ZeroWireload, no SPEF,
100 ps primary-input slew and 50/200 ps sensitivity. For each case:
`E_logic[pJ] = P_total[mW] × duration[ns]`. Aggregate energy is the sum over the
same five cases—not an unweighted average of powers with different runtimes.
Report internal, switching, leakage and total components separately.

Decision gate: ≥2× aggregate K1×8/K8 logic-energy efficiency, no nonzero case
worse than 0.95×, and ≤10% slew-induced ratio movement supports a main energy
statement. A 1.10–2× aggregate result is supporting evidence. Below 1.10× or
any correctness/annotation/check-power failure kills the energy-advantage claim.

## Hard boundary

This is standard-cell logic power only. The present top contains zero SRAM
macros; the eight weight banks are testbench/external interfaces. There is no
CTS, SPEF, extracted wire, ordered 120-record H67 stream, complete FC2 or network
schedule. Logic PTPX must never be called macro energy, energy/frame, system
energy, PPA or silicon power.

Macro-inclusive energy minimally needs an exact bank capacity/count/port
contract, matching macro `.db` or sealed access energy, per-bank access activity,
and aligned PVT. System energy additionally needs the ordered H67 FC2 stream and
full-network SRAM/DRAM/operator schedule.

M974 ran no VCS, DC, PT, PTPX, Formality, GPU or remote workload. It did not
modify canonical evidence or `docs/359`.
