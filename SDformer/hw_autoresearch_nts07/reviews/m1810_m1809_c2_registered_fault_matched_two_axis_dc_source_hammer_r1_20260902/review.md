# M1810 independent source hammer

Status: `PASS_M1810_M1809_C2_REGISTERED_FAULT_MATCHED_TWO_AXIS_DC_SOURCE_HAMMER__P0_0_P1_0_P2_0__EXACTLY_TWO_DC_RUNS_AUTHORIZED__NO_EDA`

Score: 99/100. Findings: P0=0, P1=0, P2=0.

## Verdict

M1809 is a valid two-axis, equal-public-boundary DC shell. `ARCH_MODE=0`
directly instantiates the exact M1801 K8 source admitted by M1802 and confirmed
by M1804; `ARCH_MODE=1` directly instantiates the frozen M519 K1x8 baseline.
Each target appears exactly once. The old M519 K8 top, old M519 matched top, old
M803 K8 top, and old M803 matched top are absent. The frozen M803 bundle-to-bank
adapter remains only because it is a required child of M1801.

The wrapper, M1801, and K1x8 agree on all 51 public ports, including all eight
request and response banks and 16 INT8 weight lanes per bank. Direction,
signedness, packed width, and unpacked dimensions have zero mismatches. All five
configuration parameters are forwarded identically. Eight common debug outputs
are locally sunk in both modes; M1801's five adapter-only observations are also
locally sunk, so debug state is not exposed at the comparison boundary.

The exact M1801 top SHA is the SHA frozen by M1802 and M1804. M1809 only selects
and directly wires that child; it does not alter M1801 data, ready/valid, Acc24,
completion, or registered-fault paths.

The 13-row filelist contains 13 unique, existing files and 13 unique module
definitions. The R8 setup/area Tcl accepts a parameterized top through
`ELAB_PARAMETERS`, and the generic SDC constrains `clk_core` at 3.000 ns while
applying equal delays and loads to all public inputs and outputs. The flow is
therefore statically applicable to `ARCH_MODE=0` and `ARCH_MODE=1`; hold remains
diagnostic at this logic-only, pre-CTS point.

## Authorization boundary

Exactly two fresh, source-identity-bound `dc_shell` runs are authorized: K8
`ARCH_MODE=0` and equal-bandwidth K1x8 `ARCH_MODE=1`. Both must use the same
filelist, library, Tcl, SDC, 3.000 ns clock, public ports, and reporting boundary.
Automatic retry and old DDC/netlist reuse are forbidden. All other EDA runs are
unauthorized at M1810.

This review is source-only. It proves no synthesis success, timing, area, mapped
functionality, power, energy, performance, system speedup, or paper-citable
physical result.
